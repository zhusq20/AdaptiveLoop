"""Batched greedy decoding of Ouro with per-loop KV, recording loop-2/loop-4 readouts.

The forward pass re-uses the Hugging Face modules of the official remote code
(embedding, 24 sandwich-norm decoder layers, final norm applied after every loop,
lm_head) and keeps one KV slot per (loop, layer) = loop * n_layers + layer, exactly as
Ouro's UniversalTransformerCache. Every loop always runs (no early-exit gate), so
the decoded trajectory is the "original four-loop" reference of plan.md.

Readout of loop d = lm_head(norm(h_d)) with lm_head evaluated in FP32 (bf16 hidden
state upcast, FP32 weight copy, TF32 disabled). The greedy token is argmax of the FP32
loop-4 logits, so the recorded reference argmax z4 equals the generated token by
construction.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .bounded import CHECK_GRID, center, compact_stats

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@dataclass
class SeqResult:
    tokens: list[int] = field(default_factory=list)
    finish: str = ''
    stats: list[dict] = field(default_factory=list)       # one dict of CPU tensors per step
    full: list[tuple[int, torch.Tensor, torch.Tensor]] = field(default_factory=list)


class OuroRunner:
    def __init__(self, model_dir: str, device: str = 'cuda', read_loops=(2, 4)):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        lm = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval()
        self.lm, self.inner, cfg = lm, lm.model, lm.config
        self.cfg = cfg
        self.T = cfg.total_ut_steps
        self.n_layers = cfg.num_hidden_layers
        self.layers = list(self.inner.layers[: self.n_layers])
        self.head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
        self.kv_heads = cfg.num_key_value_heads
        self.gqa = cfg.num_key_value_heads != cfg.num_attention_heads
        self.rope = sys.modules[type(self.inner).__module__].apply_rotary_pos_emb
        self.W = lm.lm_head.weight.detach().float()           # [V, H] FP32 copy
        self.device = torch.device(device)
        self.read_loops = tuple(read_loops)
        assert self.read_loops == (2, 4), 'collection stats assume loops (2, 4)'

    # ------------------------------------------------------------------ forward
    def _layer(self, layer, x, cos, sin, slot, start, mask):
        b, s, _ = x.shape
        a = layer.self_attn
        h = layer.input_layernorm(x)
        q = a.q_proj(h).view(b, s, -1, self.head_dim).transpose(1, 2)
        k = a.k_proj(h).view(b, s, -1, self.head_dim).transpose(1, 2)
        v = a.v_proj(h).view(b, s, -1, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, cos, sin)
        self.K[slot, :, :, start:start + s] = k
        self.V[slot, :, :, start:start + s] = v
        o = F.scaled_dot_product_attention(
            q, self.K[slot, :, :, :start + s], self.V[slot, :, :, :start + s],
            attn_mask=mask, scale=a.scaling, enable_gqa=self.gqa)
        o = o.transpose(1, 2).reshape(b, s, -1)
        x = x + layer.input_layernorm_2(a.o_proj(o))
        return x + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(x)))

    def _step(self, ids, start, pos, mask):
        """Run all T loops on `ids` [B, S] written at cache index `start`. Returns the
        normed hidden state of the last position after each loop in read_loops."""
        x = self.inner.embed_tokens(ids)
        cos, sin = self.inner.rotary_emb(x, pos)
        out = {}
        for s in range(self.T):
            for li, layer in enumerate(self.layers):
                x = self._layer(layer, x, cos, sin, s * self.n_layers + li, start, mask)
            x = self.inner.norm(x)
            if s + 1 in self.read_loops:
                out[s + 1] = x[:, -1]
        return out

    def _readout(self, h):
        return center(h.float() @ self.W.T)

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new_tokens: int, stop_ids: set[int],
                 stop_strings: tuple[str, ...] = (), k: int = 32, grid=CHECK_GRID,
                 force: list[list[int]] | None = None, keep_full: bool = False) -> list[SeqResult]:
        """Greedy-decode a batch. `force` replays given continuations instead of argmax
        (validation only). `keep_full` keeps full z2/z4 rows for every step (validation
        only); otherwise full rows are kept just for positions without the compact
        certificate."""
        dev = self.device
        n = len(prompts)
        P = max(len(p) for p in prompts)
        L = P + max_new_tokens
        pad = torch.tensor([P - len(p) for p in prompts], device=dev)
        ids = torch.zeros(n, P, dtype=torch.long, device=dev)
        for i, p in enumerate(prompts):
            ids[i, P - len(p):] = torch.tensor(p, device=dev)
        p0 = next(self.lm.parameters())
        self.K = torch.empty(self.T * self.n_layers, n, self.kv_heads, L, self.head_dim,
                             dtype=p0.dtype, device=dev)
        self.V = torch.empty_like(self.K)
        kidx = torch.arange(L, device=dev)
        valid = kidx[None] >= pad[:, None]                     # [B, L] real (non-pad) keys
        # prefill: causal over real keys; pad queries attend to themselves (avoids NaN rows)
        q_ = kidx[:P]
        causal = (kidx[None, :P] <= q_[:, None])[None] & valid[:, None, :P]
        causal |= torch.eye(P, dtype=torch.bool, device=dev)[None]
        pos = (kidx[None, :P] - pad[:, None]).clamp(min=0)
        hs = self._step(ids, 0, pos, causal[:, None])

        results = [SeqResult() for _ in range(n)]
        alive: list[int | None] = list(range(n))  # batch row -> sequence index (None = done)
        cur = P
        for t in range(max_new_tokens):
            z2, z4 = self._readout(hs[2]), self._readout(hs[4])
            st = compact_stats(z2, z4, k, grid)
            nxt = st['a4'].clone()
            if force is not None:
                nxt = torch.tensor([force[i][t] if i is not None and t < len(force[i]) else 0
                                    for i in alive], device=dev)
            st_cpu = {key: val.cpu() for key, val in st.items()}
            need_full = torch.ones_like(st_cpu['cert']) if keep_full else ~st_cpu['cert']
            full_rows = [row for row in need_full.nonzero().flatten().tolist() if alive[row] is not None]
            if full_rows:
                fz2, fz4 = z2[full_rows].cpu(), z4[full_rows].cpu()
            nxt_cpu = nxt.tolist()
            for row, si in enumerate(alive):
                if si is None:
                    continue
                r = results[si]
                r.stats.append({key: val[row] for key, val in st_cpu.items()})
                if row in full_rows:
                    j = full_rows.index(row)
                    r.full.append((len(r.tokens), fz2[j], fz4[j]))
                tok = nxt_cpu[row]
                r.tokens.append(tok)
                reason = None
                if force is not None:
                    if len(r.tokens) >= len(force[si]):
                        reason = 'forced'
                elif tok in stop_ids:
                    reason = 'eos'
                elif stop_strings and any(x in self.tok.decode(r.tokens[-24:]) for x in stop_strings):
                    reason = 'stop_string'
                if reason is None and len(r.tokens) >= max_new_tokens:
                    reason = 'length'
                if reason is not None:
                    r.finish = reason
                    alive[row] = None
            n_alive = sum(si is not None for si in alive)
            if n_alive == 0:
                break
            if n_alive <= len(alive) // 2:   # drop finished rows from the batch
                keep = [row for row, si in enumerate(alive) if si is not None]
                kt = torch.tensor(keep, device=dev)
                self.K = self.K.index_select(1, kt)
                self.V = self.V.index_select(1, kt)
                pad, valid, nxt = pad[kt], valid[kt], nxt[kt]
                alive = [alive[row] for row in keep]
            step_pos = (cur - pad)[:, None]
            mask = valid[:, None, None, :cur + 1]
            hs = self._step(nxt[:, None], cur, step_pos, mask)
            cur += 1
        self.K = self.V = None
        return results


def stack_stats(steps: list[dict]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([s[key] for s in steps]) for key in steps[0]}
