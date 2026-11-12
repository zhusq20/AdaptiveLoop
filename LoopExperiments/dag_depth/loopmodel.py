"""Ouro forward with access to the state at every loop boundary.

Runs the official decoder-layer modules in one parallel pass (no KV cache). Ouro's only
state passed from loop r to loop r+1 is the normed hidden state h_r of every position
(each loop has its own KV, built from that loop's own inputs), so h_r is a complete
intervention point: replacing h_r at some positions and continuing reproduces exactly
what the model would compute from that state. h_0 is the token embedding.

Readout after loop r is lm_head(h_r), the model's trained per-loop exit prediction.
Running R loops and reading loop r < R is identical to running r loops, because loop r
never depends on later loops.
"""
from __future__ import annotations

import torch

MODEL_DIR = '/mnt/disk2_from_server2/siqizhu4/models/ouro-1.4b'
DIGIT_IDS = list(range(32, 42))   # '0'..'9'

torch.backends.cuda.matmul.allow_tf32 = False


class LoopLM:
    def __init__(self, model_dir: str = MODEL_DIR, device: str = 'cuda'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval()
        self.inner = self.lm.model
        self.layers = list(self.inner.layers[: self.lm.config.num_hidden_layers])
        self.device = torch.device(device)
        self.W = self.lm.lm_head.weight.detach().float()

    def encode(self, texts: list[str]):
        ids = [self.tok(t, add_special_tokens=False).input_ids for t in texts]
        P = max(map(len, ids))
        pad = torch.tensor([P - len(x) for x in ids], device=self.device)
        out = torch.zeros(len(ids), P, dtype=torch.long, device=self.device)
        for i, x in enumerate(ids):
            out[i, P - len(x):] = torch.tensor(x, device=self.device)
        return out, pad

    @torch.no_grad()
    def run(self, ids: torch.Tensor, pad: torch.Tensor, n_loops: int, patch=None,
            keep_states: bool = False, readout_pos: int = -1):
        """ids [B, S] left-padded. patch: callable (r, h) -> h applied to h_r for
        r = 0..n_loops-1 before loop r+1 runs. Returns FP32 logits at readout_pos after
        each loop [n_loops, B, V] and, optionally, states [n_loops + 1, B, S, H] (h_0
        is the embedding; stored states are pre-patch)."""
        B, S = ids.shape
        k = torch.arange(S, device=self.device)
        valid = k[None] >= pad[:, None]
        mask = (k[None, :] <= k[:, None])[None] & valid[:, None, :]
        mask |= torch.eye(S, dtype=torch.bool, device=self.device)[None]
        mask = mask[:, None]
        pos = (k[None] - pad[:, None]).clamp(min=0)
        x = self.inner.embed_tokens(ids)
        pe = self.inner.rotary_emb(x, pos)
        states = [x] if keep_states else None
        logits = []
        for r in range(n_loops):
            if patch is not None:
                x = patch(r, x)
            for layer in self.layers:
                x = layer(x, attention_mask=mask, position_ids=pos, past_key_value=None,
                          use_cache=False, position_embeddings=pe)
            x = self.inner.norm(x)
            if keep_states:
                states.append(x)
            logits.append(x[:, readout_pos].float() @ self.W.T)
        logits = torch.stack(logits)
        if keep_states:
            return logits, torch.stack(states)
        return logits
