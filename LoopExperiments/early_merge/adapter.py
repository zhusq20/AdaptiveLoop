"""Train only a shared compact interface, using final-answer loss through frozen Ouro.

The adapter has no access to the merge operation, the opposite branch, raw programs,
or subanswer supervision. Receiver and worker backbone parameters stay frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pilot import Engine, DIGIT_IDS, rows_for_cases, pack_logits, concatenate, value


class Adapter(nn.Module):
    def __init__(self, mean_embedding, width=128):
        super().__init__()
        h = mean_embedding.numel()
        self.norm = nn.LayerNorm(h, elementwise_affine=False)
        self.up = nn.Linear(h, width)
        self.down = nn.Linear(width, h)
        nn.init.normal_(self.down.weight, std=0.0005)
        nn.init.zeros_(self.down.bias)
        self.register_buffer('origin', mean_embedding.float().clone())

    def forward(self, x):
        return self.origin + self.down(torch.nn.functional.gelu(self.up(self.norm(x.float()))))


def split_ids(data):
    # All choices use source-program identity. Validation uses 1/6 of the training
    # expressions, and held-out evaluation uses the original untouched test split.
    result = {'train': [], 'val': [], 'test': []}
    for y in range(10):
        all_train = [e['id'] for e in data['sources'] if e['y'] == y and e['split'] == 'train']
        nv = max(1, len(all_train) // 6)
        result['train'] += all_train[:-nv]
        result['val'] += all_train[-nv:]
        result['test'] += [e['id'] for e in data['sources'] if e['y'] == y and e['split'] == 'test']
    return result


def sample_pairs(data, ids, n, rng):
    groups = [[i for i in ids if data['sources'][i]['y'] == v] for v in range(10)]
    triples = [(a, b, op, value(a, op, b)) for a in range(10) for b in range(10) for op in '+-'
               if 0 <= value(a, op, b) <= 9]
    t = [triples[i] for i in rng.integers(len(triples), size=n)]
    return dict(left=np.array([rng.choice(groups[a]) for a, b, op, y in t]),
                right=np.array([rng.choice(groups[b]) for a, b, op, y in t]),
                op=np.array([0 if op == '+' else 1 for a, b, op, y in t]),
                y=np.array([y for a, b, op, y in t]))


def receiver_embeddings(adapter, hs, base, slots, left, right, op, depth):
    x = base[op].clone()
    # No other state reaches the receiver; 2 x hidden_size floats per example.
    x[:, slots[0]] = adapter(hs[left, depth]).to(x.dtype)
    x[:, slots[1]] = adapter(hs[right, depth]).to(x.dtype)
    return x


@torch.no_grad()
def evaluate(m, adapter, hs, base, slots, pairs, depth, loops=1, batch=64):
    parts = []
    for start in range(0, len(pairs['y']), batch):
        arrays = {key: torch.as_tensor(v[start:start+batch], device=m.device) for key, v in pairs.items()}
        x = receiver_embeddings(adapter, hs, base, slots, arrays['left'], arrays['right'], arrays['op'], depth)
        lg, _ = m.forward_embeds(x, loops)
        parts.append(pack_logits(lg))
    return concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--model', default='/mnt/disk2_from_server2/siqizhu4/models/ouro-2.6b')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--loops', type=int, default=1)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--seed', type=int, default=31415)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--tag', default='adapter')
    args = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = Path(args.run)
    out = run / args.tag
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((run / 'data.json').read_text())
    splits = split_ids(data)
    (out / 'splits.json').write_text(json.dumps(splits))
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    m = Engine(args.model)
    base, slots, digits = m.receiver(data['prefix'])
    hs = torch.load(run / 'source_states.pt', weights_only=True).to(m.device)
    adapter = Adapter(digits.float().mean(0), args.width).to(m.device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    val = sample_pairs(data, splits['val'], 256, np.random.default_rng(9001))
    trace, best = [], -1.
    print('trainable parameters:', sum(p.numel() for p in adapter.parameters()),
          'source split sizes:', {k: len(v) for k, v in splits.items()}, flush=True)
    start = time.perf_counter()
    if not args.eval_only:
        for step in range(1, args.steps + 1):
            pairs = sample_pairs(data, splits['train'], args.batch, rng)
            arr = {key: torch.as_tensor(v, device=m.device) for key, v in pairs.items()}
            # First 1/4 of training makes a completed-state interface; then the same
            # adapter trains over mixed depths (0 included). Total budget is fixed.
            if step <= args.steps // 4:
                depths = 4
            else:
                depths = torch.as_tensor(rng.integers(0, 5, size=args.batch), device=m.device)
            x = receiver_embeddings(adapter, hs, base, slots, arr['left'], arr['right'], arr['op'], depths)
            lg, _ = m.forward_embeds(x, args.loops, final_only=True)
            targets = m.digit_ids[arr['y']]
            loss = torch.nn.functional.cross_entropy(lg[:, -1], targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
            opt.step()
            if step == 1 or step % 25 == 0:
                rec = dict(step=step, loss=float(loss.detach()),
                           train_accuracy=float((lg[:, -1].argmax(-1) == targets).float().mean()),
                           elapsed_seconds=time.perf_counter() - start,
                           max_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
                if step % 50 == 0 or step == args.steps:
                    accuracies = []
                    for r in range(1, 5):
                        pp = evaluate(m, adapter, hs, base, slots, val, r, args.loops)
                        accuracies.append(float((pp['pred'][:, -1] == np.array(DIGIT_IDS)[val['y']]).mean()))
                    rec['val_accuracy_r1_to_r4'] = accuracies
                    score = float(np.mean(accuracies))
                    # Only mixed-depth checkpoints compete for primary selection.
                    if step > args.steps // 4 and score > best:
                        best = score
                        torch.save(adapter.state_dict(), out / 'best.pt')
                        (out / 'selection.json').write_text(json.dumps(dict(step=step, score=score,
                                                                        val_accuracy=accuracies), indent=2))
                trace.append(rec)
                print(json.dumps(rec), flush=True)
                (out / 'training.json').write_text(json.dumps(trace, indent=2))
        torch.save(adapter.state_dict(), out / 'last.pt')
        assert all(p.grad is None for p in m.lm.parameters()), 'Frozen backbone acquired gradients'
    adapter.load_state_dict(torch.load(out / 'best.pt', weights_only=True, map_location=m.device))
    rows = rows_for_cases(data, routes=3)
    (out / 'rows.json').write_text(json.dumps(rows))
    pairs = dict(left=np.array([r['left'] for r in rows]), right=np.array([r['right'] for r in rows]),
                 op=np.array([0 if r['op'] == '+' else 1 for r in rows]), y=np.array([r['y'] for r in rows]))
    summary = []
    for r in range(5):
        started = time.perf_counter()
        pp = evaluate(m, adapter, hs, base, slots, pairs, r, loops=4)
        np.savez_compressed(out / f'test_r{r}.npz', **pp)
        # Primary result uses the receiver depth on which the adapter was trained;
        # deeper readouts are descriptive, with no new checkpoint selection.
        acc = (pp['pred'] == np.array(DIGIT_IDS)[pairs['y'], None]).mean(0)
        rec = dict(r=r, accuracy_k1_to_k4=acc.tolist(), seconds=time.perf_counter()-started)
        summary.append(rec)
        print('held-out', rec, flush=True)
        (out / 'test_summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'runtime.json').write_text(json.dumps(dict(total_seconds=time.perf_counter()-start,
                                                    max_memory_gb=torch.cuda.max_memory_allocated()/1e9), indent=2))
    print('adapter complete', flush=True)


if __name__ == '__main__':
    main()
