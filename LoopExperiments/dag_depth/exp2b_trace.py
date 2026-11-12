#!/usr/bin/env python
"""Exp 2b: where does each branch value live, loop by loop?

Same TB(1,1) items as exp2_patch. For every token position t of the program and the
`Output: ` suffix and every boundary r, copy h_r[t] from source A (or source B, or the
double counterfactual) into the base run once, continue to loop 4, and record the digit
distribution after every loop.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from exp2_patch import make_item
from loopmodel import DIGIT_IDS, LoopLM
from progs import fewshot_block, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/mnt/disk2_from_server2/siqizhu4/models/ouro-2.6b')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--seed', type=int, default=23)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    fs = fewshot_block()
    items = [make_item(rng, 1) for _ in range(a.n)]
    m = LoopLM(a.model, device='cuda')
    tok = m.tok
    R = 4

    def ids_of(p):
        return tok(prompt(p, fs), add_special_tokens=False).input_ids

    n_fs = len(tok(fs, add_special_tokens=False).input_ids)
    S = len(ids_of(items[0]['base']))
    positions = list(range(n_fs, S))
    labels = [tok.decode([t]) for t in ids_of(items[0]['base'])[n_fs:]]
    srcs = ['A', 'B', 'AB']
    res = np.zeros((a.n, len(srcs), R, len(positions), R, 10), dtype=np.float16)
    clean = np.zeros((a.n, 4, R, 10), dtype=np.float32)
    dig = lambda lg: lg[:, :, DIGIT_IDS].softmax(-1).permute(1, 0, 2)  # noqa: E731
    for s in range(0, a.n, a.batch):
        chunk = items[s:s + a.batch]
        B_ = len(chunk)
        zero = torch.zeros(B_, dtype=torch.long, device=m.device)
        base_ids = torch.tensor([ids_of(it['base']) for it in chunk], device=m.device)
        src_ids = {'A': [it['srcA'][0] for it in chunk], 'B': [it['srcB'][0] for it in chunk],
                   'AB': [it['dbl'][0] for it in chunk]}
        clean[s:s + B_, 0] = dig(m.run(base_ids, zero, R)).cpu().numpy()
        for si, name in enumerate(srcs):
            sid = torch.tensor([ids_of(p) for p in src_ids[name]], device=m.device)
            lg, st = m.run(sid, zero, R, keep_states=True)
            clean[s:s + B_, si + 1] = dig(lg).cpu().numpy()
            for r in range(R):
                for pi, t in enumerate(positions):
                    def patch(rr, h, r=r, t=t):
                        if rr != r:
                            return h
                        h = h.clone()
                        h[:, t] = st[r][:, t]
                        return h
                    res[s:s + B_, si, r, pi] = dig(m.run(base_ids, zero, R, patch=patch)).cpu().numpy()
            print(f'batch {s} src {name} done', flush=True)
    meta = [dict(item=i, ys=it['ys'], altA=it['altA'], altB=it['altB'], base_text=it['base'].text,
                 srcA_text=it['srcA'][0].text, srcB_text=it['srcB'][0].text, dbl_text=it['dbl'][0].text)
            for i, it in enumerate(items)]
    (out / 'items.jsonl').write_text('\n'.join(json.dumps(x) for x in meta) + '\n')
    np.savez_compressed(out / 'trace.npz', res=res, clean=clean, positions=np.array(positions))
    (out / 'config.json').write_text(json.dumps(vars(a) | {'labels': labels, 'srcs': srcs}, indent=1))
    print('done', flush=True)


if __name__ == '__main__':
    main()
