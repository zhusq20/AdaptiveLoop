#!/usr/bin/env python
"""Exp 1: accuracy after each loop vs dependency structure (TB family).

For every cell (L1, L2, order) samples n answer-balanced programs, runs Ouro for
--loops loops in one pass and stores the 10-way digit distribution after each loop.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from loopmodel import DIGIT_IDS, MODEL_DIR, LoopLM
from progs import balanced_cell, fewshot_block, literals, prompt


def cells(max_n: int):
    for L1 in range(1, max_n):
        for L2 in range(0, L1 + 1):
            if L1 + L2 + 1 > max_n:
                continue
            for order in (('grouped',) if L2 == 0 else ('grouped', 'interleaved')):
                yield L1, L2, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--max-n', type=int, default=9)
    ap.add_argument('--loops', type=int, default=8)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fewshot-seed', type=int, default=12345)
    ap.add_argument('--model', default=MODEL_DIR)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(a.seed)
    fs = fewshot_block(a.fewshot_seed)
    items = []
    for L1, L2, order in cells(a.max_n):
        for pr in balanced_cell(rng, L1, L2, order, a.n):
            items.append(pr)
    print(f'{len(items)} items', flush=True)

    m = LoopLM(a.model, device='cuda')
    probs = np.zeros((len(items), a.loops, 10), dtype=np.float32)
    top_is_digit = np.zeros((len(items), a.loops), dtype=bool)
    order_idx = sorted(range(len(items)), key=lambda i: len(items[i].text))
    for s in range(0, len(items), a.batch):
        idx = order_idx[s:s + a.batch]
        ids, pad = m.encode([prompt(items[i], fs) for i in idx])
        lg = m.run(ids, pad, a.loops)                     # [R, B, V]
        probs[idx] = lg[:, :, DIGIT_IDS].softmax(-1).permute(1, 0, 2).cpu().numpy()
        top_is_digit[idx] = torch.isin(lg.argmax(-1), torch.tensor(DIGIT_IDS, device=lg.device)).T.cpu().numpy()
        print(f'{s + len(idx)}/{len(items)}', flush=True)

    meta = []
    for pr in items:
        d = dict(pr.meta)
        d['answer'] = pr.answer
        d['text'] = pr.text
        d['ans_is_literal'] = pr.answer in literals(pr)
        meta.append(d)
    np.savez_compressed(out / 'probs.npz', probs=probs, top_is_digit=top_is_digit)
    (out / 'items.jsonl').write_text('\n'.join(json.dumps(x) for x in meta) + '\n')
    (out / 'config.json').write_text(json.dumps(vars(a) | {'fewshot': fs}, indent=1))
    print('done', flush=True)


if __name__ == '__main__':
    main()
