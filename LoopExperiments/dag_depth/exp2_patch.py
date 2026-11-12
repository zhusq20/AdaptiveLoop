#!/usr/bin/env python
"""Exp 2: two-source interchange intervention at a loop boundary (TB(1,1) programs).

Base:     a = A0 / b = B0 / p = a op1 c1 / q = b op2 c2 / m = p opm q / print(m)
Source 1: same text except A0' and c1' (both differ from base), so p = A' != A.
Source 2: same for branch 2, q = B'.
Each target value A' (B') is reached by several routes (different A0', op1', c1').

At boundary r (h_r, the state fed into loop r+1; r = 0 is the embedding) we copy the
source states of a set of branch positions into the base run and continue to loop 4.
Position sets:
  line   - tokens of the branch's op line (`p = a + 2\n`) whose token is identical in
           base and source (the differing literal token is left unpatched)
  branch - tokens of both branch lines (init + op) whose token is identical
  full   - every token of both branch lines (r = 0: plain text substitution, i.e. the
           counterfactual program; positive control)
Modes: once (patch only at boundary r) or onward (patch at every boundary >= r).
Swaps: A (branch 1 from source 1), B, AB (both, each from its own source).
Targets: Y0 = A opm B, YA = A' opm B, YB = A opm B', YAB = A' opm B'.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch

from loopmodel import DIGIT_IDS, LoopLM
from progs import fewshot_block, prompt, two_branch


def apply(op, x, y):
    return x + y if op == '+' else x - y


def routes(target: int, init0: int, c0: int) -> list[tuple[int, str, int]]:
    """(init', op', c') with init' op' c' = target, init' != init0, c' != c0."""
    out = []
    for i0 in range(10):
        for op in '+-':
            for c in range(1, 10):
                if i0 != init0 and c != c0 and apply(op, i0, c) == target:
                    out.append((i0, op, c))
    return out


def make_item(rng: random.Random, n_routes: int):
    while True:
        base = two_branch(rng, 1, 1, 'grouped')
        if base is None:
            continue
        L = base.lines
        names = [L[0].var, L[1].var, L[2].var, L[3].var, L[4].var]
        A0, B0 = L[0].args[0], L[1].args[0]
        (_, op1, c1), (_, op2, c2) = L[2].args, L[3].args
        mop = base.meta['merge_op']
        A, B = base.meta['A'], base.meta['B']
        cands = []
        for A2, B2 in itertools.product(range(10), range(10)):
            if A2 == A or B2 == B:
                continue
            ys = [apply(mop, A, B), apply(mop, A2, B), apply(mop, A, B2), apply(mop, A2, B2)]
            if not all(0 <= y <= 9 for y in ys) or len(set(ys)) < 4:
                continue
            ra, rb = routes(A2, A0, c1), routes(B2, B0, c2)
            if len(ra) >= n_routes and len(rb) >= n_routes:
                cands.append((A2, B2, ra, rb, ys))
        if not cands:
            continue
        A2, B2, ra, rb, ys = rng.choice(cands)
        ra, rb = rng.sample(ra, n_routes), rng.sample(rb, n_routes)

        def build(a0, l1, b0, l2):
            pr = two_branch(rng, 1, 1, 'grouped', a0=a0, b0=b0, branch_lits=([l1], [l2]),
                            merge_op=mop, names=names)
            assert pr is not None
            pr.check()
            return pr
        srcA = [build(i0, (op, c), B0, (op2, c2)) for i0, op, c in ra]
        srcB = [build(A0, (op1, c1), i0, (op, c)) for i0, op, c in rb]
        dbl = [build(ra[k][0], ra[k][1:], rb[k][0], rb[k][1:]) for k in range(n_routes)]
        assert base.answer == ys[0] and all(s.answer == ys[1] for s in srcA)
        assert all(s.answer == ys[2] for s in srcB) and all(s.answer == ys[3] for s in dbl)
        # operand-copy alternative: source init with base constant (and vice versa)
        altA = [apply(mop, apply(op1, i0, c1), B) for i0, _, _ in ra]
        altB = [apply(mop, A, apply(op2, i0, c2)) for i0, _, _ in rb]
        return dict(base=base, srcA=srcA, srcB=srcB, dbl=dbl, ys=ys, A=A, B=B, A2=A2, B2=B2,
                    routesA=ra, routesB=rb, altA=altA, altB=altB)


def line_token_sets(tok, fs: str, prog, ids: list[int]):
    """Token indices of each program line (including its trailing newline)."""
    enc = tok(prompt(prog, fs), add_special_tokens=False, return_offsets_mapping=True)
    assert enc.input_ids == ids
    spans, start = [], len(fs)
    for l in prog.lines:
        end = start + len(l.render()) + 1
        spans.append((start, end))
        start = end
    sets = []
    for s, e in spans:
        sets.append([t for t, (a, b) in enumerate(enc.offset_mapping) if s <= a < e])
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/mnt/disk2_from_server2/siqizhu4/models/ouro-2.6b')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--routes', type=int, default=3)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    fs = fewshot_block()
    items = [make_item(rng, a.routes) for _ in range(a.n)]
    m = LoopLM(a.model, device='cuda')
    tok = m.tok
    R = 4
    boundaries = [0, 1, 2, 3]
    psets, modes, swaps = ['line', 'branch', 'full'], ['once', 'onward'], ['A', 'B', 'AB']

    def ids_of(p):
        return tok(prompt(p, fs), add_special_tokens=False).input_ids

    # all programs of this family have the same token length -> no padding anywhere
    lens = {len(ids_of(it['base'])) for it in items}
    assert len(lens) == 1, lens
    recs = []
    clean = {}   # unpatched accuracies (base / sources / double counterfactual), loops 1..4
    for s in range(0, len(items), a.batch):
        chunk = items[s:s + a.batch]
        B_ = len(chunk)
        base_ids = torch.tensor([ids_of(it['base']) for it in chunk], device=m.device)
        zero = torch.zeros(B_, dtype=torch.long, device=m.device)
        lg_base = m.run(base_ids, zero, R)
        dig = lambda lg: lg[:, :, DIGIT_IDS].softmax(-1).permute(1, 0, 2).cpu().numpy()  # noqa: E731
        p_base = dig(lg_base)
        # position sets per item (branch 1 = lines 0, 2 ; branch 2 = lines 1, 3)
        lsets = [line_token_sets(tok, fs, it['base'], ids_of(it['base'])) for it in chunk]
        for k in range(a.routes):
            srcA_ids = torch.tensor([ids_of(it['srcA'][k]) for it in chunk], device=m.device)
            srcB_ids = torch.tensor([ids_of(it['srcB'][k]) for it in chunk], device=m.device)
            dbl_ids = torch.tensor([ids_of(it['dbl'][k]) for it in chunk], device=m.device)
            lgA, stA = m.run(srcA_ids, zero, R, keep_states=True)
            lgB, stB = m.run(srcB_ids, zero, R, keep_states=True)
            lgD = m.run(dbl_ids, zero, R)
            pA, pB, pD = dig(lgA), dig(lgB), dig(lgD)
            for j, it in enumerate(chunk):
                key = (s + j, k)
                clean[key] = dict(base=p_base[j], srcA=pA[j], srcB=pB[j], dbl=pD[j])

            def mask_for(branch: int, pset: str, src_ids):
                M = torch.zeros_like(base_ids, dtype=torch.bool)
                for j in range(B_):
                    lines = [0, 2] if branch == 1 else [1, 3]
                    if pset == 'line':
                        lines = lines[1:]
                    pos = [t for li in lines for t in lsets[j][li]]
                    if pset != 'full':
                        pos = [t for t in pos if base_ids[j, t] == src_ids[j, t]]
                    M[j, pos] = True
                return M

            for pset in psets:
                MA, MB = mask_for(1, pset, srcA_ids), mask_for(2, pset, srcB_ids)
                for mode, r0, swap in itertools.product(modes, boundaries, swaps):
                    if mode == 'onward' and r0 == R - 1:
                        continue  # identical to 'once' at the last boundary
                    use_r = {r0} if mode == 'once' else set(range(r0, R))

                    def patch(r, h, MA=MA, MB=MB, swap=swap, use_r=use_r):
                        if r not in use_r:
                            return h
                        h = h.clone()
                        if 'A' in swap:
                            h[MA] = stA[r][MA]
                        if 'B' in swap:
                            h[MB] = stB[r][MB]
                        return h
                    lg = m.run(base_ids, zero, R, patch=patch)
                    pp = dig(lg)
                    for j in range(B_):
                        recs.append(dict(item=s + j, route=k, pset=pset, mode=mode, r=r0,
                                         swap=swap, probs=pp[j].tolist()))
        print(f'{s + B_}/{len(items)}', flush=True)

    meta = []
    for i, it in enumerate(items):
        meta.append(dict(item=i, ys=it['ys'], A=it['A'], B=it['B'], A2=it['A2'], B2=it['B2'],
                         routesA=it['routesA'], routesB=it['routesB'], altA=it['altA'],
                         altB=it['altB'], base_text=it['base'].text,
                         srcA_text=[p.text for p in it['srcA']], srcB_text=[p.text for p in it['srcB']]))
    (out / 'items.jsonl').write_text('\n'.join(json.dumps(x) for x in meta) + '\n')
    with open(out / 'patch.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    np.savez_compressed(out / 'clean.npz', keys=np.array(list(clean.keys())),
                        **{f: np.stack([v[f] for v in clean.values()]) for f in ('base', 'srcA', 'srcB', 'dbl')})
    (out / 'config.json').write_text(json.dumps(vars(a) | {'fewshot': fs}, indent=1))
    print('done', flush=True)


if __name__ == '__main__':
    main()
