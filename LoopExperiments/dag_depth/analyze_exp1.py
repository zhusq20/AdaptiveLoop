#!/usr/bin/env python
"""Exp 1 analysis: accuracy per loop by (N, D), fixed-N and fixed-D slices, and a
per-loop logistic regression of correctness on D and N with problem-clustered SEs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(run: Path) -> tuple[pd.DataFrame, np.ndarray]:
    z = np.load(run / 'probs.npz')
    items = pd.DataFrame([json.loads(l) for l in (run / 'items.jsonl').read_text().splitlines()])
    return items, z['probs']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--loops', default='1,2,3,4,6,8')
    a = ap.parse_args()
    run = Path(a.run)
    df, probs = load(run)
    R = probs.shape[1]
    loops = [int(x) for x in a.loops.split(',') if int(x) <= R]
    ans = df['answer'].to_numpy()
    for r in range(1, R + 1):
        df[f'acc{r}'] = probs[:, r - 1].argmax(-1) == ans
        df[f'p{r}'] = probs[np.arange(len(df)), r - 1, ans]
    pd.set_option('display.width', 200)
    lines = []

    def show(title, t):
        lines.append(f'\n## {title}\n')
        lines.append(t.to_string())
        print(f'\n## {title}\n{t.to_string()}')

    acc_cols = [f'acc{r}' for r in loops]
    g = df.groupby(['N', 'D', 'L1', 'L2'])[acc_cols].mean().round(2)
    g['n'] = df.groupby(['N', 'D', 'L1', 'L2']).size()
    show('Accuracy by cell (both orders pooled), columns = loop', g)

    show('Accuracy by order (L2 > 0 only)',
         df[df.L2 > 0].groupby(['order'])[acc_cols].mean().round(3))

    # fixed-N slice: vary D at constant op count / text length
    show('Fixed N: accuracy at loop 4 by D (rows N, cols D)',
         df.pivot_table(index='N', columns='D', values='acc4', aggfunc='mean').round(2))
    # fixed-D slice: vary N (parallel ops added in branch 2)
    show('Fixed D: accuracy at loop 4 by N (rows D, cols N)',
         df.pivot_table(index='D', columns='N', values='acc4', aggfunc='mean').round(2))
    show('Fixed D: accuracy at loop 1 by N (rows D, cols N)',
         df.pivot_table(index='D', columns='N', values='acc1', aggfunc='mean').round(2))
    for r in (2, 3, 4):
        if r <= R:
            show(f'Loop {r}: accuracy by L1 (rows, D = L1 + 1) x L2 (cols, extra parallel ops)',
                 df.pivot_table(index='L1', columns='L2', values=f'acc{r}', aggfunc='mean').round(2))

    # matched-N contrasts: serial chain (L2 = 0) vs every two-branch split, per loop
    rows = []
    for N, g_ in df.groupby('N'):
        ch = g_[g_.L2 == 0]
        for (L1, L2), tb in g_[g_.L2 > 0].groupby(['L1', 'L2']):
            for r in loops:
                p1, p0 = tb[f'acc{r}'].mean(), ch[f'acc{r}'].mean()
                se = np.sqrt(p1 * (1 - p1) / len(tb) + p0 * (1 - p0) / len(ch))
                rows.append(dict(N=N, split=f'({L1},{L2})', D_tb=L1 + 1, D_chain=N, loop=r,
                                 tb=round(p1, 3), chain=round(p0, 3), diff=round(p1 - p0, 3),
                                 ci95=f'[{p1 - p0 - 1.96 * se:+.3f}, {p1 - p0 + 1.96 * se:+.3f}]'))
    show('Matched N: two-branch minus serial chain (hypothesis predicts diff > 0)',
         pd.DataFrame(rows).set_index(['N', 'split', 'loop']))

    show('Shortcut check: fraction of answers equal to a literal, by D',
         df.groupby('D')['ans_is_literal'].mean().round(2).to_frame())

    # logistic regression per loop: correct ~ D + N  (separate slopes)
    try:
        import statsmodels.formula.api as smf
        rows = []
        for r in loops:
            d = df.assign(y=df[f'acc{r}'].astype(int))
            fit = smf.logit('y ~ D + N', d).fit(disp=0)
            rows.append(dict(loop=r, b_D=fit.params['D'], se_D=fit.bse['D'],
                             b_N=fit.params['N'], se_N=fit.bse['N']))
        show('Logit(correct) ~ D + N per loop', pd.DataFrame(rows).set_index('loop').round(3))
    except Exception as e:  # noqa: BLE001
        print('regression skipped:', e)

    (run / 'analysis.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
