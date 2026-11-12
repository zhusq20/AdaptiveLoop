#!/usr/bin/env python
"""Exp 2 analysis. Interchange accuracy = P(argmax at loop 4 == target) on items where
the unpatched model is right (loop 4) on the base and on every source involved."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--loop', type=int, default=4)
    a = ap.parse_args()
    run = Path(a.run)
    items = {d['item']: d for d in map(json.loads, (run / 'items.jsonl').read_text().splitlines())}
    z = np.load(run / 'clean.npz')
    keys = [tuple(k) for k in z['keys']]
    L = a.loop - 1
    ok = {}
    for i, k in enumerate(keys):
        it = items[k[0]]
        y0, ya, yb, yab = it['ys']
        ok[k] = dict(base=z['base'][i][L].argmax() == y0, A=z['srcA'][i][L].argmax() == ya,
                     B=z['srcB'][i][L].argmax() == yb, AB=z['dbl'][i][L].argmax() == yab)
    cl = pd.DataFrame(ok.values())
    out = ['## Unpatched accuracy at loop %d (base / source A / source B / double counterfactual)' % a.loop,
           cl.mean().round(3).to_string()]

    rows = []
    for line in (run / 'patch.jsonl').read_text().splitlines():
        r = json.loads(line)
        it = items[r['item']]
        y0, ya, yb, yab = it['ys']
        o = ok[(r['item'], r['route'])]
        valid = o['base'] and (o['A'] or 'A' not in r['swap']) and (o['B'] or 'B' not in r['swap'])
        pred = int(np.argmax(r['probs'][L]))
        target = {'A': ya, 'B': yb, 'AB': yab}[r['swap']]
        alt = None
        if r['swap'] == 'A':
            alt = it['altA'][r['route']]
        elif r['swap'] == 'B':
            alt = it['altB'][r['route']]
        rows.append(dict(item=r['item'], route=r['route'], pset=r['pset'], mode=r['mode'], r=r['r'],
                         swap=r['swap'], valid=valid, pred=pred, hit=pred == target, base=pred == y0,
                         single=(r['swap'] == 'AB') and pred in (ya, yb),
                         alt=(alt is not None and alt != target and pred == alt),
                         p_target=r['probs'][L][target]))
    df = pd.DataFrame(rows)
    v = df[df.valid]
    out.append(f'\nvalid (item, route) rows: {v.groupby(["item", "route"]).ngroups} of {df.groupby(["item", "route"]).ngroups}')
    for mode in ('once', 'onward'):
        for what in ('hit', 'base', 'alt', 'single'):
            t = v[v['mode'] == mode].pivot_table(index=['pset', 'swap'], columns='r', values=what, aggfunc='mean').round(3)
            out.append(f'\n## mode={mode}: P({what}) at loop {a.loop}; columns = patched boundary r (h_r feeds loop r+1)')
            out.append(t.to_string())
    # route consistency: among valid items, do the routes that reach the same value agree?
    g = v.groupby(['item', 'pset', 'mode', 'r', 'swap'])
    cons = g['pred'].agg(lambda s: s.nunique() == 1 and len(s) > 1).rename('agree')
    hits = g['hit'].mean().rename('hit_rate')
    c = pd.concat([cons, hits], axis=1).reset_index()
    out.append('\n## route consistency (fraction of items whose routes give the same patched answer)')
    out.append(c[c['mode'] == 'once'].pivot_table(index=['pset', 'swap'], columns='r', values='agree', aggfunc='mean').round(3).to_string())
    txt = '\n'.join(out)
    print(txt)
    (run / 'analysis.md').write_text(txt + '\n')


if __name__ == '__main__':
    main()
