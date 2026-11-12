#!/usr/bin/env python
"""Exp 2b analysis: P(answer moves to the source's answer) for single-position patches,
by position and loop boundary. Items: base and the source both answered right at loop 4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / 'config.json').read_text())
    z = np.load(run / 'trace.npz')
    res, clean = z['res'].astype(np.float32), z['clean']
    items = [json.loads(l) for l in (run / 'items.jsonl').read_text().splitlines()]
    ys = np.array([it['ys'] for it in items])            # [n, 4] = Y0, YA, YB, YAB
    labels = [l.replace('\n', '\\n') for l in cfg['labels']]
    R = res.shape[2]
    lines = []
    tabs = {}
    for si, name in enumerate(cfg['srcs']):
        tgt = ys[:, si + 1]
        ok = (clean[:, 0, -1].argmax(-1) == ys[:, 0]) & (clean[:, si + 1, -1].argmax(-1) == tgt)
        pred = res[ok, si, :, :, -1].argmax(-1)          # [n_ok, R, P] at loop 4
        hit = (pred == tgt[ok, None, None]).mean(0)
        stay = (pred == ys[ok, 0, None, None]).mean(0)
        tabs[name] = hit
        lines.append(f'\n## source {name}: n_valid={ok.sum()}  P(target) per (boundary r, position); '
                     f'P(base) in parentheses')
        lines.append('pos      ' + ' '.join(f'{r:>11}' for r in range(R)))
        for pi, lab in enumerate(labels):
            lines.append(f'{lab!r:9}' + ' '.join(f'{hit[r, pi]:5.2f}({stay[r, pi]:4.2f})' for r in range(R)))
    txt = '\n'.join(lines)
    print(txt)
    (run / 'analysis.md').write_text(txt + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(tabs), 1, figsize=(13, 2.1 * len(tabs) + 0.6), sharex=True)
    for ax, (name, hit) in zip(np.atleast_1d(axs), tabs.items()):
        im = ax.imshow(hit, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        ax.set_yticks(range(R), [f'h{r}' for r in range(R)])
        ax.set_ylabel(f'source {name}', fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    fig.colorbar(im, ax=axs, shrink=0.8, label='P(answer = source answer)')
    fig.suptitle('Single-position patch from source at boundary h_r (once), readout after loop 4', fontsize=10)
    fig.savefig(run / 'trace.png', dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()
