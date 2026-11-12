#!/usr/bin/env python
"""Phase-1 analysis (plan.md §3-§5): calibration of B, the four methods, metrics with
problem-cluster bootstrap CIs, exit-rate curves, figures, and the implementation
self-checks. Reads only the compact per-position arrays written by collect.py.

    python scripts/analyze.py --run runs/v1 [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from loopexp.bounded import CHECK_GRID, bounded, compact_bounded_argmax  # noqa: E402

TARGETS = (0.10, 0.25, 0.50, 0.75)
DATASETS = ('gsm8k', 'math500', 'mbppplus')
LABEL = dict(gsm8k='GSM8K', math500='MATH-500', mbppplus='MBPP+')
N_BOOT = 1000
CURVE_RATES = np.round(np.linspace(0.0, 1.0, 101), 3)


# ----------------------------------------------------------------------------- loading
class Positions:
    """All generated positions of one dataset (calibration + test)."""

    def __init__(self, run: Path, ds: str, device: str):
        files = sorted((run / ds).glob('shard*/b*.jsonl'))
        if not files:
            raise FileNotFoundError(f'no completed batches under {run / ds}')
        cols, probs, full = {}, [], []
        offset = 0
        for f in files:
            lines = [json.loads(x) for x in open(f)]
            z = np.load(f.with_suffix('.npz'))
            base = len(probs)
            probs += lines
            for key in z.files:
                if key in ('full_rows', 'full_z2', 'full_z4'):
                    continue
                v = z[key]
                if key == 'prob':
                    v = v + base
                cols.setdefault(key, []).append(v)
            if len(z['full_rows']):
                full.append((z['full_rows'] + offset, z['full_z2'], z['full_z4']))
            offset += len(z['a2'])
        a = {k: np.concatenate(v) for k, v in cols.items()}
        self.probs = probs
        self.n = len(a['a2'])
        dev = torch.device(device)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)  # noqa: E731
        self.prob = a['prob'].astype(np.int64)
        self.split = np.array([probs[i]['split'] for i in self.prob])
        self.truncated = np.array([probs[i]['finish'] == 'length' for i in self.prob])
        self.a2, self.a4, self.token = t(a['a2']).long(), t(a['a4']).long(), t(a['token']).long()
        self.m2 = t(a['m2']).float()
        self.ids = torch.cat([t(a['ids2']), t(a['ids4'])], 1).long()
        self.c2 = torch.cat([t(a['z2_ids2']), t(a['z2_ids4'])], 1).float()
        self.c4 = torch.cat([t(a['z4_ids2']), t(a['z4_ids4'])], 1).float()
        self.cert = t(a['cert']).bool()
        self.grid_arg = t(a['grid_arg']).long()
        self.gap_a4 = t(a['gap_a4']).float()      # z2[a2] - z2[a4] (0 when loops agree)
        self.rank_a4 = a['rank_a4_in_z2']
        self.full = [(t(r), t(z2), t(z4)) for r, z2, z4 in full]
        n_full = sum(len(r) for r, _, _ in self.full)
        if n_full != int((~self.cert).sum()):
            # rows stored in full must cover every uncertified row
            covered = torch.zeros(self.n, dtype=torch.bool, device=dev)
            for r, _, _ in self.full:
                covered[r] = True
            assert bool(covered[~self.cert].all()), 'uncertified rows without full logits'

    def bounded_argmax(self, B: float) -> torch.Tensor:
        out = compact_bounded_argmax(self.ids, self.c2, self.c4, B)
        for r, z2, z4 in self.full:
            out[r] = bounded(z2, z4, B).argmax(-1)
        return out


# ----------------------------------------------------------------------------- metrics
def per_problem_counts(P: Positions, mask: np.ndarray, B: float):
    """Numerators/denominators per problem for every ratio metric at bound B."""
    m = torch.from_numpy(mask).to(P.a2.device)
    exit_ = (P.m2 > 2 * B) & m
    stay = (~(P.m2 > 2 * B)) & m
    bt = P.bounded_argmax(B)
    deep = (P.a2 != P.a4) & m
    out_ee = torch.where(exit_, P.a2, P.a4)
    out_bc = torch.where(exit_, P.a2, bt)
    trunc = stay & (bt != P.a4)
    c = dict(
        pos=m, exit=exit_, exit_deep=exit_ & deep, deep=deep,
        trunc=trunc, trunc_to_a2=trunc & (bt == P.a2),
        div_bc=m & (out_bc != P.a4), div_ee=m & (out_ee != P.a4),
        deep_kept_bc=deep & (out_bc == P.a4), deep_kept_ee=deep & (out_ee == P.a4),
        stay_deep=stay & deep,
        # necessary condition for a non-exit deep correction to survive: z2 gap < 2B
        stay_deep_reachable=stay & deep & (P.gap_a4 < 2 * B),
    )
    n_prob = len(P.probs)
    idx = torch.from_numpy(P.prob).to(P.a2.device)
    counts = {k: torch.bincount(idx, weights=v.double(), minlength=n_prob).cpu().numpy()
              for k, v in c.items()}
    guarantee_viol = int((exit_ & (bt != P.a2)).sum())
    return counts, guarantee_viol


RATIOS = {  # metric: (numerator, denominator)
    'exit_rate': ('exit', 'pos'),
    'exit_disagree': ('exit_deep', 'exit'),          # among exits, argmax z2 != argmax z4
    'trunc_cost': ('trunc', 'pos'),                  # among all: not exited and z4~ != z4
    'retention_bc': ('deep_kept_bc', 'deep'),        # deep corrections kept, bounded
    'retention_ee': ('deep_kept_ee', 'deep'),        # deep corrections kept, early exit
    'div_fixed2': ('deep', 'pos'),
    'div_ee': ('div_ee', 'pos'),
    'div_bc': ('div_bc', 'pos'),
    'lost_at_exit': ('exit_deep', 'deep'),           # share of deep corrections lost by exiting
    'lost_by_trunc': ('trunc', 'deep'),              # ... lost by truncation at non-exit positions
    'trunc_among_stay_deep': ('trunc', 'stay_deep'),
    'trunc_reverts_to_a2': ('trunc_to_a2', 'trunc'),
    'reachable_among_stay_deep': ('stay_deep_reachable', 'stay_deep'),
}


def ratios(counts, weights=None):
    out = {}
    for name, (num, den) in RATIOS.items():
        a, b = counts[num], counts[den]
        if weights is None:
            out[name] = float(a.sum() / b.sum()) if b.sum() > 0 else float('nan')
        else:
            d = weights @ b
            out[name] = np.where(d > 0, (weights @ a) / np.maximum(d, 1e-12), np.nan)
    return out


def boot_weights(problem_ids: np.ndarray, n_prob: int, seed: int):
    """Multinomial resampling weights over the problems that have positions in the mask."""
    present = np.unique(problem_ids)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(present), np.full(len(present), 1 / len(present)), size=N_BOOT)
    w = np.zeros((N_BOOT, n_prob))
    w[:, present] = draws
    return w


def metrics_with_ci(P: Positions, mask, B, W):
    counts, viol = per_problem_counts(P, mask, B)
    point = ratios(counts)
    boot = ratios(counts, W)
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in boot.items()}
    return point, ci, viol, counts


# ----------------------------------------------------------------------------- calibration
def calibrate_per_dataset(m2_calib: np.ndarray):
    """B_r = q_{1-r}(m2)/2 so that P_calib(m2 > 2B) ~= r."""
    return {r: float(np.quantile(m2_calib, 1 - r) / 2) for r in TARGETS}


def calibrate_pooled(m2_by_ds: dict[str, np.ndarray]):
    """One B for all datasets: equal-weight mean of per-dataset calibration exit rates = r."""
    def rate(B):
        return np.mean([(m > 2 * B).mean() for m in m2_by_ds.values()])
    hi = max(float(m.max()) for m in m2_by_ds.values()) / 2 + 1
    out = {}
    for r in TARGETS:
        lo_, hi_ = 0.0, hi
        for _ in range(100):
            mid = (lo_ + hi_) / 2
            lo_, hi_ = (mid, hi_) if rate(mid) > r else (lo_, mid)
        out[r] = (lo_ + hi_) / 2
    return out


# ----------------------------------------------------------------------------- checks
def self_checks(P: Positions):
    res = dict(n_positions=P.n, cert_rate=float(P.cert.float().mean()),
               token_eq_a4=bool(torch.equal(P.token, P.a4)))
    mism, viol, agree_change = [], [], []
    agree = P.a2 == P.a4
    for g, B in enumerate(CHECK_GRID):
        bt = P.bounded_argmax(B)
        mism.append(int((bt != P.grid_arg[:, g]).sum()))
        ex = P.m2 > 2 * B
        viol.append(int((ex & (bt != P.a2)).sum()))
        agree_change.append(int((agree & (bt != P.a2)).sum()))
    res.update(grid=list(CHECK_GRID), compact_vs_full_mismatch=mism, exit_guarantee_violations=viol,
               agree_rows_changed=agree_change)
    res['pass'] = (res['token_eq_a4'] and sum(mism) == 0 and sum(viol) == 0)
    return res


# ----------------------------------------------------------------------------- figures
COL = dict(ee='#2a78d6', bc='#eb6834', ref='#52514e', grid='#e4e3df', ink='#0b0b0b', ink2='#52514e')


def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(COL['ink2'])
    ax.tick_params(colors=COL['ink2'], labelsize=9)
    ax.grid(True, color=COL['grid'], linewidth=0.8)
    ax.set_axisbelow(True)


def plot_main(curves, table, fixed2, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dss = [d for d in DATASETS if d in curves]
    fig, axes = plt.subplots(2, len(dss), figsize=(4.2 * len(dss), 7.0), squeeze=False)
    for j, ds in enumerate(dss):
        c = curves[ds]
        x = np.array(c['exit_rate']) * 100
        ax = axes[0, j]
        _style(ax)
        ax.axhline(fixed2[ds] * 100, color=COL['ref'], lw=1.2, ls='--')
        ax.text(99, fixed2[ds] * 100, 'fixed 2 loops ', color=COL['ink2'], fontsize=8, va='bottom', ha='right')
        ax.plot(x, np.array(c['div_ee']) * 100, color=COL['ee'], lw=2, label='margin early exit')
        ax.plot(x, np.array(c['div_bc']) * 100, color=COL['bc'], lw=2, label='bounded correction')
        for row in table[ds]:
            ax.plot(row['test']['exit_rate'] * 100, row['test']['div_bc'] * 100, 'o', ms=6,
                    color=COL['bc'], mec='white', mew=1.5, zorder=3)
            ax.plot(row['test']['exit_rate'] * 100, row['test']['div_ee'] * 100, 'o', ms=6,
                    color=COL['ee'], mec='white', mew=1.5, zorder=3)
        ax.set_title(LABEL[ds], fontsize=11, color=COL['ink'], loc='left')
        ax.set_xlabel('provable exit rate (% of positions)', fontsize=9, color=COL['ink2'])
        if j == 0:
            ax.set_ylabel('token divergence vs 4-loop Ouro (%)', fontsize=9, color=COL['ink2'])
            ax.legend(frameon=False, fontsize=8, loc='center left')
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax = axes[1, j]
        _style(ax)
        ax.plot(x, np.array(c['retention_ee']) * 100, color=COL['ee'], lw=2, label='margin early exit')
        ax.plot(x, np.array(c['retention_bc']) * 100, color=COL['bc'], lw=2, label='bounded correction')
        ax.set_xlabel('provable exit rate (% of positions)', fontsize=9, color=COL['ink2'])
        if j == 0:
            ax.set_ylabel('deep corrections kept (%)', fontsize=9, color=COL['ink2'])
            ax.legend(frameon=False, fontsize=8, loc='lower left')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'main.{ext}', dpi=160)
    plt.close(fig)


def plot_m2(P_by_ds, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dss = list(P_by_ds)
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3.2), squeeze=False)
    for j, ds in enumerate(dss):
        P = P_by_ds[ds]
        ax = axes[0, j]
        _style(ax)
        m2 = P.m2.cpu().numpy()
        deep = (P.a2 != P.a4).cpu().numpy()
        bins = np.linspace(0, np.quantile(m2, 0.995), 60)
        ax.hist(m2[~deep], bins=bins, density=True, histtype='step', lw=2, color=COL['ee'],
                label='loop 2 = loop 4')
        ax.hist(m2[deep], bins=bins, density=True, histtype='step', lw=2, color=COL['bc'],
                label='loop 4 corrects loop 2')
        ax.set_title(LABEL[ds], fontsize=11, color=COL['ink'], loc='left')
        ax.set_xlabel('loop-2 top-1 margin m2 (logits)', fontsize=9, color=COL['ink2'])
        if j == 0:
            ax.set_ylabel('density', fontsize=9, color=COL['ink2'])
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'm2_hist.{ext}', dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def fmt_pct(x, ci=None):
    s = f'{100 * x:.2f}'
    return s + (f' [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]' if ci else '')


def write_report(res, out: Path):
    L = ['# Phase-1 results: provable exit at loop 2 under bounded late-loop correction', '',
         f"Run: `{res['run']}`. Token-weighted over generated positions of the **{res['test_split']}** split; "
         '95% CIs from a problem-cluster bootstrap '
         f'({N_BOOT} resamples). Percentages.', '']
    L += ['## Self-checks', '', '| dataset | positions | certified compact rows | token == argmax z4 | '
          'compact vs full mismatches | exit-guarantee violations | agree rows changed | pass |',
          '|---|---:|---:|---|---:|---:|---:|---|']
    for ds, c in res['checks'].items():
        L.append(f"| {LABEL[ds]} | {c['n_positions']} | {100 * c['cert_rate']:.3f}% | {c['token_eq_a4']} | "
                 f"{sum(c['compact_vs_full_mismatch'])} | {sum(c['exit_guarantee_violations'])} | "
                 f"{sum(c['agree_rows_changed'])} | {c['pass']} |")
    L.append('')
    for scheme in ('per_dataset', 'pooled'):
        L += [f'## Calibrated B ({scheme.replace("_", "-")} calibration)', '']
        L += ['| dataset | target | B | calib exit | test exit | exit-pos divergence | truncation cost | '
              'deep kept (bounded) | deep kept (early exit) | div fixed-2 | div early exit | div bounded |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for ds in res['tables'][scheme]:
            for row in res['tables'][scheme][ds]:
                t, ci = row['test'], row['test_ci']
                L.append(
                    f"| {LABEL[ds]} | {int(100 * row['target'])} | {row['B']:.3f} | "
                    f"{fmt_pct(row['calib_exit_rate'])} | {fmt_pct(t['exit_rate'], ci['exit_rate'])} | "
                    f"{fmt_pct(t['exit_disagree'], ci['exit_disagree'])} | {fmt_pct(t['trunc_cost'], ci['trunc_cost'])} | "
                    f"{fmt_pct(t['retention_bc'], ci['retention_bc'])} | {fmt_pct(t['retention_ee'], ci['retention_ee'])} | "
                    f"{fmt_pct(t['div_fixed2'])} | {fmt_pct(t['div_ee'], ci['div_ee'])} | {fmt_pct(t['div_bc'], ci['div_bc'])} |")
        L.append('')
    L += ['## Where deep corrections are lost (per-dataset B, test)', '',
          '| dataset | target | lost by exiting | lost by truncation (non-exit) | truncation hits among non-exit deep | '
          'non-exit deep with z2 gap < 2B | truncated outputs that revert to loop-2 token |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for ds, rows in res['tables']['per_dataset'].items():
        for row in rows:
            t = row['test']
            L.append(f"| {LABEL[ds]} | {int(100 * row['target'])} | {fmt_pct(t['lost_at_exit'])} | "
                     f"{fmt_pct(t['lost_by_trunc'])} | {fmt_pct(t['trunc_among_stay_deep'])} | "
                     f"{fmt_pct(t['reachable_among_stay_deep'])} | {fmt_pct(t['trunc_reverts_to_a2'])} |")
    L += ['', '## Robustness: excluding length-truncated responses (per-dataset B, test)', '',
          '| dataset | target | test exit | truncation cost | deep kept (bounded) | div bounded |',
          '|---|---:|---:|---:|---:|---:|']
    for ds, rows in res['tables']['per_dataset_no_trunc'].items():
        for row in rows:
            t = row['test']
            L.append(f"| {LABEL[ds]} | {int(100 * row['target'])} | {fmt_pct(t['exit_rate'])} | "
                     f"{fmt_pct(t['trunc_cost'])} | {fmt_pct(t['retention_bc'])} | {fmt_pct(t['div_bc'])} |")
    L += ['', '## Reference trajectories', '',
          '| dataset | problems (calib/test) | positions (calib/test) | length-truncated | sanity accuracy |',
          '|---|---|---|---:|---:|']
    for ds, s in res['trajectories'].items():
        L.append(f"| {LABEL[ds]} | {s['n_calib']}/{s['n_test']} | {s['pos_calib']}/{s['pos_test']} | "
                 f"{fmt_pct(s['truncated'])} | {s['accuracy']} |")
    L += ['', 'Figures: `main.pdf` (divergence and deep-correction retention vs exit rate), '
          '`m2_hist.pdf`. Curves: `curves.json`.', '']
    (out / 'report.md').write_text('\n'.join(L))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--test-split', default='test', choices=['test', 'calib'],
                    help="'calib' only for smoke runs that contain no test problems")
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out) if a.out else run / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    P_by_ds = {ds: Positions(run, ds, a.device) for ds in a.datasets.split(',') if (run / ds).exists()}

    res = dict(run=str(run), test_split=a.test_split, checks={}, tables=dict(per_dataset={}, pooled={}, per_dataset_no_trunc={}),
               curves={}, trajectories={})
    m2_calib = {ds: P.m2.cpu().numpy()[P.split == 'calib'] for ds, P in P_by_ds.items()}
    B_pooled = calibrate_pooled(m2_calib)
    fixed2 = {}
    for ds, P in P_by_ds.items():
        print(f'[{ds}] {P.n} positions, self-checks...', flush=True)
        res['checks'][ds] = self_checks(P)
        test = P.split == a.test_split
        calib = P.split == 'calib'
        W = boot_weights(P.prob[test], len(P.probs), a.seed)
        W_nt = boot_weights(P.prob[test & ~P.truncated], len(P.probs), a.seed)
        B_ds = calibrate_per_dataset(m2_calib[ds])
        for scheme, Bs, mask, Wm in (('per_dataset', B_ds, test, W), ('pooled', B_pooled, test, W),
                                     ('per_dataset_no_trunc', B_ds, test & ~P.truncated, W_nt)):
            rows = []
            for r, B in Bs.items():
                point, ci, viol, _ = metrics_with_ci(P, mask, B, Wm)
                calib_rate = float((P.m2.cpu().numpy()[calib] > 2 * B).mean())
                rows.append(dict(target=r, B=B, calib_exit_rate=calib_rate, test=point, test_ci=ci,
                                 guarantee_violations=viol))
            res['tables'][scheme][ds] = rows
        # dense curve on the test split, parametrised by the test exit rate
        m2t = P.m2.cpu().numpy()[test]
        curve = {k: [] for k in ['B', *RATIOS]}
        for rate in CURVE_RATES:
            B = float(np.quantile(m2t, 1 - rate) / 2) if 0 < rate < 1 else (float(m2t.max()) / 2 + 1 if rate == 0 else 0.0)
            counts, _ = per_problem_counts(P, test, B)
            pt = ratios(counts)
            curve['B'].append(B)
            for k in RATIOS:
                curve[k].append(pt[k])
        res['curves'][ds] = curve
        fixed2[ds] = res['tables']['per_dataset'][ds][0]['test']['div_fixed2']
        lines = P.probs
        acc = [x['correct'] for x in lines if x.get('correct') is not None]
        syn = [x['syntax_ok'] for x in lines if 'syntax_ok' in x]
        res['trajectories'][ds] = dict(
            n_calib=sum(x['split'] == 'calib' for x in lines), n_test=sum(x['split'] == 'test' for x in lines),
            pos_calib=int(calib.sum()), pos_test=int(test.sum()),
            truncated=float(np.mean([x['finish'] == 'length' for x in lines])),
            accuracy=(f'{100 * np.mean(acc):.1f}% correct' if acc else
                      f'{100 * np.mean(syn):.1f}% syntax-valid code (pass@1 not run in phase 1)'))
    res['B_pooled'] = B_pooled
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    (out / 'curves.json').write_text(json.dumps(res['curves'], default=float))
    plot_main(res['curves'], res['tables']['per_dataset'], fixed2, out)
    plot_m2(P_by_ds, out)
    write_report(res, out)
    print((out / 'report.md').read_text())


if __name__ == '__main__':
    main()
