"""Paired analysis on held-out programs; bootstrap resamples complete quartets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pilot import DIGIT_IDS


def ci(values, seed=777):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), size=(4000, len(v)))].mean(1)
    return [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--tag', default='adapter')
    args = ap.parse_args()
    run = Path(args.run)
    out = run / args.tag
    data = json.loads((run / 'data.json').read_text())
    rows = json.loads((out / 'rows.json').read_text())
    splits = json.loads((out / 'splits.json').read_text())
    source = np.load(run / 'source_predictions.npz')
    probes = np.load(run / 'probe_predictions.npz')['pred']
    source_y = np.array([s['y'] for s in data['sources']])
    left = np.array([r['left'] for r in rows])
    right = np.array([r['right'] for r in rows])
    plus = np.array([r['op'] == '+' for r in rows])
    labels = np.array([r['y'] for r in rows])
    target_tokens = np.array(DIGIT_IDS)[labels]
    literal_match = np.array([s['y'] in [s['x']] + [c for op, c in s['steps']] for s in data['sources']])
    nc = len(data['cases'])
    assert len(rows) == nc * 3 * 4
    assert [(x['case'],x['route'],x['swap']) for x in rows] == [
        (c,route,swap) for c in range(nc) for route in range(3) for swap in ('base','A','B','AB')]
    # Convert unrestricted token predictions to digits. Invalid tokens never count
    # as a valid text subanswer, even if their index happens to produce a sum.
    lookup = {tid: d for d, tid in enumerate(DIGIT_IDS)}
    native_digits = np.vectorize(lambda x: lookup.get(int(x), -100))(source['pred'])
    base = np.array([r['swap'] == 'base' for r in rows])
    records, conditions, kgrid, predictions = [], [], [], {}
    for r in range(5):
        pp = np.load(out / f'test_r{r}.npz')
        predictions[r] = pp['pred']
        correct = pp['pred'] == target_tokens[:, None]
        cubes = correct.reshape(nc, 3, 4, 4)
        pred_cubes = pp['pred'].reshape(nc, 3, 4, 4)
        per_case = cubes[:, :, 0, 0].mean(1)
        strict = cubes[:, :, :, 0].all(2).mean(1)
        consistent = (pred_cubes[:, :, :, 0] == pred_cubes[:, :1, :, 0]).all(1).mean(1)
        prec = dict(r=r, base_accuracy=float(per_case.mean()), base_ci=ci(per_case),
                    all_accuracy=float(correct[:, 0].mean()),
                    quartet_all_four_correct=float(strict.mean()), quartet_ci=ci(strict),
                    route_consistency=float(consistent.mean()),
                    swap_accuracies={s: float(correct[np.array([z['swap'] == s for z in rows]), 0].mean())
                                     for s in ('base', 'A', 'B', 'AB')},
                    probe_accuracy=float((probes[splits['test'], r] == source_y[splits['test']]).mean()))
        no_literal = base & ~literal_match[left] & ~literal_match[right]
        prec['no_literal_n'] = int(no_literal.sum())
        prec['no_literal_accuracy'] = float(correct[no_literal, 0].mean()) if no_literal.any() else None
        if r > 0:
            lp, rp = native_digits[left, r-1], native_digits[right, r-1]
            text = np.where(plus, lp + rp, lp - rp)
            text_hit = (text == labels) & (lp >= 0) & (rp >= 0)
            text_case = text_hit.reshape(nc, 3, 4)[:, :, 0].mean(1)
            probe_ans = np.where(plus, probes[left, r] + probes[right, r], probes[left, r] - probes[right, r])
            probe_case = (probe_ans == labels).reshape(nc,3,4)[:,:,0].mean(1)
            prec.update(worker_accuracy=float((source['pred'][splits['test'], r-1]
                                               == np.array(DIGIT_IDS)[source_y[splits['test']]]).mean()),
                        text_base_accuracy=float(text_case.mean()),
                        text_quartet_all_four_correct=float(text_hit.reshape(nc,3,4).all(2).mean()),
                        probe_quartet_all_four_correct=float((probe_ans == labels).reshape(nc,3,4).all(2).mean()),
                        latent_minus_text=float((per_case-text_case).mean()),
                        latent_minus_text_ci=ci(per_case-text_case),
                        latent_minus_probe=float((per_case-probe_case).mean()),
                        latent_minus_probe_ci=ci(per_case-probe_case),
                        probe_then_combine_accuracy=float((probe_ans[base] == labels[base]).mean()))
            lwrong, rwrong = lp != source_y[left], rp != source_y[right]
            for name, select in [('both_wrong', lwrong & rwrong), ('any_wrong', lwrong | rwrong),
                                 ('both_correct', ~lwrong & ~rwrong)]:
                select &= base
                conditions.append(dict(r=r, condition=name, n=int(select.sum()),
                                       correct=int(correct[select, 0].sum()),
                                       accuracy=float(correct[select, 0].mean()) if select.any() else None,
                                       text_correct=int(text_hit[select].sum())))
        for k in range(4):
            kgrid.append(dict(r=r, k=k+1, base_accuracy=float(cubes[:, :, 0, k].mean()),
                              all_accuracy=float(correct[:, k].mean())))
        records.append(prec)
    zero_correct = (predictions[0][:, 0] == target_tokens).reshape(nc, 3, 4)[:, :, 0].mean(1)
    for rec in records:
        rr = rec['r']
        hit = (predictions[rr][:, 0] == target_tokens).reshape(nc, 3, 4)[:, :, 0].mean(1)
        rec['minus_r0'] = float((hit-zero_correct).mean())
        rec['minus_r0_ci'] = ci(hit-zero_correct)
    result = dict(primary_receiver_loops=1, records=records, conditions=conditions,
                  grid=kgrid, uncertainty='4000 bootstrap draws over base quartet IDs; all routes and swaps remain grouped. Conditional on this fixed held-out source pool and model seed.')
    (out / 'analysis.json').write_text(json.dumps(result, indent=2))
    invariants = dict(r0_predictions_per_op={})
    for op_name in ('+', '-'):
        select = np.array([row['op'] == op_name for row in rows])
        counts = [len(np.unique(predictions[0][select,k])) for k in range(4)]
        assert counts == [1,1,1,1], ('Depth-zero state unexpectedly transmits instance information', counts)
        invariants['r0_predictions_per_op'][op_name] = counts
    invariants['test_programs_disjoint_from_training_and_validation'] = not (set(splits['test']) & set(splits['train'] + splits['val']))
    assert invariants['test_programs_disjoint_from_training_and_validation']
    (out / 'analysis_checks.json').write_text(json.dumps(invariants, indent=2))
    examples = []
    early = predictions[1][:,0].reshape(nc,3,4)
    for case in data['cases']:
        c = case['id']
        desired = np.array(DIGIT_IDS)[case['ys']]
        if not np.all(early[c] == desired[None]):
            continue
        a,b,ap,bp = [v[0] for v in case['routes']]
        if native_digits[a,0] == source_y[a] and native_digits[b,0] == source_y[b]:
            continue
        examples.append(dict(case=c, op=case['op'], expected=case['ys'],
                             r1_outputs=early[c].tolist(),
                             workers=[dict(program=data['sources'][i], native_r1=int(native_digits[i,0]))
                                      for i in (a,b,ap,bp)]))
        if len(examples) == 5:
            break
    (out / 'early_examples.json').write_text(json.dumps(examples, indent=2))
    table = pd.DataFrame(records)[['r','worker_accuracy','probe_accuracy','text_base_accuracy',
                                    'base_accuracy','quartet_all_four_correct','route_consistency']]
    table.to_csv(out / 'primary.csv', index=False)
    pd.DataFrame(conditions).to_csv(out / 'conditional.csv', index=False)
    pd.DataFrame(kgrid).to_csv(out / 'depth_grid.csv', index=False)
    print(table.round(4).to_string(index=False))
    print(pd.DataFrame(conditions).round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    ax = axes[0]
    ax.plot(range(1, 5), [z['worker_accuracy'] for z in records[1:]], 'o-', label='Worker native readout')
    ax.plot(range(5), [z['probe_accuracy'] for z in records], 's--', label='Supervised linear readout')
    ax.set(title='Individual subanswer', xlabel='Worker loop depth r', ylabel='Held-out accuracy', ylim=(0,1.04), xticks=range(5))
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[1]
    ax.plot(range(5), [z['base_accuracy'] for z in records], 'o-', label='Latent merge (1 receiver loop)')
    ax.plot(range(1, 5), [z['text_base_accuracy'] for z in records[1:]], 'x--', label='Native digit -> text merge')
    ax.plot(range(1, 5), [z['probe_then_combine_accuracy'] for z in records[1:]], 's--', label='Linear readout -> text merge')
    ax.fill_between(range(5), [z['base_ci'][0] for z in records], [z['base_ci'][1] for z in records], alpha=.15)
    ax.set(title='Final composed answer', xlabel='Worker loop depth r', ylabel='Base-case accuracy', ylim=(0,1.04), xticks=range(5))
    ax.legend(fontsize=8, loc='lower right')
    grid = np.array([z['base_accuracy'] for z in kgrid]).reshape(5,4)
    ax = axes[2]
    im = ax.imshow(grid, origin='lower', vmin=0, vmax=1, aspect='auto', cmap='viridis')
    for r in range(5):
        for k in range(4):
            ax.text(k,r,f'{grid[r,k]:.1%}',ha='center',va='center',color='white' if grid[r,k]<.55 else 'black')
    ax.set(xlabel='Receiver loops k (trained at k=1)', ylabel='Worker loop depth r',
           xticks=range(4), xticklabels=range(1,5), yticks=range(5))
    fig.colorbar(im, ax=ax, label='Base-case accuracy')
    fig.savefig(out / 'accuracy.png', dpi=180)
    fig.savefig(out / 'accuracy.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
