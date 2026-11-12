#!/usr/bin/env python
"""Implementation self-check of the re-implemented Ouro forward.

(1) official vs ours: the official remote-code OuroModel run teacher-forced on
    prompt + generated tokens (no cache, one parallel pass) vs our batched incremental
    decode with per-loop KV. Compared on loop-2 and loop-4 FP32 readouts.
(2) batch vs single: the same continuations replayed with batch size 1.
Both differ only by bf16 kernel/shape effects; we report max |dz|, argmax agreement
and |dm2|, and require argmax agreement >= 99% (tolerance for bf16 near-ties).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from loopexp.bounded import center, top2_margin  # noqa: E402
from loopexp.data import build_prompt, load_problems, stop_ids  # noqa: E402
from loopexp.ouro import OuroRunner  # noqa: E402

MODEL_DIR = '/mnt/disk2_from_server2/siqizhu4/models/ouro-1.4b'


def full_rows(r):
    z2 = torch.stack([x[1] for x in r.full])
    z4 = torch.stack([x[2] for x in r.full])
    return z2, z4


def cmp(name, a2, a4, b2, b4):
    d = dict(
        n=a2.shape[0],
        max_abs_dz2=float((a2 - b2).abs().max()), max_abs_dz4=float((a4 - b4).abs().max()),
        mean_abs_dz4=float((a4 - b4).abs().mean()),
        argmax2_agree=float((a2.argmax(-1) == b2.argmax(-1)).float().mean()),
        argmax4_agree=float((a4.argmax(-1) == b4.argmax(-1)).float().mean()),
        max_abs_dm2=float((top2_margin(a2) - top2_margin(b2)).abs().max()),
        median_abs_dm2=float((top2_margin(a2) - top2_margin(b2)).abs().median()),
    )
    print(name, json.dumps(d))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--per-dataset', type=int, default=2)
    ap.add_argument('--max-new-tokens', type=int, default=96)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'validate_forward.json'))
    a = ap.parse_args()

    runner = OuroRunner(MODEL_DIR, a.device)
    tok, dev = runner.tok, runner.device
    probs = []
    for ds in ('gsm8k', 'math500', 'mbppplus'):
        probs += [p for p in load_problems(ds) if p['split'] == 'calib'][:a.per_dataset]
    prompts = [build_prompt(p, tok) for p in probs]
    stops = stop_ids(tok)
    batch = runner.generate(prompts, a.max_new_tokens, stops, keep_full=True)
    report = dict(problems=[p['pid'] for p in probs], official_vs_ours=[], single_vs_batch=[])

    for p, pr, r in zip(probs, prompts, batch, strict=True):
        ours2, ours4 = full_rows(r)
        assert all(x[0] == i for i, x in enumerate(r.full))
        assert torch.equal(ours4.argmax(-1), torch.tensor(r.tokens)), 'generated != argmax z4'
        # (1) official forward, teacher-forced, no cache
        ids = torch.tensor([pr + r.tokens[:-1]], device=dev)
        with torch.no_grad():
            _, hs, _ = runner.lm.model(input_ids=ids, use_cache=False)
        sl = slice(len(pr) - 1, len(pr) - 1 + len(r.tokens))
        off2 = center(hs[1][0, sl].float() @ runner.W.T).cpu()
        off4 = center(hs[3][0, sl].float() @ runner.W.T).cpu()
        report['official_vs_ours'].append(dict(pid=p['pid'], **cmp(p['pid'] + ' official', off2, off4, ours2, ours4)))
        # (2) batch size 1 replay of the same continuation
        single = runner.generate([pr], len(r.tokens), stops, force=[r.tokens], keep_full=True)[0]
        s2, s4 = full_rows(single)
        report['single_vs_batch'].append(dict(pid=p['pid'], **cmp(p['pid'] + ' single', s2, s4, ours2, ours4)))

    for key in ('official_vs_ours', 'single_vs_batch'):
        rows = report[key]
        n = sum(x['n'] for x in rows)
        report[key + '_summary'] = dict(
            n_positions=n,
            argmax2_agree=sum(x['argmax2_agree'] * x['n'] for x in rows) / n,
            argmax4_agree=sum(x['argmax4_agree'] * x['n'] for x in rows) / n,
            max_abs_dz4=max(x['max_abs_dz4'] for x in rows),
            max_abs_dm2=max(x['max_abs_dm2'] for x in rows))
    report['pass'] = all(report[k + '_summary'][m] >= 0.99 for k in ('official_vs_ours', 'single_vs_batch')
                         for m in ('argmax2_agree', 'argmax4_agree'))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k.endswith('summary') or k == 'pass'}, indent=1))


if __name__ == '__main__':
    main()
