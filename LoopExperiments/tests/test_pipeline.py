"""End-to-end check of save format + analysis on synthetic logits (no GPU, no model)."""
import importlib.util
import json
import sys
from pathlib import Path

import torch

from loopexp.bounded import center, compact_stats
from loopexp.ouro import SeqResult

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Tok:
    def decode(self, ids, skip_special_tokens=True):
        return '\\boxed{3}'


def _fake_run(tmp: Path, ds='gsm8k', n_prob=40, seed=0, k=8, V=300):
    collect = _load('collect')
    g = torch.Generator().manual_seed(seed)
    out = tmp / ds / 'shard0of1'
    out.mkdir(parents=True)
    probs = [dict(pid=f'p{i}', dataset=ds, split='calib' if i % 3 == 0 else 'test', gold='3')
             for i in range(n_prob)]
    for b in range(0, n_prob, 8):
        res = []
        for _ in probs[b:b + 8]:
            n = int(torch.randint(5, 30, (1,), generator=g))
            z2 = center(torch.randn(n, V, generator=g) * 3)
            z4 = center(z2 + torch.randn(n, V, generator=g) * 2)
            st = compact_stats(z2, z4, k)
            r = SeqResult(tokens=st['a4'].tolist(), finish='eos')
            r.stats = [{key: v[i] for key, v in st.items()} for i in range(n)]
            r.full = [(i, z2[i], z4[i]) for i in range(n) if not st['cert'][i]]
            res.append(r)
        collect.save_batch(out / f'b{b // 8:04d}', probs[b:b + 8], [[1, 2]] * len(res), res, _Tok())
    return tmp


def test_analysis_end_to_end(tmp_path, monkeypatch):
    run = _fake_run(tmp_path)
    analyze = _load('analyze')
    monkeypatch.setattr(analyze, 'N_BOOT', 50)
    monkeypatch.setattr(sys, 'argv', ['analyze', '--run', str(run), '--datasets', 'gsm8k'])
    analyze.main()
    res = json.loads((run / 'analysis' / 'results.json').read_text())
    c = res['checks']['gsm8k']
    assert c['pass'] and sum(c['agree_rows_changed']) == 0
    for row in res['tables']['per_dataset']['gsm8k']:
        t = row['test']
        assert row['guarantee_violations'] == 0
        # identities: bounded never beats early exit; divergence = fixed2 * (1 - retention)
        assert t['div_bc'] >= t['div_ee'] - 1e-12
        assert abs(t['div_bc'] - t['div_fixed2'] * (1 - t['retention_bc'])) < 1e-9
        assert abs(t['div_bc'] - t['div_ee'] - t['trunc_cost']) < 1e-9
    assert (run / 'analysis' / 'main.png').exists()
