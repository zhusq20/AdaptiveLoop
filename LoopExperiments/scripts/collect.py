#!/usr/bin/env python
"""Phase-1 collection: greedy-decode with the original 4-loop Ouro and record, at every
generated position, the compact loop-2 / loop-4 readout statistics (see bounded.py).

Output (resumable; one file pair per batch, jsonl written last as the completion mark):
  <out>/<dataset>/shard<i>of<n>/bNNNN.npz    per-position arrays
  <out>/<dataset>/shard<i>of<n>/bNNNN.jsonl  one line per problem (text, finish, grading)
  <out>/<dataset>/shard<i>of<n>/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from loopexp.bounded import CHECK_GRID  # noqa: E402
from loopexp.data import build_prompt, grade, load_problems, stop_ids  # noqa: E402
from loopexp.ouro import OuroRunner, stack_stats  # noqa: E402

MODEL_DIR = '/mnt/disk2_from_server2/siqizhu4/models/ouro-1.4b'
MODEL_REV = '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
INT_KEYS = ('a2', 'a4', 'ids2', 'ids4', 'grid_arg', 'rank_a4_in_z2')


def save_batch(path: Path, probs, prompts, results, tokenizer):
    arrays, lines = {}, []
    full_rows, full_z2, full_z4 = [], [], []
    offset = 0
    for j, (p, pr, r) in enumerate(zip(probs, prompts, results, strict=True)):
        st = stack_stats(r.stats)
        n = len(r.tokens)
        st['prob'] = torch.full((n,), j, dtype=torch.int32)
        st['pos'] = torch.arange(n, dtype=torch.int32)
        st['token'] = torch.tensor(r.tokens, dtype=torch.int32)
        for key, val in st.items():
            arrays.setdefault(key, []).append(val)
        for pos, z2, z4 in r.full:
            full_rows.append(offset + pos)
            full_z2.append(z2)
            full_z4.append(z4)
        offset += n
        text = tokenizer.decode(r.tokens, skip_special_tokens=True)
        g = grade(p, text)
        lines.append(dict(pid=p['pid'], dataset=p['dataset'], split=p['split'],
                          prompt_tokens=len(pr), gen_tokens=n, finish=r.finish,
                          token_eq_a4=bool((st['token'] == st['a4']).all()),
                          n_uncert=len(r.full), text=text, **g))
    out = {}
    for key, vals in arrays.items():
        v = torch.cat(vals).numpy()
        if key in INT_KEYS:
            v = v.astype(np.int32)
        elif v.dtype == np.float64:
            v = v.astype(np.float32)
        out[key] = v
    out['full_rows'] = np.asarray(full_rows, dtype=np.int64)
    out['full_z2'] = torch.stack(full_z2).numpy() if full_z2 else np.zeros((0, 0), np.float32)
    out['full_z4'] = torch.stack(full_z4).numpy() if full_z4 else np.zeros((0, 0), np.float32)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **out)
    os.replace(tmp, path.with_suffix('.npz'))
    tmp = path.with_suffix('.tmp.jsonl')
    with open(tmp, 'w') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    os.replace(tmp, path.with_suffix('.jsonl'))
    return lines


def generate_robust(runner, prompts, *args, **kw):
    """Shared GPUs: on CUDA OOM split the batch in halves and retry (down to size 1).
    Each sequence's trajectory stays self-consistent; only bf16 batch-shape noise changes."""
    try:
        return runner.generate(prompts, *args, **kw)
    except torch.OutOfMemoryError:
        runner.K = runner.V = None
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            time.sleep(60)
            return generate_robust(runner, prompts, *args, **kw)
        h = len(prompts) // 2
        print(f'OOM at batch {len(prompts)}; retrying as {h}+{len(prompts) - h}', flush=True)
        return (generate_robust(runner, prompts[:h], *args, **kw)
                + generate_robust(runner, prompts[h:], *args, **kw))


def git_rev():
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['gsm8k', 'math500', 'mbppplus'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--model-dir', default=MODEL_DIR)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-new-tokens', type=int, default=2048)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--split', default='all', choices=['all', 'calib', 'test'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--prompt-format', default='chat')
    a = ap.parse_args()

    probs = load_problems(a.dataset)
    if a.split != 'all':
        probs = [p for p in probs if p['split'] == a.split]
    probs = probs[a.shard::a.num_shards]
    if a.limit:
        probs = probs[:a.limit]
    out = Path(a.out) / a.dataset / f'shard{a.shard}of{a.num_shards}'
    out.mkdir(parents=True, exist_ok=True)

    runner = OuroRunner(a.model_dir, a.device)
    tok = runner.tok
    stops = stop_ids(tok)
    prompts = [build_prompt(p, tok, a.prompt_format) for p in probs]
    order = sorted(range(len(probs)), key=lambda i: (-len(prompts[i]), probs[i]['pid']))
    batches = [order[i:i + a.batch_size] for i in range(0, len(order), a.batch_size)]
    manifest = dict(
        args=vars(a), model_dir=a.model_dir, model_rev=MODEL_REV, git_rev=git_rev(),
        torch=torch.__version__, cuda=torch.version.cuda,
        transformers=__import__('transformers').__version__, python=platform.python_version(),
        gpu=torch.cuda.get_device_name(runner.device), host=platform.node(),
        stop_ids=sorted(stops), check_grid=list(CHECK_GRID), n_problems=len(probs),
        n_batches=len(batches), started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    t0 = time.time()
    done_tokens = 0
    for bi, idx in enumerate(batches):
        path = out / f'b{bi:04d}'
        if path.with_suffix('.jsonl').exists():
            continue
        tb = time.time()
        res = generate_robust(runner, [prompts[i] for i in idx], a.max_new_tokens, stops, k=a.k)
        lines = save_batch(path, [probs[i] for i in idx], [prompts[i] for i in idx], res, tok)
        ntok = sum(x['gen_tokens'] for x in lines)
        done_tokens += ntok
        bad = sum(not x['token_eq_a4'] for x in lines)
        print(f'[{a.dataset} {a.shard}/{a.num_shards}] batch {bi + 1}/{len(batches)} '
              f'{len(idx)} probs {ntok} tok {time.time() - tb:.0f}s '
              f'uncert={sum(x["n_uncert"] for x in lines)} token!=a4={bad} '
              f'elapsed={time.time() - t0:.0f}s', flush=True)
    (out / 'DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))


if __name__ == '__main__':
    main()
