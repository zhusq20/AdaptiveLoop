"""Frozen Ouro branch/receiver experiment. All CUDA calls respect CUDA_VISIBLE_DEVICES.

Run from the repository root with its .venv Python. Nothing uses another GPU.
Outputs contain per-example unrestricted predictions as well as digit logits.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dag_depth'))
from loopmodel import LoopLM, DIGIT_IDS


def value(x, op, c):
    return x + c if op == '+' else x - c


def program(x, op, c):
    return f'a = {x}\nb = {c}\nm = a {op} b\nprint(m)\nOutput: '


def source_program(e):
    if 'steps' not in e:
        return program(e['x'], e['op'], e['c'])
    names = ['a', 'p', 'q', 't', 'u', 'v']
    text = f'a = {e["x"]}\n'
    for j, (op, c) in enumerate(e['steps']):
        text += f'{names[j+1]} = {names[j]} {op} {c}\n'
    return text + f'print({names[len(e["steps"])]})\nOutput: '


def make_data(n=200, seed=2718, source_ops=1, per_value=9):
    assert n % 10 == 0
    rng = random.Random(seed)
    sources, by_value = [], {}
    for y in range(10):
        es = [(x, op, c) for x in range(10) for op in '+-' for c in range(1, 10)
              if value(x, op, c) == y]
        assert len(es) == 9
        rng.shuffle(es)
        by_value[y] = {'train': [], 'test': []}
        for j, (x, op, c) in enumerate(es):
            split = 'train' if j < 6 else 'test'
            sid = len(sources)
            sources.append(dict(id=sid, x=x, op=op, c=c, y=y, split=split))
            by_value[y][split].append(sid)
    shots = [sources[by_value[y]['train'][0]].copy() for y in (2, 5, 7, 0)]
    if source_ops > 1:
        # Work backwards so every intermediate stays a digit. Preserve split sizes
        # and guarantee distinct complete programs, including across splits.
        assert per_value <= 9 ** source_ops and per_value % 3 == 0
        used = set()
        sources, by_value = [], {}
        for y in range(10):
            by_value[y] = {'train': [], 'test': []}
            for j in range(per_value):
                split = 'train' if j < per_value * 2 // 3 else 'test'
                sid = len(sources)
                sources.append(dict(id=sid, y=y, split=split))
                by_value[y][split].append(sid)
        for e in sources:
            while True:
                cur, rev = e['y'], []
                for _ in range(source_ops):
                    choices = [(x, op, c) for x in range(10) for op in '+-' for c in range(1, 10)
                               if value(x, op, c) == cur]
                    x, op, c = rng.choice(choices)
                    rev.append((op, c))
                    cur = x
                key = (cur, tuple(reversed(rev)))
                if key not in used:
                    used.add(key)
                    e.update(x=cur, steps=list(reversed(rev)))
                    break
    prefix = 'Each program uses single-digit integers. Write the value it prints.\n\n'
    prefix += ''.join(program(e['x'], e['op'], e['c']) + str(e['y']) + '\n\n' for e in shots)

    candidates = {y: [] for y in range(10)}
    for a, b, op, ap, bp in itertools.product(range(10), range(10), '+-', range(10), range(10)):
        if a == ap or b == bp:
            continue
        ys = [value(a, op, b), value(ap, op, b), value(a, op, bp), value(ap, op, bp)]
        if min(ys) >= 0 and max(ys) <= 9 and len(set(ys)) == 4:
            candidates[ys[0]].append((a, b, op, ap, bp, ys))
    cases = []
    for y in range(10):
        for _ in range(n // 10):
            a, b, op, ap, bp, ys = rng.choice(candidates[y])
            routes = []
            for v in (a, b, ap, bp):
                ids = by_value[v]['test'].copy()
                rng.shuffle(ids)
                routes.append(ids)
            cases.append(dict(id=len(cases), a=a, b=b, op=op, ap=ap, bp=bp,
                              ys=ys, routes=routes))
    rng.shuffle(cases)
    for i, c in enumerate(cases):
        c['id'] = i
    return dict(seed=seed, source_ops=source_ops, per_value=per_value, prefix=prefix,
                sources=sources, by_value=by_value, cases=cases)


class Engine(LoopLM):
    def __init__(self, model):
        super().__init__(model, 'cuda:0')
        self.lm.requires_grad_(False)
        self.digit_ids = torch.tensor(DIGIT_IDS, device=self.device)
        for d, tid in enumerate(DIGIT_IDS):
            assert self.tok.decode([tid]) == str(d)

    def ids(self, texts):
        seqs = [self.tok(s, add_special_tokens=False).input_ids for s in texts]
        assert len({len(s) for s in seqs}) == 1, 'Experiment requires equal token lengths'
        return torch.tensor(seqs, device=self.device)

    def forward_embeds(self, x, loops=4, states=False, final_only=False, readout=True):
        b, s, _ = x.shape
        pos = torch.arange(s, device=self.device)[None].expand(b, -1)
        mask = torch.ones(s, s, dtype=torch.bool, device=self.device).tril()[None, None]
        pe = self.inner.rotary_emb(x, pos)
        hs = [x[:, -1].clone()] if states else None
        logits = []
        for r in range(loops):
            for layer in self.layers:
                x = layer(x, attention_mask=mask, position_ids=pos, past_key_value=None,
                          use_cache=False, position_embeddings=pe)
            x = self.inner.norm(x)
            if states:
                hs.append(x[:, -1].clone())
            if readout and (not final_only or r == loops - 1):
                logits.append(x[:, -1].float() @ self.W.T)
        return (torch.stack(logits, 1) if logits else None), torch.stack(hs, 1) if states else None

    def receiver(self, prefix):
        ids = self.ids([prefix + program(0, op, 0) for op in '+-'])
        probe = self.ids([prefix + program(3, op, 7) for op in '+-'])
        changed = (ids != probe).nonzero()
        slots = sorted(set(changed[:, 1].tolist()))
        assert len(slots) == 2 and len(changed) == 4
        assert all(ids[i, p].item() == DIGIT_IDS[0] for i in range(2) for p in slots)
        base = self.inner.embed_tokens(ids).detach()
        gold = base.clone()
        digits = self.inner.embed_tokens(self.digit_ids).detach()
        gold[:, slots[0]] = digits[3]
        gold[:, slots[1]] = digits[7]
        assert torch.equal(gold, self.inner.embed_tokens(probe)), 'Slot identification failed'
        return base, slots, digits


def pack_logits(lg):
    return dict(pred=lg.argmax(-1).cpu().numpy(),
                digit_pred=lg[..., DIGIT_IDS].argmax(-1).cpu().numpy(),
                digit_logits=lg[..., DIGIT_IDS].cpu().numpy())


def concatenate(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def batch_predict(m, embeds, batch, loops=4):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(embeds), batch):
            lg, _ = m.forward_embeds(embeds[start:start + batch], loops)
            chunks.append(pack_logits(lg))
    return concatenate(chunks)


def rows_for_cases(data, routes=1):
    rows = []
    for case in data['cases']:
        for route in range(routes):
            a, b, ap, bp = [x[route] for x in case['routes']]
            for swap, left, right, y in zip(('base', 'A', 'B', 'AB'),
                                          (a, ap, a, ap), (b, b, bp, bp), case['ys']):
                rows.append(dict(case=case['id'], route=route, swap=swap, left=left,
                                 right=right, op=case['op'], y=y))
    return rows


def collect(m, data, out, batch):
    texts = [data['prefix'] + source_program(e) for e in data['sources']]
    ids = m.ids(texts)
    # Match the official no-cache forward on the exact same tensor geometry.
    with torch.no_grad():
        ours, _ = m.forward_embeds(m.inner.embed_tokens(ids[:2]))
        _, official_hs, _ = m.inner(input_ids=ids[:2], use_cache=False)
        official = torch.stack([h[:, -1].float() @ m.W.T for h in official_hs], 1)
        diff = float((ours - official).abs().max())
        agreement = float((ours.argmax(-1) == official.argmax(-1)).float().mean())
    assert diff < 0.05 and agreement == 1., (diff, agreement)
    check = dict(official_max_logit_difference=diff, official_argmax_agreement=agreement)
    (out / 'validation.json').write_text(json.dumps(check, indent=2))
    print('forward validation', check, flush=True)
    states, parts = [], []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            lg, hs = m.forward_embeds(m.inner.embed_tokens(ids[start:start + batch]), states=True)
            states.append(hs.cpu())
            parts.append(pack_logits(lg))
    source_states = torch.cat(states)
    torch.save(source_states, out / 'source_states.pt')
    predictions = concatenate(parts)
    np.savez_compressed(out / 'source_predictions.npz', **predictions)
    y = np.array([e['y'] for e in data['sources']])
    acc = (predictions['pred'] == np.array(DIGIT_IDS)[y, None]).mean(0)
    print('worker accuracy loops 1..4:', acc.tolist(), flush=True)

    base, slots, digits = m.receiver(data['prefix'])
    oracle_rows = [dict(a=a, b=b, op=op, y=value(a, op, b))
                   for a, b, op in itertools.product(range(10), range(10), '+-')
                   if 0 <= value(a, op, b) <= 9]
    emb = base[torch.tensor([0 if r['op'] == '+' else 1 for r in oracle_rows], device=m.device)].clone()
    emb[:, slots[0]] = digits[[r['a'] for r in oracle_rows]]
    emb[:, slots[1]] = digits[[r['b'] for r in oracle_rows]]
    pred = batch_predict(m, emb, batch)
    np.savez_compressed(out / 'oracle_predictions.npz', **pred)
    (out / 'oracle_rows.json').write_text(json.dumps(oracle_rows))
    oy = np.array([r['y'] for r in oracle_rows])
    oa = (pred['pred'] == np.array(DIGIT_IDS)[oy, None]).mean(0)
    print('text-oracle accuracy loops 1..4:', oa.tolist(), flush=True)
    timings = dict(collect_and_oracle_seconds=time.perf_counter() - started,
                   source_prompt_tokens=ids.shape[1], receiver_prompt_tokens=base.shape[1],
                   slots=slots, source_accuracy=acc.tolist(), oracle_accuracy=oa.tolist())
    (out / 'collection_summary.json').write_text(json.dumps(timings, indent=2))


def native(m, data, out, batch, depths):
    hs = torch.load(out / 'source_states.pt', weights_only=True).to(m.device)
    base, slots, digits = m.receiver(data['prefix'])
    rows = rows_for_cases(data, routes=1)
    (out / 'native_rows.json').write_text(json.dumps(rows))
    idxl = torch.tensor([r['left'] for r in rows], device=m.device)
    idxr = torch.tensor([r['right'] for r in rows], device=m.device)
    op = torch.tensor([0 if r['op'] == '+' else 1 for r in rows], device=m.device)
    y = np.array([DIGIT_IDS[r['y']] for r in rows])
    target_rms = digits.float().square().mean(-1).sqrt().mean()
    summary = []
    for depth in depths:
        for style in ('raw', 'rms'):
            started = time.perf_counter()
            l, r = hs[idxl, depth], hs[idxr, depth]
            if style == 'rms':
                def normalize(x):
                    return (x.float() * target_rms / x.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)).to(x.dtype)
                l, r = normalize(l), normalize(r)
            emb = base[op].clone()
            emb[:, slots[0]], emb[:, slots[1]] = l, r
            pred = batch_predict(m, emb, batch)
            np.savez_compressed(out / f'native_{style}_r{depth}.npz', **pred)
            acc = (pred['pred'] == y[:, None]).mean(0)
            item = dict(style=style, r=depth, accuracy=acc.tolist(),
                        seconds=time.perf_counter() - started)
            summary.append(item)
            print('native', item, flush=True)
            (out / 'native_summary.json').write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/mnt/disk2_from_server2/siqizhu4/models/ouro-2.6b')
    ap.add_argument('--out', default=str(ROOT / 'early_merge/runs/pilot_2718'))
    ap.add_argument('--stage', choices=['collect', 'native', 'both'], default='both')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--source-ops', type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument('--per-value', type=int, default=9)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--depths', default='4,0,1,2,3')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES=0 to enforce the requested physical GPU.')
    torch.set_num_threads(8)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'data.json').exists():
        data = json.loads((out / 'data.json').read_text())
    else:
        data = make_data(args.n, source_ops=args.source_ops, per_value=args.per_value)
        (out / 'data.json').write_text(json.dumps(data, indent=1))
    meta = vars(args) | dict(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                            torch=torch.__version__, device=torch.cuda.get_device_name(0),
                            data_sha256=hashlib.sha256((out / 'data.json').read_bytes()).hexdigest())
    (out / f'config_{args.stage}.json').write_text(json.dumps(meta, indent=2))
    print('loading', args.model, flush=True)
    m = Engine(args.model)
    if args.stage in ('collect', 'both'):
        collect(m, data, out, args.batch)
    if args.stage in ('native', 'both'):
        native(m, data, out, args.batch, [int(x) for x in args.depths.split(',')])
    print('complete', flush=True)


if __name__ == '__main__':
    main()
