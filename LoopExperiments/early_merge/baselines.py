"""Matched native-state test, frozen serial baseline, and single-GPU model latency."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from pilot import Engine, DIGIT_IDS, native, rows_for_cases, source_program, pack_logits, concatenate
from adapter import Adapter, receiver_embeddings


def joint_program(l, r, op):
    def branch(e, names):
        text = f'{names[0]} = {e["x"]}\n'
        for j, (sign, c) in enumerate(e['steps']):
            text += f'{names[j+1]} = {names[j]} {sign} {c}\n'
        return text
    return branch(l, ['a','p','q']) + branch(r, ['b','s','t']) + f'm = q {op} t\nprint(m)\nOutput: '


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--tag', default='adapter')
    ap.add_argument('--model', default='/mnt/disk2_from_server2/siqizhu4/models/ouro-2.6b')
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--bench-n', type=int, default=16)
    args = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    torch.set_num_threads(8)
    run = Path(args.run)
    data = json.loads((run / 'data.json').read_text())
    m = Engine(args.model)
    native(m, data, run, args.batch, depths=[4])
    base_rows = [r for r in rows_for_cases(data, 3) if r['swap'] == 'base']
    texts = [data['prefix'] + joint_program(data['sources'][r['left']], data['sources'][r['right']], r['op'])
             for r in base_rows]
    ids = m.ids(texts)
    parts = []
    started = time.perf_counter()
    with torch.no_grad():
        for s in range(0, len(ids), args.batch):
            lg, _ = m.forward_embeds(m.inner.embed_tokens(ids[s:s+args.batch]), loops=8)
            parts.append(pack_logits(lg))
    preds = concatenate(parts)
    np.savez_compressed(run / 'serial_predictions.npz', **preds)
    labels = np.array([DIGIT_IDS[r['y']] for r in base_rows])
    report = dict(accuracy_k1_to_k8=(preds['pred'] == labels[:,None]).mean(0).tolist(),
                  prompt_tokens=ids.shape[1], seconds=time.perf_counter()-started,
                  caveat='Frozen backbone, no interface training. Loops 5–8 exceed the pretrained four-loop schedule.')
    (run / 'serial_summary.json').write_text(json.dumps(report, indent=2))
    print('serial', report, flush=True)

    cfg = json.loads((run / args.tag / 'config.json').read_text())
    base, slots, digits = m.receiver(data['prefix'])
    adapter = Adapter(digits.float().mean(0), cfg['width']).to(m.device)
    adapter.load_state_dict(torch.load(run / args.tag / 'best.pt', weights_only=True, map_location=m.device))
    adapter.eval()
    probe_weights = [{key:v.to(m.device) for key,v in d.items()}
                     for d in torch.load(run / 'probe_weights.pt', weights_only=True)]
    selected = base_rows[::3][:args.bench_n]
    worker_embeds = [m.inner.embed_tokens(m.ids([data['prefix'] + source_program(data['sources'][q[name]])
                                               for name in ('left','right')])).detach() for q in selected]
    ops = [torch.tensor([0 if q['op']=='+' else 1], device=m.device) for q in selected]
    linear_weights = sum(p.numel() for layer in m.layers for p in layer.parameters() if p.ndim == 2)
    width = m.lm.config.hidden_size
    nl = len(m.layers)
    def core_flops(s):
        return 2 * linear_weights * s + 4 * nl * width * s * s

    @torch.no_grad()
    def invoke(i, depth, mode):
        logits, hs = m.forward_embeds(worker_embeds[i], loops=depth, states=True,
                                      final_only=True, readout=mode=='text')
        x = base[ops[i]].clone()
        if mode == 'latent':
            v = adapter(hs[:, -1]).to(x.dtype)
        elif mode == 'probe':
            w = probe_weights[depth]
            h = hs[:, -1].float()
            h = h / h.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            chosen = ((h-w['mean']) @ w['w'] + w['ymean']).argmax(-1)
            v = digits[chosen]
        else:
            pred = logits[:, -1].argmax(-1)
            # Latency-only fallback for a non-digit output. Accuracy is evaluated
            # separately and every non-digit is counted as an invalid subanswer.
            mask = (pred[:,None] == m.digit_ids[None])
            chosen = mask.int().argmax(-1)
            v = digits[chosen]
        x[:, slots[0]], x[:, slots[1]] = v[0], v[1]
        m.forward_embeds(x, loops=1, final_only=True)

    torch.cuda.empty_cache()
    timing = []
    # Alternate order across depths to reduce one-sided warmup/order bias.
    for depth in range(1,5):
        for mode in (('latent','text','probe') if depth % 2 else ('probe','text','latent')):
            for _ in range(2):
                invoke(0, depth, mode)
            times = []
            for i in range(len(selected)):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                invoke(i, depth, mode)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-begin)*1000)
            wlen = worker_embeds[0].shape[1]
            flops = 2*depth*core_flops(wlen) + core_flops(base.shape[1])
            # Three head evaluations for text, one for latent. Adapter cost is
            # negligible relative to the Transformer but included approximately.
            flops += (3 if mode=='text' else 1)*2*width*m.W.shape[0]
            if mode=='latent':
                flops += 4*sum(p.numel() for p in adapter.parameters())
            rec = dict(mode=mode, r=depth, k=1, n=len(times), median_ms=float(np.median(times)),
                       p90_ms=float(np.quantile(times,.9)), mean_ms=float(np.mean(times)),
                       dense_forward_tflops=flops/1e12, individual_ms=times)
            timing.append(rec)
            print('latency', {k:v for k,v in rec.items() if k!='individual_ms'}, flush=True)
            (run / args.tag / 'latency.json').write_text(json.dumps(dict(results=timing,
                protocol='Physical GPU 0 only; one request with two workers batched together; prepared GPU embeddings; full worker and receiver model forward passes, local state transfer and adapter included; CPU tokenization and network transport excluded; two warmups per configuration. Text uses on-device digit embedding lookup, an optimistic text handoff. FLOPs are dense mathematical estimates, not hardware counters.'), indent=2))
    print('baselines complete', flush=True)


if __name__ == '__main__':
    main()
