"""Controlled two-worker loop-depth orchestration on frozen Ouro-2.6B.

The upstream task assignment and the final arithmetic operator are supplied.
This isolates scheduling/early-exit from learning to delegate. All calibration
uses source programs outside the held-out test split.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "early_merge"))
from pilot import DIGIT_IDS, Engine, make_data, rows_for_cases, source_program  # noqa: E402


def read_data(run: Path):
    data = json.loads((run / "data.json").read_text())
    z = np.load(run / "source_predictions.npz")
    pred = z["pred"]
    logits = z["digit_logits"].astype(np.float64)
    y = np.array([e["y"] for e in data["sources"]])
    train = np.array([e["split"] == "train" for e in data["sources"]])
    test = ~train
    return data, pred, logits, y, train, test


def margins(logits: np.ndarray) -> np.ndarray:
    top2 = np.partition(logits, -2, axis=-1)[..., -2:]
    return top2.max(-1) - top2.min(-1)


def calibrate(pred, logits, y, train):
    margin = margins(logits)
    thresholds = {}
    for depth in (2, 3):
        wrong = train & (pred[:, depth - 1] != np.array(DIGIT_IDS)[y])
        # A conservative, pre-declared threshold: reject every calibration
        # error at this depth. No held-out outcome is used to set it.
        thresholds[depth] = float(np.nextafter(margin[wrong, depth - 1].max(initial=0), np.inf))
    return thresholds, margin


def selected_depths(pred, margin, thresholds):
    n = len(pred)
    chosen = np.full(n, 4, dtype=np.int8)
    target_ids = np.array(DIGIT_IDS)
    for depth in (3, 2):
        legal = np.isin(pred[:, depth - 1], target_ids)
        accept = legal & (margin[:, depth - 1] >= thresholds[depth])
        chosen[accept] = depth
    return chosen


def score_rows(data, pred, depth):
    rows = rows_for_cases(data, routes=3)
    left = np.array([r["left"] for r in rows])
    right = np.array([r["right"] for r in rows])
    y = np.array([r["y"] for r in rows])
    plus = np.array([r["op"] == "+" for r in rows])
    lookup = np.full(max(DIGIT_IDS) + 1, -100, dtype=int)
    lookup[DIGIT_IDS] = np.arange(10)
    token = pred[np.arange(len(pred)), depth - 1]
    digit = np.full(len(token), -100, dtype=int)
    valid = (token >= 0) & (token < len(lookup))
    digit[valid] = lookup[token[valid]]
    a, b = digit[left], digit[right]
    answer = np.where(plus, a + b, a - b)
    hit = (a >= 0) & (b >= 0) & (answer == y)
    base = hit.reshape(len(data["cases"]), 3, 4)[:, :, 0]
    strict = hit.reshape(len(data["cases"]), 3, 4).all(-1)
    ld, rd = depth[left], depth[right]
    return dict(base_accuracy=float(base.mean()), all_accuracy=float(hit.mean()),
                quartet_all_four_correct=float(strict.mean()),
                mean_worker_loops=float((ld + rd).mean()),
                mean_request_critical_depth=float(np.maximum(ld, rd).mean()),
                both_exit_by_2=float(((ld == 2) & (rd == 2)).mean()),
                all_rows=len(rows), base_rows=int(base.size)), rows, hit


def analyze(run: Path, out: Path):
    data, pred, logits, y, train, test = read_data(run)
    thresholds, margin = calibrate(pred, logits, y, train)
    adaptive = selected_depths(pred, margin, thresholds)
    results = {}
    ids = np.array(DIGIT_IDS)[y]
    for name, depth in (("fixed2", np.full(len(y), 2, dtype=np.int8)),
                        ("fixed3", np.full(len(y), 3, dtype=np.int8)),
                        ("fixed4", np.full(len(y), 4, dtype=np.int8)),
                        ("adaptive", adaptive)):
        worker_hit = pred[np.arange(len(y)), depth - 1] == ids
        score, _, _ = score_rows(data, pred, depth)
        score.update(test_worker_accuracy=float(worker_hit[test].mean()),
                     train_worker_accuracy=float(worker_hit[train].mean()),
                     test_depth_counts={str(d): int(((depth == d) & test).sum()) for d in (2, 3, 4)},
                     train_depth_counts={str(d): int(((depth == d) & train).sum()) for d in (2, 3, 4)})
        results[name] = score
    assert not (adaptive < 2).any()
    assert not (adaptive > 4).any()
    report = dict(task="Two independent arithmetic workers, supplied assignment, one-loop text receiver",
                  calibration_programs=int(train.sum()), heldout_programs=int(test.sum()),
                  thresholds=thresholds,
                  calibration_rule="Margin threshold just above the largest calibration error at each depth; depth 4 fallback",
                  results=results,
                  caveat="Final-answer scoring uses exact arithmetic of the predicted worker digits. The frozen receiver was separately checked 100% on all legal operand/operator combinations; GPU benchmarking checks actual receiver outputs. Held-out case rows reuse the same 270 source programs, so row count is not independent sample size.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report, adaptive, margin


def benchmark(run: Path, out: Path, model: str, n: int, seed: int):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Run only with CUDA_VISIBLE_DEVICES=0")
    torch.set_num_threads(8)
    report, adaptive, _, = analyze(run, out)
    data = json.loads((run / "data.json").read_text())
    rows = [r for r in rows_for_cases(data, routes=3) if r["swap"] == "base"]
    rng = random.Random(seed)
    selection = rng.sample(rows, min(n, len(rows)))
    m = Engine(model)
    base, slots, digits = m.receiver(data["prefix"])
    threshold = {int(k): float(v) for k, v in report["thresholds"].items()}
    embs = []
    for row in selection:
        texts = [data["prefix"] + source_program(data["sources"][row[k]]) for k in ("left", "right")]
        embs.append(m.inner.embed_tokens(m.ids(texts)).detach())
    w = m.W
    token_ids = m.digit_ids
    mask = torch.ones(embs[0].shape[1], embs[0].shape[1], dtype=torch.bool, device=m.device).tril()[None, None]

    def one_loop(x, pos, pe):
        for layer in m.layers:
            x = layer(x, attention_mask=mask, position_ids=pos, past_key_value=None,
                      use_cache=False, position_embeddings=pe)
        return m.inner.norm(x)

    def read(x):
        logits = x[:, -1].float() @ w.T
        pred = logits.argmax(-1)
        dlog = logits[:, token_ids]
        top2 = dlog.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        valid = (pred[:, None] == token_ids[None, :]).any(-1)
        digit = (pred[:, None] == token_ids[None, :]).int().argmax(-1)
        return digit, margin, valid

    @torch.no_grad()
    def invoke(row, emb, mode):
        x = emb
        pos = torch.arange(x.shape[1], device=m.device)[None].expand(2, -1)
        pe = m.inner.rotary_emb(x, pos)
        depth = torch.full((2,), 4 if mode == "fixed4" else 3 if mode == "fixed3" else 2,
                           device=m.device, dtype=torch.int8)
        if mode.startswith("fixed"):
            for _ in range(int(depth[0])):
                x = one_loop(x, pos, pe)
            digit, _, _ = read(x)
        else:
            for _ in range(2):
                x = one_loop(x, pos, pe)
            digit, conf, valid = read(x)
            accepted = (valid & (conf >= threshold[2])).tolist()
            active = [i for i, ok in enumerate(accepted) if not ok]
            if active:
                active = list(range(2)) if mode == "joint" else active
                ix = torch.tensor(active, device=m.device)
                xx = x[ix]
                pp = pos[ix]
                ee = tuple(p[ix] for p in pe)
                xx = one_loop(xx, pp, ee)
                new_digit, conf3, valid3 = read(xx)
                digit[ix] = new_digit
                depth[ix] = 3
                rejected3 = [j for j in range(len(active)) if not (valid3[j] and conf3[j] >= threshold[3])]
                if rejected3:
                    jx = torch.tensor(rejected3, device=m.device)
                    xxx = one_loop(xx[jx], pp[jx], tuple(p[jx] for p in ee))
                    newest, _, _ = read(xxx)
                    digit[ix[jx]] = newest
                    depth[ix[jx]] = 4
        receiver = base[0 if row["op"] == "+" else 1].unsqueeze(0).clone()
        receiver[:, slots[0]] = digits[digit[0]]
        receiver[:, slots[1]] = digits[digit[1]]
        lg, _ = m.forward_embeds(receiver, loops=1, final_only=True)
        answer = int(lg[0, -1].argmax(-1))
        return answer, depth.cpu().tolist()

    modes = ("fixed2", "fixed3", "fixed4", "joint", "per_branch")
    for mode in modes:
        for j in range(3):
            invoke(selection[j], embs[j], mode)
    torch.cuda.synchronize()
    records = {mode: [] for mode in modes}
    for i, (row, emb) in enumerate(zip(selection, embs)):
        order = list(modes)
        rng.shuffle(order)
        for mode in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            answer, depths = invoke(row, emb, mode)
            torch.cuda.synchronize()
            records[mode].append(dict(row=i, ms=(time.perf_counter() - start) * 1000,
                                      correct=answer == DIGIT_IDS[row["y"]], depths=depths))
        if (i + 1) % 24 == 0:
            print(f"benchmarked {i+1}/{len(selection)}", flush=True)
    summary = {}
    for mode, rr in records.items():
        times = np.array([r["ms"] for r in rr])
        summary[mode] = dict(n=len(rr), accuracy=float(np.mean([r["correct"] for r in rr])),
                             mean_ms=float(times.mean()), median_ms=float(np.median(times)),
                             p90_ms=float(np.quantile(times, .9)),
                             mean_worker_loops=float(np.mean([sum(r["depths"]) for r in rr])),
                             mean_critical_depth=float(np.mean([max(r["depths"]) for r in rr])))
    paired = {}
    boot_rng = np.random.default_rng(13)
    for mode in ("fixed3", "fixed4", "joint"):
        delta = np.array([r["ms"] for r in records[mode]]) - np.array([r["ms"] for r in records["per_branch"]])
        indices = boot_rng.integers(0, len(delta), size=(10000, len(delta)))
        draws = delta[indices].mean(-1)
        paired[mode] = dict(mean_ms=float(delta.mean()),
                            bootstrap_95=[float(np.quantile(draws, .025)), float(np.quantile(draws, .975))])
    artifact = dict(protocol="Physical CUDA 0; two worker prompts batch=2; one-loop receiver; prepared GPU embeddings; CPU branch decision and CUDA synchronization included; tokenization, network, subagent tools excluded. Randomized policy order per item.",
                    seed=seed, selection=selection, summary=summary,
                    paired_savings_vs_per_branch_ms=paired, records=records,
                    gpu_name=torch.cuda.get_device_name(0),
                    peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
    (out / "latency.json").write_text(json.dumps(artifact, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def robustness(run: Path, out: Path, model: str, seeds: list[int], batch: int):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Run only with CUDA_VISIBLE_DEVICES=0")
    torch.set_num_threads(8)
    data, pred0, logits0, y, train, test = read_data(run)
    thresholds, _ = calibrate(pred0, logits0, y, train)
    m = Engine(model)
    report = []
    for seed in seeds:
        alternate = make_data(n=20, seed=seed, source_ops=2, per_value=81)
        assert alternate["prefix"] != data["prefix"]
        texts = [alternate["prefix"] + source_program(e) for e in data["sources"]]
        ids = m.ids(texts)
        predictions, digit_logits = [], []
        with torch.no_grad():
            for start in range(0, len(ids), batch):
                lg, _ = m.forward_embeds(m.inner.embed_tokens(ids[start:start + batch]), loops=4)
                predictions.append(lg.argmax(-1).cpu().numpy())
                digit_logits.append(lg[:, :, m.digit_ids].cpu().numpy())
        pred = np.concatenate(predictions)
        logits = np.concatenate(digit_logits)
        np.savez_compressed(out / f"prefix_seed_{seed}_predictions.npz", pred=pred, digit_logits=logits)
        margin = margins(logits)
        depth = selected_depths(pred, margin, thresholds)
        hit = pred[np.arange(len(y)), depth - 1] == np.array(DIGIT_IDS)[y]
        row_score, _, _ = score_rows(data, pred, depth)
        fixed = {}
        for d in (2, 3, 4):
            dd = np.full(len(y), d, dtype=np.int8)
            score, _, _ = score_rows(data, pred, dd)
            fixed[str(d)] = dict(test_worker_accuracy=float((pred[test, d-1] == np.array(DIGIT_IDS)[y[test]]).mean()),
                                 base_accuracy=score["base_accuracy"])
        errors = [dict(id=int(i), program=data["sources"][i], selected_depth=int(depth[i]),
                       true_digit=int(y[i]), predicted_token=int(pred[i, depth[i]-1]),
                       r2_to_r4_predicted_tokens=pred[i, 1:4].tolist(),
                       r2_to_r4_margins=margin[i, 1:4].tolist())
                  for i in np.where(test & ~hit)[0]]
        item = dict(seed=seed, new_prefix=alternate["prefix"],
                    train_worker_accuracy=float(hit[train].mean()),
                    test_worker_accuracy=float(hit[test].mean()),
                    test_source_errors=int((~hit[test]).sum()),
                    test_depth_counts={str(d): int(((depth == d) & test).sum()) for d in (2, 3, 4)},
                    score=row_score, fixed=fixed, errors=errors)
        report.append(item)
        print(json.dumps({k:v for k,v in item.items() if k!='new_prefix'}, indent=2), flush=True)
    (out / "robustness.json").write_text(json.dumps(dict(
        protocol="Same 810 source programs and labels; only four few-shot examples in the prefix change. Thresholds are frozen from the original prefix's 540 calibration programs; no recalibration.",
        thresholds=thresholds, results=report), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "early_merge/runs/ops2_full_2718"))
    ap.add_argument("--out", default=str(ROOT / "agentic_loop_depth/runs/ops2_2718"))
    ap.add_argument("--model", default="/mnt/disk2_from_server2/siqizhu4/models/ouro-2.6b")
    ap.add_argument("--mode", choices=("analyze", "benchmark", "robustness"), default="analyze")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--robustness-seeds", nargs="+", type=int, default=[1234, 5678])
    ap.add_argument("--batch", type=int, default=12)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    if args.mode == "analyze":
        analyze(run, out)
    elif args.mode == "benchmark":
        benchmark(run, out, args.model, args.n, args.seed)
    else:
        robustness(run, out, args.model, args.robustness_seeds, args.batch)


if __name__ == "__main__":
    main()
