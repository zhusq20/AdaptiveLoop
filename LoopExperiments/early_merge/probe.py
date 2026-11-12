"""Small supervised linear readout control; hyperparameters use validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adapter import split_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    run = Path(args.run)
    data = json.loads((run / 'data.json').read_text())
    splits = split_ids(data)
    states = torch.load(run / 'source_states.pt', weights_only=True).double()
    labels = torch.tensor([x['y'] for x in data['sources']])
    targets = torch.nn.functional.one_hot(labels, 10).double()
    train, val, test = [torch.tensor(splits[s]) for s in ('train', 'val', 'test')]
    predictions, reports, weights = [], [], []
    for r in range(5):
        h = states[:, r]
        h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        mean = h[train].mean(0)
        x = h - mean
        ymean = targets[train].mean(0)
        if r == 0:
            w = torch.zeros(h.shape[1], 10, dtype=h.dtype)
            best_lam = None
        else:
            gram = x[train] @ x[train].T
            scale = float(gram.diag().mean())
            best = -1
            # Larger regularization wins ties, chosen without test labels.
            for fac in (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.):
                lam = fac * scale
                alpha = torch.linalg.solve(gram + lam * torch.eye(len(train)), targets[train] - ymean)
                candidate = x[train].T @ alpha
                score = float(((x[val] @ candidate + ymean).argmax(-1) == labels[val]).double().mean())
                if score >= best:
                    best, w, best_lam = score, candidate, lam
        logits = x @ w + ymean
        pred = logits.argmax(-1)
        predictions.append(pred.numpy())
        weights.append(dict(mean=mean.float(), w=w.float(), ymean=ymean.float()))
        rec = dict(r=r, ridge=best_lam,
                   train_accuracy=float((pred[train] == labels[train]).double().mean()),
                   val_accuracy=float((pred[val] == labels[val]).double().mean()),
                   test_accuracy=float((pred[test] == labels[test]).double().mean()))
        reports.append(rec)
        print(rec, flush=True)
    np.savez_compressed(run / 'probe_predictions.npz', pred=np.stack(predictions, 1))
    torch.save(weights, run / 'probe_weights.pt')
    (run / 'probe_summary.json').write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
