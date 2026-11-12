"""Bounded late-loop correction: math shared by collection, analysis and tests.

All ranking is FP32. Tie-breaking everywhere is "smallest token id among the maxima",
which is what torch.argmax returns over a full-vocabulary row (first occurrence).

Notation (plan.md §2): z2, z4 are mean-centred full-vocab logits after loop 2 and 4,
    z4~(B) = z2 + clip(z4 - z2, -B, B),
    m2     = z2[a] - z2[b]  (top-1 minus top-2 of z2),
and m2 > 2B certifies argmax z4~(B) == argmax z2.
"""
from __future__ import annotations

import torch

# Default B grid stored per position at collection time for the compact-vs-full self-check.
CHECK_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)


def center(z: torch.Tensor) -> torch.Tensor:
    z = z.float()
    return z - z.mean(dim=-1, keepdim=True)


def bounded(z2: torch.Tensor, z4: torch.Tensor, B: float | torch.Tensor) -> torch.Tensor:
    """z4~(B). `B` may be a scalar or a tensor broadcastable against z2 (e.g. [N, 1])."""
    if isinstance(B, torch.Tensor):
        return z2 + torch.maximum(torch.minimum(z4 - z2, B), -B)
    return z2 + (z4 - z2).clamp(-B, B)


def top2_margin(z: torch.Tensor) -> torch.Tensor:
    v = torch.topk(z, 2, dim=-1).values
    return v[..., 0] - v[..., 1]


def argmax_min_id(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Row-wise argmax over candidate columns; ties -> smallest token id. Returns token ids.

    `ids` may contain duplicates (same token listed twice) and padding ids < 0 whose
    values must be -inf.
    """
    best = values.max(dim=-1, keepdim=True).values
    big = torch.iinfo(ids.dtype).max
    cand = torch.where(values == best, ids, torch.full_like(ids, big))
    return cand.min(dim=-1).values


@torch.no_grad()
def compact_stats(z2: torch.Tensor, z4: torch.Tensor, k: int, grid=CHECK_GRID, eps: float = 1e-3):
    """Per-row sufficient statistics to recompute argmax z4~(B) exactly for every B >= 0.

    z2, z4: [N, V] FP32 mean-centred logits.
    Candidate set = top-k(z2) U top-k(z4). Every z4~_v lies between z2_v and z4_v, so the
    winner must satisfy max(z2_v, z4_v) >= L := max_v min(z2_v, z4_v). If the (k+1)-th
    largest value of both z2 and z4 is < L - eps, no token outside the candidate set can
    win or tie, and the candidate set is sufficient (`cert`). Rows without the certificate
    must be stored in full by the caller.
    """
    top2 = torch.topk(z2, k + 1, dim=-1)
    top4 = torch.topk(z4, k + 1, dim=-1)
    ids2, ids4 = top2.indices[:, :k], top4.indices[:, :k]
    L = torch.minimum(z2, z4).max(dim=-1).values
    cert = (top2.values[:, k] < L - eps) & (top4.values[:, k] < L - eps)
    a2 = z2.argmax(dim=-1)
    a4 = z4.argmax(dim=-1)
    grid_t = torch.tensor(grid, dtype=z2.dtype, device=z2.device)
    grid_arg = torch.stack([bounded(z2, z4, float(b)).argmax(dim=-1) for b in grid_t.tolist()], dim=-1)
    return dict(
        a2=a2, a4=a4, m2=top2.values[:, 0] - top2.values[:, 1],
        m4=top4.values[:, 0] - top4.values[:, 1],
        ids2=ids2, z2_ids2=top2.values[:, :k], z4_ids2=z4.gather(1, ids2),
        ids4=ids4, z2_ids4=z2.gather(1, ids4), z4_ids4=top4.values[:, :k],
        cert=cert, grid_arg=grid_arg,
        # rank of the loop-4 winner under z2 (0 = top) and z2 logit gap it must close
        rank_a4_in_z2=(z2 > z2.gather(1, a4[:, None])).sum(-1),
        gap_a4=top2.values[:, 0] - z2.gather(1, a4[:, None])[:, 0],
    )


def compact_bounded_argmax(ids: torch.Tensor, z2c: torch.Tensor, z4c: torch.Tensor,
                           B: float | torch.Tensor) -> torch.Tensor:
    """argmax z4~(B) from compact candidates ([N, C] each). Ties -> smallest id."""
    return argmax_min_id(bounded(z2c, z4c, B), ids)
