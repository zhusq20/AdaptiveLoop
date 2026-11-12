import torch

from loopexp.bounded import (argmax_min_id, bounded, center, compact_bounded_argmax,
                             compact_stats, top2_margin)


def _logits(n=512, v=2000, seed=0, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    z2 = center(torch.randn(n, v, generator=g) * scale)
    # loop 4 = loop 2 + a perturbation of mixed size, so some rows change their argmax
    z4 = center(z2 + torch.randn(n, v, generator=g) * torch.rand(n, 1, generator=g) * 2 * scale)
    return z2, z4


def test_certified_exit_keeps_loop2_argmax():
    z2, z4 = _logits()
    m2 = top2_margin(z2)
    for B in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0):
        ex = m2 > 2 * B
        out = bounded(z2, z4, B).argmax(-1)
        assert torch.equal(out[ex], z2.argmax(-1)[ex])


def test_bounded_limits():
    z2, z4 = _logits()
    assert torch.equal(bounded(z2, z4, 0.0), z2)
    big = float((z4 - z2).abs().max()) + 1
    assert torch.equal(bounded(z2, z4, big).argmax(-1), z4.argmax(-1))


def test_agreeing_rows_never_change():
    """z4~ is monotone in (z2, z4): if one token wins both loops it wins every B."""
    z2, z4 = _logits(seed=1)
    same = z2.argmax(-1) == z4.argmax(-1)
    for B in (0.05, 0.3, 1.0, 3.0):
        assert torch.equal(bounded(z2, z4, B).argmax(-1)[same], z2.argmax(-1)[same])


def test_compact_matches_full_vocab():
    z2, z4 = _logits(n=256, seed=2)
    st = compact_stats(z2, z4, k=32)
    assert st['cert'].float().mean() > 0.3  # flat Gaussian logits; real LM rows certify far more often
    ids = torch.cat([st['ids2'], st['ids4']], 1)
    c2 = torch.cat([st['z2_ids2'], st['z2_ids4']], 1)
    c4 = torch.cat([st['z4_ids2'], st['z4_ids4']], 1)
    ok = st['cert']
    for B in torch.linspace(0, 8, 41).tolist():
        full = bounded(z2, z4, B).argmax(-1)
        comp = compact_bounded_argmax(ids, c2, c4, B)
        assert torch.equal(full[ok], comp[ok])
    assert torch.equal(st['a2'], z2.argmax(-1)) and torch.equal(st['a4'], z4.argmax(-1))


def test_tie_break_smallest_id():
    vals = torch.tensor([[1.0, 3.0, 3.0, -float('inf')]])
    ids = torch.tensor([[9, 7, 4, -1]])
    assert argmax_min_id(vals, ids).item() == 4
    # torch.argmax on a full row returns the first maximal index == smallest id
    assert torch.tensor([[0.0, 2.0, 2.0]]).argmax(-1).item() == 1
