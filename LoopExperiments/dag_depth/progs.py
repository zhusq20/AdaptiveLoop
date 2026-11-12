"""Synthetic straight-line programs with a controlled dependency DAG.

Every program is SSA: init lines `v = c`, op lines `v = u + c`, `v = u - c` (one
variable operand, one literal) or `v = u + w`, `v = u - w` (two variables), and a final
`print(v)`. All values, intermediate and final, are single digits 0..9, so the answer
is one token.

Two-branch family TB(L1, L2): two init variables a, b. Branch 1 applies L1 var-lit ops
to a, branch 2 applies L2 var-lit ops to b, and one var-var op merges the two branch
ends. N = L1 + L2 + 1 ops, longest chain D = max(L1, L2) + 1. L2 = 0 is a serial chain
(b is merged in at the end). For fixed N every split has the same line count, the same
number of var-lit / var-var ops and (because lines tokenize to fixed lengths) the same
token count. Only the dependency structure changes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

LETTERS = list('abcdefghjkmnpqrstuvwxyz')  # no i, l, o


@dataclass
class Line:
    var: str
    kind: str                 # 'init' | 'lit' | 'var' | 'print'
    args: tuple = ()          # init: (c,), lit: (u, op, c), var: (u, op, w), print: (v,)
    branch: int = 0           # 1, 2 for branch lines, 0 for merge / print
    step: int = 0             # index within the branch (0 = init line)

    def render(self) -> str:
        if self.kind == 'init':
            return f'{self.var} = {self.args[0]}'
        if self.kind in ('lit', 'var'):
            u, op, w = self.args
            return f'{self.var} = {u} {op} {w}'
        return f'print({self.args[0]})'


@dataclass
class Prog:
    lines: list[Line]
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return '\n'.join(l.render() for l in self.lines) + '\n'

    def values(self) -> dict[str, int]:
        env: dict[str, int] = {}
        for l in self.lines:
            if l.kind == 'init':
                env[l.var] = l.args[0]
            elif l.kind in ('lit', 'var'):
                u, op, w = l.args
                x = env[u]
                y = env[w] if l.kind == 'var' else w
                env[l.var] = x + y if op == '+' else x - y
        return env

    @property
    def answer(self) -> int:
        return self.values()[self.lines[-1].args[0]]

    def check(self) -> None:
        env: dict[str, int] = {}
        exec(self.text.replace('print(', '__out = ('), {}, env)  # noqa: S102 (our own text)
        assert env['__out'] == self.answer, (self.text, env['__out'], self.answer)
        assert all(0 <= v <= 9 for k, v in env.items() if not k.startswith('__')), self.text


def _lit_step(rng: random.Random, v: int, consts=range(1, 10)) -> tuple[str, int]:
    opts = [(op, c) for op in '+-' for c in consts if 0 <= (v + c if op == '+' else v - c) <= 9]
    return rng.choice(opts)


def two_branch(rng: random.Random, L1: int, L2: int, order: str = 'grouped',
               a0: int | None = None, b0: int | None = None,
               branch_lits: tuple[list, list] | None = None, merge_op: str | None = None,
               names: list[str] | None = None, consts=range(1, 10)) -> Prog | None:
    """Sample a TB(L1, L2) program. Optional arguments pin parts of it (used to build
    token-aligned counterfactuals). Returns None if pinned parts leave 0..9."""
    n_vars = 2 + L1 + L2 + 1
    names = names or rng.sample(LETTERS, n_vars)
    a, b, m = names[0], names[1], names[-1]
    p = names[2:2 + L1]
    q = names[2 + L1:2 + L1 + L2]
    va = rng.randrange(10) if a0 is None else a0
    vb = rng.randrange(10) if b0 is None else b0
    lines = [Line(a, 'init', (va,), 1, 0), Line(b, 'init', (vb,), 2, 0)]
    br: list[list[Line]] = [[], []]
    ends = []
    for bi, (src, val, vs, L) in enumerate(((a, va, p, L1), (b, vb, q, L2))):
        prev, v = src, val
        for k in range(L):
            if branch_lits is not None:
                op, c = branch_lits[bi][k]
            else:
                op, c = _lit_step(rng, v, consts)
            v = v + c if op == '+' else v - c
            if not 0 <= v <= 9:
                return None
            br[bi].append(Line(vs[k], 'lit', (prev, op, c), bi + 1, k + 1))
            prev = vs[k]
        ends.append((prev, v))
    (pe, pv), (qe, qv) = ends
    ops = [op for op in '+-' if 0 <= (pv + qv if op == '+' else pv - qv) <= 9]
    if merge_op is not None:
        ops = [merge_op] if merge_op in ops else []
    if not ops:
        return None
    mop = rng.choice(ops)
    if order == 'grouped':
        body = br[0] + br[1]
    elif order == 'interleaved':
        body = []
        for k in range(max(L1, L2)):
            for bi in (0, 1):
                if k < len(br[bi]):
                    body.append(br[bi][k])
    else:
        raise ValueError(order)
    lines += body + [Line(m, 'var', (pe, mop, qe), 0, 0), Line('', 'print', (m,), 0, 0)]
    prog = Prog(lines, dict(family='TB', L1=L1, L2=L2, N=L1 + L2 + 1, D=max(L1, L2) + 1,
                            order=order, A=pv, B=qv, merge_op=mop))
    return prog


def literals(prog: Prog) -> list[int]:
    out = []
    for l in prog.lines:
        if l.kind == 'init':
            out.append(l.args[0])
        elif l.kind == 'lit':
            out.append(l.args[2])
    return out


def balanced_cell(rng: random.Random, L1: int, L2: int, order: str, n: int,
                  max_tries: int = 200000, consts=range(1, 10)) -> list[Prog]:
    """n programs of one cell with the answer distribution uniform over 0..9."""
    assert n % 10 == 0
    per = n // 10
    buckets: dict[int, list[Prog]] = {d: [] for d in range(10)}
    for _ in range(max_tries):
        pr = two_branch(rng, L1, L2, order, consts=consts)
        if pr is None:
            continue
        d = pr.answer
        if len(buckets[d]) < per:
            pr.check()
            buckets[d].append(pr)
        if all(len(v) == per for v in buckets.values()):
            break
    else:
        raise RuntimeError(f'could not balance cell {(L1, L2, order)}')
    out = [p for d in range(10) for p in buckets[d]]
    rng.shuffle(out)
    return out


HEADER = 'Each program uses single-digit integers. Write the value it prints.\n\n'


def fewshot_block(seed: int = 12345, shapes=((1, 0), (1, 1), (2, 1), (3, 0)), consts=range(1, 10)) -> str:
    """Fixed few-shot examples (shallow shapes, distinct answers)."""
    rng = random.Random(seed)
    used: set[int] = set()
    parts = []
    for L1, L2 in shapes:
        while True:
            pr = two_branch(rng, L1, L2, 'grouped', consts=consts)
            if pr is not None and (pr.answer not in used or len(used) >= 10):
                break
        used.add(pr.answer)
        parts.append(pr.text + f'Output: {pr.answer}\n\n')
    return HEADER + ''.join(parts)


def prompt(prog: Prog, fewshot: str) -> str:
    # Digits tokenize as ' ' + digit, so the prompt ends with the space token and the
    # answer is exactly one digit token.
    return fewshot + prog.text + 'Output: '
