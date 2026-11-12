"""Problems, calibration/test split, prompts and (sanity-level) grading."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
DATASETS = ('gsm8k', 'math500', 'mbppplus')
SPLIT_SALT = 'loopexp-split-v1'
CALIB_FRACTION = 0.30

MATH_SUFFIX = 'Please reason step by step, and put your final answer within \\boxed{}.'


def split_of(dataset: str, pid: str) -> str:
    """Deterministic problem-level split; every token of a response follows its problem."""
    h = int(hashlib.sha256(f'{SPLIT_SALT}:{dataset}:{pid}'.encode()).hexdigest(), 16)
    return 'calib' if (h % 10_000) < CALIB_FRACTION * 10_000 else 'test'


def load_problems(dataset: str) -> list[dict]:
    import pandas as pd
    if dataset == 'gsm8k':
        df = pd.read_parquet(next((RAW / 'gsm8k' / 'main').glob('test-*.parquet')))
        out = [dict(pid=f'gsm8k-{i:04d}', question=r.question,
                    gold=r.answer.split('####')[-1].strip().replace(',', ''))
               for i, r in enumerate(df.itertuples())]
    elif dataset == 'math500':
        rows = [json.loads(line) for line in open(RAW / 'MATH-500' / 'test.jsonl')]
        out = [dict(pid=r['unique_id'], question=r['problem'], gold=r['answer'],
                    level=r.get('level'), subject=r.get('subject')) for r in rows]
    elif dataset == 'mbppplus':
        df = pd.read_parquet(next((RAW / 'mbppplus' / 'data').glob('test-*.parquet')))
        out = [dict(pid=f'Mbpp/{r.task_id}', question=r.prompt, tests=list(r.test_list),
                    test_imports=list(r.test_imports)) for r in df.itertuples()]
    else:
        raise ValueError(dataset)
    for p in out:
        p['dataset'] = dataset
        p['split'] = split_of(dataset, p['pid'])
    return out


def user_message(p: dict) -> str:
    if p['dataset'] in ('gsm8k', 'math500'):
        return f"{p['question']}\n{MATH_SUFFIX}"
    test = p['tests'][0]
    return (f"{p['question'].strip()}\nYour code should pass this test:\n{test}\n"
            'Write the complete Python function in a single ```python code block.')


def build_prompt(p: dict, tokenizer, fmt: str = 'chat') -> list[int]:
    if fmt != 'chat':
        raise ValueError(f'unknown prompt format {fmt}')
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': user_message(p)}],
                                         tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)['input_ids']


def stop_ids(tokenizer) -> set[int]:
    ids = {tokenizer.eos_token_id}
    for t in ('<|im_end|>', '<|endoftext|>'):
        i = tokenizer.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tokenizer.unk_token_id:
            ids.add(i)
    return {i for i in ids if i is not None}


# ----------------------------------------------------------------------------- grading
def last_boxed(text: str) -> str | None:
    i = text.rfind('\\boxed')
    if i < 0:
        return None
    j = text.find('{', i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {'{': 1, '}': -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def _num(s: str) -> float | None:
    s = s.replace(',', '').replace('$', '').strip()
    m = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', s)
    if not m:
        return None
    x = m[-1]
    try:
        if '/' in x:
            a, b = x.split('/')
            return float(a) / float(b)
        return float(x)
    except (ValueError, ZeroDivisionError):
        return None


def grade(p: dict, text: str) -> dict:
    """Sanity-level correctness of the reference trajectory (not a phase-2 metric)."""
    if p['dataset'] == 'gsm8k':
        box = last_boxed(text)
        pred = _num(box) if box is not None else _num(text)
        gold = float(p['gold'])
        return dict(pred=box, correct=pred is not None and abs(pred - gold) < 1e-6)
    if p['dataset'] == 'math500':
        from math_verify import parse, verify
        box = last_boxed(text)
        try:
            ok = bool(verify(parse(f"\\boxed{{{p['gold']}}}"), parse(text)))
        except Exception:  # noqa: BLE001 - grader failures count as wrong
            ok = False
        return dict(pred=box, correct=ok)
    code = extract_code(text)
    try:
        ast.parse(code)
        syntax_ok = bool(code.strip())
    except SyntaxError:
        syntax_ok = False
    return dict(pred=None, correct=None, code=code, syntax_ok=syntax_ok)


def extract_code(text: str) -> str:
    m = re.findall(r'```(?:python)?\n(.*?)```', text, flags=re.S)
    if m:
        return m[0]
    m = re.findall(r'```(?:python)?\n(.*)', text, flags=re.S)
    return m[0] if m else text
