"""
v20 탐색: v10 엔진으로 예시에 맞는 풀이를 최대 K개 모은 뒤 (첫 해 비용 + 창 이내),
'기본 재료 기준 크기'가 가장 작은 풀이를 고른다 (기본 재료 오컴).
개념은 '빨리 찾는' 데 쓰고, '무엇이 진짜 설명인가'는 펼쳤을 때의 단순함으로 판정.
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import v11search as S_
from v11search import variants, prim_costs, lambda_banks, Ctx, TCFG, T_LIST, INT, BOOL, grow_fits
from v81 import arrows, compile_term, _call, base_size

def search(task, lib, W, budget, time_cap=None, cap_default=1.0):
    t0 = time.time(); cap = time_cap or cap_default; deadline = t0 + cap
    req = task['type']; vis = task['ex'][:4]; hid = task['ex'][4:]
    args_t, ret_t = arrows(req)
    costs, vcost = prim_costs(W, len(lib))
    vl = variants(lib)
    vl_fo = [v for v in vl if not any(isinstance(a, str) for a in v[1])]
    needed = set(a for v in vl for a in v[1] if isinstance(a, str))
    in_envs = [tuple(reversed(inp)) for inp, _ in vis]
    nargs_rev = [args_t[len(args_t) - 1 - k] for k in range(len(args_t))]
    ints = set([0, 1, 2, 3]); lists = []
    for inp, out in vis:
        for v in list(inp) + [out]:
            if isinstance(v, bool): continue
            if isinstance(v, int): ints.add(v)
            else:
                ints.update(v[:6]); lists.append(v)
                if len(v) > 1: lists.append(v[1:])
    ints = sorted(ints)[:8]; lists = (lists + [()])[:6]
    raw = lambda_banks(lib, vl_fo, costs, vcost, in_envs, nargs_rev, ints, lists, t0 + cap * 0.35, needed)
    lam_banks = {}
    for kind, bank in raw.items():
        lam_banks[kind] = {}
        for c, items in bank.items():
            lst = []
            for lam, _ in items:
                try: lst.append((lam, tuple(compile_term(lam, lib)(env) for env in in_envs)))
                except Exception: continue
            lam_banks[kind][c] = lst
    target = (ret_t, tuple(out for _, out in vis))
    ctx = Ctx(in_envs, [(('v', k), nargs_rev[k]) for k in range(len(args_t))], 0)
    counter = [0]
    fits = grow_fits(ctx, lib, vl, costs, vcost, TCFG['MAX_COST'], lam_banks, deadline, target, counter)
    if not fits: return dict(prog=None, tried=counter[0], hidden_ok=False, n_fits=0)
    best = None
    for term, c in fits:
        prog = term
        for _ in range(len(args_t)): prog = ('l', prog)
        key = (base_size(prog, lib), c)
        if best is None or key < best[0]: best = (key, prog, c)
    prog = best[1]
    try:
        f = compile_term(prog, lib)(())
        hok = all(_call(f, inp) == out for inp, out in hid)
    except Exception:
        hok = False
    return dict(prog=prog, cost=best[2], tried=counter[0], hidden_ok=hok, n_fits=len(fits))
