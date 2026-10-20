"""
v11 탐색: v10 엔진(아래→위 + 관찰 동등성) + 다수결로 과적합 잡기 + 고전 개념 탐지기(관찰 전용)

다수결 (오컴 + 합의)
  예시에 맞는 프로그램을 최대 K개 모은다 (첫 해 비용 + 창 이내).
  처음 보는 입력 12개(고정 난수)에서 각 후보의 출력을 계산 -> 가장 많은 후보가 동의하는 동작을 선택,
  동점이면 가장 싼(짧은) 후보. 우연히 예시에만 맞는 엉터리 규칙은 혼자라서 다수결에서 탈락.

고전 개념 탐지기
  김코어의 개념이 사람들이 '합계/최댓값/뒤집기/정렬...'이라 부르는 동작과 같은지 몰래 검사.
  김코어 학습에는 절대 쓰지 않음 (출석부에 몰래 동그라미만 침).
"""
import os, sys, time, itertools, collections
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('V10_OUT', os.path.join(HERE, '_v10_import_tmp'))
sys.path.insert(0, HERE)
import numpy as np
import v10
from v10 import variants, prim_costs, lambda_banks, Ctx, _compositions, TCFG, T_LIST, INT, BOOL
from v81 import arrows, compile_term, _call, LIST

K_FITS = int(os.environ.get('V11_K', 6)); OCCAM_WINDOW = 4

def _extra_inputs(arg_types, seed):
    r = np.random.default_rng(seed)
    def one(t):
        if t == INT: return int(r.integers(0, 10))
        return tuple(int(x) for x in r.integers(0, 10, size=int(r.integers(0, 8))))
    return [tuple(one(t) for t in arg_types) for _ in range(12)]

def grow_fits(ctx, lib, vlist, costs, vcost, max_cost, lam_banks, deadline, target, counter):
    """v10.grow과 같되, 목표에 맞는 항을 K개까지 모은다 (동등성 검사보다 목표 검사를 먼저)"""
    P = len(ctx.envs)
    seen = {INT: {}, BOOL: {}, T_LIST: {}}
    by_cost = collections.defaultdict(lambda: collections.defaultdict(list))
    fits = []; first_c = [None]
    def add(term, t, vals, c):
        if t == target[0] and vals == target[1]:
            fits.append((term, c))
            if first_c[0] is None: first_c[0] = c
        if vals in seen[t]: return
        seen[t][vals] = term; by_cost[c][t].append((term, vals))
    for term, t in ctx.vars:
        add(term, t, tuple(env[term[1]] for env in ctx.envs), vcost)
    for c in range(1, max_cost + 1):
        if time.time() > deadline or len(fits) >= K_FITS: break
        if first_c[0] is not None and c > first_c[0] + OCCAM_WINDOW: break
        for (pi, ats, ret) in vlist:
            pc = int(costs[pi]); rem = c - pc
            if rem < 0 or (rem == 0 and ats) or (rem > 0 and not ats): continue
            fn0 = lib.values[pi]
            if not ats:
                vals = tuple(fn0 for _ in range(P))
                try: hash(vals)
                except TypeError: continue
                add(('p', pi), ret, vals, c); continue
            k = len(ats)
            for split in _compositions(rem, k):
                pools = []
                for at, sc in zip(ats, split):
                    pools.append(lam_banks.get(at, {}).get(sc, []) if isinstance(at, str) else (by_cost[sc][at] if sc in by_cost else []))
                    if not pools[-1]: break
                if len(pools) < k or not all(pools): continue
                it = itertools.product(*pools)
                n_combo = 1
                for pl in pools: n_combo *= len(pl)
                if n_combo > TCFG['COMBO_CAP']: it = itertools.islice(it, TCFG['COMBO_CAP'])
                cnt = 0
                for combo in it:
                    cnt += 1
                    if cnt % 2000 == 0 and time.time() > deadline: break
                    term = ('p', pi)
                    for (a_term, _) in combo: term = ('a', term, a_term)
                    try:
                        vals = []
                        for p in range(P):
                            f = fn0
                            for (a_term, a_vals) in combo: f = f(a_vals[p])
                            vals.append(f)
                        vals = tuple(vals); hash(vals)
                    except Exception:
                        continue
                    counter[0] += 1
                    add(term, ret, vals, c)
                    if len(fits) >= K_FITS: return fits
                    if len(by_cost[c][ret]) > TCFG['CAP_PER_LEVEL']: break
    return fits

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
    extra = _extra_inputs(args_t, hash(str(task['ex'][0])) % (2 ** 31))
    groups = collections.OrderedDict()
    for term, c in fits:
        prog = term
        for _ in range(len(args_t)): prog = ('l', prog)
        try:
            f = compile_term(prog, lib)(())
            beh = []
            for inp in extra:
                try: beh.append(_call(f, inp))
                except Exception: beh.append('ERR')
            beh = tuple(beh)
        except Exception:
            continue
        g = groups.setdefault(beh, [0, c, prog]); g[0] += 1; g[1] = min(g[1], c)
    if not groups: return dict(prog=None, tried=counter[0], hidden_ok=False, n_fits=0)
    best = sorted(groups.values(), key=lambda g: (-g[0], g[1]))[0]
    prog = best[2]
    try:
        f = compile_term(prog, lib)(())
        hok = all(_call(f, inp) == out for inp, out in hid)
    except Exception:
        hok = False
    return dict(prog=prog, cost=best[1], tried=counter[0], hidden_ok=hok, n_fits=len(fits))

# ------------------------------------------------------------------ 고전 개념 탐지기 (관찰 전용)
def _classics():
    def srt(xs): return tuple(sorted(xs))
    def dedup(xs):
        out = []
        for x in xs:
            if x not in out: out.append(x)
        return tuple(out)
    return {
        ('list->int', '합계 sum'): lambda xs: sum(xs),
        ('list->int', '곱 product'): lambda xs: int(np.prod(xs)) if xs else 1,
        ('list->int', '최댓값 max'): lambda xs: max(xs) if xs else 'ERR',
        ('list->int', '최솟값 min'): lambda xs: min(xs) if xs else 'ERR',
        ('list->int', '마지막 last'): lambda xs: xs[-1] if xs else 'ERR',
        ('list->int', '짝수 개수'): lambda xs: sum(1 for x in xs if x % 2 == 0),
        ('list->list', '뒤집기 reverse'): lambda xs: tuple(reversed(xs)),
        ('list->list', '정렬 sort'): srt,
        ('list->list', '중복 제거 dedup'): dedup,
        ('list->list', '짝수만 evens'): lambda xs: tuple(x for x in xs if x % 2 == 0),
        ('list->list', '홀수만 odds'): lambda xs: tuple(x for x in xs if x % 2 == 1),
        ('list->list', '두 배 double'): lambda xs: tuple(2 * x for x in xs),
        ('list->list', '제곱 square'): lambda xs: tuple(x * x for x in xs),
        ('list->list', '+1 increment'): lambda xs: tuple(x + 1 for x in xs),
        ('list->list', '누적합 prefix sum'): lambda xs: tuple(itertools.accumulate(xs)),
        ('list->list', '마지막 빼기 init'): lambda xs: xs[:-1] if xs else 'ERR',
        ('list->bool', '정렬됐나 is_sorted'): lambda xs: all(a <= b for a, b in zip(xs, xs[1:])),
        ('list->bool', '대칭인가 palindrome'): lambda xs: xs == xs[::-1],
        ('int->int', '제곱 square'): lambda n: n * n,
        ('int->int', '1부터 합 triangle'): lambda n: n * (n + 1) // 2,
        ('int->int', '팩토리얼 factorial'): lambda n: int(np.prod(range(1, n + 1))) if n > 0 else 1,
        ('int->list', '거꾸로 세기 countdown'): lambda n: tuple(range(n, 0, -1)),
        ('int->list', '1부터 n'): lambda n: tuple(range(1, n + 1)),
    }

_REF = None
def _ref_table():
    global _REF
    if _REF is None:
        r = np.random.default_rng(777)
        L_in = [tuple(int(x) for x in r.integers(0, 10, size=int(r.integers(1, 8)))) for _ in range(16)]
        I_in = [int(x) for x in r.integers(0, 8, size=10)]
        _REF = dict(L_in=L_in, I_in=I_in)
    return _REF

def detect_classics(lib, start=None):
    """lib의 1차 개념(int/[int] 인자 하나)이 고전 동작과 같으면 [(개념 이름, 고전 이름)]"""
    ref = _ref_table(); found = []
    C = _classics()
    for i in range(start if start is not None else lib.n_base, len(lib)):
        for (pi, ats, ret) in variants(lib):
            if pi != i or len(ats) != 1: continue
            kind = {(T_LIST, INT): 'list->int', (T_LIST, T_LIST): 'list->list', (T_LIST, BOOL): 'list->bool',
                    (INT, INT): 'int->int', (INT, T_LIST): 'int->list'}.get((ats[0], ret))
            if kind is None: continue
            ins = ref['L_in'] if ats[0] == T_LIST else ref['I_in']
            try:
                mine = []
                for x in ins:
                    try: mine.append(lib.values[i](x))
                    except Exception: mine.append('ERR')
            except Exception:
                continue
            for (k, name), fn in C.items():
                if k != kind: continue
                theirs = []
                for x in ins:
                    try: theirs.append(fn(x))
                    except Exception: theirs.append('ERR')
                if mine == theirs: found.append((lib.names[i], name))
    return found
