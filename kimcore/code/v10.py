"""
김코어 v10 (D1) - 탐색 엔진 폭발: 아래에서 위로 쌓기 + 관찰 동등성 가지치기

v8.1 탐색: 위에서 아래로 프로그램을 하나씩 만들고 전부 실행. 결과가 같은 프로그램 수천 개를 따로 시험.
v10 탐색: 작은 프로그램부터 아래에서 위로 쌓으면서, '모든 예시에서 결과가 같은' 프로그램은 하나만 남긴다.
          (관찰 동등성, observational equivalence). 살아남는 건 '서로 다른 동작'뿐이라 탐색 공간이 지수적으로 줄어든다.
  - 람다(map/filter/fold의 함수 인자)는 별도 문맥에서 먼저 쌓는다: 예시에 나온 수들을 표본 점으로 써서
    같은 동작의 람다는 하나만 남김.
  - 비용 = 인식망이 준 확률의 -log. 인식망이 '이 재료 쓸 것 같다'고 하면 싸게, 아니면 비싸게 -> 학습이 탐색 순서를 바꿈.
  - 발명된 개념은 그냥 재료가 된다 (고차 개념도 람다 문맥으로 처리).
나머지(출제, 개념 발명, 꿈, 인식망, 탐침)는 v8.1 그대로. 탐색 엔진만 교체.
"""
import os, sys, time, math, itertools, collections
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(os.environ.get('V10_SEED', 1234))
OUT = os.environ.get('V10_OUT', os.path.join(HERE, f'v10_out_s{SEED}'))
os.makedirs(OUT, exist_ok=True)
os.environ['V8_OUT'] = OUT
os.environ.setdefault('V8_SEED', str(SEED))
sys.path.insert(0, HERE)
import numpy as np
import v81 as B
from v81 import INT, BOOL, LIST, arrows, instantiate, tvars, compile_term, _call

TCFG = dict(MAX_COST=int(os.environ.get('V10_MAX_COST', 22)), LAMBDA_COST=int(os.environ.get('V10_LAMBDA_COST', 8)),
            CAP_PER_LEVEL=int(os.environ.get('V10_CAP', 4000)), COMBO_CAP=60000)
T_LIST = LIST(INT)
LAM_KINDS = {('->', INT, INT): 'ii', ('->', INT, BOOL): 'ib', ('->', INT, ('->', INT, INT)): 'iii',
             ('->', INT, ('->', T_LIST, T_LIST)): 'ill'}

# ------------------------------------------------------------------ 재료의 '바닥 타입' 변형 (다형 -> int / [int])
_VAR_CACHE = {}
def variants(lib):
    """lib의 각 재료/개념을 int, [int]로 구체화한 변형들: [(i, arg_types, ret)] (인자 타입은 INT/BOOL/LIST 또는 람다 종류)"""
    key_ = tuple(str(t) for t in lib.types)       # 개념 '타입' 기준 캐시 (개체마다 개념이 달라도 안전)
    if key_ in _VAR_CACHE: return _VAR_CACHE[key_]
    out = []
    for i in range(len(lib)):
        t = lib.types[i]; vs = sorted(tvars(t))
        seen = set()
        for combo in itertools.product([INT, T_LIST], repeat=len(vs)):
            m = dict(zip(vs, combo))
            def g(x):
                if x[0] == 'tv': return m[x[1]]
                if x[0] == 'list': return ('list', g(x[1]))
                if x[0] == '->': return ('->', g(x[1]), g(x[2]))
                return x
            gt = g(t); args, ret = arrows(gt)
            if ret not in (INT, BOOL, T_LIST): continue
            ok = True; at = []
            for a in args:
                if a in (INT, BOOL, T_LIST): at.append(a)
                elif a in LAM_KINDS: at.append(LAM_KINDS[a])
                else: ok = False; break
            if ok and (tuple(at), ret) not in seen:
                seen.add((tuple(at), ret)); out.append((i, tuple(at), ret))
    _VAR_CACHE[key_] = out
    return out

def prim_costs(W, n):
    lg = W[:n].astype(float); m = lg.max()
    lp = lg - (m + math.log(np.exp(lg - m).sum()))
    c = np.clip(np.round(-lp * 0.75), 1, 6).astype(int)
    m2 = max(m, float(W[-1]))                     # 변수 가중치가 커도 넘치지 않게 (안정적 logsumexp)
    vlp = W[-1] - (m2 + math.log(np.exp(lg - m2).sum() + math.exp(W[-1] - m2)))
    return c, int(np.clip(round(-vlp * 0.75), 1, 4))

# ------------------------------------------------------------------ 한 문맥에서 아래에서 위로 쌓기
class Ctx:
    """points: 평가 점(각 점의 환경 튜플). vars: [(항, 타입)]"""
    def __init__(self, envs, vars_, depth):
        self.envs, self.vars, self.depth = envs, vars_, depth

def grow(ctx, lib, vlist, costs, vcost, max_cost, lam_banks, deadline, target=None, counter=None):
    """비용 순으로 서로 다른 동작의 항을 쌓는다. -> by_cost: c -> type -> [(term, vals)]"""
    P = len(ctx.envs)
    seen = {INT: {}, BOOL: {}, T_LIST: {}}
    by_cost = collections.defaultdict(lambda: collections.defaultdict(list))
    def add(term, t, vals, c):
        if vals in seen[t]: return None
        seen[t][vals] = term; by_cost[c][t].append((term, vals))
        if target is not None and t == target[0] and vals == target[1]: return term
        return None
    # 변수
    for term, t in ctx.vars:
        vals = tuple(env[term[1]] for env in ctx.envs)
        r = add(term, t, vals, vcost)
        if r is not None: return by_cost, r
    for c in range(1, max_cost + 1):
        if time.time() > deadline: break
        for (pi, ats, ret) in vlist:
            pc = int(costs[pi])
            rem = c - pc
            if rem < 0 or (rem == 0 and ats) or (rem > 0 and not ats): continue
            fn0 = lib.values[pi]
            if not ats:
                if rem != 0: continue
                vals = tuple(fn0 for _ in range(P))
                try: hash(vals)
                except TypeError: continue
                r = add(('p', pi), ret, vals, c)
                if r is not None: return by_cost, r
                continue
            k = len(ats)
            # 인자 비용 분배
            for split in _compositions(rem, k):
                pools = []
                for at, sc in zip(ats, split):
                    if isinstance(at, str):
                        pools.append(lam_banks.get(at, {}).get(sc, []))
                    else:
                        pools.append(by_cost[sc][at] if sc in by_cost else [])
                    if not pools[-1]: break
                if len(pools) < k or not all(pools): continue
                n_combo = 1
                for pl in pools: n_combo *= len(pl)
                it = itertools.product(*pools)
                if n_combo > TCFG['COMBO_CAP']: it = itertools.islice(it, TCFG['COMBO_CAP'])
                cnt = 0
                for combo in it:
                    cnt += 1
                    if cnt % 2000 == 0 and time.time() > deadline: break
                    term = ('p', pi)
                    for (a_term, _) in combo: term = ('a', term, a_term)
                    vals = []
                    try:
                        for p in range(P):
                            f = fn0
                            for at, (a_term, a_vals) in zip(ats, combo):
                                f = f(a_vals[p] if not isinstance(at, str) else a_vals[p])
                            vals.append(f)
                        vals = tuple(vals)
                        hash(vals)
                    except Exception:
                        continue
                    if counter is not None: counter[0] += 1
                    r = add(term, ret, vals, c)
                    if r is not None: return by_cost, r
                    if len(by_cost[c][ret]) > TCFG['CAP_PER_LEVEL']: break
    return by_cost, None

def _compositions(n, k):
    if k == 1:
        if n >= 1: yield (n,)
        return
    for i in range(1, n - k + 2):
        for rest in _compositions(n - i, k - 1): yield (i,) + rest

# ------------------------------------------------------------------ 람다 은행 (표본 점에서 같은 동작이면 하나만)
def lambda_banks(lib, vlist_fo, costs, vcost, in_envs, nargs, ints, lists, deadline, needed):
    banks = {}
    def mk(kind, pts_envs, vars_, wrap):
        ctx = Ctx(pts_envs, vars_, None)
        bc, _ = grow(ctx, lib, vlist_fo, costs, vcost, TCFG['LAMBDA_COST'], {}, deadline)
        bank = collections.defaultdict(list)
        ret = {'ii': INT, 'ib': BOOL, 'iii': INT, 'ill': T_LIST}[kind]
        nfree = {'ii': 1, 'ib': 1, 'iii': 2, 'ill': 2}[kind]
        for c, d in bc.items():
            for term, vals in d.get(ret, []):
                # 함수 값: 실제 호출 가능한 클로저 (예시마다 환경이 다름) -> 예시별 클로저 묶음
                lam = wrap(term)
                bank[c + 1].append((lam, None))
        return bank
    # 예시별 환경: (입력들 역순)
    if 'ii' in needed or 'ib' in needed:
        pts = [(x,) + env for env in in_envs for x in ints]
        vars_ = [(('v', 0), INT)] + [(('v', 1 + j), t) for j, t in enumerate(nargs)]
        for kind in ('ii', 'ib'):
            if kind in needed: banks[kind] = mk(kind, pts, vars_, lambda b: ('l', b))
    if 'iii' in needed:
        pts = [(acc, a) + env for env in in_envs for a in ints[:5] for acc in ints[:5]]
        vars_ = [(('v', 1), INT), (('v', 0), INT)] + [(('v', 2 + j), t) for j, t in enumerate(nargs)]
        banks['iii'] = mk('iii', pts, vars_, lambda b: ('l', ('l', b)))
    if 'ill' in needed:
        pts = [(acc, a) + env for env in in_envs for a in ints[:4] for acc in lists[:4]]
        vars_ = [(('v', 1), INT), (('v', 0), T_LIST)] + [(('v', 2 + j), t) for j, t in enumerate(nargs)]
        banks['ill'] = mk('ill', pts, vars_, lambda b: ('l', ('l', b)))
    return banks

def search(task, lib, W, budget, time_cap=None):
    """v8.1 search와 같은 인터페이스. 아래에서 위로 + 관찰 동등성."""
    t0 = time.time(); cap = time_cap or B.CFG['TIME_CAP']; deadline = t0 + cap
    req = task['type']; vis = task['ex'][:4]; hid = task['ex'][4:]
    args_t, ret_t = arrows(req)
    n = len(lib)
    costs, vcost = prim_costs(W, n)
    vl = variants(lib)
    vl_fo = [v for v in vl if not any(isinstance(a, str) for a in v[1])]
    needed = set(a for v in vl for a in v[1] if isinstance(a, str))
    in_envs = [tuple(reversed(inp)) for inp, _ in vis]
    nargs_rev = [args_t[len(args_t) - 1 - k] for k in range(len(args_t))]   # ('v',k) -> 타입
    ints = set([0, 1, 2, 3])
    lists = []
    for inp, out in vis:
        for v in list(inp) + [out]:
            if isinstance(v, bool): continue
            if isinstance(v, int): ints.add(v)
            else:
                ints.update(v[:6]); lists.append(v)
                if len(v) > 1: lists.append(v[1:])
    ints = sorted(ints)[:8]; lists = (lists + [()])[:6]
    # 람다는 '예시별 실제 클로저'가 필요: 람다 항을 예시 환경에서 컴파일
    raw_banks = lambda_banks(lib, vl_fo, costs, vcost, in_envs, nargs_rev, ints, lists, t0 + cap * 0.35, needed)
    lam_banks = {}
    for kind, bank in raw_banks.items():
        lam_banks[kind] = {}
        for c, items in bank.items():
            lst = []
            for lam, _ in items:
                try:
                    fns = tuple(compile_term(lam, lib)(env) for env in in_envs)
                except Exception:
                    continue
                lst.append((lam, fns))
            lam_banks[kind][c] = lst
    target = (ret_t, tuple(out for _, out in vis))
    ctx = Ctx(in_envs, [(('v', k), nargs_rev[k]) for k in range(len(args_t))], 0)
    counter = [0]
    _, found = grow(ctx, lib, vl, costs, vcost, TCFG['MAX_COST'], lam_banks, deadline, target, counter)
    if found is None:
        return dict(prog=None, tried=counter[0], hidden_ok=False)
    prog = found
    for _ in range(len(args_t)): prog = ('l', prog)
    try:
        f = compile_term(prog, lib)(())
        assert all(_call(f, inp) == out for inp, out in vis)
        hok = all(_call(f, inp) == out for inp, out in hid)
    except Exception:
        return dict(prog=None, tried=counter[0], hidden_ok=False)
    return dict(prog=prog, cost=0, tried=counter[0], hidden_ok=hok)

# ------------------------------------------------------------------ 엔진 교체 + 비교 측정
def benchmark(n=150):
    """같은 탐침에서 v8.1 엔진 vs v10 엔진: 해결 수와 시간"""
    lib = B.Library(); W = np.zeros(len(lib) + 1)
    probes = B.probe_set(9999)[:n]
    old = B.search
    res = {}
    for name, fn in (('v8.1 위→아래', old), ('v10 아래→위+동등성', search)):
        B.search = fn
        t = time.time()
        rs = B.solve_all(lib, None, probes, B.CFG['SEARCH_BUDGET'])
        ok = sum(r['prog'] is not None and r['hidden_ok'] for r in rs)
        res[name] = (ok, time.time() - t, [r['prog'] is not None and r['hidden_ok'] for r in rs])
    B.search = search
    a, b = res['v8.1 위→아래'], res['v10 아래→위+동등성']
    only_new = sum(1 for x, y in zip(a[2], b[2]) if y and not x)
    only_old = sum(1 for x, y in zip(a[2], b[2]) if x and not y)
    return a, b, only_new, only_old

if __name__ == '__main__':
    B.search = search
    if os.environ.get('V10_BENCH'):
        a, b, nn, no = benchmark(int(os.environ.get('V10_BENCH')))
        print(f"v8.1 엔진: {a[0]}개 / {a[1]:.0f}s   |   v10 엔진: {b[0]}개 / {b[1]:.0f}s   |   v10만 푼 것 {nn}, v8.1만 푼 것 {no}")
    else:
        B.log(" ⚙️ [v10 엔진] 아래에서 위로 쌓기 + 관찰 동등성 가지치기 (출제/개념발명/꿈/인식망/탐침은 v8.1 동일)")
        B.main()
