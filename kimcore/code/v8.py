"""
김코어 v8.0 - 복리로 쌓이는 지능 (도메인 무관 RSI)

주는 것: 계산 재료뿐 (CPU 명령어 수준)
  수: 0, 1, +, -, *, mod      비교: eq, <      조건: if
  목록: nil, cons, car, cdr, empty?, length, range
  반복: fold, map, filter      함수: 람다 (변수 묶기)
주지 않는 것: '합계', '최대값', '뒤집기', '정렬', '짝수', '세기' ... 개념은 하나도 없음.

루프 (한 세대)
  1. 출제(깨어있기): 지금까지 발명한 개념까지 섞어서 숨은 프로그램을 무작위 생성 -> 입력/출력 예시 6개.
                    김코어는 4개만 보고 프로그램을 찾고, 숨겨둔 2개로 진짜 맞았는지 판정.
  2. 탐색: 타입이 맞는 프로그램만 확률 순서로 열거 (인식망이 예시를 보고 어떤 재료가 필요할지 감을 줌)
  3. 잠자기(개념 발명): 푼 프로그램 전체에서 반복되는 구조를 '구멍 뚫린 틀'로 추출(반통일화)
                        -> 전체 설명 길이를 가장 많이 줄이는 틀을 새 개념(함수)으로 등록 -> 코퍼스 재작성
  4. 꿈꾸기: 현재 개념들로 가상 문제를 무한 생성해 인식망 훈련
  -> 개념 위에 개념이 쌓이면, 같은 탐색 예산으로 '기본 재료 기준 훨씬 큰' 프로그램을 찾게 됨 = 복리

성장 측정 (벤치마크 아님, 김코어 내부)
  - 고정 탐침 문제집: 기본 재료로 만든 숨은 프로그램, 크기 단계별. 학습에 절대 안 씀. 같은 예산으로 몇 % 푸나
  - 해결한 프로그램의 '기본 재료 기준 크기' 최대/평균: 개념이 쌓일수록 커져야 함
  - 이해도: 푼 문제 전체의 설명 길이(재료 개수) 압축률
  - 개념 깊이: 개념이 다른 개념 위에 몇 층 쌓였나
  - 과적합 거짓말: 보이는 4개는 맞추고 숨긴 2개에서 틀림
  - 과거의 나 vs 지금의 나: 0세대 김코어와 최종 김코어를 최종 세대가 낸 문제로 대결
"""
import os, sys, time, json, math, random, itertools, collections, copy
import multiprocessing as mp
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = dict(
    SEED=int(os.environ.get('V8_SEED', 1234)),
    GENS=int(os.environ.get('V8_GENS', 20)),
    TASKS_PER_GEN=int(os.environ.get('V8_TASKS', 120)),
    SEARCH_BUDGET=int(os.environ.get('V8_BUDGET', 4000)),     # 문제당 시험해 보는 프로그램 수
    PROBE_PER_BIN=int(os.environ.get('V8_PROBE', 30)),
    WORKERS=int(os.environ.get('V8_WORKERS', 4)),
    MAX_INVENT_PER_GEN=3, MAX_PRODS=160, DREAMS=600, TIME_CAP=float(os.environ.get('V8_TIME_CAP', 2.5)),
    PROBE_N=int(os.environ.get('V8_PROBE_N', 300)), EASY_KEEP=0.2, QUICK=150,
    COST_STEP=1.5, MAX_COST=24.0, EPS=0.08, MAX_DEPTH=7,
)
OUT = os.environ.get('V8_OUT', os.path.join(HERE, f"v8_out_s{CFG['SEED']}"))
os.makedirs(OUT, exist_ok=True)
_T0 = time.time()
_LOG = open(os.path.join(OUT, 'run.txt'), 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); _LOG.write(s + '\n'); _LOG.flush()

# ==============================================================================
# 1. 타입 (Hindley-Milner 축약판)
# ==============================================================================
INT, BOOL = ('int',), ('bool',)
def LIST(t): return ('list', t)
def ARR(*ts):
    t = ts[-1]
    for a in reversed(ts[:-1]): t = ('->', a, t)
    return t
def TV(i): return ('tv', i)

_tv = itertools.count(1000)
def fresh(): return TV(next(_tv))

def tvars(t, acc=None):
    acc = set() if acc is None else acc
    if t[0] == 'tv': acc.add(t[1])
    elif t[0] == 'list': tvars(t[1], acc)
    elif t[0] == '->': tvars(t[1], acc); tvars(t[2], acc)
    return acc

def instantiate(t):
    m = {v: fresh() for v in tvars(t)}
    def go(x):
        if x[0] == 'tv': return m[x[1]]
        if x[0] == 'list': return ('list', go(x[1]))
        if x[0] == '->': return ('->', go(x[1]), go(x[2]))
        return x
    return go(t)

def walk(t, s):
    while t[0] == 'tv' and t[1] in s: t = s[t[1]]
    return t

def resolve(t, s):
    t = walk(t, s)
    if t[0] == 'list': return ('list', resolve(t[1], s))
    if t[0] == '->': return ('->', resolve(t[1], s), resolve(t[2], s))
    return t

def occurs(v, t, s):
    t = walk(t, s)
    if t[0] == 'tv': return t[1] == v
    if t[0] == 'list': return occurs(v, t[1], s)
    if t[0] == '->': return occurs(v, t[1], s) or occurs(v, t[2], s)
    return False

def unify(a, b, s):
    a, b = walk(a, s), walk(b, s)
    if a == b: return s
    if a[0] == 'tv':
        if occurs(a[1], b, s): return None
        s = dict(s); s[a[1]] = b; return s
    if b[0] == 'tv': return unify(b, a, s)
    if a[0] != b[0]: return None
    if a[0] == 'list': return unify(a[1], b[1], s)
    if a[0] == '->':
        s = unify(a[1], b[1], s)
        return None if s is None else unify(a[2], b[2], s)
    return s if a == b else None

def arrows(t):
    args = []
    while t[0] == '->': args.append(t[1]); t = t[2]
    return args, t

def tshow(t):
    if t[0] == 'tv': return f"t{t[1] % 100}"
    if t[0] == 'list': return f"[{tshow(t[1])}]"
    if t[0] == '->': return f"({tshow(t[1])}→{tshow(t[2])})"
    return t[0]

# ==============================================================================
# 2. 계산 재료 (도메인 개념 0개)
# ==============================================================================
class EvalError(Exception): pass
LIM_INT, LIM_LEN = 10 ** 6, 40
def ci(x):
    if not isinstance(x, (int, bool)) or abs(x) > LIM_INT: raise EvalError
    return x
def cl(xs):
    if len(xs) > LIM_LEN: raise EvalError
    return xs
def _car(xs):
    if not xs: raise EvalError
    return xs[0]
def _cdr(xs):
    if not xs: raise EvalError
    return xs[1:]
def _mod(a):
    def m(b):
        if b == 0: raise EvalError
        return ci(a % b)
    return m
def _range(n):
    if not (0 <= n <= 20): raise EvalError
    return tuple(range(n))
def _fold(xs):
    def z_(z):
        def f_(f):
            acc = z
            for x in reversed(xs): acc = f(x)(acc)
            return acc
        return f_
    return z_

A, B = TV(0), TV(1)
BASE = [
    ('0', INT, 0), ('1', INT, 1),
    ('+', ARR(INT, INT, INT), lambda a: lambda b: ci(a + b)),
    ('-', ARR(INT, INT, INT), lambda a: lambda b: ci(a - b)),
    ('*', ARR(INT, INT, INT), lambda a: lambda b: ci(a * b)),
    ('mod', ARR(INT, INT, INT), _mod),
    ('eq', ARR(INT, INT, BOOL), lambda a: lambda b: a == b),
    ('<', ARR(INT, INT, BOOL), lambda a: lambda b: a < b),
    ('if', ARR(BOOL, A, A, A), lambda c: lambda a: lambda b: a if c else b),
    ('nil', LIST(A), ()),
    ('cons', ARR(A, LIST(A), LIST(A)), lambda x: lambda xs: cl((x,) + xs)),
    ('car', ARR(LIST(A), A), _car),
    ('cdr', ARR(LIST(A), LIST(A)), _cdr),
    ('empty?', ARR(LIST(A), BOOL), lambda xs: len(xs) == 0),
    ('length', ARR(LIST(A), INT), lambda xs: len(xs)),
    ('range', ARR(INT, LIST(INT)), _range),
    ('fold', ARR(LIST(A), B, ARR(A, B, B), B), _fold),
    ('map', ARR(ARR(A, B), LIST(A), LIST(B)), lambda f: lambda xs: cl(tuple(f(x) for x in xs))),
    ('filter', ARR(ARR(A, BOOL), LIST(A), LIST(A)), lambda p: lambda xs: tuple(x for x in xs if p(x))),
]

# ==============================================================================
# 3. 프로그램 (de Bruijn 람다 항)
#    ('p', i) 재료/개념 i  |  ('v', k) 변수  |  ('l', body) 람다  |  ('a', f, x) 적용  |  ('h', j) 구멍(틀 전용)
# ==============================================================================
class Library:
    def __init__(self):
        self.names, self.types, self.values, self.bodies, self.depth, self.base_leaves = [], [], [], [], [], []
        for n, t, v in BASE:
            self.add(n, t, v, None, 0, 1)
    def add(self, name, t, value, body, depth, base_leaves):
        self.names.append(name); self.types.append(t); self.values.append(value); self.bodies.append(body)
        self.depth.append(depth); self.base_leaves.append(base_leaves)
        return len(self.names) - 1
    def __len__(self): return len(self.names)
    @property
    def n_base(self): return len(BASE)

def leaves(t):
    k = t[0]
    if k == 'a': return leaves(t[1]) + leaves(t[2])
    if k == 'l': return leaves(t[1])
    return 1

def base_size(t, lib):
    k = t[0]
    if k == 'a': return base_size(t[1], lib) + base_size(t[2], lib)
    if k == 'l': return base_size(t[1], lib)
    if k == 'p': return lib.base_leaves[t[1]]
    return 1

def show(t, lib, d=0):
    k = t[0]
    if k == 'p': return lib.names[t[1]]
    if k == 'v': return f"${t[1]}"
    if k == 'h': return f"?{t[1]}"
    if k == 'l': return f"(λ {show(t[1], lib, d + 1)})"
    f, args = t, []
    while f[0] == 'a': args.append(f[2]); f = f[1]
    return f"({show(f, lib, d)} {' '.join(show(a, lib, d) for a in reversed(args))})"

def compile_term(t, lib):
    k = t[0]
    if k == 'p':
        v = lib.values[t[1]]; return lambda env: v
    if k == 'v':
        i = t[1]; return lambda env: env[i]
    if k == 'l':
        b = compile_term(t[1], lib); return lambda env: (lambda x: b((x,) + env))
    f = compile_term(t[1], lib); x = compile_term(t[2], lib)
    return lambda env: f(env)(x(env))

def run(t, lib, args):
    fn = compile_term(t, lib)(())
    for a in args: fn = fn(a)
    return fn

# ==============================================================================
# 4. 타입 추론 (개념 등록 시 타입 계산)
# ==============================================================================
def infer(t, lib, env=(), s=None):
    s = {} if s is None else s
    k = t[0]
    if k == 'p': return instantiate(lib.types[t[1]]), s
    if k == 'v':
        if t[1] >= len(env): raise TypeError
        return env[t[1]], s
    if k == 'l':
        a = fresh(); bt, s = infer(t[1], lib, (a,) + env, s)
        return ('->', a, bt), s
    ft, s = infer(t[1], lib, env, s); xt, s = infer(t[2], lib, env, s)
    r = fresh(); s2 = unify(ft, ('->', xt, r), s)
    if s2 is None: raise TypeError
    return r, s2

def generalize(t):
    m = {}
    def go(x):
        if x[0] == 'tv':
            if x[1] not in m: m[x[1]] = TV(len(m))
            return m[x[1]]
        if x[0] == 'list': return ('list', go(x[1]))
        if x[0] == '->': return ('->', go(x[1]), go(x[2]))
        return x
    return go(t)

# ==============================================================================
# 5. 탐색: 타입 맞는 프로그램을 확률 순서(비용 창)로 열거
# ==============================================================================
def enum(req, env, s, budget, W, lib, depth):
    req = resolve(req, s)
    if req[0] == '->':
        for body, s2, c in enum(req[2], (req[1],) + env, s, budget, W, lib, depth):
            yield ('l', body), s2, c
        return
    if depth > CFG['MAX_DEPTH'] or budget <= 0: return
    cands = []
    for i, vt in enumerate(env):
        args, ret = arrows(resolve(vt, s)); s2 = unify(ret, req, s)
        if s2 is not None: cands.append((('v', i), args, s2, W[-1]))
    for pi in range(len(lib)):
        args, ret = arrows(instantiate(lib.types[pi])); s2 = unify(ret, req, s)
        if s2 is not None: cands.append((('p', pi), args, s2, W[pi]))
    if not cands: return
    lg = np.array([c[3] for c in cands]); lp = lg - (np.log(np.exp(lg - lg.max()).sum()) + lg.max())
    for k in np.argsort(-lp):
        leaf, args, s2, _ = cands[k]
        c0 = -lp[k] + CFG['EPS']
        if c0 > budget: continue
        yield from _apply(leaf, args, env, s2, budget - c0, c0, W, lib, depth)

def _apply(expr, args, env, s, budget, cost, W, lib, depth):
    if not args:
        yield expr, s, cost; return
    for a, s2, c in enum(args[0], env, s, budget, W, lib, depth + 1):
        yield from _apply(('a', expr, a), args[1:], env, s2, budget - c, cost + c, W, lib, depth)

def search(task, lib, W, budget, time_cap=None):
    """-> dict(prog, cost, tried, hidden_ok)"""
    req = task['type']; vis = task['ex'][:4]; hid = task['ex'][4:]
    tried = 0; hi = CFG['COST_STEP']; t0 = time.time(); cap = time_cap or CFG['TIME_CAP']; seen = 0
    while hi <= CFG['MAX_COST'] and tried < budget:
        lo = hi - CFG['COST_STEP']
        for prog, s, c in enum(req, (), {}, hi, W, lib, 0):
            seen += 1
            if seen % 200 == 0 and time.time() - t0 > cap:
                return dict(prog=None, tried=tried, hidden_ok=False)
            if c <= lo: continue
            tried += 1
            try:
                f = compile_term(prog, lib)(())
                ok = all(_call(f, inp) == out for inp, out in vis)
            except Exception:
                ok = False
            if ok:
                try: hok = all(_call(f, inp) == out for inp, out in hid)
                except Exception: hok = False
                return dict(prog=prog, cost=c, tried=tried, hidden_ok=hok)
            if tried >= budget: break
        hi += CFG['COST_STEP']
    return dict(prog=None, tried=tried, hidden_ok=False)

def _call(f, inp):
    for a in inp: f = f(a)
    return f

# ==============================================================================
# 6. 출제기: 현재 개념까지 섞어서 숨은 프로그램 무작위 생성
# ==============================================================================
REQUESTS = [ARR(LIST(INT), LIST(INT)), ARR(LIST(INT), INT), ARR(INT, LIST(INT)), ARR(LIST(INT), BOOL),
            ARR(INT, INT), ARR(LIST(INT), LIST(INT), LIST(INT)), ARR(INT, LIST(INT), LIST(INT))]

def sample(req, env, s, W, lib, rng, depth, maxdepth):
    req = resolve(req, s)
    if req[0] == '->':
        r = sample(req[2], (req[1],) + env, s, W, lib, rng, depth, maxdepth)
        return None if r is None else (('l', r[0]), r[1])
    cands = []
    for i, vt in enumerate(env):
        args, ret = arrows(resolve(vt, s)); s2 = unify(ret, req, s)
        if s2 is not None: cands.append((('v', i), args, s2, W[-1] + 1.0))
    for pi in range(len(lib)):
        args, ret = arrows(instantiate(lib.types[pi])); s2 = unify(ret, req, s)
        if s2 is not None and (depth < maxdepth or not args): cands.append((('p', pi), args, s2, W[pi]))
    if not cands: return None
    lg = np.array([c[3] for c in cands]); p = np.exp(lg - lg.max()); p /= p.sum()
    leaf, args, s2, _ = cands[rng.choice(len(cands), p=p)]
    expr = leaf
    for a in args:
        r = sample(a, env, s2, W, lib, rng, depth + 1, maxdepth)
        if r is None: return None
        expr = ('a', expr, r[0]); s2 = r[1]
    return expr, s2

def rand_input(t, rng):
    if t == INT: return int(rng.integers(0, 10))
    if t == LIST(INT): return tuple(int(x) for x in rng.integers(0, 10, size=int(rng.integers(0, 8))))
    raise ValueError

def closed_bloat(t, lib, d=0):
    """변수 없이 상수만 계산하는 덩어리(재료 3개 이상)가 있으면 True - 크기만 부풀린 가짜 난이도"""
    for st, dd in subterms(t):
        if st[0] == 'a' and min_free(st) >= 10 ** 6 and leaves(st) >= 3 and not _has_lambda(st):
            return True
    return False

def _has_lambda(t):
    if t[0] == 'l': return True
    if t[0] == 'a': return _has_lambda(t[1]) or _has_lambda(t[2])
    return False

def make_task(lib, W, rng, maxdepth, min_base=1, max_base=99):
    for _ in range(200):
        req = REQUESTS[rng.integers(len(REQUESTS))]
        r = sample(req, (), {}, W, lib, rng, 0, maxdepth)
        if r is None: continue
        prog = r[0]; bs = base_size(prog, lib)
        if not (min_base <= bs <= max_base) or closed_bloat(prog, lib): continue
        args, ret = arrows(req)
        try:
            f = compile_term(prog, lib)(())
            ex = []
            for _ in range(6):
                inp = tuple(rand_input(a, rng) for a in args); ex.append((inp, _call(f, inp)))
        except Exception:
            continue
        outs = [o for _, o in ex]
        if (len(set(outs)) < 2) if isinstance(outs[0], bool) else (len(set(outs)) < 3): continue   # 거의 상수 함수 제외
        if any(all(i[k] == o for i, o in ex) for k in range(len(args))): continue  # 입력 하나를 그대로 돌려주는 함수 제외
        if any(isinstance(o, tuple) and len(o) > 15 for o in outs): continue
        return dict(type=req, ex=ex, src=prog, base=bs)
    return None

# ==============================================================================
# 7. 잠자기: 개념 발명 (반통일화 + 압축 이득)
# ==============================================================================
def shift(t, d, cut=0):
    k = t[0]
    if k == 'v': return ('v', t[1] + d) if t[1] >= cut else t
    if k == 'l': return ('l', shift(t[1], d, cut + 1))
    if k == 'a': return ('a', shift(t[1], d, cut), shift(t[2], d, cut))
    return t

def min_free(t, cut=0):
    """t 안에서 cut 밖을 가리키는 변수 중 가장 작은 (상대) 번호, 없으면 큰 수"""
    k = t[0]
    if k == 'v': return t[1] - cut if t[1] >= cut else 10 ** 6
    if k == 'l': return min_free(t[1], cut + 1)
    if k == 'a': return min(min_free(t[1], cut), min_free(t[2], cut))
    return 10 ** 6

def subterms(t, d=0):
    yield t, d
    if t[0] == 'l': yield from subterms(t[1], d + 1)
    elif t[0] == 'a': yield from subterms(t[1], d); yield from subterms(t[2], d)

def anti_unify(a, b):
    holes = {}
    def go(x, y, d):
        if x[0] == y[0] == 'a': return ('a', go(x[1], y[1], d), go(x[2], y[2], d))
        if x[0] == y[0] == 'l': return ('l', go(x[1], y[1], d + 1))
        if x == y:
            if x[0] != 'v' or x[1] < d: return x       # 재료/개념, 또는 틀 안쪽에 묶인 변수 -> 그대로
            # 틀 바깥을 가리키는 변수 -> 구멍으로
        elif min_free(x) < d or min_free(y) < d:
            return None                                 # 틀 안쪽 변수를 구멍에 넣을 수 없음
        key = (x, y)
        if key not in holes: holes[key] = ('h', len(holes))
        return holes[key]
    def chk(t):
        if t is None: return False
        if t[0] == 'a': return chk(t[1]) and chk(t[2])
        if t[0] == 'l': return chk(t[1])
        return True
    r = go(a, b, 0)
    return (r, len(holes)) if chk(r) else (None, 0)

def fmatch(f, t, d, binds):
    k = f[0]
    if k == 'h':
        if min_free(t) < d: return False
        v = shift(t, -d)
        if f[1] in binds: return binds[f[1]] == v
        binds[f[1]] = v; return True
    if k != t[0]: return False
    if k == 'a': return fmatch(f[1], t[1], d, binds) and fmatch(f[2], t[2], d, binds)
    if k == 'l': return fmatch(f[1], t[1], d + 1, binds)
    return f == t

def rewrite(t, f, inv, nh):
    """t 안의 틀 f 출현을 (inv 구멍들...) 로 치환. -> (새 항, 치환 횟수)"""
    b = {}
    if fmatch(f, t, 0, b) and len(b) == nh:
        e = ('p', inv); n = 1
        for j in range(nh):
            sub, k = rewrite(b[j], f, inv, nh); n += k; e = ('a', e, sub)
        return e, n
    if t[0] == 'a':
        x, n1 = rewrite(t[1], f, inv, nh); y, n2 = rewrite(t[2], f, inv, nh); return ('a', x, y), n1 + n2
    if t[0] == 'l':
        x, n = rewrite(t[1], f, inv, nh); return ('l', x), n
    return t, 0

def holes_to_body(f, nh):
    def go(t, d):
        k = t[0]
        if k == 'h': return ('v', d + nh - 1 - t[1])
        if k == 'v': return ('v', t[1] + (nh if t[1] >= d else 0))
        if k == 'l': return ('l', go(t[1], d + 1))
        if k == 'a': return ('a', go(t[1], d), go(t[2], d))
        return t
    body = go(f, 0)
    for _ in range(nh): body = ('l', body)
    return body

def prim_leaves(f):
    if f[0] == 'a': return prim_leaves(f[1]) + prim_leaves(f[2])
    if f[0] == 'l': return prim_leaves(f[1])
    return 1 if f[0] == 'p' else 0

def frag_leaves(f):
    if f[0] == 'a': return frag_leaves(f[1]) + frag_leaves(f[2])
    if f[0] == 'l': return frag_leaves(f[1])
    return 0 if f[0] == 'h' else 1

def invent(lib, corpus, rng, gen):
    """corpus: {task_id: prog}. 가장 압축 이득이 큰 틀을 개념으로 등록. -> 로그 목록"""
    logs = []
    for _ in range(CFG['MAX_INVENT_PER_GEN']):
        if len(lib) >= CFG['MAX_PRODS']: break
        subs = []
        for tid, p in corpus.items():
            for st, d in subterms(p):
                if 3 <= leaves(st) <= 14 and st[0] in ('a', 'l'): subs.append((tid, st))
        if len(subs) < 4: break
        groups = collections.defaultdict(list)
        for tid, st in subs:
            sig = st[0] + (st[1][0] if st[0] == 'a' else 'l') + str(leaves(st) // 3)
            groups[sig].append((tid, st))
        cands = set()
        for g in groups.values():
            g = g if len(g) <= 60 else rng.sample(g, 60)
            for (t1, a), (t2, b) in itertools.combinations(g, 2):
                if t1 == t2: continue
                f, nh = anti_unify(a, b)
                if f is not None and nh <= 3 and prim_leaves(f) >= 2: cands.add((f, nh))
        best, best_gain = None, 0.0
        before = sum(leaves(p) for p in corpus.values())
        for f, nh in cands:
            body = holes_to_body(f, nh)
            try:
                tt, s = infer(body, lib); tt = generalize(resolve(tt, s))
            except (TypeError, RecursionError):
                continue
            inv = len(lib)
            lib_tmp_names = lib.names
            used = 0; after = 0
            for tid, p in corpus.items():
                np_, n = rewrite(p, f, inv, nh); after += leaves(np_)
                used += (n > 0)
            if used < 3: continue
            gain = before - after - frag_leaves(f)
            if gain < max(4, 0.01 * before): continue
            if gain > best_gain: best, best_gain = (f, nh, body, tt), gain
        if best is None: break
        f, nh, body, tt = best
        depth = 1 + max([lib.depth[x[1]] for x, _ in subterms(body) if x[0] == 'p'] or [0])
        bl = base_size(body, lib)
        try:
            val = compile_term(body, lib)(())
        except Exception:
            break
        name = f"개념{len(lib) - lib.n_base + 1}"
        idx = lib.add(name, tt, val, body, depth, max(1, bl - nh))
        for tid in list(corpus): corpus[tid] = rewrite(corpus[tid], f, idx, nh)[0]
        after = sum(leaves(p) for p in corpus.values())
        logs.append(f"  💡 [개념 발명] {name} : {tshow(tt)}  =  {show(body, lib)}   (압축 {before}→{after}, 층 {depth})")
    return logs

# ==============================================================================
# 8. 인식망: 예시를 보고 어떤 재료/개념이 필요할지 감 잡기
# ==============================================================================
NF = 28
def stats(v):
    if isinstance(v, bool): return [0, 0, 1, float(v), 0, 0, 0, 0]
    if isinstance(v, int): return [1, 0, 0, 0, v / 10, 0, 0, 0]
    xs = list(v)
    if not xs: return [0, 1, 0, 0, 0, 0, 0, 0]
    a = np.array(xs, float)
    return [0, 1, 0, 0, a.mean() / 10, a.max() / 10, len(xs) / 10, float(all(x <= y for x, y in zip(xs, xs[1:])))]

def task_features(task):
    rows = []
    for inp, out in task['ex'][:4]:
        fi = stats(inp[0]); fo = stats(out)
        rel = [0.0] * 8
        if isinstance(out, tuple) and isinstance(inp[0], tuple):
            si, so = inp[0], out
            # 일반 통계만 (뒤집기/정렬 같은 '개념' 힌트는 넣지 않음)
            rel = [float(len(so) == len(si)), len(so) / (len(si) + 1),
                   float(collections.Counter(so) == collections.Counter(si)), float(set(so) <= set(si)),
                   float(len(so) < len(si)), float(len(so) > len(si)),
                   float(np.mean([a - b for a, b in zip(so, si)])) / 10 if len(so) == len(si) and so else 0.0, 0.0]
        elif isinstance(out, int) and isinstance(inp[0], tuple):
            rel = [0, 0, 0, float(out in inp[0]), 0, 0, 0, 0]
        rows.append(fi + fo + rel)
    v = np.mean(np.array(rows, float), 0)
    t = task['type']; args, ret = arrows(t)
    sig = [float(len(args) == 2), float(ret == INT), float(ret == BOOL), float(ret == LIST(INT))]
    out = np.zeros(NF, np.float32); x = np.concatenate([v, sig])[:NF]; out[:len(x)] = x
    return out

class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(NF, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.head = nn.Linear(128, CFG['MAX_PRODS'] + 1)   # 마지막 = 변수
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self, x): return self.head(self.f(x))

def weights_for(net, task, n):
    with torch.no_grad():
        lg = net(torch.from_numpy(task_features(task))[None])[0].numpy()
    W = np.concatenate([lg[:n], lg[-1:]])
    return W

def prod_counts(p, n):
    c = np.zeros(CFG['MAX_PRODS'] + 1, np.float32)
    for st, _ in subterms(p):
        if st[0] == 'p': c[st[1]] += 1
        elif st[0] == 'v': c[-1] += 1
    return c

def train_recognizer(net, data, n, iters=300, bs=64):
    if len(data) < 16: return 0.0
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); net.train(); tl = 0.0
    mask = torch.full((CFG['MAX_PRODS'] + 1,), -1e9); mask[:n] = 0; mask[-1] = 0
    for _ in range(iters):
        b = random.sample(data, min(bs, len(data)))
        x = torch.from_numpy(np.stack([f for f, _ in b])); y = torch.from_numpy(np.stack([c for _, c in b]))
        y = y / y.sum(1, keepdim=True).clamp(min=1)
        lp = torch.log_softmax(net(x) + mask, -1)
        loss = -(y * lp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss)
    net.eval(); return tl / iters

# ==============================================================================
# 9. 병렬 (풀기 / 경계선 출제)
# ==============================================================================
_G = {}
def _weights(task):
    lib, net = _G['lib'], _G['net']
    return weights_for(net, task, len(lib)) if net is not None else np.zeros(len(lib) + 1)

def _job_solve(args):
    task, budget = args
    return search(task, _G['lib'], _weights(task), budget)

def _job_gen(seed):
    """경계선 출제: 지금 김코어가 짧은 탐색(QUICK)으로 바로 푸는 문제는 대부분 버림"""
    lib = _G['lib']; rng = np.random.default_rng(seed)
    W = np.zeros(len(lib) + 1); W[lib.n_base:len(lib)] = _G['concept_bias']
    for _ in range(30):
        tk = make_task(lib, W, rng, _G['maxdepth'])
        if tk is None: continue
        r = search(tk, lib, _weights(tk), CFG['QUICK'], time_cap=0.4)
        if r['prog'] is not None and rng.random() > CFG['EASY_KEEP']: continue
        tk['quick_solved'] = r['prog'] is not None
        return tk
    return None

def _pool(fn, items):
    with mp.get_context('fork').Pool(CFG['WORKERS']) as pool:
        return pool.map(fn, items, chunksize=1)

def solve_all(lib, net, tasks, budget):
    _G.update(lib=lib, net=net)
    return _pool(_job_solve, [(t, budget) for t in tasks])

def gen_frontier(lib, net, n, maxdepth, concept_bias, seed):
    _G.update(lib=lib, net=net, maxdepth=maxdepth, concept_bias=concept_bias)
    out = []; k = 0
    while len(out) < n:
        got = _pool(_job_gen, [seed * 1000003 + k + i for i in range(n - len(out))])
        k += n; out += [t for t in got if t is not None]
    return out[:n]

# ==============================================================================
# 10. 탐침 (고정 문제집, 난이도 = 0세대 김코어가 얼마나 고생했나)
# ==============================================================================
def probe_set(seed):
    lib0 = Library(); rng = np.random.default_rng(seed); W = np.zeros(len(lib0) + 1)
    out, seen = [], set()
    while len(out) < CFG['PROBE_N']:
        t = make_task(lib0, W, rng, maxdepth=int(rng.integers(3, 7)))
        if t is None: continue
        sig = (t['type'], tuple(o for _, o in t['ex']))
        if sig in seen: continue
        seen.add(sig); out.append(t)
    return out

LEVELS = ['쉬움', '고생', '0세대 실패']
def level_of(r0):
    if r0['prog'] is None or not r0['hidden_ok']: return '0세대 실패'
    return '쉬움' if r0['tried'] <= 100 else '고생'

def probe_eval(lib, net, probes, levels):
    rs = solve_all(lib, net, probes, CFG['SEARCH_BUDGET'])
    by = {l: [0, 0] for l in LEVELS}; over = 0; tried = {l: [] for l in LEVELS}
    for t, r, lv in zip(probes, rs, levels):
        by[lv][1] += 1
        if r['prog'] is not None:
            if r['hidden_ok']: by[lv][0] += 1; tried[lv].append(r['tried'])
            else: over += 1
    tot = sum(v[0] for v in by.values())
    return 100.0 * tot / len(probes), by, over, {l: (float(np.mean(v)) if v else 0.0) for l, v in tried.items()}, rs

def fmt_lv(by):
    return ' '.join(f"{l} {v[0]}/{v[1]}" for l, v in by.items())

# ==============================================================================
# 11. 메인
# ==============================================================================
def main():
    random.seed(CFG['SEED']); np.random.seed(CFG['SEED']); torch.manual_seed(CFG['SEED'])
    rng = np.random.default_rng(CFG['SEED'])
    lib = Library(); net = Recognizer()
    t = time.time()
    probes = probe_set(9999)
    log("=" * 104)
    log(" [김코어 v8.0] 복리 지능: 계산 재료만 주고, 개념은 스스로 발명")
    log(f"  기본 재료 {lib.n_base}개: {' '.join(lib.names)}")
    log(f"  탐침 {len(probes)}문제 (기본 재료로 만든 숨은 프로그램, 학습 금지) | 문제당 탐색 {CFG['SEARCH_BUDGET']}개 / {CFG['TIME_CAP']}초")
    p0, by0_raw, ov0, tr0, rs0 = probe_eval(Library(), None, probes, ['쉬움'] * len(probes))
    levels = [level_of(r) for r in rs0]
    by0 = {l: [sum(1 for lv, r in zip(levels, rs0) if lv == l and r['prog'] is not None and r['hidden_ok']),
               sum(1 for lv in levels if lv == l)] for l in LEVELS}
    log(f"  탐침 난이도 분류 (0세대 김코어 기준): " + ' '.join(f"{l} {by0[l][1]}개" for l in LEVELS) + f" | {time.time() - t:.0f}s")
    log("=" * 104)
    log(f"\n📊 [0세대 김코어 = 대조군] 탐침 {p0:.1f}% | {fmt_lv(by0)} | 과적합 {ov0}")
    metrics = dict(config=CFG, probe0=p0, probe0_by=by0, gens=[])
    corpus = {}; rec_data = collections.deque(maxlen=8000)
    maxdepth = 3; tid = 0; concept_bias = 1.5
    for gen in range(1, CFG['GENS'] + 1):
        tg = time.time()
        tasks = gen_frontier(lib, net, CFG['TASKS_PER_GEN'], maxdepth, concept_bias, gen)
        for tk in tasks: tk['id'] = tid; tid += 1
        rs = solve_all(lib, net, tasks, CFG['SEARCH_BUDGET'])
        solved = over = 0; sizes = []
        for tk, r in zip(tasks, rs):
            if r['prog'] is None: continue
            if not r['hidden_ok']: over += 1; continue
            solved += 1; corpus[tk['id']] = r['prog']; sizes.append(base_size(r['prog'], lib))
        rate = solved / len(tasks)
        ilogs = invent(lib, corpus, random.Random(gen), gen)
        dream = []
        Wd = np.zeros(len(lib) + 1); Wd[lib.n_base:len(lib)] = 1.0
        while len(dream) < CFG['DREAMS']:
            tk = make_task(lib, Wd, rng, maxdepth=4)
            if tk is not None: dream.append((task_features(tk), prod_counts(tk['src'], len(lib))))
        for tk in tasks:
            if tk['id'] in corpus: rec_data.append((task_features(tk), prod_counts(corpus[tk['id']], len(lib))))
        loss = train_recognizer(net, list(rec_data) + dream, len(lib))
        cb = sum(base_size(p, lib) for p in corpus.values()); cl_ = sum(leaves(p) for p in corpus.values())
        rec = dict(gen=gen, rate=rate, overfit=over, lib=len(lib) - lib.n_base, max_depth=max(lib.depth),
                   max_base_solved=max(sizes) if sizes else 0, mean_base_solved=float(np.mean(sizes)) if sizes else 0,
                   compression=cb / max(1, cl_), corpus=len(corpus), loss=loss, secs=time.time() - tg, gen_depth=maxdepth)
        line = (f"[제 {gen:02d} 세대 | 출제 깊이 {maxdepth}] 경계선 문제 {len(tasks)} → 해결 {solved} (과적합 {over}) | "
                f"푼 프로그램 기본재료 크기 평균 {rec['mean_base_solved']:.1f} 최대 {rec['max_base_solved']} | "
                f"개념 {rec['lib']}개 (최고 {rec['max_depth']}층) | 이해도 ×{rec['compression']:.2f} | {rec['secs']:.0f}s")
        for l in ilogs: line += "\n" + l
        if gen % 4 == 0 or gen == CFG['GENS']:
            pg, byg, ovg, trg, _ = probe_eval(lib, net, probes, levels)
            rec.update(probe=pg, probe_by=byg, probe_over=ovg)
            line += (f"\n   🔒 탐침 (고정, 학습 금지): 0세대 {p0:.1f}% → 지금 {pg:.1f}% | {fmt_lv(byg)} | 과적합 {ovg}"
                     f" | 고생 문제 평균 탐색 {trg['고생']:.0f}개")
        log(line)
        metrics['gens'].append(rec)
        if rate >= 0.5 and maxdepth < 7: maxdepth += 1
        elif rate < 0.2 and maxdepth > 3: maxdepth -= 1
        with open(os.path.join(OUT, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)

    log("\n" + "=" * 104)
    final_tasks = gen_frontier(lib, net, 100, 6, 2.0, 31337)
    final_tasks = [dict(type=t['type'], ex=t['ex'], base=t['base']) for t in final_tasks]
    r0 = solve_all(Library(), None, final_tasks, CFG['SEARCH_BUDGET'])
    rf = solve_all(lib, net, final_tasks, CFG['SEARCH_BUDGET'])
    s0 = sum(r['prog'] is not None and r['hidden_ok'] for r in r0); sf = sum(r['prog'] is not None and r['hidden_ok'] for r in rf)
    log(f" 🥊 [과거의 나 vs 지금의 나] 최종 김코어가 낸 경계선 문제 100개: 0세대 {s0}개  vs  최종 {sf}개")
    log("\n 🧠 [발명된 개념 전체]")
    for i in range(lib.n_base, len(lib)):
        log(f"   {lib.names[i]} ({lib.depth[i]}층, 기본재료 {lib.base_leaves[i]}개) : {tshow(lib.types[i])} = {show(lib.bodies[i], lib)}")
    pg, byg, ovg, trg, _ = probe_eval(lib, net, probes, levels)
    log("\n" + "=" * 104)
    log(" 🏆 [김코어 v8 성장 기록]")
    log(f"   탐침 (고정, 학습 금지): 0세대 {p0:.1f}% → 최종 {pg:.1f}%   [{fmt_lv(by0)}]  →  [{fmt_lv(byg)}]")
    log(f"   '0세대가 못 푼 문제' 중 최종이 푼 것: {byg['0세대 실패'][0]}/{byg['0세대 실패'][1]}")
    log(f"   과거의 나 vs 지금의 나: {s0} vs {sf}")
    log(f"   개념 {len(lib) - lib.n_base}개, 최고 {max(lib.depth)}층 | 총 {time.time() - _T0:.0f}s")
    for g in metrics['gens']:
        extra = f" | 탐침 {g['probe']:.1f}% (0세대 실패 중 {g['probe_by']['0세대 실패'][0]})" if 'probe' in g else ''
        log(f"   Gen {g['gen']:02d} | 해결 {g['rate'] * 100:5.1f}% | 푼 크기 평균 {g['mean_base_solved']:5.1f} 최대 {g['max_base_solved']:3d} | "
            f"개념 {g['lib']:2d} ({g['max_depth']}층) | 이해도 ×{g['compression']:.2f}{extra}")
    metrics['final'] = dict(probe=pg, probe0=p0, probe_by=byg, past_vs_now=(s0, sf), concepts=len(lib) - lib.n_base, max_depth=max(lib.depth))
    with open(os.path.join(OUT, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)

if __name__ == '__main__':
    main()
