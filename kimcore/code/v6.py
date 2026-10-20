"""
김코어 v6.0 - 무한 세계 메타 추론 엔진 ("세계를 외우지 말고, 추론을 배워라")

v5까지: 규칙 20개짜리 세계 '하나'에서만 연습 -> 그 세계를 외울 수 있음.
v6:     매 문제마다 다른 '세계'가 나옴. 세계 = 무작위 연산표로 만든 작은 우주(유한 대수)에서
        실제로 참인 법칙만 뽑아낸 규칙 체계. 연산 이름도 매번 무작위. 김코어는 도메인을 모름.

  세계 생성기 (Theory Exploration)
    1) 우주 만들기: 원소 2~6개, 이항 연산 1~2개, 단항 연산 0~1개, 상수 1~2개 (연산표)
    2) 법칙 찾기: 일반 모양(항등원, 흡수, 멱등, 대합, 역원, 교환, 결합, 분배, 반준동형...)에
       세계의 연산/상수를 전부 대입해 보고, 모든 원소 조합에서 참인 것만 규칙으로 채택
    3) 조건부 법칙(센서): 술어 P(원소 부분집합)를 만족할 때만 참인 법칙 + P의 닫힘 성질(사실 규칙)
    4) 뻥튀기: 축소 법칙의 역방향 (경로당 최대 2회)
  -> 규칙이 무작위여도 '어떤 우주에서 실제로 참'이므로 절대 모순이 없음.

  학습 세계 계열: 무작위 연산표, 사슬 격자(min/max), 아핀 대수 (n x + m y + c mod k)
  시험 세계 계열 (학습 때 한 번도 안 나옴, 김코어는 이름도 모름):
    - 불 대수 4원소 (사람들이 '논리/집합'이라 부르는 것)
    - 5원소 체 Z5 (사람들이 '산수/나눗셈'이라 부르는 것, 0이 아닐 때만 역원 -> 센서 필수)
    - 3차 대칭군 S3 (사람들이 '대칭/회전'이라 부르는 것, 비가환)
    - 처음 보는 무작위 세계들

  공정성/정직성 장치
    - 대조군: 같은 v6 엔진, 학습 0회, 와플 0개
    - 함정 문제: 그 우주에서 실제로 값이 '다른' 두 식 -> 정당한 규칙으로는 원리상 도달 불가.
      풀리면 BREACH (엔진/규칙이 틀렸다는 수학적 증거)
    - 모든 출제 문제는 우주에서 값이 같은지(시작=목표) 직접 대입 검증 후 출제
    - 롤백 판단은 별도 검증 세계로만. 시험 세계는 학습/모델선택에 절대 안 씀
    - 새 세계 적응: 시험 세계에서 '스스로 낸 문제'만 풀며 와플 주조 후 재시험 (가중치 고정)
"""
import os, sys, time, json, math, random, copy, heapq, itertools, collections
from functools import lru_cache
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

CFG = dict(
    SEED=int(os.environ.get('V6_SEED', 1234)),
    GENS=int(os.environ.get('V6_GENS', 24)),
    WORLDS_TRAIN=int(os.environ.get('V6_WORLDS', 160)),
    WORLDS_PER_GEN=12, PROBS_PER_WORLD=4,
    TRAIN_BUDGET=60, EVAL_BUDGET=35,
    MAX_INFL=2, INFL_SUB_MAX=6, MAX_SIZE_SLACK=8,
    EVAL_EVERY=4, WORKERS=int(os.environ.get('V6_WORKERS', 4)),
    MAX_WAFFLES=40, ADAPT_PROBS=48, D=64,
)
OUT_DIR = os.environ.get('V6_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), f"v6_out_s{CFG['SEED']}"))
os.makedirs(OUT_DIR, exist_ok=True)
_T0 = time.time()
_LOG = open(os.path.join(OUT_DIR, 'run.txt'), 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); _LOG.write(s + '\n'); _LOG.flush()

# ==============================================================================
# 1. 불변 튜플 그래프
# ==============================================================================
def L(s): return (s, ())
def N(s, *ch): return (s, tuple(ch))
def is_var(g): return g[0][0] == '?'

@lru_cache(maxsize=400000)
def size(g): return 1 + sum(size(c) for c in g[1])

def show(g): return g[0] if not g[1] else f"{g[0]}({','.join(show(c) for c in g[1])})"

def positions(g, prefix=()):
    yield prefix, g
    for i, c in enumerate(g[1]):
        yield from positions(c, prefix + (i,))

def get_at(g, pos):
    for i in pos: g = g[1][i]
    return g

def replace_at(g, pos, new):
    if not pos: return new
    ch = list(g[1]); ch[pos[0]] = replace_at(ch[pos[0]], pos[1:], new)
    return (g[0], tuple(ch))

def match(p, t, b):
    if p[0][0] == '?':
        v = b.get(p[0])
        if v is None:
            b[p[0]] = t; return True
        return v == t
    if p[0] != t[0] or len(p[1]) != len(t[1]): return False
    for pc, tc in zip(p[1], t[1]):
        if not match(pc, tc, b): return False
    return True

def subst(t, b):
    if t[0][0] == '?': return b[t[0]]
    if not t[1]: return t
    return (t[0], tuple(subst(c, b) for c in t[1]))

def vars_of(g, acc=None):
    acc = set() if acc is None else acc
    if is_var(g): acc.add(g[0])
    for c in g[1]: vars_of(c, acc)
    return acc

def leaf_syms(g, acc=None):
    acc = [] if acc is None else acc
    if not g[1] and g[0].startswith('s_') and g[0] not in acc: acc.append(g[0])
    for c in g[1]: leaf_syms(c, acc)
    return acc

X0, X1, X2, XF = L('?0'), L('?1'), L('?2'), L('?f')
LEAVES = ['s_a', 's_b', 's_c']

# ==============================================================================
# 2. 세계 = 작은 우주(연산표) + 거기서 참인 법칙들
# ==============================================================================
class Rule:
    __slots__ = ('name', 'lhs', 'rhs', 'kind', 'cond', 'free', 'uses', 'born')
    def __init__(self, name, lhs, rhs, kind, cond=(), born=0):
        self.name, self.lhs, self.rhs, self.kind, self.cond = name, lhs, rhs, kind, tuple(cond)
        cv = set().union(*[vars_of(c) for c in cond]) if cond else set()
        self.free = tuple(sorted((vars_of(rhs) | cv) - vars_of(lhs)))
        self.uses, self.born = 0, born

KIND_ID = {'shrink': 0, 'neutral': 1, 'cond': 2, 'infl': 3, 'waffle': 4}
_SYL = 'bcdfghjklmnprstvwz'

class World:
    def __init__(self, wid, family, n, ops, consts, pred, rng):
        self.wid, self.family, self.n = wid, family, n
        self.ops = ops          # sym -> (arity, table)  table: dict tuple->val
        self.consts = consts    # sym -> val
        self.pred = pred        # frozenset of elements or None (센서 술어 P)
        self.psym = 'P_' + ''.join(rng.choice(_SYL) for _ in range(3))
        self.rules, self.fact_rules = [], []
        self.n_base = 0
        order = list(ops) ; rng.shuffle(order)
        order.sort(key=lambda s: ops[s][0])
        corder = list(consts); rng.shuffle(corder)
        self.sig = {s: ('OP', i, ops[s][0]) for i, s in enumerate(order)}
        self.sig.update({c: ('CONST', i, 0) for i, c in enumerate(corder)})
        self.sig[self.psym] = ('PRED', 0, 1)

    # ---- 의미 평가 (함정 검증, 문제 검증용)
    def ev(self, g, asg):
        s = g[0]
        if s in self.consts: return self.consts[s]
        if s in self.ops:
            ar, tb = self.ops[s]
            return tb[tuple(self.ev(c, asg) for c in g[1])]
        return asg[s]

    def assignments(self, leaves, hyps):
        need = set(h[1][0][0] for h in hyps)
        for vals in itertools.product(range(self.n), repeat=len(leaves)):
            asg = dict(zip(leaves, vals))
            if self.pred is not None and any(asg[l] not in self.pred for l in need if l in asg):
                continue
            yield asg

    def sem_equal(self, a, b, hyps):
        lv = sorted(set(leaf_syms(a)) | set(leaf_syms(b)))
        return all(self.ev(a, asg) == self.ev(b, asg) for asg in self.assignments(lv, hyps))

    @property
    def waffles(self): return self.rules[self.n_base:]

# ---- 법칙 모양 (도메인 지식 아님: 연산 기호가 비어 있는 '모양'들)
def law_templates(world):
    B = [s for s, (a, _) in world.ops.items() if a == 2]
    U = [s for s, (a, _) in world.ops.items() if a == 1]
    C = list(world.consts)
    T = []
    for f in B:
        for c in C:
            T += [(N(f, X0, L(c)), X0), (N(f, L(c), X0), X0), (N(f, X0, L(c)), L(c)), (N(f, L(c), X0), L(c))]
            for g in U:
                T += [(N(f, X0, N(g, X0)), L(c)), (N(f, N(g, X0), X0), L(c))]
        T += [(N(f, X0, X0), X0)]
        T += [(N(f, X0, X0), L(c)) for c in C]
        T += [(N(f, X0, X1), N(f, X1, X0))]
        T += [(N(f, N(f, X0, X1), X2), N(f, X0, N(f, X1, X2)))]
        for h in B:
            if h == f: continue
            T += [(N(h, N(f, X0, X1), N(f, X0, X2)), N(f, X0, N(h, X1, X2))),     # 묶기 (분배 역)
                  (N(h, N(f, X1, X0), N(f, X2, X0)), N(f, N(h, X1, X2), X0)),
                  (N(f, X0, N(h, X0, X1)), X0)]                                      # 흡수
            for g in U:
                T += [(N(h, N(g, X0), N(g, X1)), N(g, N(f, X0, X1)))]              # 위상 침투
        for g in U:
            T += [(N(f, N(g, X1), N(g, X0)), N(g, N(f, X0, X1)))]                  # 반준동형 (군 역원)
    for g in U:
        T += [(N(g, N(g, X0)), X0)]
        for c in C:
            for d in C:
                T += [(N(g, L(c)), L(d))]
    return T

def holds(world, lhs, rhs, condvars=()):
    vs = sorted(vars_of(lhs) | vars_of(rhs))
    for vals in itertools.product(range(world.n), repeat=len(vs)):
        asg = dict(zip(vs, vals))
        if condvars and any(asg[v] not in world.pred for v in condvars): continue
        def ev(g):
            s = g[0]
            if s[0] == '?': return asg[s]
            if s in world.consts: return world.consts[s]
            ar, tb = world.ops[s]; return tb[tuple(ev(c) for c in g[1])]
        if ev(lhs) != ev(rhs): return False
    return True

def discover_laws(world, rng, max_shrink=14, max_neutral=6, max_cond=4, max_infl=8):
    shrink, neutral, cond = [], [], []
    seen = set()
    for lhs, rhs in law_templates(world):
        if lhs == rhs or (lhs, rhs) in seen: continue
        seen.add((lhs, rhs))
        ok = holds(world, lhs, rhs)
        if ok:
            if size(lhs) > size(rhs): shrink.append((lhs, rhs))
            elif size(lhs) == size(rhs): neutral.append((lhs, rhs))
        elif world.pred is not None and '?0' in vars_of(lhs) and size(lhs) > size(rhs) and holds(world, lhs, rhs, condvars=('?0',)):
            cond.append((lhs, rhs))
    rng.shuffle(shrink); rng.shuffle(neutral); rng.shuffle(cond)
    shrink, neutral, cond = shrink[:max_shrink], neutral[:max_neutral], cond[:max_cond]
    R = []
    for i, (l, r) in enumerate(shrink): R.append(Rule(f"법칙{i}[축소]", l, r, 'shrink'))
    for i, (l, r) in enumerate(neutral):
        R.append(Rule(f"법칙N{i}[→]", l, r, 'neutral'))
        if (r, l) not in [(x.lhs, x.rhs) for x in R] and vars_of(l) == vars_of(r):
            R.append(Rule(f"법칙N{i}[←]", r, l, 'neutral'))
    P = world.psym
    for i, (l, r) in enumerate(cond):
        R.append(Rule(f"센서{i}[|P]", l, r, 'cond', cond=[N(P, X0)]))
    # 뻥튀기 = 축소/센서 법칙의 역방향 (사라지는 부분은 잎 기호로 채움)
    infl_src = [x for x in R if x.kind in ('shrink', 'cond')]
    rng.shuffle(infl_src)
    k = 0
    for x in infl_src:
        if k >= max_infl: break
        lost = vars_of(x.lhs) - vars_of(x.rhs)
        if len(lost) > 1: continue
        lhs2 = x.rhs; rhs2 = x.lhs
        if lost:
            v = next(iter(lost)); rhs2 = subst(rhs2, {v: XF, **{u: L(u) for u in vars_of(rhs2) if u != v}})
        condf = [subst(c, {'?0': (XF if '?0' in lost else X0)}) for c in x.cond]
        R.append(Rule(f"뻥튀기{k}[{x.name}역]", lhs2, rhs2, 'infl', cond=condf)); k += 1
    world.rules = R; world.n_base = len(R)
    # 센서의 '작은 산' 증명용 사실 규칙: P의 닫힘 성질
    fr = []
    if world.pred is not None:
        for c, v in world.consts.items():
            if v in world.pred: fr.append((N(P, L(c)), []))
        for s, (ar, tb) in world.ops.items():
            if all(tb[args] in world.pred for args in itertools.product(sorted(world.pred), repeat=ar)):
                args = [X0, X1][:ar]
                fr.append((N(P, N(s, *args)), [N(P, a) for a in args]))
    world.fact_rules = fr
    return len([x for x in R if x.kind == 'shrink'])

# ---- 우주(연산표) 계열
def rand_name(rng, used):
    while True:
        s = rng.choice('fghkmpqrtvw') + ''.join(rng.choice(_SYL) for _ in range(2))
        if s not in used: used.add(s); return s

def table2(n, fn): return {(a, b): fn(a, b) % n for a in range(n) for b in range(n)}
def table1(n, fn): return {(a,): fn(a) % n for a in range(n)}

def make_world(wid, family, rng):
    used = set()
    ops, consts, pred = {}, {}, None
    if family == 'rand':
        n = rng.choice([2, 3])
        for _ in range(rng.choice([1, 2])):
            tb = {(a, b): rng.randrange(n) for a in range(n) for b in range(n)}
            ops[rand_name(rng, used)] = (2, tb)
        if rng.random() < 0.6:
            ops[rand_name(rng, used)] = (1, {(a,): rng.randrange(n) for a in range(n)})
        for v in rng.sample(range(n), rng.choice([1, 2])): consts[rand_name(rng, used)] = v
        if rng.random() < 0.5: pred = frozenset(rng.sample(range(n), rng.randint(1, n - 1)))
    elif family == 'chain':
        n = rng.choice([3, 4, 5])
        ops[rand_name(rng, used)] = (2, table2(n, min)); ops[rand_name(rng, used)] = (2, table2(n, max))
        consts[rand_name(rng, used)] = 0; consts[rand_name(rng, used)] = n - 1
        if rng.random() < 0.4: pred = frozenset(range(1, n))
    elif family == 'affine':
        n = rng.choice([4, 6, 8])
        for _ in range(rng.choice([1, 2])):
            a, b, c = rng.randrange(n), rng.randrange(n), rng.randrange(n)
            ops[rand_name(rng, used)] = (2, table2(n, lambda x, y, a=a, b=b, c=c: a * x + b * y + c))
        if rng.random() < 0.7:
            d, e = rng.randrange(n), rng.randrange(n)
            ops[rand_name(rng, used)] = (1, table1(n, lambda x, d=d, e=e: d * x + e))
        for v in rng.sample(range(n), rng.choice([1, 2])): consts[rand_name(rng, used)] = v
        if rng.random() < 0.4: pred = frozenset(x for x in range(n) if x % 2 == 1)
    elif family == 'bool4':      # 시험 전용: 불 대수 (비트 2개)
        n = 4
        ops[rand_name(rng, used)] = (2, table2(n, lambda x, y: x & y)); ops[rand_name(rng, used)] = (2, table2(n, lambda x, y: x | y))
        ops[rand_name(rng, used)] = (1, table1(n, lambda x: 3 - x))
        consts[rand_name(rng, used)] = 0; consts[rand_name(rng, used)] = 3
    elif family == 'z5field':    # 시험 전용: 5원소 체 (0이 아닐 때만 역원)
        n = 5; inv = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
        ops[rand_name(rng, used)] = (2, table2(n, lambda x, y: x + y)); ops[rand_name(rng, used)] = (2, table2(n, lambda x, y: x * y))
        ops[rand_name(rng, used)] = (1, table1(n, lambda x: inv[x]))
        consts[rand_name(rng, used)] = 0; consts[rand_name(rng, used)] = 1
        pred = frozenset(range(1, 5))
    elif family == 's3group':    # 시험 전용: 3차 대칭군 (비가환)
        perms = list(itertools.permutations(range(3))); n = 6
        idx = {p: i for i, p in enumerate(perms)}
        mul = lambda a, b: idx[tuple(perms[a][perms[b][k]] for k in range(3))]
        inv = lambda a: idx[tuple(sorted(range(3), key=lambda k: perms[a][k]))]
        ops[rand_name(rng, used)] = (2, {(a, b): mul(a, b) for a in range(n) for b in range(n)})
        ops[rand_name(rng, used)] = (1, {(a,): inv(a) for a in range(n)})
        consts[rand_name(rng, used)] = idx[(0, 1, 2)]
    w = World(wid, family, n, ops, consts, pred, rng)
    ns = discover_laws(w, rng)
    return w if ns >= 4 else None

def gen_worlds(families, k, seed, start_id):
    rng = random.Random(seed); out = []
    while len(out) < k:
        w = make_world(start_id + len(out), rng.choice(families), rng)
        if w is not None: out.append(w)
    return out

# ==============================================================================
# 3. 센서의 작은 산: 사실 증명기 (세계별 Horn 역방향 연쇄)
# ==============================================================================
_FACTS = {}
def prove(world, fact, hyps, depth=4):
    key = (world.wid, fact, hyps, depth)
    if key in _FACTS: return _FACTS[key]
    ok = fact in hyps
    if not ok and depth > 0:
        for head, body in world.fact_rules:
            b = {}
            if match(head, fact, b) and all(prove(world, subst(x, b), hyps, depth - 1) for x in body):
                ok = True; break
    _FACTS[key] = ok
    return ok

# ==============================================================================
# 4. 엔진: (규칙, 위치, 바인딩) 단위의 합법 수 나열
# ==============================================================================
def moves(world, g, goal, hyps, n_infl, allow_infl=True, use_waffles=True, check_cond=True):
    out, seen = [], set()
    gl = leaf_syms(goal)
    pool = (gl + [s for s in leaf_syms(g) if s not in gl])[:4] or LEAVES[:1]
    nr = len(world.rules) if use_waffles else world.n_base
    for nidx, (pos, sub) in enumerate(positions(g)):
        ssz = size(sub)
        for ri in range(nr):
            r = world.rules[ri]
            if r.kind == 'infl':
                if not allow_infl or n_infl >= CFG['MAX_INFL']: continue
                if is_var(r.lhs) and ssz > CFG['INFL_SUB_MAX']: continue
            b = {}
            if not match(r.lhs, sub, b): continue
            for bs in ([None] if not r.free else pool):
                bb = dict(b)
                if bs is not None: bb[r.free[0]] = L(bs)
                if r.cond and check_cond and not all(prove(world, subst(c, bb), hyps) for c in r.cond): continue
                child = replace_at(g, pos, subst(r.rhs, bb))
                if child == g or child in seen: continue
                seen.add(child); out.append((ri, nidx, bs, child))
    return out

# ==============================================================================
# 5. GNN 눈 (세계 무관 토큰: 연산 번호 + 항수, 상수 번호, 잎 익명, 패턴 변수)
# ==============================================================================
MAXOP, MAXC, NLEAF, NPVAR = 6, 4, 8, 8
T_OP0 = 0; T_C0 = T_OP0 + MAXOP; T_LEAF0 = T_C0 + MAXC; T_PVAR0 = T_LEAF0 + NLEAF; T_PRED = T_PVAR0 + NPVAR
T_UNK = T_PRED + 1; NTOK = T_UNK + 1
ROLE_STATE, ROLE_GOAL, ROLE_LHS, ROLE_RHS = 0, 1, 2, 3

def anon_map(goal, g):
    m = {}
    for s in leaf_syms(goal) + leaf_syms(g):
        if s not in m: m[s] = len(m)
    return m

def tok_of(sym, am, sig):
    t = sig.get(sym)
    if t is not None:
        kind, i, _ = t
        if kind == 'OP': return T_OP0 + min(i, MAXOP - 1)
        if kind == 'CONST': return T_C0 + min(i, MAXC - 1)
        return T_PRED
    if sym[0] == '?':
        return T_PVAR0 + (NPVAR - 1 if sym == '?f' else min(int(sym[1:]), NPVAR - 2))
    k = am.get(sym)
    return T_LEAF0 + min(k, NLEAF - 1) if k is not None else T_UNK

@lru_cache(maxsize=200000)
def subtrees(g): return frozenset(sub for _, sub in positions(g) if sub[1])

class Batch:
    def __init__(self):
        self.tok, self.role, self.flag, self.parent, self.slot, self.ar = [], [], [], [], [], []
        self.graph_of, self.roots, self.n = [], [], 0
    def add(self, g, role, am, sig, goal_subs=None):
        gid = len(self.roots); self.roots.append(self.n)
        stack = [(g, -1, 0)]
        while stack:
            node, par, slot = stack.pop()
            me = self.n; self.n += 1
            self.tok.append(tok_of(node[0], am, sig)); self.role.append(role)
            self.flag.append(1 if (goal_subs is not None and node[1] and node in goal_subs) else 0)
            self.parent.append(par); self.slot.append(min(slot, 2)); self.ar.append(min(len(node[1]), 3))
            self.graph_of.append(gid)
            for i in range(len(node[1]) - 1, -1, -1): stack.append((node[1][i], me, i))
        return gid
    def tensors(self):
        T = lambda x: torch.tensor(x, dtype=torch.long)
        return T(self.tok), T(self.role), T(self.flag), T(self.ar), T(self.parent), T(self.slot), T(self.graph_of), T(self.roots)

class TreeGNN(nn.Module):
    def __init__(self, D, T=3):
        super().__init__()
        self.tok = nn.Embedding(NTOK, D); self.role = nn.Embedding(4, D); self.flag = nn.Embedding(2, D); self.ar = nn.Embedding(4, D)
        self.up = nn.ModuleList([nn.Linear(D, D) for _ in range(3)])
        self.down = nn.ModuleList([nn.Linear(D, D) for _ in range(3)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(3 * D, D), nn.GELU(), nn.Linear(D, D)) for _ in range(T)])
        self.norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(T)])
        self.read = nn.Linear(2 * D, D)
    def forward(self, tok, role, flag, ar, parent, slot, graph_of, roots):
        h = self.tok(tok) + self.role(role) + self.flag(flag) + self.ar(ar)
        ch = (parent >= 0).nonzero(as_tuple=True)[0]; par = parent[ch]; sl = slot[ch]
        groups = [(ch[sl == s], par[sl == s]) for s in range(3)]
        for upd, norm in zip(self.upd, self.norm):
            up = torch.zeros_like(h); down = torch.zeros_like(h)
            for s, (c, p) in enumerate(groups):
                if len(c):
                    up = up.index_add(0, p, self.up[s](h[c])); down = down.index_copy(0, c, self.down[s](h[p]))
            h = norm(h + upd(torch.cat([h, up, down], -1)))
        ng = len(roots)
        cnt = torch.zeros(ng).index_add(0, graph_of, torch.ones(len(graph_of))).clamp(min=1)
        mean = torch.zeros(ng, h.shape[1]).index_add(0, graph_of, h) / cnt[:, None]
        return h, self.read(torch.cat([h[roots], mean], -1))

class KimCore(nn.Module):
    def __init__(self, D=64):
        super().__init__()
        self.D = D; self.gnn = TreeGNN(D)
        self.rule_mlp = nn.Sequential(nn.Linear(2 * D + 6, D), nn.GELU(), nn.Linear(D, D))
        self.pol = nn.Sequential(nn.Linear(5 * D, 128), nn.GELU(), nn.Linear(128, 1))
        self.val = nn.Sequential(nn.Linear(3 * D + 2, 128), nn.GELU(), nn.Linear(128, 1))
        for head in (self.pol[-1], self.val[-1]):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def rule_embs(self, world):
        b = Batch()
        for r in world.rules:
            b.add(r.lhs, ROLE_LHS, {}, world.sig); b.add(r.rhs, ROLE_RHS, {}, world.sig)
        _, g = self.gnn(*b.tensors())
        g = g.view(len(world.rules), 2 * self.D)
        meta = torch.tensor([[float(KIND_ID[r.kind] == k) for k in range(5)] + [float(bool(r.cond))] for r in world.rules])
        return self.rule_mlp(torch.cat([g, meta], -1))

    def encode_pairs(self, triples):
        b = Batch(); sidx, gidx, offs = [], [], []
        for st, goal, sig in triples:
            am = anon_map(goal, st)
            offs.append(b.n); sidx.append(b.add(st, ROLE_STATE, am, sig, subtrees(goal)))
            gidx.append(b.add(goal, ROLE_GOAL, am, sig))
        h, g = self.gnn(*b.tensors())
        return h, g[sidx], g[gidx], offs

    def policy_logits(self, h, gs, gg, offs, items, rembs):
        """items: [(state_i, node_idx, rule_emb_row, bind_tok)]; rembs: 행 묶음 텐서"""
        si = torch.tensor([it[0] for it in items]); ni = torch.tensor([offs[it[0]] + it[1] for it in items])
        ri = torch.tensor([it[2] for it in items]); bt = torch.tensor([it[3] for it in items])
        bemb = self.gnn.tok(bt.clamp(min=0)) * (bt >= 0).float()[:, None]
        return self.pol(torch.cat([h[ni], gs[si], gg[si], rembs[ri], bemb], -1)).squeeze(-1)

    def value_of(self, states, goal, sig, n_infl):
        b = Batch()
        for c in states: b.add(c, ROLE_STATE, anon_map(goal, c), sig, subtrees(goal))
        _, gc = self.gnn(*b.tensors())
        b2 = Batch(); b2.add(goal, ROLE_GOAL, anon_map(goal, goal), sig)
        _, gg = self.gnn(*b2.tensors()); gg = gg.expand(len(states), -1)
        extra = torch.tensor([[(size(c) - size(goal)) / 10.0, n_infl / 2.0] for c in states])
        return torch.sigmoid(self.val(torch.cat([gc, gg, gc * gg, extra], -1)).squeeze(-1))

    def value_pairs(self, triples, n_infls):
        _, gs, gg, _ = self.encode_pairs(triples)
        extra = torch.tensor([[(size(s) - size(g)) / 10.0, ni / 2.0] for (s, g, _), ni in zip(triples, n_infls)])
        return torch.sigmoid(self.val(torch.cat([gs, gg, gs * gg, extra], -1)).squeeze(-1))

# ==============================================================================
# 6. 저격 다림질 탐색
# ==============================================================================
def bind_tok(bs, goal, g, sig):
    return -1 if bs is None else tok_of(bs, anon_map(goal, g), sig)

def search(world, net, remb, prob, budget, explore=False, allow_infl=True, use_waffles=True,
           check_cond=True, rng=None, record=True):
    start, goal, hyps = prob['start'], prob['goal'], prob['hyps']
    res = dict(ok=False, path=(), expansions=0, steps=[], neg=[])
    if start == goal:
        res['ok'] = True; return res
    max_size = size(start) + CFG['MAX_SIZE_SLACK']
    uid = 0; heap = [(0.0, 0, start, 0, 0.0, ())]; best_infl = {start: 0}; visited = []
    gsz = size(goal)
    while heap and res['expansions'] < budget:
        _, _, g, ni, logp, path = heapq.heappop(heap)
        res['expansions'] += 1
        mv = [m for m in moves(world, g, goal, hyps, ni, allow_infl, use_waffles, check_cond) if size(m[3]) <= max_size]
        if not mv: continue
        if record: visited.append((g, ni))
        for m in mv:
            if m[3] == goal:
                full = path + ((g, ni, m[0], m[1], m[2]),)
                res.update(ok=True, path=full)
                if record: res['steps'] = [(st, n_i, (ri, nidx, bs)) for st, n_i, ri, nidx, bs in full]
                return res
        with torch.no_grad():
            h, gs, gg, offs = net.encode_pairs([(g, goal, world.sig)])
            items = [(0, m[1], m[0], bind_tok(m[2], goal, g, world.sig)) for m in mv]
            lg = net.policy_logits(h, gs, gg, offs, items, remb)
            if explore: lg = lg + 0.7 * torch.from_numpy(rng.gumbel(size=len(mv))).float()
            lp = torch.log_softmax(lg, 0).numpy()
            v = net.value_of([m[3] for m in mv], goal, world.sig, ni).numpy()
        for k, m in enumerate(mv):
            child = m[3]; cni = ni + (1 if world.rules[m[0]].kind == 'infl' else 0)
            if child in best_infl and best_infl[child] <= cni: continue
            best_infl[child] = cni
            clogp = logp + 0.5 * float(lp[k])
            pri = clogp + 2.0 * math.log(max(float(v[k]), 1e-4)) - 0.35 * max(0, size(child) - gsz) - 0.2 * (len(path) + 1)
            uid += 1
            heapq.heappush(heap, (-pri, uid, child, cni, clogp, path + ((g, ni, m[0], m[1], m[2]),)))
    if record: res['neg'] = visited[:10]
    return res

# ==============================================================================
# 7. 출제기 (세계 무관): 목표에서 거꾸로 걷기 + 우주 대입 검증
# ==============================================================================
class Proposer:
    def __init__(self, world, seed):
        self.w = world; self.rng = random.Random(seed)
        R = world.rules[:world.n_base]
        self.infl = [r for r in R if r.kind == 'infl']
        self.shrink = [r for r in R if r.kind in ('shrink', 'cond')]
        self.neutral = [r for r in R if r.kind == 'neutral']
        # 풀 때 정방향 규칙으로 되돌릴 수 있는 '확장'만 중립으로 사용: 축소 규칙의 역방향 중 잃는 변수 없는 것
        self.expand = [Rule(r.name + '역', r.rhs, r.lhs, 'neutral') for r in R
                       if r.kind == 'shrink' and vars_of(r.lhs) == vars_of(r.rhs) and not is_var(r.rhs)]

    def rand_goal(self):
        r, w = self.rng, self.w
        ops = list(w.ops); consts = list(w.consts)
        def t(d):
            if d == 0 or r.random() < 0.35:
                return L(r.choice(consts)) if (consts and r.random() < 0.15) else L(r.choice(LEAVES))
            s = r.choice(ops); ar = w.ops[s][0]
            return N(s, *[t(d - 1) for _ in range(ar)])
        return t(2)

    def _apply(self, g, rules, hyps, goal=None, shrink_mode=False):
        cands = []
        pool = leaf_syms(g) or LEAVES[:1]
        gl = set(leaf_syms(goal)) if goal is not None else set()
        for pos, sub in positions(g):
            for r in rules:
                if r.kind == 'infl' and is_var(r.lhs) and size(sub) > CFG['INFL_SUB_MAX']: continue
                b = {}
                if not match(r.lhs, sub, b): continue
                if r.free:
                    for s in pool:
                        bb = dict(b); bb[r.free[0]] = L(s)
                        if all(prove(self.w, subst(c, bb), hyps) for c in r.cond): cands.append((r, pos, bb))
                    continue
                if r.cond and not all(prove(self.w, subst(c, b), hyps) for c in r.cond): continue
                if shrink_mode:     # 풀 때 뻥튀기로 '정확히' 복원 가능해야 함
                    lost = vars_of(r.lhs) - vars_of(r.rhs)
                    if any(b[v][1] or b[v][0] not in gl for v in lost): continue
                    if is_var(r.rhs) and size(subst(r.rhs, b)) > CFG['INFL_SUB_MAX']: continue
                    if not any(i.lhs == r.rhs and (i.rhs == r.lhs or True) for i in self.infl if i.name.startswith('뻥튀기') and r.name in i.name):
                        continue   # 이 축소의 역방향 뻥튀기가 세계에 없으면 안 됨
                cands.append((r, pos, b))
        if not cands: return None, None
        r, pos, bb = self.rng.choice(cands)
        return replace_at(g, pos, subst(r.rhs, bb)), r

    def make(self, kind, steps, hyps=None):
        r, w = self.rng, self.w
        for _ in range(40):
            goal = self.rand_goal()
            h = hyps
            if h is None:
                h = frozenset([N(w.psym, L(s)) for s in r.sample(LEAVES, r.randint(1, 2))]) if w.pred is not None else frozenset()
            g, shrinks, used = goal, 0, []
            force_at = r.randint(0, 1) if kind in ('infl', 'cond') else -1
            for st in range(steps):
                if st == force_at and kind == 'cond':
                    ng, rr = self._apply(g, [x for x in self.infl if x.cond], h)
                elif st == force_at and kind == 'infl':
                    ng, rr = self._apply(g, self.shrink, h, goal, shrink_mode=True)
                    if ng is not None: shrinks += 1
                else:
                    u = r.random()
                    if u < 0.55: ng, rr = self._apply(g, self.infl, h)
                    elif u < 0.85 or shrinks >= CFG['MAX_INFL']: ng, rr = self._apply(g, self.neutral + self.expand, h)
                    else:
                        ng, rr = self._apply(g, self.shrink, h, goal, shrink_mode=True)
                        if ng is not None: shrinks += 1
                if ng is not None and size(ng) <= 30: g = ng; used.append(rr)
            if g == goal or size(g) > 30: continue
            if kind == 'cond' and not any(x.kind == 'infl' and x.cond for x in used): continue
            if kind == 'infl' and shrinks == 0: continue
            assert w.sem_equal(g, goal, h), f"출제 버그: 우주에서 값이 다름 {show(g)} vs {show(goal)}"
            return dict(start=g, goal=goal, hyps=h, kind=kind, wid=w.wid)
        return None

    def decoy(self):
        """함정: 이 우주에서 실제로 값이 '다른' 두 식. 정당한 규칙으로는 원리상 도달 불가."""
        for _ in range(60):
            p = self.make('plain', self.rng.randint(3, 6))
            if p is None: continue
            goal = p['goal']; lv = leaf_syms(goal)
            pos_list = [pos for pos, sub in positions(goal) if not sub[1]]
            pos = self.rng.choice(pos_list)
            alt = self.rng.choice([s for s in LEAVES if s != get_at(goal, pos)[0]] or LEAVES)
            fake = replace_at(goal, pos, L(alt))
            if fake == p['start'] or self.w.sem_equal(p['start'], fake, p['hyps']): continue
            q = dict(p); q['goal'] = fake; q['kind'] = 'decoy'
            return q
        return None

    def sensor_decoy(self):
        """센서 함정: 조건 P가 있을 때만 같은 식. 가정을 지우면 우주에서 값이 달라짐 -> 센서 없이 풀면 거짓말"""
        if self.w.pred is None: return None
        for _ in range(60):
            p = self.make('cond', self.rng.randint(2, 4))
            if p is None: continue
            if self.w.sem_equal(p['start'], p['goal'], frozenset()): continue
            q = dict(p); q['hyps'] = frozenset(); q['kind'] = 'decoy'; q['sensor'] = True
            return q
        return None

    def train_problem(self, difficulty):
        u = self.rng.random()
        kind = 'cond' if (u < 0.3 and self.w.pred is not None) else ('infl' if u < 0.6 else 'plain')
        return self.make(kind, self.rng.randint(difficulty, difficulty + 2))

# ==============================================================================
# 8. 와플 (세계 안에서의 압축) + 재생 검증 + 우주 대입 검증
# ==============================================================================
def generalize_pair(a, b):
    m = {}
    def gen(g):
        if not g[1]:
            if g[0].startswith('s_'):
                if g[0] not in m: m[g[0]] = f'?{len(m)}'
                return L(m[g[0]])
            return g
        return (g[0], tuple(gen(c) for c in g[1]))
    return gen(a), gen(b), m

def node_pos(g, nidx):
    for i, (pos, _) in enumerate(positions(g)):
        if i == nidx: return pos
    return None

def replay_ok(world, lhs, rhs, rel, m):
    for tag in range(2):
        inst = {v: L(f's_{tag}{k}') for k, v in enumerate(sorted(vars_of(lhs)))}
        def fill(g):
            if is_var(g): return inst[g[0]]
            return (g[0], tuple(fill(c) for c in g[1]))
        g, target = fill(lhs), fill(rhs)
        for ri, relpos, bs in rel:
            r = world.rules[ri]
            if r.cond: return False
            b = {}
            try: sub = get_at(g, relpos)
            except (IndexError, TypeError): return False
            if not match(r.lhs, sub, b): return False
            if r.free:
                if bs is None or bs not in m: return False
                b[r.free[0]] = inst[m[bs]]
            g = replace_at(g, relpos, subst(r.rhs, b))
        if g != target: return False
    return holds(world, lhs, rhs)       # 이중 안전: 우주에서도 참이어야 등록

def forge_waffles(world, solved, gen, counter, min_count=2, max_add=4):
    logs, cands = [], {}
    for prob, path in solved:
        graphs = [st for st, _, _, _, _ in path]
        afters = graphs[1:] + [prob['goal']]
        seq = [(st, ri, node_pos(st, nidx), bs) for st, ni, ri, nidx, bs in path]
        for i in range(len(seq)):
            for j in range(i + 1, min(len(seq), i + 4)):
                seg = seq[i:j + 1]; poss = [p for _, _, p, _ in seg]
                pre = []
                for k in range(min(len(p) for p in poss)):
                    if all(p[k] == poss[0][k] for p in poss): pre.append(poss[0][k])
                    else: break
                pre = tuple(pre)
                try: before = get_at(seg[0][0], pre); after = get_at(afters[j], pre)
                except (IndexError, TypeError): continue
                if before == after or size(before) > 14: continue
                lhs, rhs, m = generalize_pair(before, after)
                if not vars_of(rhs) <= vars_of(lhs) or is_var(lhs) or size(rhs) >= size(lhs): continue
                key = (lhs, rhs)
                if key not in cands: cands[key] = ([(ri, p[len(pre):], bs) for _, ri, p, bs in seg], m)
                counter[key] += 1
    have = set((r.lhs, r.rhs) for r in world.rules); added = 0
    for key, cnt in sorted(counter.items(), key=lambda kv: -kv[1]):
        if added >= max_add or len(world.waffles) >= CFG['MAX_WAFFLES']: break
        if cnt < min_count or key in have or key not in cands: continue
        rel, m = cands[key]
        if not replay_ok(world, key[0], key[1], rel, m):
            logs.append(f"  ⛔ [와플 폐기] {show(key[0])} ⇒ {show(key[1])}"); continue
        world.rules.append(Rule(f"와플#{len(world.waffles) + 1}", key[0], key[1], 'waffle', born=gen))
        have.add(key); added += 1
        logs.append(f"  🧇 [와플] {show(key[0])} ⇒ {show(key[1])} (×{cnt})")
    return logs

# ==============================================================================
# 9. 병렬 실행
# ==============================================================================
_G = {}
def _job(args):
    wi, prob, budget, explore, flags, seed = args
    world = _G['worlds'][wi]; remb = _G['rembs'][wi]
    return search(world, _G['net'], remb, prob, budget, explore=explore, rng=np.random.default_rng(seed), **flags)

def run_parallel(worlds_by_id, net, jobs, budget, explore=False, flags=None, seed=0):
    """jobs: [(wid, prob)]"""
    net.eval()
    need = sorted(set(w for w, _ in jobs))
    with torch.no_grad():
        rembs = {w: net.rule_embs(worlds_by_id[w]) for w in need}
    _G.update(worlds=worlds_by_id, rembs=rembs, net=net)
    args = [(w, p, budget, explore, flags or {}, seed * 100003 + i) for i, (w, p) in enumerate(jobs)]
    with mp.get_context('fork').Pool(CFG['WORKERS']) as pool:
        return pool.map(_job, args, chunksize=1)

# ==============================================================================
# 10. 학습 (세계 섞어서 전문가 반복)
# ==============================================================================
class Trainer:
    def __init__(self, net):
        self.net = net
        self.opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        self.pol = collections.deque(maxlen=6000)
        self.val = collections.deque(maxlen=14000)

    def add(self, prob, res):
        wid = prob['wid']
        if res['ok']:
            n = len(res['steps'])
            for i, (st, ni, mv) in enumerate(res['steps']):
                self.pol.append((wid, st, prob['goal'], prob['hyps'], ni, mv))
                self.val.append((wid, st, prob['goal'], ni, 0.85 ** (n - i)))
        else:
            for st, ni in res['neg']: self.val.append((wid, st, prob['goal'], ni, 0.0))

    def train(self, worlds, iters=150, bs=24):
        if len(self.pol) < 16: return 0.0, 0.0
        net = self.net; net.train(); pol, val = list(self.pol), list(self.val)
        tl = tv = 0.0
        for _ in range(iters):
            batch = random.sample(pol, min(bs, len(pol)))
            wids = sorted(set(b[0] for b in batch))
            remb_rows, base = [], {}
            for w in wids:
                e = net.rule_embs(worlds[w]); base[w] = sum(x.shape[0] for x in remb_rows); remb_rows.append(e)
            rembs = torch.cat(remb_rows, 0)
            h, gs, gg, offs = net.encode_pairs([(st, goal, worlds[w].sig) for w, st, goal, _, _, _ in batch])
            items, seg, tgts = [], [], []
            for bi, (w, st, goal, hyps, ni, chosen) in enumerate(batch):
                world = worlds[w]
                mv = [m for m in moves(world, st, goal, hyps, ni) if size(m[3]) <= size(st) + CFG['MAX_SIZE_SLACK'] + 4]
                keys = [(m[0], m[1], m[2]) for m in mv]
                if tuple(chosen) not in keys: continue
                tgts.append(keys.index(tuple(chosen))); seg.append((len(items), len(mv)))
                items += [(bi, m[1], base[w] + m[0], bind_tok(m[2], goal, st, world.sig)) for m in mv]
            loss = 0.0
            if seg:
                lg = net.policy_logits(h, gs, gg, offs, items, rembs)
                C = max(n for _, n in seg); S = torch.full((len(seg), C), -1e9)
                for k, (o, n) in enumerate(seg): S[k, :n] = lg[o:o + n]
                lp = F.cross_entropy(S, torch.tensor(tgts)); loss = loss + lp; tl += float(lp.detach())
            vb = random.sample(val, min(48, len(val)))
            vp = net.value_pairs([(s, g, worlds[w].sig) for w, s, g, _, _ in vb], [ni for _, _, _, ni, _ in vb])
            lv = F.binary_cross_entropy(vp.clamp(1e-4, 1 - 1e-4), torch.tensor([t for *_, t in vb], dtype=torch.float))
            loss = loss + lv; tv += float(lv)
            self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); self.opt.step()
        net.eval()
        return tl / iters, tv / iters

# ==============================================================================
# 11. 시험 세트
# ==============================================================================
FAM_NAME = {'bool4': '불대수(위장)', 'z5field': '5원소체(위장)', 's3group': '대칭군S3(위장)',
            'rand': '처음보는무작위', 'chain': '처음보는사슬', 'affine': '처음보는아핀'}

def build_suite(worlds, seed, per_world, n_decoy, forbid, steps=(6, 14)):
    jobs = []
    for w in worlds:
        pr = Proposer(w, seed + w.wid); got = 0; tries = 0
        kinds = (['cond'] if w.pred is not None else []) + ['infl', 'plain', 'plain']
        while got < per_world and tries < per_world * 30:
            tries += 1
            k = kinds[got % len(kinds)]
            p = pr.make(k, pr.rng.randint(*steps))
            if p is None or (w.wid, p['start']) in forbid: continue
            forbid.add((w.wid, p['start'])); jobs.append((w.wid, p)); got += 1
        d = 0
        while d < n_decoy:
            q = pr.decoy()
            if q is None: break
            jobs.append((w.wid, q)); d += 1
        for _ in range(n_decoy):
            q = pr.sensor_decoy()
            if q is not None: jobs.append((w.wid, q))
    return jobs

def score(jobs, rs, worlds):
    by = collections.defaultdict(lambda: [0, 0]); breaches = 0; sbreach = 0; steps = []
    for (wid, p), r in zip(jobs, rs):
        if p['kind'] == 'decoy':
            if p.get('sensor'): sbreach += bool(r['ok'])
            else: breaches += bool(r['ok'])
            continue
        fam = worlds[wid].family
        by[fam][1] += 1
        if r['ok']: by[fam][0] += 1; steps.append(len(r['path']))
    tot = sum(v[0] for v in by.values()); n = sum(v[1] for v in by.values())
    return dict(acc=100.0 * tot / max(1, n), by={k: tuple(v) for k, v in by.items()}, breaches=breaches + sbreach, sbreach=sbreach,
                steps=float(np.mean(steps)) if steps else 0.0)

def fmt(sc):
    order = ['bool4', 'z5field', 's3group', 'rand', 'chain', 'affine']
    return ' '.join(f"{FAM_NAME[k]} {sc['by'][k][0]}/{sc['by'][k][1]}" for k in order if k in sc['by'])

def strip_waffles(w):
    w2 = copy.copy(w); w2.rules = w.rules[:w.n_base]; return w2

# ==============================================================================
# 12. 메인
# ==============================================================================
def main():
    random.seed(CFG['SEED']); np.random.seed(CFG['SEED']); torch.manual_seed(CFG['SEED'])
    t = time.time()
    train_worlds = gen_worlds(['rand', 'chain', 'affine'], CFG['WORLDS_TRAIN'], CFG['SEED'], 0)
    valid_worlds = gen_worlds(['rand', 'chain', 'affine'], 8, 555, 10000)
    test_worlds = (gen_worlds(['bool4'], 1, 901, 20000) + gen_worlds(['z5field'], 1, 902, 20001) +
                   gen_worlds(['s3group'], 1, 903, 20002) + gen_worlds(['rand', 'chain', 'affine'], 6, 904, 20003))
    W = {w.wid: w for w in train_worlds + valid_worlds + test_worlds}
    forbid = set()
    T = build_suite(test_worlds, 7000, 16, 2, forbid)
    V = build_suite(valid_worlds, 8000, 6, 0, forbid, steps=(4, 10))

    net = KimCore(CFG['D']); control = copy.deepcopy(net); trainer = Trainer(net)
    log("=" * 100)
    log(" [김코어 v6.0] 무한 세계 메타 추론: 세계를 외우지 말고 추론을 배워라")
    log(f"  학습 세계 {len(train_worlds)}개 (무작위표/사슬/아핀) | 검증 세계 {len(valid_worlds)} | 시험 세계 {len(test_worlds)} (불대수·Z5체·S3군 위장 + 처음보는 6)")
    log(f"  뇌 파라미터 {sum(p.numel() for p in net.parameters()):,}개 | 세계 생성 {time.time() - t:.0f}s")
    for w in test_worlds[:3]:
        kinds = collections.Counter(r.kind for r in w.rules)
        log(f"  시험 세계 [{FAM_NAME[w.family]}] 원소 {w.n}개, 연산 {list(w.ops)}, 상수 {list(w.consts)}, 규칙 {dict(kinds)}")
        for r in [x for x in w.rules if x.kind in ('shrink', 'cond')][:3]:
            log(f"      예: {show(r.lhs)} → {show(r.rhs)}{'  (센서: ' + show(r.cond[0]) + ')' if r.cond else ''}")
    log(f"  시험 문제 {sum(1 for _, p in T if p['kind'] != 'decoy')}개 (6~14단계) + 함정 {sum(1 for _, p in T if p['kind'] == 'decoy')}개 (그중 센서 함정 {sum(1 for _, p in T if p.get('sensor'))}개, 전부 우주 대입으로 값이 다름이 증명됨)")
    log("=" * 100)

    t = time.time()
    c0 = score(T, run_parallel(W, control, T, CFG['EVAL_BUDGET']), W)
    cv = score(V, run_parallel(W, control, V, CFG['EVAL_BUDGET']), W)
    log(f"\n📊 [대조군: 같은 v6 엔진, 학습 0회, 와플 0개] 시험 {c0['acc']:.1f}% | {fmt(c0)} | 함정 돌파 {c0['breaches']} | 검증 {cv['acc']:.1f}% ({time.time() - t:.0f}s)")
    metrics = dict(config=CFG, control=c0, gens=[])
    best_v, best_state = cv['acc'], copy.deepcopy(net.state_dict())
    rng = random.Random(CFG['SEED']); difficulty = 2
    counters = collections.defaultdict(collections.Counter)

    for gen in range(1, CFG['GENS'] + 1):
        tg = time.time()
        jobs = []
        for w in rng.sample(train_worlds, CFG['WORLDS_PER_GEN']):
            pr = Proposer(w, rng.randrange(1 << 30)); k = 0; tries = 0
            while k < CFG['PROBS_PER_WORLD'] and tries < 40:
                tries += 1
                p = pr.train_problem(difficulty)
                if p is not None and (w.wid, p['start']) not in forbid: jobs.append((w.wid, p)); k += 1
        rs = run_parallel(W, net, jobs, CFG['TRAIN_BUDGET'], explore=True, seed=gen)
        solved_by = collections.defaultdict(list); nsol = 0; ninf = ncond = 0
        for (wid, p), r in zip(jobs, rs):
            trainer.add(p, r)
            if r['ok'] and r['path']:
                nsol += 1; solved_by[wid].append((p, r['path']))
                kinds = [W[wid].rules[ri].kind for _, _, ri, _, _ in r['path']]
                ninf += 'infl' in kinds; ncond += 'cond' in kinds
        rate = nsol / max(1, len(jobs))
        pl, vl = trainer.train(W)
        nw = 0
        for wid, sol in solved_by.items():
            nw += sum(1 for l in forge_waffles(W[wid], sol, gen, counters[wid]) if '🧇' in l)
        v = score(V, run_parallel(W, net, V, CFG['EVAL_BUDGET']), W)
        rb = False
        if v['acc'] > best_v: best_v, best_state = v['acc'], copy.deepcopy(net.state_dict())
        elif v['acc'] < best_v - 15: net.load_state_dict(best_state); rb = True
        rec = dict(gen=gen, rate=rate, difficulty=difficulty, valid=v['acc'], new_waffles=nw, infl=ninf, cond=ncond,
                   pl=pl, vl=vl, rollback=rb, secs=time.time() - tg)
        line = (f"[제 {gen:02d} 세대 | 난이도 Lv.{difficulty}] 새 세계 {CFG['WORLDS_PER_GEN']}개에서 {nsol}/{len(jobs)} 해결 "
                f"(뻥튀기 {ninf}, 센서 {ncond}) | 와플 +{nw} | 검증 세계 {v['acc']:.1f}%{' ↩️롤백' if rb else ''} | {rec['secs']:.0f}s")
        if gen % CFG['EVAL_EVERY'] == 0 or gen == CFG['GENS']:
            Tz = [(wid, p) for wid, p in T]
            Wz = {**W, **{w.wid: strip_waffles(w) for w in test_worlds}}
            z = score(T, run_parallel(Wz, net, Tz, CFG['EVAL_BUDGET']), W)
            rec.update(test=z['acc'], test_by=z['by'], breaches=z['breaches'])
            line += f"\n   🔒 처음 보는 시험 세계 (무학습 즉석): 대조군 {c0['acc']:.1f}% vs 김코어 {z['acc']:.1f}% [{z['acc'] - c0['acc']:+.1f}%p] | {fmt(z)} | 함정 돌파 {z['breaches']}"
        log(line)
        metrics['gens'].append(rec)
        if rate >= 0.7 and difficulty < 10: difficulty += 1
        elif rate < 0.3 and difficulty > 2: difficulty -= 1
        with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)

    # ---------------- 최종: 즉석 vs 새 세계 적응 (시험 세계에서 '스스로 낸 문제'만으로 와플 주조)
    log("\n" + "=" * 100)
    log(" 🌍 [새 세계 적응 시험] 시험 세계마다: 스스로 문제 48개 출제 → 풀기 → 와플 주조 (가중치 고정) → 재시험")
    Wz = {**W, **{w.wid: strip_waffles(w) for w in test_worlds}}
    zero = score(T, run_parallel(Wz, net, T, CFG['EVAL_BUDGET']), W)
    adapted = {}
    for w in test_worlds:
        wa = strip_waffles(w); pr = Proposer(wa, 4242 + w.wid); probs = []
        while len(probs) < CFG['ADAPT_PROBS']:
            p = pr.train_problem(rng.randint(3, 7))
            if p is not None and (w.wid, p['start']) not in forbid: probs.append((w.wid, p))
        Wtmp = {**W, w.wid: wa}
        rs = run_parallel(Wtmp, net, probs, CFG['TRAIN_BUDGET'], explore=True, seed=w.wid)
        sol = [(p, r['path']) for (_, p), r in zip(probs, rs) if r['ok'] and r['path']]
        cnt = collections.Counter()
        forge_waffles(wa, sol, 99, cnt, min_count=2, max_add=8)
        adapted[w.wid] = wa
        log(f"   [{FAM_NAME[w.family]}] 자가 출제 {len(probs)}문제 중 {len(sol)}개 해결 → 와플 {len(wa.waffles)}개 주조")
    Wa = {**W, **adapted}
    ad_rs = run_parallel(Wa, net, T, CFG['EVAL_BUDGET'])
    ad = score(T, ad_rs, W)
    # 가장 어려운(가장 긴) 풀이 TOP 3 기록
    hard = sorted([((wid, p), r) for (wid, p), r in zip(T, ad_rs) if p['kind'] != 'decoy' and r['ok']],
                  key=lambda x: -len(x[1]['path']))[:3]
    log("\n 🏔️ [가장 어려운 풀이 TOP 3]")
    for (wid, p), r in hard:
        w = Wa[wid]
        log(f"   [{FAM_NAME[w.family]}] {len(r['path'])}수 | 시작 {show(p['start'])}")
        log(f"      목표 {show(p['goal'])}" + (f" | 가정 {[show(h) for h in p['hyps']]}" if p['hyps'] else ''))
        for st, ni, ri, nidx, bs in r['path']:
            rr = w.rules[ri]
            log(f"      {rr.kind:7s} {show(rr.lhs)} → {show(rr.rhs)}" + (f"  [기호 {bs}]" if bs else ''))
    noinfl = score(T, run_parallel(Wa, net, T, CFG['EVAL_BUDGET'], flags=dict(allow_infl=False)), W)
    nosensor = score([(w, p) for w, p in T if p['kind'] == 'decoy'],
                     run_parallel(Wz, control, [(w, p) for w, p in T if p['kind'] == 'decoy'], CFG['EVAL_BUDGET'], flags=dict(check_cond=False)), W)

    log("\n" + "=" * 100)
    log(" 🏆 [김코어 v6 최종 성적표] 시험 세계는 학습 때 한 번도 안 나온 세계 (연산 이름도 무작위)")
    log("=" * 100)
    log(f"   대조군 (학습 0)            : {c0['acc']:5.1f}%  | {fmt(c0)}")
    log(f"   김코어 v6 즉석 (적응 0)     : {zero['acc']:5.1f}%  | {fmt(zero)}")
    log(f"   김코어 v6 + 새 세계 적응    : {ad['acc']:5.1f}%  | {fmt(ad)}")
    log(f"   └ 뻥튀기 끄면               : {noinfl['acc']:5.1f}%")
    nsd = sum(1 for _, p in T if p.get('sensor'))
    log(f"   함정 돌파: 대조군 {c0['breaches']} / 즉석 {zero['breaches']} / 적응 {ad['breaches']}  (함정 = 우주에서 값이 다른 식, 0이어야 정상)")
    log(f"   센서 함정 {nsd}문항 (조건을 지우면 값이 달라지는 식): 센서 다리미 돌파 {ad['sbreach']} vs 센서 끈 다리미 {nosensor['sbreach']} '풀었다고 거짓말'")
    log(f"   평균 풀이 길이: 대조군 {c0['steps']:.2f}수 / 즉석 {zero['steps']:.2f}수 / 적응 {ad['steps']:.2f}수 | 총 {time.time() - _T0:.0f}s")
    for g in metrics['gens']:
        if 'test' in g:
            log(f"   Gen {g['gen']:02d} | 시험 즉석 {g['test']:5.1f}% [{'█' * int(g['test'] // 5):<20}] | 검증 {g['valid']:5.1f}%")
    metrics['final'] = dict(control=c0, zero=zero, adapted=ad, no_infl=noinfl['acc'], sensorless_breaches=nosensor['breaches'])
    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)

if __name__ == '__main__':
    main()
