"""
김코어 v6.5 - 세계가 진화하고, 기술이 세계를 건넌다

v6의 약점 3개를 정면 수리:
  1) 법칙 모양 목록(템플릿) 폐기 -> QuickSpec식 법칙 발견
     우주에서 작은 식을 전부 만들어 '모든 입력에 대한 값표(지문)'를 비교. 값표가 같으면 같은 식.
     작은 대표식만 재료로 쓰기 때문에 크기 7짜리 식(분배법칙 같은 것)까지 폭발 없이 탐색.
     사람이 '어떤 모양의 법칙이 있을 것'이라고 알려주지 않음.
  2) 세계 출제기도 학습 (열린 진화형 커리큘럼)
     세계 계열 x 크기 수준 = 12개 팔(arm). 김코어의 '학습 진척도'가 큰 팔에서 새 세계를 더 많이 만든다.
     다 익힌 세계는 은퇴, 아슬아슬한 세계는 재방문.
  3) 범용 다리미 (세계를 건너는 기술)
     여러 세계의 풀이에서 반복된 '규칙 모양 연쇄'를 연산 이름 없이 저장.
     처음 보는 세계에 들어가면, 그 세계의 실제 규칙들로 모양을 다시 조립(단일화/narrowing)해서
     그 세계 전용 합성 규칙을 즉석 생성. 조립은 정당한 규칙 연쇄라 원리상 참이고, 우주 대입으로 이중 확인.

시험
  - 시험지 V6: v6와 완전히 같은 세계/문제 (직접 비교, v6 김코어 93.8%)
  - 시험지 H: 더 크고 깊은 처음 보는 계열 (3비트 불대수 8원소, Z7체, 정사각형 대칭군 D4, 12의 약수 격자)
             + 처음 보는 생성 세계, 10~20단계 문제, 함정/센서 함정 포함
  - 팔(arm): 대조군 / 학습만 / 학습+범용 다리미 / 학습+범용 다리미+새 세계 자가 적응
"""
import os, sys, time, json, math, random, copy, itertools, collections
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(os.environ.get('V65_SEED', 1234))
OUT = os.environ.get('V65_OUT', os.path.join(HERE, f'v65_out_s{SEED}'))
os.makedirs(OUT, exist_ok=True)
os.environ['V6_OUT'] = os.path.join(OUT, '_v6_base')       # v6 모듈 import가 기존 v6 로그를 덮어쓰지 않게
sys.path.insert(0, HERE)
import v6 as B
from v6 import L, N, size, show, positions, get_at, replace_at, match, subst, vars_of, is_var, Rule

CFG = dict(
    SEED=SEED, GENS=int(os.environ.get('V65_GENS', 30)),
    NEW_WORLDS_PER_GEN=5, REVISIT_PER_GEN=7, PROBS_PER_WORLD=4,
    SMAX=7, REPMAX=40, MAX_SHRINK=22, MAX_NEUTRAL=6, MAX_COND=4, MAX_INFL=8,
    MACRO_MIN_WORLDS=3, MAX_MACROS=25, MAX_MACRO_WAFFLES=12,
    EVAL_EVERY=6,
)
B.CFG['WORKERS'] = int(os.environ.get('V65_WORKERS', 4))
_T0 = time.time()
_LOG = open(os.path.join(OUT, 'run.txt'), 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); _LOG.write(s + '\n'); _LOG.flush()

VX, VY, VZ = 'x', 'y', 'z'

# ==============================================================================
# 1. 템플릿 없는 법칙 발견 (QuickSpec식, numpy 지문)
# ==============================================================================
def discover_laws_free(world, rng):
    n = world.n
    grid = np.indices((n, n, n)).reshape(3, -1)          # 모든 (x,y,z) 조합
    rows = grid.shape[1]
    tabs = {}
    for s, (ar, tb) in world.ops.items():
        if ar == 1:
            t = np.zeros(n, dtype=np.int64)
            for (a,), v in tb.items(): t[a] = v
        else:
            t = np.zeros((n, n), dtype=np.int64)
            for (a, b), v in tb.items(): t[a, b] = v
        tabs[s] = (ar, t)
    mask = None
    if world.pred is not None:
        mask = np.isin(grid[0], sorted(world.pred))     # 센서 조건: x ∈ P
    atoms = [(L(VX), grid[0]), (L(VY), grid[1]), (L(VZ), grid[2])]
    atoms += [(L(c), np.full(rows, v, dtype=np.int64)) for c, v in world.consts.items()]
    classes, cclasses = {}, {}
    reps = collections.defaultdict(list)
    eqs, ceqs = [], []

    def consider(term, fp):
        key = fp.tobytes()
        if key in classes:
            eqs.append((term, classes[key]))
        else:
            classes[key] = term
            if len(reps[size(term)]) < CFG['REPMAX']:
                reps[size(term)].append((term, fp))
        if mask is not None and VX in [x[0] for x in _leaves(term)]:
            ck = fp[mask].tobytes()
            if ck in cclasses:
                r = cclasses[ck]
                if fp.tobytes() != _fp_of(r) and size(term) > size(r):
                    ceqs.append((term, r))
            else:
                cclasses[ck] = term
                _FP[term] = key

    _FP = {}
    def _fp_of(t): return _FP.get(t, b'')
    def _leaves(t):
        if not t[1]: return [t]
        out = []
        for c in t[1]: out += _leaves(c)
        return out

    for t, fp in atoms:
        _FP[t] = fp.tobytes(); consider(t, fp)
    for s in range(2, CFG['SMAX'] + 1):
        cand = []
        for op, (ar, tb) in tabs.items():
            if ar == 1:
                for t, fp in reps[s - 1]:
                    cand.append((N(op, t), tb[fp]))
            else:
                for sa in range(1, s - 1):
                    sb = s - 1 - sa
                    for ta, fa in reps[sa]:
                        for tb_, fb in reps[sb]:
                            cand.append((N(op, ta, tb_), tb[fa, fb]))
        if len(cand) > 30000:
            cand = rng.sample(cand, 30000)
        for t, fp in cand:
            _FP.setdefault(t, fp.tobytes())
            consider(t, fp)

    def to_pattern(lhs, rhs, x_first=True):
        order = []
        def walk(g):
            if not g[1] and g[0] in (VX, VY, VZ) and g[0] not in order: order.append(g[0])
            for c in g[1]: walk(c)
        walk(lhs)
        if x_first and VX in order: order.remove(VX); order.insert(0, VX)
        m = {v: f'?{i}' for i, v in enumerate(order)}
        def conv(g):
            if not g[1] and g[0] in (VX, VY, VZ): return L(m.get(g[0], '?BAD'))
            return (g[0], tuple(conv(c) for c in g[1]))
        return conv(lhs), conv(rhs), m

    shrink, neutral, cond = [], [], []
    seen = set()
    ground = 0
    # 같은 크기면 변수가 있는 '일반 법칙'을 먼저: 상수끼리의 계산 사실은 대부분 일반 법칙의 특수한 경우라 자동 중복 제거
    for t, r in sorted(eqs, key=lambda e: (size(e[0]) + size(e[1]), 0 if vars_of(to_pattern(*e)[0]) else 1, show(e[0]))):
        lhs, rhs, _ = to_pattern(t, r)
        if '?BAD' in show(rhs) or is_var(lhs) or (lhs, rhs) in seen: continue
        seen.add((lhs, rhs))
        if size(lhs) > size(rhs):
            if len(shrink) >= CFG['MAX_SHRINK']: continue
            if any(match(x.lhs, sub, {}) for x in shrink for _, sub in positions(lhs)):
                continue                 # 이미 고른 법칙으로 줄일 수 있으면 중복
            if not vars_of(lhs):
                if ground >= 4: continue
                ground += 1
            shrink.append(Rule(f"법칙{len(shrink)}[축소]", lhs, rhs, 'shrink'))
        elif size(lhs) == size(rhs) and lhs != rhs and vars_of(lhs) == vars_of(rhs):
            neutral.append((lhs, rhs))
        if len(shrink) >= CFG['MAX_SHRINK'] and len(neutral) >= CFG['MAX_NEUTRAL'] * 3: break
    def _vx(e): return vars_of(e[0])
    P = world.psym
    for t, r in sorted(ceqs, key=lambda e: size(e[0]) + size(e[1])):
        if len(cond) >= CFG['MAX_COND']: break
        lhs, rhs, m = to_pattern(t, r)
        if '?BAD' in show(rhs) or is_var(lhs) or m.get(VX) != '?0': continue
        if any(c.lhs == lhs for c in cond): continue
        if not B.holds(world, lhs, rhs, condvars=('?0',)) or B.holds(world, lhs, rhs): continue
        cond.append(Rule(f"센서{len(cond)}[|P]", lhs, rhs, 'cond', cond=[N(P, L('?0'))]))
    rng.shuffle(neutral)
    R = list(shrink[:CFG['MAX_SHRINK']])
    for i, (l, r) in enumerate(neutral[:CFG['MAX_NEUTRAL']]):
        R.append(Rule(f"법칙N{i}[→]", l, r, 'neutral')); R.append(Rule(f"법칙N{i}[←]", r, l, 'neutral'))
    R += cond
    # 뻥튀기: 축소/센서 법칙의 역방향 (잃는 변수는 최대 1개, 잎 기호로 채움)
    src = [x for x in R if x.kind in ('shrink', 'cond')]; rng.shuffle(src); k = 0
    for x in src:
        if k >= CFG['MAX_INFL']: break
        lost = vars_of(x.lhs) - vars_of(x.rhs)
        if len(lost) > 1: continue
        lhs2, rhs2 = x.rhs, x.lhs
        if lost:
            v = next(iter(lost)); rhs2 = subst(rhs2, {v: B.XF, **{u: L(u) for u in vars_of(rhs2) if u != v}})
        condf = [subst(c, {'?0': (B.XF if '?0' in lost else L('?0'))}) for c in x.cond]
        R.append(Rule(f"뻥튀기{k}[{x.name}역]", lhs2, rhs2, 'infl', cond=condf)); k += 1
    for x in R:   # 안전 확인: 모든 무조건 규칙은 우주에서 참이어야 함
        if x.kind in ('shrink', 'neutral') and not B.holds(world, x.lhs, x.rhs):
            raise AssertionError(f"법칙 발견 버그: {show(x.lhs)} = {show(x.rhs)}")
    world.rules = R; world.n_base = len(R)
    fr = []
    if world.pred is not None:
        for c, v in world.consts.items():
            if v in world.pred: fr.append((N(P, L(c)), []))
        for s, (ar, tb) in world.ops.items():
            if all(tb[a] in world.pred for a in itertools.product(sorted(world.pred), repeat=ar)):
                args = [L('?0'), L('?1')][:ar]
                fr.append((N(P, N(s, *args)), [N(P, a) for a in args]))
    world.fact_rules = fr
    return len(shrink)

# ==============================================================================
# 2. 우주 계열 (학습용 3계열 x 4수준, 시험 전용 4계열)
# ==============================================================================
def t2(n, fn): return {(a, b): fn(a, b) % n for a in range(n) for b in range(n)}
def t1(n, fn): return {(a,): fn(a) % n for a in range(n)}

def make_universe(family, level, rng):
    used = set(); nm = lambda: B.rand_name(rng, used)
    ops, consts, pred = {}, {}, None
    if family == 'rand':
        n = 2 + level
        for _ in range(1 + (level >= 2 and rng.random() < 0.6)):
            ops[nm()] = (2, {(a, b): rng.randrange(n) for a in range(n) for b in range(n)})
        if rng.random() < 0.6: ops[nm()] = (1, {(a,): rng.randrange(n) for a in range(n)})
        for v in rng.sample(range(n), rng.choice([1, 2])): consts[nm()] = v
        if rng.random() < 0.5: pred = frozenset(rng.sample(range(n), rng.randint(1, n - 1)))
    elif family == 'chain':
        n = 3 + level
        ops[nm()] = (2, t2(n, min)); ops[nm()] = (2, t2(n, max))
        if rng.random() < 0.5: ops[nm()] = (1, {(a,): min(a + 1, n - 1) for a in range(n)})
        consts[nm()] = 0; consts[nm()] = n - 1
        if rng.random() < 0.4: pred = frozenset(range(1, n))
    elif family == 'affine':
        n = [4, 6, 8, 9][level]
        for _ in range(rng.choice([1, 2])):
            a, b, c = rng.randrange(n), rng.randrange(n), rng.randrange(n)
            ops[nm()] = (2, t2(n, lambda x, y, a=a, b=b, c=c: a * x + b * y + c))
        if rng.random() < 0.7:
            d, e = rng.randrange(n), rng.randrange(n); ops[nm()] = (1, t1(n, lambda x, d=d, e=e: d * x + e))
        for v in rng.sample(range(n), rng.choice([1, 2])): consts[nm()] = v
        if rng.random() < 0.4: pred = frozenset(x for x in range(n) if math.gcd(x, n) == 1)
    elif family == 'bool8':
        n = 8
        ops[nm()] = (2, t2(n, lambda x, y: x & y)); ops[nm()] = (2, t2(n, lambda x, y: x | y)); ops[nm()] = (1, t1(n, lambda x: 7 - x))
        consts[nm()] = 0; consts[nm()] = 7
    elif family == 'z7field':
        n = 7; inv = {0: 0, **{a: pow(a, 5, 7) for a in range(1, 7)}}
        ops[nm()] = (2, t2(n, lambda x, y: x + y)); ops[nm()] = (2, t2(n, lambda x, y: x * y)); ops[nm()] = (1, t1(n, lambda x: inv[x]))
        consts[nm()] = 0; consts[nm()] = 1; pred = frozenset(range(1, 7))
    elif family == 'd4group':
        r = (1, 2, 3, 0); s = (0, 3, 2, 1)
        comp = lambda p, q: tuple(p[q[i]] for i in range(4))
        el = {(0, 1, 2, 3)}; frontier = [(0, 1, 2, 3)]
        while frontier:
            p = frontier.pop()
            for g in (r, s):
                q = comp(g, p)
                if q not in el: el.add(q); frontier.append(q)
        el = sorted(el); idx = {p: i for i, p in enumerate(el)}; n = len(el)
        inv = lambda p: tuple(sorted(range(4), key=lambda k: p[k]))
        ops[nm()] = (2, {(a, b): idx[comp(el[a], el[b])] for a in range(n) for b in range(n)})
        ops[nm()] = (1, {(a,): idx[inv(el[a])] for a in range(n)})
        consts[nm()] = idx[(0, 1, 2, 3)]
    elif family == 'divlat12':
        D = [1, 2, 3, 4, 6, 12]; n = 6; ix = {d: i for i, d in enumerate(D)}
        ops[nm()] = (2, {(a, b): ix[math.gcd(D[a], D[b])] for a in range(n) for b in range(n)})
        ops[nm()] = (2, {(a, b): ix[D[a] * D[b] // math.gcd(D[a], D[b])] for a in range(n) for b in range(n)})
        consts[nm()] = 0; consts[nm()] = 5
    return n, ops, consts, pred

def make_world65(wid, family, level, rng):
    n, ops, consts, pred = make_universe(family, level, rng)
    w = B.World(wid, family, n, ops, consts, pred, rng)
    w.level = level
    try:
        ns = discover_laws_free(w, rng)
    except AssertionError as e:
        log('  !! ' + str(e)); return None
    return w if ns >= 4 and w.fact_rules is not None else None

# ==============================================================================
# 3. 범용 다리미: 규칙 '모양' 연쇄를 세계 간 이전 (단일화로 재조립)
# ==============================================================================
def rule_shape(r):
    ops, cs, vs = {}, {}, {}
    def canon(g):
        s = g[0]
        if s[0] == '?':
            if s not in vs: vs[s] = f'?v{len(vs)}'
            return L(vs[s])
        if not g[1]:
            if s not in cs: cs[s] = f'C{len(cs)}'
            return L(cs[s])
        if s not in ops: ops[s] = f'F{len(g[1])}_{len(ops)}'
        return (ops[s], tuple(canon(c) for c in g[1]))
    return (canon(r.lhs), canon(r.rhs))

def _walk(t, s):
    while is_var(t) and t[0] in s: t = s[t[0]]
    return t

def _occurs(v, t, s):
    t = _walk(t, s)
    if is_var(t): return t[0] == v
    return any(_occurs(v, c, s) for c in t[1])

def unify(a, b, s):
    a, b = _walk(a, s), _walk(b, s)
    if a == b: return s
    if is_var(a):
        if _occurs(a[0], b, s): return None
        s = dict(s); s[a[0]] = b; return s
    if is_var(b): return unify(b, a, s)
    if a[0] != b[0] or len(a[1]) != len(b[1]): return None
    for x, y in zip(a[1], b[1]):
        s = unify(x, y, s)
        if s is None: return None
    return s

def resolve(t, s):
    t = _walk(t, s)
    if is_var(t) or not t[1]: return t
    return (t[0], tuple(resolve(c, s) for c in t[1]))

def rename(t, tag):
    if is_var(t): return L(f'{t[0]}#{tag}')
    if not t[1]: return t
    return (t[0], tuple(rename(c, tag) for c in t[1]))

def normalize_vars(lhs, rhs):
    m = {}
    def scan(g):
        if is_var(g) and g[0] not in m: m[g[0]] = f'?{len(m)}'
        for c in g[1]: scan(c)
    scan(lhs)
    def conv(g):
        if is_var(g): return L(m.get(g[0], '?BAD'))
        if not g[1]: return g
        return (g[0], tuple(conv(c) for c in g[1]))
    return conv(lhs), conv(rhs)

def compose(rules_seq, relposes):
    """규칙 연쇄를 하나의 합성 규칙으로 (narrowing). 실패하면 None"""
    start = rename(rules_seq[0].lhs, 0); cur = rename(rules_seq[0].rhs, 0); s = {}
    for j in range(1, len(rules_seq)):
        rp = relposes[j]
        cur_r = resolve(cur, s)
        try: sub = get_at(cur_r, rp)
        except (IndexError, TypeError): return None
        l = rename(rules_seq[j].lhs, j); r = rename(rules_seq[j].rhs, j)
        s = unify(sub, l, s)
        if s is None: return None
        cur = replace_at(resolve(cur, s), rp, r)
    lhs, rhs = normalize_vars(resolve(start, s), resolve(cur, s))
    if '?BAD' in show(rhs) or is_var(lhs) or size(rhs) >= size(lhs): return None
    return lhs, rhs

class MacroBank:
    """세계를 건너는 기술 창고"""
    def __init__(self):
        self.seen = collections.defaultdict(set)    # key -> 등장한 세계들
        self.items = []                              # [(key, n_worlds)]

    def mine(self, W, solved):
        for wid, prob, path in solved:
            w = W[wid]
            steps = [(w.rules[ri], B.node_pos(st, nidx)) for st, ni, ri, nidx, bs in path]
            for i in range(len(steps)):
                for k in (2, 3):
                    seg = steps[i:i + k]
                    if len(seg) < k or any(r.kind not in ('shrink', 'neutral') for r, _ in seg): continue
                    p0 = seg[0][1]
                    if any(p is None or p[:len(p0)] != p0 for _, p in seg): continue
                    key = tuple((rule_shape(r), p[len(p0):]) for r, p in seg)
                    self.seen[key].add(wid)
        ranked = sorted(((k, len(v)) for k, v in self.seen.items() if len(v) >= CFG['MACRO_MIN_WORLDS']), key=lambda x: -x[1])
        self.items = ranked[:CFG['MAX_MACROS']]

    def instantiate(self, w, cap=None):
        """처음 보는 세계에서 모양을 실제 규칙으로 재조립 -> 그 세계 전용 합성 규칙(와플) 생성"""
        cap = cap or CFG['MAX_MACRO_WAFFLES']
        base = [r for r in w.rules[:w.n_base] if r.kind in ('shrink', 'neutral')]
        by_shape = collections.defaultdict(list)
        for r in base: by_shape[rule_shape(r)].append(r)
        have = set((r.lhs, r.rhs) for r in w.rules); added = 0
        for key, nw in self.items:
            if added >= cap: break
            pools = [by_shape.get(sh, []) for sh, _ in key]
            if not all(pools): continue
            for combo in itertools.islice(itertools.product(*pools), 20):
                res = compose(list(combo), [rp for _, rp in key])
                if res is None or res in have: continue
                lhs, rhs = res
                if not B.holds(w, lhs, rhs): continue          # 이중 확인 (원리상 항상 참)
                w.rules.append(Rule(f"범용다리미#{added + 1}", lhs, rhs, 'waffle'))
                have.add(res); added += 1
                break
        return added

def strip(w):
    w2 = copy.copy(w); w2.rules = w.rules[:w.n_base]; return w2

def with_macros(w, bank):
    w2 = strip(w); bank.instantiate(w2); return w2

# ==============================================================================
# 4. 열린 진화형 세계 커리큘럼 (학습 진척도 밴딧)
# ==============================================================================
class WorldCurriculum:
    FAMS = ['rand', 'chain', 'affine']
    def __init__(self, rng):
        self.rng = rng
        self.arms = [(f, l) for f in self.FAMS for l in range(4)]
        self.p = {a: 0.5 for a in self.arms}; self.lp = {a: 0.3 for a in self.arms}; self.cnt = {a: 0 for a in self.arms}
        self.active = {}      # wid -> dict(world, ema, n, arm)
        self.retired = 0; self.next_id = 0

    def weight(self, a):
        w = self.lp[a] + 0.05
        if self.p[a] > 0.97 and self.cnt[a] > 6: w *= 0.2      # 다 익힌 계열은 덜
        return w

    def spawn(self, k):
        out = []
        for _ in range(k):
            for _try in range(20):
                ws = [self.weight(a) for a in self.arms]
                a = self.rng.choices(self.arms, weights=ws)[0]
                w = make_world65(self.next_id, a[0], a[1], self.rng); self.next_id += 1
                if w is not None:
                    self.active[w.wid] = dict(world=w, ema=0.5, n=0, arm=a); out.append(w); break
        return out

    def revisit(self, k):
        cand = [v for v in self.active.values() if v['n'] > 0]
        cand.sort(key=lambda v: abs(v['ema'] - 0.6))               # 아슬아슬한 세계(성공률 0.6 근처) 우선
        return [v['world'] for v in cand[:k]]

    def update(self, wid, ok):
        v = self.active.get(wid)
        if v is None: return
        v['n'] += 1; v['ema'] = 0.7 * v['ema'] + 0.3 * float(ok)
        a = v['arm']; old = self.p[a]
        self.p[a] = 0.9 * self.p[a] + 0.1 * float(ok); self.cnt[a] += 1
        self.lp[a] = 0.8 * self.lp[a] + 0.2 * abs(self.p[a] - old) * 10
        if v['n'] >= 8 and v['ema'] > 0.95:
            del self.active[wid]; self.retired += 1

    def summary(self):
        return ' '.join(f"{f[:2]}{l}:{self.weight((f, l)):.2f}" for f, l in self.arms)

# ==============================================================================
# 5. 시험지
# ==============================================================================
FAM_NAME = dict(B.FAM_NAME, bool8='불대수8(위장)', z7field='Z7체(위장)', d4group='대칭군D4(위장)', divlat12='약수격자12(위장)')

def fmt(sc):
    order = ['bool4', 'z5field', 's3group', 'bool8', 'z7field', 'd4group', 'divlat12', 'rand', 'chain', 'affine']
    return ' '.join(f"{FAM_NAME[k]} {sc['by'][k][0]}/{sc['by'][k][1]}" for k in order if k in sc['by'])

def suite_v6():
    """v6 main()과 완전히 같은 시드/순서로 같은 시험 세계와 문제를 재생성"""
    valid_worlds = B.gen_worlds(['rand', 'chain', 'affine'], 8, 555, 10000)
    tw = (B.gen_worlds(['bool4'], 1, 901, 20000) + B.gen_worlds(['z5field'], 1, 902, 20001) +
          B.gen_worlds(['s3group'], 1, 903, 20002) + B.gen_worlds(['rand', 'chain', 'affine'], 6, 904, 20003))
    forbid = set()
    T = B.build_suite(tw, 7000, 16, 2, forbid)
    return tw, T

def suite_hard():
    rng = random.Random(4646); tw = []; wid = 30000
    for fam in ['bool8', 'z7field', 'd4group', 'divlat12']:
        while True:
            w = make_world65(wid, fam, 0, rng); wid += 1
            if w is not None: tw.append(w); break
    while len(tw) < 8:
        f = rng.choice(['rand', 'chain', 'affine']); w = make_world65(wid, f, rng.randint(1, 3), rng); wid += 1
        if w is not None: tw.append(w)
    T = B.build_suite(tw, 9100, 12, 2, set(), steps=(10, 20))
    return tw, T

# ==============================================================================
# 6. 메인
# ==============================================================================
def evaluate(W, net, T, flags=None):
    return B.score(T, B.run_parallel(W, net, T, B.CFG['EVAL_BUDGET'], flags=flags), W)

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rng = random.Random(SEED)
    t = time.time()
    tw6, T6 = suite_v6(); twH, TH = suite_hard()
    log("=" * 104)
    log(" [김코어 v6.5] 템플릿 없는 법칙 발견 + 진화하는 세계 커리큘럼 + 세계를 건너는 범용 다리미")
    log(f"  시험지 V6: {sum(p['kind'] != 'decoy' for _, p in T6)}문항 (v6와 동일) | 시험지 H: {sum(p['kind'] != 'decoy' for _, p in TH)}문항 "
        f"(10~20단계) + 함정 {sum(p['kind'] == 'decoy' for _, p in TH)} (센서 함정 {sum(1 for _, p in TH if p.get('sensor'))}) | 준비 {time.time() - t:.0f}s")
    for w in twH[:4]:
        kinds = collections.Counter(r.kind for r in w.rules)
        log(f"  시험 세계 [{FAM_NAME[w.family]}] 원소 {w.n}개 | 발견된 규칙 {dict(kinds)}")
        for r in [x for x in w.rules if x.kind in ('shrink', 'cond')][:4]:
            log(f"      {show(r.lhs)} → {show(r.rhs)}" + (f"   (센서 {show(r.cond[0])})" if r.cond else ''))
    W = {w.wid: w for w in tw6 + twH}
    net = B.KimCore(B.CFG['D']); control = copy.deepcopy(net); trainer = B.Trainer(net)
    bank = MacroBank(); cur = WorldCurriculum(rng)
    log(f"  뇌 파라미터 {sum(p.numel() for p in net.parameters()):,}개")
    log("=" * 104)

    W0 = {**W, **{w.wid: strip(w) for w in tw6 + twH}}
    c6 = evaluate(W0, control, T6); cH = evaluate(W0, control, TH)
    log(f"\n📊 [대조군] 시험지 V6 {c6['acc']:.1f}% | 시험지 H {cH['acc']:.1f}% [{fmt(cH)}] | 함정 돌파 {c6['breaches'] + cH['breaches']}")
    metrics = dict(config=CFG, control=dict(v6=c6['acc'], H=cH['acc'], H_by=cH['by']), gens=[])
    difficulty = 2; counters = collections.defaultdict(collections.Counter)
    for w in cur.spawn(12): W[w.wid] = w
    valid_rng = random.Random(555)

    for gen in range(1, CFG['GENS'] + 1):
        tg = time.time()
        fresh = cur.spawn(CFG['NEW_WORLDS_PER_GEN'])
        worlds = fresh + cur.revisit(CFG['REVISIT_PER_GEN'])
        Wt = dict(W)
        for w in worlds:
            if w.wid not in Wt: Wt[w.wid] = w
            if len(w.waffles) == 0 and bank.items: bank.instantiate(w)
        jobs = []
        for w in worlds:
            pr = B.Proposer(w, rng.randrange(1 << 30)); k = tries = 0
            while k < CFG['PROBS_PER_WORLD'] and tries < 40:
                tries += 1; p = pr.train_problem(difficulty)
                if p is not None: jobs.append((w.wid, p)); k += 1
        rs = B.run_parallel(Wt, net, jobs, B.CFG['TRAIN_BUDGET'], explore=True, seed=gen)
        solved = []; nsol = 0; mac_used = 0
        for (wid, p), r in zip(jobs, rs):
            trainer.add(p, r); cur.update(wid, r['ok'])
            if r['ok'] and r['path']:
                nsol += 1; solved.append((wid, p, r['path']))
                mac_used += any(Wt[wid].rules[ri].name.startswith('범용') for _, _, ri, _, _ in r['path'])
        rate = nsol / max(1, len(jobs))
        pl, vl = trainer.train(Wt)
        bank.mine(Wt, solved)
        for w in worlds:   # 기존 세계에도 새 범용 다리미 반영
            bank.instantiate(w, cap=max(0, CFG['MAX_MACRO_WAFFLES'] - len(w.waffles)))
        W.update({w.wid: w for w in worlds})
        rec = dict(gen=gen, rate=rate, difficulty=difficulty, macros=len(bank.items), macro_used=mac_used,
                   active=len(cur.active), retired=cur.retired, arms=cur.summary(), pl=pl, vl=vl, secs=time.time() - tg)
        line = (f"[제 {gen:02d} 세대 | 난이도 Lv.{difficulty}] 세계 {len(worlds)}개(신규 {len(fresh)}) {nsol}/{len(jobs)} 해결 | "
                f"범용 다리미 {len(bank.items)}종 (풀이에 사용 {mac_used}) | 활성 세계 {len(cur.active)}, 졸업 {cur.retired} | {rec['secs']:.0f}s\n"
                f"   세계 출제 가중치: {cur.summary()}")
        if gen % CFG['EVAL_EVERY'] == 0 or gen == CFG['GENS']:
            WM = {**W, **{w.wid: with_macros(w, bank) for w in tw6 + twH}}
            e6 = evaluate(WM, net, T6); eH = evaluate(WM, net, TH)
            rec.update(v6=e6['acc'], H=eH['acc'], H_by=eH['by'], breaches=e6['breaches'] + eH['breaches'])
            line += (f"\n   🔒 시험지 V6: 대조군 {c6['acc']:.1f}% → {e6['acc']:.1f}% (v6 김코어 93.8%) | "
                     f"시험지 H: 대조군 {cH['acc']:.1f}% → {eH['acc']:.1f}% | 함정 돌파 {rec['breaches']}")
        log(line)
        metrics['gens'].append(rec)
        if rate >= 0.7 and difficulty < 12: difficulty += 1
        elif rate < 0.3 and difficulty > 2: difficulty -= 1
        with open(os.path.join(OUT, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)

    # ---------------- 최종: 팔(arm)별 비교
    log("\n" + "=" * 104)
    log(" 🔬 [최종 비교] 같은 시험, 다른 무기")
    Wz = {**W, **{w.wid: strip(w) for w in tw6 + twH}}
    WM = {**W, **{w.wid: with_macros(w, bank) for w in tw6 + twH}}
    z6, zH = evaluate(Wz, net, T6), evaluate(Wz, net, TH)
    m6, mH = evaluate(WM, net, T6), evaluate(WM, net, TH)
    log(f"   범용 다리미 {len(bank.items)}종. 시험 세계에 재조립된 개수: " +
        ', '.join(f"{FAM_NAME[w.family]} {len(WM[w.wid].waffles)}" for w in twH))
    for w in twH[:4]:
        for r in WM[w.wid].waffles[:2]:
            log(f"      [{FAM_NAME[w.family]}] {r.name}: {show(r.lhs)} ⇒ {show(r.rhs)}")
    # 새 세계 자가 적응 (가중치 고정, 스스로 낸 문제로만)
    WA = {}
    for w in tw6 + twH:
        wa = with_macros(w, bank); pr = B.Proposer(wa, 4242 + w.wid); probs = []
        tries = 0
        while len(probs) < 48 and tries < 400:
            tries += 1; p = pr.train_problem(rng.randint(4, 9))
            if p is not None: probs.append((w.wid, p))
        rs = B.run_parallel({**W, w.wid: wa}, net, probs, B.CFG['TRAIN_BUDGET'], explore=True, seed=w.wid)
        sol = [(p, r['path']) for (_, p), r in zip(probs, rs) if r['ok'] and r['path']]
        B.forge_waffles(wa, sol, 99, collections.Counter(), min_count=2, max_add=8)
        WA[w.wid] = wa
    WAll = {**W, **WA}
    a6, aH = evaluate(WAll, net, T6), evaluate(WAll, net, TH)
    nf = evaluate(WAll, net, TH, flags=dict(allow_infl=False))
    dec = [(w, p) for w, p in T6 + TH if p['kind'] == 'decoy']
    sl = B.score(dec, B.run_parallel(Wz, control, dec, B.CFG['EVAL_BUDGET'], flags=dict(check_cond=False)), W)

    log("\n" + "=" * 104)
    log(" 🏆 [김코어 v6.5 최종 성적표]")
    log("=" * 104)
    log(f"   {'':28s} {'시험지 V6 (v6 동일)':>20s} {'시험지 H (더 크고 깊음)':>24s}")
    log(f"   {'대조군 (학습 0)':28s} {c6['acc']:>19.1f}% {cH['acc']:>23.1f}%")
    log(f"   {'v6 김코어 (참고)':28s} {'93.8%':>20s} {'-':>24s}")
    log(f"   {'v6.5 학습만':28s} {z6['acc']:>19.1f}% {zH['acc']:>23.1f}%")
    log(f"   {'v6.5 + 범용 다리미':28s} {m6['acc']:>19.1f}% {mH['acc']:>23.1f}%")
    log(f"   {'v6.5 + 범용 다리미 + 자가적응':28s} {a6['acc']:>19.1f}% {aH['acc']:>23.1f}%")
    log(f"   └ 뻥튀기 끄면 (시험지 H)        {nf['acc']:.1f}%")
    log(f"   시험지 H 세부: 대조군 [{fmt(cH)}]")
    log(f"                  최종   [{fmt(aH)}]")
    log(f"   함정 돌파: 최종 {a6['breaches'] + aH['breaches']}건 | 센서 함정: 센서 다리미 {aH['sbreach'] + a6['sbreach']} vs 센서 끈 다리미 {sl['sbreach']} '거짓말'")
    log(f"   졸업한 세계 {cur.retired}개 | 만든 세계 {cur.next_id}개 | 총 {time.time() - _T0:.0f}s")
    for g in metrics['gens']:
        if 'H' in g:
            log(f"   Gen {g['gen']:02d} | V6 {g['v6']:5.1f}% | H {g['H']:5.1f}% [{'█' * int(g['H'] // 5):<20}] | 범용 다리미 {g['macros']}")
    metrics['final'] = dict(control_v6=c6['acc'], control_H=cH['acc'], train_v6=z6['acc'], train_H=zH['acc'],
                            macro_v6=m6['acc'], macro_H=mH['acc'], adapt_v6=a6['acc'], adapt_H=aH['acc'],
                            no_infl_H=nf['acc'], breaches=a6['breaches'] + aH['breaches'], sensorless=sl['sbreach'])
    with open(os.path.join(OUT, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)

if __name__ == '__main__':
    main()
