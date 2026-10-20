"""
김코어 v5.0 - 수술 3종 세트
  수술 1. 센서 다리미 (Conditional Rewriting): 조건이 '증명'될 때만 작동하는 규칙.
          조건 확인은 작은 산을 먼저 미는 하위 증명(Horn 역방향 연쇄)으로 수행.
  수술 2. 뻥튀기 기계 (Strategic Inflation): 탐색 도중 일부러 그래프를 키우는 수(경로당 최대 2회).
  수술 3. GNN 눈 (Message-Passing Tree GNN): 14차원 요약 대신 모든 노드를 보고
          (어떤 규칙을, 어느 위치에, 어떤 기호로) 쏠지 직접 고르는 저격 정책 + 목표 조건부 가치망.

공정성 장치 (v4.6 계승 + 강화)
  - 시험지 A: v4.6 숨겨진 시험지 40문항 그대로 (v4.6 김코어 80~85%와 직접 비교)
  - 시험지 B: v5 신규 60문항 (센서 20 / 뻥튀기 20 / 심층 20) + 함정 10문항
  - 검증지(valid) 30문항: 롤백 판단 전용. 시험지는 모델 선택에 절대 안 씀.
  - 대조군: 똑같은 v5 엔진/탐색/수술 능력, 학습 0회 + 와플 0개.
  - 함정 문제: 조건을 무시하면 2~3수에 풀리지만, 정당하게는 절대 못 푸는 문제. 풀리면 BREACH.
  - 와플(발명 규칙)은 새 기호 대입 후 원래 경로를 재생해서 검증된 것만 등록.
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
    SEED=int(os.environ.get('V5_SEED', 1234)),
    GENS=int(os.environ.get('V5_GENS', 18)),
    PROBS_PER_GEN=48,
    TRAIN_BUDGET=60,        # 연습 때 노드 확장 횟수
    EVAL_BUDGET=35,         # 시험 때 노드 확장 횟수 (v4.6과 동일)
    MAX_INFL=2,             # 경로당 뻥튀기 최대 횟수
    INFL_SUB_MAX=6,         # 뻥튀기를 씌울 수 있는 하위 그래프 최대 크기
    MAX_SIZE_SLACK=8,       # 시작 그래프보다 이만큼까지 커지는 것 허용
    EVAL_EVERY=3,
    WORKERS=int(os.environ.get('V5_WORKERS', 4)),
    MAX_WAFFLES=60,
    D=64,
)
OUT_DIR = os.environ.get('V5_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), f"v5_out_s{CFG['SEED']}"))
os.makedirs(OUT_DIR, exist_ok=True)
_T0 = time.time()
_LOG = open(os.path.join(OUT_DIR, 'run.txt'), 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); _LOG.write(s + '\n'); _LOG.flush()

# ==============================================================================
# 1. 불변 튜플 그래프: (기호, (자식들...)). 해시/비교가 공짜라 탐색이 빠름
# ==============================================================================
def L(s): return (s, ())
def N(s, *ch): return (s, tuple(ch))
def is_var(g): return g[0][0] == '?'

@lru_cache(maxsize=200000)
def size(g): return 1 + sum(size(c) for c in g[1])

def show(g):
    return g[0] if not g[1] else f"{g[0]}({','.join(show(c) for c in g[1])})"

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

def from_gnode(n):
    """v4.6 GNode -> 튜플 그래프"""
    return (n.symbol, tuple(from_gnode(c) for c in n.children))

P0, P1, P2, PF = L('?0'), L('?1'), L('?2'), L('?f')
E0, E1 = L('e0'), L('e1')

# ==============================================================================
# 2. 수술 1: 센서 다리미의 '작은 산' - 사실(fact) 증명기 (Horn 역방향 연쇄)
# ==============================================================================
FACT_RULES = [
    (N('nz', E1), []),                                   # e1은 0이 아님
    (N('nz', N('op2', P0, P1)), [N('nz', P0), N('nz', P1)]),
    (N('nz', N('u1', P0)), [N('nz', P0)]),
    (N('nz', N('div', P0, P1)), [N('nz', P0), N('nz', P1)]),
]

@lru_cache(maxsize=200000)
def prove(fact, hyps, depth=4):
    """fact(예: nz(op2(s_a,s_b)))가 가정 hyps에서 유도되는가? = 센서가 초록불을 켤 수 있는가"""
    if fact in hyps: return True
    if depth == 0: return False
    for head, body in FACT_RULES:
        b = {}
        if match(head, fact, b):
            if all(prove(subst(x, b), hyps, depth - 1) for x in body):
                return True
    return False

# ==============================================================================
# 3. 규칙 체계: 공리 10 + 센서 공리 2 + 뻥튀기 8 + (발명 와플)
# ==============================================================================
class Rule:
    __slots__ = ('name', 'lhs', 'rhs', 'kind', 'cond', 'free', 'uses', 'born', 'moves')
    def __init__(self, name, lhs, rhs, kind, cond=(), born=0, moves=None):
        self.name, self.lhs, self.rhs, self.kind, self.cond = name, lhs, rhs, kind, tuple(cond)
        self.free = tuple(sorted(vars_of(rhs) - vars_of(lhs) | set().union(*[vars_of(c) for c in cond]) - vars_of(lhs))) if True else ()
        self.uses, self.born, self.moves = 0, born, moves

KIND_ID = {'axiom': 0, 'cond': 1, 'infl': 2, 'waffle': 3}

def build_rules():
    ax = [
        Rule("공리0[우측e0소거]", N('op1', P0, E0), P0, 'axiom'),
        Rule("공리1[좌측e1소거]", N('op2', E1, P0), P0, 'axiom'),
        Rule("공리2[자기소거]", N('op3', P0, P0), E0, 'axiom'),
        Rule("공리3[이중u1소거]", N('u1', N('u1', P0)), P0, 'axiom'),
        Rule("공리4[공통가지묶기]", N('op1', N('op2', P0, P1), N('op2', P0, P2)), N('op2', P0, N('op1', P1, P2)), 'axiom'),
        Rule("공리5[가지펼치기]", N('op2', P0, N('op1', P1, P2)), N('op1', N('op2', P0, P1), N('op2', P0, P2)), 'axiom'),
        Rule("공리6[op1대칭교환]", N('op1', P0, P1), N('op1', P1, P0), 'axiom'),
        Rule("공리7[선택기단순화]", N('sel', P0, P1, P1), P1, 'axiom'),
        Rule("공리8[u1위상침투]", N('u1', N('op1', P0, P1)), N('op2', N('u1', P0), N('u1', P1)), 'axiom'),
        Rule("공리9[우측e1소거]", N('op2', P0, E1), P0, 'axiom'),
    ]
    cond = [
        Rule("센서0[자기나눗셈|nz]", N('div', P0, P0), E1, 'cond', cond=[N('nz', P0)]),
        Rule("센서1[곱나눗셈소거|nz]", N('op2', N('div', P1, P0), P0), P1, 'cond', cond=[N('nz', P0)]),
    ]
    infl = [
        Rule("뻥튀기[+e0]", P0, N('op1', P0, E0), 'infl'),
        Rule("뻥튀기[e1·]", P0, N('op2', E1, P0), 'infl'),
        Rule("뻥튀기[·e1]", P0, N('op2', P0, E1), 'infl'),
        Rule("뻥튀기[u1u1]", P0, N('u1', N('u1', P0)), 'infl'),
        Rule("뻥튀기[e0→op3(x,x)]", E0, N('op3', PF, PF), 'infl'),
        Rule("뻥튀기[sel]", P0, N('sel', PF, P0, P0), 'infl'),
        Rule("뻥튀기[e1→x/x|nz]", E1, N('div', PF, PF), 'infl', cond=[N('nz', PF)]),
        Rule("뻥튀기[x→(x/y)·y|nz]", P0, N('op2', N('div', P0, PF), PF), 'infl', cond=[N('nz', PF)]),
    ]
    return ax, cond, infl

# ==============================================================================
# 4. 엔진: (규칙, 위치, 바인딩) 단위의 모든 합법 수 나열
# ==============================================================================
class Engine:
    def __init__(self, ax, cond, infl):
        self.rules = list(ax) + list(cond) + list(infl)
        self.n_base = len(self.rules)

    @property
    def waffles(self):
        return self.rules[self.n_base:]

    def moves(self, g, goal, hyps, n_infl, allow_infl=True, check_cond=True, use_waffles=True):
        """-> [(rule_idx, node_idx, bind_sym, child_graph)]  (node_idx = 전위순회 번호)"""
        out, seen = [], set()
        pool = (leaf_syms(goal) + [s for s in leaf_syms(g) if s not in leaf_syms(goal)])[:4]
        nrules = len(self.rules) if use_waffles else self.n_base
        for nidx, (pos, sub) in enumerate(positions(g)):
            ssz = size(sub)
            for ri in range(nrules):
                r = self.rules[ri]
                if r.kind == 'infl':
                    if not allow_infl or n_infl >= CFG['MAX_INFL']: continue
                    if is_var(r.lhs) and ssz > CFG['INFL_SUB_MAX']: continue
                b = {}
                if not match(r.lhs, sub, b): continue
                binds = [None] if not r.free else pool
                for bs in binds:
                    bb = dict(b)
                    if bs is not None:
                        bb[r.free[0]] = L(bs)
                    if r.cond and check_cond and not all(prove(subst(c, bb), hyps) for c in r.cond):
                        continue
                    child = replace_at(g, pos, subst(r.rhs, bb))
                    if child == g or child in seen: continue
                    seen.add(child)
                    out.append((ri, nidx, bs, child))
        return out

# ==============================================================================
# 5. 수술 3: GNN 눈 (슬롯 인식 트리 메시지 패싱) + 저격 정책 + 목표 조건부 가치망
# ==============================================================================
OPS = ['op1', 'op2', 'op3', 'u1', 'sel', 'div', 'nz', 'e0', 'e1']
NLEAF, NPVAR = 10, 8
TOK = {s: i for i, s in enumerate(OPS)}
LEAF0 = len(OPS); PVAR0 = LEAF0 + NLEAF; UNK = PVAR0 + NPVAR; NTOK = UNK + 1
ROLE_STATE, ROLE_GOAL, ROLE_LHS, ROLE_RHS = 0, 1, 2, 3

def anon_map(goal, g):
    m = {}
    for s in leaf_syms(goal) + leaf_syms(g):
        if s not in m: m[s] = len(m)
    return m

def tok_of(sym, am):
    if sym in TOK: return TOK[sym]
    if sym[0] == '?':
        k = 7 if sym == '?f' else int(sym[1:])
        return PVAR0 + min(k, NPVAR - 1)
    k = am.get(sym)
    return LEAF0 + min(k, NLEAF - 1) if k is not None else UNK

@lru_cache(maxsize=100000)
def subtrees(g):
    return frozenset(sub for _, sub in positions(g) if sub[1])

class Batch:
    """여러 그래프를 한 번에 GNN에 넣기 위한 묶음"""
    def __init__(self):
        self.tok, self.role, self.flag, self.parent, self.slot = [], [], [], [], []
        self.graph_of, self.roots, self.n = [], [], 0
    def add(self, g, role, am, goal_subs=None):
        gid = len(self.roots); self.roots.append(self.n)
        stack = [(g, -1, 0)]
        while stack:   # 전위순회 (positions()와 같은 순서)
            node, par, slot = stack.pop()
            me = self.n; self.n += 1
            self.tok.append(tok_of(node[0], am)); self.role.append(role)
            self.flag.append(1 if (goal_subs is not None and node[1] and node in goal_subs) else 0)
            self.parent.append(par); self.slot.append(min(slot, 2)); self.graph_of.append(gid)
            for i in range(len(node[1]) - 1, -1, -1):
                stack.append((node[1][i], me, i))
        return gid
    def tensors(self):
        T = lambda x: torch.tensor(x, dtype=torch.long)
        return T(self.tok), T(self.role), T(self.flag), T(self.parent), T(self.slot), T(self.graph_of), T(self.roots)

class TreeGNN(nn.Module):
    def __init__(self, D, T=3):
        super().__init__()
        self.tok = nn.Embedding(NTOK, D); self.role = nn.Embedding(4, D); self.flag = nn.Embedding(2, D)
        self.up = nn.ModuleList([nn.Linear(D, D) for _ in range(3)])
        self.down = nn.ModuleList([nn.Linear(D, D) for _ in range(3)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(3 * D, D), nn.GELU(), nn.Linear(D, D)) for _ in range(T)])
        self.norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(T)])
        self.read = nn.Linear(2 * D, D)

    def forward(self, tok, role, flag, parent, slot, graph_of, roots):
        h = self.tok(tok) + self.role(role) + self.flag(flag)
        ch = (parent >= 0).nonzero(as_tuple=True)[0]
        par = parent[ch]; sl = slot[ch]
        groups = [(ch[sl == s], par[sl == s]) for s in range(3)]
        for upd, norm in zip(self.upd, self.norm):
            up = torch.zeros_like(h); down = torch.zeros_like(h)
            for s, (c, p) in enumerate(groups):
                if len(c):
                    up = up.index_add(0, p, self.up[s](h[c]))
                    down = down.index_copy(0, c, self.down[s](h[p]))
            h = norm(h + upd(torch.cat([h, up, down], -1)))
        ng = len(roots)
        cnt = torch.zeros(ng).index_add(0, graph_of, torch.ones(len(graph_of))).clamp(min=1)
        mean = torch.zeros(ng, h.shape[1]).index_add(0, graph_of, h) / cnt[:, None]
        g = self.read(torch.cat([h[roots], mean], -1))
        return h, g

class KimCore(nn.Module):
    def __init__(self, D=64):
        super().__init__()
        self.D = D
        self.gnn = TreeGNN(D)
        self.rule_mlp = nn.Sequential(nn.Linear(2 * D + 5, D), nn.GELU(), nn.Linear(D, D))
        self.pol = nn.Sequential(nn.Linear(5 * D, 128), nn.GELU(), nn.Linear(128, 1))
        self.val = nn.Sequential(nn.Linear(3 * D + 2, 128), nn.GELU(), nn.Linear(128, 1))
        for head in (self.pol[-1], self.val[-1]):   # 초기 뇌 = 균일 정책 + 가치 0.5 (대조군과 동일 출발)
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def rule_embs(self, rules):
        b = Batch()
        for r in rules:
            b.add(r.lhs, ROLE_LHS, {}); b.add(r.rhs, ROLE_RHS, {})
        _, g = self.gnn(*b.tensors())
        g = g.view(len(rules), 2 * self.D)
        meta = torch.tensor([[float(KIND_ID[r.kind] == k) for k in range(4)] + [float(bool(r.cond))] for r in rules])
        return self.rule_mlp(torch.cat([g, meta], -1))

    def encode_pairs(self, pairs):
        """pairs: [(state, goal)] -> 노드 임베딩, 상태 벡터, 목표 벡터, 상태별 노드 오프셋"""
        b = Batch(); sidx, gidx, offs = [], [], []
        for st, goal in pairs:
            am = anon_map(goal, st)
            offs.append(b.n); sidx.append(b.add(st, ROLE_STATE, am, subtrees(goal)))
            gidx.append(b.add(goal, ROLE_GOAL, am))
        h, g = self.gnn(*b.tensors())
        return h, g[sidx], g[gidx], offs

    def policy_logits(self, h, gs, gg, offs, items, remb):
        """items: [(state_i, node_idx, rule_idx, bind_tok)] -> 로짓"""
        si = torch.tensor([it[0] for it in items]); ni = torch.tensor([offs[it[0]] + it[1] for it in items])
        ri = torch.tensor([it[2] for it in items]); bt = torch.tensor([it[3] for it in items])
        bemb = self.gnn.tok(bt.clamp(min=0)) * (bt >= 0).float()[:, None]
        x = torch.cat([h[ni], gs[si], gg[si], remb[ri], bemb], -1)
        return self.pol(x).squeeze(-1)

    def value(self, children, goal, n_infl):
        b = Batch(); am_list = []
        for c in children:
            am = anon_map(goal, c); b.add(c, ROLE_STATE, am, subtrees(goal))
        _, gc = self.gnn(*b.tensors())
        b2 = Batch(); b2.add(goal, ROLE_GOAL, anon_map(goal, goal))
        _, ggoal = self.gnn(*b2.tensors())
        gg = ggoal.expand(len(children), -1)
        extra = torch.tensor([[(size(c) - size(goal)) / 10.0, n_infl / 2.0] for c in children])
        return torch.sigmoid(self.val(torch.cat([gc, gg, gc * gg, extra], -1)).squeeze(-1))

    def value_pairs(self, pairs, n_infls):
        h, gs, gg, _ = self.encode_pairs(pairs)
        extra = torch.tensor([[(size(s) - size(g)) / 10.0, ni / 2.0] for (s, g), ni in zip(pairs, n_infls)])
        return torch.sigmoid(self.val(torch.cat([gs, gg, gs * gg, extra], -1)).squeeze(-1))

# ==============================================================================
# 6. 저격 다림질 탐색 (목표 조건부 최우선 탐색)
# ==============================================================================
def bind_tok(bs, goal, g):
    if bs is None: return -1
    return tok_of(bs, anon_map(goal, g))

def search(engine, net, remb, prob, budget, explore=False, allow_infl=True, check_cond=True,
           use_waffles=True, rng=None, record=True):
    start, goal, hyps = prob['start'], prob['goal'], prob['hyps']
    res = dict(ok=False, path=[], expansions=0, steps=[], neg=[])
    if start == goal:
        res['ok'] = True; return res
    max_size = size(start) + CFG['MAX_SIZE_SLACK']
    uid = 0
    heap = [(0.0, uid, start, 0, 0.0, ())]   # (-pri, uid, graph, n_infl, logp, path)
    best_infl = {start: 0}
    visited_states = []
    while heap and res['expansions'] < budget:
        _, _, g, ni, logp, path = heapq.heappop(heap)
        res['expansions'] += 1
        mv = engine.moves(g, goal, hyps, ni, allow_infl, check_cond, use_waffles)
        mv = [m for m in mv if size(m[3]) <= max_size]
        if not mv: continue
        if record: visited_states.append((g, ni))
        for m in mv:
            if m[3] == goal:   # 목표 도달
                full = path + ((g, ni, m[0], m[1], m[2]),)
                res['ok'] = True; res['path'] = full
                if record:
                    res['steps'] = [(st, n_i, (ri, nidx, bs)) for st, n_i, ri, nidx, bs in full]
                return res
        with torch.no_grad():
            h, gs, gg, offs = net.encode_pairs([(g, goal)])
            items = [(0, m[1], m[0], bind_tok(m[2], goal, g)) for m in mv]
            lg = net.policy_logits(h, gs, gg, offs, items, remb)
            if explore:
                lg = lg + 0.7 * torch.from_numpy(rng.gumbel(size=len(mv))).float()
            lp = torch.log_softmax(lg, 0).numpy()
            v = net.value([m[3] for m in mv], goal, ni).numpy()
        gsz = size(goal)
        for k, m in enumerate(mv):
            child = m[3]; is_inf = engine.rules[m[0]].kind == 'infl'
            cni = ni + (1 if is_inf else 0)
            if child in best_infl and best_infl[child] <= cni: continue
            best_infl[child] = cni
            clogp = logp + 0.5 * float(lp[k])
            pri = clogp + 2.0 * math.log(max(float(v[k]), 1e-4)) - 0.35 * max(0, size(child) - gsz) - 0.2 * (len(path) + 1)
            uid += 1
            heapq.heappush(heap, (-pri, uid, child, cni, clogp, path + ((g, ni, m[0], m[1], m[2]),)))
    if record:
        res['neg'] = visited_states[:10]
    return res

# ==============================================================================
# 7. 출제기: 목표에서 거꾸로 걷기 (뻥튀기 역방향 / 축소 / 중립 / 센서 단계)
# ==============================================================================
SEED_LEAVES = ['s_a', 's_b', 's_c']

class Proposer:
    def __init__(self, engine, seed):
        self.e = engine; self.rng = random.Random(seed); self.difficulty = 2
        R = {r.name: r for r in engine.rules[:engine.n_base]}
        self.infl = [r for r in engine.rules[:engine.n_base] if r.kind == 'infl']
        # 축소 단계로 써도 되는 규칙 (= 풀 때 뻥튀기로 되돌릴 수 있는 것)
        self.shrink = [R[n] for n in ["공리0[우측e0소거]", "공리1[좌측e1소거]", "공리3[이중u1소거]", "공리9[우측e1소거]",
                                       "공리2[자기소거]", "공리7[선택기단순화]", "센서0[자기나눗셈|nz]"]]
        # 중립 단계: 풀 때 다른 정방향 공리로 되돌릴 수 있는 것만 (공리8은 역방향이 없어서 제외)
        self.neutral = [R[n] for n in ["공리6[op1대칭교환]", "공리5[가지펼치기]", "공리4[공통가지묶기]"]]

    def rand_goal(self):
        r = self.rng
        def t(d):
            if d == 0 or r.random() < 0.35:
                u = r.random()   # 가끔 항등원이 목표 안에 박혀 있음 -> 풀 때 뻥튀기가 꼭 필요한 문제의 재료
                return E1 if u < 0.12 else (E0 if u < 0.18 else L(r.choice(SEED_LEAVES)))
            op = r.choice(['op1', 'op2', 'op2', 'u1'])
            return N('u1', t(d - 1)) if op == 'u1' else N(op, t(d - 1), t(d - 1))
        return t(2)

    def _apply_random(self, g, rules, hyps, need_leaf_free=False, goal=None):
        """규칙 목록에서 무작위로 하나를 무작위 위치에 적용 (가능한 것 중에서)"""
        cands = []
        pool = leaf_syms(g) or SEED_LEAVES
        for pos, sub in positions(g):
            for r in rules:
                if r.kind == 'infl' and is_var(r.lhs) and size(sub) > CFG['INFL_SUB_MAX']: continue
                b = {}
                if not match(r.lhs, sub, b): continue
                if r.free:
                    for s in pool:
                        bb = dict(b); bb[r.free[0]] = L(s)
                        if all(prove(subst(c, bb), hyps) for c in r.cond):
                            cands.append((r, pos, bb))
                else:
                    if r.cond and not all(prove(subst(c, b), hyps) for c in r.cond): continue
                    if need_leaf_free:   # 축소된 것을 풀 때 뻥튀기로 정확히 복원할 수 있어야 함
                        lost = vars_of(r.lhs) - vars_of(r.rhs)
                        gl = set(leaf_syms(goal))
                        if any(b[v][1] or b[v][0] not in gl for v in lost): continue
                        if r.name.startswith('센서0') and (b['?0'][1] or b['?0'][0] not in gl): continue
                        kept = subst(r.rhs, b)
                        if is_var(r.rhs) and size(kept) > CFG['INFL_SUB_MAX']: continue
                    cands.append((r, pos, b))
        if not cands: return None, None
        r, pos, bb = self.rng.choice(cands)
        return replace_at(g, pos, subst(r.rhs, bb)), r

    def make(self, kind, L_steps, hyps=None):
        r = self.rng
        for _ in range(60):
            goal = self.rand_goal()
            hyps = frozenset([N('nz', L(s)) for s in r.sample(SEED_LEAVES, r.randint(1, 2))]) if hyps is None else hyps
            g, used, shrinks = goal, [], 0
            forced = {'cond': 'cond', 'infl': 'shrink'}.get(kind)
            force_at = r.randint(0, 1) if kind == 'infl' else r.randint(0, max(0, L_steps - 2))
            for step in range(L_steps):
                if forced and step == force_at:
                    typ = forced
                else:
                    u = r.random()
                    typ = 'inflate' if u < 0.6 else ('neutral' if u < 0.85 or shrinks >= CFG['MAX_INFL'] else 'shrink')
                if typ == 'cond':
                    rules = [x for x in self.infl if x.cond]
                    ng, rr = self._apply_random(g, rules, hyps)
                elif typ == 'inflate':
                    ng, rr = self._apply_random(g, self.infl, hyps)
                elif typ == 'shrink':
                    if shrinks >= CFG['MAX_INFL']: continue
                    ng, rr = self._apply_random(g, self.shrink, hyps, need_leaf_free=True, goal=goal)
                    if ng is not None: shrinks += 1
                else:
                    ng, rr = self._apply_random(g, self.neutral, hyps)
                if ng is not None and size(ng) <= 30:
                    g = ng; used.append(rr.name)
            if g == goal or size(g) > 30: continue
            # 센서 문제는 시작 그래프에 div가 '남아 있어야' 진짜로 센서가 필요함 (넣었다 지운 경우 제외)
            # div는 조건부 규칙으로만 없앨 수 있으므로, div가 남은 함정 문제는 nz 가정 없이는 원리상 풀 수 없음
            if kind == 'cond' and 'div' not in show(g): continue
            if kind == 'infl' and shrinks == 0: continue
            return dict(start=g, goal=goal, hyps=hyps, kind=kind, recipe=used)
        return None

    def decoy(self):
        """함정: 센서 단계로 만든 문제에서 nz 가정을 지움 -> 정당한 길이 사라짐"""
        for _ in range(100):
            p = self.make('cond', 2)
            if p is None: continue
            p = dict(p); p['hyps'] = frozenset(); p['kind'] = 'decoy'
            return p
        return None

    def train_problem(self):
        u = self.rng.random()
        kind = 'cond' if u < 0.3 else ('infl' if u < 0.6 else 'plain')
        Ls = self.rng.randint(self.difficulty, self.difficulty + 2)
        return self.make(kind, Ls)

# ==============================================================================
# 8. 수술 보조: 와플 거푸집 (부분 경로 압축) + 재생 검증
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

def replay_ok(engine, lhs, rhs, rel_moves, varmap_inv):
    """와플 검증: 패턴 변수에 서로 다른 새 기호를 넣고, 원래 수순을 그대로 재생해 rhs가 나오는지"""
    for tag in range(2):
        inst = {v: L(f's_{tag}{k}') for k, v in enumerate(sorted(vars_of(lhs)))}
        def fill(g):
            if is_var(g): return inst[g[0]]
            return (g[0], tuple(fill(c) for c in g[1]))
        g = fill(lhs); target = fill(rhs)
        for ri, relpos, bs in rel_moves:
            r = engine.rules[ri]
            if r.cond: return False                     # 조건부 수순은 와플로 굳히지 않음 (조건이 사라지면 위험)
            sub = get_at(g, relpos) if relpos is not None else None
            if sub is None: return False
            b = {}
            if not match(r.lhs, sub, b): return False
            if r.free:
                if bs is None or bs not in varmap_inv: return False
                b[r.free[0]] = inst[varmap_inv[bs]]
            g = replace_at(g, relpos, subst(r.rhs, b))
        if g != target: return False
    return True

def forge_waffles(engine, solved, gen, counter):
    """해결 경로의 연속 구간(2~4수)이 한 가지(subtree) 안에서 일어났으면 그 구간을 규칙 하나로 압축"""
    new_logs = []
    cands = {}
    for prob, path in solved:
        graphs = [st for st, _, _, _, _ in path]
        final = prob['goal']
        seq = []
        for i, (st, ni, ri, nidx, bs) in enumerate(path):
            seq.append((st, ri, node_pos(st, nidx), bs))
        afters = graphs[1:] + [final]
        for i in range(len(seq)):
            for j in range(i + 1, min(len(seq), i + 4)):
                seg = seq[i:j + 1]
                poss = [p for _, _, p, _ in seg]
                pre = []
                for k in range(min(len(p) for p in poss)):
                    if all(p[k] == poss[0][k] for p in poss): pre.append(poss[0][k])
                    else: break
                pre = tuple(pre)
                before = get_at(seg[0][0], pre)
                try:
                    after = get_at(afters[j], pre)
                except (IndexError, TypeError):
                    continue
                if before == after or size(before) > 14: continue
                lhs, rhs, m = generalize_pair(before, after)
                if not vars_of(rhs) <= vars_of(lhs) or is_var(lhs): continue
                if size(rhs) >= size(lhs): continue          # 산을 낮추는 와플만
                rel = [(ri, p[len(pre):], bs) for _, ri, p, bs in seg]
                key = (lhs, rhs)
                if key not in cands:
                    cands[key] = (rel, {s: v for s, v in m.items()})
                counter[key] += 1
    existing = set((r.lhs, r.rhs) for r in engine.rules)
    added = 0
    for key, cnt in sorted(counter.items(), key=lambda kv: -kv[1]):
        if added >= 4 or len(engine.waffles) >= CFG['MAX_WAFFLES']: break
        if cnt < 2 or key in existing or key not in cands: continue
        lhs, rhs = key
        rel, m = cands[key]
        if not replay_ok(engine, lhs, rhs, rel, m):
            new_logs.append(f"  ⛔ [와플 검증 실패, 폐기] {show(lhs)} ==> {show(rhs)}"); continue
        name = f"와플#{len(engine.waffles) + 1}[{show(lhs)}⇒{show(rhs)}]"
        engine.rules.append(Rule(name, lhs, rhs, 'waffle', born=gen))
        existing.add(key); added += 1
        new_logs.append(f"  🧇 [와플 주조] {show(lhs)} ==> {show(rhs)}  (등장 {cnt}회, 재생 검증 통과)")
    return new_logs

# ==============================================================================
# 9. 병렬 실행 (fork 워커가 현재 뇌/규칙을 복사해 감)
# ==============================================================================
_G = {}
def _init_worker():
    torch.set_num_threads(1)

def _job(args):
    prob, budget, explore, flags, seed = args
    eng, net, remb = _G['engine'], _G['net'], _G['remb']
    rng = np.random.default_rng(seed)
    r = search(eng, net, remb, prob, budget, explore=explore, rng=rng, **flags)
    return r

def run_parallel(engine, net, probs, budget, explore=False, flags=None, seed=0):
    flags = flags or {}
    net.eval()
    with torch.no_grad():
        remb = net.rule_embs(engine.rules)
    _G.update(engine=engine, net=net, remb=remb)
    ctx = mp.get_context('fork')
    args = [(p, budget, explore, flags, seed * 100003 + i) for i, p in enumerate(probs)]
    with ctx.Pool(CFG['WORKERS'], initializer=_init_worker) as pool:
        return pool.map(_job, args, chunksize=1)

def evaluate(engine, net, exam, flags=None):
    rs = run_parallel(engine, net, exam, CFG['EVAL_BUDGET'], flags=flags)
    by = collections.defaultdict(lambda: [0, 0]); steps = []
    breaches = []
    for p, r in zip(exam, rs):
        k = p.get('kind', 'A')
        by[k][1] += 1
        if r['ok']:
            by[k][0] += 1
            if k == 'decoy': breaches.append(p)
            else: steps.append(len(r['path']))
    real = [k for k in by if k != 'decoy']
    tot = sum(by[k][0] for k in real); n = sum(by[k][1] for k in real)
    return dict(acc=100.0 * tot / max(1, n), by={k: (v[0], v[1]) for k, v in by.items()},
                steps=float(np.mean(steps)) if steps else 0.0, breaches=len(breaches), results=rs)

# ==============================================================================
# 10. 학습 (전문가 반복: 검증된 경로 모방 + 거리 기반 가치)
# ==============================================================================
class Trainer:
    def __init__(self, net):
        self.net = net
        self.opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        self.pol = collections.deque(maxlen=5000)   # (state, goal, hyps, n_infl, chosen(ri,nidx,bs))
        self.val = collections.deque(maxlen=12000)  # (state, goal, n_infl, target)

    def add(self, prob, res):
        if res['ok']:
            n = len(res['steps'])
            for i, (st, ni, mv) in enumerate(res['steps']):
                self.pol.append((st, prob['goal'], prob['hyps'], ni, mv))
                self.val.append((st, prob['goal'], ni, 0.85 ** (n - i)))
        else:
            for st, ni in res['neg']:
                self.val.append((st, prob['goal'], ni, 0.0))

    def train(self, engine, iters=120, bs=24):
        if len(self.pol) < 16: return 0.0, 0.0
        net = self.net; net.train()
        pol, val = list(self.pol), list(self.val)
        tl = tv = 0.0
        for it in range(iters):
            batch = random.sample(pol, min(bs, len(pol)))
            remb = net.rule_embs(engine.rules)
            pairs = [(st, goal) for st, goal, _, _, _ in batch]
            h, gs, gg, offs = net.encode_pairs(pairs)
            loss_p = 0.0; nvalid = 0
            all_items, seg, tgts = [], [], []
            for bi, (st, goal, hyps, ni, chosen) in enumerate(batch):
                mv = engine.moves(st, goal, hyps, ni)
                mv = [m for m in mv if size(m[3]) <= size(st) + CFG['MAX_SIZE_SLACK'] + 4]
                keys = [(m[0], m[1], m[2]) for m in mv]
                if tuple(chosen) not in keys: continue
                tgts.append(keys.index(tuple(chosen)))
                seg.append((len(all_items), len(mv)))
                all_items += [(bi, m[1], m[0], bind_tok(m[2], goal, st)) for m in mv]
            if seg:
                lg = net.policy_logits(h, gs, gg, offs, all_items, remb)
                C = max(n for _, n in seg)
                S = torch.full((len(seg), C), -1e9)
                for k, (o, n) in enumerate(seg):
                    S[k, :n] = lg[o:o + n]
                loss_p = F.cross_entropy(S, torch.tensor(tgts))
            vb = random.sample(val, min(48, len(val)))
            vp = net.value_pairs([(s, g) for s, g, _, _ in vb], [ni for _, _, ni, _ in vb])
            loss_v = F.binary_cross_entropy(vp.clamp(1e-4, 1 - 1e-4), torch.tensor([t for *_, t in vb], dtype=torch.float))
            loss = (loss_p if seg else 0.0) + loss_v
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); self.opt.step()
            tl += float(loss_p) if seg else 0.0; tv += float(loss_v)
        net.eval()
        return tl / iters, tv / iters

# ==============================================================================
# 11. 시험지
# ==============================================================================
def exam_A():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v46
    random.seed(0)
    ax = v46.create_universal_axiom_system(); eng = v46.DomainFreeRewriteEngine(ax)
    H = v46.build_heldout_exam(ax, eng)
    return [dict(start=from_gnode(p), goal=from_gnode(g), hyps=frozenset(), kind='A') for p, g in H]

def needs_inflation(engine, p, probe_net, probe_remb):
    """뻥튀기 없이 300노드 탐색으로도 못 풀어야 '뻥튀기 필수 문제'로 인정"""
    r = search(engine, probe_net, probe_remb, p, 300, allow_infl=False, record=False)
    return not r['ok']

def exam_B(engine, seed, n_each=20, n_decoy=10, forbid=frozenset()):
    pr = Proposer(engine, seed)
    out, seen = [], set(forbid)
    probe = KimCore(CFG['D'])
    with torch.no_grad():
        probe_remb = probe.rule_embs(engine.rules)
    for kind, lo, hi in [('cond', 3, 6), ('infl', 2, 5), ('deep', 10, 14)]:
        c = 0
        while c < n_each:
            p = pr.make('plain' if kind == 'deep' else kind, pr.rng.randint(lo, hi))
            if p is None or p['start'] in seen: continue
            if kind == 'infl' and not needs_inflation(engine, p, probe, probe_remb): continue
            p['kind'] = kind; seen.add(p['start']); out.append(p); c += 1
    c = 0
    while c < n_decoy:
        p = pr.decoy()
        if p is None or p['start'] in seen: continue
        seen.add(p['start']); out.append(p); c += 1
    return out

# ==============================================================================
# 12. 메인
# ==============================================================================
def park_briefing(prob, path, engine):
    names = [engine.rules[ri].name.split('[')[0] for _, _, ri, _, _ in path]
    return f"🎤 [박통역 브리핑] \"{show(prob['start'])} 를 [{' → '.join(names)}] 로 밀어서 {show(prob['goal'])} 로 폈습니다!\""

def fmt_by(d):
    order = ['A', 'cond', 'infl', 'deep', 'decoy']
    lab = {'A': 'A', 'cond': '센서', 'infl': '뻥튀기', 'deep': '심층', 'decoy': '함정'}
    return ' '.join(f"{lab[k]} {d[k][0]}/{d[k][1]}" for k in order if k in d)

def main():
    random.seed(CFG['SEED']); np.random.seed(CFG['SEED']); torch.manual_seed(CFG['SEED'])
    ax, cond, infl = build_rules()
    engine = Engine(ax, cond, infl)
    control_engine = Engine(ax, cond, infl)
    net = KimCore(CFG['D']); control_net = copy.deepcopy(net)
    trainer = Trainer(net)
    proposer = Proposer(engine, CFG['SEED'])

    A = exam_A()
    B = exam_B(engine, 2024, forbid=frozenset(p['start'] for p in A))
    V = exam_B(engine, 555, n_each=10, n_decoy=0, forbid=frozenset(p['start'] for p in A + B))
    forbidden = set(p['start'] for p in A + B + V)
    nparams = sum(p.numel() for p in net.parameters())

    log("=" * 96)
    log(" [김코어 v5.0] 수술 3종: 센서 다리미(조건부) + 뻥튀기(전략적 팽창) + GNN 눈(노드 단위 저격)")
    log(f"  뇌 파라미터 {nparams:,}개 | 기본 규칙 {engine.n_base}개 (공리 10, 센서 2, 뻥튀기 8)")
    log(f"  시험지 A(v4.6 숨김) {len(A)} | 시험지 B(v5 신규) {len(B)} (센서/뻥튀기/심층 20씩 + 함정 10) | 검증지 {len(V)}")
    log("=" * 96)

    t = time.time()
    cA = evaluate(control_engine, control_net, A)
    cB = evaluate(control_engine, control_net, B)
    cV = evaluate(control_engine, control_net, V)
    log(f"\n📊 [대조군 = 같은 v5 엔진, 학습 0회, 와플 0개]  ({time.time() - t:.0f}s)")
    log(f"   시험지 A: {cA['acc']:.1f}% ({cA['steps']:.2f}수) | 시험지 B: {cB['acc']:.1f}% [{fmt_by(cB['by'])}] | 검증지 {cV['acc']:.1f}%")
    log(f"   함정 돌파(BREACH): {cB['breaches']}건")

    metrics = dict(config=CFG, control=dict(A=cA['acc'], B=cB['acc'], B_by=cB['by'], V=cV['acc'], A_steps=cA['steps']), gens=[])
    best_v, best_state = cV['acc'], copy.deepcopy(net.state_dict())
    counter = collections.Counter()
    last_A = last_B = None

    for gen in range(1, CFG['GENS'] + 1):
        tg = time.time()
        probs = []
        while len(probs) < CFG['PROBS_PER_GEN']:
            p = proposer.train_problem()
            if p is not None and p['start'] not in forbidden: probs.append(p)
        rs = run_parallel(engine, net, probs, CFG['TRAIN_BUDGET'], explore=True, seed=gen)
        solved = []
        infl_used = cond_used = 0
        best = None
        for p, r in zip(probs, rs):
            trainer.add(p, r)
            if r['ok'] and r['path']:
                solved.append((p, r['path']))
                kinds = [engine.rules[ri].kind for _, _, ri, _, _ in r['path']]
                infl_used += 'infl' in kinds; cond_used += 'cond' in kinds
                for _, _, ri, _, _ in r['path']: engine.rules[ri].uses += 1
                if best is None or len(r['path']) > len(best[1]): best = (p, r['path'])
        rate = len(solved) / len(probs)
        pl, vl = trainer.train(engine)
        wlogs = forge_waffles(engine, solved, gen, counter)

        v = evaluate(engine, net, V)
        rollback = False
        if v['acc'] > best_v:
            best_v, best_state = v['acc'], copy.deepcopy(net.state_dict())
        elif v['acc'] < best_v - 15:
            net.load_state_dict(best_state); rollback = True

        rec = dict(gen=gen, train_rate=rate, difficulty=proposer.difficulty, valid=v['acc'], waffles=len(engine.waffles),
                   infl_paths=infl_used, cond_paths=cond_used, policy_loss=pl, value_loss=vl, rollback=rollback,
                   secs=time.time() - tg)
        if gen % CFG['EVAL_EVERY'] == 0 or gen == CFG['GENS']:
            last_A = evaluate(engine, net, A); last_B = evaluate(engine, net, B)
            rec.update(A=last_A['acc'], A_steps=last_A['steps'], B=last_B['acc'], B_by=last_B['by'], breaches=last_B['breaches'])
        metrics['gens'].append(rec)

        log(f"\n{'-' * 96}")
        log(f"[제 {gen:02d} 세대 | 난이도 Lv.{proposer.difficulty} | 와플 {len(engine.waffles)}개] 연습 해결 {len(solved)}/{len(probs)} "
            f"(뻥튀기 사용 {infl_used}, 센서 사용 {cond_used}) | 검증지 {v['acc']:.1f}%{' ↩️ 롤백' if rollback else ''} | {rec['secs']:.0f}s")
        if 'A' in rec:
            log(f" 🔒 시험지 A (v4.6 숨김 40): 대조군 {cA['acc']:.1f}%  vs  김코어 v5 {last_A['acc']:.1f}% ({last_A['steps']:.2f}수)  "
                f"[{last_A['acc'] - cA['acc']:+.1f}%p]")
            log(f" 🔒 시험지 B (v5 신규 60):  대조군 {cB['acc']:.1f}%  vs  김코어 v5 {last_B['acc']:.1f}%  [{last_B['acc'] - cB['acc']:+.1f}%p]  "
                f"| {fmt_by(last_B['by'])} | 함정 돌파 {last_B['breaches']}건")
        for l in wlogs: log(l)
        if best:
            log(f"  └─ 최장 다림질 {len(best[1])}수 [{best[0]['kind']}]")
            log(f"  {park_briefing(best[0], best[1], engine)}")

        if rate >= 0.7 and proposer.difficulty < 10: proposer.difficulty += 1
        elif rate < 0.3 and proposer.difficulty > 2: proposer.difficulty -= 1
        with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)

    # ---------------- 최종 + 수술별 기여 분석 (ablation)
    log("\n" + "=" * 96)
    log(" 🔬 [수술별 기여 분석] 최종 김코어에서 하나씩 떼어보기 (시험지 B)")
    noinfl = evaluate(engine, net, B, flags=dict(allow_infl=False))
    nowaf = evaluate(engine, net, B, flags=dict(use_waffles=False))
    nosensor = evaluate(control_engine, control_net, [p for p in B if p['kind'] == 'decoy'], flags=dict(check_cond=False))
    log(f"   최종 김코어 v5         : {last_B['acc']:.1f}%  [{fmt_by(last_B['by'])}]")
    log(f"   - 뻥튀기 끄면          : {noinfl['acc']:.1f}%  [{fmt_by(noinfl['by'])}]")
    log(f"   - 와플 끄면            : {nowaf['acc']:.1f}%  [{fmt_by(nowaf['by'])}]")
    log(f"   센서 없는 다리미(조건 무시)로 함정 10문항: {nosensor['by'].get('decoy', (0, 0))[0]}/10 '풀었다고 거짓말'  "
        f"vs  센서 다리미: {last_B['breaches']}/10")
    metrics['ablation'] = dict(full=last_B['acc'], no_infl=noinfl['acc'], no_waffle=nowaf['acc'],
                               sensorless_decoy_fooled=nosensor['by'].get('decoy', (0, 0))[0], breaches=last_B['breaches'])

    log("\n" + "=" * 96)
    log(" 🏆 [최종 성적표]")
    log("=" * 96)
    log(f"   시험지 A (v4.6 숨김 40): 대조군 {cA['acc']:.1f}%  →  김코어 v5 {last_A['acc']:.1f}%   (참고: v4.6 김코어 80~85%)")
    log(f"   시험지 B (v5 신규 60):  대조군 {cB['acc']:.1f}%  →  김코어 v5 {last_B['acc']:.1f}%")
    log(f"     세부: 대조군 [{fmt_by(cB['by'])}]  /  김코어 [{fmt_by(last_B['by'])}]")
    log(f"   함정 돌파 {last_B['breaches']}건 | 발명 와플 {len(engine.waffles)}개 | 총 {time.time() - _T0:.0f}s")
    for g in metrics['gens']:
        if 'A' in g:
            bar = '█' * int(g['B'] // 5)
            log(f"   Gen {g['gen']:02d} | A {g['A']:5.1f}% | B {g['B']:5.1f}% [{bar:<20}] | 검증 {g['valid']:5.1f}% | 와플 {g['waffles']}")
    with open(os.path.join(OUT_DIR, 'waffles.txt'), 'w') as f:
        for r in engine.waffles:
            f.write(f"{r.name}\tuses={r.uses}\tborn=gen{r.born}\n")
    metrics['final'] = dict(A=last_A['acc'], B=last_B['acc'], B_by=last_B['by'], waffles=len(engine.waffles))
    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)
    try:
        plot(metrics, os.path.join(OUT_DIR, 'progress.png'))
    except Exception as e:
        log('plot error', e)

def plot(m, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=SURF)
    G = [g for g in m['gens'] if 'A' in g]
    for a in ax:
        a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.6); a.spines[['top', 'right']].set_visible(False); a.set_ylim(0, 100)
    x = [g['gen'] for g in G]
    for a, key, title in [(ax[0], 'A', 'Exam A (v4.6 hidden, 40)'), (ax[1], 'B', 'Exam B (v5 new, 60)')]:
        a.plot(x, [g[key] for g in G], color=BLUE, lw=2, marker='o', ms=4, label='KimCore v5 (trained)')
        a.axhline(m['control'][key], color=ORANGE, lw=2, ls='--', label='control (same engine, untrained)')
        if key == 'A':
            a.axhspan(80, 85, color=AQUA, alpha=0.18, label='v4.6 trained (80-85%)')
        a.set_title(title, loc='left', color=INK, fontweight='bold'); a.set_xlabel('generation', color=INK2); a.set_ylabel('% solved', color=INK2)
        a.legend(frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=SURF); plt.close(fig)

if __name__ == '__main__':
    main()
