import os
import json
import math
import time
import random
from collections import Counter
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==============================================================================
# 0. 실행 설정
# ==============================================================================
MAX_GENS = int(os.environ.get("RSI_MAX_GENS", "80"))
TIME_LIMIT_SEC = float(os.environ.get("RSI_TIME_LIMIT", str(25 * 60)))
PROBS_PER_GEN = int(os.environ.get("RSI_PROBS", "25"))
EVAL_EVERY = int(os.environ.get("RSI_EVAL_EVERY", "5"))
SEARCH_STEPS = 220
EVAL_STEPS = 20  # 홀드아웃은 탐색 예산을 빡빡하게: 학습 안 한 뇌가 약 37%만 푸는 수준
MAX_PATH = 16
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

# ==============================================================================
# 1. 순수 수식 트리 (Symbolic AST) 정의
# ==============================================================================
class Node:
    def __init__(self, op, children=None, val=None):
        self.op = op
        self.children = children if children is not None else []
        self.val = val

    def clone(self):
        return Node(self.op, [c.clone() for c in self.children], self.val)

    def __eq__(self, other):
        if not isinstance(other, Node): return False
        return self.op == other.op and self.val == other.val and self.children == other.children

    def __hash__(self):
        return hash((self.op, self.val, tuple(self.children)))

    def __repr__(self):
        if self.op == 'num': return str(self.val)
        if self.op == 'var': return str(self.val)
        if self.op == '=': return f"{self.children[0]} = {self.children[1]}"
        if self.op == 'or': return "  또는  ".join(str(c) for c in self.children)
        if self.op == '^': return f"({self.children[0]})^{self.children[1]}"
        if self.op == '*':
            l = f"({self.children[0]})" if self.children[0].op == '+' else f"{self.children[0]}"
            r = f"({self.children[1]})" if self.children[1].op == '+' else f"{self.children[1]}"
            if self.children[0].op == 'num' and self.children[1].op == 'var':
                return f"{self.children[0].val}{self.children[1].val}"
            return f"{l}*{r}"
        if self.op == '+':
            res = str(self.children[0])
            for c in self.children[1:]:
                s = str(c)
                if s.startswith('-'): res += f" - {s[1:]}"
                else: res += f" + {s}"
            return res
        return "?"

def Num(n): return Node('num', val=n)
def Var(v='x'): return Node('var', val=v)
def Add(*args):
    flat = []
    for a in args:
        if a.op == '+': flat.extend([c.clone() for c in a.children])
        else: flat.append(a.clone())
    return Node('+', flat) if len(flat) > 1 else flat[0]
def Mul(a, b): return Node('*', [a.clone(), b.clone()])
def Pow(a, b): return Node('^', [a.clone(), b.clone()])
def Eq(a, b): return Node('=', [a.clone(), b.clone()])
def Or(*eqs):
    flat = []
    for e in eqs:
        if e.op == 'or': flat.extend([c.clone() for c in e.children])
        else: flat.append(e.clone())
    uniq = []
    for e in flat:
        if e not in uniq: uniq.append(e)
    return Node('or', uniq) if len(uniq) > 1 else uniq[0]

X = Var('x')

def terms(node):
    return [c for c in node.children] if node.op == '+' else [node]

def mono(t):
    """항 분류: ('x2', 계수) / ('x1', 계수) / ('c', 값) / None"""
    if t.op == 'num': return ('c', t.val)
    if t.op == 'var': return ('x1', 1)
    if t.op == '^' and t.children[0] == X and t.children[1] == Num(2): return ('x2', 1)
    if t.op == '*':
        a, b = t.children
        if a == X and b == X: return ('x2', 1)
        if a.op == 'num' and b == X: return ('x1', a.val)
        if b.op == 'num' and a == X: return ('x1', b.val)
        if a.op == 'num' and b.op == '^' and b.children[0] == X and b.children[1] == Num(2): return ('x2', a.val)
    return None

def make_mono(kind, coef):
    if kind == 'x2': return Pow(X, Num(2)) if coef == 1 else Mul(Num(coef), Pow(X, Num(2)))
    if kind == 'x1': return Mul(Num(coef), X)
    return Num(coef)

# ==============================================================================
# 2. 검증기 (Lean 역할): SymPy 정확 검증. 신경망 0개.
#    모든 단계는 해집합을 보존해야 하고, 최종 답은 원래 방정식의 해집합과 정확히 같아야 한다.
# ==============================================================================
SX = sp.Symbol('x')

def to_sp(n):
    if n.op == 'num': return sp.Integer(n.val)
    if n.op == 'var': return SX
    if n.op == '+': return sp.Add(*[to_sp(c) for c in n.children])
    if n.op == '*': return sp.Mul(*[to_sp(c) for c in n.children])
    if n.op == '^': return sp.Pow(to_sp(n.children[0]), to_sp(n.children[1]))
    raise ValueError(n.op)

def solution_set(state):
    if state.op == 'or':
        return sp.Union(*[solution_set(c) for c in state.children])
    lhs, rhs = state.children
    return sp.solveset(sp.expand(to_sp(lhs) - to_sp(rhs)), SX, domain=sp.S.Reals)

class ExactVerifier:
    def __init__(self):
        self.checks = 0
        self.rejections = 0

    def verify(self, problem, states, final_state):
        """problem → states... → final_state 경로 전체를 검사. 하나라도 틀리면 거부."""
        self.checks += 1
        try:
            target = solution_set(problem)
            prev = target
            for st in states[1:] + [final_state]:
                cur = solution_set(st)
                if cur != prev:
                    self.rejections += 1
                    return False, "단계가 해집합을 바꿈"
                prev = cur
            answers = set()
            leaves = final_state.children if final_state.op == 'or' else [final_state]
            for leaf in leaves:
                if not (leaf.children[0] == X and leaf.children[1].op == 'num'):
                    self.rejections += 1
                    return False, "최종 형태가 x = 수 가 아님"
                answers.add(sp.Integer(leaf.children[1].val))
            if sp.FiniteSet(*answers) != target:
                self.rejections += 1
                return False, "최종 답이 진짜 해와 다름"
            return True, "ok"
        except Exception as e:
            self.rejections += 1
            return False, f"검증 예외: {e}"

# ==============================================================================
# 3. 일반화된 대수 공리 (모두 해집합 보존)
# ==============================================================================
class PureAlgebraEnv:
    def __init__(self):
        self.atomic_axioms = [
            ("공리0: 상수 산술 계산", self.axiom_eval_constants),
            ("공리1: 지수 정의 풀어쓰기 (A^2 -> A*A)", self.axiom_expand_power),
            ("공리2: 지수 정의 묶기 (A*A -> A^2)", self.axiom_fold_power),
            ("공리3: 일차항 쪼개기 (bx -> px + qx, p+q=b, pq=c)", self.axiom_ac_split),
            ("공리4: 상수항을 옆 계수로 인수분해 (c -> q*(c/q))", self.axiom_factor_const),
            ("공리5: 앞부분 분배법칙 묶기", self.axiom_factor_front),
            ("공리6: 뒷부분 분배법칙 묶기", self.axiom_factor_back),
            ("공리7: 영인자 법칙 (A*B=0 -> A=0 또는 B=0)", self.axiom_zero_product),
            ("공리8: 일차식 정리 (x + c = r -> x = r - c)", self.axiom_isolate),
            ("공리9: 우변을 좌변으로 이항", self.axiom_move_left),
            ("공리10: 공통 계수로 양변 나누기", self.axiom_divide_gcd),
            ("공리11: 양변 바꾸기", self.axiom_swap),
            ("공리12: 분배법칙 풀기 (A*(B+C) -> A*B + A*C)", self.axiom_distribute),
        ]
        self.invented_macros = []

    @property
    def total_actions(self):
        return len(self.atomic_axioms) + len(self.invented_macros)

    def action_name(self, a):
        return self.atomic_axioms[a][0] if a < len(self.atomic_axioms) else self.invented_macros[a - len(self.atomic_axioms)][0]

    # ---- 공리 구현 (방정식 하나 eq 에 적용) ----
    def axiom_eval_constants(self, eq):
        def _eval(node):
            node = node.clone()
            node.children = [_eval(c) for c in node.children]
            if node.op == '*' and node.children[0].op == 'num' and node.children[1].op == 'num':
                return Num(node.children[0].val * node.children[1].val)
            if node.op == '+':
                nums = [c.val for c in node.children if c.op == 'num']
                others = [c for c in node.children if c.op != 'num']
                if len(nums) > 1 or (len(nums) == 1 and nums[0] == 0 and len(others) > 0):
                    s = sum(nums)
                    if s != 0 or not others:
                        others.append(Num(s))
                    return Add(*others) if len(others) > 1 else others[0]
            return node
        return _eval(eq)

    def axiom_expand_power(self, eq):
        def _exp(node):
            if node.op == '^' and node.children[1] == Num(2):
                base = node.children[0].clone()
                return Mul(base, base)
            return Node(node.op, [_exp(c) for c in node.children], node.val)
        return Eq(_exp(eq.children[0]), eq.children[1].clone())

    def axiom_fold_power(self, eq):
        def _fold(node):
            if node.op == '*' and node.children[0] == node.children[1]:
                return Pow(node.children[0].clone(), Num(2))
            return Node(node.op, [_fold(c) for c in node.children], node.val)
        return Eq(_fold(eq.children[0]), eq.children[1].clone())

    def axiom_ac_split(self, eq):
        lhs, rhs = eq.children
        if rhs != Num(0): return eq
        ms = [mono(t) for t in terms(lhs)]
        if any(m is None for m in ms): return eq
        kinds = [m[0] for m in ms]
        if kinds.count('x2') != 1 or kinds.count('c') != 1 or kinds.count('x1') > 1: return eq
        a = [m[1] for m in ms if m[0] == 'x2'][0]
        if a != 1: return eq
        b = sum(m[1] for m in ms if m[0] == 'x1')
        c = [m[1] for m in ms if m[0] == 'c'][0]
        for p in range(-abs(c) - 1, abs(c) + 2):
            q = b - p
            if p * q == c and p != 0 and q != 0:
                return Eq(Add(Pow(X, Num(2)), make_mono('x1', p), make_mono('x1', q), Num(c)), Num(0))
        return eq

    def axiom_factor_const(self, eq):
        lhs = eq.children[0]
        if lhs.op != '+': return eq
        ch = [c.clone() for c in lhs.children]
        for i in range(1, len(ch)):
            if ch[i].op == 'num' and ch[i].val != 0:
                m = mono(ch[i - 1])
                if m and m[0] == 'x1' and m[1] != 0 and ch[i].val % m[1] == 0:
                    ch[i] = Mul(Num(m[1]), Num(ch[i].val // m[1]))
                    return Eq(Add(*ch), eq.children[1].clone())
        return eq

    def _extract_common(self, term1, term2):
        if term1.op == '*' and term2.op == '*':
            a, b = term1.children
            c, d = term2.children
            if a == c: return Mul(a, Add(b, d))
            if b == d: return Mul(Add(a, c), b)
            if a == d: return Mul(a, Add(b, c))
            if b == c: return Mul(b, Add(a, d))
        return None

    def axiom_factor_front(self, eq):
        lhs = eq.children[0].clone()
        if lhs.op == '+' and len(lhs.children) >= 2:
            factored = self._extract_common(lhs.children[0], lhs.children[1])
            if factored:
                rest = lhs.children[2:]
                return Eq(Add(factored, *rest), eq.children[1].clone())
        return eq

    def axiom_factor_back(self, eq):
        lhs = eq.children[0].clone()
        if lhs.op == '+' and len(lhs.children) >= 2:
            factored = self._extract_common(lhs.children[-2], lhs.children[-1])
            if factored:
                rest = lhs.children[:-2]
                return Eq(Add(*rest, factored), eq.children[1].clone())
        return eq

    def axiom_zero_product(self, eq):
        lhs, rhs = eq.children
        if rhs != Num(0): return eq
        if lhs.op == '^' and lhs.children[1] == Num(2):
            return Eq(lhs.children[0].clone(), Num(0))
        if lhs.op == '*':
            a, b = lhs.children
            if a.op == 'num' or b.op == 'num': return eq
            if a == b: return Eq(a.clone(), Num(0))
            return Or(Eq(a, Num(0)), Eq(b, Num(0)))
        return eq

    def axiom_isolate(self, eq):
        lhs, rhs = eq.children
        if rhs.op != 'num': return eq
        if lhs.op == '+' and len(lhs.children) == 2:
            t0, t1 = lhs.children
            if t0 == X and t1.op == 'num': return Eq(X, Num(rhs.val - t1.val))
            if t1 == X and t0.op == 'num': return Eq(X, Num(rhs.val - t0.val))
        m = mono(lhs)
        if m and m[0] == 'x1' and m[1] not in (0, 1) and rhs.val % m[1] == 0:
            return Eq(X, Num(rhs.val // m[1]))
        return eq

    def axiom_move_left(self, eq):
        lhs, rhs = eq.children
        if rhs.op != 'num' or rhs.val == 0: return eq
        return self.axiom_eval_constants(Eq(Add(lhs, Num(-rhs.val)), Num(0)))

    def axiom_divide_gcd(self, eq):
        lhs, rhs = eq.children
        if rhs != Num(0): return eq
        if lhs.op == '*' and lhs.children[0].op == 'num' and lhs.children[0].val != 0 and lhs.children[1].op != 'num':
            return Eq(lhs.children[1].clone(), Num(0))
        ms = [mono(t) for t in terms(lhs)]
        if any(m is None for m in ms) or len(ms) < 2: return eq
        g = 0
        for _, c in ms: g = math.gcd(g, abs(c))
        if g <= 1: return eq
        return Eq(Add(*[make_mono(k, c // g) for k, c in ms]), Num(0))

    def axiom_swap(self, eq):
        lhs, rhs = eq.children
        if lhs.op == 'num' and rhs.op != 'num': return Eq(rhs, lhs)
        return eq

    def axiom_distribute(self, eq):
        def _dist(node, done):
            if done[0]: return node.clone()
            if node.op == '*':
                a, b = node.children
                if b.op == '+':
                    done[0] = True
                    return Add(*[Mul(a, t) for t in b.children])
                if a.op == '+':
                    done[0] = True
                    return Add(*[Mul(t, b) for t in a.children])
            new = [_dist(c, done) for c in node.children]
            if node.op == '+': return Add(*new)
            return Node(node.op, new, node.val)
        done = [False]
        return Eq(_dist(eq.children[0], done), eq.children[1].clone())

    # ---- 상태(방정식 또는 갈래들)에 행동 적용 ----
    def leaf_solved(self, eq):
        return eq.op == '=' and eq.children[0] == X and eq.children[1].op == 'num'

    def is_solved(self, state):
        if state.op == 'or': return all(self.leaf_solved(c) for c in state.children)
        return self.leaf_solved(state)

    def _apply_atomic(self, state, a):
        if state.op == 'or':
            ch = [c.clone() for c in state.children]
            for i, c in enumerate(ch):
                if not self.leaf_solved(c):
                    ch[i] = self.atomic_axioms[a][1](c)
                    return Or(*ch)
            return state
        return self.atomic_axioms[a][1](state)

    def apply_action(self, state, act_idx):
        if act_idx < len(self.atomic_axioms):
            return self._apply_atomic(state, act_idx)
        curr = state.clone()
        for a in self.invented_macros[act_idx - len(self.atomic_axioms)][1]:
            curr = self._apply_atomic(curr, a)
        return curr

# ==============================================================================
# 4. 자율 출제기: 답(근)을 먼저 정하고 위장된 방정식을 거꾸로 만든다
#    난이도 조절기가 해결률 띠(40~80%)를 유지하도록 스스로 손잡이를 돌린다
# ==============================================================================
class SelfProposer:
    KNOBS = [("근 범위 R", 2, 12, 1), ("서로 다른 근 확률", 0.0, 1.0, 0.25),
             ("위장 단계", 0, 4, 1), ("배율 최대", 1, 4, 1)]

    def __init__(self):
        self.state = {name: lo for name, lo, hi, st in self.KNOBS}
        self.turn = 0
        self.last_bump = None
        self.frontier = []
        self.seen = set()

    def difficulty_index(self):
        tot = 0.0
        for name, lo, hi, st in self.KNOBS:
            tot += (self.state[name] - lo) / (hi - lo)
        return tot / len(self.KNOBS)

    def _make(self, r1, r2, disguise, scale):
        p, q = -r1, -r2
        b, c = p + q, p * q
        if disguise == 0:
            if r1 == r2: return Eq(Pow(Add(X, Num(p)), Num(2)), Num(0))
            return Eq(Mul(Add(X, Num(p)), Add(X, Num(q))), Num(0))
        if disguise == 1:
            return Eq(Add(Mul(X, Add(X, Num(p))), Mul(Num(q), Add(X, Num(p)))), Num(0))
        quad = [Pow(X, Num(2))] + ([make_mono('x1', b)] if b != 0 else []) + [Num(c)]
        if disguise == 2:
            return Eq(Add(*quad), Num(0))
        if disguise == 3:
            m = random.choice([v for v in range(-9, 10) if v != 0])
            quad[-1] = Num(c + m)
            eq = Eq(Add(*quad), Num(m))
            return Eq(eq.children[1], eq.children[0]) if random.random() < 0.3 else eq
        s = random.randint(2, max(2, scale))
        sq = [make_mono('x2', s)] + ([make_mono('x1', s * b)] if b != 0 else []) + [Num(s * c)]
        if random.random() < 0.5:
            m = random.choice([v for v in range(-9, 10) if v != 0])
            sq[-1] = Num(s * c + m)
            return Eq(Add(*sq), Num(m))
        return Eq(Add(*sq), Num(0))

    def sample(self, verifier_target_check=True):
        if self.frontier and random.random() < 0.2:
            return self.frontier.pop(random.randrange(len(self.frontier)))
        R = int(self.state["근 범위 R"])
        nz = [v for v in range(-R, R + 1) if v != 0]
        for _ in range(20):
            r1 = random.choice(nz)
            r2 = random.choice([v for v in nz if v != r1]) if random.random() < self.state["서로 다른 근 확률"] else r1
            disguise = random.randint(0, int(self.state["위장 단계"]))
            scale = int(self.state["배율 최대"])
            if disguise == 4 and scale < 2: disguise = 3
            prob = self._make(r1, r2, disguise, scale)
            key = str(prob)
            truth = sp.FiniteSet(sp.Integer(r1), sp.Integer(r2))
            if verifier_target_check and solution_set(prob) != truth:
                continue
            if key in self.seen and random.random() < 0.7:
                continue
            self.seen.add(key)
            return prob
        return prob

    def adapt(self, rate):
        if rate >= 80.0:
            for k in range(len(self.KNOBS)):
                name, lo, hi, st = self.KNOBS[(self.turn + k) % len(self.KNOBS)]
                if self.state[name] < hi:
                    self.state[name] = min(hi, self.state[name] + st)
                    self.last_bump = (name, st)
                    self.turn = (self.turn + k + 1) % len(self.KNOBS)
                    return f"난이도 ↑ {name} → {self.state[name]}"
            return "난이도 최대"
        if rate < 40.0 and self.last_bump:
            name, st = self.last_bump
            lo = [k for k in self.KNOBS if k[0] == name][0][1]
            self.state[name] = max(lo, self.state[name] - st)
            self.last_bump = None
            return f"난이도 ↓ {name} → {self.state[name]}"
        return "유지"

def build_heldout(n=30, seed=777):
    rng = random.Random(seed)
    saved = random.getstate()
    random.seed(seed)
    prop = SelfProposer()
    for name, lo, hi, st in SelfProposer.KNOBS: prop.state[name] = hi
    out = [prop.sample() for _ in range(n)]
    random.setstate(saved)
    return out

# ==============================================================================
# 5. 특징 추출 & 확장형 추론 뇌
# ==============================================================================
N_FEATS = 12

def ast_to_feature_vector(state):
    leaves = state.children if state.op == 'or' else [state]
    env_unsolved = [l for l in leaves if not (l.children[0] == X and l.children[1].op == 'num')]
    eq = env_unsolved[0] if env_unsolved else leaves[0]
    s = str(eq)
    lhs, rhs = eq.children

    def count_op(node, op):
        c = 1 if node.op == op else 0
        return c + sum(count_op(ch, op) for ch in node.children)

    def max_depth(node):
        if not node.children: return 1
        return 1 + max(max_depth(ch) for ch in node.children)

    return [
        count_op(lhs, '^') / 3.0,
        count_op(lhs, '*') / 5.0,
        count_op(lhs, '+') / 5.0,
        (len(lhs.children) if lhs.op == '+' else 1) / 5.0,
        max_depth(lhs) / 5.0,
        1.0 if lhs.op == '^' else 0.0,
        1.0 if lhs.op == '*' else 0.0,
        1.0 if lhs.op == 'var' else 0.0,
        float(rhs.val) / 20.0 if rhs.op == 'num' else 0.5,
        len(s) / 40.0,
        len(leaves) / 2.0,
        len(env_unsolved) / 2.0,
    ]

class ASTReasoningBrain(nn.Module):
    def __init__(self, num_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATS, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(h + self.fc2(h))
        return self.policy_head(h), torch.sigmoid(self.value_head(h))

    def expand_brain(self, new_num_actions, device):
        old_w = self.policy_head.weight.data
        old_b = self.policy_head.bias.data
        old_n = old_w.size(0)
        if new_num_actions <= old_n: return
        new_head = nn.Linear(self.fc2.out_features, new_num_actions).to(device)
        new_head.weight.data[:old_n] = old_w
        new_head.bias.data[:old_n] = old_b
        new_head.bias.data[old_n:] = 0.8
        self.policy_head = new_head

# ==============================================================================
# 6. 탐색기 (Best-First) & 복리 개념 압축기
# ==============================================================================
def mcts_solve(env, brain, start_eq, max_steps=250, device='cpu'):
    """반환: (성공여부, 경로[(상태, 행동)], 최종상태, 펼친 노드 수)"""
    brain.eval()
    queue = [(0.0, 0, start_eq, [])]
    visited = {str(start_eq)}
    uid = 0
    expanded = 0

    for _ in range(max_steps):
        if not queue: break
        queue.sort(key=lambda x: x[0], reverse=True)
        _, _, curr_eq, path = queue.pop(0)
        expanded += 1

        if env.is_solved(curr_eq):
            return True, path, curr_eq, expanded
        if len(path) >= MAX_PATH:
            continue

        feat = torch.tensor([ast_to_feature_vector(curr_eq)], dtype=torch.float, device=device)
        with torch.no_grad():
            logits, _ = brain(feat)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        children = []
        for act in range(env.total_actions):
            next_eq = env.apply_action(curr_eq, act)
            s_next = str(next_eq)
            if s_next == str(curr_eq) or len(s_next) > 90 or s_next in visited:
                continue
            visited.add(s_next)
            children.append((act, next_eq))
        if not children:
            continue
        feats = torch.tensor([ast_to_feature_vector(n) for _, n in children], dtype=torch.float, device=device)
        with torch.no_grad():
            _, vals = brain(feats)
        for (act, next_eq), val in zip(children, vals.squeeze(1).tolist()):
            uid += 1
            leaves = next_eq.children if next_eq.op == 'or' else [next_eq]
            lhs_terms = sum(len(l.children[0].children) if l.children[0].op == '+' else 1 for l in leaves)
            compression_bonus = (5 - lhs_terms) * 0.15
            score = val * 2.0 + probs[act] * 1.5 + compression_bonus - (len(path) * 0.2) + random.uniform(0, 0.05)
            queue.append((score, uid, next_eq, path + [(curr_eq, act)]))

    return False, [], None, expanded

def compress_and_invent_skill(env, success_paths):
    seq_counter = Counter()
    n_atomic = len(env.atomic_axioms)
    for path in success_paths:
        acts = [a for _, a in path]
        for i in range(len(acts) - 1):
            a1, a2 = acts[i], acts[i + 1]
            raw1 = [a1] if a1 < n_atomic else env.invented_macros[a1 - n_atomic][1]
            raw2 = [a2] if a2 < n_atomic else env.invented_macros[a2 - n_atomic][1]
            combined = tuple(raw1 + raw2)
            if 2 <= len(combined) <= 8:
                seq_counter[combined] += 1

    existing = {tuple(m[1]) for m in env.invented_macros}
    for seq, count in seq_counter.most_common(5):
        if count >= 4 and seq not in existing:
            skill_id = len(env.invented_macros) + 1
            skill_name = f"발명스킬#{skill_id}[공리{list(seq)}압축]"
            env.invented_macros.append((skill_name, list(seq)))
            return skill_name, list(seq)
    return None, None

def atomic_len(env, path):
    n_atomic = len(env.atomic_axioms)
    return sum(1 if a < n_atomic else len(env.invented_macros[a - n_atomic][1]) for _, a in path)

def evaluate(env, brain, problems, verifier, device):
    solved, nodes, steps = 0, [], []
    for prob in problems:
        ok, path, final, expanded = mcts_solve(env, brain, prob, max_steps=EVAL_STEPS, device=device)
        if ok:
            good, _ = verifier.verify(prob, [s for s, _ in path], final)
            if good:
                solved += 1
                nodes.append(expanded)
                steps.append(len(path))
    rate = 100.0 * solved / len(problems)
    return rate, (sum(nodes) / len(nodes) if nodes else float('nan')), (sum(steps) / len(steps) if steps else float('nan'))

def policy_entropy(brain, memory, device):
    if not memory: return float('nan')
    batch = random.sample(memory, min(len(memory), 256))
    feats = torch.tensor([x[0] for x in batch], dtype=torch.float, device=device)
    with torch.no_grad():
        logits, _ = brain(feats)
        p = F.softmax(logits, dim=-1)
        ent = -(p * torch.log(p + 1e-9)).sum(-1).mean().item()
    return ent / math.log(logits.size(-1))

# ==============================================================================
# 7. 메인 RSI 루프: 자율 출제 → 탐색 → 정확 검증 → 학습 → 스킬 발명 → 모니터링
# ==============================================================================
def main():
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = PureAlgebraEnv()
    verifier = ExactVerifier()
    proposer = SelfProposer()
    brain = ASTReasoningBrain(env.total_actions).to(device)
    optimizer = optim.AdamW(brain.parameters(), lr=0.004)
    control_init = ASTReasoningBrain(env.total_actions).to(device)
    control_init.load_state_dict(brain.state_dict())
    heldout = build_heldout()

    memory, anchor = [], []
    metrics = []
    best_heldout, best_state = -1.0, None

    print("==========================================================================")
    print(" [Pure AST RSI Engine v3.0] 자율 출제 + 정확 검증기(SymPy) + 실시간 모니터링")
    print("==========================================================================")
    print(f" 홀드아웃 시험: 최고 난이도 문제 {len(heldout)}개 (학습에 절대 사용 안 함)")
    print(f" 예시: {heldout[0]}  |  {heldout[1]}")

    for gen in range(1, MAX_GENS + 1):
        if time.time() - t0 > TIME_LIMIT_SEC:
            print(f"\n⏱ 시간 제한 도달 ({TIME_LIMIT_SEC/60:.0f}분). 종료합니다.")
            break
        solved, verified_fail = 0, 0
        gen_paths, best_demo, nodes_solved = [], None, []

        for _ in range(PROBS_PER_GEN):
            prob = proposer.sample()
            ok, path, final, expanded = mcts_solve(env, brain, prob, max_steps=SEARCH_STEPS, device=device)
            if not ok:
                if len(proposer.frontier) < 200: proposer.frontier.append(prob)
                continue
            good, why = verifier.verify(prob, [s for s, _ in path], final)
            if not good:
                verified_fail += 1
                continue
            solved += 1
            nodes_solved.append(expanded)
            gen_paths.append(path)
            if not best_demo or atomic_len(env, path) > atomic_len(env, best_demo[1]):
                best_demo = (prob, path)
            for idx, (st, act) in enumerate(path):
                target_val = ((idx + 1) / len(path)) ** 0.5
                rec = (ast_to_feature_vector(st), act, target_val)
                memory.append(rec)
                if len(anchor) < 600 and random.random() < 0.3: anchor.append(rec)

        rate = 100.0 * solved / PROBS_PER_GEN
        diff = proposer.difficulty_index()
        knob_str = ", ".join(f"{k}={v:g}" for k, v in proposer.state.items())
        print(f"\n[제 {gen} 세대] 해결률(검증 통과) {rate:5.1f}% | 검증 거부 {verified_fail} | 난이도 {diff:.2f} ({knob_str}) | 스킬 {env.total_actions}개 | 평균 탐색 노드 {sum(nodes_solved)/max(1,len(nodes_solved)):.1f}")

        if best_demo and gen % 5 == 1:
            prob_eq, path = best_demo
            print(f"  └─ [추론 과정 해부] 문제: {prob_eq}  ({len(path)}단계)")
            curr = prob_eq.clone()
            for step_i, (_, act_idx) in enumerate(path):
                curr = env.apply_action(curr, act_idx)
                print(f"       Step {step_i+1}: [{env.action_name(act_idx)}]  ==>  {curr}")

        new_name = None
        if len(gen_paths) >= 5 and len(env.invented_macros) < 15:
            new_name, seq = compress_and_invent_skill(env, gen_paths)
            if new_name:
                print(f"  ★ [개념 발명] 새 스킬 {new_name}")
                brain.expand_brain(env.total_actions, device)
                optimizer = optim.AdamW(brain.parameters(), lr=0.004)

        loss_val = float('nan')
        if memory:
            memory = memory[-3000:]
            pool = memory + anchor
            brain.train()
            for _ in range(20):
                batch = random.sample(pool, min(len(pool), 64))
                feats = torch.tensor([x[0] for x in batch], dtype=torch.float, device=device)
                acts = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
                vals = torch.tensor([[x[2]] for x in batch], dtype=torch.float, device=device)
                optimizer.zero_grad()
                logits, pred_vals = brain(feats)
                loss = F.cross_entropy(logits, acts) + F.mse_loss(pred_vals, vals)
                loss.backward()
                optimizer.step()
                loss_val = loss.item()

        ent = policy_entropy(brain, memory, device)
        msg = proposer.adapt(rate)
        print(f"  ↳ 출제기: {msg} | 정책 엔트로피 {ent:.2f} | 손실 {loss_val:.3f} | 검증 누적 {verifier.checks}건 중 거부 {verifier.rejections}건")
        if gen > 3 and ent == ent and ent < 0.15:
            print("  ⚠ [붕괴 경보] 정책 엔트로피가 너무 낮음 (한 행동만 고집). 기억 버퍼의 기준 데이터 비율 유지 중.")

        row = {"gen": gen, "rate": rate, "verify_reject": verified_fail, "difficulty": diff,
               "knobs": dict(proposer.state), "skills": env.total_actions, "new_skill": new_name,
               "nodes_per_solve": (sum(nodes_solved) / len(nodes_solved)) if nodes_solved else None,
               "entropy": ent, "loss": loss_val, "elapsed_min": (time.time() - t0) / 60}

        if gen % EVAL_EVERY == 0 or gen == 1:
            h_rate, h_nodes, h_steps = evaluate(env, brain, heldout, verifier, device)
            control = ASTReasoningBrain(env.total_actions).to(device)
            control.load_state_dict({k: v for k, v in control_init.state_dict().items() if not k.startswith('policy_head')}, strict=False)
            c_rate, c_nodes, _ = evaluate(env, control, heldout, verifier, device)
            row.update({"heldout": h_rate, "heldout_nodes": h_nodes, "heldout_steps": h_steps,
                        "control": c_rate, "control_nodes": c_nodes})
            lift = h_rate - c_rate
            print(f"  📊 [홀드아웃 시험] 학습한 뇌 {h_rate:5.1f}% (노드 {h_nodes:.1f}) vs 학습 안 한 뇌 {c_rate:5.1f}% (노드 {c_nodes:.1f}) → 학습 효과 {lift:+.1f}%p")
            if h_rate > best_heldout:
                best_heldout = h_rate
                best_state = ({k: v.clone() for k, v in brain.state_dict().items()}, env.total_actions)
            elif best_state and h_rate < best_heldout - 15.0:
                sd, n_act = best_state
                restored = ASTReasoningBrain(n_act).to(device)
                restored.load_state_dict(sd)
                restored.expand_brain(env.total_actions, device)
                brain = restored
                optimizer = optim.AdamW(brain.parameters(), lr=0.004)
                print(f"  ⏪ [롤백] 홀드아웃이 최고점({best_heldout:.1f}%)보다 15%p 이상 하락 → 최고 시점의 뇌로 복원")
                row["rollback"] = True

        metrics.append(row)
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=1)

    print("\n==========================================================================")
    print(" 최종 요약")
    print("==========================================================================")
    hs = [m for m in metrics if "heldout" in m]
    if hs:
        print(f" 홀드아웃: 첫 측정 {hs[0]['heldout']:.1f}% → 마지막 {hs[-1]['heldout']:.1f}% (최고 {best_heldout:.1f}%)")
        print(f" 대조군(학습 안 한 뇌): 마지막 {hs[-1]['control']:.1f}%")
    print(f" 최종 난이도 {proposer.difficulty_index():.2f} | 스킬 {env.total_actions}개 (발명 {len(env.invented_macros)}개)")
    print(f" 검증기: 총 {verifier.checks}건 검사, {verifier.rejections}건 거부")
    for name, seq in env.invented_macros:
        print(f"   - {name}: {[env.atomic_axioms[a][0].split(':')[0] for a in seq]}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = [m["gen"] for m in metrics]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0][0].plot(g, [m["rate"] for m in metrics], label="in-loop solve %")
        ax[0][0].plot([m["gen"] for m in hs], [m["heldout"] for m in hs], "o-", label="held-out (trained)")
        ax[0][0].plot([m["gen"] for m in hs], [m["control"] for m in hs], "s--", label="held-out (untrained control)")
        ax[0][0].set_title("Solve rate"); ax[0][0].legend()
        ax[0][1].plot(g, [m["difficulty"] for m in metrics]); ax[0][1].set_title("Self-proposed difficulty index")
        ax[1][0].plot(g, [m["skills"] for m in metrics]); ax[1][0].set_title("Skills (axioms + invented)")
        ax[1][1].plot([m["gen"] for m in hs], [m["heldout_nodes"] for m in hs], "o-", label="trained")
        ax[1][1].plot([m["gen"] for m in hs], [m["control_nodes"] for m in hs], "s--", label="control")
        ax[1][1].set_title("Held-out search nodes per solve (lower = smarter)"); ax[1][1].legend()
        for a in ax.flat: a.set_xlabel("generation")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "progress.png"), dpi=110)
        print(f" 그래프 저장: {os.path.join(OUT_DIR, 'progress.png')}")
    except Exception as e:
        print("그래프 생략:", e)

if __name__ == "__main__":
    main()
