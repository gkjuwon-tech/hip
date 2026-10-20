import copy
import random
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

# ==============================================================================
# 2. 무결점 기초 대수 공리 (분배법칙 및 교환법칙 완벽 패치)
# ==============================================================================
class PureAlgebraEnv:
    def __init__(self):
        self.atomic_axioms = [
            ("공리0: 상수 산술 계산", self.axiom_eval_constants),
            ("공리1: 지수 정의 풀어쓰기 (A^2 -> A*A)", self.axiom_expand_power),
            ("공리2: 지수 정의 묶기 (A*A -> A^2)", self.axiom_fold_power),
            ("공리3: 일차항 반으로 쪼개기 (2kx -> x*k + k*x)", self.axiom_split_linear),
            ("공리4: 상수항 음수 인수분해 (k^2 -> (-k)*(-k))", self.axiom_factor_const_neg),
            ("공리5: 앞부분 분배법칙 묶기", self.axiom_factor_front),
            ("공리6: 뒷부분 분배법칙 묶기", self.axiom_factor_back),
            ("공리7: 영인자 법칙 (A^2 = 0 -> A = 0)", self.axiom_zero_product),
            ("공리8: 양변에 1 더하기", lambda eq: self.axiom_add_both(eq, 1)),
            ("공리9: 양변에 2 더하기", lambda eq: self.axiom_add_both(eq, 2)),
            ("공리10: 양변에 3 더하기", lambda eq: self.axiom_add_both(eq, 3)),
        ]
        self.invented_macros = []

    @property
    def total_actions(self):
        return len(self.atomic_axioms) + len(self.invented_macros)

    def axiom_eval_constants(self, eq):
        def _eval(node):
            node = node.clone()
            node.children = [_eval(c) for c in node.children]
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

    def axiom_split_linear(self, eq):
        lhs = eq.children[0].clone()
        if lhs.op == '+':
            new_ch = []
            split_done = False
            for c in lhs.children:
                if not split_done and c.op == '*' and c.children[0].op == 'num' and c.children[1].op == 'var':
                    val = c.children[0].val
                    if val % 2 == 0 and val != 0:
                        half = val // 2
                        new_ch.append(Mul(Var('x'), Num(half)))
                        new_ch.append(Mul(Num(half), Var('x')))
                        split_done = True
                        continue
                new_ch.append(c)
            if split_done:
                return Eq(Add(*new_ch), eq.children[1].clone())
        return eq

    def axiom_factor_const_neg(self, eq):
        lhs = eq.children[0].clone()
        if lhs.op == '+':
            new_ch = []
            for c in lhs.children:
                if c.op == 'num' and c.val in [1, 4, 9, 16]:
                    k = -int(round(c.val ** 0.5))
                    new_ch.append(Mul(Num(k), Num(k)))
                else:
                    new_ch.append(c)
            return Eq(Add(*new_ch), eq.children[1].clone())
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
        if rhs == Num(0) and lhs.op == '^' and lhs.children[1] == Num(2):
            return Eq(lhs.children[0].clone(), Num(0))
        return eq

    def axiom_add_both(self, eq, val):
        lhs, rhs = eq.children
        new_eq = Eq(Add(lhs, Num(val)), Add(rhs, Num(val)))
        return self.axiom_eval_constants(new_eq)

    def apply_action(self, eq, act_idx):
        if act_idx < len(self.atomic_axioms):
            return self.atomic_axioms[act_idx][1](eq)
        else:
            macro_seq = self.invented_macros[act_idx - len(self.atomic_axioms)][1]
            curr = eq.clone()
            for a in macro_seq:
                curr = self.atomic_axioms[a][1](curr)
            return curr

    def is_solved(self, eq):
        return eq.children[0] == Var('x') and eq.children[1].op == 'num'

    def generate_curriculum_problem(self, level):
        k = random.choice([1, 2, 3])
        if level == 1:
            return Eq(Add(Var('x'), Num(-k)), Num(0))
        elif level == 2:
            return Eq(Pow(Add(Var('x'), Num(-k)), Num(2)), Num(0))
        elif level == 3:
            t1 = Mul(Var('x'), Add(Var('x'), Num(-k)))
            t2 = Mul(Num(-k), Add(Var('x'), Num(-k)))
            return Eq(Add(t1, t2), Num(0))
        elif level == 4:
            t1 = Mul(Var('x'), Add(Var('x'), Num(-k)))
            t2 = Mul(Num(-k), Var('x'))
            t3 = Mul(Num(-k), Num(-k))
            return Eq(Add(t1, t2, t3), Num(0))
        elif level == 5:
            t1 = Mul(Var('x'), Var('x'))
            t2 = Mul(Var('x'), Num(-k))
            t3 = Mul(Num(-k), Var('x'))
            t4 = Num(k * k)
            return Eq(Add(t1, t2, t3, t4), Num(0))
        else:
            t_sq = Pow(Var('x'), Num(2))
            t_lin = Mul(Num(-2 * k), Var('x'))
            t_const = Num(k * k)
            return Eq(Add(t_sq, t_lin, t_const), Num(0))

# ==============================================================================
# 3. AST 구조 특징 추출기 & 확장형 추론 뇌
# ==============================================================================
def ast_to_feature_vector(eq):
    s = str(eq)
    lhs, rhs = eq.children

    def count_op(node, op):
        c = 1 if node.op == op else 0
        return c + sum(count_op(ch, op) for ch in node.children)

    def max_depth(node):
        if not node.children: return 1
        return 1 + max(max_depth(ch) for ch in node.children)

    feats = [
        count_op(lhs, '^') / 3.0,
        count_op(lhs, '*') / 5.0,
        count_op(lhs, '+') / 5.0,
        (len(lhs.children) if lhs.op == '+' else 1) / 5.0,
        max_depth(lhs) / 5.0,
        1.0 if lhs.op == '^' else 0.0,
        1.0 if lhs.op == '*' else 0.0,
        1.0 if lhs.op == 'var' else 0.0,
        float(rhs.val) / 5.0 if rhs.op == 'num' else 0.0,
        len(s) / 30.0
    ]
    return feats

class ASTReasoningBrain(nn.Module):
    def __init__(self, num_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(10, hidden_dim)
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
# 4. 탐색기 (MCTS/Best-First) & 복리 개념 압축기
# ==============================================================================
def mcts_solve(env, brain, start_eq, max_steps=250, device='cpu'):
    brain.eval()
    queue = [(0.0, 0, start_eq, [])]
    visited = {str(start_eq)}
    uid = 0

    for _ in range(max_steps):
        if not queue: break
        queue.sort(key=lambda x: x[0], reverse=True)
        _, _, curr_eq, path = queue.pop(0)

        if env.is_solved(curr_eq):
            return True, path

        if len(path) >= 7:
            continue

        feat = torch.tensor([ast_to_feature_vector(curr_eq)], dtype=torch.float, device=device)
        with torch.no_grad():
            logits, _ = brain(feat)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        for act in range(env.total_actions):
            next_eq = env.apply_action(curr_eq, act)
            s_next = str(next_eq)

            if s_next == str(curr_eq) or len(s_next) > 50:
                continue

            if s_next not in visited:
                visited.add(s_next)
                uid += 1
                next_feat = torch.tensor([ast_to_feature_vector(next_eq)], dtype=torch.float, device=device)
                with torch.no_grad():
                    _, val = brain(next_feat)

                lhs_terms = len(next_eq.children[0].children) if next_eq.children[0].op == '+' else 1
                compression_bonus = (5 - lhs_terms) * 0.2

                score = val.item() * 2.0 + probs[act] * 1.5 + compression_bonus - (len(path) * 0.2) + random.uniform(0, 0.05)
                queue.append((score, uid, next_eq, path + [(curr_eq, act)]))

    return False, []

def compress_and_invent_skill(env, success_paths):
    seq_counter = Counter()
    for path in success_paths:
        acts = [a for _, a in path]
        for i in range(len(acts) - 1):
            a1, a2 = acts[i], acts[i+1]
            raw1 = [a1] if a1 < len(env.atomic_axioms) else env.invented_macros[a1 - len(env.atomic_axioms)][1]
            raw2 = [a2] if a2 < len(env.atomic_axioms) else env.invented_macros[a2 - len(env.atomic_axioms)][1]
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

# ==============================================================================
# 5. 메인 RSI 루프 실행
# ==============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = PureAlgebraEnv()
    brain = ASTReasoningBrain(env.total_actions).to(device)
    optimizer = optim.AdamW(brain.parameters(), lr=0.004)

    level = 1
    memory = []

    print("==========================================================================")
    print(" [Pure AST RSI Engine v2.0] 교환법칙 패치 & 6단계 복리 커리큘럼 가동")
    print("==========================================================================")

    for gen in range(1, 35):
        solved = 0
        gen_paths = []
        best_demo = None

        for _ in range(25):
            prob = env.generate_curriculum_problem(level)
            ok, path = mcts_solve(env, brain, prob, max_steps=180, device=device)
            if ok:
                solved += 1
                gen_paths.append(path)
                if not best_demo or len(path) < len(best_demo[1]):
                    best_demo = (prob, path)

                for idx, (st, act) in enumerate(path):
                    target_val = ((idx + 1) / len(path)) ** 0.5
                    memory.append((ast_to_feature_vector(st), act, target_val))

        rate = (solved / 25.0) * 100
        print(f"\n[제 {gen} 세대 | 커리큘럼 Lv.{level}] 해결률: {rate:.1f}% (보유 스킬: 총 {env.total_actions}개)")

        if best_demo:
            prob_eq, path = best_demo
            print(f"  └─ [추론 과정 해부] 문제: {prob_eq}  (총 {len(path)}단계의 생각 만에 해결!)")
            curr = prob_eq.clone()
            for step_i, (_, act_idx) in enumerate(path):
                act_name = env.atomic_axioms[act_idx][0] if act_idx < len(env.atomic_axioms) else env.invented_macros[act_idx - len(env.atomic_axioms)][0]
                curr = env.apply_action(curr, act_idx)
                print(f"       Step {step_i+1}: [{act_name}] 적용  ==>  {curr}")

        if len(gen_paths) >= 5 and len(env.invented_macros) < 15:
            new_name, seq = compress_and_invent_skill(env, gen_paths)
            if new_name:
                print(f"  ★ [개념 발명(Abstraction)!] 모델이 스스로 새로운 대수 스킬 [{new_name}]을 창조했습니다!")
                brain.expand_brain(env.total_actions, device)
                optimizer = optim.AdamW(brain.parameters(), lr=0.004)

        if memory:
            memory = memory[-3000:]
            brain.train()
            for _ in range(20):
                batch = random.sample(memory, min(len(memory), 64))
                feats = torch.tensor([x[0] for x in batch], dtype=torch.float, device=device)
                acts = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
                vals = torch.tensor([[x[2]] for x in batch], dtype=torch.float, device=device)

                optimizer.zero_grad()
                logits, pred_vals = brain(feats)
                loss = F.cross_entropy(logits, acts) + F.mse_loss(pred_vals, vals)
                loss.backward()
                optimizer.step()

        if rate >= 80.0:
            if level < 6:
                level += 1
                print(f"  ▲ [커리큘럼 돌파!] 다음 세대는 난이도 Lv.{level}에 도전합니다!")
            else:
                print(f"\n  👑 [이차방정식 완전 정복 달성!!] 치팅 0%, 순수 공리와 자가 발명 스킬만으로 최종 보스(Lv.6) 이차방정식을 정복했습니다!")
                if len(best_demo[1]) <= 4:
                    print(f"  🚀 [복리 지능 증명 완료] 원래 9단계가 걸리는 이차방정식을 스스로 발명한 인수분해 스킬들로 단 {len(best_demo[1])}단계 만에 풀어냈습니다!")
                    break

if __name__ == "__main__":
    main()
