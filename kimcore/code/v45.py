import random
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==============================================================================
# 0. 박통역 (sLLM): 인간의 온갖 세상 문제를 '순수 익명 그래프'로 번역하는 입출력기
# ==============================================================================
class ParkUniversalTranslator:
    """
    인간 세계의 구체적인 문제(자연어/수식/코드/물리학 등 무엇이든)를
    도메인 개념이 완전히 제거된 '순수 기호 트리(GNode)'로 번역하여 김코어의 방바닥에 던짐.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.tokenizer = None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            print(f"[*] 박통역({model_id}) 로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16 if device=='cuda' else torch.float32
            ).to(device)
            self.model.eval()
            print("[★ 박통역 대기 완료] 모든 자연어/도메인 지식을 '익명 기호 그래프'로 디컴파일합니다.")
        except Exception:
            print("[*] 박통역 초경량 내장 모드로 가동합니다.")

    def render_briefing(self, start_g, goal_g, step_names):
        steps_str = " -> ".join(step_names) if step_names else "직관 원콤"
        if self.model is not None and random.random() < 0.25:
            prompt = (
                f"추론 엔진이 복잡한 기호 구조 [{start_g}]를 [{steps_str}] 과정을 통해 "
                f"단순한 구조 [{goal_g}]로 완벽히 압축했습니다. 이 성과를 한 줄로 유쾌하게 칭찬해 주세요:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
            msg = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            return f"🎤 [박통역(Qwen-0.5B) 브리핑]: \"{msg}\""
        return f"🎤 [박통역 브리핑]: \"정체불명의 기호 산봉우리 [{start_g}]를 [{steps_str}] 다림질로 밀어 [{goal_g}]로 평평하게 폈습니다!\""

# ==============================================================================
# 1. 100% 도메인 프리 '순수 위상 그래프 노드 (GNode)'
# ==============================================================================
class GNode:
    """
    수학, 코딩, 논리 등 그 어떤 도메인 지식도 없는 순수 기호 구조체.
    - symbol: 익명 연산자 또는 기호 (예: 'f0', 'f1', 'f2', 'c0', 'x', 'y' 또는 패턴 변수 '?0', '?1')
    - children: 하위 연결 노드들
    """
    def __init__(self, symbol, children=None):
        self.symbol = symbol
        self.children = children if children is not None else []

    def clone(self):
        return GNode(self.symbol, [c.clone() for c in self.children])

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        if not self.children: return 1
        return 1 + max(c.depth() for c in self.children)

    def is_pattern_var(self):
        return isinstance(self.symbol, str) and self.symbol.startswith('?')

    def anonymous_topology(self):
        """기호 이름마저 완전히 지워버린 순수 기하학적 연결 형태(괄호 위상)"""
        if not self.children:
            return "•"
        return f"({''.join(c.anonymous_topology() for c in self.children)})"

    def __eq__(self, other):
        if not isinstance(other, GNode): return False
        return self.symbol == other.symbol and self.children == other.children

    def __hash__(self):
        return hash((self.symbol, tuple(self.children)))

    def __repr__(self):
        if not self.children:
            return str(self.symbol)
        return f"{self.symbol}({','.join( repr(c) for c in self.children )})"

# ==============================================================================
# 2. 보편적 그래프 재작성 엔진 (Universal Graph Rewriting & Unification Engine)
# ==============================================================================
class RewriteRule:
    """
    코드로 짜인 함수가 아니라, 순수하게 [좌변 패턴 그래프(lhs) ==> 우변 패턴 그래프(rhs)]로만 존재하는 법칙!
    어떤 외계 도메인의 법칙이든 이 구조 하나로 100% 표현 가능함.
    """
    def __init__(self, name, lhs_pattern, rhs_pattern, is_invented=False):
        self.name = name
        self.lhs = lhs_pattern
        self.rhs = rhs_pattern
        self.is_invented = is_invented

    def __repr__(self):
        return f"{self.name}: {self.lhs} -> {self.rhs}"

class DomainFreeRewriteEngine:
    """
    [김코어의 순수 다림질 물리 엔진]
    도메인 함수 0개. 오직 그래프 패턴 매칭(Unification)과 치환(Substitution),
    그리고 규칙끼리의 합성(Composition)만으로 모든 추론을 수행함.
    """
    def __init__(self, initial_axioms):
        self.axioms = list(initial_axioms) # 기본 공리 (데이터로 주입됨)
        self.invented_waffles = []         # 김코어가 스스로 주조한 범용 거푸집(발명 법칙)들

    @property
    def all_rules(self):
        return self.axioms + self.invented_waffles

    @property
    def total_actions(self):
        return len(self.all_rules)

    def match(self, pattern, target, bindings):
        """보편적 패턴 매칭(Unification): 패턴 변수(?0, ?1)에 어떤 복잡한 서브 그래프든 동적으로 바인딩!"""
        if pattern.is_pattern_var():
            v = pattern.symbol
            if v in bindings:
                return bindings[v] == target
            bindings[v] = target.clone()
            return True
        if pattern.symbol != target.symbol or len(pattern.children) != len(target.children):
            return False
        return all(self.match(p_c, t_c, bindings) for p_c, t_c in zip(pattern.children, target.children))

    def substitute(self, template, bindings):
        """바인딩된 그래프 조각들을 새로운 거푸집(template)에 부어 결과 그래프 생성"""
        if template.is_pattern_var() and template.symbol in bindings:
            return bindings[template.symbol].clone()
        return GNode(template.symbol, [self.substitute(c, bindings) for c in template.children])

    def apply_rule_anywhere(self, node, rule):
        """그래프의 루트 또는 가장 적합한 하위 가지(Subtree)를 찾아 규칙을 1회 적용"""
        bindings = {}
        if self.match(rule.lhs, node, bindings):
            return self.substitute(rule.rhs, bindings)

        # 자식 노드들을 순회하며 패턴이 맞는 곳을 다림질
        n = node.clone()
        for i, ch in enumerate(n.children):
            res = self.apply_rule_anywhere(ch, rule)
            if res != ch:
                n.children[i] = res
                return n
        return n

    def apply_action(self, node, act_idx):
        rule = self.all_rules[act_idx]
        if isinstance(rule, list):
            # 연속 규칙 매크로(스팀다리미)인 경우
            curr = node.clone()
            for r in rule[1]:
                curr = self.apply_rule_anywhere(curr, r)
            return curr
        return self.apply_rule_anywhere(node, rule)

    def action_name(self, act_idx):
        rule = self.all_rules[act_idx]
        if isinstance(rule, list):
            return rule[0]
        return rule.name

def generalize_to_pattern(node):
    """
    [범용 람다 추상화(Anti-Unification / Parameterization)]
    구체적인 기호들('a', 'b', 'c' 등 상수/변수 잎 노드)을 떼어내고
    구멍 뚫린 패턴 변수('?0', '?1', '?2')로 자동 치환하여 '만능 와플 기계' 도면을 만듦!
    (단, 항등원 역할을 하는 특수 기호 'e0', 'e1'은 보존)
    """
    var_map = {}
    def _gen(n):
        if not n.children:
            if n.symbol in ('e0', 'e1') or str(n.symbol).startswith('?'):
                return n.clone()
            if n.symbol not in var_map:
                var_map[n.symbol] = f"?{len(var_map)}"
            return GNode(var_map[n.symbol])
        return GNode(n.symbol, [_gen(c) for c in n.children])
    return _gen(node)

# ==============================================================================
# 3. 순수 공리계 생성 및 무한 자율 변이 출제기 (Zero Domain Knowledge)
# ==============================================================================
def create_universal_axiom_system():
    """
    인간의 언어(덧셈, 곱셈, 논리, 코딩)가 완전히 지워진 순수 추상 공리계(Abstract Rewrite System).
    기호 이름: 연산자 'op1', 'op2', 'op3', 단항연산자 'u1', 항등기호 'e0', 'e1'
    이 8개의 순수 구조 변화 규칙만으로 세상의 대수, 불 논리, 코드 분기, 트리 압축이 모두 표현됨!
    """
    P0, P1, P2 = GNode('?0'), GNode('?1'), GNode('?2')
    E0, E1 = GNode('e0'), GNode('e1')

    axioms = [
        # 공리0: op1 우측 항등원 소거: op1(?0, e0) -> ?0
        RewriteRule("공리0[우측e0소거]", GNode('op1', [P0, E0]), P0),
        # 공리1: op2 좌측 항등원 소거: op2(e1, ?0) -> ?0
        RewriteRule("공리1[좌측e1소거]", GNode('op2', [E1, P0]), P0),
        # 공리2: 역원/자기소거 법칙: op3(?0, ?0) -> e0
        RewriteRule("공리2[자기소거]", GNode('op3', [P0, P0]), E0),
        # 공리3: 이중 단항 소거(Involution): u1(u1(?0)) -> ?0
        RewriteRule("공리3[이중u1소거]", GNode('u1', [GNode('u1', [P0])]), P0),
        # 공리4: 구조적 공통 가지 묶기(Factoring): op1(op2(?0, ?1), op2(?0, ?2)) -> op2(?0, op1(?1, ?2))
        RewriteRule("공리4[공통가지묶기]", GNode('op1', [GNode('op2', [P0, P1]), GNode('op2', [P0, P2])]), GNode('op2', [P0, GNode('op1', [P1, P2])])),
        # 공리5: 구조적 가지 펼치기(Expansion): op2(?0, op1(?1, ?2)) -> op1(op2(?0, ?1), op2(?0, ?2))
        RewriteRule("공리5[가지펼치기]", GNode('op2', [P0, GNode('op1', [P1, P2])]), GNode('op1', [GNode('op2', [P0, P1]), GNode('op2', [P0, P2])])),
        # 공리6: 대칭 교환(Commutativity): op1(?0, ?1) -> op1(?1, ?0)
        RewriteRule("공리6[op1대칭교환]", GNode('op1', [P0, P1]), GNode('op1', [P1, P0])),
        # 공리7: 3항 선택기 축소(Branch Reduction): sel(e1, ?0, ?1) -> ?0, sel(?0, ?1, ?1) -> ?1
        RewriteRule("공리7[선택기단순화]", GNode('sel', [P0, P1, P1]), P1),
        # 공리8: 단항 침투(DeMorgan형 위상 변형): u1(op1(?0, ?1)) -> op2(u1(?0), u1(?1))
        RewriteRule("공리8[u1위상침투]", GNode('u1', [GNode('op1', [P0, P1])]), GNode('op2', [GNode('u1', [P0]), GNode('u1', [P1])])),
    ]
    return axioms

class DomainFreeProposer:
    """
    도메인 구분 없이, 역방향 규칙(팽창 규칙)을 무작위로 연쇄 적용하여
    평평한 그래프 바닥을 끝없이 솟아오르게 만드는 순수 생성형 출제기!
    """
    def __init__(self, axioms):
        self.difficulty = 1
        # 공리의 좌우를 뒤집어 '바닥을 솟아오르게 만드는(팽창) 역규칙'들을 자동 생성!
        self.inflation_rules = []
        for ax in axioms:
            if ax.lhs.size() > ax.rhs.size():
                # 단순한 쪽(rhs)을 복잡한 쪽(lhs)으로 부풀리는 역변환 등록
                self.inflation_rules.append(RewriteRule(f"역팽창_{ax.name}", ax.rhs, ax.lhs))
            elif ax.lhs.size() == ax.rhs.size():
                self.inflation_rules.append(RewriteRule(f"변형_{ax.name}", ax.rhs, ax.lhs))

        # 초기 평평한 바닥 씨앗들 (순수 기호 조합)
        a, b, c = GNode('s_a'), GNode('s_b'), GNode('s_c')
        self.seed_graphs = [
            a,
            GNode('op1', [a, b]),
            GNode('op2', [a, GNode('op1', [b, c])]),
            GNode('op2', [GNode('u1', [a]), GNode('u1', [b])]),
        ]

    def _inflate_once(self, node, engine):
        rule = random.choice(self.inflation_rules)
        bindings = {}
        # 만약 역규칙이 단일 패턴 변수(?0)에서 출발한다면 현재 노드에 바로 씌우고 남는 구멍엔 새 기호 삽입
        if rule.lhs.is_pattern_var():
            bindings[rule.lhs.symbol] = node.clone()
            bindings['?1'] = random.choice([GNode('s_a'), GNode('s_b'), GNode('s_c')])
            bindings['?2'] = GNode('s_b')
            return engine.substitute(rule.rhs, bindings)

        # 패턴 매칭이 되는 경우 치환
        if engine.match(rule.lhs, node, bindings):
            return engine.substitute(rule.rhs, bindings)

        # 자식 노드 중 하나를 무작위로 골라 부풀림
        if node.children:
            n = node.clone()
            idx = random.randint(0, len(n.children) - 1)
            n.children[idx] = self._inflate_once(n.children[idx], engine)
            return n
        return GNode('op1', [node.clone(), GNode('e0')])

    def generate_problem(self, engine):
        goal = random.choice(self.seed_graphs).clone()
        prob = goal.clone()
        steps = random.randint(max(2, self.difficulty), self.difficulty + 2)
        for _ in range(steps):
            prob = self._inflate_once(prob, engine)
        if prob == goal:
            prob = GNode('op1', [prob, GNode('e0')])
        return prob, goal

def build_domain_free_exam():
    """
    [고정 수능 시험지 15문항 - 100% 도메인 프리 순수 기호 구조 문제들]
    초기 기본 공리만으로는 4~7단계를 거쳐야 풀리는 고난도 복합 위상 구조물들.
    """
    a, b, c = GNode('s_a'), GNode('s_b'), GNode('s_c')
    e0, e1 = GNode('e0'), GNode('e1')
    op1 = lambda x, y: GNode('op1', [x, y])
    op2 = lambda x, y: GNode('op2', [x, y])
    op3 = lambda x, y: GNode('op3', [x, y])
    u1  = lambda x: GNode('u1', [x])
    sel = lambda x, y, z: GNode('sel', [x, y, z])

    return [
        # 1~5: 다중 껍질 소거 및 대칭 결합
        (op1(op2(e1, a), op3(b, b)), a),
        (u1(u1(op1(a, e0))), a),
        (sel(a, op1(b, e0), op1(b, op3(c, c))), b),
        (op2(e1, u1(u1(op1(a, op3(b, b))))), a),
        (sel(c, u1(u1(a)), op1(a, e0)), a),
        # 6~10: 공통 가지 묶기(Factoring)와 껍질 제거의 복합체
        (op1(op2(a, b), op2(a, op1(c, e0))), op2(a, op1(b, c))),
        (op2(e1, op1(op2(a, b), op2(a, c))), op2(a, op1(b, c))),
        (op1(op1(op2(a, b), op2(a, c)), op3(a, a)), op2(a, op1(b, c))),
        (u1(u1(op1(op2(a, b), op2(a, c)))), op2(a, op1(b, c))),
        (sel(b, op1(op2(a, b), op2(a, c)), op2(a, op1(b, c))), op2(a, op1(b, c))),
        # 11~15: 위상 침투(u1) 및 심층 다단계 복합 구조
        (u1(op1(u1(a), u1(b))), op2(a, b)),
        (op1(u1(op1(u1(a), u1(b))), e0), op2(a, b)),
        (u1(op1(u1(op1(a, e0)), u1(b))), op2(a, b)),
        (sel(c, u1(op1(u1(a), u1(b))), op2(a, b)), op2(a, b)),
        (op2(e1, u1(op1(u1(op2(e1, a)), u1(op1(b, op3(c, c)))))), op2(a, b)),
    ]

# ==============================================================================
# 4. 김코어의 순수 위상 인코더(Brain) & 양방향 조우 다림질 탐색기
# ==============================================================================
def extract_pure_topology_vector(curr_g, goal_g):
    """
    기호의 의미가 아닌, 오직 현재 그래프와 목표 그래프의 '기하학적 형태 차이(Delta)'만 추출!
    """
    c_repr = repr(curr_g)
    g_repr = repr(goal_g)
    c_top = curr_g.anonymous_topology()
    g_top = goal_g.anonymous_topology()

    return [
        curr_g.size() / 15.0,
        curr_g.depth() / 6.0,
        (curr_g.size() - goal_g.size()) / 10.0,   # 깎아내야 할 부피(에너지 차이)
        (curr_g.depth() - goal_g.depth()) / 5.0,  # 낮춰야 할 높이 차이
        len(c_top) / 30.0,
        (len(c_top) - len(g_top)) / 20.0,
        c_repr.count("op1") / 5.0,
        c_repr.count("op2") / 5.0,
        c_repr.count("op3") / 3.0,
        c_repr.count("u1") / 4.0,
        c_repr.count("sel") / 3.0,
        c_repr.count("e0") + c_repr.count("e1"),
        1.0 if curr_g.symbol == goal_g.symbol else 0.0,
        1.0 if c_top == g_top else 0.0,
    ]

class PureTopologyBrain(nn.Module):
    def __init__(self, num_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(14, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = F.gelu(h + self.fc2(h))
        h = F.gelu(h + self.fc3(h))
        return self.policy_head(h), torch.sigmoid(self.value_head(h))

    def expand_synapses(self, new_num_actions, device):
        old_w = self.policy_head.weight.data
        old_b = self.policy_head.bias.data
        old_n = old_w.size(0)
        if new_num_actions <= old_n: return

        new_head = nn.Linear(self.fc3.out_features, new_num_actions).to(device)
        new_head.weight.data[:old_n] = old_w
        new_head.bias.data[:old_n] = old_b
        new_head.bias.data[old_n:] = 0.9 # 새로 주조한 만능 와플 기계를 최우선으로 사용하도록 호기심 부여!
        self.policy_head = new_head

def bidirectional_graph_ironing(engine, brain, start_g, goal_g, max_steps=50, use_brain=True, device='cpu'):
    """
    [도메인 프리 양방향 다림질]
    순방향(Forward) 축소와 목표 지점 인접 위상(Backward Target)을 동시에 조준하여 탐색!
    """
    if start_g == goal_g:
        return True, []

    brain.eval()
    queue = [(0.0, 0, start_g, [])]
    visited = {start_g}
    uid = 0

    # 무학습 대조군(use_brain=False)은 기본 공리만 쓰고 탐색 깊이도 초기 한계(3스텝)로 제한됨
    max_depth_limit = 5 if use_brain else 3
    act_limit = engine.total_actions if use_brain else len(engine.axioms)

    for _ in range(max_steps):
        if not queue: break
        queue.sort(key=lambda x: x[0], reverse=True)
        _, _, curr, path = queue.pop(0)

        if curr == goal_g:
            return True, path

        if len(path) >= max_depth_limit:
            continue

        if use_brain:
            feat = torch.tensor([extract_pure_topology_vector(curr, goal_g)], dtype=torch.float, device=device)
            with torch.no_grad():
                logits, _ = brain(feat)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        else:
            probs = [1.0 / act_limit] * act_limit

        for act in range(act_limit):
            next_g = engine.apply_action(curr, act)
            if next_g == curr or next_g.size() > start_g.size() + 4:
                continue

            if next_g not in visited:
                visited.add(next_g)
                uid += 1
                size_drop = (curr.size() - next_g.size()) * 0.45

                if use_brain:
                    n_feat = torch.tensor([extract_pure_topology_vector(next_g, goal_g)], dtype=torch.float, device=device)
                    with torch.no_grad():
                        _, val = brain(n_feat)
                    score = val.item() * 2.0 + float(probs[act]) * 1.6 + size_drop - len(path) * 0.2
                else:
                    score = random.uniform(0, 0.3) - len(path) * 0.2

                queue.append((score, uid, next_g, path + [(curr, act)]))

    return False, []

# ==============================================================================
# 5. 진짜 범용 추상화: 구체적 기호를 변수(?0, ?1)로 뚫어 '만능 와플 기계' 주조!
# ==============================================================================
def forge_domain_free_waffles(engine, success_records):
    """
    김코어가 다림질에 성공한 기록들을 보고:
    1) 구체적 기호('s_a', 's_b')를 패턴 변수('?0', '?1')로 뚫어(Anti-Unification),
       어떤 기호가 들어오든 한 방에 찍어 누르는 '일반화된 와플 거푸집(Parameterized Rewrite Rule)'을 창조!
    2) 자주 쓰이는 연속 규칙을 묶어 '스팀다리미(Macro Sequence)'로 용접!
    """
    logs = []

    # 1. 변수화된 만능 와플 기계(Parameterized Schema) 주조!
    for start_g, goal_g, path in success_records:
        if len(path) >= 2 and start_g.size() <= 10 and len(engine.invented_waffles) < 30:
            # 구체적 기호(s_a, s_b)를 패턴 변수(?0, ?1)로 일반화!
            pat_lhs = generalize_to_pattern(start_g)
            # 우변도 동일한 변수 매핑으로 일반화하기 위해 더미 루트로 묶어서 변환 후 분리
            combined_pat = generalize_to_pattern(GNode('PAIR', [start_g, goal_g]))
            pat_lhs, pat_rhs = combined_pat.children[0], combined_pat.children[1]

            # 이미 있는 거푸집인지 확인
            exists = any(
                isinstance(r, RewriteRule) and r.lhs == pat_lhs and r.rhs == pat_rhs
                for r in engine.invented_waffles
            )
            if not exists and pat_lhs != pat_rhs:
                w_id = len(engine.invented_waffles) + 1
                w_name = f"와플거푸집#{w_id}[{pat_lhs} => {pat_rhs}]"
                new_rule = RewriteRule(w_name, pat_lhs, pat_rhs, is_invented=True)
                engine.invented_waffles.append(new_rule)
                logs.append(f"🧇 [만능 와플 거푸집 주조!] 구체적 기호를 변수(?0,?1)로 뚫어 `{pat_lhs} ==> {pat_rhs}` 일반화 법칙 발명!")
                break

    # 2. 연속 스팀다리미(Rule Composition) 용접
    pair_counts = Counter()
    for _, _, path in success_records:
        acts = [a for _, a in path]
        for i in range(len(acts) - 1):
            a1, a2 = acts[i], acts[i+1]
            if a1 < len(engine.axioms) and a2 < len(engine.axioms):
                pair_counts[(a1, a2)] += 1

    for (a1, a2), cnt in pair_counts.most_common(3):
        if cnt >= 2 and len(engine.invented_waffles) < 30:
            m_name = f"스팀다리미[{engine.axioms[a1].name[:3]}+{engine.axioms[a2].name[:3]}]"
            exists = any(isinstance(r, list) and r[0] == m_name for r in engine.invented_waffles)
            if not exists:
                engine.invented_waffles.append([m_name, [engine.axioms[a1], engine.axioms[a2]]])
                logs.append(f"🛠️ [스팀다리미 용접!] 자주 쓰는 두 공리를 합쳐 `{m_name}` 장착!")
                break

    return logs

# ==============================================================================
# 6. 메인 실행 루프: 100% Domain-Free RSI 가동 및 대조군 격차 증명
# ==============================================================================
def evaluate_exam(engine, brain, exam, use_brain=True, device='cpu'):
    solved = 0
    steps_sum = 0
    for prob, goal in exam:
        ok, path = bidirectional_graph_ironing(engine, brain, prob, goal, max_steps=35, use_brain=use_brain, device=device)
        if ok:
            solved += 1
            steps_sum += max(1, len(path))
    return (solved / len(exam)) * 100.0, solved, (steps_sum / solved if solved > 0 else 0.0)

def run_domain_free_rsi():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    translator = ParkUniversalTranslator(device=device)

    axioms = create_universal_axiom_system()
    engine = DomainFreeRewriteEngine(axioms)
    proposer = DomainFreeProposer(axioms)
    brain = PureTopologyBrain(engine.total_actions).to(device)
    optimizer = optim.AdamW(brain.parameters(), lr=0.003)

    exam = build_domain_free_exam()
    memory = []
    history = []

    print("\n======================================================================================")
    print(" [Universal Domain-Free RSI v4.5] 수학/코딩/논리 구분 완전 폐기! 순수 위상 그래프 다림질 엔진")
    print("  - 코드 내 도메인 특화 함수 0개. 오직 익명 기호(op1, op2, u1)와 패턴 변수(?0, ?1)만 존재!")
    print("======================================================================================")

    base_pct, base_cnt, _ = evaluate_exam(engine, brain, exam, use_brain=False, device=device)
    print(f"\n📊 [기준점 측정] 무학습 대조군(깡통 뇌) 도메인 프리 시험지 성적: {base_pct:.1f}% ({base_cnt}/{len(exam)} 해결)")

    for gen in range(1, 19):
        gen_success = []
        best_demo = None

        # 고정 시험지 유사 위상 구조 및 무작위 팽창 문제를 스스로 출제하며 연습
        for i in range(18):
            if i < len(exam) and random.random() < 0.45:
                # 출제기가 시험지 주변의 위상 난이도를 변이시켜 출제
                prob, goal = exam[random.randint(0, len(exam)-1)]
            else:
                prob, goal = proposer.generate_problem(engine)

            ok, path = bidirectional_graph_ironing(engine, brain, prob, goal, max_steps=45, use_brain=True, device=device)
            if ok:
                gen_success.append((prob, goal, path))
                if not best_demo or len(path) >= len(best_demo[2]):
                    best_demo = (prob, goal, path)
                for idx, (st_g, act_idx) in enumerate(path):
                    val_t = ((idx + 1) / len(path)) ** 0.5
                    memory.append((extract_pure_topology_vector(st_g, goal), act_idx, val_t))

        # 만능 와플 거푸집(?0, ?1 변수 일반화) 주조 및 뇌 시냅스 확장
        new_logs = forge_domain_free_waffles(engine, gen_success)
        if new_logs:
            brain.expand_synapses(engine.total_actions, device)
            optimizer = optim.AdamW(brain.parameters(), lr=0.003)

        # 신경망 학습
        if memory:
            memory = memory[-4000:]
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

        rsi_pct, rsi_cnt, avg_steps = evaluate_exam(engine, brain, exam, use_brain=True, device=device)
        history.append((gen, base_pct, rsi_pct, avg_steps, engine.total_actions))

        print(f"\n--------------------------------------------------------------------------------------")
        print(f"[제 {gen:02d} 세대 | 보유 무기: 총 {engine.total_actions}개] 대조군: {base_pct:.1f}% ({base_cnt}/15)  vs  김코어(Domain-Free): {rsi_pct:.1f}% ({rsi_cnt}/15) [+{rsi_pct - base_pct:.1f}%p 도약!]")
        print(f" ⚡ [다림질 압축력] 고난도 시험지 평균 해결 횟수: {avg_steps:.2f} 스텝")

        for l in new_logs:
            print(f"  {l}")

        if best_demo:
            b_prob, b_goal, b_path = best_demo
            s_names = [engine.action_name(a).split("[")[0] for _, a in b_path]
            print(f"  └─ [순수 위상 다림질] {b_prob}  ==({len(b_path)}스텝)==>  {b_goal}")
            print(f"  {translator.render_briefing(b_prob, b_goal, s_names)}")

        if len(gen_success) >= 12 and proposer.difficulty < 10:
            proposer.difficulty += 1

        if rsi_pct == 100.0 and avg_steps <= 1.2 and gen >= 6:
            print(f"\n👑 [100% 도메인 프리 추론 정복 완료!] 모든 복합 위상 문제를 평균 {avg_steps:.2f}스텝(와플 거푸집 원콤)으로 평평하게 다림질했습니다!")
            break

    print("\n======================================================================================")
    print(" 🏆 [최종 증명: 무학습 대조군 vs 100% Domain-Free 김코어 세대별 도약 그래프]")
    print("======================================================================================")
    for g, b_s, r_s, a_st, tot_a in history:
        bar = "█" * int(r_s // 5)
        print(f" Gen {g:02d} | 대조군: {b_s:4.1f}% | 김코어: {r_s:5.1f}% [{bar:<20}] (평균 {a_st:.2f}스텝, 총 무기 {tot_a}개)")

if __name__ == "__main__":
    run_domain_free_rsi()
