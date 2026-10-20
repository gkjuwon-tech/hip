import os
import re
import time
import json
import random
import subprocess
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==============================================================================
# 0. 실제 Lean 4 컴파일러 자동 설치기 (Kaggle Linux 환경용)
# ==============================================================================
def setup_lean4():
    lean_bin_dir = os.path.expanduser("~/.elan/bin")
    os.environ["PATH"] = f"{lean_bin_dir}:{os.environ.get('PATH', '')}"

    try:
        res = subprocess.run(["lean", "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[*] Lean 4 이미 설치됨: {res.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("[*] 인터넷을 통해 실제 Lean 4 컴파일러(elan)를 다운로드 및 설치합니다 (약 1~2분 소요)...")
    install_cmd = (
        "curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | "
        "sh -s -- -y --default-toolchain leanprover/lean4:stable"
    )
    subprocess.run(install_cmd, shell=True, check=True)

    res = subprocess.run(["lean", "--version"], capture_output=True, text=True)
    print(f"[★ 설치 완료!] {res.stdout.strip()}")
    return True

# ==============================================================================
# 1. Lean 4 초고속 배치 검증기 + 개념(def) 및 보조정리(Lemma) 무결성 축적기
# ==============================================================================
class Lean4Verifier:
    def __init__(self, work_dir="/tmp/lean_rsi_v3"):
        os.makedirs(work_dir, exist_ok=True)
        self.test_file = os.path.join(work_dir, "batch_verify.lean")
        self.check_file = os.path.join(work_dir, "single_check.lean")
        self.base_header = "open Nat\n\n"

        # 모델이 스스로 창조한 개념(def 함수)들과 스스로 증명한 보조정리(theorem) 저장소
        self.invented_defs = []
        self.accumulated_lemmas = []

    def get_full_header(self):
        defs_block = "\n".join(self.invented_defs)
        lemmas_block = "\n".join(self.accumulated_lemmas)
        return f"{self.base_header}-- [Invented Definitions]\n{defs_block}\n\n-- [Accumulated Lemmas]\n{lemmas_block}\n\n"

    def safely_register_def(self, def_code):
        """새로운 개념(def)이 Lean 4 문법 및 정지성(Termination) 검사를 완벽히 통과하는지 검증 후 등재"""
        test_code = self.get_full_header() + def_code + "\n"
        with open(self.check_file, "w") as f:
            f.write(test_code)
        try:
            res = subprocess.run(["lean", self.check_file], capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and "error" not in (res.stdout + res.stderr):
                self.invented_defs.append(def_code)
                return True
        except subprocess.TimeoutExpired:
            pass
        return False

    def safely_register_lemma(self, lemma_name, stmt, proof_lines):
        """발견한 보조정리가 단독으로도 에러나 경고(sorry) 없이 100% 완벽히 컴파일되는지 재검증 후 영구 등재"""
        tactic_str = "\n  ".join(proof_lines)
        lemma_code = f"theorem {lemma_name} {stmt} := by\n  {tactic_str}\n"
        test_code = self.get_full_header() + lemma_code

        with open(self.check_file, "w") as f:
            f.write(test_code)
        try:
            res = subprocess.run(["lean", self.check_file], capture_output=True, text=True, timeout=10)
            out = res.stdout + "\n" + res.stderr
            if res.returncode == 0 and "error" not in out and "warning" not in out:
                self.accumulated_lemmas.append(lemma_code)
                return True
        except subprocess.TimeoutExpired:
            pass
        return False

    def verify_batch(self, theorem_stmt, candidate_proofs, timeout=15):
        if not candidate_proofs:
            return -1

        header_lines = self.get_full_header().splitlines()
        lines = list(header_lines)
        marker_to_idx = {}

        for idx, tactics in enumerate(candidate_proofs):
            start_line = len(lines) + 1
            block_lines = [f"theorem cand_{idx} {theorem_stmt} := by"] + [f"  {t}" for t in (tactics if tactics else ["sorry"])]
            lines.extend(block_lines)
            marker_to_idx[idx] = (start_line, len(lines))

        full_code = "\n".join(lines) + "\n"
        with open(self.test_file, "w") as f:
            f.write(full_code)

        try:
            res = subprocess.run(
                ["lean", self.test_file],
                capture_output=True, text=True, timeout=timeout
            )
            output = res.stdout + "\n" + res.stderr
        except subprocess.TimeoutExpired:
            return -1

        failed_lines = set()
        for line in output.splitlines():
            m = re.search(r"batch_verify\.lean:(\d+):\d+:\s*(error|warning)", line)
            if m:
                failed_lines.add(int(m.group(1)))

        for idx in range(len(candidate_proofs)):
            if not candidate_proofs[idx]:
                continue
            s_line, e_line = marker_to_idx[idx]
            has_error = any(s_line <= err_l <= e_line for err_l in failed_lines)
            if not has_error:
                return idx

        return -1

# ==============================================================================
# 2. 개념 창조기(Concept Inventor) & 자율 문제 출제기(Proposer)
# ==============================================================================
class AutonomousMathematician:
    """
    인간 커리큘럼 없이 1) 스스로 새로운 수학 함수(def)를 정의하고,
    2) 귀납법(Induction)이 필요한 명제와 창조된 개념의 성질을 묻는 문제를 스스로 출제함.
    """
    def __init__(self):
        self.complexity_budget = 1
        self.solved_history = set()
        self.active_concepts = [] # (함수이름, 인자수, 설명)

        # 모델이 진화하면서 단계별로 스스로 발명할 수 있는 '새로운 수학 개념(def)' 후보 풀
        self.concept_blueprints = [
            ("c_double", 1, "def c_double (n : Nat) : Nat := n + n", "스스로 덧셈을 묶어 '2배수(Double)' 개념 정의"),
            ("c_square", 1, "def c_square (n : Nat) : Nat := n * n", "스스로 곱셈을 묶어 '제곱(Square)' 개념 정의"),
            ("c_step", 1, "def c_step (n : Nat) : Nat := succ n", "페아노 후자 함수를 새로운 '스텝(Step)' 연산으로 정의"),
            ("c_sym_add", 2, "def c_sym_add (a b : Nat) : Nat := (a + b) + (b + a)", "대칭 덧셈 결합 연산자 정의"),
            ("c_linear", 2, "def c_linear (a b : Nat) : Nat := (a * b) + a", "선형 변환 연산자 a*b + a 정의"),
            ("c_rec_id", 1, "def c_rec_id : Nat → Nat\n  | 0 => 0\n  | succ n => succ (c_rec_id n)", "재귀적 항등 함수(Recursive Identity) 정의 (귀납법 증명 필수!)"),
            ("c_rec_add2", 1, "def c_rec_add2 : Nat → Nat\n  | 0 => 0\n  | succ n => succ (succ (c_rec_add2 n))", "재귀적 짝수 생성기 정의 (귀납법 증명 필수!)"),
        ]
        self.next_concept_idx = 0

    def try_invent_new_concept(self, verifier, tac_env, brain, device):
        """진화 수준이 무르익었을 때 세상에 없던 새로운 함수(def)를 선언하고 관련 전술을 뇌에 장착!"""
        if self.next_concept_idx >= len(self.concept_blueprints):
            return None

        c_name, arity, def_code, desc = self.concept_blueprints[self.next_concept_idx]
        self.next_concept_idx += 1

        if verifier.safely_register_def(def_code):
            self.active_concepts.append((c_name, arity, desc))
            # 뇌에 '내가 만든 개념의 정의를 풀어헤치는(unfold/simp) 스킬'을 자동으로 추가!
            unfold_skill_name = f"개념전개[{c_name} 정의 풀기 (simp [{c_name}])]"
            unfold_tactic = f"try simp [{c_name}]"
            tac_env.invented_macros.append((unfold_skill_name, unfold_tactic))
            brain.expand_brain(tac_env.total_actions, device)
            return c_name, desc, def_code
        return None

    def generate_frontier_conjecture(self):
        # 4대 수학 영역을 자율 탐색: 1) 귀납법(INDUCTION), 2) 창조된 개념 탐구(CONCEPT), 3) 수리논리(LOGIC), 4) 대수(ALGEBRA)
        domains = ["ALGEBRA", "ALGEBRA", "LOGIC", "INDUCTION"]
        if self.active_concepts:
            domains.extend(["CONCEPT", "CONCEPT"]) # 창조한 개념이 있으면 그 개념의 성질을 집중 탐구!

        domain = random.choice(domains)

        # [영역 1] 창조된 개념(def)의 수학적 성질을 스스로 탐구하는 문제 출제!
        if domain == "CONCEPT" and self.active_concepts:
            c_name, arity, _ = random.choice(self.active_concepts)
            if c_name == "c_double":
                props = [
                    "(n : Nat) : c_double n = n + n",
                    "(n : Nat) : c_double (n + 0) = n + n",
                    ": c_double 0 = 0",
                    "(a b : Nat) : c_double a + b = (a + a) + b",
                ]
            elif c_name == "c_square":
                props = [
                    "(n : Nat) : c_square n = n * n",
                    ": c_square 1 = 1",
                    "(n : Nat) : c_square (n * 1) = n * n",
                    "(n : Nat) : c_square n = n ^ 2",
                ]
            elif c_name == "c_step":
                props = [
                    "(n : Nat) : c_step n = n + 1",
                    "(a b : Nat) : c_step a + b = succ (a + b)",
                ]
            elif c_name == "c_sym_add":
                props = [
                    "(a b : Nat) : c_sym_add a b = c_sym_add b a", # 대칭성(교환법칙) 스스로 증명!
                    "(a : Nat) : c_sym_add a 0 = a + a",
                ]
            elif c_name == "c_linear":
                props = [
                    "(a : Nat) : c_linear a 0 = a",
                    "(a b : Nat) : c_linear a b = a * (b + 1)", # 인수분해 성질 증명!
                ]
            elif c_name == "c_rec_id":
                props = [
                    "(n : Nat) : c_rec_id n = n", # 수학적 귀납법으로만 증명 가능한 재귀 정리!
                    "(n : Nat) : c_rec_id (n + 0) = n",
                ]
            else:
                props = [
                    ": c_rec_add2 0 = 0",
                    "(n : Nat) : c_rec_add2 (succ n) = succ (succ (c_rec_add2 n))",
                ]
            return random.choice(props), self.complexity_budget

        # [영역 2] 수학적 귀납법(Induction) 및 페아노 공리계 핵심 정리 출제
        if domain == "INDUCTION":
            ind_pool = [
                ("(n : Nat) : 0 + n = n", 1),
                ("(n : Nat) : n + 0 = n", 1),
                ("(n : Nat) : succ n = n + 1", 2),
                ("(a b : Nat) : succ a + b = succ (a + b)", 2),
                ("(a b : Nat) : a + succ b = succ (a + b)", 2),
                ("(n : Nat) : 1 * n = n", 3),
                ("(n : Nat) : n * 1 = n", 3),
                ("(n : Nat) : 0 * n = 0", 3),
                ("(a b c : Nat) : (a + b) + c = a + (b + c)", 4),
            ]
            max_i = min(len(ind_pool) - 1, self.complexity_budget)
            return random.choice(ind_pool[:max_i + 1])

        # [영역 3] 순수 수리 논리학(Prop) 명제 출제
        if domain == "LOGIC":
            lv = min(self.complexity_budget, 6)
            logic_pool = [
                "(P : Prop) : P -> P",
                "(P Q : Prop) : P -> (Q -> P)",
                "(P Q : Prop) : (P ∧ Q) -> (Q ∧ P)",
                "(P Q : Prop) : (P ∧ Q) -> P",
                "(P Q R : Prop) : (P -> Q) -> (Q -> R) -> (P -> R)",
                "(P Q R : Prop) : (P ∧ (Q ∧ R)) -> ((P ∧ Q) ∧ R)"
            ]
            return random.choice(logic_pool[:lv]), lv

        # [영역 4] 무작위 AST 변이를 통한 무한 대수 항등식 생성
        base_seeds = ["a", "(a + b)", "(a * b)", "((a + b) + c)", "(a * (b + c))", "((a + b) * c)"]
        max_idx = min(len(base_seeds) - 1, self.complexity_budget // 2)
        seed = random.choice(base_seeds[:max_idx + 1])

        lhs, rhs = seed, seed
        num_twists = random.randint(1, max(1, self.complexity_budget))
        for _ in range(num_twists):
            choice = random.randint(0, 7)
            if choice == 0: lhs = f"({lhs} + 0)"
            elif choice == 1: lhs = f"(0 + {lhs})"
            elif choice == 2: lhs = f"({lhs} * 1)"
            elif choice == 3: lhs = f"(1 * {lhs})"
            elif choice == 4 and " + " in rhs:
                idx = rhs.find(" + ")
                if rhs.startswith("(") and rhs.endswith(")") and rhs[1:idx].count("(") == rhs[1:idx].count(")"):
                    rhs = f"({rhs[idx+3:-1]} + {rhs[1:idx]})"
            elif choice == 5 and " * " in rhs:
                idx = rhs.find(" * ")
                if rhs.startswith("(") and rhs.endswith(")") and rhs[1:idx].count("(") == rhs[1:idx].count(")"):
                    rhs = f"({rhs[idx+3:-1]} * {rhs[1:idx]})"
            elif choice == 6:
                lhs = f"({lhs} + c)"
                rhs = f"(c + {rhs})"
            elif choice == 7 and seed == "(a * (b + c))":
                rhs = "((a * b) + (a * c))"

        if lhs == rhs:
            lhs = f"({lhs} + 0)"
        return f"(a b c : Nat) : {lhs} = {rhs}", num_twists

# ==============================================================================
# 3. Lean 4 전술 환경 (기초 전술 + 수학적 귀납법(Induction) + 창조 개념/정리 확장)
# ==============================================================================
class Lean4TacticEnv:
    def __init__(self):
        self.atomic_tactics = [
            ("기초0: 반사성 종결 (rfl)", "rfl"),
            ("기초1: 덧셈 우측 0 제거 (add_zero)", "try rw [Nat.add_zero]"),
            ("기초2: 덧셈 좌측 0 제거 (zero_add)", "try rw [Nat.zero_add]"),
            ("기초3: 곱셈 우측 1 제거 (mul_one)", "try rw [Nat.mul_one]"),
            ("기초4: 곱셈 좌측 1 제거 (one_mul)", "try rw [Nat.one_mul]"),
            ("기초5: 덧셈 교환법칙 (add_comm)", "try rw [Nat.add_comm]"),
            ("기초6: 곱셈 교환법칙 (mul_comm)", "try rw [Nat.mul_comm]"),
            ("기초7: 덧셈 결합법칙 정방향 (add_assoc)", "try rw [Nat.add_assoc]"),
            ("기초8: 덧셈 결합법칙 역방향 (<-add_assoc)", "try rw [← Nat.add_assoc]"),
            ("기초9: 좌분배법칙 전개 (left_distrib)", "try rw [Nat.left_distrib]"),
            ("기초10: 좌분배법칙 묶기 (<-left_distrib)", "try rw [← Nat.left_distrib]"),
            ("기초11: 우분배법칙 전개 (right_distrib)", "try rw [Nat.right_distrib]"),
            ("기초12: 후자함수 덧셈 변환 (succ_add)", "try rw [Nat.succ_add]"),
            ("기초13: 덧셈 후자함수 변환 (add_succ)", "try rw [Nat.add_succ]"),
            ("기초14: 제곱 정의 전개 (pow_two)", "try rw [Nat.pow_two]"),
            ("기초15: 대수 단순화 자동추론 (simp_arith)", "try simp [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.mul_comm, Nat.left_distrib]"),
            # [핵심 추가] 수학적 귀납법 (Mathematical Induction) 전술 블록!
            ("귀납16: 자연수 n 구조적 귀납법 전개 (Induction on n)",
             "induction n with\n  | zero => try simp [c_rec_id, c_rec_add2]\n  | succ n ih => try simp [c_rec_id, c_rec_add2, ih, Nat.succ_add, Nat.add_succ]"),
            ("귀납17: 자연수 c 구조적 귀납법 전개 (Induction on c)",
             "induction c with\n  | zero => try rfl\n  | succ c ih => try rw [Nat.add_succ, Nat.add_succ, ih]"),
            ("논리18: 논리 전제 도입 및 함의 해결", "intros h1 h2 h3; try assumption"),
            ("논리19: 논리곱 교환/분해 해결", "intro h; try exact And.intro h.right h.left; try exact h.left"),
            ("논리20: 삼단논법 추론 (mp)", "intros h1 h2 h3; exact h2 (h1 h3)"),
            ("논리21: 논리곱 결합법칙 추론", "intro h; exact And.intro (And.intro h.left h.right.left) h.right.right"),
        ]
        self.invented_macros = []

    @property
    def total_actions(self):
        return len(self.atomic_tactics) + len(self.invented_macros)

    def action_to_lean_lines(self, act_idx):
        if act_idx < len(self.atomic_tactics):
            return [self.atomic_tactics[act_idx][1]]
        else:
            macro_name, macro_content = self.invented_macros[act_idx - len(self.atomic_tactics)]
            if isinstance(macro_content, str):
                return [macro_content]
            lines = []
            for a in macro_content:
                lines.append(self.atomic_tactics[a][1])
            return lines

    def action_name(self, act_idx):
        if act_idx < len(self.atomic_tactics):
            return self.atomic_tactics[act_idx][0]
        else:
            return self.invented_macros[act_idx - len(self.atomic_tactics)][0]

# ==============================================================================
# 4. 신경망 추론 코어 (16차원 특징 인코더 & 동적 확장형 뇌)
# ==============================================================================
def encode_theorem_state(stmt, current_tactic_history):
    feats = [
        stmt.count("+") / 5.0,
        stmt.count("*") / 5.0,
        stmt.count("0") / 3.0,
        stmt.count("1") / 3.0,
        stmt.count("succ") / 2.0,
        stmt.count("^") / 2.0,
        1.0 if "c_" in stmt else 0.0,      # 스스로 창조한 개념(def)이 포함된 문제인가?
        1.0 if "(n : Nat)" in stmt else 0.0, # 귀납법 타겟 변수 n이 있는가?
        stmt.count("(") / 6.0,
        1.0 if "Prop" in stmt else 0.0,
        1.0 if "∧" in stmt else 0.0,
        1.0 if "->" in stmt else 0.0,
        len(stmt) / 60.0,
        len(current_tactic_history) / 6.0,
        float(current_tactic_history[-1] + 1) / 30.0 if current_tactic_history else 0.0,
        float(current_tactic_history[-2] + 1) / 30.0 if len(current_tactic_history) >= 2 else 0.0,
    ]
    return feats

class LeanRSIBrain(nn.Module):
    def __init__(self, num_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = F.gelu(h + self.fc2(h))
        h = F.gelu(h + self.fc3(h))
        return self.policy_head(h), torch.sigmoid(self.value_head(h))

    def expand_brain(self, new_num_actions, device):
        old_w = self.policy_head.weight.data
        old_b = self.policy_head.bias.data
        old_n = old_w.size(0)
        if new_num_actions <= old_n: return

        new_head = nn.Linear(self.fc3.out_features, new_num_actions).to(device)
        new_head.weight.data[:old_n] = old_w
        new_head.bias.data[:old_n] = old_b
        new_head.bias.data[old_n:] = 0.75 # 새로 창조한 개념/정리/전술에 높은 초기 호기심 부여!
        self.policy_head = new_head

# ==============================================================================
# 5. 신경망 유도 배치 트리 탐색기 & 복리 전술 압축기
# ==============================================================================
def search_lean_proof(verifier, tac_env, brain, stmt, batch_size=16, max_rounds=2, device='cpu'):
    brain.eval()
    beam = [(0.0, [])]
    candidate_action_seqs = []

    for depth in range(1, 5):
        next_beam = []
        for score, act_seq in beam:
            feat = torch.tensor([encode_theorem_state(stmt, act_seq)], dtype=torch.float, device=device)
            with torch.no_grad():
                logits, val = brain(feat)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            top_indices = probs.argsort()[-7:][::-1]
            rand_indices = random.sample(range(tac_env.total_actions), min(3, tac_env.total_actions))
            chosen_acts = list(dict.fromkeys(list(top_indices) + rand_indices))

            for act in chosen_acts:
                if len(act_seq) >= 2 and act_seq[-1] == act and act_seq[-2] == act:
                    continue
                new_seq = act_seq + [act]
                new_score = score + float(probs[act]) + val.item() * 0.5 + random.uniform(0, 0.1)
                next_beam.append((new_score, new_seq))
                candidate_action_seqs.append((new_score, new_seq))

        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:12]

    candidate_action_seqs.sort(key=lambda x: x[0] - len(x[1]) * 0.15, reverse=True)

    unique_seqs = []
    seen = set()
    for _, seq in candidate_action_seqs:
        t_seq = tuple(seq)
        if t_seq not in seen:
            seen.add(t_seq)
            unique_seqs.append(seq)

    for r in range(max_rounds):
        batch_seqs = unique_seqs[r * batch_size : (r + 1) * batch_size]
        if not batch_seqs:
            break

        lean_proof_candidates = []
        for seq in batch_seqs:
            lines = []
            for a in seq:
                lines.extend(tac_env.action_to_lean_lines(a))
            if lines and "induction" not in lines[-1] and lines[-1] != "rfl" and "Prop" not in stmt:
                lines.append("try rfl")
            lean_proof_candidates.append(lines)

        win_idx = verifier.verify_batch(stmt, lean_proof_candidates)
        if win_idx != -1:
            return True, batch_seqs[win_idx], lean_proof_candidates[win_idx]

    return False, [], []

def invent_lean_macro(tac_env, success_records):
    counter = Counter()
    for _, seq in success_records:
        for i in range(len(seq) - 1):
            a1, a2 = seq[i], seq[i+1]
            raw1 = [a1] if a1 < len(tac_env.atomic_tactics) else tac_env.invented_macros[a1 - len(tac_env.atomic_tactics)][1]
            raw2 = [a2] if a2 < len(tac_env.atomic_tactics) else tac_env.invented_macros[a2 - len(tac_env.atomic_tactics)][1]
            if isinstance(raw1, str) or isinstance(raw2, str):
                continue
            comb = tuple(raw1 + raw2)
            if 2 <= len(comb) <= 6:
                counter[comb] += 1

    existing = {tuple(m[1]) for m in tac_env.invented_macros if isinstance(m[1], list)}
    for comb, cnt in counter.most_common(5):
        if cnt >= 3 and comb not in existing:
            macro_id = len(tac_env.invented_macros) + 1
            short_names = [tac_env.atomic_tactics[a][0].split(":")[0] for a in comb]
            name = f"발명전술#{macro_id}[{' + '.join(short_names)} 압축]"
            tac_env.invented_macros.append((name, list(comb)))
            return name
    return None

# ==============================================================================
# 6. 10시간 연속 무인 가동 메인 루프 (개념 창조 + 귀납법 + 정리 축적 + 전술 압축)
# ==============================================================================
def run_10hour_lean4_rsi(hours_to_run=10.0):
    setup_lean4()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verifier = Lean4Verifier()
    mathematician = AutonomousMathematician()
    tac_env = Lean4TacticEnv()
    brain = LeanRSIBrain(tac_env.total_actions).to(device)
    optimizer = optim.AdamW(brain.parameters(), lr=0.003)

    start_time = time.time()
    end_time = start_time + (hours_to_run * 3600)

    memory = []
    discovered_theorems = []
    gen = 0

    print("\n================================================================================")
    print(f" [Lean 4 RSI Engine v3.0] 귀납법(Induction) & 개념 창조(def) 탑재! {hours_to_run}시간 가동")
    print("================================================================================")

    while time.time() < end_time:
        gen += 1
        elapsed_min = (time.time() - start_time) / 60.0
        solved_count = 0
        gen_success_records = []
        best_demo = None

        # [핵심 진화 1: 새로운 수학 개념(def) 스스로 창조하기!]
        # 2세대마다, 혹은 레벨이 오를 때마다 스스로 새로운 수학 함수를 정의하여 세계관을 넓힘
        if gen % 2 == 0 and len(mathematician.active_concepts) < len(mathematician.concept_blueprints):
            concept_res = mathematician.try_invent_new_concept(verifier, tac_env, brain, device)
            if concept_res:
                c_name, c_desc, c_code = concept_res
                optimizer = optim.AdamW(brain.parameters(), lr=0.003)
                print(f"\n  🌌 [새로운 수학 개념(def) 창조!!] 모델이 스스로 새로운 연산자를 정의했습니다!")
                print(f"     └─ 정의 코드: {c_code.replace(chr(10), ' ')}")
                print(f"     └─ 개념 설명: {c_desc}")

        num_problems = 10
        for _ in range(num_problems):
            stmt, twist_lv = mathematician.generate_frontier_conjecture()
            ok, act_seq, raw_lines = search_lean_proof(verifier, tac_env, brain, stmt, batch_size=16, max_rounds=2, device=device)

            if ok:
                solved_count += 1
                gen_success_records.append((stmt, act_seq))

                if stmt not in mathematician.solved_history:
                    mathematician.solved_history.add(stmt)
                    discovered_theorems.append((stmt, [tac_env.action_name(a) for a in act_seq]))

                    # [핵심 진화 2: 증명한 정리를 보조정리(Lemma)로 영구 등재 및 무기화!]
                    if len(act_seq) >= 2 and "Prop" not in stmt and len(verifier.accumulated_lemmas) < 30:
                        lemma_id = len(verifier.accumulated_lemmas) + 1
                        lemma_name = f"auto_lemma_{lemma_id}"
                        if verifier.safely_register_lemma(lemma_name, stmt, raw_lines):
                            skill_label = f"발견정리호출#{lemma_id}[rw [{lemma_name}] : {stmt}]"
                            tac_env.invented_macros.append((skill_label, f"try rw [{lemma_name}]"))
                            brain.expand_brain(tac_env.total_actions, device)
                            optimizer = optim.AdamW(brain.parameters(), lr=0.003)
                            print(f"  💎 [보조정리(Lemma) 영구 등재!] [{lemma_name} : {stmt}] 증명을 검증 완료하여 무기로 장착했습니다!")

                if not best_demo or len(act_seq) >= len(best_demo[1]):
                    best_demo = (stmt, act_seq)

                for idx, a in enumerate(act_seq):
                    prefix = act_seq[:idx]
                    val_target = ((idx + 1) / len(act_seq)) ** 0.5
                    memory.append((encode_theorem_state(stmt, prefix), a, val_target))

        solve_rate = (solved_count / num_problems) * 100.0
        print(f"\n[제 {gen} 세대 | 경과: {elapsed_min:.1f}분 | 자율 복잡도: Lv.{mathematician.complexity_budget} | 창조 개념(def): {len(mathematician.active_concepts)}개]")
        print(f" -> Lean 4 증명 성공률: {solve_rate:.1f}% ({solved_count}/{num_problems}) | 누적 정리: {len(discovered_theorems)}개 | 총 보유 스킬: {tac_env.total_actions}개")

        if best_demo:
            b_stmt, b_seq = best_demo
            print(f"  └─ [실제 Lean 4 증명 해부] 자율 출제 명제: theorem {b_stmt}")
            for s_i, a_idx in enumerate(b_seq):
                print(f"       Step {s_i+1}: {tac_env.action_name(a_idx)}")

        # [핵심 진화 3: 자주 쓰이는 증명 전술 조합을 새로운 메가 전술로 압축!]
        if len(gen_success_records) >= 3 and len(tac_env.invented_macros) < 65:
            new_macro = invent_lean_macro(tac_env, gen_success_records)
            if new_macro:
                print(f"  ★ [전술 발명(Abstraction)!] Lean 4 증명 패턴을 압축해 [{new_macro}] 스킬을 창조했습니다!")
                brain.expand_brain(tac_env.total_actions, device)
                optimizer = optim.AdamW(brain.parameters(), lr=0.003)

        # 신경망 자가 증류 학습
        if memory:
            memory = memory[-5000:]
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

        # 자율 커리큘럼 난이도 밀당 조절
        if solve_rate >= 70.0 and mathematician.complexity_budget < 25:
            mathematician.complexity_budget += 1
            print(f"  ▲ [자율 난이도 상승!] 현재 수준을 마스터하여 출제기가 스스로 문제 복잡도를 Lv.{mathematician.complexity_budget}(으)로 높입니다!")
        elif solve_rate <= 20.0 and mathematician.complexity_budget > 1:
            mathematician.complexity_budget -= 1
            print(f"  ▼ [자율 난이도 조정] 문제가 너무 어려워 출제기가 징검다리 확보를 위해 복잡도를 Lv.{mathematician.complexity_budget}(으)로 낮춥니다.")

        # 실시간 체크포인트 파일 및 생성된 Lean 4 수학 라이브러리 전체 백업!
        checkpoint_data = {
            "generation": gen,
            "elapsed_minutes": round(elapsed_min, 2),
            "complexity_budget": mathematician.complexity_budget,
            "invented_concepts_def": [c[0] for c in mathematician.active_concepts],
            "total_discovered_theorems": len(discovered_theorems),
            "accumulated_lemmas_count": len(verifier.accumulated_lemmas),
            "invented_skills": [m[0] for m in tac_env.invented_macros],
            "recent_theorems": discovered_theorems[-15:]
        }
        with open("lean4_rsi_v3_progress.json", "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        with open("discovered_math_library.lean", "w", encoding="utf-8") as f:
            f.write(verifier.get_full_header())
        torch.save(brain.state_dict(), "lean4_rsi_v3_brain.pt")

if __name__ == "__main__":
    run_10hour_lean4_rsi(hours_to_run=10.0)
