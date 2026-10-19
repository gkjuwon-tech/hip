# Red Team Critique: Expert-Iteration RSI Engine (Search Proposal v1)

**Target Document**: `work/proposal_search_v1.md`  
**Target System**: 79.8M Gated DeltaNet MCTS Reasoner ("Aphasic Genius")  
**Reviewer Persona**: Top ML/RL Researcher, Kaggle Grandmaster (T4/TPU Limits), and Benchmark-Integrity Auditor  
**Binding Directives**: AUDIT_FROM_CLAUDE.md (§0, §1, §2) and BRIEF.md (§2, §4 Phase B)  
**Strict Prohibition**: NO CODE anywhere in this document (equations, arithmetic tables, ASCII schemas, and numbered plain-language steps only).

---

## 1. Executive Summary & Audit Scorecard

The Search Proposal (`proposal_search_v1.md`) presents an ambitious vision: an AlphaZero-style expert-iteration engine coupling an attention-free Gated DeltaNet core (<100M parameters) with Monte Carlo Tree Search (MCTS) and 0-parameter exact verifiers. 

However, beneath its elegant surface lies a collection of **fatal mathematical fallacies, physical latency impossibilities, severe RSI blind spots, and citation laundering**. Most egregiously:
1. **Recurrent depth ($r \in [1, 8]$) was completely omitted from all FLOP and throughput arithmetic**, underestimating rollout and training compute by **2.66× to 4.88×**.
2. **Sequential autoregressive decoding latency on NVIDIA T4 was computed assuming peak batch tensor throughput**, underestimating token generation time by **over 300×**. A 64-simulation MCTS pass takes **2.0 to 4.5 seconds**, not 0.047 seconds. Consequently, one generation takes **8.7 to 14.0 hours**, not 1.07 hours, destroying the claimed "14 generations per week."
3. **The Reading Verifier Hole (AUDIT §1.4) is wide open**: the engine verifies program execution, but has zero mechanism to verify that the generated program correctly formalizes the natural-language input. The model can hallucinate or overfit reading while receiving 100% verifier reward.
4. **Experiment 1 contains 80 lines of banned Python code** (violating AUDIT §0), which constitutes a hardcoded template-matching facade rather than an actual neural learning experiment.
5. **Multiple instances of citation laundering, reverse-engineered scoreboard targets, unjustified probabilities, and omitted parameters** violate the mandatory project rules.

### Red Team Audit Scorecard

| Area | Primary Weakness | Severity | Verdict |
|---|---|---|---|
| **RSI Engine & MCTS** | Reading verifier hole; MCTS cold-start stall; own-level prison; hand-waved domain ascension | **FATAL** | **REJECT AS CURRENTLY SPECIFIED** |
| **FLOP & Quota Math** | Recurrent depth $r$ omitted ($2.66\times - 4.88\times$ error); 5B pretrain compute understated | **FATAL** | **MUST RECALCULATE WITH REAL $r$** |
| **Inference Latency** | Memory-bandwidth bound ignored; claimed 0.0007s/pass vs physical 0.213s on T4 | **FATAL** | **PHYSICALLY IMPOSSIBLE ON T4** |
| **Missing Brief Items** | No-teacher ablation missing; outdated frontier names; lumped PrOntoQA/MuSR; 405B teacher | **MAJOR** | **MUST RESTORE & COMPLIANCE-FIX** |
| **Scoreboard & ARC-AGI** | ARC-AGI target 46.5% exceeds attention-free evidence (29.6%); unverified $T$ tags; fake probabilities | **MAJOR** | **MUST LABEL [추정]/[가설] & DOWNGRADE** |
| **Parameter Table** | Omitted DROP pointer head, formal input embeddings, short convs, reader layer doubling | **MAJOR** | **RECOUNT REQUIRED (STAYS <100M)** |
| **Citation Integrity** | Laundering on distillation, Stitch, AlignScore, Gated DeltaNet math, fake dataset names | **MAJOR** | **MUST RETRACT & REVISE CITATIONS** |
| **Experiment 1 & Code** | 80 lines of banned Python code; hardcoded template facade | **FATAL** | **PURGE CODE; REPLACE WITH PLAN** |
| **12-Week Feasibility** | Scope explosion (Lean 4 + Z3 + SymPy + MCTS + JAX + Triton + Stitch) for 1 person | **MAJOR** | **PRUNE TO STAGED S-CURVE MVP** |

---

## 2. Attack 1: The RSI Engine & Self-Play Loopholes

### 2.1 The Deepest Hole: The Reading Verifier Void (AUDIT §1.4)
- **Severity**: **FATAL**
- **Technical Reasoning**:  
  The proposal relies on Python, SymPy, Z3, and Lean 4 sandboxes returning binary rewards $z \in \{-1, +1\}$. In §1.2 and §2.1, it asserts that mechanical verifiers provide "infinite-fuel ground truth."  
  This is true for self-contained games (Go, Chess) and pure formal systems (Lean theorems with formal premises). However, the Aphasic Genius core must solve natural-language problems (GSM8K, SVAMP, DROP, MuSR, StrategyQA, KMMLU).  
  The external sandbox checks only that `eval(program)` runs without throwing an exception and returns a numeric or symbolic value matching an expected output. **It never checks that the emitted program faithfully formalizes the natural-language question.**  
  Inside an autonomous proposer-solver self-play loop without human labels:
  - The proposer emits a problem description $x_{\text{text}}$ and a target constraint.
  - The solver emits a formal program $P$.
  - The sandbox executes $P$.
  If the solver develops an inductive bias that misinterprets complex linguistic clauses (e.g., distractor conditions in GSM-Symbolic or negation in MuSR) but the proposer co-adapts by emitting problem texts where that flawed formalization happens to execute successfully, the system receives full positive reward $z = +1$.  
  The core is an *aphasic reasoner*: it cannot explain its reading in natural language. Thus, the RSI loop reward-hacks reading comprehension, achieving 99% in-loop verified accuracy while catastrophic comprehension collapse occurs on held-out human-written problems.
- **Concrete Fix (Preserving Ambition)**:  
  Enforce a mandatory **Tri-Verification Reading Harness** for all in-loop natural-language tasks:
  1. *Program-First Reverse Generation*: The procedural generator must generate the formal ground-truth AST / program $P_{\text{gold}}$ and target answer *first*, and then render it into natural language $x_{\text{text}}$ via deterministic grammars and certified templates. For in-loop training, the core's emitted AST $P_{\text{pred}}$ is verified not only by execution outcome, but by **semantic equivalence to $P_{\text{gold}}$** using SymPy / Z3 equivalence checks.
  2. *Paraphrase Consistency Check*: Every proposed problem is rendered into three linguistically divergent surface forms $(x_1, x_2, x_3)$. The solver must produce formally equivalent programs across all three variations. Any variance triggers $z = -1$.
  3. *Perturbation Twin Gating*: Apply systematic number-swapping and distractor clause insertion (GSM-Plus / GSM-Symbolic protocol). The solver's emitted program must exhibit the exact mathematically predicted functional change.

### 2.2 The Own-Level Prison & The Zero-Density Cold-Start Trap
- **Severity**: **FATAL**
- **Technical Reasoning**:  
  In §2.1, the proposer reward is formulated as:
  $$R_{\text{proposer}} = \mathbb{I}(\text{solvable}) \cdot \left(1 - \text{Accuracy}_{\text{solver}}(x_{\text{spec}})\right)$$
  This formulation creates two catastrophic dynamic failure modes:
  1. *The Zero-Density Trap (Cold-Start Stall)*: If the proposer generates a problem requiring an unseen reasoning primitive (e.g., ascending from arithmetic to quadratic polynomials, or introducing Lean tactic chaining), the solver's accuracy is $0.0\%$. Under the indicator function $\mathbb{I}(\text{solvable})$, who proves the problem is solvable? If the solver cannot solve it, solvability is unproven, yielding $R_{\text{proposer}} = 0$. The proposer receives zero learning signal to guide it toward the solvable frontier.
  2. *The Trivial Modification Basin (The Own-Level Prison)*: The proposer quickly discovers that it can reliably harvest maximal reward ($R \approx 0.5$) by introducing trivial variations (such as adding arbitrary constants or permuting variable names) to problems the solver solves with $50\%$ probability. Venturing outside this comfort zone risks $0\%$ solver accuracy and zero reward.
  3. *MCTS Sampling Collapse on Hard Formal Domains*: In game domains, random play reaches a terminal state with a win/loss signal. In Lean 4 theorem proving or LiveCodeBench algorithmic coding, random token exploration has a probability of hitting a valid proof or passing unit tests of literally $P < 10^{-6}$. With only 64 MCTS simulations, the search will return $z = -1$ on 100% of rollouts. The expert replay buffer $\mathcal{D}_g^*$ remains empty, and Expert Iteration completely halts.
- **Concrete Fix (Preserving Ambition)**:  
  1. *Adopt Depth Regret*: Replace the fragile proposer reward with parameter-free **Depth Regret**:
     $$R_{\text{depth}} = \max\left(0, \text{Score}_{\text{MCTS\_deep}}(x) - \text{Score}_{\text{Policy\_shallow}}(x)\right)$$
     This guarantees that the curriculum selects problems that the current policy *cannot* solve greedily, but which MCTS search *can* solve.
  2. *Dual-Curriculum Injection*: Mandate that exactly $50\%$ of tasks per generation are drawn from frozen, stepped procedural generators (Reasoning Gym, re-ARC, SymPy generators) with fixed difficulty ladders. This forcibly pulls the model out of its own-level prison.
  3. *Backward Goal-Decomposition for Formal Domains*: For Lean 4 and code synthesis, replace naive forward MCTS with bidirectional proof-state decomposition. The proposer generates problems by applying known valid tactics in reverse from terminal axioms, guaranteeing non-zero solvability and dense intermediate rewards.

### 2.3 Hand-Waved "Domain Ascension" (AUDIT §1.3)
- **Severity**: **MAJOR**
- **Technical Reasoning**:  
  In §2.4, the proposal defines domain transition as: "1. Compositional Synthesis; 2. Autoformalization Mapping; 3. Zero-Parameter Simulator Ingestion."  
  This is a purely descriptive human wishlist. BRIEF.md §1 Decision 3 mandates that finding or building the next verifier domain is *itself an RSI skill*. Furthermore, BRIEF.md §2.7 forbids the model from writing or modifying its own verifiers.  
  Autoformalization without an exact equivalence checker is a hallucination factory: the proposal states "validating syntax via Lean compiler." Syntactically valid Lean code can formalize complete nonsense (e.g., proving `True = True` instead of Fermat's Last Theorem). Compiling syntax does not verify semantic alignment with informal source text.
- **Concrete Fix**:  
  Define an explicit, safe, autonomous domain transition protocol:
  - Pre-package an isolated, read-only **Domain Registry** containing sandbox environments (Arithmetic $\to$ Symbolic CAS $\to$ Z3 SMT $\to$ Lean 4 $\to$ PlanBench VAL $\to$ RDKit).
  - The model does not invent code for new verifiers (preventing reward tampering). Instead, the model's meta-action space includes **Domain Bridging Programs**: emitting typed reduction compilers that map problems from a new unverified domain $D_{k+1}$ into equivalent representations in verified base domains $D_1 \dots D_k$ (e.g., reducing graph coloring to a Z3 SAT instance).
  - Domain ascension is triggered when the Saturation Metric $\Psi = \frac{\text{pass@1}}{\text{pass@64}} \times (1 - \Delta_{\text{DL}} / \text{DL}) \ge 0.85$ holds for two consecutive generations.

---

## 3. Attack 2: Recurrent Depth FLOP Recalculation (AUDIT §1.1)

### 3.1 The Missing Recursion Multiplier
- **Severity**: **FATAL**
- **Technical Reasoning**:  
  In §2.5 and §8.2, the proposal computes compute cost using standard Transformer single-pass formulas:
  $$\text{FLOPs/token}_{\text{inference}} = 2N, \quad \text{FLOPs/token}_{\text{training}} = 6N$$
  with $N = 79.82\text{M}$, yielding $1.596 \times 10^8$ and $4.788 \times 10^8$ FLOPs/token respectively.  
  However, §5.3 explicitly specifies that the reasoning core is an **8-layer weight-tied Gated DeltaNet iterated across recurrent depth $r \in [1, 8]$**.  
  AUDIT §1.1 mandates the correct arithmetic:
  $$\text{FLOPs/token}_{\text{inference}} \approx 2 \times \left(N_{\text{non\_core}} + r \times N_{\text{core}}\right)$$
  $$\text{FLOPs/token}_{\text{training}} \approx 6 \times \left(N_{\text{non\_core}} + r \times N_{\text{core}}\right)$$
  Let us decompose the parameter budget from Table 5.3:
  - Core Trunk Parameters: $N_{\text{core}} = 44,252,160 \approx 44.25\text{M}$.
  - Non-Core Parameters (Embeddings, Reader, Heads, Key Proj):  
    $N_{\text{non\_core}} = 79,817,985 - 44,252,160 = 35,565,825 \approx 35.57\text{M}$.

Let us calculate the true computational cost per token across recursion depths:

```
+----------------------------------------------------------------------------------------------------+
| Recurrent Depth | Effective Params (N_eff) | Inference FLOPs/tok | Training FLOPs/tok | Understatement Factor |
+----------------------------------------------------------------------------------------------------+
| r = 1 (Proposal)| 79.82 M                  | 1.596 x 10^8        | 4.789 x 10^8       | 1.00x (Baseline)      |
| r = 2           | 124.07 M                 | 2.481 x 10^8        | 7.444 x 10^8       | 1.55x                 |
| r = 4 (Median)  | 212.57 M                 | 4.251 x 10^8        | 1.275 x 10^9       | 2.66x                 |
| r = 8 (Full)    | 389.58 M                 | 7.792 x 10^8        | 2.337 x 10^9       | 4.88x                 |
+----------------------------------------------------------------------------------------------------+
```

### 3.2 Recalculation of Generation Rollout Compute
In §2.5, the proposal budgets:
$$\text{Tokens} = 25,000 \text{ tasks} \times 64 \text{ MCTS sims} \times 150 \text{ tokens} = 2.40 \times 10^8 \text{ tokens}$$
The proposal claimed this requires $3.83 \times 10^{16}$ FLOPs, taking $1,178\text{ seconds} \approx 0.33\text{ hours (20 min)}$ on 2× T4 GPUs at 25% MFU (32.5 TFLOPS sustained).

Recalculating with real recurrent depth:
- **At median recursion ($r = 4$)**:
  $$\text{FLOPs} = 2.40 \times 10^8 \times 4.251 \times 10^8 = 1.020 \times 10^{17} \text{ FLOPs}$$
  $$\text{Time}_{\text{T4x2}} = \frac{1.020 \times 10^{17} \text{ FLOPs}}{3.25 \times 10^{13} \text{ FLOP/s}} = 3,139 \text{ seconds} \approx \mathbf{0.872 \text{ hours (52 min)}}$$
- **At full reasoning recursion ($r = 8$)**:
  $$\text{FLOPs} = 2.40 \times 10^8 \times 7.792 \times 10^8 = 1.870 \times 10^{17} \text{ FLOPs}$$
  $$\text{Time}_{\text{T4x2}} = \frac{1.870 \times 10^{17} \text{ FLOPs}}{3.25 \times 10^{13} \text{ FLOP/s}} = 5,754 \text{ seconds} \approx \mathbf{1.60 \text{ hours (96 min)}}$$

### 3.3 Recalculation of Expert Iteration Training Compute
In §2.5, training is budgeted at 20M verified tokens:
- At $r = 4$: $\text{FLOPs} = 20 \times 10^6 \times 1.275 \times 10^9 = 2.551 \times 10^{16} \text{ FLOPs}$.
  On Kaggle TPU v5e-8 at 30% MFU ($472.8 \text{ TFLOPS}$ sustained):
  $$\text{Time}_{\text{TPU}} = \frac{2.551 \times 10^{16}}{4.728 \times 10^{14}} = 53.95 \text{ seconds} \approx \mathbf{0.015 \text{ hours}}$$
  On Kaggle 2× T4: $785 \text{ seconds} \approx \mathbf{0.218 \text{ hours}}$.
- At $r = 8$: $\text{FLOPs} = 20 \times 10^6 \times 2.337 \times 10^9 = 4.675 \times 10^{16} \text{ FLOPs}$.
  On TPU v5e-8: $98.88 \text{ seconds} \approx \mathbf{0.027 \text{ hours}}$.
  On 2× T4: $1,438 \text{ seconds} \approx \mathbf{0.400 \text{ hours}}$.

### 3.4 Recalculation of Stage 0 Bootloader Pretraining
In §8.2, pretraining on 5.0 Billion tokens was claimed to consume:
$$6 \times 7.98 \times 10^7 \times 5.0 \times 10^9 = 2.39 \times 10^{18} \text{ FLOPs} \implies 1.40 \text{ hours on TPU v5e-8}$$
If pretraining trains the recurrent core under depth supervision (e.g., mean $r = 4$):
$$\text{FLOPs} = 6 \times 2.126 \times 10^8 \times 5.0 \times 10^9 = 6.377 \times 10^{18} \text{ FLOPs}$$
$$\text{Time}_{\text{TPU}} = \frac{6.377 \times 10^{18}}{4.728 \times 10^{14}} = 13,488 \text{ seconds} \approx \mathbf{3.75 \text{ hours}}$$
At $r = 8$: $\text{FLOPs} = 1.169 \times 10^{19} \implies \mathbf{6.86 \text{ hours}}$.  
While 3.75–6.86 hours still fits within the weekly 20-hour TPU quota, it represents a **2.68× to 4.90× increase** over the proposal's naive claim. The proposal designer completely omitted this.

- **Concrete Fix**:
  Update all FLOP tables, rollout timers, and pretraining schedules to explicitly state $r_{\text{mean}} = 4$ and $r_{\text{max}} = 8$. Adjust the rollout token allocation per generation (e.g., scale tasks from 25,000 to 12,500 with adaptive tree depth) to keep compute strictly bounded.

---

## 4. Attack 3: Latency Realism & Memory-Bandwidth Bounds on T4 (AUDIT §1.2)

### 4.1 The Myth of 0.0007-Second Generation
- **Severity**: **FATAL**
- **Technical Reasoning**:  
  In §8.3, the proposal states:
  > "Greedy Pass (150 tokens): $150 \times 1.596 \times 10^8 \approx 2.39 \times 10^{10} \text{ FLOPs} \approx 0.0007 \text{ seconds on T4}$."  
  > "Snell Adaptive MCTS Search (64 simulations): $64 \times 2.39 \times 10^{10} \approx 1.53 \times 10^{12} \text{ FLOPs} \approx 0.047 \text{ seconds on T4}$."

  This calculation is an utter physical impossibility. The author took total FLOPs and divided by peak sustained GPU FLOP/s:
  $$\frac{2.39 \times 10^{10} \text{ FLOPs}}{3.25 \times 10^{13} \text{ FLOP/s}} = 0.000735 \text{ seconds}$$
  This treats sequential autoregressive decoding as if it were a single dense matrix multiplication running at peak batched tensor-core efficiency.  
  As AUDIT §1.2 explicitly notes, **sequential token-by-token generation is strictly memory-bandwidth bound and kernel-launch bound**.
  
  Let us compute the true physical latency on an NVIDIA T4 GPU:
  - **Memory Bandwidth**: Peak theoretical GDDR6 bandwidth is $300 \text{ GB/s}$. Realistic sustained streaming bandwidth on Turing is $\approx 230 \text{ GB/s}$.
  - **Bytes Read per Token at Batch Size 1**:  
    Generating a single token requires streaming all active weights from HBM to SRAM.  
    At fp16 precision ($2 \text{ bytes/weight}$):
    - Non-core weights: $35.57\text{M} \times 2 = 71.14 \text{ MB}$.
    - Recurrent core weights: $44.25\text{M} \times 2 = 88.50 \text{ MB}$, read $r$ times per token.
    - At $r = 4$: Total weight memory read per token:
      $$B_{\text{token}} = 71.14\text{ MB} + 4 \times 88.50\text{ MB} = 425.14 \text{ MB} \approx 0.425 \text{ GB}$$
  - **Minimum Memory-Read Time per Token**:
    $$T_{\text{mem}} = \frac{0.425 \text{ GB}}{300 \text{ GB/s}} \approx 1.417 \text{ ms/token}$$
    At realistic bandwidth ($230 \text{ GB/s}$): $T_{\text{mem}} \approx 1.848 \text{ ms/token}$.
  - **Kernel Launch & Driver Overhead**:  
    Each Gated DeltaNet layer requires chunked associative scans, short 1D convolutions, RMSNorm operations, and gate projections. This amounts to at least 8 kernel launches per layer. Across 4 reader layers and 8 core layers iterated $r=4$ times, a single step launches dozens of CUDA kernels. On Kaggle's shared Xeon host, kernel launch latency contributes at least $0.15 \text{ ms}$ per step.
  - **True Single-Sequence Greedy Latency (150 tokens)**:
    $$T_{\text{greedy}} = 150 \text{ tokens} \times (1.42 \text{ ms} + 0.15 \text{ ms}) \approx \mathbf{0.235 \text{ seconds (235 ms)}}$$
  The proposal's claimed $0.0007 \text{ seconds}$ is off by a factor of **$335\times$**!

### 4.2 The True Cost of 64-Simulation MCTS
In MCTS, tree expansions are sequential or batched across small branch frontiers.
- Even if simulations are batched across 8 parallel worker branches ($B = 8$), generating actions requires sequential node expansion.
- Furthermore, intermediate and terminal verifications require IPC serialization across Unix pipes to the Python/SymPy sandbox running on Kaggle's 4 vCPUs.
- Realistic wall-clock budget for 64 MCTS simulations on a single problem:
  - Batched forward generation ($8 \times 8$ simulations, mean depth 12): $\approx 1.50 \text{ seconds}$.
  - Sandbox verification checks (64 visits, amortized cache hit 50% $\implies 32$ Python/SymPy executions at $3 \text{ ms}$ each): $\approx 0.10 \text{ seconds}$.
  - Tree bookkeeping, PUCT updates, Dirichlet noise injection: $\approx 0.05 \text{ seconds}$.
  - **Realistic Query Latency**: $\mathbf{1.65 \text{ to } 3.00 \text{ seconds}}$ per problem!
  The proposal claimed: **$0.047 \text{ seconds}$** (underestimated by **$35\times$ to $64\times$**).

### 4.3 The Generation Cycle Wall: 14 Hours, Not 1 Hour
If a single problem rollout takes $2.0 \text{ seconds}$ on 2× T4:
- Sustained throughput on 2× T4: $1.0 \text{ problem/second}$.
- For 25,000 tasks:
  $$\text{Time}_{\text{rollout}} = \frac{25,000}{1.0} = 25,000 \text{ seconds} \approx \mathbf{6.94 \text{ hours}}$$
- Adding verification overhead ($\approx 1.5 \text{ hours}$) and training ($\approx 0.2 \text{ hours}$):
  $$\text{Total Generation Time} \approx \mathbf{8.64 \text{ hours}}$$
  If batched less efficiently, it easily reaches **12 to 14 hours**.
- **Fatal Quota Violation**:  
  With Kaggle's 30 GPU-hours/week, running a single generation takes 8.6 to 14.0 hours. You can complete **at most 2 to 3 generations per week**, NOT 14 generations!

- **Concrete Fix (Preserving Ambition)**:  
  1. *Downscale In-Loop Task Volume*: Reduce task count from 25,000 to 5,000 tasks per generation.
  2. *Adaptive Test-Time Simulation Allocation (Snell Scaling)*: Allocate 64 simulations only to ambiguous problems (root value entropy $H(V) > 0.5$). For easy/obvious problems, exit at 8 simulations. Mean simulations drops from 64 to 18.
  3. *Rollout Time Recalculation*:
     $$5,000 \text{ tasks} \times 18 \text{ sims} \times 100 \text{ tokens} = 9.0 \times 10^6 \text{ tokens}$$
     With batched inference on 2× T4, this takes $\approx 1.25 \text{ hours}$. Total cycle time lands at $\approx 2.0 \text{ hours}$, permitting **10 to 12 generations per week** within quota.

---

## 5. Attack 4: Missing Required Elements (AUDIT §1.5)

### 5.1 Missing "No-Teacher" Ablation
- **Severity**: **FATAL**
- **Technical Reasoning**:  
  BRIEF.md §2.6 and AUDIT §1.5 explicitly mandate that every proposal must include an explicit **"no-teacher" ablation** and teacher-data logging protocol.  
  Proposal v1 describes a 5B-token Stage 0 bootloader using OpenMathInstruct-2 and Python-Edu (§8.1), but contains zero mention of how the engine performs when the teacher is withheld from day 1.  
  Without a no-teacher ablation, it is impossible to verify whether the core's capability is genuine autonomous self-improvement or merely residual distillation from the 5B-token teacher corpus.
- **Concrete Fix**:  
  Specify a parallel control arm: **Track 0-NT (No-Teacher)**.
  - Pretraining data consists strictly of procedural generators (CLRS traces, SymPy equations, Reasoning Gym algorithmic instances, re-ARC training tasks) with 0 tokens derived from open LLM teachers.
  - Run identical MCTS Expert Iteration for 5 generations. Report the delta:
    $$\Delta_{\text{teacher\_boost}} = \text{Score}_{\text{bootloaded}}(g) - \text{Score}_{\text{no\_teacher}}(g)$$
  - Quantify the generation $g^*$ at which Track 0-NT overtakes the static teacher baseline.

### 5.2 Outdated Frontier "Tin Can" Names
- **Severity**: **MINOR (Compliance)**
- **Technical Reasoning**:  
  In §1.1 and §7.2, the proposal repeatedly names Claude 3.5 Sonnet, GPT-4o, Claude 3.7 Sonnet (Thinking), and Gemini 2.5 Pro.  
  AUDIT §1.5 explicitly mandates:
  > "Use the brief's names: **Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra**."
- **Concrete Fix**:  
  Replace all occurrences in §1.1, §7.2, and §10 with the mandatory canon names: Claude Opus 5.5, Gemini 4 Argon, and GPT-6 Astra.

### 5.3 Lumped Scoreboard Rows (PrOntoQA and MuSR)
- **Severity**: **MAJOR**
- **Technical Reasoning**:  
  In Table 7.1:
  - Row A12 lumps "FOLIO / ProofWriter / PrOntoQA" into a single combined cell: `62.7 / 71.5 / 85.0 -> 75.2 / 81.5 / 95.0`. PrOntoQA has a very high GPT-3.5 target ($T = 85.0$), requiring $G = 95.0$. Lumping it hides the unique reading-comprehension bottleneck of depth-5 multi-hop deduction.
  - Row A14 lumps "MuSR (MM / OP / TA)" into `70.0 / 58.2 / 78.8 -> 80.0 / 72.1 / 88.8`. AUDIT §1.5 explicitly mandates: **"MuSR must be gated per subtask, not on the average."** Murder Mysteries (MM) involves 1,000-word murder-mystery narrative comprehension, where a 100M encoder faces severe context-compression degradation. Averaging allows high performance on Team Allocation (TA) to mask failure on MM.
- **Concrete Fix**:  
  Split Table 7.1 into separate, dedicated rows:
  - Row A12a: FOLIO ($T = 62.7, G = 75.2, N = 204$).
  - Row A12b: ProofWriter ($T = 71.5, G = 81.5, N = 600$).
  - Row A12c: PrOntoQA ($T = 85.0, G = 95.0, N = 500, \text{CI} = \pm 1.9$).
  - Row A14a: MuSR Murder Mysteries ($T = 70.0, G = 80.0, N = 250, \text{CI} = \pm 5.0$).
  - Row A14b: MuSR Object Placement ($T = 58.2, G = 72.1, N = 256, \text{CI} = \pm 5.5$).
  - Row A14c: MuSR Team Allocation ($T = 78.8, G = 88.8, N = 250, \text{CI} = \pm 3.9$).

### 5.4 The Cord-Cut Teacher Scale Contradiction
- **Severity**: **MAJOR**
- **Technical Reasoning**:  
  In §7.1 and §8.1, the teacher dataset is OpenMathInstruct-2, which was generated by **Llama-3.1-405B**. In §2.3, the cord-cut metric requires evaluating teacher failure:
  $$\Delta_{\text{TE}}(g) = \frac{|\{x \in \mathcal{D}_{\text{held-out}}^{\text{teacher\_fail}} : \text{Solve}(x) = 1\}|}{|\mathcal{D}_{\text{held-out}}^{\text{teacher\_fail}}|}$$
  AUDIT §1.5 explicitly warns:
  > "The cord-cut teacher must be something that can actually run on free compute ($\le 14\text{B}$ on 2×T4, $\sim 27\text{B}$-class on TPU per compute.md), or be evaluated from a fixed, pre-computed teacher-failure set. A 70B teacher cannot run here."
  A 405B teacher cannot be executed to sample $k=8$ solutions per problem on Kaggle T4 or TPU!
- **Concrete Fix**:  
  Formally specify that the teacher benchmark failure split $\mathcal{D}_{\text{held-out}}^{\text{teacher\_fail}}$ is pre-computed offline using **Qwen2.5-Math-7B-Instruct** (runnable on 2× T4 via vLLM) or drawn from a frozen, commit-revealed benchmark failure manifest of an open 14B model.

---

## 6. Attack 5: Scoreboard Integrity, Projections vs Results, ARC-AGI & Unjustified Probabilities (AUDIT §1.6 - §1.8)

### 6.1 Unjustified Probabilities (AUDIT §1.6)
- **Severity**: **MAJOR**
- **Technical Reasoning**:  
  In §7.2, the proposal asserts:
  - "Rung 2: GPT-4 Level Reasoning (Probability: 65%)"
  - "Rung 3: o1 / DeepSeek-R1 Level Reasoning (Probability: 40%)"
  - "Rung 4: Frontier 'Tin Can' Parity (Probability: 25%)"
  - "Rung 5: North Star / Superhuman ASI (Probability: 10%)"
  These numbers are pseudo-scientific decorations pulled out of thin air. AUDIT §1.6 mandates:
  > "Unjustified probabilities: '100% probability', 'P > 98%', '95%', '70% for GPT-4-level' are unsupported and violate the honesty rule. Every probability needs a stated basis or must be replaced with a qualitative risk level plus the experiment that would measure it."
- **Concrete Fix**:  
  Delete all speculative percentage numbers. Replace with **Qualitative Feasibility Ratings** backed by empirical gating criteria:
  - Rung 2 (GPT-4 Parity): Feasibility = Moderate-High. Gating Metric: Solving $>30\%$ on AIME 2024 via Lean 4 proof search.
  - Rung 3 (o1/R1 Parity): Feasibility = Moderate. Gating Metric: Codeforces rating $>1800$ verified under LiveCodeBench execution.
  - Rung 4 (Frontier Tin Cans): Feasibility = Low-Moderate. Gating Metric: Outperforming Claude Opus 5.5 and Gemini 4 Argon on GPQA-Diamond without external retrieval hints.

### 6.2 Projections Disguised as Results (AUDIT §1.7)
- **Severity**: **MAJOR**
- **Technical Reasoning**:  
  In Table 7.1, columns "Expected Gen ($g$)" and "Target Score" present numbers such as "Gen 4: 89.2" (GSM8K), "Gen 4: 76.5" (GSM-Plus), and "Gen 7: 64.8" (MATH).  
  These numbers sit conveniently 1.1 to 2.7 points above the mandatory $G$ threshold ($88.1 \to 89.2$, $74.1 \to 76.5$, $62.1 \to 64.8$). They look reverse-engineered to appear victorious on paper.  
  AUDIT §1.7 explicitly orders:
  > "Expected scores, '95% CI lower bound' columns, and 'passed at generation N' cells must be labeled **[추정]** or **[가설]**. They are projections, not measurements."
- **Concrete Fix**:  
  Add the explicit label **[추정]** or **[가설]** to every cell in Table 7.1 that contains an expected generation, target score, or projected margin. Provide the explicit mathematical extrapolation model (e.g., log-linear scaling over search rollouts) used to derive each score.

### 6.3 ARC-AGI Target Exceeds Attention-Free Evidence (AUDIT §1.8)
- **Severity**: **MAJOR**
- **Technical Reasoning**:  
  In Row A6, the proposal claims a target score of **46.5%** on ARC-AGI-1 public eval at Gen 5.  
  AUDIT §1.8 and `tiny_sota.md` establish the verified state of the art:
  - TRM-Att (7M parameters) achieved 44.6% using **multi-head self-attention**.
  - TRM-MLP (the attention-free variant) collapsed to **29.6%** on ARC-AGI-1 and **0.0%** on Maze-Hard!
  The Search Proposal uses an attention-free Gated DeltaNet backbone. Gated DeltaNet maintains a fixed-size recurrent state matrix $S_t \in \mathbb{R}^{d \times d}$. It lacks associative softmax attention over 2D spatial coordinate tokens.  
  Claiming 46.5% with an attention-free backbone exceeds all published empirical evidence by **+16.9 percentage points** without providing any replacement mechanism for 2D spatial lookup.
- **Concrete Fix**:  
  Either:
  1. Lower the ARC-AGI-1 target score to **32.0% [추정]**, acknowledging the attention-free capacity limit; OR
  2. Specify an explicit 0-parameter non-attention 2D inductive mechanism: a **2D Cellular-Automaton Spatial Convolutional Adapter** (kernel size $3 \times 3$, 8 channels, 0.05M params) and an **Object-Centric Test-Time Training (TTT) Loop** operating over re-ARC augmented grids per test puzzle.

### 6.4 Unverified GPT-3.5 Baselines Falsely Tagged as [검증됨] (AUDIT §2)
- **Severity**: **MAJOR**
- **Technical Reasoning**:  
  In Table 7.1:
  - Row A3: MATH $T = 43.1$ is tagged `[검증됨: https://arxiv.org/abs/2312.11805]`. In `reason_baseline.md`, arXiv 2312.11805 reports **34.1**! The 43.1 figure was an unverified claim from an OpenAI website.
  - Row A11: DROP $T = 70.2$ is tagged `[검증됨: https://arxiv.org/abs/2303.08774]`. In arXiv 2303.08774 (GPT-4 Report), GPT-3.5 DROP F1 is verified at **64.1**! 70.2 is unverified.
  - Row B3: StrategyQA $T = 70.3$ is tagged `[검증됨: Faithful-COT]`. Faithful-COT reports **51.5**! 70.3 is an unverified secondary CoT snippet.
  - Row A5: BBH $T = 70.7$ is tagged `[검증됨: https://arxiv.org/abs/2312.11805]`. arXiv 2312.11805 reports 66.6; 70.7 is text-davinci-003 from CoT-Hub.
- **Concrete Fix**:  
  Correct all baseline targets to verified primary values:
  - MATH: $T = 34.1 \implies G = 34.1 + \max(10, 22.0) = \mathbf{56.1}$ (target 64.8 clears it).
  - DROP F1: $T = 64.1 \implies G = 64.1 + \max(10, 12.0) = \mathbf{76.1}$ (target 81.8 clears it).
  - StrategyQA: $T = 51.5 \implies G = 51.5 + \max(10, 16.2) = \mathbf{67.7}$ (target 82.0 clears it).
  - Label all unverified historical numbers strictly as `[UNVERIFIED: secondary source]`.

---

## 7. Attack 6: Parameter Accounting & Omitted Modules (AUDIT §1.9)

### 7.1 Missing Modules from Inference Budget
- **Severity**: **MAJOR**
- **Technical Reasoning**:  
  AUDIT §1.9 mandates that every parameter executed at inference must be counted toward the <100M budget.  
  Careful inspection of Table 5.3 reveals several missing modules required by the proposal's own claims:
  1. *DROP Span Pointer Head*: In §5.1 and §7.1 (Row A11), the model is claimed to solve DROP by extracting pointer spans from input text. A linear projection over $V_{\text{out}} = 2,048$ formal IR tokens cannot output arbitrary token spans from a 1,000-token reading passage. Emitting pointer spans requires a **Pointer Network Head** (dual linear projections of dimension $d \times d$ plus start/end scoring vectors):
     $$\text{Params}_{\text{pointer}} = 640 \times 640 + 2 \times 640 = 410,880 \text{ parameters}$$
  2. *Formal IR Token Input Embeddings*: The core autoregressively emits formal tokens during MCTS rollout. In §5.3, the proposal counts input token embeddings ($V_{\text{in}} = 16,384 \times 640 = 10,485,760$), but lists the formal IR vocabulary as a distinct set of $V_{\text{out}} = 2,048$ tokens. When an emitted formal token is fed back into the recurrent core at step $t+1$, how is it embedded? If it has an independent embedding matrix, that requires:
     $$\text{Params}_{\text{formal\_emb}} = 2,048 \times 640 = 1,310,720 \text{ parameters}$$
     If it shares the input embedding table, the proposal must state this and ensure token indices are strictly mapped.
  3. *Bidirectional Reader Layer Multiplier*: §5.3 lists "4 Layers Bi-Gated DeltaNet ($d=640, d_{\text{ff}}=1600$): $22,126,080$ params."  
     A true bidirectional recurrent layer consists of two independent recurrent sweeps (forward and backward). If there are 4 bidirectional layers, that is $4 \times 2 = 8$ recurrent blocks ($44,252,160$ parameters), not 22.13M! If the proposal meant 2 forward and 2 backward blocks, it must explicitly specify 2 layers.
  4. *Gated DeltaNet Conv1D and Gate Projections*: The mathematical formulation of Gated DeltaNet includes 1D short convolutions (kernel size $k=4$) on $q, k, v$ projections ($3 \times 4 \times 640 = 7,680$ params per layer) and RMSNorm scale parameters.

### 7.2 Recounted True Parameter Table
Let us recalculate the parameter budget honestly:

```
+----------------------------------------------------------------------------------------------------+
| Module                          | Specification                                  | Honest Parameter Count  |
+----------------------------------------------------------------------------------------------------+
| Input Token Embeddings          | 16,384 vocab x 640 dim                         | 10,485,760              |
| Formal IR Input Embeddings      | 2,048 formal vocab x 640 dim                   | 1,310,720               |
| Bidirectional Input Reader      | 2 Bi-Layers (4 directional blocks, d=640)      | 22,159,360              |
| Recurrent Reasoning Core        | 8 Layers Weight-Tied Gated DeltaNet (r in 1..8)| 44,318,720              |
| Policy Head (MCTS Action)       | Linear d=640 -> 2,048                          | 1,312,768               |
| Value Head (MCTS Critic)        | MLP 640 -> 256 -> 1                            | 164,353                 |
| Proposer Head (Self-Play)       | Linear d=640 -> 2,048                          | 1,312,768               |
| DROP Span Pointer Head          | Pointer Network Projections (Start/End)        | 410,880                 |
| Dense Retriever Key Projection  | Projection d=640 -> 256                        | 164,096                 |
| External Verifiers & Tools      | Python, SymPy, Z3, Lean, BM25 (0 params)       | 0                       |
+----------------------------------------------------------------------------------------------------+
| TOTAL RECOUNTED PARAMETERS      | All inference-executed trainable weights       | 81,639,425              |
+----------------------------------------------------------------------------------------------------+
```

*Verdict*: $81,639,425 < 100,000,000$. The model remains compliant with the <100M parameter limit (18.36M headroom), but the proposal's original table was sloppy and omitted necessary functional modules.

---

## 8. Attack 7: Citation Laundering & Unverified References (AUDIT §2)

Proposal v1 exhibits multiple instances of citation laundering, where real papers are cited for claims they do not make:

1. **Distillation Ceiling (arXiv:2303.08774)**:
   - *Proposal Claim (§1.1)*: Cites the GPT-4 Technical Report (`https://arxiv.org/abs/2303.08774`) for the mathematical theorem that distillation caps student support within teacher support.
   - *Reality*: The GPT-4 report contains zero mathematical theory on distillation.
   - *Fix*: Replace with verified citation: arXiv 2504.13837, and accurately report its finding: distillation can add new reasoning patterns, whereas pure RL narrows the reasoning boundary.
2. **Stitch Symbolic Abstraction (arXiv:2006.08381v1)**:
   - *Proposal Claim (§2.1)*: Cites DreamCoder (`https://arxiv.org/abs/2006.08381v1`) as "Stitch-style symbolic abstraction."
   - *Reality*: DreamCoder uses its own wake-sleep compression. Stitch is an entirely distinct paper (Bowers et al., 2023, arXiv:2310.19791).
   - *Fix*: Cite Bowers et al. (arXiv:2310.19791) directly for Stitch.
3. **Gated DeltaNet Update Formula Error (§5.4)**:
   - *Proposal Claim (§5.4)*:
     $$S_t = S_{t-1} \odot (1 - \beta_t k_t^T) + v_t k_t^T$$
   - *Reality*: This formula is mathematically invalid. It takes the Hadamard product $\odot$ of a matrix with an outer-product transpose.
   - *Fix*: State the exact matrix update equation from arXiv:2412.06464:
     $$S_t = \alpha_t S_{t-1} (I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
4. **Invented Dataset Names (§8.1)**:
   - *Proposal Claim*: Lists "FineWeb-2 Korean Educational Subset" (0.4B) and "Cosmopedia v2 Formal STEM" (1.0B).
   - *Reality*: Neither subset exists under those names in `data.md`.
   - *Fix*: Use exact names from `data.md`: FineWeb-2 Korean split and Cosmopedia v2.
5. **AlignScore Double-Blind Swap Test (§9.3)**:
   - *Proposal Claim*: Cites AlignScore (`https://arxiv.org/abs/2305.16739`) as the source for double-blind swap tests and suffix-automaton masking.
   - *Reality*: AlignScore is an offline RoBERTa factual consistency scorer. Suffix-automaton masking originates from `io_split.md`.
   - *Fix*: Attribute suffix-automaton masking to `io_split.md` / CoG (arXiv:2307.06962), and keep AlignScore strictly as an offline factual auditor.
6. **Broken Model Links (§9.1)**:
   - *Proposal Claim*: Gemma 3 4B linked to `https://huggingface.co/google/gemma-3-1b-it`.
   - *Fix*: Correct link to the verified Gemma 3 4B checkpoint.

---

## 9. Attack 8: Banned Code & The Experiment 1 Facade (AUDIT §0 & §2)

### 9.1 Direct Violation of the Universal Code Ban
- **Severity**: **FATAL**
- **Technical Reasoning**:  
  AUDIT §0 Rule 2 explicitly mandates:
  > "**No code in any proposal, blueprint, or section.** This replaces the BRIEF.md instruction to include pseudo-code for Experiment 1. Experiments must be specified as a **plan only**: inputs, procedure, what is measured and how, the baseline/control arm, the pass criterion, the kill criterion, and the expected compute and wall-clock time."
  Section 12.1 of Proposal v1 contains **80 lines of executable Python code** (lines 404–484). This is a direct violation of binding audit instructions.

### 9.2 The "Hardcoded Success" Facade
- **Severity**: **FATAL**
- **Technical Reasoning**:  
  Beyond violating the formatting ban, the code provided in §12.1 is an engineering facade that proves none of the proposal's architectural claims:
  - In lines 432–436, `self.learned_templates` is initialized with two hardcoded lambda functions.
  - In line 465, `distill()` merely appends a newly discovered Python lambda expression directly to a list:
    `self.learned_templates.append(...)`
  - There is NO neural network, NO Gated DeltaNet state update, NO policy-gradient calculation, NO value head evaluation, and NO continuous representation. It is a toy symbolic template-accumulator that hardcodes its own success. It completely circumvents the real scientific challenge: *can a 79.8M neural policy distill discrete MCTS tree rollouts into neural weights?*

- **Concrete Fix**:  
  Completely purge the Python code from §12.1. Replace it with a rigorous **Plain-Language Experimental Plan**:
  1. *Inputs*: 500 procedurally generated Game of 24 tuples with known combinatorial solution depth.
  2. *Procedure (Numbered Steps)*:
     - Step 1: Initialize 79.8M Gated DeltaNet weights $\theta_0, \phi_0$ from Stage 0 arithmetic pretraining.
     - Step 2: Measure baseline greedy pass@1 accuracy on held-out 100 tuples ($g_0$).
     - Step 3: Run MCTS (64 simulations, PUCT exploration $c = 1.4$, Dirichlet noise $\alpha = 0.3$) across 400 training tuples using the Python arithmetic verifier sandbox. Record visit distributions $\pi$ and outcomes $z$.
     - Step 4: Execute Expert Iteration update on 2× T4: train policy head with cross-entropy to $\pi$ and value head with MSE to $z$ for 3 epochs (AdamW, lr = $2 \times 10^{-4}$). Produce generation $g_1$.
     - Step 5: Evaluate $g_1$ greedy pass@1 on the 100 held-out tuples.
  3. *Control Arm*: Parallel identical training run where outcome $z$ is replaced with random noise $z_{\text{rand}} \sim \text{Bernoulli}(0.5)$.
  4. *Pass Criterion*: $g_1$ pass@1 exceeds $g_0$ by $\ge 10.0$ percentage points, and beats the random-reward control arm by $\ge 8.0$ percentage points.
  5. *Kill Criterion*: $g_1$ score fails to exceed $g_0$ ($\Delta_{g1} \le 0$), or random-reward arm matches true reward arm within 3 points (spurious reward collapse).
  6. *Compute & Time*: Rollouts: 0.15 GPU-hours on 2× T4. Training: 0.05 GPU-hours. Total wall-clock: 25 minutes.

---

## 10. Attack 9: Fair Play, Secret Transformer Check, Transferability, Safety & 12-Week Feasibility

### 10.1 Secret Transformer Check
- **Investigation**: Does the Gated DeltaNet architecture sneak in softmax attention?
- **Finding**: Pure Gated DeltaNet (arXiv:2412.06464) uses an associative scan over state matrices:
  $$S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T, \quad y_t = S_t q_t$$
  This is strictly attention-free ($\mathcal{O}(1)$ memory per token, no $N \times N$ token attention matrix).
- **Caveat**: The official NVlabs repository includes hybrid configurations that interleave Gated DeltaNet with Sliding Window Attention (SWA). **Any import of SWA blocks violates the non-Transformer mandate.**
- **Enforceable Fix**: Formally certify that the PyTorch/JAX implementation pins the configuration to `attn_mode="none"` and `hybrid_swa_layers=0`. Verify zero calls to `scaled_dot_product_attention`.

### 10.2 Transferability to 1B and 10B Scales
- **Evaluation**: The proposal claims non-parametric capability transfer (L3 program libraries carry over without retraining).
- **Critique**: While the L3 macro library transfers losslessly, the proposal ignores that **MCTS search speed scales inversely with model size**.
  - A 0.1B core does 64 simulations in 2 seconds.
  - A 1B core on the same hardware would take 15–20 seconds per query.
  - A 10B core cannot run MCTS rollouts on free compute at all.
- **Fix**: Clarify that scaling to 1B/10B relies on **Depth-to-Width Distillation**: using the 0.1B core + MCTS to generate dense verified pretraining data that trains the 1B/10B cores to solve problems in a shallow greedy pass ($r=1$), minimizing test-time search cost at larger scale.

### 10.3 Safety Mechanisms: Enforceable vs. Decorative
- **Finding**: The Day 1 filesystem read-only permissions and network isolation (`--network none`) are genuine and enforceable.
- **Weakness**: In §3, the "canary token trigger" is described as detecting any gradient or file access attempt. Canary tokens detect dataset memorization in pretraining; they cannot detect memory-resident reward hacking in real time.
- **Fix**: Replace vague canary file monitors with **Process-Level Sandbox Seccomp BPF filters** that physically trap and terminate any socket syscall or directory traversal attempt outside `/tmp/eval_sandbox`.

### 10.4 12-Week Feasibility for a Solo Builder (0 KRW)
- **Severity**: **MAJOR**
- **Technical Reasoning**:  
  Proposal v1 proposes building:
  1. A custom Gated DeltaNet implementation with custom Triton 3.2 kernels on T4 AND JAX scans on TPU.
  2. A complete MCTS tree-search engine with batched GPU-CPU rollouts.
  3. Four distinct formal verification sandboxes (Python, SymPy, Z3, Lean 4).
  4. An Absolute-Zero self-play proposer head with learnability regret rewards.
  5. A Stitch DSL compression engine.
  6. A 5B-token Stage 0 pretraining pipeline and evaluation across 20 benchmark rows.
  For a solo developer with 0 KRW in 12 weeks, this is a recipe for complete structural collapse.
- **Concrete Fix (The Phased S-Curve MVP Roadmap)**:
  - **Weeks 1–4 (MVP S-Curve 1: Arithmetic & Symbolic Logic)**:
    - Stick to pure-PyTorch parallel chunked scans (avoiding custom Triton kernel debt).
    - Single verifier sandbox: Python 3.12 + SymPy.
    - Validate MCTS Expert Iteration on Game of 24 and GSM-Symbolic.
    - Measure $\Delta_{g1} > 0$ and achieve cord-cut on math.
  - **Weeks 5–8 (S-Curve 2: Code Execution & Algorithm Synthesis)**:
    - Add Unit Test execution for HumanEval+ / MBPP+.
    - Integrate Depth Regret proposer.
    - Port scan kernels to JAX for TPU v5e-8 throughput scaling.
  - **Weeks 9–12 (S-Curve 3: Formal Proving & SMT Solvers)**:
    - Ingest Z3 and Lean 4 sandboxes for Tier A logic rows.
    - Execute official benchmark audit and publish commit-revealed results.

---

## 11. Concrete Revision Mandates for Phase C

The Lead Designer for Search must implement the following mandatory revisions in `work/proposal_search_v2.md`:

1. **Recompute All Compute and Quotas with Real Recursion**: Use $r=4$ (median) and $r=8$ (max) across all FLOP formulas ($2 \times [N_{\text{non\_core}} + r \times N_{\text{core}}]$ for inference, $6\times$ for training).
2. **Correct Inference Latency**: Replace the 0.0007s fantasy with physical memory-bandwidth-bound timings (0.235s per greedy pass; 1.65–3.0s per 64-sim MCTS pass). Downscale task rollouts to 5,000 tasks/generation with Snell adaptive search to ensure 10–12 generations/week within Kaggle quotas.
3. **Plug the Reading Verifier Hole**: Integrate Reverse Program-First Generation, Paraphrase Consistency Checks, and Perturbation Twin Gating inside the self-play loop.
4. **Purge All Code from Experiment 1**: Delete the 80 lines of Python. Replace with a numbered, step-by-step scientific plan including a random-reward control arm.
5. **Fix Parameter Omissions**: Add the DROP pointer head ($0.41\text{M}$) and formal IR embeddings ($1.31\text{M}$); clarify bidirectional reader layers to confirm the total sits transparently at $\approx 81.6\text{M} < 100\text{M}$.
6. **Correct Scoreboard & Baseline Tags**: Split PrOntoQA and MuSR subtasks into dedicated rows; correct MATH, DROP, StrategyQA, and BBH baselines to verified values; label all projections as **[추정]** or **[가설]**; downgrade ARC-AGI-1 to 32% or provide an explicit 2D spatial adapter.
7. **Eliminate Citation Laundering**: Clean up citations for distillation theory, Stitch, AlignScore, Gated DeltaNet math, and dataset names.
8. **Restore Required Items**: Add the mandatory No-Teacher ablation protocol, use official frontier model names (Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra), and bound the cord-cut teacher to $\le 14\text{B}$.
