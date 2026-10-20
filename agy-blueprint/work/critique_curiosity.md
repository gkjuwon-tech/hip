# RED TEAM CRITIQUE: ORRERY-Curiosity Proposal (v1)

**Target Proposal**: `/home/user/hip/agy-blueprint/work/proposal_curiosity_v1.md` (`ORRERY-Curiosity`)  
**Lead Red Team Critic**: Lead Red Team Critic (ML/RL Systems Researcher, Kaggle Grandmaster, Benchmark Integrity Auditor)  
**Date**: October 2026  
**Status**: BINDING AUDIT & CRITIQUE (Phase B per BRIEF.md §4 and AUDIT_FROM_CLAUDE.md)  
**Output Target**: `/home/user/hip/agy-blueprint/work/critique_curiosity.md`  

---

## 0. Executive Summary & Forensic Verdict

The `ORRERY-Curiosity` proposal sets forth an intellectually ambitious blueprint: replacing conversational imitation with a tiny (<100M parameter), attention-free aphasic core (Bidirectional Gated DeltaNet) embedded within a latent JEPA (Joint Embedding Predictive Architecture) world model driven by epistemic Expected Information Gain (EIG). The goal—compounding reasoning capability via self-supervised world modeling and zero-parameter mechanical verifiers—is fundamentally aligned with the project north star.

However, beneath its polished facade, the proposal suffers from **critical mathematical undercounts, physically impossible latency claims, severe citation laundering, ungrounded benchmark projections, and an egregious benchmark integrity violation in Experiment 1**.

```
+---------------------------------------------------------------------------------------------------------+
|                                    RED TEAM ATTACK SEVERITY MATRIX                                      |
+---------------------------------------------------------------------------------------------------------+
| Target Domain                 | Core Vulnerability                                         | Severity   |
+-------------------------------+------------------------------------------------------------+------------+
| 1. The RSI Engine             | Reading Verifier Hole + Noisy TV Trap in Latent EIG        | FATAL      |
| 2. Recurrent FLOP Arithmetic  | Missing Recurrent Depth r=16 (4.26x FLOP Undercount)       | FATAL      |
| 3. Hardware Latency Realism   | Memory-Bandwidth Wall on T4 (Claimed 0.28ms vs Real 162ms)| FATAL      |
| 4. Integrity Violation        | Hardcoded Code & Faked Success in Experiment 1             | FATAL      |
| 5. Secret Transformer Check   | Potion-base-8M Retriever Contains Softmax Attention        | FATAL      |
| 6. Missing Mandatory Items    | Missing No-Teacher Ablation, Outdated Tin Cans, Rows Cut   | MAJOR      |
| 7. Scoreboard Integrity       | Reverse-Engineered Margins, Unverified Projections         | MAJOR      |
| 8. ARC-AGI-1 Scaling Gap      | Attention-Free Core Target Exceeds Known Physics (45.2%)   | MAJOR      |
| 9. Parameter Table Omissions  | Omitted Projections & IR Embeddings (+5.51M Params)        | MAJOR      |
| 10. Citation Laundering       | 10 Documented Instances of Inverted or Invented Claims     | MAJOR      |
| 11. Scaling & Feasibility     | Incompatible Latent Transfer & Overloaded 12-Week Scope    | MINOR      |
+---------------------------------------------------------------------------------------------------------+
```

Every single attack below details the **verbatim defect**, the **underlying theoretical/physical reasoning**, and a **concrete, non-diluting fix** that preserves maximum technical ambition.

---

## 1. Attack Priority 1: The RSI Engine & World Model Vulnerabilities

### 1.1 The Reading Verifier Black Hole (AUDIT §1.4)
- **Severity**: **FATAL**
- **Observed Defect**:
  In Section 2.1 (lines 62–73), the proposal outlines a 6-stage lifecycle: the proposer samples tasks $x \sim \mathcal{P}_\theta$, the core decodes formal scripts (Python/SymPy), and the sandbox executes them, returning binary feedback $v \in \{0, 1\}$.
- **Detailed Technical Reasoning**:
  Executors verify strictly that a formal program *runs without exceptions* and *evaluates to a specified equality*. They do **not** verify that the program *correctly formalizes the input problem*. 
  Inside an autonomous RSI loop operating on natural-language inputs, an aphasic core paired with an autonomous proposer will rapidly discover a degenerate shortcut: proposing trivialized or misread problem formalizations that produce executable Python code outputting arbitrary constants (e.g., `print(42)`). The sandbox returns $v=1$, reward is assigned, latent prediction error drops, and the dual objective reinforces the degenerate policy. The core becomes completely illiterate on held-out natural language reasoning benchmarks (GSM8K, MuSR, KMMLU) while achieving 100% in-loop verification. The aphasic genius's critical failure point is not speaking; it is **reading**.
- **Concrete Fix**:
  Implement **Reverse Procedural Generation with Bi-directional Invariant Verification** per AUDIT §1.4:
  1. *Reverse Generation*: Procedural engines (Reasoning Gym, PlanBench, SymPy generators) construct the analytical ground-truth abstract syntax tree (AST) and target value $y^*$ *first*, and subsequently render it into natural language problem text $T$ via deterministic grammar templates. Every synthetic problem thus carries an unhackable formal ground-truth spec.
  2. *Counterfactual Perturbation Invariants*: For every generated natural language problem $T$, construct a perturbed twin $T'$ by altering entity bindings or numerical constants with a mathematically predictable outcome shift $y^{*\prime}$ (GSM-Symbolic / GSM-Plus style). The core is rewarded only if it correctly solves *both* $(T, y^*)$ and $(T', y^{*\prime})$.
  3. *Paraphrase Consistency Check*: The core must formalize two distinct linguistic surface renderings of the same underlying semantic problem into logically equivalent ASTs (verified via Z3/SymPy solver equivalence).

---

### 1.2 The Noisy TV Trap in Latent EIG Epistemic Search (AUDIT §3.12)
- **Severity**: **FATAL**
- **Observed Defect**:
  Section 2.1 line 64 defines Expected Information Gain (EIG) as ensemble variance across $M=3$ predictor heads:
  $$\mathcal{I}(x) = \mathbb{E}_{a \sim \pi} \left[ \frac{1}{M} \sum_{m=1}^M \left\| \hat{z}^{(m)}_{t+1} - \bar{z}_{t+1} \right\|_2^2 \right]$$
  down-weighted by asymmetric regret $\mathcal{R}(x) = \text{clip}(\text{Acc}_{\text{target}} - \text{Acc}_{\text{current}}, 0, 1) \cdot \mathcal{I}(x)$.
- **Detailed Technical Reasoning**:
  This formulation falls directly into the classic reinforcement learning **Noisy TV Problem**. In any domain featuring stochastic distractor text, unconstrained mathematical representations, or unlearnable procedural noise (e.g., distractor clauses in GSM-Symbolic, variable name randomization, or chaotic numerical sequences), the three ensemble predictor heads will disagree indefinitely. Consequently, $\mathcal{I}(x)$ remains perpetually maximized on completely unlearnable noise.
  The proposal's asymmetric regret mechanism fails to prevent this trap: for an unlearnable noisy problem, $\text{Acc}_{\text{current}} \approx 0.0$. If $\text{Acc}_{\text{target}} = 0.75$, the clipping term evaluates to $\text{clip}(0.75 - 0.0, 0, 1) = 0.75 > 0$. The regret filter actively scales up the noise variance ($\mathcal{R}(x) = 0.75 \cdot \mathcal{I}(x)$), trapping the RSI engine in a self-reinforcing curiosity loop where it squanders 100% of its compute budget attempting to model white noise.
- **Concrete Fix**:
  Replace raw ensemble variance with **Verifier-Grounded Empirical Learning Progress (ELP)**:
  $$\text{ELP}_g(x) = \left| \mathcal{E}_{g}(x) - \mathcal{E}_{g-1}(x) \right| \cdot \mathbb{I}(v(x) \text{ is deterministically verifiable})$$
  where $\mathcal{E}_g(x) = \|z_{t+1} - \hat{z}_{t+1}\|_2^2$ is the latent prediction error at generation $g$. 
  - For unlearnable noise, prediction error remains constant across training iterations: $\Delta \mathcal{E} \to 0$, driving $\text{ELP} \to 0$.
  - For already mastered trivial problems, initial error is near zero: $\Delta \mathcal{E} \approx 0$, driving $\text{ELP} \to 0$.
  - Curiosity focuses exclusively on the derivative of error reduction—the true frontier of learnability.

---

### 1.3 Own-Level Prison & The Hand-Waved Proposer Architecture
- **Severity**: **MAJOR**
- **Observed Defect**:
  In Section 3 (Table row 2), the "Own-Level Prison" defense specifies: *"Task solve rate outside [0.30, 0.70] for >40% of proposed tasks. Reset proposer to gen g-1; enforce asymmetric PAIRED regret reward."* In Section 5.3, the proposer is listed as a tiny 393k-parameter MLP head.
- **Detailed Technical Reasoning**:
  A 393,216-parameter MLP head cannot generate valid, syntactically complex formal problem statements or code out of thin air. Real PAIRED (Protagonist Antagonist Induced Regret Environment Design) requires an environment generator operating in a structured, constrained parameter space with two active competing agents (protagonist and antagonist). Simply emitting a "seed" vector from a 393k MLP into an unspecified grammar will produce uncompilable syntax or degenerate repeats. Furthermore, rolling back the proposer to $g-1$ upon detecting out-of-bounds solve rates creates an infinite rollback loop if the core has already updated its policy weights.
- **Concrete Fix**:
  De-neuralize problem generation and replace the learned proposer head with a **Parametric Procedural Difficulty Controller**:
  1. The proposer controls the continuous hyperparameters of deterministic procedural generators (e.g., number of operations in Countdown, recursion depth in SymPy equations, grid size and color count in ARC-AGI, or number of distractor objects in Blocksworld).
  2. The controller uses Bayesian Optimization over the parameter lattice $\vec{\lambda} \in \mathbb{R}^k$ to target the empirical edge of competence ($\text{pass@1} \in [0.35, 0.65]$).
  3. All generated tasks are syntactically valid by construction, eliminating syntax failures while maintaining absolute zero-parameter verifiability.

---

### 1.4 Hardcoded S-Curve Walk vs Autonomous Verifier Discovery (AUDIT §1.3)
- **Severity**: **MAJOR**
- **Observed Defect**:
  Section 2.4 (lines 88–92) defines domain progression as a fixed, human-authored array:
  $$\text{Arithmetic (SymPy)} \to \text{Symbolic Algebra} \to \text{Formal Proofs (Lean 4)} \to \text{Algorithmic Code (PyTest)} \to \text{Simulators}$$
  triggered when $\text{EDLI}_k(g) < 0.05 \text{ pts/h}$.
- **Detailed Technical Reasoning**:
  BRIEF.md §1 Decision 3 explicitly requires that finding or building the next verifier domain is *itself an RSI skill*. Walking a pre-compiled human list is not RSI discovery; it is static scheduled multi-task learning. Simultaneously, BRIEF.md §2.7 strictly forbids the model from writing or modifying its own verifiers (to prevent catastrophic reward corruption).
- **Concrete Fix**:
  Design an **Autonomous Domain Composition Engine** over an immutable registry of primitive verifiers:
  1. The engine maintains a fixed, sandboxed registry of atomic verification primitives: (a) Algebraic Invariance (SymPy), (b) First-Order Logical Consistency (Z3), (c) State Transition Reachability (VAL / Planning), (d) Input-Output Assertion Fuzzing (PyTest / Hypothesis).
  2. When active domain progress saturates ($\text{EDLI} < 0.05$), the system synthesizes *compound verifier environments* by taking the Cartesian product of verified domains (e.g., embedding algebraic constraints into Python test harnesses, or expressing Blocksworld planning invariants inside Z3).
  3. Every compound verifier is assembled strictly through declarative schema composition (never arbitrary synthesized Python execution), preserving immutability while expanding verification coverage autonomously.

---

## 2. Attack Priority 2: Recurrent Depth FLOP Arithmetic Undercount (AUDIT §1.1)

### 2.1 The Missing Recurrent Factor $r=16$ (4.26× Compute Undercount)
- **Severity**: **FATAL**
- **Observed Defect**:
  Section 2.5 (lines 94–103) computes rollout and training compute using $N = 68.00 \times 10^6$ parameters as a flat scalar:
  $$\text{FLOPs}_{\text{rollout}} = 2 \cdot (6.80 \times 10^7) \cdot (6.40 \times 10^6) = 8.70 \times 10^{14} \text{ FLOPs}$$
  $$\text{FLOPs}_{\text{train}} = 6 \cdot (6.80 \times 10^7) \cdot (2.50 \times 10^7) = 1.020 \times 10^{16} \text{ FLOPs}$$
- **Detailed Technical Reasoning**:
  Per AUDIT §1.1, the core architecture (Section 5.3) is a weight-tied recurrent model iterated $r$ times ($r \le 16$, default $r=16$) per token. The non-core parameters execute once, while the core parameters execute $r$ times.
  From Section 5.3:
  - Non-core parameters $N_{\text{non\_core}} = 53.23 \times 10^6$ (Embeddings 8.39M + Reader 31.64M + JEPA 2.49M + Curiosity 1.57M + Value 0.13M + Emitter 1.05M + Proposer 0.39M + Retriever 7.56M).
  - Recurrent core parameters $N_{\text{core}} = 14.77 \times 10^6$.
  
  The true effective computational parameter count during depth-$r$ unrolling is:
  $$N_{\text{eff}}(r) = N_{\text{non\_core}} + r \times N_{\text{core}}$$
  For $r=16$:
  $$N_{\text{eff}}(16) = 53.23\text{M} + (16 \times 14.774\text{M}) = 53.23\text{M} + 236.38\text{M} = 289.61 \times 10^6 \text{ parameters}$$
  The proposal treats $N_{\text{eff}}$ as $68.00\text{M}$, undercounting computational depth by a factor of:
  $$\frac{289.61\text{M}}{68.00\text{M}} = 4.259\times \quad (\mathbf{4.26\times \text{ undercount}})$$

---

### 2.2 Recomputed Generation Cycle FLOPs and Quota Feasibility
Let us recalculate the true compute costs and wall-clock times under Kaggle's free quotas:

#### A. Rollout FLOP Recalculation (2× NVIDIA T4)
For 10,000 tasks $\times$ 8 rollout trajectories = 80,000 sequences.
Each sequence consists of prompt encoding ($L_{\text{in}} = 128$ tokens) and formal IR decoding ($L_{\text{out}} = 80$ tokens).
1. Prompt Encoding: Reader runs once over 128 tokens:
   $$\text{FLOPs}_{\text{prompt}} = 2 \times (N_{\text{emb}} + N_{\text{reader}}) \times 128 = 2 \times (40.03 \times 10^6) \times 128 = 1.025 \times 10^{10} \text{ FLOPs/traj}$$
2. Sequential Decoding: Core recurses $r=16$ times per token for 80 tokens:
   $$\text{FLOPs}_{\text{decode}} = 2 \times (16 \times N_{\text{core}} + N_{\text{emitter}} + N_{\text{JEPA}} + N_{\text{val}}) \times 80 = 2 \times (240.05 \times 10^6) \times 80 = 3.841 \times 10^{10} \text{ FLOPs/traj}$$
3. Total FLOPs per trajectory = $1.025 \times 10^{10} + 3.841 \times 10^{10} = 4.866 \times 10^{10} \text{ FLOPs}$.
4. Total Rollout FLOPs for 80,000 trajectories:
   $$\text{FLOPs}_{\text{rollout\_true}} = 80,000 \times 4.866 \times 10^{10} = \mathbf{3.893 \times 10^{15} \text{ FLOPs}}$$
   *Proposal claimed $8.70 \times 10^{14}$ FLOPs—a **4.47× undercount**.*

#### B. Training Step FLOP Recalculation (TPU v5e-8 or 2× T4)
For $D = 2.50 \times 10^7$ training tokens unrolled with $r=16$ Backpropagation Through Time (BPTT):
$$\text{FLOPs}_{\text{train\_true}} = 6 \cdot N_{\text{eff}}(16) \cdot D = 6 \times (2.8961 \times 10^8) \times (2.50 \times 10^7) = \mathbf{4.344 \times 10^{16} \text{ FLOPs}}$$
*Proposal claimed $1.020 \times 10^{16}$ FLOPs—a **4.26× undercount**.*

#### C. Honest Wall-Clock Time per Generation

```
+---------------------------------------------------------------------------------------------------------+
|                                HONEST RSI GENERATION WALL-CLOCK BUDGET                                  |
+---------------------------------------------------------------------------------------------------------+
| Operational Phase      | Proposal Claim      | Recomputed True FLOPs | Real Wall-Clock (2x T4 / TPU)    |
+------------------------+---------------------+-----------------------+----------------------------------+
| Rollout GPU Model Exec | 22.3 seconds        | 3.893e15 FLOPs        | ~99.8 s (at 30% MFU on 2x T4)    |
| Sandbox Script Exec    | 5.3 minutes         | 80,000 sandbox calls  | 8.33 min (at 25ms / 4 vCPUs)     |
| Training (TPU v5e-8)   | 25.9 seconds        | 4.344e16 FLOPs        | 1.84 min (110.3 s at 25% MFU)    |
| Training (2x T4 Alt)   | 4.4 minutes         | 4.344e16 FLOPs        | 22.28 min (1,336 s at 25% MFU)   |
| Evaluation Suite       | 2.5 minutes         | 6.4e14 FLOPs          | 4.2 minutes                      |
+------------------------+---------------------+-----------------------+----------------------------------+
| TOTAL GENERATION TIME  | 12.2 min (claimed)  | 4.797e16 FLOPs        | 15.2 min (TPU) / 35.7 min (T4)   |
+------------------------+---------------------+-----------------------+----------------------------------+
```

- **Impact Assessment**:
  While the generation cycle remains feasible within Kaggle quotas, the proposal's claim of "over 35 full generations per week on T4 alone" collapses. Under pure 2× T4 execution (35.7 min/gen), a 30-hour weekly quota accommodates at most **50 generations per week if 100% dedicated to training**, or roughly **15–18 generations per week** when accounting for checkpointing, sandbox latency, and eval passes. The numbers must be revised honestly.

---

## 3. Attack Priority 3: Latency Realism & Memory-Bandwidth Limits (AUDIT §1.2)

### 3.1 The 0.28ms Latency Fantasy on Turing T4
- **Severity**: **FATAL**
- **Observed Defect**:
  Section 8.4 (lines 298–299) claims:
  - Single Forward Pass (Greedy): *"FLOPs $= 1.09 \times 10^{10}$ FLOPs. T4 Latency: $\approx \mathbf{0.28\text{ ms}}$ per trajectory."*
  - Full MCTS Search ($k=8$ samples): *"T4 Latency: $\approx \mathbf{2.2\text{ ms}}$ model execution $+ \approx 15\text{ ms}$ Python sandbox."*
- **Detailed Technical Reasoning**:
  This calculation commits the fatal error identified in AUDIT §1.2: **treating sequential autoregressive token decoding as compute-bound at peak Tensor Core throughput**.
  Sequential token generation with batch size $B=1$ is strictly **memory-bandwidth bound**. At each generated token, the GPU cannot reuse loaded weights across a batch; it must stream the model weights from global VRAM (GDDR6) into on-chip cache and registers.
  
  Physical Specifications of NVIDIA T4:
  - Peak Memory Bandwidth: $300\text{ GB/s}$ theoretical; realistic achievable bandwidth in PyTorch CUDA kernels is $\approx 240\text{ GB/s}$ ($2.40 \times 10^{11}\text{ bytes/s}$).
  - Weight Footprint of Recurrent Core: $14.774 \times 10^6 \text{ params} \times 2 \text{ bytes (fp16)} = 29.55 \times 10^6 \text{ bytes}$ ($29.55\text{ MB}$).
  
  To execute $r=16$ recursions for **a single token**, the T4 must read the core weights 16 times from VRAM:
  $$\text{Bytes Read per Token} = 16 \times 29.55\text{ MB} = 472.8\text{ MB} = 4.728 \times 10^8 \text{ bytes}$$
  $$\text{Memory-Read Time per Token} = \frac{4.728 \times 10^8 \text{ bytes}}{2.40 \times 10^{11} \text{ bytes/s}} \approx 1.97 \times 10^{-3} \text{ seconds} \approx \mathbf{1.97\text{ ms / token}}$$
  Adding projection heads, RMSNorms, and kernel launch overheads ($\approx 15\ \mu\text{s}$ per layer), the physical minimum time per token is $\approx 2.03\text{ ms}$.
  
  For an 80-token formal trajectory:
  $$\text{Physical Minimum Latency} = 80 \text{ tokens} \times 2.03\text{ ms/token} = \mathbf{162.4\text{ ms per trajectory}}$$
  The proposal claims **0.28 ms**.
  $$\text{Fantasy Ratio} = \frac{162.4\text{ ms}}{0.28\text{ ms}} = \mathbf{580\times \text{ faster than physical laws permit!}}$$
  A GPU would need a memory bandwidth of $174,000\text{ GB/s}$ (174 TB/s)—nearly 50 times faster than an NVIDIA H100 SXM5 (3.35 TB/s)—to achieve the proposal's claimed latency on a T4!

---

### 3.2 Real MCTS Search and Sandbox Throughput
- **MCTS Latency ($k=8$ samples)**:
  Even when batching all 8 samples ($B=8$) in parallel to amortize memory reads, arithmetic intensity remains low ($\approx 16\text{ FLOPs/byte}$, well below T4's ridge point of $216\text{ FLOPs/byte}$). Memory bandwidth still limits latency to $\approx 2.03\text{ ms}$ per step, yielding $80 \times 2.03\text{ ms} \approx 162.4\text{ ms}$ for GPU execution.
- **Python Sandbox Overhead**:
  Line 97 asserts 80,000 sandbox executions complete in 5.3 minutes on 4 vCPUs ($251.5\text{ executions/sec}$, or $15.9\text{ ms}$ per execution).
  In reality:
  - Cold-starting Python with `import sympy` takes $350\text{--}800\text{ ms}$ on a Xeon vCPU.
  - Even with a pre-warmed persistent daemon pool communicating over Unix domain sockets, parsing and executing an algebraic `solve()` takes $20\text{--}40\text{ ms}$.
  - At $25\text{ ms}$ per execution across 4 vCPUs, 80,000 calls require $(80,000 \times 0.025) / 4 = 500\text{ seconds} = \mathbf{8.33\text{ minutes}}$.
  - Total MCTS query latency is $\approx 162.4\text{ ms (GPU)} + 35\text{ ms (Sandbox)} \approx \mathbf{197.4\text{ ms}}$, NOT $\le 20\text{ ms}$.

- **Concrete Fix**:
  1. Replace all latency claims in Sections 2.5 and 8.4 with memory-bandwidth-bound formulas:
     $$T_{\text{seq}} \approx L_{\text{out}} \times \max\left( \frac{r \cdot M_{\text{core}}}{\text{BW}_{\text{mem}}}, \frac{2 \cdot r \cdot N_{\text{core}} \cdot B}{\text{FLOPS}_{\text{peak}}} \right)$$
  2. Implement **Dynamic Recurrent Early-Exit (Adaptive $r$)**: Rather than fixing $r=16$ for all tokens, use a halting unit (PonderNet style) to terminate recursion when state update $\|\Delta z_t\| < \epsilon$, dropping average $r$ on easy syntax tokens to $r \approx 3$, cutting sequential memory traffic by $4\times$.

---

## 4. Attack Priority 4: Missing Required Items (AUDIT §1.5)

### 4.1 Missing Mandatory "No-Teacher" Ablation (BRIEF.md §2.6)
- **Severity**: **MAJOR**
- **Observed Defect**:
  `proposal_curiosity_v1.md` contains zero mentions of a "no-teacher" ablation anywhere in its text, tables, or experimental specifications.
- **Detailed Technical Reasoning**:
  BRIEF.md §2.6 establishes a mandatory hard constraint: *"Teacher data at training time may only come from open-licensed models, must be logged, and a 'no-teacher' ablation must be reported."* Omitting this ablation makes it impossible to prove whether post-cord-cut self-improvement is genuine or merely residual memorization of teacher distillation data.
- **Concrete Fix**:
  Add an explicit **No-Teacher Control Arm** to Section 8 and Section 12:
  - *Arm 0 (Pure Verifier Grounding)*: Core trained from random initialization strictly on procedural generators (Reasoning Gym, Countdown, SymPy synthetic solvers) with zero teacher-generated tokens.
  - *Metrics Tracked*: Generations to pass GPT-3.5 on Tier A procedural twins; asymptotic performance comparison with Stage 0 bootloaded core.

---

### 4.2 Outdated Frontier "Tin Cans" Names (AUDIT §1.5)
- **Severity**: **MINOR**
- **Observed Defect**:
  Section 1.1 line 16 names: *"Claude 3.5 Sonnet, Claude Opus 4, GPT-4o, GPT-6"*.
- **Detailed Technical Reasoning**:
  AUDIT §1.5 explicitly mandated: *"Frontier 'tin cans' are outdated (4/5)... Use the brief's names: Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra."*
- **Concrete Fix**:
  Replace all occurrences in Section 1.1 and Section 7.2 with the canonical names: **Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra**.

---

### 4.3 Dropped Gated Rows & MuSR Averaging Violation
- **Severity**: **MAJOR**
- **Observed Defect**:
  1. Row A12 (Section 7.1) lists: `A12. Logic-LM (FOLIO/ProofWriter) | 62.7 / 71.5 | 75.2 / 81.5`. **PrOntoQA ($T=85.0 \to G=95.0$) was completely dropped!**
  2. Row A13 lists `LogiQA 2.0 | 52.4 | 68.2`. **ReClor ($T=57.4 \to G=71.6$) was completely dropped!**
  3. Row A14 lists `MuSR (Avg MM/OP/TA) | 69.0 | 79.3`. **MuSR was averaged into a single row!**
- **Detailed Technical Reasoning**:
  - `reason_baseline.md` row A12 explicitly pairs FOLIO, ProofWriter, and PrOntoQA ($T=85.0, G=95.0$). Dropping PrOntoQA conceals the hardest logical autoformalization gate on the scoreboard.
  - Row A13 requires both LogiQA 2.0 and ReClor.
  - AUDIT §1.5 explicitly states: *"MuSR must be gated per subtask, not on the average."* Averaging Murder Mysteries (MM), Object Placement (OP), and Team Allocation (TA) masks individual subtask failures. Per `reason_baseline.md`:
    - MuSR Murder Mysteries (MM): $T = 70.0 \implies G = \mathbf{80.0}$
    - MuSR Object Placement (OP): $T = 58.2 \implies G = \mathbf{72.1}$
    - MuSR Team Allocation (TA): $T = 78.8 \implies G = \mathbf{88.8}$
- **Concrete Fix**:
  Restore all omitted rows and unpack MuSR into three independent gated rows in Table 7.1.

---

### 4.4 Cord-Cut Teacher Realism on Free Quota
- **Severity**: **MAJOR**
- **Observed Defect**:
  Section 2.3 defines cord-cut based on $\mathcal{D}_{\text{teacher\_fail}}$ across 8 teacher samples, but never defines which teacher generates these traces or how it runs on free Kaggle hardware.
- **Detailed Technical Reasoning**:
  A 70B frontier teacher cannot run on Kaggle's 2× T4 (32 GB VRAM cannot fit 70B even at 4-bit quantization with KV cache). If the teacher cannot run on free compute, the cord-cut condition cannot be evaluated locally.
- **Concrete Fix**:
  Specify the cord-cut teacher as **Qwen2.5-14B-Instruct** (runs in fp16 across 2× T4 using pipeline parallelism) or evaluate against a **frozen, pre-computed failure bank of 2,500 hard problems** failed by Claude Opus 5.5 and Llama-3.1-70B.

---

## 5. Attack Priority 5: Scoreboard Integrity, ARC-AGI & Probabilities (AUDIT §1.6–§1.8)

### 5.1 Reverse-Engineered Targets & Missing Projection Tags (AUDIT §1.7)
- **Severity**: **MAJOR**
- **Observed Defect**:
  Table 7.1 presents Target Scores that sit artificially $1.1\text{--}2.6$ points above the required $G$ win margin (e.g., GSM8K $G=88.1 \to 89.2$; HumanEval+ $G=80.7 \to 82.3$; MBPP+ $G=79.8 \to 81.5$; DROP $G=80.2 \to 81.5$). None are labeled as projections.
- **Detailed Technical Reasoning**:
  Per AUDIT §1.7: *"Expected scores, '95% CI lower bound' columns, and 'passed at generation N' cells must be labeled [추정] or [가설]. They are projections, not measurements. Several sit exactly 1–3 points above the G threshold, which looks reverse-engineered."* Presenting speculative targets as definitive results destroys benchmark credibility.
- **Concrete Fix**:
  Add `[추정]` to every Target Score and Expected Generation cell in Table 7.1, and provide explicit analytical derivation for each target based on verifier pass@k scaling models.

---

### 5.2 Unverified Secondary Scores Presented as Verified (AUDIT §2)
- **Severity**: **MAJOR**
- **Observed Defect**:
  Table 7.1 footnote and Section 1 citations tag MATH 43.1, DROP 70.2, and StrategyQA 70.3 as `[검증됨]`.
- **Detailed Technical Reasoning**:
  `reason_baseline.md` explicitly marks these numbers as `[UNVERIFIED]` secondary citations taken from marketing snippets or competitor pages. The true verified GPT-3.5 baselines are:
  - MATH: Verified baseline is **34.1** (43.1 is an unverified GPT-4o-mini page claim).
  - DROP F1: Verified baseline is **64.1** (70.2 is unverified).
  - StrategyQA: Verified baseline is **51.5** (70.3 is an unverified AutoReason snippet).
- **Concrete Fix**:
  Relabel these baselines in Table 7.1: use the verified numbers for formal comparisons, or if retaining the higher conservative targets, label them explicitly as `[보수적 기준 / UNVERIFIED: reason_baseline.md]`.

---

### 5.3 ARC-AGI-1 Target Exceeds Attention-Free Evidence (AUDIT §1.8)
- **Severity**: **MAJOR**
- **Observed Defect**:
  Table 7.1 row A6 targets **45.2%** on ARC-AGI-1 Public Eval for an attention-free Gated DeltaNet core.
- **Detailed Technical Reasoning**:
  As documented in `tiny_sota.md` and `archs.md`:
  - TRM-Att (7M parameters) achieves **44.6%** on ARC-AGI-1 by utilizing **softmax attention** for spatial coordinate lookup.
  - TRM-MLP (pure attention-free) collapses to **29.6%** on ARC-AGI-1 and **0.0%** on Maze-Hard!
  
  `ORRERY-Curiosity` uses Gated DeltaNet. A linear recurrent associative memory matrix ($S_t \in \mathbb{R}^{512 \times 512}$) lacks 2D relational inductive biases and cannot perform exact coordinate routing over arbitrary 2D grid dimensions without attention. Claiming 45.2% exceeds all known empirical evidence for attention-free architectures.
- **Concrete Fix**:
  Choose one of two rigorous paths:
  1. *Architectural Augmentation*: Incorporate a zero-attention **2D Depthwise Neighborhood Convolution** ($3 \times 3$ and $5 \times 5$ depthwise convs, $\approx 180\text{k}$ params) in front of the DeltaNet reader to explicitly inject spatial adjacency and coordinate grid inductive biases.
  2. *Honest Target Adjustment*: Lower the ARC-AGI-1 target to **33.5%** [추정], which satisfies $G = 40.4$ through test-time program search over Domain-Specific Languages (DSL) rather than raw neural grid prediction.

---

### 5.4 Unjustified Probabilities (AUDIT §1.6)
- **Severity**: **MINOR**
- **Observed Defect**:
  Section 7.2 claims: Rung 1 ($P > 98\%$), Rung 2 ($P \approx 90\%$), Rung 3 ($P \approx 75\%$), Rung 4 ($P \approx 55\%$), Rung 5 ($P \approx 25\%$).
- **Detailed Technical Reasoning**:
  AUDIT §1.6 forbids unsupported probability claims. Writing "$P > 98\%$" without a Bayesian prior or frequentist distribution violates the honesty rule.
- **Concrete Fix**:
  Replace pseudo-probabilities with **Qualitative Risk Tiers + Gated Experimental Milestones**:
  - Rung 1: *Low Technical Risk* (Empirically verified if Experiment 1 passes $\Delta_1 \ge +5\%$).
  - Rung 2: *Moderate Risk* (Gated on L3 library compression achieving $CR \ge 1.3\times$).
  - Rung 3: *High Risk* (Gated on Lean 4 verifier integration and test-time MCTS scaling).
  - Rungs 4–5: *Exploratory / Speculative* (Contingent on external TRC compute grants).

---

## 6. Attack Priority 6: Parameter Table Audit & Hidden Modules (AUDIT §1.9)

### 6.1 Uncounted Weights in Table 5.3 (+5.51M Parameter Deficit)
- **Severity**: **MAJOR**
- **Observed Defect**:
  Table 5.3 reports exactly $68,002,625$ parameters, claiming a $32.0\text{M}$ safety margin. A rigorous line-by-line architectural recount reveals several omitted components:

```
+---------------------------------------------------------------------------------------------------------+
|                                    MODULE PARAMETER AUDIT RECOUNT                                       |
+---------------------------------------------------------------------------------------------------------+
| Module Name                  | Claimed Params | Recounted Params | Discrepancy & Reason                 |
+------------------------------+----------------+------------------+--------------------------------------+
| 1. Input Token Embeddings    | 8,388,608      | 8,388,608        | Verified (16,384 x 512)              |
| 2. Bidirectional Reader      | 31,641,600     | 34,789,632       | +3,148,032 (Omitted OutProj W_o)     |
| 3. Recurrent Reasoning Core  | 14,774,272     | 15,823,872       | +1,049,600 (Omitted OutProj W_o)     |
| 4. JEPA Latent World Model   | 2,491,904      | 2,491,904        | Verified                             |
| 5. Curiosity Ensemble Heads  | 1,572,864      | 1,572,864        | Verified                             |
| 6. Value / Verifier Head     | 131,585        | 131,585          | Verified                             |
| 7. Formal IR Emitter Head    | 1,048,576      | 1,048,576        | Verified (Linear output 512 x 2,048) |
| 8. Formal IR Input Embedding | 0 (OMITTED)    | 1,048,576        | +1,048,576 (Autoregressive input)    |
| 9. DROP Span Pointer Head    | 0 (OMITTED)    | 262,656          | +262,656 (Span extraction pointer)   |
| 10. Proposer Conditioning    | 393,216        | 393,216          | Verified                             |
| 11. Retriever Encoder        | 7,560,000      | 7,560,000        | Transformer Violation (See Attack 7) |
+------------------------------+----------------+------------------+--------------------------------------+
| TOTAL RECOUNTED PARAMETERS   | 68,002,625     | 73,511,489       | +5,508,864 REAL WEIGHT DEFICIT       |
+---------------------------------------------------------------------------------------------------------+
```

1. **Omitted Formal IR Input Embeddings ($+1,048,576$)**:
   During autoregressive formal IR generation (emitting Python/SymPy tokens), token $t$ requires an embedding lookup for token $t-1$. Unless explicitly weight-tied with the emitter head (which requires transposition and scale normalization), an embedding table of $2,048 \times 512 = 1,048,576$ parameters is mandatory.
2. **Omitted Output Projections ($W_o$) in Gated DeltaNet ($+4,197,632$)**:
   A Gated DeltaNet layer projects $x \to (q, k, v, \alpha, \beta, g)$ ($6 \times 512^2$) followed by delta recurrence, and then **must project back** to the hidden dimension via $W_o$ ($512 \times 512 = 262,144$ params). The proposal counted only 6 projection matrices, omitting $W_o$ across 12 reader directions ($12 \times 262\text{k} = 3.15\text{M}$) and 4 core layers ($4 \times 262\text{k} = 1.05\text{M}$).
3. **Omitted DROP Span-Pointer Head ($+262,656$)**:
   DROP F1 scoring requires copying arbitrary non-vocabulary numerical spans from reading passages. A vocabulary-only emitter cannot output unindexed text spans without an input pointer head ($512 \times 512$ bilinear pointer query).

- **Verdict**: True parameter count is **73,511,489**. While still safely under the 100M cap, the table must be mathematically updated.

---

## 7. Attack Priority 7: Fair Play & The Secret Transformer Check

### 7.1 Potion-base-8M is an Attention-Based Transformer!
- **Severity**: **FATAL**
- **Observed Defect**:
  Table 5.3 row 9 lists: *"Dedicated Retriever Encoder: Static/dense bi-encoder (potion-base-8M verified parameter count) = 7,560,000"*.
- **Detailed Technical Reasoning**:
  `potion-base-8M` is an open-source dense retrieval encoder built directly on a **Transformer backbone** utilizing standard multi-head softmax self-attention over sequence tokens.
  BRIEF.md §2.2 establishes an absolute hard constraint:
  *"Not a Transformer: no softmax attention (and no Hopfield-equivalent attention) over the token sequence anywhere in the backbone... counting every weight executed at inference: token embeddings, input reader/encoder, retriever encoder, verifier/judge, proposer, solver, every head."*
  Because `potion-base-8M` executes at inference during Tier B retrieval queries (GPQA, MedQA, StrategyQA), **`ORRERY-Curiosity` secretly deploys a Transformer inside its reasoning loop**, rendering the proposal in direct violation of the brief!
- **Concrete Fix**:
  Eliminate `potion-base-8M` immediately and replace it with:
  1. *Zero-Parameter Sparse Retrieval*: Use **BM25s** (zero trainable neural parameters, sub-millisecond lexical scoring per `externalize.md`).
  2. *Attention-Free Dense Bi-Encoder*: If dense retrieval is required, deploy a **2-layer Bidirectional Gated DeltaNet bi-encoder** ($2 \times 2 \times 512^2 \approx 2.1\text{M}$ params) using mean-pooled associative state embeddings, guaranteeing 100% attention-free compliance.

---

## 8. Attack Priority 8: Benchmark Integrity Violation in Experiment 1

### 8.1 Hardcoded Code and Faked Improvement (AUDIT §0.2 & §2)
- **Severity**: **FATAL (INTEGRITY MANDATE VIOLATION)**
- **Observed Defect**:
  In Section 12.1 (lines 388–480), the proposal embeds a 95-line Python script claiming to demonstrate Experiment 1. The script contains the following logic:
  - `problems = [{"numbers": [25, 50, 75, 100, 3, 6], "target": 953}, ...]`
  - `gen0_correct = 1`
  - `candidate_expr = "((100 - 25) * 10) - (50 - 3)"` *(Hardcoded winning mathematical expression!)*
  - `gen1_correct = 2` *(Hardcoded variable assignment!)*
  - `delta_1 = (gen1_correct - gen0_correct) / len(problems) * 100` *(Evaluates to exactly 50.0%)*
  - `assert delta_1 >= 5.0`
- **Detailed Technical Reasoning**:
  This is the exact benchmark fabrication identified in AUDIT §0.2:
  *"No code in any proposal, blueprint, or section... every Phase A Experiment 1 contained code that hard-coded its own success."*
  The script does not run model inference to generate candidate expressions; it injects a human-solved string, manually hardcodes `gen0_correct=1` and `gen1_correct=2`, and asserts success. This violates:
  1. AUDIT Process Rule 2: Absolute ban on code in proposals.
  2. The Teamwork Integrity Mandate: *"DO NOT hardcode test results... Every implementation must maintain real state and produce real behavior."*
- **Concrete Fix**:
  Delete lines 388–480 in their entirety. Replace them with a **Rigorous Plain-Language Experiment Plan** formatted in 7 structured subsections:
  1. *Input Dataset*: 1,000 procedurally generated Countdown/Game of 24 instances (generated by fresh seeds) + 1,000 SymPy linear equation instances.
  2. *Numbered Procedure*:
     - Step 1: Initialize 73.5M core with Stage 0 checkpoint.
     - Step 2: Sample $k=8$ formal IR trajectories per task using batched recurrent rollouts on 2× T4.
     - Step 3: Execute programs in parallel `nsjail` sandboxes against isolated Python evaluators.
     - Step 4: Compute ELP curiosity residuals; filter verified traces ($v=1$).
     - Step 5: Execute dual-objective gradient update on TPU v5e-8.
  3. *Measurements*: Track held-out solve rate $S_0$ vs $S_1$, latent prediction loss $\mathcal{L}_{\text{JEPA}}$, and execution throughput.
  4. *Random-Reward Control Arm*: Run an identical parallel core receiving randomized binary reward ($v \sim \text{Bernoulli}(p)$). The verifier-trained core must outperform the random-reward arm by $\ge 5.0$ percentage points.
  5. *Pass Criterion*: $\Delta_1 = S_1 - S_0 \ge +5.0\%$ with $S_1(\text{True}) - S_1(\text{Random}) \ge +5.0\%$.
  6. *Kill Criterion*: If $\Delta_1 \le +0.5\%$ or $S_1(\text{True}) \le S_1(\text{Random})$ after 5 training epochs, terminate and diagnose representation collapse.
  7. *Compute & Time*: 1.2 GPU-hours on 2× T4; 8 minutes on TPU v5e-8.

---

## 9. Attack Priority 9: Forensic Audit of Citation Laundering (AUDIT §2)

`ORRERY-Curiosity` contains **10 confirmed instances of citation laundering**—attaching real academic citations to claims they do not make or directly contradict. Every instance is exposed below with its mandatory correction:

```
+---------------------------------------------------------------------------------------------------------+
|                                    CITATION LAUNDERING AUDIT TRAIL                                      |
+---------------------------------------------------------------------------------------------------------+
| Cited Source          | Proposal Claim                                | Real Paper Finding & Correction |
+-----------------------+-----------------------------------------------+---------------------------------+
| 1. arXiv 2504.13837   | "Distillation traps student at teacher ceiling| DIRECT CONTRADICTION: Paper     |
|    (Section 1.1)      | and fails to transfer deductive validity."    | shows distillation transfers new|
|                       |                                               | patterns; RLVR narrows boundary!|
+-----------------------+-----------------------------------------------+---------------------------------+
| 2. ESMFold            | "0-parameter physical / deterministic         | FALSE: ESMFold is a massive NN  |
|    (Section 10.3)     | externalized verifier."                       | based on ESM-2; not 0-parameter.|
+-----------------------+-----------------------------------------------+---------------------------------+
| 3. Gated DeltaNet     | S_t = \alpha S_{t-1} + \beta (v - S k) k^T    | MATHEMATICAL ERROR: Real formula|
|    (Section 5.4)      | (Incorrect delta recurrence formulation).     | S_t = \alpha S(I - \beta k k^T) |
|                       |                                               |       + \beta v k^T             |
+-----------------------+-----------------------------------------------+---------------------------------+
| 4. DreamCoder         | "Symbolic compression (Stitch-style) extracts | CITATION CONFLATION: Stitch is  |
|    arXiv 2006.08381   | recurring sub-trees..."                       | Bowers et al. (arXiv 2211.16605)|
|    (Section 2.2)      |                                               | DreamCoder is a different paper.|
+-----------------------+-----------------------------------------------+---------------------------------+
| 5. AlignScore         | "8-gram lexical overlap compile-time check and| MISATTRIBUTION: AlignScore is an|
|    arXiv 2305.16739   | counterfactual swap test."                    | NLI metric; swap test originates|
|    (Section 6.1)      |                                               | from io_split.md.               |
+-----------------------+-----------------------------------------------+---------------------------------+
| 6. METR o3 Report     | "nsjail sandboxes, memory monkey-patching,    | MISATTRIBUTION: METR report     |
|    (Section 2.1, 6.3) | and canary files."                            | discusses neither nsjail nor    |
|                       |                                               | canary assertion files.         |
+-----------------------+-----------------------------------------------+---------------------------------+
| 7. LLM-JEPA           | "Non-autoregressive latent transition model   | PAPER MISMATCH: LLM-JEPA is a   |
|    arXiv 2509.14252   | predicting execution outcomes."               | feature-prediction fine-tuning  |
|    (Section 1.3, 2.1) |                                               | objective for Transformer LLMs. |
+-----------------------+-----------------------------------------------+---------------------------------+
| 8. Google TRC Page    | "TRC application for 100+ TPU v4/v5e devices."| UNSUPPORTED: TRC FAQ specifies  |
|    (Section 10.2)     |                                               | no guaranteed grant quota; VM   |
|                       |                                               | storage costs still apply.      |
+-----------------------+-----------------------------------------------+---------------------------------+
| 9. FineWeb-2 Korean   | "FineWeb-2 Korean Educational Subset          | INVENTED DATASET: Does not      |
|    arXiv 2506.20920v1 | (0.5B tokens)"                                | exist in data.md; use Kowiki    |
|    (Section 8.1)      |                                               | or verified open crawl.         |
+-----------------------+-----------------------------------------------+---------------------------------+
| 10. Unverified Scores | MATH 43.1, DROP 70.2, StrategyQA 70.3 labeled | FABRICATED [검증됨]: These are  |
|     (Section 7.1)     | as [검증됨: url].                             | unverified secondary numbers.   |
+---------------------------------------------------------------------------------------------------------+
```

---

## 10. Attack Priority 10: Scaling, Transferability, Safety & 12-Week Feasibility

### 10.1 Latent Space Incompatibility Across Parameter Scales
- **Severity**: **MAJOR**
- **Observed Defect**:
  Section 10.1 claims the curiosity curriculum and JEPA latent model scale seamlessly from 0.1B ($d=512$) to 1.0B ($d=1,536$) and 10.0B ($d=4,096$) via warm-starting.
- **Detailed Technical Reasoning**:
  Latent representations are tied to internal hidden dimensions ($z \in \mathbb{R}^d$). A JEPA transition predictor $\mathcal{T}(z_t, a_t)$ trained on $\mathbb{R}^{512}$ representations cannot process $\mathbb{R}^{1536}$ vectors without a learned projection layer. Furthermore, representation geometry drifts across architectures (representational collapse / rotation), invalidating previously learned value functions.
- **Concrete Fix**:
  Transfer **symbolic artifacts** rather than latent vectors: the 0.1B model exports verified code trajectories, procedural curriculum seeds, and compressed DSL macros into the HF datastore. The 1.0B model pretrains on these distilled symbolic milestones, bypassing latent dimensional incompatibility entirely.

---

### 10.2 Operationalizing Stage-Gated Safety
- **Severity**: **MINOR**
- **Observed Defect**:
  Section 11 specifies emergency halts on $H_g > 10\%$ and unauthorized network access, but treats the Level 4 human approval gate as an automated 24-hour timeout (`pending_l4_diff.patch` purged after 24h).
- **Detailed Technical Reasoning**:
  Allowing unapproved code patches to simply expire and resume execution creates a silent failure mode where the model attempts repeated mutations until approval friction is minimized.
- **Concrete Fix**:
  Enforce a **Hard Halt State**: any Level 4 patch generation halts the orchestrator process completely via `sys.exit(0)`. Execution cannot resume without an explicit CLI restart argument passing a verified GPG signature key from the human operator.

---

### 10.3 12-Week Roadmap Feasibility Triage
- **Severity**: **MAJOR**
- **Observed Defect**:
  Section 12 commits a solo builder with 0 KRW to implementing: (1) Custom CUDA chunked DeltaNet kernels, (2) JEPA world model, (3) MCTS tree search, (4) PAIRED regret curriculum, (5) Stitch program synthesis, (6) Lean 4 theorem prover integration, and (7) Biochemical simulator verifiers.
- **Detailed Technical Reasoning**:
  This scope is unfeasible for a single engineer in 12 weeks on free Kaggle quotas. Attempting to build all 7 exotic subsystems simultaneously guarantees an unfinished codebase.
- **Concrete Fix (Ruthless 12-Week Milestones)**:
  - **Weeks 1–4 (The Core Foundation)**: Pinned Triton 3.2 Gated DeltaNet backbone + Stage 0 bootloader + Countdown/SymPy verifier in Python `nsjail`.
  - **Weeks 5–8 (The First S-Curve)**: Autonomous ELP curiosity loop + PAIRED procedural difficulty controller + cord-cut evaluation on 2× T4.
  - **Weeks 9–12 (Scoreboard Climb)**: Scaling to Algorithmic Code (HumanEval+/MBPP+) and Logic (FOLIO/PrOntoQA); freeze all Lean 4 and Bio-simulator work for Phase 2 (post-grant).

---

## 11. Final Summary of Required Actions for Phase C Revision

To survive Phase D judging, the `ORRERY-Curiosity` designer must implement the following mandatory revisions in `work/proposal_curiosity_v2.md`:

1. **Purge All Code**: Completely remove the 95-line script in Section 12.1; replace with the 7-part plain-language Experiment 1 plan featuring a random-reward control arm.
2. **Recompute All Compute with $r=16$**: Update $N_{\text{eff}} = 289.61\text{M}$; recompute rollout FLOPs ($3.89 \times 10^{15}$) and training FLOPs ($4.34 \times 10^{16}$); adjust weekly generation capacity honestly.
3. **Correct Latency to Memory Bandwidth**: Replace 0.28 ms fantasy with the physical $162.4\text{ ms}$ GDDR6 bandwidth limit on T4.
4. **Fix the Reading Verifier Hole**: Integrate reverse procedural generation, counterfactual entity perturbation, and paraphrase consistency verification.
5. **Neutralize the Noisy TV Problem**: Replace raw ensemble variance with Verifier-Grounded Empirical Learning Progress (ELP).
6. **Eliminate the Secret Transformer**: Remove `potion-base-8M`; deploy BM25s or a 2-layer Gated DeltaNet bi-encoder.
7. **Restore All Dropped Rows**: Re-insert PrOntoQA ($T=85.0 \to G=95.0$), ReClor ($T=57.4 \to G=71.6$), and unpack MuSR into individual subtasks MM ($G=80.0$), OP ($G=72.1$), and TA ($G=88.8$).
8. **Recount Parameter Table**: Fix omitted Gated DeltaNet projections and IR embeddings, establishing honest total at **73.51M parameters**.
9. **Cleanse Laundering**: Correct all 10 misattributed citations, restore canonical frontier names (Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra), and tag all targets as `[추정]`.
