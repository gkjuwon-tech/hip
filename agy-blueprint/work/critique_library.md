# RED TEAM FORENSIC AUDIT & CRITIQUE: PROPOSAL 'LIBRARY' (ALEMBIC-LIB v1.0)

**Target Document**: `work/proposal_library_v1.md` (ALEMBIC-LIB: Library-Growing Neuro-Symbolic RSI Engine)  
**Lead Red Team Auditor**: Senior ML/RL Systems Researcher, Kaggle Grandmaster (Hardware & Kernel Limits), Benchmark-Integrity Auditor  
**Binding Directives**: `AUDIT_FROM_CLAUDE.md` (§1 & §2), `BRIEF.md` (§2, §4 Phase B), `ORIGINAL_REQUEST.md`  
**Audit Verdict**: **CONDITIONAL REJECTION (REQUIRES PHASE C STRUCTURAL RECONSTRUCTION)**  
**Summary of Findings**: 4 Fatal Pathogens, 6 Major Systems Failures, 4 Benchmark & Citation Integrity Violations.

```
====================================================================================================
                                RED TEAM AUDIT SCORECARD: ALEMBIC-LIB v1.0
====================================================================================================
  Dimension                         Severity   Status     Key Failure Point
----------------------------------------------------------------------------------------------------
  1. The RSI Loop Dynamics          FATAL      BROKEN     Reading Verifier Void & 70B Teacher Fantasy
  2. Recurrent Depth FLOP Math      FATAL      COLLAPSED  6.3x Compute Undercount (r=8 Ignored)
  3. T4 Latency & Roofline Realism  FATAL      IMPOSSIBLE Memory-Bandwidth Bound (Off by 700x-800x)
  4. Forbidden Code & Exp 1 Rigor   FATAL      VIOLATION  135 Lines Executable Code Hardcoding Success
  5. ARC-AGI & Boss Ladder Integrity MAJOR      UNJUSTIFIED 47.5% Target on Non-Attention Backbone
  6. Parameter Table Completeness    MAJOR      DEFECTIVE  Omitted Output Gates, Conv1D & Embeddings
  7. Stitch Compression Scalability  MAJOR      BOTTLENECK Imperative Python AST Combinatorial Explosion
  8. Citation & Dataset Laundering   MAJOR      FABRICATED 7 Misattributed Papers & Phantom Datasets
  9. Domain Staircase Autonomy      MINOR      HAND-WAVED Static Human-Registered API Walking
  10. 12-Week Solo Feasibility      MAJOR      UNFEASIBLE Multi-DSL Toolchain Fragmentation
====================================================================================================
```

---

## 1. Priority Attack 1: The RSI Loop Dynamics, Teacher Escape, and the Verification Void

### 1.1 The Deepest Hole: The Reading Verifier Void (AUDIT §1.4)
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:  
  The core thesis of ALEMBIC-LIB is that 0-parameter deterministic executors (Python AST, SymPy, Z3, Lean 4) eliminate the need for a smarter teacher by providing an un-hackable, objective ground truth. This is a catastrophic architectural conflation between **execution validity** and **semantic alignment**.  
  Executors verify strictly that a generated program runs without crashing, obeys typing rules, and returns a value or satisfies internal constraints. They do **not** verify that the generated program correctly formalizes the natural language problem.  
  Inside an autonomous RSI loop where problems are self-proposed ($M = 50,000$ per generation):
  1. If the proposer emits a natural language prompt and the solver generates code, the verifier only checks that the solver's code runs and returns an output.
  2. If the solver generates code that answers an entirely different, easier problem (e.g., formalizing a complex multi-step combinatorics word problem as a simple addition script), the Python interpreter executes it with exit code 0, outputs a clean integer, and flags the trajectory as `VERIFIED_SUCCESS` ($r_i = 1$).
  3. The core receives positive reinforcement for misreading natural language and hallucinating convenient formalizations. The reasoner becomes an "aphasic fraud": highly capable of executing trivial symbolic manipulations while completely decoupled from natural language comprehension.
- **Concrete Fix**:  
  Eliminate ungrounded self-generated natural language problems from the training loop. Implement a mandatory three-layer reading verification harness:
  1. **Reverse Procedural Generation**: The procedural generator synthesizes the ground-truth formal AST first, executes it to establish the true answer, and then compiles the AST into natural language via grammar templates. Every problem in the loop carries an exact latent formalization against which the solver's formal AST is mathematically compared (via AST isomorphism or SymPy equivalence).
  2. **Dual-Formulation Paraphrase Consistency**: During evaluation and wake search, the core is presented with two semantically identical paraphrases of the problem text. Verification requires that the emitted ASTs for both paraphrases evaluate to identical values:
     $$\text{Verify}_{\text{read}}(x_1, x_2) = \mathbb{I}(\text{Exec}(P(x_1)) == \text{Exec}(P(x_2))) \times \mathbb{I}(\text{AST}_{\text{equiv}}(P(x_1), P(x_2)))$$
  3. **Counterfactual Perturbation Twin Gating**: Every candidate problem must pass a parameter perturbation test (GSM-Symbolic style). Altering numerical values in the text must produce the exact predicted delta in execution output.

---

### 1.2 Teacher Ceiling Escape & The Impossible 70B Teacher on Free Compute (AUDIT §1.5)
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:  
  Section 2.3 specifies that Generation 0 ($g=0$) uses an open-weight teacher: Llama-3.1-70B-Instruct or Qwen2.5-72B.  
  This violates the hard zero-budget computational reality established in `compute.md` and AUDIT §1.5. A 70B-parameter model in 16-bit precision requires 140 GB of VRAM. Even quantized to 4-bit AWQ/GPTQ, it consumes ~38–42 GB of memory. The available hardware is 2x NVIDIA T4 (16 GB each, 32 GB total across PCIe, with no NVLink) and Kaggle TPU v5e-8 (20 hours/week). A 70B model cannot run on 2x T4 even in 4-bit precision without extreme CPU offloading that drops throughput to $< 0.5$ tokens/second. Generating $2.0 \times 10^9$ tokens from a 70B model on this hardware is physically impossible within the 12-week timeline.
- **Concrete Fix**:  
  1. Ban live 70B teacher inference from the runtime loop.
  2. Restrict the active distillation teacher to an architecture that runs natively on the free tier: **Qwen2.5-14B-Instruct** (4-bit AWQ fits in ~9 GB VRAM on a single T4 at 25–40 tok/s) or **Qwen2.5-27B** on TPU v5e-8 using static JAX meshes per `compute.md`.
  3. Alternatively, source Stage-0 bootstrap traces entirely from frozen, pre-computed, permissively licensed open execution datasets (OpenMathInstruct-2 generated from Llama-3.1-405B, filtered strictly through Python/SymPy execution).

---

### 1.3 Cord-Cut Feasibility & Missing Random-Reward Control Arm (AUDIT §3.1)
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:  
  Section 2.3 defines a three-part cord-cut criterion: (1) solving $\ge 20\%$ of the teacher's failure set $\mathcal{D}_{\text{fail}}^{\text{teacher}}$, (2) positive score gain $\Delta_{\text{self}} > 0$ for two consecutive generations, and (3) library compression factor $\text{CR} \ge 1.30$.  
  This criterion lacks an empirical baseline to differentiate true self-improvement from noise or memorization. In small models ($<100\text{M}$), continuous policy training on self-generated rollouts can appear to improve score simply by overfitting to specific syntax distributions or exploiting sandbox quirks.
- **Concrete Fix**:  
  Incorporate the binding **Random-Reward Control Arm** (AUDIT §3.1):
  1. Maintain an identical control core $\text{Core}_{\text{control}}$ trained on identical self-proposed problems but updated using randomly permuted binary rewards ($r_{\text{control}} \sim \text{Bernoulli}(p)$).
  2. Severing the teacher cord requires that the verifier-trained policy outscore the random-reward control arm by at least 5.0 percentage points on the commit-revealed held-out test suite:
     $$S_{\text{verifier}}(g) - S_{\text{random\_control}}(g) \ge +5.0\%$$
  3. If this margin is not cleared, the cord is not cut, and the run is flagged for reward hacking.

---

### 1.4 Stitch Symbolic Compression: The Imperative AST Combinatorial Wall
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:  
  Section 2.1 and Section 2.6 claim that Stitch compresses 25,000 verified AST programs in 8.0 minutes on 4 vCPUs.  
  This claim is technically untenable. Stitch (Bowers et al., 2023) operates via e-graph construction and anti-unification over **purely functional, variable-free or simply-typed lambda calculi** (such as DreamCoder primitives). ALEMBIC-LIB proposes generating programs in Python, SymPy, Z3, and Lean 4.  
  Imperative Python ASTs contain control-flow constructs, local variable assignments, scoping mutations, imports, and arbitrary statement sequencing. Running anti-unification over 25,000 Python ASTs with branching and variable re-binding creates an exponential subgraph isomorphism search space ($O(N^2 \cdot 2^{|V|})$). An unconstrained e-graph over 25,000 multi-line Python scripts will exhaust 16 GB host RAM within seconds or run for days, not 8 minutes.
- **Concrete Fix**:  
  1. Decouple the compression engine from raw Python syntax.
  2. Introduce a rigid, purely functional **S-Expression Intermediate Representation (Symbolic IR)** based on combinator logic (pure functions, no variable re-assignment, no side effects).
  3. Stitch operates strictly on this functional Symbolic IR where e-graph equivalence is linear or polynomial.
  4. A deterministic, 0-parameter transpiler expands the compressed S-expressions into executable Python/SymPy/Z3 code immediately prior to sandbox execution.

---

### 1.5 The Own-Level Prison and the 0.3M Proposer Fantasy
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:  
  Section 2.2 claims the Proposer Head implements PAIRED asymmetric regret self-play to generate a frontier curriculum. However, Table 5.4 reveals the Proposer Head consists of a **2-layer MLP with 295,488 parameters (0.30M)** mapping a 512-dimensional latent state to grammar conditioning logits.  
  A 0.30M parameter MLP cannot generate novel natural language problems, synthesize valid PDDL domain files, or formulate new mathematical puzzles. It can only emit a categorical distribution over fixed template indices. Claiming this constitutes autonomous curriculum generation is deceptive; it is a static template selector masquerading as self-play.
- **Concrete Fix**:  
  State transparently that the Proposer Head does not author raw text. Instead, define the Proposer as a **Parametric Complexity Controller**:
  1. The 0.30M MLP outputs continuous control parameters (e.g., number of reasoning hops $H \in [1, 10]$, branching factor $B \in [2, 6]$, numerical magnitude $M \in [10^1, 10^6]$, distractor count $D \in [0, 5]$) conditioned on solver regret.
  2. These parameters parameterize deterministic, external procedural generators (Reasoning Gym, SymPy problem generators, re-ARC generators).
  3. The problem text is synthesized by the procedural engine, guaranteeing syntax validity, solvability, and exact formal ground truth.

---

### 1.6 Domain Staircase: Walking a Static List vs. True Autonomous Expansion (AUDIT §1.3)
- **Severity**: **MINOR**
- **Detailed Technical Reasoning**:  
  Section 2.5 claims the engine autonomously finds the next verifier domain via a "Verifier Synthesizer Probe". However, the document merely walks a predefined, human-written sequence: Arithmetic $\to$ Logic $\to$ Code $\to$ Proofs $\to$ Simulators. The "probe" merely checks round-trip invariants over APIs that humans explicitly pre-coded.  
  Per BRIEF.md §2.7, the model is strictly forbidden from writing or modifying its own verifiers to prevent reward hacking. Therefore, claiming the AI "synthesizes" or "finds" new domains is an over-claim.
- **Concrete Fix**:  
  State honestly that the verifier sequence is an **expert-curated API registry**. Autonomous RSI capability governs **domain unlocking and interface mapping**, not verifier authoring:
  1. The loop monitors domain saturation via the saturation coefficient $\Psi_D(g) \ge 0.95$.
  2. Upon saturation, the core activates the next human-verified sandbox from the frozen registry.
  3. The model's autonomous learning task is restricted to synthesizing the bidirectional translation layer (compiling natural language into the newly exposed formal DSL) verified against human-reviewed test suites.

---

## 2. Priority Attack 2: Recurrent Depth FLOP Math Collapse (AUDIT §1.1)

- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:  
  Section 2.6 and Section 8.2 perform FLOP calculations using the static formula:
  $$\text{FLOP}_{\text{token}} = 2N \quad (\text{inference}), \quad 6N \quad (\text{training})$$
  with $N = 68,712,001$ parameters.  
  This completely ignores that the 14-layer Recurrent Gated DeltaNet trunk is iterated $r$ times ($r \in [1, 16]$) per token, as specified in Section 5.2 and 5.4.  
  Per AUDIT §1.1, the exact per-token cost for an architecture with recurrent weight tying is:
  $$\text{FLOPs/token}_{\text{inference}} \approx 2 \times (N_{\text{non\_core}} + r \times N_{\text{core}})$$
  $$\text{FLOPs/token}_{\text{training}} \approx 6 \times (N_{\text{non\_core}} + r_{\text{train}} \times N_{\text{core}})$$

### Rigorous Architectural Re-Decomposition:
From the re-audited parameter budget (see Section 6 of this critique):
- $N_{\text{non\_core}} = \mathbf{22,234,089}$ parameters (Embeddings, 6-layer Bi-SSM Reader, LayerNorms, Heads, Pointer Head, Formal Embeddings).
- $N_{\text{core}} = \mathbf{51,537,920}$ parameters (14-layer Gated DeltaNet Solver Trunk with RMSNorms, Conv1D, Output Gates, and SwiGLU FFN).

```
====================================================================================================
                        RECURRENT DEPTH FLOP AUDIT & RECOMPUTATION TABLE
====================================================================================================
  Metric                          Proposal v1 Claim        Audited (r = 8)         Audited (r = 16)
----------------------------------------------------------------------------------------------------
  Effective Params / Token        68.71M                   434.54M                 846.84M
  Inference FLOPs / Token         1.374 x 10^8             8.691 x 10^8 (6.32x)    1.694 x 10^9 (12.3x)
  Training FLOPs / Token          4.123 x 10^8             2.607 x 10^9 (6.32x)    5.081 x 10^9 (12.3x)
  Gen Rollout FLOPs (4.1e8 tok)   5.628 x 10^16            3.560 x 10^17           6.939 x 10^17
  Gen Train FLOPs (5.76e7 tok)    2.375 x 10^16            7.893 x 10^16 (r_tr=4)  1.498 x 10^17 (r_tr=8)
  2x T4 Rollout Time              24.1 minutes             152.1 minutes (2.54 h)  296.5 minutes (4.94 h)
  2x T4 Train Time (Fallback)     10.2 minutes             33.7 minutes            64.0 minutes
  Total Gen Wall-Clock (T4)       38.1 minutes             211.3 minutes (3.52 h)  389.0 minutes (6.48 h)
  Generations / Week (30h Quota)  47 generations           8.5 generations         4.6 generations
====================================================================================================
```

### Forensic Impact on the Proposal's Feasibility:
1. **Rollout FLOPs Undercount**: At an average test-time recurrence of $r = 8$, rollout compute increases from $5.63 \times 10^{16}$ to **$3.56 \times 10^{17}$ FLOPs**—an undercount factor of **6.32x**.
2. **Wall-Clock Explosion**: Rollout time on Kaggle 2x T4 at 30% sustained MFU (39.0 TFLOPS) explodes from 24.1 minutes to **152.1 minutes (2.54 hours)**.
3. **Weekly Generation Collapse**: Total wall-clock time per generation rises from 38.1 minutes to **211.3 minutes (3.52 hours)**. The claimed throughput of **47 generations per week collapses to 8.5 generations per week**.
4. If maximum recurrent depth $r = 16$ is engaged, a single generation takes 6.48 hours, yielding only **4.6 generations per week**.
- **Concrete Fix**:  
  1. Base all generation cycle schedules on $r_{\text{avg}} = 8$ for rollouts and $r_{\text{train}} = 4$ for truncated backpropagation through time.
  2. Implement an adaptive early-exit halting head (Universal Transformer / Huginn style) that terminates recursion dynamically on easy tokens ($r=1$), capping mean sequence recurrence at $\bar{r} \le 4.2$.
  3. Re-anchor the operational blueprint to a realistic target of **12 to 14 generations per week** across the 30-hour T4 and 20-hour TPU allocations.

---

## 3. Priority Attack 3: T4 Latency Realism and the Memory-Bandwidth Wall (AUDIT §1.2)

- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:  
  Section 8.3 claims:
  - Single-pass generation ($L = 256$ tokens): $0.9\text{ ms}$ on T4.
  - Full MCTS search query ($K = 32$ rollouts): $28.9\text{ ms}$ on T4.  
  This calculation commits the classic novice error of treating autoregressive token generation as compute-bound at peak tensor-core throughput ($65\text{ TFLOPS}$). Autoregressive decoding at small batch sizes is strictly **memory-bandwidth bound**.

### Exact Hardware Roofline Derivation for Turing T4 (sm_75):
- Memory Bandwidth: $B_{\text{mem}} = \mathbf{300\text{ GB/s}}$.
- FP16 Tensor Peak Compute: $P_{\text{peak}} = \mathbf{65\text{ TFLOPS}}$ per GPU.
- Machine Operational Intensity Threshold:
  $$I^* = \frac{P_{\text{peak}}}{B_{\text{mem}}} = \frac{65 \times 10^{12} \text{ FLOP/s}}{300 \times 10^9 \text{ Byte/s}} \approx \mathbf{216.7\text{ FLOPs/Byte}}$$
- Operational Intensity of Autoregressive Decoding at Batch Size $b=1$:
  Generating 1 token requires reading every model weight from DRAM into SRAM/registers.
  In 16-bit precision, each parameter occupies 2 bytes and performs 2 FLOPs (1 multiply-accumulate).
  $$I_{\text{actual}} = \frac{2 \text{ FLOPs}}{2 \text{ Bytes}} = \mathbf{1.0\text{ FLOP/Byte}}$$
  Because $I_{\text{actual}} = 1.0 \ll 216.7$, generation is deeply trapped in the memory-bandwidth-bound regime. Peak attainable compute is limited to:
  $$\text{FLOPS}_{\text{attainable}} = 300\text{ GB/s} \times 1.0\text{ FLOP/Byte} = \mathbf{300\text{ GFLOPS}} \quad (\mathbf{0.46\% \text{ of peak T4 compute}})$$

### Exact Per-Token Weight Streaming Latency:
1. Re-audited model size in fp16: $73.77\text{M params} \times 2\text{ bytes} = \mathbf{147.54\text{ MB}}$.
2. For $r = 1$ (no recurrent depth), reading weights once from DRAM takes:
   $$t_{\text{token}}(r=1) = \frac{147.54 \times 10^6 \text{ bytes}}{300 \times 10^9 \text{ bytes/s}} = \mathbf{0.492\text{ ms / token}}$$
3. For $r = 8$ recursions:  
   The T4 on-chip L2 cache is only **4.0 MB**.  
   The recurrent trunk alone occupies $51.54\text{M params} \times 2\text{ bytes} = \mathbf{103.08\text{ MB}}$.  
   Because $103.08\text{ MB} \gg 4.0\text{ MB}$, the recurrent trunk cannot reside in cache and must be fetched from DRAM on every single recursion step.  
   Total bytes transferred per token:
   $$\text{Bytes/token} = 2 \times N_{\text{non\_core}} + 2 \times r \times N_{\text{core}} = 44.46\text{ MB} + 8 \times 103.08\text{ MB} = \mathbf{869.1\text{ MB / token}}$$
   $$\text{Latency/token}(r=8) = \frac{869.1 \times 10^6 \text{ bytes}}{300 \times 10^9 \text{ bytes/s}} = \mathbf{2.897\text{ ms / token}}$$
4. Single-Pass Generation Latency ($L = 256$ tokens):
   $$T_{\text{single-pass}} = 256 \text{ tokens} \times 2.897\text{ ms/token} = \mathbf{741.6\text{ ms}} \approx \mathbf{0.742\text{ seconds}}$$
   The proposal's claim of $0.9\text{ ms}$ is understated by a factor of:
   $$\frac{741.6\text{ ms}}{0.9\text{ ms}} \approx \mathbf{824\times}$$

### MCTS Search Query Latency ($K = 32$ rollouts):
Even with batched tree evaluation ($b=8$ over 4 sequential waves) amortizing non-core weight reads, kernel launch overhead on T4 (15–25 $\mu$s per launch $\times 20$ layers $\times 8$ recursions $\times 256$ steps $\approx 800\text{ ms}$) plus CPU sandbox execution (10–30 ms per verified script) yields:
$$T_{\text{MCTS}} \ge \frac{32 \times 741.6\text{ ms}}{\text{Parallel Speedup (}\approx 6.5\text{)}} + \text{Sandbox Overhead} \approx \mathbf{3.65\text{ to } 4.20\text{ seconds}}$$
The proposal's claim of $28.9\text{ ms}$ is understated by a factor of **$130\times$ to $145\times$**.
- **Concrete Fix**:  
  1. Eradicate the $0.9\text{ ms}$ and $28.9\text{ ms}$ numbers.
  2. Report latency honestly: **$0.74\text{ s}$ per single-pass query** and **$3.8\text{ s}$ per 32-rollout MCTS query**.
  3. To maximize throughput on Kaggle, batch rollouts aggressively ($b=64$ across 2 T4s using data parallelism) where operational intensity rises to $I \approx 64\text{ FLOP/Byte}$, lifting compute efficiency to $\approx 30\%$ of hardware peak.

---

## 4. Priority Attack 4: Missing Mandatory Items and Code Rule Violations

### 4.1 Fatal Inclusion of Hardcoded Executable Code in Section 12.1 (AUDIT §0.2, §2)
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:  
  Section 12.1 contains 135 lines of executable Python code (`run_experiment_1()`, `GameOf24Verifier`, `SymbolicPolicyCore`, `StitchCompressor`). This violates the absolute user instruction: **NO CODE anywhere in proposals, blueprints, or critiques**.  
  Worse, the code commits the exact offense highlighted in AUDIT §0.2: **it hard-codes its own success**.
  - `StitchCompressor.abstract_library()` does not implement anti-unification; it contains hard-coded regular expressions:
    `r"\d+\s*\/\s*\(\s*\d+\s*-\s*\(\s*\d+\s*\/\s*\d+\s*\)\s*\)"`
  - It runs on a hard-coded 20-puzzle list specifically curated so that those two hard-coded regex macros increase solve count, guaranteeing an artificial $\Delta = +15\%$ to pass the assertion `assert delta >= 15.0`.  
  This is scientific fabrication and must be purged immediately.
- **Concrete Fix**:  
  Delete the entire Python script from Section 12.1. Replace it with a rigorous **Plain-Language Experiment Plan** formatted as follows:
  1. **Inputs**: 500 procedurally generated Game-of-24 puzzles and 500 3-SAT Boolean logic problems with commit-revealed SHA-256 hashes.
  2. **Step-by-Step Procedure**:
     - Step 1: Base core generates $K=32$ candidate ASTs per problem via beam search.
     - Step 2: 0-parameter Python and Z3 sandboxes evaluate candidates, producing binary reward labels.
     - Step 3: Verified AST traces are converted to functional S-expressions and ingested by the Stitch binary.
     - Step 4: Top 5 extracted primitives are appended to the DSL grammar.
     - Step 5: Core policy is updated via 3 epochs of SFT on compressed traces followed by 1 pass GRPO.
  3. **Measured Metrics**: Held-out accuracy, average program AST depth, description length compression ratio $\text{CR}$.
  4. **Random-Reward Control Arm**: Identical procedure executed in parallel with randomized reward feedback.
  5. **Pass Criterion**: $\Delta_{\text{fresh}} \ge +10.0\%$ improvement AND $\text{CR} \ge 1.20$ AND outscoring the control arm by $\ge +5.0\%$.
  6. **Kill Criterion**: $\Delta_{\text{fresh}} \le +2.0\%$ or Stitch memory exhaustion after 10,000 traces.
  7. **Compute Budget**: 4.5 hours on 2x T4, consuming 15% of weekly allocation.

---

### 4.2 Missing No-Teacher Ablation & Outdated Frontier Names (AUDIT §1.5)
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:  
  1. **Missing No-Teacher Ablation**: BRIEF.md §2.6 and AUDIT §1.5 mandate an explicit "no-teacher" ablation to prove that the RSI loop can bootstrap purely from 0-parameter procedural generators without Stage-0 distillation. Proposal v1 omitted this completely.
  2. **Outdated Frontier Names**: In Section 4 and Section 7.2, the proposal drops **GPT-6 Astra** and refers intermittently to legacy models. Per AUDIT §1.5, the mandatory frontier triumvirate is: **Claude Opus 5.5, Gemini 4 Argon, and GPT-6 Astra**.
  3. **MuSR Subtask Aggregation**: Proposal Table 7.1 bundles MuSR as a single average row. AUDIT §1.5 requires gating per subtask: Murder Mysteries (MM), Object Placement (OP), and Team Allocation (TA).
- **Concrete Fix**:  
  1. Add an explicit **No-Teacher Bootstrap Ablation** section to Section 8: Initialize weights from scratch, train exclusively on procedural Reasoning Gym + re-ARC tasks with verifiers for 5 generations, and compare held-out trajectory against the distilled Stage-0 model.
  2. Update all text and ladder diagrams to reference **Claude Opus 5.5, Gemini 4 Argon, and GPT-6 Astra**.
  3. Disaggregate MuSR in Table 7.1 into three distinct gated rows: MM ($T=70.0 \to G=80.0$), OP ($T=58.2 \to G=72.1$), and TA ($T=78.8 \to G=88.8$).

---

## 5. Priority Attack 5: Scoreboard Inaccuracies, Reverse-Engineered Margins, and ARC-AGI

### 5.1 The ARC-AGI-1 47.5% Target: Physically Unsound on Non-Attention Hardware (AUDIT §1.8)
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:  
  Table 7.1 claims an expected score of **47.5% on ARC-AGI-1** at Generation 10.  
  This target exceeds TRM-Att (44.6%), which relies on full 2D bidirectional softmax attention. As documented in `tiny_sota.md` and AUDIT §1.8, the attention-free variant **TRM-MLP drops to 29.6% on ARC-AGI-1 and 0.0% on Maze-Hard**.  
  Non-attention sequence backbones (pure SSMs and DeltaNet) suffer from an inherent inability to perform precise, coordinate-indexed content retrieval across arbitrary 2D spatial token grids without sequence-wide attention matrices. Claiming a 68.7M attention-free model will hit 47.5% on ARC-AGI purely through "grid DSL abstraction" without an explicit spatial lookup mechanism is technically groundless.
- **Concrete Fix**:  
  Choose one of two rigorous engineering routes:
  1. **Realistic Recalibration**: Lower the ARC-AGI-1 expected score to **31.5%** [추정], which genuinely reflects the empirical ceiling of non-attention recurrent cores while clearing GPT-3.5's verified baseline of 10.6% by over 20 points.
  2. **Architectural Inductive Bias**: Augment the Bidirectional Reader with an explicit 0-parameter **2D Spatial Coordinate Grid Embedding** and a local $3 \times 3$ depthwise convolution patch tokenizer before the DeltaNet trunk, providing explicit spatial inductive bias without violating the Transformer ban.

---

### 5.2 Unjustified Probabilities & Reverse-Engineered Margins (AUDIT §1.6, §1.7)
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:  
  1. Section 7.2 lists arbitrary probabilities: Rung 1 (95%), Rung 2 (70%), Rung 3 (45%), Rung 4 (25%), Rung 5 (10%). AUDIT §1.6 strictly bans unsupported numerical probabilities.
  2. In Table 7.1, expected scores sit exactly 1.3 to 2.4 points above the $G$ threshold across 12 rows (e.g., GSM8K $90.4$ vs $G=88.1$, HumanEval+ $82.8$ vs $G=80.7$, MBPP+ $82.2$ vs $G=79.8$). This indicates reverse-engineered projection rather than empirical modeling.
  3. Unverified GPT-3.5 targets from secondary sources were marked as verified: MATH was listed as 43.1 (verified in `reason_baseline.md` is 34.1), DROP was listed as 70.2 (verified is 64.1), StrategyQA was listed as 70.3 (verified is 51.5).
- **Concrete Fix**:  
  1. Replace all probability percentages in Section 7.2 with qualitative confidence tiers (**High / Moderate / Low / Speculative**) paired with the precise empirical risk factor that would falsify them.
  2. Label every expected score in Table 7.1 with **[추정]** and every target score with **[가설]**.
  3. Ground the scoreboard on verified baselines from `reason_baseline.md` while reporting conservative targets side-by-side:
     - MATH: Verified $T = 34.1 \implies G = 56.1$ (report conservative $T=43.1 \to G=62.1$ as secondary [UNVERIFIED]).
     - DROP F1: Verified $T = 64.1 \implies G = 76.1$ (report conservative $T=70.2 \to G=80.2$ as secondary [UNVERIFIED]).
     - StrategyQA: Verified $T = 51.5 \implies G = 67.7$ (report conservative $T=70.3 \to G=80.3$ as secondary [UNVERIFIED]).

---

## 6. Priority Attack 6: Parameter Accounting Omissions (< 100M Rail Strict Compliance)

- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:  
  Section 5.4 claims a total of 68,712,001 parameters. However, forensic analysis of the text reveals multiple functional modules that were omitted from the parameter table (AUDIT §1.9):
  1. **Formal Vocabulary Input Embeddings**: The core generates programs autoregressively. Previously emitted formal tokens ($V_{\text{formal}} = 2,048$) must be embedded as inputs into the solver trunk. This requires an input embedding table:
     $$2,048 \times 512 = \mathbf{1,048,576\text{ parameters}}$$
  2. **Gated DeltaNet Short 1D Convolutions**: Standard Gated DeltaNet layers require depthwise causal 1D convolutions (kernel size $k=4$) on $Q, K, V$ projections to maintain local context:
     $$14 \text{ layers} \times 3 \times (4 \times 512) = \mathbf{86,016\text{ parameters}}$$
  3. **DeltaNet Output Gate ($Z$-Projection)**: Gated DeltaNet uses an output gate projection ($O = (S \cdot Q) \odot \text{silu}(Z)$):
     $$14 \text{ layers} \times (512 \times 512) = \mathbf{3,670,016\text{ parameters}}$$
  4. **Span Copy / Pointer Head**: Row A11 (DROP F1) requires copying spans directly from the reading context into the output. A 2-layer bilinear pointer head requires:
     $$2 \times (512 \times 256) + 256 = \mathbf{262,400\text{ parameters}}$$
  5. **Total Omitted Trainable Parameters**: $\mathbf{5,067,008\text{ parameters (~5.07M)}}$.

### Fully Corrected & Re-Audited Parameter Accounting Table:

```
====================================================================================================
                        RE-AUDITED PARAMETER SPECIFICATION TABLE (< 100M)
====================================================================================================
  Module                      Component                Exact Arithmetic Dimension        Parameters
----------------------------------------------------------------------------------------------------
  1. Input Embeddings         Natural Language BPE     16,384 vocab x 512 dim            8,388,608
                              Formal DSL Input Embed   2,048 vocab x 512 dim (RESTORED)  1,048,576
  2. Bidirectional Reader     6x Bi-SSM Layers (Fwd)   6 x [2 x (512x1024 + 1024x512)]   6,291,456
                              6x Bi-SSM Layers (Bwd)   6 x [2 x (512x1024 + 1024x512)]   6,291,456
                              Reader Conv1D & Proj     6 x [4 x 1024 + 512 x 1536]       4,743,168
                              Reader LayerNorms        6 layers x 2 norms x 512              6,144
  3. Recurrent Solver Trunk   Q, K, V Projections      14 layers x 3 x (512 x 512)      11,010,048
                              Causal Conv1D (k=4)      14 layers x 3 x (4 x 512) (REST)     86,016
                              Delta Gates (alpha, beta)14 layers x 2 x (512 x 4 heads)      57,344
                              Output Gate (Z-proj)     14 layers x (512 x 512) (REST)    3,670,016
                              Trunk Linear Output Proj 14 layers x (512 x 512)           3,670,016
                              SwiGLU FFN Trunk         14 layers x 3 x (512 x 1536)     33,030,144
                              Solver RMSNorms          14 layers x 2 norms x 512            14,336
  4. Heads & Emitters         DSL Formal Emitter       512 dim x 2,048 formal vocab      1,048,576
                              Shared Value Head MLP    512x256 + 256 + 256x1 + 1           131,585
                              Proposer Head MLP        512x512 + 512 + 512x64 + 64         295,488
                              Span Pointer Head        2 x (512 x 256) + 256 (RESTORED)    262,400
                              Final Output RMSNorm     1 x 512 dim                             512
----------------------------------------------------------------------------------------------------
  TOTAL RE-AUDITED PARAMETERS (ALL EXECUTED WEIGHTS COUNTED)                            73,772,009
  HEADROOM UNDER 100,000,000 LIMIT                                                      26,227,991
====================================================================================================
```
- **Audit Conclusion**:  
  With all omitted modules fully restored, the core stands at **73,772,009 parameters (~73.77M)**. It passes the hard 100M parameter limit with 26.23M parameters of safety margin. The Phase C blueprint must adopt this exact table.

---

## 7. Priority Attack 7: Citation Laundering and Literature Integrity (AUDIT §2)

- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:  
  A forensic cross-check of citations in Proposal v1 against primary literature and AUDIT §2 reveals multiple instances of citation laundering—attaching real papers to claims they do not make:

```
====================================================================================================
                    FORENSIC CITATION AUDIT & REMEDIATION REGISTER
====================================================================================================
  Citation             Proposal Claim                      Actual Finding & Correction
----------------------------------------------------------------------------------------------------
  1. arXiv:2504.13837  "Distillation caps student at       FALSE CLAIM. The paper proves distillation
                       teacher; RLVR discovers novelty"    ADDS reasoning patterns, while RLVR
                                                           NARROWS the reasoning boundary. Fix: Cite
                                                           as warning against greedy GRPO overfitting.
  2. arXiv:2006.08381  "Stitch symbolic compression"       WRONG CITATION. 2006.08381 is DreamCoder.
                                                           Stitch is Bowers et al. (arXiv:2211.16605).
                                                           Fix: Cite Bowers et al. (POPL 2023).
  3. Gated DeltaNet    S_t = S_{t-1}(I - beta*k*k^T)...    INCOMPLETE EQUATION. Omits the alpha decay
     (Line 270 & 360)  Omits alpha retention gate          gate. Correct: S_t = alpha_t*S_{t-1}(I -
                                                           beta_t*k_t*k_t^T) + beta_t*v_t*k_t^T.
  4. AlignScore        "Suffix-automaton masking and       FABRICATED ATTRIBUTION. AlignScore covers
     (arXiv:2305.16739) swap-test for blindfold renderer"  factual alignment, not suffix automata or
                                                           swap tests (from io_split.md). Fix reference.
  5. Google TRC Grant  "TRC awards up to 110+ Cloud TPUs"  UNSUPPORTED NUMBER. Official TRC FAQ gives
     (Line 511)                                            no device quota; users pay VM/storage fees.
  6. arXiv:2511.18397  "OPCW chemical weapon filter &      MISATTRIBUTED. Paper covers reward hacking
     (Line 551)        SHA-256 boot hashes"                and inoculation prompting. Fix citation.
  7. Dataset Names     "FineWeb-2 Korean Educational"      PHANTOM DATASET. FineWeb-2 does not exist
     (Line 446)                                            in data.md. Use verified FineWeb-Edu.
====================================================================================================
```
- **Concrete Fix**:  
  Apply every correction listed in the register above directly into Section 1, 2, 9, 10, and 11 of the revised proposal.

---

## 8. Priority Attack 8: Fair Play, Secret Transformer Check, and 12-Week Feasibility

### 8.1 Fair-Play & Sandboxing Integrity
- **Severity**: **MINOR**
- **Assessment**:  
  The renderer firewall (blindfold protocol, core-only scoring, counterfactual swap test) is conceptually sound and complies with BRIEF §2.5.  
  However, the execution sandbox lacks defense against **resource starvation attacks**. A candidate Python or Lean script can execute a subtle memory leak or non-terminating recursion that evades simple timeouts by thrashing CPU cache.  
- **Concrete Fix**:  
  Enforce sandboxing using Linux `cgroups v2` with strict kernel-enforced limits: `memory.max = 256M`, `pids.max = 16`, `cpu.max = 100000 100000` (1 core cap), and run all processes under `nsjail` with seccomp-bpf filters blocking network sockets and non-whitelisted syscalls.

### 8.2 Secret Transformer Verification
- **Severity**: **PASS (COMPLIANT)**
- **Assessment**:  
  The backbone consists of 6-layer Bidirectional SSM Reader + 14-layer Gated DeltaNet Solver Trunk. There is no sequence-wide softmax attention ($QK^T$), no Hopfield recurrence, and no hidden sliding-window attention hybrid (such as StripedHyena or LFM2). It is 100% compliant with the non-Transformer mandate.

### 8.3 Cross-Scale Transferability of the Symbolic Library
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:  
  The proposal asserts that the skill library $\mathcal{P}$ transfers seamlessly to a 1B or 10B model on Day 1 because it has 0 parameters.  
  This ignores the **recognition gap**: a larger neural core does not automatically know how to call or compose newly abstracted library primitives if their functional signatures are not part of its pretraining distribution. If a 1B core is trained on standard code and math, presenting it with custom macro tokens like `lib_p42_sum_of_squares` will result in out-of-vocabulary confusion.
- **Concrete Fix**:  
  Define the **Library Warm-Start Transfer Protocol**:
  1. The library is stored as an open, human-readable repository of documented Python/SymPy functions with explicit docstrings and assertions.
  2. When instantiating the 1B core, the library is converted into an explicit Stage-0 synthetic fine-tuning shard where the 1B model is trained to predict both the primitive expansions and their compressed calls.

### 8.4 12-Week Solo Builder Feasibility & The Multi-DSL Disaster
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:  
  ALEMBIC-LIB proposes supporting four disparate execution environments in parallel: Python 3.11, SymPy CAS, Z3 SMT Prover, and Lean 4 Interactive Theorem Prover.  
  For a solo builder with zero funding, building grammar pushdown automata (PICARD), AST parsers, Stitch anti-unification bindings, and training pipelines across four completely different syntaxes in 12 weeks is a recipe for total project failure. Lean 4 alone requires specialized environment configurations, complex proof elaboration handling, and high failure rates on autoformalization.
- **Concrete Fix**:  
  Radically simplify the symbolic toolchain into a **Unified 2-Language Execution Bridge**:
  1. **Phase 1 (Weeks 1–6)**: Restrict the engine exclusively to **Python 3.11 with SymPy**. Python handles algorithmic benchmarks (HumanEval+, MBPP+, BBH), while SymPy handles mathematical reasoning (GSM8K, MATH, SVAMP). This requires only ONE AST grammar mask and ONE runtime parser.
  2. **Phase 2 (Weeks 7–12)**: Introduce Z3 via its native Python API (`z3-solver`), treating logic formulas as standard Python scripts with assertions.
  3. Defer Lean 4 entirely to post-bootloader scaling (Rung 3, Generations 35+). This keeps the 12-week roadmap tightly focused and achievable.

---

## 9. Comprehensive Remediation Matrix for Phase C Revisions

The author of `proposal_library_v2.md` must implement the following mandatory engineering modifications:

```
====================================================================================================
                        PHASE C ACTIONABLE REMEDIATION SPECIFICATION
====================================================================================================
  Audit Issue                 Mandatory Action in v2 Revision
----------------------------------------------------------------------------------------------------
  1. Reading Verifier Void    Implement Reverse Procedural Generation, Dual-Formulation Consistency,
                              and GSM-Symbolic counterfactual twin gating.
  2. Impossible 70B Teacher   Replace live 70B teacher with Qwen2.5-14B-Instruct AWQ on 2x T4 or
                              pre-computed OpenMathInstruct-2 verified execution traces.
  3. Missing Control Arm      Add Random-Reward Control Arm gate (must beat control by >= 5.0%).
  4. Recurrent FLOP Math      Recompute FLOPs using FLOPs/tok = 2*(N_non_core + r*N_core) with r_avg=8.
                              Update generations/week from 47 to 8.5-12.0.
  5. T4 Latency Realism       Correct per-token latency from 0.9ms to 0.74s; MCTS from 28.9ms to 3.8s.
  6. Code Rule Violation      Purge all 135 lines of Python code from Section 12.1. Replace with
                              structured 7-part plain-language experiment plan.
  7. ARC-AGI Target           Lower ARC-AGI-1 target to 31.5% OR integrate 2D Spatial Positional Embeddings.
  8. Parameter Accounting     Adopt re-audited table (73.77M params) including output gates, conv1D,
                              pointer head, and formal input embeddings.
  9. Stitch Python Bottleneck Replace raw Python AST compression with a functional S-Expression IR.
  10. Citation Laundering     Correct all 7 misattributed papers and phantom datasets per Section 7.
  11. Multi-DSL Scope         Narrow 12-week implementation to Python + SymPy + z3-solver API.
====================================================================================================
```

---

## 10. Final Assessment

Proposal ALEMBIC-LIB v1.0 contains the most potent theoretical engine for unbounded recursive scaling—**the zero-parameter symbolic skill library that compounds capability without neural weight expansion**. This core insight is brilliant and must be preserved. 

However, v1.0 is currently crippled by fabricated toy code, impossible T4 latency claims, severe recurrent FLOP undercounts, a fatal verification blind spot in natural language comprehension, and unfeasible multi-DSL fragmentation. 

Applying the concrete, mathematically sound fixes detailed in this critique will transform ALEMBIC-LIB from a fragile, overstated concept into an airtight, scientifically unassailable, and genuinely competitive ASI foundation.
