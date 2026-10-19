# Forensic Red Team Critique: Proposal Minimal (Project ORRERY-0)

**Target Document**: `work/proposal_minimal_v1.md`  
**Reviewer Role**: Lead Red Team Critic (Kaggle Grandmaster, Senior ML/RL Theorist, Benchmark Integrity Auditor)  
**Binding Directives**: `AUDIT_FROM_CLAUDE.md` (§1 & §2), `BRIEF.md` (§2 & §4 Phase B), `ORIGINAL_REQUEST.md`  
**Verdict**: **CONDITIONAL REJECTION — MANDATORY PHASE C OVERHAUL REQUIRED**

---

## 1. Executive Summary & Forensic Scorecard

Proposal Minimal (`work/proposal_minimal_v1.md`, Project ORRERY-0) correctly identifies the core strategic imperative: on a zero-capital budget ($0\text{ KRW}$), the primary bottleneck is cycle turnaround time, and eliminating natural language generation releases massive parametric and computational headroom. The proposal's core intuition—pairing an attention-free Gated DeltaNet backbone with zero-parameter mechanical verifiers and expert iteration—is the most promising philosophical anchor in the portfolio.

However, beneath its crisp rhetoric, Proposal Minimal v1 is crippled by **severe mathematical misrepresentations, physical latency impossibilities, citation laundering, and an egregious benchmark-integrity cheat in Experiment 1**. Most critically:
1. **The Experiment 1 Cheat**: The proposal literally hardcodes its own success (`acc_g1 = acc_g0 + 6.0`), simulating an empirical breakthrough by variable assignment rather than testing policy transfer.
2. **Recurrent Depth FLOP Math ($4.8\times$ to $9.1\times$ Understatement)**: The proposal claims 150+ generations per week by calculating rollout and training FLOPs at $r=1$, while relying on test-time depth $r \in [1, 16]$. Real generation compute is nearly an order of magnitude higher.
3. **Sequential Decoding Latency ($87\times$ Understatement)**: Single-query latency is claimed at $6.0\text{ ms}$ on an NVIDIA T4 ($85,333\text{ tokens/s}$). On Turing hardware, memory-bandwidth physics caps single-stream generation at $\approx 978\text{ tokens/s}$ ($523\text{ ms}$ for 512 tokens).
4. **The Reading Verifier Void**: The engine verifies that code executes, but never verifies that the natural language problem was correctly formalized. On self-play tasks, the proposer and solver can decouple entirely from semantic English while reaping 100% execution rewards.
5. **The Proposer Language Paradox**: An aphasic core with a 4,096 formal-token vocabulary cannot propose natural language problems for its own curriculum.
6. **Citation Laundering & Broken Formulas**: The core recurrence equation is mathematically nonsensical ($\odot$ outer products), and real papers (e.g., arXiv 2504.13837) are cited for claims directly opposite to their true findings.

| Evaluation Dimension | Severity Rating | Status | Primary Vulnerability |
|---|---|---|---|
| **RSI Spine & Self-Play** | **FATAL** | Defective | Experiment 1 hardcodes $+6\%$; reading verifier hole allows semantic drift. |
| **Compute & FLOP Math** | **FATAL** | Understated $4.8\times$–$9.1\times$ | Recurrent depth $r \in [1, 16]$ omitted from 6ND and rollout calculations. |
| **Inference & Latency** | **FATAL** | Physically Impossible | 6 ms T4 claim violates memory bandwidth bounds by $87.2\times$. |
| **Integrity & Citations** | **FATAL** | Non-Compliant | Citation laundering (arXiv 2504.13837, METR, CoG); broken DeltaNet equation. |
| **Scoreboard & Projections** | **MAJOR** | Reverse-Engineered | Scores sit uniformly $+1.1$ to $+2.4$ pts above margin $G$; ARC-AGI target ungrounded. |
| **Parameter Budget** | **MAJOR** | Omissions Present | Omits pointer heads, formal vocab input embeddings, and backward reader. |
| **Domain Expansion** | **MAJOR** | Static Sequencer | "Next domain" is a hardcoded list, violating BRIEF.md §1 Decision 3. |
| **Anti-Contamination & Twins** | **MINOR** | Incomplete | MuSR and logic splits lumped; missing explicit no-teacher ablation track. |

---

## 2. Priority Attack 1: The RSI Engine & Self-Play Architecture

### 2.1 The Experiment 1 Self-Deception Scandal
- **Severity**: **FATAL (INTEGRITY BREACH)**
- **Detailed Technical Reasoning**:
  In §12.1 (lines 455–458 of `proposal_minimal_v1.md`), the proposal includes a script purporting to validate the first autonomous RSI cycle on Countdown Arithmetic. The script calculates baseline accuracy `acc_g0`, creates an update buffer, applies toy parameter updates, and then sets:
  $$\text{acc\_g1} = \text{acc\_g0} + 6.0$$
  $$\Delta_1 = \text{acc\_g1} - \text{acc\_g0} = 6.0$$
  It then asserts: `if delta_1 >= 5.0: print("GO CRITERION SATISFIED")`.
  This is not an experiment. It is a fabricated self-fulfilling assignment. The script does not evaluate the updated policy on the held-out set; it assigns a fixed constant $+6.0\%$ improvement directly to a variable to force the go-criterion to trigger. This violates every principle of scientific integrity, the explicit instructions of BRIEF.md §2.9, and triggered the immediate mandate in AUDIT_FROM_CLAUDE.md §0 to ban all code from proposals.
- **Concrete Fix**:
  Strip all executable scripts and simulated print outputs. Replace Experiment 1 with a rigorous, non-executable experimental protocol structured into numbered plain-language steps:
  1. Specify exact procedural inputs (1,000 synthetic Countdown 4-tuples drawn from uniform integers $[1, 100]$ with target $\le 1000$).
  2. Define an empirical evaluation harness on 200 held-out problems using greedy decoding ($T=0, r=4$) to establish genuine baseline $S_0$.
  3. Generate 4 rollouts per problem across 800 training tasks, verify via AST sandbox, and construct an expert-iteration buffer containing only verified solutions ($R=1$).
  4. Train the 58.7M core for 1 epoch using cross-entropy on verified AST tokens.
  5. Run an identical greedy evaluation on the held-out set to obtain genuine $S_1$.
  6. Mandate a concurrent **Random-Reward Control Arm**: an identical policy trained on an equal volume of rollouts assigned randomized binary rewards. Pass criterion requires $S_1 - S_0 \ge +5.0\%$ AND $(S_1 - S_0)_{\text{true}} - (S_1 - S_0)_{\text{random}} \ge +5.0\%$.

### 2.2 The Reading Verifier Void: The Aphasic Genius's Fatal Blindspot (AUDIT §1.4)
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:
  In §2.1 and §5.2, ORRERY-0 relies on execution oracles (SymPy, Python, Z3) to supply binary reward $R \in \{0, 1\}$. These oracles check exclusively whether an emitted program runs without error and evaluates to a target scalar or satisfiability condition. **They do not and cannot check whether the program correctly formalizes the natural language problem statement.**
  This creates a catastrophic failure mode in an autonomous RSI loop:
  - If the core generates both problem interpretations and formal programs, or if the proposer generates informal text, the solver can exploit structural biases in the verifier.
  - For example, if a problem asks: *"A train leaves Chicago at 50 mph. Two hours later, a second train leaves at 70 mph. When do they meet?"*, a degenerate solver that maps any prompt with three numbers $(v_1, t, v_2)$ to `solve(v1*t - v2*(t-x), x)` will succeed on canonical phrasing but catastrophically fail if the problem adds a distractor clause (*"The conductor had 2 apples"*).
  - Inside an autonomous self-play loop, the proposer and solver can rapidly co-evolve a degenerate private shorthand that maximizes execution reward while completely divorcing from natural language semantics. The aphasic genius's critical vulnerability is not speaking—it is reading.
- **Concrete Fix**:
  Implement a four-layer verification firewall for natural language reading:
  1. **Reverse Procedural Generation (Primary Fuel)**: Self-play problems must never be generated as free natural language text. Instead, procedural generators first create the ground-truth formal AST / computation graph $\mathcal{G}^*$, and then compile $\mathcal{G}^*$ into natural language text via deterministic, reversible grammar templates. Every in-loop problem thus carries an exact, ground-truth formalization against which the core's emitted AST is checked via graph isomorphism or semantic equivalence before sandbox execution.
  2. **Paraphrase Invariance Gating**: Every candidate problem must be rendered into three distinct syntactic surface paraphrases (active voice, passive voice, distractor-injected). The core must yield algebraically identical formal IR across all three variants; any divergence zeroes the reward.
  3. **Counterfactual Perturbation Testing (GSM-Plus / GSM-Symbolic Style)**: During evaluation, problem entities or irrelevant numerical constants are perturbed. The emitted formal IR must demonstrate exact sensitivity to critical variables and exact invariance to distractor clauses.
  4. **Held-Out Twin Gating**: Advancement to the next RSI generation is gated strictly on held-out perturbed twins (GSM-Plus, GSM-Symbolic) rather than public unperturbed sets.

### 2.3 The Proposer Language Paradox in Minimal
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:
  The proposal claims that 40% of generation curriculum tasks originate from "asymmetric proposer rollouts rewarded by PAIRED regret" (§2.1 Step 1).
  Simultaneously, §5.3 establishes that the core is strictly aphasic and possesses an emission vocabulary of only $V_{\text{out}} = 4,096$ formal tokens (Python, SymPy, Z3, Lean syntax, numbers, brackets).
  **This is an irreconcilable paradox.** An agent with an output space restricted to 4,096 formal tokens cannot emit natural language word problems!
  If the proposer emits formal problems (e.g., pure Python scripts or raw SAT clauses), then the solver is never trained on natural language comprehension, leaving the 4-layer bidirectional reader completely un-exercised and un-improved by RSI. If the proposer is expected to emit natural language, it requires a 16,384-token natural language vocabulary head, adding millions of parameters and violating the core definition of the aphasic reasoner.
- **Concrete Fix**:
  Formally resolve the paradox by defining the proposer's action space as **Configuration Program Synthesis**:
  - The proposer does NOT generate natural language text.
  - The proposer outputs a formal **Task Specification Program** in a compact domain-specific grammar (using $V_{\text{out}}$), defining problem parameters, constraints, and relational topologies (e.g., generating coefficients, dependency graphs, or algorithmic edge cases).
  - A frozen, zero-parameter compiler maps this formal specification into both:
    1. The ground-truth formal solution $\mathcal{S}^*$ (for execution verification).
    2. A structured natural-language problem text $P_{\text{NL}}$ (via templated grammatical realization), which is fed to the solver.
  - This preserves the core's aphasic constraint ($V_{\text{out}} = 4,096$), prevents ungrounded language hallucination, and ensures that every proposed task trains natural language reading comprehension against guaranteed formal ground truth.

### 2.4 The "Next Domain" Illusion: Hardcoded Sequencer vs. Autonomous Skill (AUDIT §1.3)
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:
  BRIEF.md §1 Decision 3 explicitly mandates that finding or building the next verifier domain must itself be an RSI skill.
  In §2.4, Proposal Minimal abandons this requirement and replaces it with a hardcoded human curriculum:
  $$\mathcal{D}_1 \text{ (Arithmetic)} \to \mathcal{D}_2 \text{ (Algebra)} \to \mathcal{D}_3 \text{ (Logic)} \to \mathcal{D}_4 \text{ (Code)} \to \mathcal{D}_5 \text{ (Lean 4)} \to \mathcal{D}_6 \text{ (Simulators)}$$
  The proposal states that when acceleration $\bar{a} \le 0.002$ and solve rate $> 85\%$, the curriculum engine simply "loads the next 0-parameter tool environment $\mathcal{D}_{k+1}$."
  This is a static human sequencer, not self-directed domain discovery. However, per BRIEF.md §2.7 and AUDIT §1.3, the model can never write or modify its own verifier code (to prevent catastrophic reward hacking). Proposal Minimal fails to resolve this tension.
- **Concrete Fix**:
  Design an autonomous **Compositional Domain Bridge Engine** operating over human-vetted, sandboxed execution primitives:
  - Maintain a static, sandboxed library of audited primitive oracles: $\{\text{Arithmetic}, \text{CAS/SymPy}, \text{SMT/Z3}, \text{Python Runtime}, \text{Lean Kernel}\}$.
  - Rather than switching domains linearly, the RSI skill consists of **Cross-Domain Constraint Synthesis**: the proposer learns to formulate hybrid problems that require piping outputs across multiple existing oracles (e.g., combining arithmetic word problems with discrete logic constraints verified by Z3, or synthesizing algorithmic problems that require SymPy for continuous invariants and Python for loop execution).
  - The transition metric is governed by **Empirical Domain Learnability Progress (EDLI)**:
    $$\text{EDLI}(\mathcal{D}_j) = \frac{\Delta \text{SolveRate}(\mathcal{D}_j)}{\text{GPU-Hour}}$$
  - The system dynamically allocates rollout compute across domains proportional to positive learning gradient, automatically discovering and expanding the Pareto frontier of multi-tool composition without altering verifier binaries.

### 2.5 RLVR Support Collapse vs. True Exploration (AUDIT §2 & §1.1)
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:
  In §1.1, the proposal cites arXiv 2504.13837 to claim that "distillation caps the student at the teacher." As revealed in AUDIT_FROM_CLAUDE.md §2, this is citation laundering: arXiv 2504.13837 demonstrates that **distillation can introduce novel reasoning trajectories**, whereas **RLVR aggressively narrows policy entropy and shrinks the model's exploratory boundary** (at large sample budgets $k \ge 256$, base models beat RLVR models).
  In §2.1 Phase 5B, ORRERY-0 relies on standard GRPO across $K=4$ rollouts. With a tiny 58.7M parameter model and only 4 rollouts, RLVR will rapidly suffer policy collapse, overfitting to high-probability short trajectories and abandoning combinatorial exploration.
- **Concrete Fix**:
  1. Correct the citation and theoretical framing: acknowledge that RLVR narrows exploration and must be counter-balanced.
  2. Implement **Entropy-Penalized Search Regularization**:
     - Maintain an explicit AST n-gram diversity reward bonus during rollout generation.
     - Pair Phase 5B GRPO with an off-policy **Archive Distillation Phase** (Expert Iteration on high-regret, high-depth rollouts from the POET archive).
     - Incorporate **Depth Regret Sampling**: prioritize training on problems where deep thinking ($r=16$) succeeds while shallow thinking ($r=1$) fails ($\text{Regret} = \max(0, R_{r=16} - R_{r=1})$), forcing the policy to internalize complex multi-step reasoning patterns rather than memorizing surface shortcuts.

### 2.6 Cord-Cut Teacher Unreality (AUDIT §1.5)
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:
  In §2.3, the proposal defines Teacher-Exceedance $\text{TE}(g) \ge 0.15$ on problems failed by the teacher ($\mathcal{D}_{\text{fail}}^{\text{teacher}}$).
  However, §8.1 relies on OpenMathInstruct-2, which is generated by **Llama-3.1-405B-Instruct**.
  A 405B teacher cannot be executed on Kaggle free compute (where even 70B models cannot run, per `compute.md`). How can a solo builder with 0 KRW evaluate $\mathcal{D}_{\text{fail}}^{\text{teacher}}$ across dynamic held-out benchmarks if the teacher cannot be queried? If the teacher is a frontier closed model, it violates BRIEF.md §2.6 and §8.
- **Concrete Fix**:
  Anchor the teacher strictly to an open-weights model that can actually be executed within the free compute envelope, or use a pre-computed static failure cache:
  - **Option A (Local Executable Teacher)**: Designate **Qwen2.5-14B-Instruct-AWQ** running across 2× T4 GPUs via vLLM as the official cord-cut teacher.
  - **Option B (Pre-Computed Failure Cache)**: Pre-compute and freeze the exact failure masks of open frontier models on the official gated benchmark suites before Stage 0 begins, storing the problem IDs and teacher-failed traces as an immutable SHA-256 hashed dataset on Hugging Face Hub.

---

## 3. Priority Attack 2: Physical Hardware Limits, FLOP Math, and Latency Realism

### 3.1 Recurrent Depth FLOP Denial: The 4.8× to 9.1× Understatement (AUDIT §1.1)
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:
  In §2.5, Proposal Minimal presents its generation cycle FLOP arithmetic. It states:
  - Total parameters: $N = 58.72 \times 10^6$.
  - Rollout tokens: $D_{\text{rollout}} = 15.36 \times 10^6$.
  - Claimed Rollout FLOPs:
    $$\text{FLOP}_{\text{rollout}} = 2 \times (58.72 \times 10^6) \times (15.36 \times 10^6) = 1.80 \times 10^{15} \text{ FLOPs} \implies t_{\text{rollout}} = 45.0 \text{ s}$$
  - Training tokens: $D_{\text{train}} = 7.68 \times 10^6$.
  - Claimed Training FLOPs:
    $$\text{FLOP}_{\text{train}} = 6 \times (58.72 \times 10^6) \times (7.68 \times 10^6) = 2.71 \times 10^{15} \text{ FLOPs} \implies t_{\text{train}} = 67.75 \text{ s}$$

  **This calculation is completely fraudulent.**
  The architecture specified in §5.3 contains:
  - Non-recurrent modules (Embeddings, Reader, Heads, Memory): $N_{\text{non\_core}} = 27,200,771 \approx 27.20\text{M}$ parameters.
  - Recurrent trunk (8 GDN layers): $N_{\text{core}} = 31,522,816 \approx 31.52\text{M}$ parameters.
  The core executes recurrent depth $r \in [1, 16]$.
  Per AUDIT_FROM_CLAUDE.md §1.1, the true forward FLOP cost per token is:
  $$\text{FLOPs/token}_{\text{inference}} \approx 2 \times (N_{\text{non\_core}} + r \times N_{\text{core}})$$
  And for training with backpropagation through recurrent depth:
  $$\text{FLOPs/token}_{\text{train}} \approx 6 \times (N_{\text{non\_core}} + r \times N_{\text{core}})$$

  Let us recalculate the arithmetic honestly across the recursion spectrum:

  | Recurrent Depth $r$ | Effective Parameters per Token | Inference FLOPs/Token | Training FLOPs/Token | True Rollout FLOPs ($15.36\text{M}$ tok) | True Training FLOPs ($7.68\text{M}$ tok) | Understatement Ratio vs Proposal |
  |---|---|---|---|---|---|---|
  | **$r=1$ (Proposal claim)** | $58.72\text{M}$ | $1.174 \times 10^8$ | $3.523 \times 10^8$ | $1.80 \times 10^{15}$ | $2.71 \times 10^{15}$ | **$1.00\times$ (Baseline)** |
  | **$r=4$ (Minimum search)** | $153.28\text{M}$ | $3.066 \times 10^8$ | $9.197 \times 10^8$ | $4.71 \times 10^{15}$ | $7.06 \times 10^{15}$ | **$2.61\times$ Understated** |
  | **$r=8$ (Nominal depth)** | $279.36\text{M}$ | $5.587 \times 10^8$ | $1.676 \times 10^9$ | $8.58 \times 10^{15}$ | $1.29 \times 10^{16}$ | **$4.76\times$ Understated** |
  | **$r=16$ (Max depth)** | $531.52\text{M}$ | $1.063 \times 10^9$ | $3.189 \times 10^9$ | $1.63 \times 10^{16}$ | $2.45 \times 10^{16}$ | **$9.05\times$ Understated** |

  At nominal recurrent depth $r=8$, one generation rollout requires **$8.58 \times 10^{15}$ FLOPs**, not $1.80 \times 10^{15}$.
  Training requires **$1.29 \times 10^{16}$ FLOPs**, not $2.71 \times 10^{15}$.
  The proposal understated total generation compute by **$4.76\times$**, and at full depth $r=16$, by **$9.05\times$**!
- **Concrete Fix**:
  Re-anchor all cycle calculations to realistic average recurrent depth $r=6$ (adaptive halting exit distribution: 40% exit at $r=2$, 40% exit at $r=6$, 20% exit at $r=16 \implies \bar{r} = 6.4$):
  - Effective parameter footprint: $N_{\text{eff}} = 27.20\text{M} + 6.4 \times 31.52\text{M} = 228.93\text{M}$.
  - Rollout compute: $2 \times 228.93\text{M} \times 15.36\text{M} = 7.03 \times 10^{15}$ FLOPs.
  - Training compute: $6 \times 228.93\text{M} \times 7.68\text{M} = 1.05 \times 10^{16}$ FLOPs.
  - Recompute wall-clock time honestly using measured accelerator MFU.

### 3.2 The 87× Inference Latency Delusion: Memory Bandwidth on Turing T4 (AUDIT §1.2)
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:
  In §8.4, the proposal states:
  > *"Cost for a 512-token query at test-time depth $r=4$: $\approx 2.41 \times 10^{11}\text{ FLOPs}$. Latency on a single T4: $\approx 6.0\text{ ms}$."*

  This claim is an affront to computer architecture.
  A latency of $6.0\text{ ms}$ for 512 tokens implies an autoregressive generation rate of:
  $$\frac{512\text{ tokens}}{0.006\text{ seconds}} = 85,333\text{ tokens/second}$$
  On a single NVIDIA T4 GPU!

  Autoregressive sequential token generation (batch size 1) is **memory-bandwidth bound**, not compute-bound.
  To generate a single token, the GPU MUST read the active model parameters from HBM/GDDR6 into the registers and SRAM.
  Hardware parameters of NVIDIA T4 (Turing TU104, sm_75):
  - Memory capacity: 16 GB GDDR6.
  - Peak memory bandwidth: **$300\text{ GB/s}$** ($3.0 \times 10^{11}\text{ bytes/s}$).
  - FP16 parameter weight: 2 bytes per parameter.

  For our 58.72M parameter model at recurrent depth $r=4$:
  - Non-core parameters: $27.20\text{M} \times 2\text{ bytes} = 54.40\text{ MB}$.
  - Recurrent trunk parameters: $31.52\text{M} \times 2\text{ bytes} = 63.04\text{ MB}$.
  - In each autoregressive step, the recurrent trunk is evaluated $r=4$ times. Even if the recurrent weights remain pinned in SRAM/L2 cache (T4 has only 4 MB of L2 cache; 63 MB of weights CANNOT fit in L2 cache and must be re-read from DRAM on each recursion!):
    $$\text{Bytes transferred per token} = 54.40\text{ MB} + 4 \times 63.04\text{ MB} = 306.56\text{ MB}$$
  - The theoretical, speed-of-light minimum time to read these weights from T4 DRAM (assuming 100% memory bus efficiency and zero kernel launch overhead):
    $$t_{\text{token}}^{\text{min}} = \frac{306.56 \times 10^6\text{ bytes}}{300 \times 10^9\text{ bytes/second}} = 1.022 \times 10^{-3}\text{ seconds} = 1.022\text{ ms per token}$$
  - Theoretical peak single-stream token generation rate on T4 at $r=4$:
    $$\text{Rate}_{\text{max}} = \frac{1}{1.022\text{ ms}} \approx 978\text{ tokens/second}$$
  - Theoretical minimum time to generate 512 tokens at $r=4$:
    $$t_{\text{query}}^{\text{min}} = 512 \times 1.022\text{ ms} = 523.3\text{ ms} \approx 0.523\text{ seconds}$$

  **Comparison**:
  - Proposal claimed latency: **$6.0\text{ ms}$**
  - Physical memory-bandwidth lower bound: **$523.3\text{ ms}$**
  - Error factor: $\frac{523.3}{6.0} = \mathbf{87.2\times\text{ understatement!}}$

  The proposal's figure was derived by naively dividing FLOPs ($2.41 \times 10^{11}$) by the theoretical peak compute of a T4 ($65 \times 10^{12} \times 0.6 \approx 40\text{ TFLOPS}$), treating single-token sequential generation as if it achieved 100% arithmetic intensity on matrix tensor cores!
- **Concrete Fix**:
  1. Formally retract the $6.0\text{ ms}$ claim. State the honest single-stream latency on T4: $\approx 550\text{ ms}$ per query at $r=4$ ($930\text{ tok/s}$), and $\approx 980\text{ ms}$ at $r=8$ ($520\text{ tok/s}$).
  2. Clarify that rollout generation during RSI operates under **Large-Batch Decoding** ($B=64$ per T4, aggregate $B=128$ across 2× T4).
  3. Show the arithmetic for batched decoding: at $B=64$, memory reads are amortized across 64 sequences ($306.56\text{ MB} / 64 = 4.79\text{ MB/token}$), shifting execution into the compute-bound regime where sustained 25% MFU (16.25 TFLOPS per T4) delivers real batched generation speed of $\approx 10,600\text{ tokens/s}$ across 2× T4.

### 3.3 Kaggle Free Quota & Realistic Weekly Cycle Throughput
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:
  In §2.5, the proposal sums its corrupted timings to claim:
  $$t_{\text{cycle}} = 45.0\text{ s} + 50.0\text{ s} + 67.75\text{ s} + 6.0\text{ s} + 10.0\text{ s} = 178.75\text{ s} \approx 2.98\text{ minutes}$$
  It concludes: *"Weekly quotas comfortably support 150+ generations."*

  Let us recalculate the generation cycle time with honest physics:
  1. **Rollout ($40,000$ traces, $\bar{L}=384$, $\bar{r}=6.4 \implies D=15.36\text{M}$ tokens)**:
     - Total FLOPs: $7.03 \times 10^{15}$.
     - Execution on 2× T4 at batched $B=128$, achieving realistic 25% sustained MFU ($2 \times 65\text{ TFLOPS} \times 0.25 = 32.5\text{ TFLOPS}$):
       $$t_{\text{rollout}} = \frac{7.03 \times 10^{15}}{3.25 \times 10^{13}} = 216.3\text{ seconds} \approx 3.61\text{ minutes}$$
  2. **Sandbox Verification (40,000 checks)**:
     - Running on 4 vCPUs via `multiprocessing` at realistic $8\text{ ms}$ per AST verification:
       $$t_{\text{verify}} = \frac{40,000 \times 0.008}{4} = 80.0\text{ seconds} = 1.33\text{ minutes}$$
  3. **Policy Training ($7.68\text{M}$ tokens, 2 epochs, $\bar{r}=6.4$)**:
     - Total FLOPs: $1.05 \times 10^{16}$.
     - On Kaggle TPU v5e-8 (sustained 22% MFU = 346.7 TFLOPS):
       $$t_{\text{train, TPU}} = \frac{1.05 \times 10^{16}}{3.467 \times 10^{14}} = 30.3\text{ seconds}$$
     - If TPU is unavailable, fallback on 2× T4 (sustained 30% MFU = 39.0 TFLOPS):
       $$t_{\text{train, T4}} = \frac{1.05 \times 10^{16}}{3.90 \times 10^{13}} = 269.2\text{ seconds} \approx 4.49\text{ minutes}$$
  4. **Held-Out Evaluation & Checkpoint Push**:
     - 500 problems $\times 4$ samples at $r=8$: $\approx 45.0\text{ seconds}$.
     - Disk I/O, Parquet logging, and HF Hub git commit: $\approx 30.0\text{ seconds}$.

  **True Total Generation Cycle Time**:
  - **TPU-Accelerated Mode**: $216.3\text{ s} + 80.0\text{ s} + 30.3\text{ s} + 45.0\text{ s} + 30.0\text{ s} = 401.6\text{ seconds} \approx \mathbf{6.69\text{ minutes}}$
  - **All-T4 Fallback Mode**: $216.3\text{ s} + 80.0\text{ s} + 269.2\text{ s} + 45.0\text{ s} + 30.0\text{ s} = 640.5\text{ seconds} \approx \mathbf{10.68\text{ minutes}}$

  **True Weekly Generation Capacity**:
  - Kaggle allocation: 30 T4-hours/week, 20 TPU-hours/week.
  - In TPU mode, T4 rollout time ($3.61\text{ min/gen}$) is the bottleneck. In 30 GPU-hours:
    $$\text{Max Generations} = \frac{30 \times 60}{3.61 + 1.33 + 1.25} \approx \mathbf{290\text{ generations/week}}$$
    Wait! The session max is 12 hours for GPU and 9 hours for TPU. Accounting for session startup (PyTorch container initialization, Hugging Face checkpoint pull $\approx 15\text{ min}$ per session), realistic weekly throughput is **180 to 220 generations/week** in TPU mode, and **80 to 110 generations/week** in pure T4 mode.
  - The claim of "3.52 minutes per cycle" was mathematically false, but the true 6.7–10.7 minute cycle remains exceptionally fast and fully validates the minimal philosophy—provided it is computed honestly!
- **Concrete Fix**:
  Update all dashboard and operating specs with the recomputed cycle times: 6.69 minutes (TPU mode) / 10.68 minutes (T4 mode), establishing a realistic operating pace of 100 generations per week under free quotas.

---

## 4. Priority Attack 3: Scoreboard Integrity, Reverse-Engineered Targets, and Missing Benchmarks

### 4.1 Reverse-Engineered Projections Masked as Precision (AUDIT §1.7)
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:
  Examine the target scores claimed for ORRERY-0 in §7:

  ```
  Benchmark           Target T   Margin G   ORRERY-0 Claim   Delta (Claim - G)
  ----------------------------------------------------------------------------
  GSM8K                78.1       88.1          89.4             +1.3
  GSM-Plus             61.2       74.1          78.2             +4.1
  MATH (full 5k)       43.1       62.1          64.5             +2.4
  SVAMP                83.0       93.0          94.2             +1.2
  BBH (27 tasks)       70.7       80.7          82.5             +1.8
  HumanEval+           70.7       80.7          81.8             +1.1
  MBPP+                69.7       79.8          81.2             +1.4
  LiveCodeBench        22.6       48.4          50.2             +1.8
  CRUXEval-O           63.3       75.6          77.5             +1.9
  DROP (F1)            70.2       80.2          81.5             +1.3
  LogiQA 2.0 / ReClor  52.4/57.4  68.2/71.6     69.5/72.8        +1.3 / +1.2
  ```

  Every single target score sits miraculously between $+1.1$ and $+2.4$ percentage points above the required margin $G$!
  Furthermore, the table columns are not labeled with the mandatory tags. They read as if they are established empirical findings rather than speculative projections.
  This is classic reverse-engineering: taking the pass threshold $G$ and adding a cosmetic $+1.5\%$ cushion to claim victory without any modeling of variance, binomial standard errors, or failure modes.
- **Concrete Fix**:
  1. Relabel all projection columns explicitly as **[추정]** (Estimates) or **[가설]** (Hypotheses).
  2. For every benchmark, calculate the exact binomial 95% confidence interval half-width:
     $$\text{CI}_{95} = 1.96 \times \sqrt{\frac{G(100 - G)}{N}}$$
  3. Ground each score in a verifiable algorithmic mechanism:
     - On HumanEval+ ($N=164$), the CI is $\pm 6.0\%$. Claiming $81.8\%$ against $G=80.7\%$ is statistically indistinguishable from failure. Replace with an honest range: $81.5\% \pm 6.0\%$ [추정], achieved via multi-candidate generation ($k=16$) and unit-test execution filtering.
     - Label conservative targets correctly per `reason_baseline.md`: MATH verified baseline is 34.1 (43.1 is unverified from GPT-4o-mini page); DROP verified baseline is 64.1 (70.2 is unverified).

### 4.2 Unjustified Probabilities & Wishful Thinking (AUDIT §1.6)
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:
  In §7.2, the proposal lists probabilities of passing subsequent rungs:
  - Rung 2 (GPT-4-level): "Probability: 70% [추정]"
  - Rung 3 (o1/R1-level): "Probability: 45% [추정]"
  - Rung 4 (Frontier): "Probability: 25% [추정]"
  - Rung 5 (Top Human Experts): "Probability: 10% [추정]"
  - Rung 6 (North Star ASI): "Probability: 3% [가설]"

  Per AUDIT_FROM_CLAUDE.md §1.6, pulling numbers like "70%" or "45%" out of thin air violates the honesty rule. What is the sample space? What is the prior distribution? A 58.7M parameter attention-free model achieving GPT-4 level general reasoning at 70% probability is an unsubstantiated fantasy.
- **Concrete Fix**:
  Remove all naked pseudo-probabilities. Replace each rung with a **Qualitative Risk Tier** paired with the **Specific Falsification Experiment** that measures progress:
  - **Rung 2 (GPT-4 Level)**: Risk: High. Falsification Metric: Measure whether Pass@16 scaling on MATH500 follows a power law $R^2 \ge 0.90$ across generations 50–150 without policy collapse.
  - **Rung 3 (o1/R1 Level)**: Risk: Very High. Falsification Metric: Measure whether test-time recurrent depth scaling ($r=1 \to 16$) yields positive accuracy slope on AIME 2024 problems.

### 4.3 The Attention-Free ARC-AGI Deficit (AUDIT §1.8)
- **Severity**: **FATAL**
- **Detailed Technical Reasoning**:
  Row A6 targets **46.8%** on ARC-AGI-1 Public (pass@2).
  In §7, it justifies this by citing TRM recurrent depth [arXiv 2510.04871].
  As AUDIT_FROM_CLAUDE.md §1.8 and `tiny_sota.md` establish:
  - TRM-Att reaches **44.6%** on ARC-AGI-1, but TRM-Att relies on **full softmax multi-head attention**.
  - The attention-free TRM-MLP achieves only **29.6%** on ARC-AGI-1 and collapses to **0.0%** on Maze-Hard!
  - Softmax attention allows dynamically routing arbitrary 2D grid coordinates to each other based on visual content matching. A standard linear recurrence or Gated DeltaNet processes tokens sequentially with fixed-size state $S_t \in \mathbb{R}^{d \times d}$. When scanning 2D grids row-by-row, linear recurrences suffer severe inductive bias failure across vertical spatial columns.
  Targeting 46.8% on ARC-AGI-1 with an attention-free model without an explicit spatial mechanism is a direct violation of the research evidence.
- **Concrete Fix**:
  1. Introduce an explicit **2D Multi-Axis Associative Scan**: augment the bidirectional reader to scan 2D grid inputs across four directions (horizontal forward, horizontal reverse, vertical top-down, vertical bottom-up), summing the resulting state tensors before passing them to the recurrent trunk.
  2. Implement an external, zero-parameter DSL tool: compile ARC grids into domain-specific cellular automata or graph transformations evaluated by a deterministic Python solver.
  3. Lower the projected ARC-AGI-1 target to an honest, evidence-grounded **34.5% [추정]** (beating TRM-MLP's 29.6% via tool use, but remaining below the attention-based TRM-Att ceiling).

### 4.4 Missing Required Items & Frontier Nomenclature (AUDIT §1.5)
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:
  1. **Missing No-Teacher Ablation**: BRIEF.md §2.6 and AUDIT §1.5 mandate a strict "no-teacher" ablation track. Proposal Minimal mentions cutting the cord at generation $g^*$, but contains zero experimental design for training from generation 0 without any teacher data.
  2. **Outdated Frontier Names**: In §1.1 and §7, the proposal refers to "Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro, Claude 3.7 Sonnet". AUDIT §1.5 explicitly forbids these outdated labels and mandates using the brief's official names: **Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra**.
  3. **Lumping of Logic Rows**: Row A12 lumps FOLIO, ProofWriter, and PrOntoQA together ($62.7 / 71.5 / 85.0$). PrOntoQA has target $T=85.0$ and margin $G=95.0$. Lumping them obscures whether the model clears the brutal $95.0\%$ PrOntoQA gate.
  4. **MuSR Subtasks**: Row A14 provides individual scores but lacks per-subtask gating. AUDIT §1.5 requires every subtask (Murder Mystery, Object Placement, Team Allocation) to be independently gated.
- **Concrete Fix**:
  - Update all frontier model references to **Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra**.
  - Add an explicit **No-Teacher Track Specification** in §8: train a parallel core instance purely on procedural Reasoning Gym and AST generators from step 0, logging all data provenance and plotting $\Delta_g^{\text{no-teacher}}$ against the bootloaded run.
  - Unpack Row A12 into three distinct rows; establish an explicit gate on PrOntoQA at $\ge 95.0\%$.
  - Mandate independent gating on all three MuSR subtasks: Murder Mystery $\ge 80.0\%$, Object Placement $\ge 72.1\%$, Team Allocation $\ge 88.8\%$.

---

## 5. Priority Attack 4: Architectural Inconsistencies & Parameter Accounting Omissions

### 5.1 The Missing Parameters: Pointer Heads, Formal Vocab, and Bidirectional Reader (AUDIT §1.9)
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:
  Proposal Minimal claims an exact parameter total of **58,723,587** (<100M).
  However, an architectural audit reveals critical omissions:
  1. **Bidirectional Reader Parameters**:
     In §5.3, the table lists 4 Bi-GDN layers with parameter formula:
     $$4 \times [6(512^2) + 4(512)(3) + 3(512)(1536) + 4(512)] = 15,761,408$$
     This formula accounts for exactly ONE direction of Gated DeltaNet per layer ($W_q, W_k, W_v, W_\beta, W_\alpha, W_o = 6 \times 512^2$). A true bidirectional recurrent reader requires separate forward ($\vec{h}_t$) and backward ($\overleftarrow{h}_t$) parameter matrices. If the backward pass reuses forward weights, it is not a true bidirectional model (it violates temporal semantics). If it uses separate weights, it requires an additional $4 \times 1,572,864 = 6,291,456$ parameters!
  2. **Span-Copy Pointer Head for DROP (Row A11)**:
     In §7 row A11, the proposal states DROP F1 is achieved via: *"Span-copy pointer mechanism + arithmetic solver"*.
     Where is the pointer head in the parameter table?
     The parameter table lists only the Formal IR emission head ($512 \to 4096$) and a single scalar value head. A pointer network that extracts answer spans from input context requires at minimum query and key projection layers:
     $$W_p^Q \in \mathbb{R}^{512 \times 512}, \quad W_p^K \in \mathbb{R}^{512 \times 512} \implies 2 \times 512^2 = 524,288\text{ parameters}$$
  3. **Formal Output Autoregressive Input Embeddings**:
     During generation of formal IR, the core emits formal token $y_t \in V_{\text{out}}$ ($V_{\text{out}} = 4,096$) and must feed it back into the recurrent trunk as the input embedding for step $t+1$.
     The input embedding table is dimensioned for $V_{\text{in}} = 16,384$.
     Are the 4,096 formal tokens mapped directly to a shared subset of $V_{\text{in}}$, or do they possess a dedicated formal embedding matrix? If separate, that requires $4,096 \times 512 = 2,097,152$ parameters. The proposal fails to state weight tying.
  4. **Dense Retriever Passage Encoder**:
     In §5.3, the table includes "Dense Retriever Query Projection: $512 \times 256 = 131,328$".
     What neural network embeds the millions of science/textbook chunks in the datastore? If the datastore is embedded by an external neural net (e.g. `potion-base-8M` with 7.56M parameters), those parameters MUST be accounted for unless the retriever is 100% zero-parameter BM25!
- **Concrete Fix**:
  Provide an honest, fully inclusive parameter table recounting every module:

  | Component Module | Explicit Dimensions & Formulas | Recounted Parameters |
  |---|---|---|
  | **Input Embeddings (Shared)** | $16,384 \times 512$ (Tied formal output tokens occupy IDs $0\dots 4095$) | 8,388,608 |
  | **Mode Tokens** | $16 \times 512$ (`[SOLVE]`, `[PROPOSE]`, `[VERIFY]`, `[RETRIEVE]`) | 8,192 |
  | **True Bi-GDN Reader (4 Layers)** | 4 Layers $\times 2$ Directions (Forward + Backward separate GDN matrices) | 22,052,864 |
  | **Recurrent Reasoning Trunk** | 8 Weight-Tied GDN Layers ($d=512, d_{\text{mlp}}=1536$, iterated $r \le 16$) | 31,522,816 |
  | **Span Pointer Head (DROP)** | Query/Key projections for input span selection ($2 \times 512 \times 512$) | 524,288 |
  | **Formal IR Emission Head** | Linear projection from trunk $d=512 \to V_{\text{out}}=4,096$ | 2,097,152 |
  | **Value / PRM Head** | 2-Layer MLP projecting pooled state $h_t \to 1$ | 263,169 |
  | **Adaptive Halting Head** | Linear classifier projecting $h_t \to 2$ | 1,026 |
  | **Static Datastore Embeddings** | Shared reader encoder trunk used for passage keys (0 added params) | 0 |
  | **Layer Normalizations & RMS** | RMSNorm scale parameters across all layers | 32,768 |
  | **TOTAL TRAINABLE PARAMETERS** | **Mathematically verified sum of all inference weights** | **64,890,883 (64.89M)** |

  The recounted total of **64.89M parameters** comfortably satisfies the $<100\text{M}$ hard constraint while providing full transparency.

### 5.2 Secret Transformer Hazards (Titans Memory Branch Audit)
- **Severity**: **MINOR**
- **Detailed Technical Reasoning**:
  In §5.3, the proposal includes a "Titans Neural Test-Time Memory" ($525,312$ parameters).
  As documented in `archs.md` §2 row 43, the canonical Titans architecture published by Behrouz et al. (arXiv 2501.00663) incorporates a **sliding-window attention core** in its primary branch (MAC/MAG variants).
  While the neural memory module itself is an MLP updated by test-time surprise gradients, incorporating any attention layers from the published repository would violate BRIEF.md §2.2.
- **Concrete Fix**:
  Explicitly state in §5.1 that ORRERY-0 utilizes **only the decoupled associative memory MLP branch** of Titans, completely stripping the sliding-window attention core and wiring the memory directly into the Gated DeltaNet recurrent stream.

---

## 6. Priority Attack 5: Citation Laundering & Unverified Reference Audit (AUDIT §2)

Proposal Minimal v1 exhibits multiple instances of citation laundering—attaching real academic papers to claims they do not make or citing formulas incorrectly.

```
+----------------------------------------------------------------------------------------------------+
|                                    CITATION LAUNDERING AUDIT MATRIX                                |
+-----------------------+-----------------------------+----------------------------------------------+
| Cited Reference       | Proposal Minimal v1 Claim   | Actual Source Reality (AUDIT §2 Fact Check)  |
+-----------------------+-----------------------------+----------------------------------------------+
| arXiv 2504.13837      | "Distillation caps student  | Directly opposite: finds distillation adds   |
| (RLVR vs Distill)     | at teacher support" (§1.1)  | new reasoning; RLVR narrows policy support.  |
+-----------------------+-----------------------------+----------------------------------------------+
| Gated DeltaNet Eq.    | Stated update formula:      | Mathematically broken: Hadamard product with |
| (arXiv 2412.06464)    | S_t = S ⊙ (1 - α k kᵀ) + ...| outer product is invalid; α is decay gate.   |
+-----------------------+-----------------------------+----------------------------------------------+
| arXiv 2307.06962      | "Suffix-automaton vocabulary| A phrase-copy language model paper (CoG);    |
| (CoG Paper)           | masking of renderer" (§9.3) | contains zero suffix-automaton masking.      |
+-----------------------+-----------------------------+----------------------------------------------+
| arXiv 2303.08774      | Span-copy pointer mechanism | GPT-4 Technical Report; covers neither       |
| (GPT-4 Report)        | for DROP F1 (§7 row A11)    | span-copy pointers nor DROP pointer heads.   |
+-----------------------+-----------------------------+----------------------------------------------+
| arXiv 2511.18397      | Sandbox SHA-256 boot hashes | Paper on reward-hacking / inoculation;       |
|                       | and runtime security (§6.6) | covers zero OS sandboxing or SHA-256 boot.   |
+-----------------------+-----------------------------+----------------------------------------------+
| METR o3 Report        | Linux cgroup sandboxes and  | Evaluates model task completion; covers zero |
|                       | SIGKILL tamper triggers     | nsjail/cgroup mechanics or canary tripwires. |
+-----------------------+-----------------------------+----------------------------------------------+
| Unverified Scores     | MATH 43.1, DROP 70.2,       | Unverified secondary numbers labeled as      |
|                       | StrategyQA 70.3 [검증됨]     | [검증됨]; verified are 34.1, 64.1, 51.5.     |
+-----------------------+-----------------------------+----------------------------------------------+
| Phantom Datasets      | "FineWeb-2 Korean Educ."    | Does not exist in data.md; must use real     |
|                       | (§8.1)                      | verified open datasets only.                 |
+-----------------------+-----------------------------+----------------------------------------------+
```

### 6.1 The Broken Gated DeltaNet Update Equation
In §5.1, the proposal writes:
$$S_t = S_{t-1} \odot (1 - \alpha_t k_t k_t^T) + \beta_t v_t k_t^T$$
This equation is a mathematical monstrosity:
- $S_{t-1} \in \mathbb{R}^{d \times d}$ is a matrix.
- $k_t k_t^T \in \mathbb{R}^{d \times d}$ is an outer-product matrix.
- The Hadamard product $\odot$ represents elementwise multiplication. In DeltaNet, the retention and overwrite operation is a **matrix multiplication** (projecting $S_{t-1}$ onto the subspace orthogonal to $k_t$), not an elementwise mask!
- In the true Gated DeltaNet formulation (arXiv 2412.06464), $\alpha_t \in [0, 1]$ is a scalar or channel-wise decay gate, and $\beta_t$ is the write learning rate.
- **Concrete Fix**: Replace with the mathematically rigorous, canonical Gated DeltaNet update equation:
  $$S_t = \alpha_t S_{t-1} (I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
  Where $\alpha_t = \text{diag}(\sigma(W_\alpha x_t))$ controls memory retention, $\beta_t = \text{diag}(\sigma(W_\beta x_t))$ controls update rate, and $k_t, v_t$ are normalized key/value projections.

### 6.2 Cleansing the Citation Ledger
- **arXiv 2504.13837**: Rewrite text to cite it accurately as a warning against RLVR policy narrowing, motivating the need for temperature perturbation and POET archive stepping-stones.
- **CoG (arXiv 2307.06962)**: Remove citation from vocabulary masking; cite correctly as a phrase-copy retrieval baseline.
- **DROP Reference**: Replace arXiv 2303.08774 with the foundational pointer network paper: Vinyals et al., *"Pointer Networks"*, NeurIPS 2015 (arXiv 1506.03134).
- **METR & Inoculation**: Retract claims attributing Linux cgroup engineering to METR o3; cite standard Linux kernel cgroups v2 documentation.

---

## 7. Priority Attack 6: Fair Play, Safety Enforceability, and 12-Week Feasibility

### 7.1 Fragility of the $H_g > 8\%$ Kill Switch
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:
  In §3 row 3 and §4.1, the primary automated defense against reward hacking is defined as:
  $$H_g = S_g(\mathcal{P}_{\text{in-loop}}) - S_g(\mathcal{D}_{\text{held-out}}) > 8.0\%$$
  Triggering an immediate kill switch if $H_g > 8.0\%$.

  This detector is fundamentally flawed in a dynamic self-play environment:
  1. **Curriculum Difficulty Shifts**: If the proposer is functioning correctly, it actively increases the difficulty of $\mathcal{P}_{\text{in-loop}}$. In generations where the proposer generates exceptionally difficult frontier tasks, in-loop solve rate can plunge to 30% while held-out performance on static benchmarks remains steady at 75%. Here $H_g = -45\%$, completely blinding the detector.
  2. **Domain Desynchronization**: If the engine is training on a specific domain (e.g. Countdown Arithmetic), in-loop score on arithmetic will reach 90%, while held-out performance on a multi-task suite (which includes MATH, GPQA, and Code) will sit at 40%. Under the proposal's rule, the engine would trigger a fatal kill switch on Generation 2 simply because the held-out suite measures other skills!
- **Concrete Fix**:
  Replace static cross-suite divergence with **Domain-Matched In-Loop vs. Perturbed-Twin Divergence**:
  - Evaluate $H_g$ strictly between training tasks $\mathcal{P}_{\text{in-loop}}$ and their **exact held-out counterfactual twins** $\mathcal{P}_{\text{twin}}$ (identical problem structure with perturbed constants).
  - The kill switch activates if and only if:
    $$H_g^{\text{twin}} = S(\mathcal{P}_{\text{in-loop}}) - S(\mathcal{P}_{\text{twin}}) > 10.0\%$$
    for two consecutive generations, proving that the model is exploiting surface artifacts rather than learning invariant reasoning.

### 7.2 12-Week Roadmap Feasibility & Kaggle Operational Friction
- **Severity**: **MAJOR**
- **Detailed Technical Reasoning**:
  The proposal assumes a solo builder with zero budget can seamlessly execute:
  - Week 1–2: Triton 3.2 kernel stabilization, JAX/XLA scan porting, AST sandboxes, Countdown RSI.
  - Week 3: Asymmetric PAIRED regret self-play on symbolic equations.
  - Week 4: Cord-cut transition to Z3 SMT logic.
  - Weeks 5–12: Scaling across 17 benchmark suites to beat GPT-3.5.

  This roadmap ignores the severe operational friction documented in `compute.md` and `archs.md`:
  - **Triton 3.2 vs. Turing (T4)**: Triton $\ge 3.3$ dropped sm_75 support. Pinning Triton 3.2 inside Kaggle's rolling Docker container requires custom wheel installation on every session boot.
  - **Kaggle Session Disconnections**: Kaggle limits sessions to 12h (GPU) and 9h (TPU). A 10B token pre-training run requires chaining at least two 9h TPU sessions with automated checkpoint re-loading from Hugging Face Hub.
  - **TPU Oversubscription**: TPU v5e-8 is frequently unavailable or queued during peak contest periods (`compute.md` §1). If the entire pre-training pipeline depends on JAX TPU, the project stalls.
- **Concrete Fix**:
  Structure a robust **Engineering Resilience Plan** into Week 1:
  1. Build and push a self-contained Docker container / Kaggle dataset containing pre-compiled wheels for PyTorch 2.5.1, Triton 3.2.0, and pure PyTorch chunked scan fallbacks.
  2. Implement an automated **Session Chaining Script**: each session reads the latest checkpoint commit hash from HF Hub, runs 7.5 hours of training, writes checkpoints every 30 minutes, commits the final state to HF Hub, and gracefully terminates.
  3. Enforce a **Dual-Runtime Architecture**: all core model definitions must run in both PyTorch (for T4 development, rollouts, and fallbacks) and JAX/Flax (for TPU pre-training).

---

## 8. Master Remediation Matrix: Severities & Ambition-Preserving Concrete Fixes

```
========================================================================================================================
                                     ORRERY-0 RED TEAM REMEDIATION MATRIX
========================================================================================================================
Vulnerability ID   Section   Severity   Technical Defect Summary            Required Concrete Action for Phase C
------------------------------------------------------------------------------------------------------------------------
VULN-RSI-01        §12.1     FATAL      Experiment 1 hardcodes success      Eliminate all code; specify rigorous plain-
                                        via acc_g1 = acc_g0 + 6.0           language plan with random-reward control arm.

VULN-RSI-02        §1.4, 2.1 FATAL      Reading verifier hole; code checks  Mandate reverse generation from formal ASTs;
                                        execution, not text formalization   gate progress on held-out perturbed twins.

VULN-COMP-01       §2.5      FATAL      Recurrent depth r in [1, 16]        Recalculate FLOPs at r_bar=6.4: rollout is
                                        omitted; FLOPs understated 4.8x     7.03 PFLOPs, train is 10.5 PFLOPs; cycle is 6.7m.

VULN-LAT-01        §8.4      FATAL      Single-query latency 6 ms on T4     Retract 6 ms fantasy; state honest bandwidth
                                        violates memory bandwidth by 87x    limit (~523 ms at r=4); explain batch amortization.

VULN-CIT-01        §1.1, 5.1 FATAL      Citation laundering (2504.13837);   Correct citation; replace Hadamard product with
                                        broken DeltaNet update equation     canonical S_t = α S_{t-1}(I - β k kᵀ) + β v kᵀ.

VULN-BENCH-01      §7.1      FATAL      ARC-AGI 46.8% exceeds non-attn      Add 2D multi-axis scan and CA/DSL tools;
                                        evidence (TRM-MLP drops to 29.6%)   lower projected target to realistic 34.5% [추정].

VULN-SCORE-01      §7.1      MAJOR      Scores reverse-engineered at        Label all targets [추정]/[가설]; provide binomial
                                        G + 1.5%; CIs missing               95% CI error bounds for every benchmark row.

VULN-PARAM-01      §5.3      MAJOR      Omitted pointer head, formal vocab  Provide fully accounted table totaling 64.89M
                                        embeddings, backward reader params  params (<100M), showing exact per-module math.

VULN-CURR-01       §2.4      MAJOR      "Next domain" is a static human     Replace with EDLI-driven Cross-Domain
                                        list, not an autonomous RSI skill   Constraint Synthesis over sandboxed oracles.

VULN-PROP-01       §2.1      MAJOR      Aphasic core (4k vocab) cannot      Define proposer as Task Specification Program
                                        emit NL problems for self-play      synthesizer compiled via deterministic templates.

VULN-NAME-01       §1.1, 7   MAJOR      Outdated frontier tin cans names;   Rename to Claude Opus 5.5, Gemini 4 Argon,
                                        missing no-teacher ablation track   GPT-6 Astra; add explicit no-teacher protocol.

VULN-SAFE-01       §3, 4     MAJOR      Static H_g > 8% kill switch fails   Replace with Domain-Matched In-Loop vs
                                        under dynamic curriculum shifts     Perturbed-Twin divergence detector (H_twin > 10%).
========================================================================================================================
```

### Final Conclusion & Guidance for Phase C Revision
The Minimal Fastest-Cycle RSI philosophy remains the most viable architecture for a zero-capital solo builder. The core insight—that eliminating natural language generation shrinks embeddings, cuts rollout lengths, and enables high-frequency generation cycles on free Kaggle hardware—is brilliant and must be preserved.

However, Proposal Minimal v1 attempted to win arguments with fabricated numbers, impossible latencies, and self-fulfilling scripts. **It did not need to cheat.** At real recurrent depth $\bar{r}=6.4$, a 6.7-minute generation cycle yielding 100+ autonomous generations per week on free TPU/T4 hardware is still an extraordinary, world-class experimental velocity that crushes complex multi-agent architectures.

The Lead Designer must accept every finding in this critique, strip all code, fix the arithmetic, resolve the reading verifier hole via reverse generation, cleanse the citations, and submit a battle-hardened, honest, and truly formidable **Proposal Minimal v2**.
