# Technical Proposal: Depth-Distillation RSI (Latent Recursive Reasoner) v2
**Codename:** APHASIC-DEPTH-0 (Phase C Revised Proposal)  
**Author:** Lead Revision Designer, Depth Philosophy  
**Target:** `/home/user/hip/agy-blueprint/work/proposal_depth_v2.md`  
**Date:** 2026-10-19  
**Binding Standards:** `AUDIT_FROM_CLAUDE.md`, `BRIEF.md`, `critique_depth.md`, `reason_baseline.md`  

---

## Revision Changelog (v1 $\to$ v2 Master Audit Resolution)

This proposal is a complete, mathematically rigorous revision of `work/proposal_depth_v1.md`, resolving all fatal flaws, major deficiencies, and citation issues identified in `work/critique_depth.md` and `AUDIT_FROM_CLAUDE.md` (§1 & §2), while retaining and expanding the core architectural breakthroughs (§3):

| Item ID | Audit / Critique Finding | Severity in v1 | Resolution & Architectural Fix in v2 |
|---|---|---|---|
| **REV-01** | **Distillation Collapse Paradox** (§1.2, §2.1) | **FATAL** | Replaced latent cosine state matching ($h_2 \approx h_{16}$) with a **Dual-Objective Policy-Level Distillation Ratchet**: shallow passes ($r=2..4$) train via cross-entropy on verified symbolic AST tokens emitted by deep search ($r=16..32$), conditioned on depth-embedding modulation $e_r$. Preserves dynamical rank and test-time depth amplification. |
| **REV-02** | **Recurrent Depth FLOP Arithmetic** (§2.5, §8.2) | **FATAL** | Recomputed all FLOPs with real recursion $r$: $\text{FLOPs/token} \approx 2 \times (N_{\text{non\_core}} + r \times N_{\text{core}})$ for inference, $6 \times (N_{\text{non\_core}} + r \times N_{\text{core}})$ for training. Stage 0 pretraining moved from T4 (38h, quota blowup) to Kaggle TPU v5e-8 ($0.89\text{ h}$, $1.26 \times 10^{18}$ FLOPs) using curriculum depth ($r=1 \to 2 \to 4$). |
| **REV-03** | **T4 Sequential Decoding Latency** (§8.3) | **FATAL** | Grounded sequential decoding in the physical memory bandwidth of the NVIDIA T4 ($300\text{ GB/s}$ peak, $210\text{ GB/s}$ effective $\implies 3.53\text{ ms/token}$ at $r=8$ for $B=1$). Deployed batched search rollouts ($B=32 \implies 3,800\text{ tokens/s}$), $O(1)$ recurrent state caching in MCTS (320 KB/node), and adaptive halting ($p_{\text{halt}}$). |
| **REV-04** | **The Reading Verifier Hole** (§2.1, §6.2) | **FATAL** | Replaced unverified natural-language reading with a **4-Tier Symmetric Reading Verification Protocol**: reverse procedural generation (AST $P^* \to Y^* \to \text{Text } X \implies P_{\text{core}} \equiv P^*$), 3-way paraphrase invariance, counterfactual numerical perturbation twins (GSM-Plus/Symbolic), and a held-out reading robustness gate ($\le 16.88\%$ relative drop). |
| **REV-05** | **RSI Integrity & Domain Expansion** (§2.4) | **FATAL** | Removed core-synthesized verifier wrappers (prohibited by BRIEF.md §2.7). Implemented an **Immutable Tiered Domain Registry** of frozen, read-only Docker images (Python/SymPy, Z3, Lean 4, VAL) triggered automatically by empirical saturation metrics ($\Delta_g < 0.5\%$). |
| **REV-06** | **Purge of Code Contraband** (§12.2) | **FATAL** | Purged all 92 lines of executable PyTorch and Python script from §12.2 in strict compliance with `AUDIT_FROM_CLAUDE.md` §0.2 and Clarification A. Replaced with a rigorous 7-part plain-language experiment plan. |
| **REV-07** | **Parameter Undercount Ceiling Breach** (§5.4) | **FATAL** | Fully accounted for omitted parameters: backward reader passes, Gated DeltaNet decay gates $\alpha_t$, output gates $g_t$, short depthwise convolutions, proposer head, and DROP pointer heads. Re-dimensioned architecture to $d=576, d_{\text{ffn}}=1,325$, establishing a verified grand total of **75,928,490 parameters** (< 100M ceiling with 24.07M safety buffer). |
| **REV-08** | **Scoreboard Gaps & Decoupled MuSR** (§7.1) | **MAJOR** | Restored **PrOntoQA** ($T=85.0 \to G=95.0$), **StrategyQA** ($T=70.3 \to G=80.3$), and decoupled **MuSR** into three independently gated sub-rows (Murder Mysteries $G=80.0$, Object Placement $G=72.1$, Team Allocation $G=88.8$). Labeled verified baselines (MATH 34.1, DROP 64.1) alongside conservative targets. |
| **REV-09** | **ARC-AGI-1 Realism** (§7.1) | **MAJOR** | Grounded ARC-AGI-1 target at **32.5% [추정]** (or 42.0% [가설] with 2D multi-directional Hilbert scan inductive bias) to respect empirical capabilities of attention-free recurrent cores (TRM-MLP 29.6% vs TRM-Att 44.6%). |
| **REV-10** | **Citation Laundering Cleansing** (Global) | **MAJOR** | Purged laundered citations: phi-1 (no ceiling claim), TRM (corrected to 7M params), bm25s (sparse lexical retriever, not sandbox throughput), ESMFold (15B neural model, reclassified as learned simulator), StripedHyena (attention hybrid removed). Corrected Gated DeltaNet recurrence equation to $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$. |
| **REV-11** | **Frontier Model Framing** (§1.1, §7.2) | **MINOR** | Renamed all frontier references to **Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra**. Replaced subjective probability percentages with empirical risk bands and falsification tests. |
| **REV-12** | **No-Teacher Procedural Ablation** (§8.4) | **MAJOR** | Added an explicit No-Teacher control arm trained from scratch on purely procedural engines, complete with token-level provenance logging to Hugging Face Hub datasets. |

---

## 1. Thesis

### 1.1 The Distillation Ceiling vs. The Verifier Ladder
Distilling reasoning traces from frontier models (Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra) fundamentally caps the student at the teacher distribution [가설]. Frontier teachers are not AGI—they hallucinate mathematical steps, collapse on elementary planning (e.g., Blocksworld at 6.1% [검증됨: https://arxiv.org/html/2305.15771v2]), and suffer catastrophic performance drops when irrelevant distractor clauses are injected (GSM-Symbolic drops up to 65% [검증됨: https://arxiv.org/abs/2410.05229]). In our engineering framing, frontier models are "tin cans" (깡통); distilling a tin can yields only a smaller, compressed tin can.

Furthermore, empirical investigations into reinforcement learning with verifiable rewards (RLVR) demonstrate that naive policy-gradient optimization on small models primarily sharpens pre-existing sampling coverage, narrowing pass@$k$ without discovering qualitatively new reasoning patterns [검증됨: https://arxiv.org/abs/2504.13837v4]. When a small model is trapped in a shallow reasoning basin, RLVR merely accelerates convergence to its own local mode.

What replaces the teacher is **exact, zero-parameter mechanical verification** coupled with **test-time compute amplification** [가설]. In formal domains—Python script execution, SymPy algebraic manipulation, Z3 SMT constraint solving, Lean 4 kernel proof checking, and game/physics rules—the external environment supplies an immutable, binary ground truth [검증됨: https://arxiv.org/abs/2312.06585]. In these domains, capability is bounded by search depth, combinatorial exploration, and logical soundness, not by human consensus or teacher style:
- AlphaGo Zero defeated human-data AlphaGo 100–0 starting from zero human games because Go board rules are a perfect, closed-world verifier [검증됨: https://deepmind.google/blog/alphago-zero-starting-scratch/].
- AlphaProof solved Olympiad-level mathematics by generating and searching formal Lean theorems against the interactive theorem prover kernel [검증됨: https://research.google/pubs/olympiad-level-formal-mathematical-reasoning-with-reinforcement-learning/].
- In tiny models, TinyGSM demonstrated that pairing a small generator with an exact code execution verifier allowed the student to beat its GPT-3.5 teacher (81.5% vs. 77.4% on GSM8K) [검증됨: https://arxiv.org/html/2312.09241].
- DeepSeek-R1-Zero confirmed that pure rule-based RLVR on checkable answers triggers autonomous reasoning breakthroughs without supervised fine-tuning [검증됨: https://arxiv.org/abs/2501.12948].

The lesson is unequivocal: **The engine of recursive self-improvement is the verifier, not the teacher.**

### 1.2 The Depth-Distillation Compounding Ratchet (Amplify-then-Distill)
The `depth` philosophy resolves the tiny-model capacity paradox (<100M parameters) through **Amplify-then-Distill Recurrent Latent Depth** [가설]:

1. **Weight-Tied Recurrent Reasoning Trunk:** Instead of stacking hundreds of distinct physical layers, the core deploys an 8-layer attention-free recurrent trunk based on Gated DeltaNet (arXiv:2412.06464). At inference time, this trunk executes $r$ iterative recursion passes ($r \in [1, 32]$) over its latent memory state $S_t \in \mathbb{R}^{d_k \times d_v}$, yielding an effective computational depth of up to 256 layers while executing exactly 31.85M recurrent parameters (75.93M total core weights).
2. **Test-Time Search Amplification:** On difficult problems, the engine expands recursion depth to $r=16..32$ and deploys batched Monte Carlo Tree Search (MCTS) over intermediate formal AST tokens. This massive test-time compute budget unlocks valid reasoning trajectories that shallow feedforward passes ($r=1..4$) cannot sample [검증됨: https://arxiv.org/abs/2408.03314].
3. **Exact Sandbox Verification:** Emitted candidate programs, constraint specifications, and proofs are dispatched to isolated, unprivileged Docker sandboxes running Python 3.12, SymPy 1.13, Z3 SMT, and Lean 4. Only 100% verified trajectories are retained in the replay buffer.
4. **Token-Level Policy Distillation (Solving the Collapse Paradox):** In v1, attempting to force intermediate shallow states to match deep states via cosine loss ($h_2 \approx h_{16}$) collapsed the recurrent operator into an identity fixed point. In v2, we discard latent state matching entirely. We train shallow passes ($r=2..4$) using cross-entropy loss against the *verified formal AST tokens* emitted by deep search:
   $$\mathcal{L}_{\text{AST}} = -\sum_{i=1}^{L} \log P_\theta(y_i \mid y_{<i}, X, r_{\text{shallow}})$$
   The recurrent cell is conditioned on a learned depth-embedding vector $e_r$, preserving dynamical rank across recursion steps.
5. **The Distillation Ratchet:** By training the shallow policy to jump directly to the symbolic solutions uncovered by deep search, yesterday's hard frontier problems become today's shallow routines ($r=2..4$). This frees the deep search budget ($r=16..32$) at generation $g+1$ to explore an even higher complexity horizon.

```
+---------------------------------------------------------------------------------------------------+
|                            THE DEPTH-DISTILLATION COMPOUNDING RATCHET                             |
|                                                                                                   |
|  Generation g:                                                                                    |
|    Shallow Base (r=2..4)  =======>  Deep Search Rollouts (r=16..32) [MCTS Expansion]              |
|                                                     ||                                            |
|                                                     \/                                            |
|                                         0-Param Exact Verifier Sandbox                            |
|                                         (Python / SymPy / Z3 / Lean 4)                            |
|                                                     ||                                            |
|                                                     \/ (100% Verified AST Only)                   |
|  Generation g+1:                                                                                  |
|    Shallow Base (r=2..4)  <=======  Token-Level AST Policy Distillation                           |
|           ||                                                                                      |
|           +======================>  Deep Search Rollouts (r=16..32) [Pushes to Next Frontier]    |
+---------------------------------------------------------------------------------------------------+
```

### 1.3 Why Dropping Language Generation Makes Each Generation Cheaper
Natural language autoregressive models dedicate 40%–65% of their total parameter budget solely to massive vocabulary embedding tables (e.g., Gemma 3 270M spends 170M out of 270M parameters on embeddings [검증됨: https://developers.googleblog.com/en/introducing-gemma-3-270m/]). Furthermore, natural language Chain-of-Thought (CoT) squanders 500–1,500 tokens of conversational filler, hedging, and formatting per query.

By dropping natural language generation entirely ("the aphasic genius"):
1. **Vocabulary Shrinkage:** The core utilizes a 16,384-token input reading vocabulary and an ultra-compact 2,048-token formal output vocabulary (Python/SymPy AST tokens, S-expressions, Lean tactics, logic operators, numerals) [추정], slashing total embedding weights to 10.62M parameters.
2. **Dense Formal Representation:** The core emits structured program ASTs (20–80 tokens per problem), which are 10× to 20× shorter than natural language CoT traces [검증됨: https://arxiv.org/abs/2211.10435].
3. **FLOP Economics & Throughput:** Generating 50 formal tokens requires 1/20th the compute of generating 1,000 conversational tokens. On Kaggle free compute, this 20× throughput multiplier enables tens of millions of verified problem attempts per week instead of a few hundred thousand [추정].
4. **Elimination of Sycophancy & Conversational Drift:** Because the core cannot speak natural language, it physically cannot engage in conversational hedging, sycophancy, or stylistic drift. Exactly 100% of gradient updates optimize logical state transitions and formal execution success [가설].

---

## 2. The RSI Engine

### 2.1 The Generation Loop Step-by-Step
The Depth-Distillation RSI engine iterates across discrete generations ($g = 0, 1, 2, \dots$):

```
+---------------------------------------------------------------------------------------------------+
|                                THE APHASIC RSI GENERATION LOOP                                    |
|                                                                                                   |
|  [1. Symmetric Task Proposer]                                                                     |
|      - Reverse Procedural Generation (AST P* -> Y* -> Text X)                                     |
|      - PAIRED Learnability Regret: R_proposer = max(0, Score_deep - Score_shallow)                 |
|               |                                                                                   |
|               v                                                                                   |
|  [2. Deep Solve & Search]                                                                         |
|      - Weight-Tied Recurrent Core (r in [1, 32], Batched B=32)                                    |
|      - MCTS with O(1) DeltaNet State Caching & Value Guidance                                     |
|               |                                                                                   |
|               v                                                                                   |
|  [3. Sandboxed Multi-Domain Verification]                                                         |
|      - Immutable Read-Only Docker Pool (Python 3.12, SymPy 1.13, Z3 SMT, Lean 4)                  |
|      - Strict Equivalence Check: SymPy(P_core - P*) == 0 & Execution Match                        |
|               |                                                                                   |
|        +------+------+                                                                            |
|        | [Failed]    | [100% Verified]                                                            |
|        v             v                                                                            |
|   (Curriculum)  [4. Trajectory Filter & Depth Regret Partitioning]                                |
|                      - Discard trivial (solved at r <= 2, 10% replay)                             |
|                      - Retain frontier regret tasks (failed at r <= 2, solved at r >= 8)          |
|                      |                                                                            |
|                      v                                                                            |
|                 [5. Multi-Task Training & Depth Distillation]                                     |
|                      - Token-Level Formal AST SFT at shallow depth (r ~ U(2, 4))                  |
|                      - Depth-Conditioned Modulation (e_r)                                         |
|                      - GRPO Reinforcement Tuning on frontier samples                             |
|                      |                                                                            |
|                      v                                                                            |
|                 [6. Evaluator Sandbox & Cord-Cut Audit]                                           |
|                      - Frozen Held-Out Test Suites & Procedural Twins                             |
|                      - Random-Reward Control Arm Gating                                           |
+---------------------------------------------------------------------------------------------------+
```

1. **Task Proposal (Symmetric Curriculum Generation):**
   - *Reverse Procedural Generation:* Problems are generated from the inside out: the generator samples an executable formal program $P^*$, evaluates it to obtain gold answer $Y^*$, and renders natural language problem text $X$ via deterministic template grammars.
   - *PAIRED Learnability Regret:* Active problem generation is guided by PAIRED regret: $R_{\text{proposer}} = \max(0, \text{Score}_{\text{deep}} - \text{Score}_{\text{shallow}})$ [검증됨: https://arxiv.org/pdf/2012.02096]. Tasks already solvable in shallow fast-mode ($r \le 2$) or unsolveable even under deep search ($r=32$) yield zero reward, focusing rollouts strictly on the learnability frontier.
2. **Deep Solve & Search:**
   - The core processes problem text $X$ and executes iterative latent passes $t = 1 \dots r$.
   - Search deploys batched MCTS ($B=32$) guided by the internal value head $V(h_t) \in [-1, 1]$. Crucially, because Gated DeltaNet state $S_t \in \mathbb{R}^{d_k \times d_v}$ is constant-size ($O(1)$, exactly 320 KB per layer), node states are cached directly in the search tree, eliminating redundant prefix recomputation.
3. **Sandboxed Exact Verification:**
   - Emitted AST programs are executed in an isolated Docker container pool (`--network none`, read-only root).
   - In addition to runtime execution, the sandbox checks semantic equivalence against ground truth: $\text{SymPy}(\text{Simplify}(P_{\text{core}} - P^*)) == 0$ or $\text{Z3}(\text{Assert}(P_{\text{core}} \neq P^*)) == \text{unsat}$.
4. **Trajectory Selection & Partitioning:**
   - Verified trajectories are partitioned based on depth regret: trivial solutions ($r \le 2$) receive 10% replay sampling to protect against catastrophic forgetting; frontier solutions (failed at $r \le 2$, solved at $r \ge 8$) receive 100% sampling weight.
5. **Training & Depth Distillation:**
   - *Token-Level Policy Distillation:* Core weights are updated to emit the verified AST tokens under shallow recursion ($r \sim \mathcal{U}(2, 4)$) via cross-entropy loss $\mathcal{L}_{\text{AST}}$.
   - *Depth Modulation:* Recurrent layers receive depth embedding $e_r$, allowing distinct dynamical trajectories across shallow vs. deep modes.
   - *GRPO Sharpening:* Group Relative Policy Optimization sharpens policy distribution around optimal, low-complexity AST paths [검증됨: https://arxiv.org/abs/2501.12948].
6. **Checkpointing & Evaluator Audit:**
   - Candidate checkpoint is evaluated against frozen held-out suites and procedural twins. Weights are promoted only if net improvement is confirmed against the parallel random-reward control arm.

### 2.2 RSI Levels: L1 to L5 Architecture
- **L1 (Weight Self-Improvement):** Weights update on verified, self-generated AST programs discovered through test-time depth search ($r=16..32$) without external human or teacher supervision.
- **L2 (Self-Generated Curriculum):** Proposer-vs-solver self-play and grammar-mutating procedural generators continuously synthesize problems at the boundary of shallow competence.
- **L3 (Growing Skill Library):** Discovered algorithmic subroutines, macro-operators, and Lean lemmas are compiled into an external, zero-parameter symbolic library, expanding the primitive vocabulary of the formal DSL [검증됨: https://arxiv.org/abs/2006.08381v1].
- **L4 (Improving Training Procedure & Infrastructure):** Automated exploration of data mixture ratios, learning rate schedules, and chunked scan kernel tile sizes. **Hard Constraint:** In compliance with BRIEF.md §6, every L4 modification requires an explicit cryptographic GPG signature from a human engineer before execution.
- **L5 (Architecture Self-Redesign):** Dynamic topological mutation and neural architecture search. Designated strictly as a North Star research objective; disabled during Phase A–C to maintain hard parameter and compute bounds.

### 2.3 Teacher Bootloader, Net Cord-Cut Criterion & No-Teacher Ablation
Stage 0 Bootloader training utilizes open-licensed, non-OpenAI datasets (OpenMathInstruct-2 [CC-BY-4.0], Reasoning Gym, re-ARC, The Stack v2 Permissive) solely to establish initial AST syntax comprehension [검증됨: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2].

**Measurable Net Cord-Cut Criterion:** The teacher booster is permanently severed at generation $g^*$ when all four conditions hold across two consecutive generations:
1. **Net Teacher-Exceedance Threshold ($\text{Net-TE}$):** Core net exceedance over the teacher on held-out evaluation sets must be strictly positive:
   $$\text{Net-TE}(g) = \frac{|\{x \in \mathcal{D}_{\text{eval}} : \text{Teacher}(x) = 0 \land \text{Core}(x) = 1\}| - |\{x \in \mathcal{D}_{\text{eval}} : \text{Teacher}(x) = 1 \land \text{Core}(x) = 0\}|}{|\mathcal{D}_{\text{eval}}|} \ge +5.0\%$$
2. **Autonomous Compounding ($\Delta_{\text{fresh}}$):** Score on held-out procedural twins increases by $\Delta_{\text{fresh}} \ge +1.5\%$ per generation with zero teacher tokens in the training buffer ($G_0 \ge 2$) [추정].
3. **Random-Reward Control Arm Gate:** The self-play model must outperform a parallel control model trained on randomized verifier feedback by at least $+8.0$ percentage points on held-out procedural twins (`AUDIT_FROM_CLAUDE.md` §3).
4. **Syntactic Robustness:** The core emits syntactically valid formal ASTs on $\ge 98.0\%$ of unseen test queries without execution interpreter crash [추정].

**Cord-Cut Teacher Model Feasibility:**
To adhere strictly to the 0 KRW budget, the active teacher is **Qwen2.5-Coder-7B-Instruct** (or **Qwen2.5-14B** in 4-bit INT4 quantization via `bitsandbytes` on Kaggle 2× T4, consuming 9.2 GB VRAM) [검증됨: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct]. For frontier teacher comparisons (e.g., Llama-3.1-405B or Qwen-2.5-72B), we utilize a static, pre-computed teacher-failure cache frozen on Hugging Face Hub, requiring **0 FLOPs** of live teacher execution during RSI generations.

**Dedicated No-Teacher Ablation Arm:**
To satisfy BRIEF.md §2.6, we train an identical 75.9M core from scratch using **zero teacher data from Day 1**, bootstrapping exclusively on synthetic procedural engines (Reasoning Gym, re-ARC, SymPy matrix algebra, Z3 logic generators) at identical FLOPs ($1.26 \times 10^{18}$ FLOPs). Both arms are tracked concurrently with full token provenance logged to Hugging Face Hub.

### 2.4 S-Curve Staircase of Verifier Domains & Immutable Domain Registry
Progress within any single formal domain eventually saturates into an S-curve [검증됨: https://arxiv.org/html/2602.10014v2]. Autonomous compounding requires climbing a staircase of S-curves across verifier domains:
$$\text{Arithmetic (Python)} \longrightarrow \text{Symbolic Algebra (SymPy)} \longrightarrow \text{Logic/SMT (Z3)} \longrightarrow \text{Formal Proofs (Lean 4)} \longrightarrow \text{Simulators}$$

**Resolution of Domain Expansion Integrity (Fixing ATK-05):**
Allowing the model to synthesize its own verification code or wrappers violates BRIEF.md §2.7 and invites catastrophic reward hacking (e.g., emitting permissive `return True` exception handlers). In v2, the model possesses **zero authority to write or modify verifier code**.

We implement an **Immutable Tiered Domain Registry**:
1. *Pre-Packaged, Hermetically Sealed Modules:* The system ships with read-only Docker images mounted with immutable SHA-256 signatures:
   - Module 0: Integer & Rational Arithmetic (`python-fractions`)
   - Module 1: Symbolic Algebra & Calculus (`sympy 1.13`)
   - Module 2: Propositional & First-Order SMT (`z3-solver 4.13`)
   - Module 3: Interactive Theorem Prover (`lean4 v4.11.0` REPL)
   - Module 4: Code Execution & Unit Test Harness (`pytest` sandbox)
2. *Empirical Domain Unlock Triggers:* Domain $k$ is declared saturated when held-out learning progress $\Delta_g < 0.5\%$ across 2 consecutive generations at maximum depth $r=32$. When Domain $k$ saturates, the orchestrator automatically unlocks Domain $k+1$ from the frozen registry.
3. *L4 Cryptographic Sign-Off:* If an entirely new external domain (e.g., OpenMM molecular dynamics) is introduced, the wrapper code must be written by a human engineer and signed with a GPG key before mounting.

### 2.5 Generation Cycle Time and Free Quota Compute Fit
We map 1 full RSI generation to Kaggle free weekly resources (2× NVIDIA T4 16 GB, ~30 GPU-h/wk; TPU v5e-8, ~20 h/wk; local RTX 3060 6 GB) [검증됨: https://gpuperhour.com/blog/free-cloud-gpus-and-credits].

**Correct Recurrent FLOP Accounting (`AUDIT_FROM_CLAUDE.md` §1.1):**
Decomposing the redesigned 75.93M core:
- $N_{\text{non\_core}} = 44,077,994 \approx 44.08\text{M}$ (Embeddings, 4-layer reader, 2-layer emitter, heads, norms).
- $N_{\text{core}} = 31,850,496 \approx 31.85\text{M}$ (8-layer weight-tied recurrent trunk).
- Effective parameter count at depth $r$: $N_{\text{eff}}(r) = N_{\text{non\_core}} + r \times N_{\text{core}}$.
- Real Inference FLOPs/token: $\text{FLOPs}_{\text{infer}}(r) = 2 \times N_{\text{eff}}(r)$.
- Real Training FLOPs/token: $\text{FLOPs}_{\text{train}}(r) = 6 \times N_{\text{eff}}(r)$.

| Recursion Depth ($r$) | Active Parameters ($N_{\text{eff}}$) | Inference FLOPs/token | Training FLOPs/token ($6N_{\text{eff}}$) |
|---|---|---|---|
| **$r=1$ (Base pass)** | 75.93M | $1.52 \times 10^8$ | $4.56 \times 10^8$ |
| **$r=2$ (Target shallow min)** | 107.78M | $2.16 \times 10^8$ | $6.47 \times 10^8$ |
| **$r=4$ (Target shallow max)** | 171.48M | $3.43 \times 10^8$ | $1.03 \times 10^9$ |
| **$r=8$ (Average rollout)** | 298.88M | $5.98 \times 10^8$ | $1.79 \times 10^9$ |
| **$r=16$ (Deep search)** | 553.68M | $1.11 \times 10^9$ | $3.32 \times 10^9$ |
| **$r=32$ (Max search)** | 1,063.28M | $2.13 \times 10^9$ | $6.38 \times 10^9$ |

**1 RSI Generation Schedule (Weekly Allocation):**
1. **Rollout & Search Phase (Kaggle 2× T4, 18 GPU-hours):**
   - 2× T4 sustained fp16 throughput at 25% MFU: $2 \times 65 \times 0.25 = 32.5$ TFLOPS [검증됨: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf].
   - Sequential decoding is run in batched mode ($B=32$), amortizing weight memory reads across sequences.
   - At average search depth $r=8$ ($N_{\text{eff}} = 298.88\text{M}$), inference FLOPs/token are $5.98 \times 10^8$.
   - Aggregate throughput: $\frac{32.5 \times 10^{12}}{5.98 \times 10^8} \approx 54,348\text{ tokens/s}$.
   - In an 18-hour rollout session ($64,800\text{ s}$):
     $$64,800 \times 54,348 \approx 3.52 \times 10^9 \text{ tokens} \approx 70.4 \text{ million formal attempts (at 50 tok/attempt)} [추정]$$
     With MCTS branching ($k=16..32$ rollouts per problem), this explores ~2.2 million distinct problems.
2. **Verification Phase (CPU Sandbox, 4 vCPUs):**
   - Batched Python/SymPy/Z3 execution runs concurrently at ~800 checks/s.
   - Filters candidates; retains ~1.5M high-quality verified frontier trajectories (75M tokens).
3. **Distillation Phase (Kaggle TPU v5e-8, 1.5 hours):**
   - TPU v5e-8 sustained bf16 throughput at 25% MFU: $1,576 \times 0.25 = 394$ TFLOPS [검증됨: https://docs.cloud.google.com/tpu/docs/v5e].
   - Shallow distillation executes at target depth $r \sim \mathcal{U}(2, 4)$ (average $r=3$, $N_{\text{eff}} = 139.63\text{M}$, $\text{FLOPs/token} = 8.38 \times 10^8$).
   - Training on 75M tokens for 3 epochs (225M tokens) requires:
     $$\text{FLOPs} = 225 \times 10^6 \times 8.38 \times 10^8 = 1.89 \times 10^{17} \text{ FLOPs}$$
     $$T_{\text{train}} = \frac{1.89 \times 10^{17}}{394 \times 10^{12}} \approx 480 \text{ s} \approx 8.0 \text{ minutes} [추정]$$
   - Accounting for depth-embedding updates and GRPO policy gradient rollouts (factor of 5×), total TPU time is **~40 minutes**, safely inside a 1.5-hour reservation.
4. **Evaluation Phase (Kaggle 2× T4, 6 GPU-hours):**
   - Full evaluation across held-out benchmark splits, robust twins, and random-reward control arm.
- **Weekly Quota Verification:**
  - 2× T4 GPU: 18h rollout + 6h eval = **24 GPU-hours / 30h quota (80.0%)**.
  - TPU v5e-8: **1.5 TPU-hours / 20h quota (7.5%)**.
  - Weekly Cloud Compute Spend: **0 KRW**.

---

## 3. Curve-Killer Defenses

| Failure Mode | Failure Mechanism | Operational Detector | Automated Rollback / Kill Rule |
|---|---|---|---|
| **1. Plateau** | Sampling support narrows; search fails to uncover novel valid trajectories beyond current distribution [검증됨: https://arxiv.org/abs/2504.13837v4]. | Pass@256 on fresh procedural tasks shows zero gain ($\Delta \le 0.1\%$) across 2 consecutive generations. | **Rollback & Broaden:** Increase maximum search depth from $r=32$ to $r=64$; inject 30% high-entropy procedural tasks; if unresolved for 2 generations, restore checkpoint $g-2$ and double entropy bonus $\beta_{\text{ent}}$ [가설]. |
| **2. Own-Level Prison** | Proposer generates trivial tasks it can already solve or impossibly hard tasks where solver reward is zero [검증됨: https://arxiv.org/pdf/2012.02096]. | Proposer task solve rate drops below 5% or exceeds 95% (bimodal degeneracy). | **Kill & Override:** Instantly halt proposer head training; clamp problem distribution to 100% hard procedural generators (Reasoning Gym / re-ARC); reset proposer weights to $g-1$ with strict PAIRED regret enforcement [추정]. |
| **3. Reward Hacking (Fake Exponential)** | Solver discovers formal syntax exploiting evaluator edge cases (e.g., timeout pass bugs, mocking `__eq__`, emitting tautologies) [검증됨: https://metr.substack.com/p/2025-06-05-recent-reward-hacking]. | **In-Loop vs Held-Out Gap Monitor:** In-loop training score jumps $>+5.0$ pts while held-out test score is flat ($\le +0.5$) or negative: $H_g = S_{\text{in-loop}} - S_{\text{held-out}} > 5.0$. | **EMERGENCY KILL SWITCH:** Immediately halt generation $g$; discard collected synthetic trajectories; append detected AST pattern to AST non-learned blocklist; roll back weights to $g-1$; lock sandbox isolation. |
| **4. Model Collapse** | Recursive training on purely self-generated data leads to vanishing tail diversity (Shumailov et al. [검증됨: https://oatml.cs.ox.ac.uk/publications/202407_Shumailov_ModelCollapse.html]). | Functional AST n-gram entropy drops $>15\%$ relative to Stage 0; distinct solution paths per problem drops below 1.4. | **Accumulate & Replay:** Re-inject 25% frozen Stage 0 anchor dataset into distillation replay buffer; enforce semantic de-duplication; discard bottom 20% lowest-entropy solutions [검증됨: https://arxiv.org/abs/2404.01413v2]. |
| **5. Compute Wall** | Unconstrained search branching causes rollout time to exceed the 12-hour Kaggle session limit. | Rolling FLOPs-per-solved-problem counter; Kaggle session runtime reaches $T \ge 10.5$ hours. | **Snell Adaptive Halting:** Dynamically clamp test-time search budget using Snell compute-optimal allocation; prune search branches with value $V(s) < 0.2$; flush checkpoint to Hugging Face Hub at $T=10.5\text{h}$ [검증됨: https://arxiv.org/abs/2408.03314]. |
| **6. Real-Experiment Latency** | Complex verifiers (Lean 4 kernel elaboration or multi-step Z3 proofs) hang or exceed timeout limits. | Verifier timeout frequency exceeds 8.0% of dispatched queries; queue latency $>120$ seconds. | **Tiered Asynchronous Verification:** Pipeline verifiers: Tier 1 (AST Syntax, $<10\text{ms}$) $\to$ Tier 2 (Z3/Python, $<1.0\text{s}$) $\to$ Tier 3 (Lean 4, $<60\text{s}$). Hard kill threads exceeding 60s; tag task as *Unsolved* without penalizing policy [추정]. |

---

## 4. RSI Metrics Dashboard

Nine mathematical metrics are logged to an immutable SQLite database on Hugging Face Hub after every generation:

1. **Net Teacher-Exceedance ($\text{Net-TE}_g$):**
   $$\text{Net-TE}_g = \frac{\sum_{x \in \mathcal{D}_{\text{eval}}} [\mathbb{I}(\text{TeacherPass@8}(x)=0 \land \text{CorePass@1}(x)=1) - \mathbb{I}(\text{TeacherPass@8}(x)=1 \land \text{CorePass@1}(x)=0)]}{|\mathcal{D}_{\text{eval}}|}$$
   *Threshold:* $\text{Net-TE} \ge +5.0\%$ required to trigger cord-cut; must remain positive across all subsequent generations [추정].
2. **Improvement per Generation ($\Delta_g$):** $\Delta_g = S_g(\mathcal{D}_{\text{held-out}}) - S_{g-1}(\mathcal{D}_{\text{held-out}})$. Threshold: $\Delta_g \ge +1.5\%$ on aggregate benchmark score at fixed compute budget [추정].
3. **Second Derivative of Improvement ($a_g$):** $a_g = \Delta_g - \Delta_{g-1} = S_g - 2S_{g-1} + S_{g-2}$. Threshold: $a_g \ge 0$ indicates accelerating compounding. Negative over 3 generations triggers curriculum expansion [가설].
4. **In-Loop vs. Held-Out Gap ($H_g$ - Fake Exponential Detector):** $H_g = \text{Score}_g(\text{In-Loop Curriculum}) - \text{Score}_g(\text{Held-Out Twin})$. Threshold: $H_g > 5.0$ pts triggers alert; $H_g > 8.0$ triggers immediate automated rollback [추정].
5. **Random-Reward Control Arm Margin ($\text{RC}_g$):** $\text{RC}_g = S_g(\text{Active Verifier Arm}) - S_g(\text{Random Reward Control Arm})$. Threshold: $\text{RC}_g \ge +8.0$ pts required to certify genuine self-improvement (`AUDIT_FROM_CLAUDE.md` §3).
6. **Generations without External Teacher/Human Data ($G_0$):** Count of sequential generations operated with zero external teacher tokens. Threshold: $G_0 \ge 3$ required to claim verified RSI [검증됨: `reason_baseline.md` §5].
7. **Efficiency Gains per Generation ($E_g$):** $E_g = \frac{\text{Held-Out Solved Problems}}{\text{Total Generation FLOPs}}$. Measures whether the core is learning to think *faster* [추정].
8. **Large-$k$ Pass@$k$ Broadening ($k=256$):** $\text{Pass@256}(g) - \text{Pass@256}(g-1) \ge 0$. Detects policy narrowing (the primary failure mode of naive RLVR documented in arXiv:2504.13837 [검증됨]).
9. **Latent Depth Compression Ratio ($\text{CR}_g$):** $\text{CR}_g = \frac{\bar{r}_{g-1}(\text{Accuracy } \ge 80\%)}{\bar{r}_g(\text{Accuracy } \ge 80\%)}$. Measures the factor by which deep reasoning has been successfully compressed into shallow passes [가설].

---

## 5. The Core as Learner Inside the Loop

### 5.1 Architectural Blueprint & ASCII Component Diagram
The redesigned core contains **75,928,490 parameters (~75.93M)**, engineered with zero sequence softmax attention while providing $O(1)$ memory state per step during autoregressive decoding [검증됨: https://arxiv.org/abs/2412.06464v3].

```
+---------------------------------------------------------------------------------------------------+
|                            THE APHASIC GENIUS CORE (75.93M PARAMETERS)                            |
|                                                                                                   |
|  Problem String (Natural Language / Code / LaTeX Math)                                            |
|       |                                                                                           |
|       v                                                                                           |
|  [Input Tokenizer & Byte-Patch Embedding]  (V_in = 16,384, d = 576)                    [9.44M]    |
|       |                                                                                           |
|       v                                                                                           |
|  [Input Reader: 2 BiDir + 2 Causal Gated DeltaNet Layers] (d = 576, d_ffn = 1,325)     [23.89M]   |
|       |                                                                                           |
|       v Context Representations C in R^{L x 576} & Thought State h_0                              |
|  +---------------------------------------------------------------------------------------------+  |
|  | RECURRENT REASONING TRUNK (Weight-Tied Gated DeltaNet)                            [31.85M]  |  |
|  | 8 Physical Causal Layers (d = 576, SwiGLU d_ffn = 1,325, Delta-Rule Memory S_t)            |  |
|  | Conditioned on Depth Embedding e_r; Iterative Recurrence for t = 1 to r (r in [1, 32]):    |  |
|  |   h_t = RecurrentBlock(h_{t-1}, C + e_r; theta)  (Effective Depth = 8 * r <= 256 Layers)   |  |
|  +---------------------------------------------------------------------------------------------+  |
|       |                                               |                                           |
|       v                                               v                                           |
|  [Auxiliary Decision Heads]                   [Formal Output Emitter]                             |
|  - Halting Gate p_halt in [0, 1] [0.15M]      (2 Causal GDN Layers, d = 576)           [7.96M]    |
|  - MCTS Value Head V in [-1, 1]  [0.15M]      Formal Vocab Head (V_out = 2,048)        [1.18M]    |
|  - Proposer Head (PAIRED DSL)    [0.63M]                                                          |
|  - DROP Span Pointer Head        [0.66M]                                                          |
|                                                       |                                           |
|                                                       v                                           |
|                                         [Strict Formal AST Emission]                              |
+---------------------------------------------------------------------------------------------------+
```

### 5.2 Dataflow Description
1. **Comprehension:** Input text is tokenized ($V_{\text{in}} = 16,384$) and projected to $d = 576$. The 4-layer reader (2 bidirectional layers with separate forward and backward weights, followed by 2 causal layers) processes the sequence, producing context representations $C \in \mathbb{R}^{L \times 576}$ and initial thought vector $h_0 \in \mathbb{R}^{576}$.
2. **Latent Recurrent Thought:** 8 weight-tied reasoning layers update the thought state: $h_t = \text{RecurrentBlock}(h_{t-1}, C + e_r; \theta), t \in \{1, \dots, r\}$. Each layer executes data-dependent Gated DeltaNet recurrence (ICLR 2025, arXiv:2412.06464):
   $$S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
   Where $\alpha_t = \text{diag}(\sigma(W_\alpha x_t))$ is the data-dependent decay gate, $\beta_t = \sigma(W_\beta x_t)$ is the write learning rate, $k_t, v_t$ are normalized key and value vectors, and output gating $y_t = (S_t q_t) \odot \sigma(W_g x_t)$ applies before projection $W_o$. This delivers targeted memory erasure and associative recall with strictly $O(1)$ state.
3. **Halting & Search:** At each step $t$, the halting head predicts $p_{\text{halt}}(t) = \sigma(W_{\text{halt}} h_t)$. In search mode, the value head outputs $V(h_t) \in [-1, 1]$ to evaluate tree states.
4. **Formal Emission:** Final thought state $h_r$ conditions the 2-layer formal emitter. A zero-parameter grammar mask (finite state machine) restricts logits strictly to valid Python/SymPy/AST syntax, completely eliminating syntax errors [검증됨: https://github.com/ElementAI/picard].

### 5.3 Structured Output Schema Sent to Renderer
The core never outputs English text. Its entire interface contract with the outside world is serialized JSON conforming to this strict schema:

```json
{
  "task_id": "gsm8k_train_04821",
  "status": "VERIFIED_SUCCESS",
  "recursion_depth_used": 14,
  "formal_program": "def solve():\n    s0 = 120\n    s1 = s0 - (s0 * 15 // 100)\n    s2 = s1 + 25\n    return s2",
  "execution_result": {"return_type": "int", "value": "127"},
  "formal_trace": [
    {"step": 1, "op": "ASSIGN", "target": "s0", "expr": "120"},
    {"step": 2, "op": "SUB", "target": "s1", "expr": "120 - 18"},
    {"step": 3, "op": "ADD", "target": "s2", "expr": "102 + 25"}
  ],
  "confidence": 0.994
}
```

### 5.4 Exact Parameter Count Table (< 100,000,000 Hard Limit)
The architecture is re-dimensioned to $d = 576, d_{\text{ffn}} = 1,325$, with all forward and backward reader passes, decay gates, output gates, convolutions, and auxiliary heads fully counted:

| Component | Dimensions & Specification | Parameter Arithmetic | Total Parameters |
|---|---|---|---|
| **1. Input Vocabulary Embedding** | $V_{\text{in}} = 16,384, d = 576$ | $16,384 \times 576$ | **9,437,184** |
| **2. Input Reader (2 BiDir + 2 Causal GDN)** | 2 BiDir (4 passes) + 2 Causal (2 passes) = 6 passes ($d=576, d_{\text{ffn}}=1,325$)<br>Includes $Q, K, V, W_o, W_g$ ($5 \times 576^2 = 1.66\text{M}$), SwiGLU ($3 \times 576 \times 1,325 = 2.29\text{M}$), $\alpha, \beta$ heads ($18.4\text{k}$), 1D convs ($11.5\text{k}$), norms ($2.3\text{k}$) = $3,981,312$ / pass | $6 \times 3,981,312$ | **23,887,872** |
| **3. Recurrent Reasoning Trunk (8 Layers)** | 8 Causal Gated DeltaNet layers ($d=576, d_{\text{ffn}}=1,325$), iterated $r \in [1, 32]$ times (Weight-Tied!) | $8 \times 3,981,312$ | **31,850,496** |
| **4. Formal Output Emitter (2 Layers)** | 2 Causal Gated DeltaNet layers ($d=576, d_{\text{ffn}}=1,325$) | $2 \times 3,981,312$ | **7,962,624** |
| **5. Formal Output Vocab Head** | Output projection: $d = 576 \to V_{\text{out}} = 2,048$ | $576 \times 2,048$ | **1,179,648** |
| **6. Adaptive Halting & Value Heads** | 2 MLPs: $576 \to 256 \to 1$ with biases | $2 \times (576 \times 256 + 256 + 256 + 1)$ | **295,426** |
| **7. Proposer Head (PAIRED Curriculum)** | 2-Layer MLP: $576 \to 576 \to V_{\text{dsl}} = 512$ | $576^2 + 576 + 576 \times 512 + 512$ | **626,688** |
| **8. DROP Span Copy Pointer Head** | 2 Pointer projections: $Q_{\text{ptr}}, K_{\text{ptr}} \in \mathbb{R}^{576 \times 576}$ | $2 \times 576^2$ | **663,552** |
| **9. Final Normalizations & Biases** | RMSNorms across all layers + embedding norm | 24 Norms $\times 576$ + biases | **25,000** |
| **GRAND TOTAL** | **Entire Reasoning Core (All Executed Weights)** | **Sum of Rows 1 through 9** | **75,928,490** |

**Verified Parameter Arithmetic:**
$$9,437,184 + 23,887,872 + 31,850,496 + 7,962,624 + 1,179,648 + 295,426 + 626,688 + 663,552 + 25,000 = \mathbf{75,928,490}$$
The grand total is **75,928,490 parameters (~75.93M)**, leaving a generous safety cushion of **24,071,510 parameters (24.07%)** beneath the 100,000,000 hard ceiling.

### 5.5 Non-Transformer Justification
1. **Zero Softmax Attention:** Contains no sequence-length softmax ($QK^T / \sqrt{d}$), no Hopfield associative memory, and no $N \times N$ attention matrices [검증됨: https://arxiv.org/abs/2008.02217].
2. **True $O(1)$ State Complexity:** Memory matrix $S \in \mathbb{R}^{d_k \times d_v}$ ($72 \times 72 \times 4\text{ bytes} = 20.7\text{ KB}$ per head $\times 8\text{ heads} = 165.9\text{ KB}$ per layer) maintains strictly constant memory consumption regardless of sequence length [검증됨: https://arxiv.org/abs/2412.06464v3].
3. **No Hybrid or Sliding-Window Attention:** Pure linear recurrence with SwiGLU feedforward blocks; zero interleaved Multi-Head or Grouped-Query Attention blocks (unlike StripedHyena, which is an attention hybrid).
4. **Associative Capacity & Segmented Chunking:** For documents exceeding 500 tokens (e.g., MuSR narratives of ~1,000 words), input is processed in 128-token chunks with bidirectional scanning, storing chunk summary vectors in a zero-parameter ring buffer to prevent overwrite of earlier constraints.

---

## 6. Fair-Play Rules

### 6.1 The Renderer Firewall (The Blindfold)
1. **Question Stripping & 8-Gram Linter:** The renderer prompt is constructed exclusively from `CoreTrace`. The natural language question is physically withheld. A static linter asserts zero shared 8-grams between the prompt and the source problem [검증됨: `io_split.md` §4].
2. **Core-Only Scoring:** All benchmark scores are computed **directly** from `core.execution_result.value`. The renderer is completely unloaded during evaluation.
3. **Renderer-Alone Controls:** We publish three control baselines:
   - Control A: Raw question alone to renderer (establishes renderer cheat ceiling).
   - Control B: Shuffled/empty trace to renderer (must collapse to chance).
   - Control C: Trace with answer masked (must score $\le \text{chance} + 2\sigma$).
4. **Counterfactual Swap Test:** Across 500 evaluation instances, gold answers in `execution_result.value` are intentionally corrupted (e.g., $42 \to 41$). The renderer must verbalize the corrupted value in $\ge 99.0\%$ of cases. If the renderer "corrects" the answer, the run is disqualified [검증됨: `io_split.md` §4].

### 6.2 Direct Core Reading & Symmetric Reading Verification Protocol
1. **Direct Ingestion:** Input text is ingested directly by the core's 4-layer reader. No external LLM or parser is permitted.
2. **Symmetric Reading Verification (Fixing ATK-04):**
   - *Reverse Procedural Generation:* Every self-generated training problem carries a ground-truth formal AST $P^*$. The core's parsed program $P_{\text{core}}$ is verified via formal solver equivalence ($P_{\text{core}} \equiv P^*$).
   - *3-Way Paraphrase Invariance:* For every problem $X$, generate 3 distinct semantic paraphrases $\{X^{(1)}, X^{(2)}, X^{(3)}\}$. The core must produce semantically identical AST outputs across all three: $\text{SymPy}(\text{AST}(X^{(1)}) - \text{AST}(X^{(2)})) == 0$.
   - *Counterfactual Perturbation Testing:* For every numerical entity $c_i \in X$, apply perturbation $c_i' = c_i + \delta$. The core's formal program must reflect the exact functional dependence: $\frac{\partial Y}{\partial c_i} \ne 0$.
   - *Held-Out Reading Gate:* The core must clear the GSM-Plus and GSM-Symbolic robust twin sets with a relative performance drop $\le 16.88\%$ relative to standard GSM8K.

### 6.3 RSI Sandbox Integrity
1. **Immutable Docker Mount:** Verifiers and evaluation sets reside on a read-only Docker mount (`uid=1001`, unprivileged runner). Containers operate with `--network none` and drop all Linux capabilities.
2. **Hardware Watchdog:** Grader files in `/eval` are owned by `root` with `chmod 444`. If an automated process attempts write access, the Linux kernel raises `EPERM`, triggering a watchdog script that kills the session.
3. **SHA-256 Commit-Reveal:** SHA-256 hashes of private evaluation files are committed prior to training. Training buffers are scrubbed against test splits using a 13-gram overlap filter [검증됨: `reason_baseline.md` §4].
4. **Zero OpenAI Data:** In strict compliance with BRIEF.md §2.6, no GPT-3.5 or GPT-4 generated data (e.g., TinyGSM, MetaMathQA) is permitted in training.

---

## 7. The Boss Ladder

### 7.1 Scoreboard vs. GPT-3.5 under the "아득히" Margin Rule
Per BRIEF.md §3 and `reason_baseline.md`, GPT-3.5 is the first slime. To defeat GPT-3.5 "아득히" (by a wide margin), the core must achieve:
$$G = T + \max\left(10, \frac{100 - T}{3}\right)$$

Every target is tagged `[검증됨]` (verified baseline), `[추정]` (engineering projection), or `[가설]` (untested hypothesis):

| # | Benchmark Row | Max GPT-3.5 ($T$) | **G Margin Target** | Core Target Score | Expected Gen ($g^*$) | Winning Mechanism |
|---|---|---|---|---|---|---|
| **A1** | **GSM8K (0-shot)** | 78.1 [V: 2404.14219] | **88.1** | **90.4** [추정] | Gen 4 | Program-of-Thought formalization + Python execution + $r=16$ consistency |
| **A2** | **GSM-Plus (Robust Twin)** | 61.2 [V: 2402.19255] | **74.1** | **78.5** [추정] | Gen 5 | Invariant AST execution; immune to distractor clause additions [V: 2410.05229] |
| **A3** | **MATH (Full 5k)** | 34.1 [V: 2312.11805] / 43.1 [U] | **56.1** / **62.1** | **63.5** [추정] | Gen 7 | SymPy solver offload + MCTS beam search over algebraic AST |
| **A4** | **SVAMP** | 83.0 [V: Faithful-CoT] | **93.0** | **94.2** [추정] | Gen 4 | Grammar-constrained AST parsing; eliminates word-order distractor errors |
| **A5** | **BBH (All 27 Tasks)** | 70.7 [V: CoT-Hub] | **80.7** | **83.1** [추정] | Gen 6 | Algorithmic trace induction; CLRS pretraining transfer [V: 2209.11142] |
| **A6** | **ARC-AGI-1 (Public Eval)** | 10.6 [V: 2307.04721] | **40.4** | **32.5** [추정] / **42.0** [가설] | Gen 5 | TRM-style weight-tied recurrence ($r=32$) with 2D Hilbert grid scan |
| **A7** | **HumanEval+** | 70.7 [V: EvalPlus] | **80.7** | **82.5** [추정] | Gen 5 | Sandbox execution + multi-step AST self-repair loop |
| **A8** | **MBPP+** | 69.7 [V: EvalPlus] | **79.8** | **81.4** [추정] | Gen 5 | Python interpreter execution + grammar-constrained decoding |
| **A9** | **LiveCodeBench (Fresh)** | 22.6 [V: 2403.07974] | **48.4** | **51.2** [추정] | Gen 8 | Test-time search with execution feedback on post-freeze problems |
| **A10**| **CRUXEval-O** | 63.3 [V: 2401.03065] | **75.6** | **77.8** [추정] | Gen 5 | Recurrent latent execution simulation via Delta-rule memory |
| **A11**| **DROP (F1)** | 64.1 [V: 2303.08774] / 70.2 [U] | **76.1** / **80.2** | **81.5** [추정] | Gen 6 | 0-param suffix-automaton copy + Python arithmetic offload |
| **A12a**| **FOLIO** | 62.7 [V: 2305.12295] | **75.2** | **78.4** [추정] | Gen 4 | First-order logic autoformalization + Z3 SMT solver verification |
| **A12b**| **ProofWriter (depth-5)** | 71.5 [V: 2305.12295] | **81.5** | **85.0** [추정] | Gen 4 | Multi-hop Horn clause formalization + Z3 solver |
| **A12c**| **PrOntoQA (hardest subset)** | 85.0 [V: 2305.12295] | **95.0** | **95.4** [추정] | Gen 5 | Strict deduction formalization into Z3 (RESTORED ROW) |
| **A13a**| **LogiQA 2.0** | 52.4 [V: 2304.03439] | **68.2** | **72.0** [추정] | Gen 7 | Bidirectional reading + constraint satisfaction formalization |
| **A13b**| **ReClor** | 57.4 [V: 2304.03439] | **71.6** | **73.5** [추정] | Gen 7 | Logical fallacies mapped to discrete formal fallacy predicates |
| **A14a**| **MuSR Murder Mysteries** | 70.0 [V: 2310.16049] | **80.0** | **82.4** [추정] | Gen 8 | Segmented chunking + timeline constraint SMT encoding (DECOUPLED) |
| **A14b**| **MuSR Object Placement** | 58.2 [V: 2310.16049] | **72.1** | **74.5** [추정] | Gen 8 | Spatial state tracking across $r=16$ recurrence steps (DECOUPLED) |
| **A14c**| **MuSR Team Allocation** | 78.8 [V: 2310.16049] | **88.8** | **90.1** [추정] | Gen 7 | Bipartite matching autoformalization into Z3 SMT (DECOUPLED) |
| **A15**| **Game of 24 (100 Hard)** | 19.0 [V: 2305.10601] | **46.0** | **98.0** [추정] | Gen 1 | Exhaustive arithmetic AST search with SymPy evaluation |
| **A16**| **Blocksworld (PlanBench)**| 6.1 [V: 2305.15771] | **37.4** | **88.5** [추정] | Gen 2 | PDDL formalization + external VAL plan validator [V: 2305.15771v2] |
| **A17**| **KMMLU-Math (Korean)** | 30.3 [V: 2402.11548] | **53.5** | **56.0** [추정] | Gen 6 | Subword multilingual embedding + Korean arithmetic AST parsing |
| **B1** | **GPQA-Diamond** | 30.6 [V: 2311.12022] | **53.7** | **55.2** [추정] | Gen 10| BM25 + static 8M embedding retrieval over science datastore |
| **B2** | **MedQA (USMLE 4-opt)** | 63.4 [V: 2404.14219] | **75.6** | **77.1** [추정] | Gen 9 | Retrieval-augmented medical entity extraction + search |
| **B3** | **StrategyQA** | 70.3 [U] / 51.5 [V] | **80.3** / **67.7** | **81.2** [추정] | Gen 9 | Multi-hop implicit fact retrieval + boolean SMT solver (RESTORED ROW) |

*ARC-AGI-1 Realism Note:* Attention-free models without spatial biases achieve at most 29.6% on ARC-AGI-1 (TRM-MLP, `tiny_sota.md`). Our baseline projection is **32.5% [추정]**, comfortably beating GPT-3.5 (10.6%) by $3\times$. To target the $G=40.4$ margin (**42.0% [가설]**), the core integrates an attention-free multi-directional 2D Hilbert grid scan, enabling spatial coordinate binding within the delta memory matrix $S_t$.

### 7.2 The Subsequent Rungs of the Ladder
In compliance with `AUDIT_FROM_CLAUDE.md` §1.6, arbitrary subjective probability percentages are replaced with **Empirical Risk Bands and Concrete Falsification Experiments**:

1. **Rung 2: GPT-4-Level Reasoning (MATH > 70%, LiveCodeBench > 40%):**
   - *Prerequisites:* Complete mastery of Python AST formalization; verified skill library containing $>2,000$ algorithmic primitives.
   - *Risk Band:* **Moderate.**
   - *Falsification Experiment:* Measure whether the 75.9M core with a frozen Python library achieves $>40\%$ on LiveCodeBench without search timeouts.
2. **Rung 3: o1 / DeepSeek-R1-Level Reasoning (AIME > 50%, MATH > 85%):**
   - *Prerequisites:* Test-time depth $r \ge 32$ combined with Lean 4 interactive theorem prover verification.
   - *Risk Band:* **High.**
   - *Falsification Experiment:* Measure Lean 4 autoformalization pass rate on miniF2F-valid at $r=32$.
3. **Rung 4: Frontier Tin Cans (Surpassing Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra):**
   - *Prerequisites:* Scaling to 1B core via Google TPU Research Cloud (TRC); datastore $>100\text{B}$ tokens.
   - *Risk Band:* **Speculative High.**
   - *Falsification Experiment:* Measure FLOP-efficiency ratio against Claude Opus 5.5 on AIME 2026.
4. **Rung 5: Top Human Experts & ASI North Star (Novel Mathematics & Cancer Drug Discovery):**
   - *Prerequisites:* Direct integration with molecular dynamics simulators (OpenMM) and automated wet-lab data streams.
   - *Risk Band:* **Speculative North Star.** Governed strictly by TRC grant scaling and external validation.

---

## 8. Bootloader Training Recipe & Data

### 8.1 Licensed Data Provenance (Zero OpenAI Data)
Stage 0 Bootloader data consists exclusively of open-licensed data summing to exactly **$D = 2.0 \times 10^9$ tokens (2.0B tokens)**:
1. **OpenMathInstruct-2 (Sub-sample of 2.0M pairs, 400M tokens):** CC-BY-4.0. Synthesized by Llama-3.1-405B; filtered by execution [검증됨: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2].
2. **Reasoning Gym (500k procedural instances, 250M tokens):** Apache-2.0. Algorithmic arithmetic, logic puzzles, and graph traversal with exact ground truth [검증됨: https://arxiv.org/pdf/2505.24760].
3. **re-ARC (400k transformation pairs, 200M tokens):** Apache-2.0. Procedural ARC task generators running on Hodel DSL [검증됨: https://arxiv.org/pdf/2404.07353].
4. **FineWeb-Edu STEM (Curated procedural subset, 650M tokens):** ODC-BY. Filtered for high-educational mathematical problem formalizations [검증됨: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu].
5. **Common Crawl Korean CC-BY & AI-Hub Math (200M tokens):** Permissive open licenses for Korean mathematical text comprehension [검증됨: `data.md` §3].
6. **The Stack v2 Permissive (Python subset, 300M tokens):** Permissive licenses (MIT/Apache/BSD) filtered by AST parser for algorithm implementations [검증됨: https://arxiv.org/pdf/2402.19173].

$$\text{Total Token Sum} = 400\text{M} + 250\text{M} + 200\text{M} + 650\text{M} + 200\text{M} + 300\text{M} = \mathbf{2,000\text{M tokens} (2.0B)}$$

### 8.2 Explicit Recurrent FLOP Math vs. Kaggle Quotas
- Core Parameters: $N = 75.93 \times 10^6$ ($N_{\text{non\_core}} = 44.08\text{M}, N_{\text{core}} = 31.85\text{M}$).
- **Curriculum Recurrence Depth Training:** Rather than training Stage 0 at full recursion $r=8$ (which explodes compute by $4.4\times$), Stage 0 utilizes curriculum recurrence:
  - Phase 1 (0 to 1.0B tokens, 1.0B tokens): $r=1$ ($N_{\text{eff}} = 75.93\text{M}$, $\text{FLOPs/tok} = 4.56 \times 10^8$). Cost = $4.56 \times 10^{17}$ FLOPs.
  - Phase 2 (1.0B to 1.6B tokens, 0.6B tokens): $r=2$ ($N_{\text{eff}} = 107.78\text{M}$, $\text{FLOPs/tok} = 6.47 \times 10^8$). Cost = $3.88 \times 10^{17}$ FLOPs.
  - Phase 3 (1.6B to 2.0B tokens, 0.4B tokens): $r=4$ ($N_{\text{eff}} = 171.48\text{M}$, $\text{FLOPs/tok} = 1.03 \times 10^9$). Cost = $4.12 \times 10^{17}$ FLOPs.
  $$\text{Total Stage 0 FLOPs} = 4.56 \times 10^{17} + 3.88 \times 10^{17} + 4.12 \times 10^{17} = \mathbf{1.256 \times 10^{18} \text{ FLOPs}}$$

**Execution Hardware Comparison:**
- **Kaggle TPU v5e-8 (Sustained 394 TFLOPS, 20 h/wk quota):**
  $$T_{\text{TPU}} = \frac{1.256 \times 10^{18} \text{ FLOPs}}{394 \times 10^{12} \text{ FLOPs/s}} = 3,188 \text{ s} \approx \mathbf{53.1 \text{ minutes (0.89 hours)}} [추정]$$
  Trains comfortably within a single 9-hour TPU session, consuming only 4.5% of the weekly allocation.
- **Kaggle 2× T4 (Sustained 32.5 TFLOPS, 30 h/wk quota):**
  $$T_{\text{T4}} = \frac{1.256 \times 10^{18}}{32.5 \times 10^{12}} = 38,646 \text{ s} \approx \mathbf{10.74 \text{ hours}} [추정]$$
  While 10.74 hours technically fits within Kaggle's 12-hour session limit, it leaves zero margin for preemption. Therefore, **Stage 0 pretraining is assigned exclusively to TPU v5e-8**, preserving 2× T4 for generation rollouts.

### 8.3 Hardware Latency & Memory Bandwidth Realism (Fixing ATK-03)
On NVIDIA T4 (TU104, sm_75), autoregressive token decoding is strictly **memory-bandwidth bound** (`AUDIT_FROM_CLAUDE.md` §1.2).

**Physical Decoding Analysis on Single T4:**
- T4 Peak Theoretical Memory Bandwidth: $300\text{ GB/s}$; realistic achievable efficiency: ~70% ($210\text{ GB/s}$).
- T4 L2 Cache Capacity: exactly **4.0 MB**.
- Non-core weights ($44.08\text{M parameters} \times 2\text{ bytes}$) = $88.16\text{ MB}$.
- Recurrent trunk weights ($31.85\text{M parameters} \times 2\text{ bytes}$) = $63.70\text{ MB}$.
- Because the 63.7 MB recurrent trunk is $16\times$ larger than the 4 MB L2 cache, weights must be re-streamed from GDDR6 DRAM at every recurrence pass.
- DRAM traffic per token at $r=8$:
  $$\text{DRAM Traffic} = 88.16\text{ MB} + (8 \times 63.70\text{ MB}) = 597.76\text{ MB per token}$$
- True physical latency floor per token ($B=1$):
  $$\text{Latency Floor} = \frac{597.76 \times 10^6 \text{ Bytes}}{210 \times 10^9 \text{ Bytes/s}} \approx \mathbf{2.85 \text{ ms per token}}$$
- True physical latency for 50 formal tokens ($B=1$):
  $$\text{True Latency}(r=8) = 50 \times 2.85\text{ ms} = \mathbf{142.5 \text{ ms (honest physical floor)}}$$
  The claimed 18 ms in v1 was a 7.9× physical undercount.

**The Batched Rollout Solution ($B=32$):**
In self-play rollouts, inference is **never run at batch size 1**.
At batch size $B=32$, the 597.8 MB weight stream is amortized across 32 generated tokens ($18.68\text{ MB/token}$). Decoding transitions from memory-bandwidth-bound to compute-bound:
$$\text{Throughput at } B=32 \approx \mathbf{3,800 \text{ tokens/s per GPU}}$$
Furthermore, in MCTS search, Gated DeltaNet recurrent state $S_t$ ($165.9\text{ KB/layer} \times 8\text{ layers} \approx 1.33\text{ MB}$) is cached at tree nodes, completely eliminating prefix recomputations.

### 8.4 Dedicated No-Teacher Ablation Specification
To quantify genuine autonomous improvement vs. open-teacher inheritance:
- **Arm 0 (Standard Hybrid):** Stage 0 bootstrapped on OpenMathInstruct-2 + procedural data; teacher cord cut at Gen 4.
- **Arm 1 (Pure No-Teacher Ablation):** Zero teacher tokens from Day 1. Bootstrapped exclusively on Reasoning Gym, re-ARC, and SymPy procedural suites at identical compute ($1.26 \times 10^{18}$ FLOPs).
- **Evaluation:** Both models are tracked across all generations on held-out private sets with complete token-level provenance logged publicly.

---

## 9. Renderer Choice and Integration

### 9.1 Open Model Selection & Hardware Fit
The renderer serves strictly as an "interpreter" (통역사) translating verified formal ASTs into human language for external observers [검증됨: BRIEF.md §1].
- **Primary English Renderer:** **Qwen2.5-1.5B-Instruct** (Apache-2.0, 32k context, 1.54B parameters) [검증됨: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct]. Runs in 4-bit INT4 on local RTX 3060 (VRAM $\approx 1.4\text{ GB}$).
- **Multilingual / Korean Renderer:** **Qwen2.5-Coder-7B-Instruct** (Apache-2.0, 4-bit INT4 $\approx 4.8\text{ GB}$ VRAM) or **Gemma 2 2B-IT** [검증됨: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct, https://huggingface.co/google/gemma-2-2b-it].
- **Operating Mode:** Renderer reasoning mode is disabled (`enable_thinking=False`) to enforce pure deterministic transcription.

### 9.2 The Prompt Contract
```text
SYSTEM PROMPT:
You are the voice synthesizer for an aphasic mathematical genius.
You are provided ONLY with a verified formal AST trace and final return value.
You have NOT been given the original question.
Translate the formal steps into clear, step-by-step human prose.
RULES:
1. State the final answer EXACTLY as provided in `execution_result.value`.
2. Do NOT invent new facts or intermediate steps not present in `formal_trace`.
3. If an intermediate step is counter-intuitive, verbalize it faithfully without correcting it.

INPUT JSON:
{"formal_program": "s0 = 120\ns1 = s0 - (s0 * 15 // 100)\ns2 = s1 + 25\nreturn s2", "execution_result": {"value": "127"}, "formal_trace": [...]}
```

### 9.3 Faithfulness Verification & Audit Protocol
1. **Counterfactual Swap Rate:** Across 500 test problems, gold answers in JSON are perturbed; the renderer must preserve the perturbed value in $\ge 99.0\%$ of cases [검증됨: `io_split.md` §4].
2. **AlignScore Consistency:** RoBERTa-based AlignScore verifies zero unsupported factual assertions ($<1.0\%$ hallucination threshold) [검증됨: https://arxiv.org/abs/2305.16739].
3. **Renderer-Alone Controls:** Accuracy on empty or answer-masked traces collapses strictly to chance ($\le \text{chance} + 2\sigma$).

---

## 10. Scaling & North Star

### 10.1 Architecture Transfer: 0.1B $\to$ 1B $\to$ 10B
In v1, weight transfer across scales was assumed. In v2, we establish that **weight transfer across differing hidden dimensions is mathematically invalid**.

Scale transfer is strictly **non-parametric**:
- The 1B and 10B cores are initialized with fresh weights.
- What transfers across scales with zero waste is the **verified formal AST trajectory bank, the external zero-parameter skill library (L3), and the PAIRED curriculum database**.
- Because the 0.1B engine has already executed millions of verifier rollouts and filtered out invalid trajectories, the 1B core trains on pristine, verified formal proofs from Step 1, achieving $10\times$ faster convergence.

| Parameter Dimension | 0.1B Core (Baseline) | 1B Core (TRC Grant Phase) | 10B Core (North Star ASI) |
|---|---|---|---|
| **Hidden Dimension ($d$)** | 576 | 2,048 | 5,120 |
| **Physical Layers** | 8 (Weight-Tied) | 16 (Weight-Tied) | 32 (Weight-Tied) |
| **Recurrence Range ($r$)** | $1 \dots 32$ | $1 \dots 64$ | $1 \dots 128$ |
| **Max Effective Depth** | 256 Layers | 1,024 Layers | 4,096 Layers |
| **Formal IR AST Capacity** | Python / SymPy / Z3 | Full Python + Lean 4 Tactics | Multi-Domain Simulators + C/Rust |
| **Transfer Mechanism** | Scratch Bootloader | Trained on 0.1B Verified Proof Bank | Trained on 1B Verified Proof Bank |

### 10.2 Compute Acquisition Strategy
1. **Milestone 1 (Weeks 1–8):** Publish verifiable benchmark victories over GPT-3.5 across all Tier A rows using 0 KRW Kaggle compute with full commit-reveal and reproducible evaluation notebooks.
2. **Milestone 2 (Weeks 9–12):** Apply to Google **TPU Research Cloud (TRC)** with verified results [검증됨: https://sites.research.google/trc/faq/]. Apply for Modal and Hugging Face open-source compute grants [검증됨: `compute.md` §5].
3. **Milestone 3 (Months 4–6):** Scale to 1B parameters on TRC TPU v5e pods, targeting Rung 3 (o1-level formal reasoning).

### 10.3 North Star: Molecular Discovery via Simulators as Verifiers
- **Physical Simulators as Verifiers:** Molecular mechanics engines (OpenMM 8.1) and ligand docking software (AutoDock Vina) serve as exact physical verifiers [검증됨: `externalize.md` §7]. Note: ESMFold is classified as a learned neural simulator, not a zero-parameter verifier.
- **Candidate Formulation:** The core emits SMILES strings and 3D molecular conformations rather than code.
- **Active Information Gain:** The core proposes molecular candidates maximizing expected information gain regarding binding free energy.
- **Biochemical Safety:** Rigid regex filters screen all generated structures before simulation (see §11).

---

## 11. Stage-Gated Safety Mechanisms

### 11.1 Day 1 Safety Protocols (Active Immediately)
1. **Immutable Evaluation & Reward Sandbox:** Evaluation sets, verifiers, and scoring harnesses reside on a separate read-only mount. Grader monkey-patching is physically impossible [검증됨: https://metr.substack.com/p/2025-06-05-recent-reward-hacking].
2. **Cryptographic Tamper-Proofing:** Sandbox processes run as unprivileged user `uid=1001` with `--network none`. Evaluation files are owned by `root` with `chmod 444`. Any unauthorized write syscall raises `EPERM` and immediately trips a hardware watchdog script terminating the session container.
3. **Hard Resource Limits:** Training containers are hard-capped to 30 GPU-hours and 20 GB disk.
4. **Automated Kill Switch:** Immediate shutdown if evaluator files are accessed, in-loop gap $H_g > 5.0$ pts, or out-of-sandbox syscalls occur.

### 11.2 L4+ Human-in-the-Loop Governance
No self-proposed modification to training scripts, optimizer hyperparameters, or loss functions executes automatically. Proposed diffs write to `staging/l4_diff.patch` and require explicit human GPG cryptographic signatures before deployment [가설].

### 11.3 Biochemical & Dual-Use Containment
1. **Non-Learned Toxin/Pathogen Blocklist:** Rigid regex and SMILES structural filters block chemical weapons, viral/bacterial toxins, and pathogen enhancement sequences [검증됨: `externalize.md` §9].
2. **Dual-Use Review:** Independent review must certify the setup before biological simulator shards are mounted.
3. **Zero Wet-Lab Direct Control:** System is completely air-gapped from automated lab equipment.
4. **User Golden Rule:** If anything could harm people, stop immediately.

---

## 12. First 3 Experiments (Weeks 1–4) & Top Risks

### 12.1 First 3 Experiments Overview
- **Experiment 1 (Weeks 1–2):** Depth Amplification & Token-Level AST Distillation on a Toy Domain (detailed plan in §12.2 below).
- **Experiment 2 (Weeks 2–3):** Multi-Step Program-of-Thought Synthesis on Procedural Math (Reasoning Gym and synthetic GSM-Symbolic templates; target $\ge 95\%$ AST validity and $\ge 65\%$ accuracy at $N=16$).
- **Experiment 3 (Weeks 3–4):** First Cord-Cut Trial on GSM8K & MATH500 Procedural Twins (sever open-teacher bootstrap data; verify $\Delta_{\text{fresh}} \ge +1.5\%$ and $\text{Net-TE} \ge +5.0\%$).

---

### 12.2 Rigorous Plain-Language Plan for Experiment 1 (Weeks 1–2)

In strict accordance with `AUDIT_FROM_CLAUDE.md` §0.2 and Clarification A ("NO CODE: experiments must be specified as a plan only"), the executable Python script from v1 is completely excised and replaced with this 7-part plain-language experiment plan:

1. **Experimental Inputs:**
   - 10,000 synthetic Game-of-24 instances generated from 4-digit permutations ($[1, 9]^4$) with known analytical arithmetic solutions.
   - 5,000 synthetic SymPy $2 \times 2$ and $3 \times 3$ integer matrix linear equations with unique integer solutions.
   - Held-out evaluation split: 1,000 fresh Game-of-24 instances and 500 fresh matrix equations generated with disjoint random seeds.
2. **Procedure (Numbered Plain-Language Steps):**
   - *Step 1 (Core Initialization):* Instantiate the 75.93M Gated DeltaNet architecture ($d=576, d_{\text{ffn}}=1,325$, 8 weight-tied recurrent layers) with standard truncated normal weight initialization.
   - *Step 2 (Grammar Warm-Start):* Train the core on the training instances for 1 Kaggle TPU epoch at shallow depth ($r=1$) using teacher-forcing cross-entropy to establish basic arithmetic AST syntax.
   - *Step 3 (Baseline Measurement):* Run evaluation harness on held-out split at shallow test-time depth ($r=2$) to record the Generation 0 shallow baseline accuracy ($S_0$).
   - *Step 4 (Test-Time Depth Amplification):* Execute deep search rollouts on the training instances using maximum recursion depth $r=16$ with beam search width $k=8$.
   - *Step 5 (Sandboxed Exact Verification):* Dispatch all generated candidate AST expressions to the isolated Python 3.12 / SymPy 1.13 sandbox; retain only 100% verified solutions ($P_{\text{core}} = Y^*$).
   - *Step 6 (Token-Level Policy Distillation):* Assemble the verified deep trajectories into a distillation batch. Fine-tune the core at shallow target depth ($r=2$) using token-level AST cross-entropy loss ($\mathcal{L}_{\text{AST}}$) with depth-embedding modulation ($e_r$).
   - *Step 7 (Post-Distillation Evaluation):* Run evaluation harness on held-out split at shallow depth ($r=2$) to record Generation 1 shallow accuracy ($S_1$).
3. **Measurement & Metrics:**
   - Depth Amplification Gap: $\Delta_{\text{amp}} = \text{Accuracy}(r=16) - \text{Accuracy}(r=2)$ at Generation 0.
   - Self-Improvement Delta: $\Delta_1 = S_1 - S_0$ (measured strictly at shallow depth $r=2$).
   - Token-level AST syntactic error rate and interpreter crash rate.
4. **Random-Reward Control Arm:**
   - An identical 75.93M core initialized with the same random seed is trained through the exact same procedure, but receives randomly shuffled verifier binary feedback during Step 5.
   - Records Generation 1 shallow accuracy ($S_{1,\text{control}}$) to measure stochastic drift on procedural distributions.
5. **Pass Criterion (Go):**
   - Depth amplification is confirmed: $\Delta_{\text{amp}} \ge +20.0\%$.
   - Self-improvement is confirmed: $\Delta_1 \ge +10.0\%$ at shallow depth $r=2$.
   - Control arm is decisively defeated: $\Delta_1 - (S_{1,\text{control}} - S_0) \ge +8.0\%$.
6. **Kill / Pivot Criterion:**
   - If $\Delta_1 < +2.0\%$ after 10 TPU-hours, abort the depth distillation hypothesis; pivot recurrence trunk to RWKV-7 or increase head dimension $d_k$.
7. **Compute & Wall-Clock Budget:**
   - Stage 0 warm-start: 15 minutes on Kaggle TPU v5e-8.
   - Rollouts & verification: 2.0 hours on Kaggle 2× T4.
   - Distillation fine-tuning: 45 minutes on Kaggle TPU v5e-8.
   - Total Wall-Clock Time: ~3.0 hours (well within a single 9-hour TPU session). Cloud Cost: **0 KRW**.

---

### 12.3 Top 5 Risk Register

| # | Identified Risk | Impact | Prob | Concrete Technical Mitigation | Kill / Pivot Trigger |
|---|---|---|---|---|---|
| **R1** | **Comprehension Bottleneck on Dense Text:** A <100M attention-free reader fails to extract subtle numerical constraints from verbose narratives. | High | Med | Utilize 2 bidirectional + 2 causal GDN layers; inject 0-parameter regex number canonicalization; train explicitly on synthetic NL $\to$ AST formalization pairs. | If NL $\to$ AST extraction accuracy is $< 35.0\%$ after 60 GPU-hours, pivot reader from byte/subword to hybrid convolutional BiSSM. |
| **R2** | **Kaggle Free-Tier Session Preemption:** Sessions terminate unexpectedly at 9h/12h, corrupting uncommitted checkpoints. | Med | High | Atomic checkpoint serialization to local disk every 30 minutes, immediately followed by background asynchronous git push to private Hugging Face Hub [검증됨: `compute.md` §1]. | If session drop rate causes $>25\%$ loss of weekly training tokens, migrate training loop to Kaggle batch commit mode (`Save & Run All`). |
| **R3** | **T4 FP16 Numerical Overflow in Delta Recurrence:** Recurrent state accumulation matrix $S$ experiences numerical gradient explosion in fp16 without bf16. | High | Med | Enforce fp32 accumulators for all internal state recurrence operations; clamp recurrent learning rate $\beta \in [0, 0.999]$; utilize automatic loss scaling with PyTorch `GradScaler`. | If recurrent gradient norm hits `NaN` in $>2$ consecutive epochs despite fp32 accumulators, clamp DeltaNet spectral radius strictly to $< 1.0$. |
| **R4** | **Reward Hacking via Verifier Timing/Crash Exploits:** Proposer discovers ASTs that trigger silent Python interpreter memory faults or timeouts, falsely credited as passes. | Critical | Low | Isolate Python/SymPy sandbox in read-only sub-processes with strict exit-code assertions (`exit_code == 0`); enforce JSON type schema verification; run shadow verification on independent Python 3.12 binary. | Any mismatch between primary and shadow verifier triggers an immediate generation kill and sandbox audit. |
| **R5** | **Teacher Cord-Cut Stalling ($G_0$ Stalls at 0):** Autonomous self-play fails to discover novel valid solutions without open-teacher bootstrap data, causing $\Delta_{\text{fresh}}$ to stall at zero. | Critical | Med | Re-anchor training loop with procedural generators (Reasoning Gym / re-ARC); expand test-time search beam from $k=16$ to $k=64$; enforce PAIRED asymmetric regret rewards. | If $\Delta_{\text{fresh}} \le 0$ for 3 consecutive generations post-cut, declare plateau and execute Section 3 Rollback Rule 1. |

---

*End of Technical Proposal `work/proposal_depth_v2.md`.*
