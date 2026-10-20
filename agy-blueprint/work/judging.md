# Phase D Judging & Comparative Evaluation Report: The Aphasic Genius RSI Engine

**Evaluation Authority**: Lead Judge, Phase D Architecture Review Board  
**Target Specification**: `/home/user/hip/agy-blueprint/work/judging.md`  
**Governing Documents**: `BRIEF.md` (§4 Phase D), `AUDIT_FROM_CLAUDE.md` (§1, §2, §3), `reason_baseline.md`  
**Evaluation Scope**: Comparative Forensic Audit & Scoring of Five Revised v2 Candidate Proposals:
1. `proposal_library_v2.md` (`ARCHIMEDES-LIBRARY` / `ALEMBIC-LIB v2.0`)
2. `proposal_depth_v2.md` (`APHASIC-DEPTH-0`)
3. `proposal_search_v2.md` (`EXPERT-ITERATION-MCTS v2.0`)
4. `proposal_curiosity_v2.md` (`ORRERY-Curiosity v2.0`)
5. `proposal_minimal_v2.md` (`Project ORRERY-0 v2.0`)
**Binding Constraint**: Strictly **NO CODE** anywhere in this document (per `AUDIT_FROM_CLAUDE.md` §0.2 and `DISPATCH.md`).

---

## 1. Executive Summary & Verdict Headline

### 1.1 The Evaluation Mission
Phase D serves as the decisive architectural tribunal for the Aphasic Genius RSI Engine project. The single objective is to select the most credible, feasible, and scalable foundation for recursive self-improvement (RSI) under the binding constraints established in `BRIEF.md`:
- **Core Learner**: A tiny (<100M parameter), non-Transformer, reasoning-only "aphasic genius" that comprehends natural language but never emits conversational text, emitting strictly formal structured representations.
- **Escape from the Teacher**: Transcending the ceiling of frontier teacher distillation via exact mechanical verifiers, active self-play curricula, and compounding improvement loops.
- **Zero-Capital Constraint (0 KRW)**: Full operational execution on free Kaggle resources (2× NVIDIA T4 16GB GPUs, TPU v5e-8) and a laptop RTX 3060 (6GB VRAM) across a 6–12 month development timeline.
- **Boss Ladder Clearance**: Decisively defeating GPT-3.5 by the binding G-rule margin ($G = T + \max(10, (100 - T)/3)$) on reasoning benchmarks, with an honest, verifiable pathway to climbing frontier rungs.

Following Phase B Red Team critiques and the binding external audit (`AUDIT_FROM_CLAUDE.md`), all five design philosophies underwent thorough Phase C revisions to eliminate v1 flaws (recurrent depth undercounting, physical latency fantasies, the reading verifier void, parameter omissions, and citation laundering). This Phase D evaluation assesses the resulting v2 specifications across three mandatory evaluation lenses:
- **Lens 1: RSI Credibility (Weight ×2)**: Genuineness of escape from the teacher, compounding mechanics, curve-killer defenses, measurable cord-cut criterion, and domain expansion mechanism.
- **Lens 2: Feasibility (Weight ×1)**: Single-builder execution in 6–12 months with 0 KRW on free Kaggle hardware, exact parameter counts (<100M), real recurrent FLOP arithmetic, and memory-bandwidth-bound latency realism.
- **Lens 3: Boss Ladder (Weight ×1)**: Rigor and probability of clearing the GPT-3.5 scoreboard by the G-rule margin under fair-play rules, accompanied by credible scaling to GPT-4, o1/R1, and frontier rungs.

---

### 1.2 Verdict Headline: The Winning RSI Spine

```
====================================================================================================
                        PHASE D OFFICIAL VERDICT: WINNING RSI SPINE
====================================================================================================

      WINNER: EXPERT-ITERATION RSI (AlphaZero-Style Policy-Value MCTS)
      PROPOSAL: proposal_search_v2.md
      FINAL WEIGHTED SCORE: 37.3 / 40.0 (Normalized: 9.33 / 10.0) — RANK 1

      CORE ARCHITECTURAL RATIONALE:
      1. Mathematical Compounding: Tree search provably expands policy support beyond its sampling
         manifold, providing the only mathematically certified antidote to RLVR policy contraction.
      2. Non-Transformer Hardware Synergy: Gated DeltaNet maintains a strictly constant-size recurrent
         state matrix (O(1) memory, zero KV-cache) per tree node, executing MCTS expansions 8x-10x
         faster than Transformers on memory-constrained Turing T4 hardware.
      3. Operational Sweet-Spot: A 1.0- to 2.0-hour generation cycle (10-12 generations/week) fits
         Kaggle free quotas with 25% safety margin, avoiding both the slow multi-day batches of Depth
         and the hyperparameter fragility of Curiosity.

====================================================================================================
```

The tribunal conclusively selects **`proposal_search_v2.md` (Expert-Iteration RSI / AlphaZero-Style MCTS)** as the **Core RSI Spine** for the Phase E Chief Architect Blueprint. 

Crucially, this selection does not discard the distinct breakthroughs of the competing philosophies. Instead, Phase E will execute an aggressive, principled **Grafting Synthesis**: the robust MCTS Expert-Iteration spine will be reinforced with:
1. **The Zero-Parameter Skill Library** and **Saturation Metric $\Psi$** from `proposal_library_v2.md`.
2. **Depth Regret ($R = \max(0, S_{\text{deep}} - S_{\text{shallow}})$)** and **Depth-Modulated Policy Distillation ($e_r$)** from `proposal_depth_v2.md`.
3. **Value-Head Latent Pruning** and **Verifier-Grounded Empirical Learning Progress (ELP)** from `proposal_curiosity_v2.md`.
4. **The Random-Reward Control Arm Gate** and **Large-Batch Rollout Amortization** from `proposal_minimal_v2.md`.

---

## 2. Master Scorecard Matrix

### 2.1 Comparative Scorecard Table

Every candidate proposal was rigorously audited and scored from 0.0 to 10.0 across the three mandatory lenses. In accordance with `DISPATCH.md`, Lens 1 (RSI Credibility) carries a weight of **×2**, while Lens 2 (Feasibility) and Lens 3 (Boss Ladder) each carry a weight of **×1**, yielding a total weighted maximum score of **40.0 points** (normalized to a 10.0-point scale by dividing by 4).

| Rank | Proposal Name & Target Philosophy | Lens 1: RSI Credibility (Weight ×2) | Lens 2: Feasibility (Weight ×1) | Lens 3: Boss Ladder (Weight ×1) | Total Weighted Score (Max 40.0) | Normalized Score (0–10 Scale) | Verdict / Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **`proposal_search_v2.md`**<br>*(Expert-Iteration MCTS)* | **9.4 / 10**<br>(Weighted: **18.8**) | **9.1 / 10**<br>(Weighted: **9.1**) | **9.4 / 10**<br>(Weighted: **9.4**) | **37.3 / 40.0** | **9.33 / 10** | **SELECTED WINNING RSI SPINE** |
| **2** | **`proposal_library_v2.md`**<br>*(Library-Growing / Stitch)* | **9.3 / 10**<br>(Weighted: **18.6**) | **8.6 / 10**<br>(Weighted: **8.6**) | **9.2 / 10**<br>(Weighted: **9.2**) | **36.4 / 40.0** | **9.10 / 10** | **Primary Grafting Donor** *(L3 Library & $\Psi$)* |
| **3** | **`proposal_depth_v2.md`**<br>*(Depth-Distillation Ratchet)* | **8.9 / 10**<br>(Weighted: **17.8**) | **8.2 / 10**<br>(Weighted: **8.2**) | **8.9 / 10**<br>(Weighted: **8.9**) | **34.9 / 40.0** | **8.73 / 10** | **Key Grafting Donor** *(Depth Regret & $e_r$)* |
| **4** | **`proposal_minimal_v2.md`**<br>*(Minimal Fastest-Cycle)* | **8.3 / 10**<br>(Weighted: **16.6**) | **9.6 / 10**<br>(Weighted: **9.6**) | **8.6 / 10**<br>(Weighted: **8.6**) | **34.8 / 40.0** | **8.70 / 10** | **Control Donor** *(Random-Reward Gate)* |
| **5** | **`proposal_curiosity_v2.md`**<br>*(Curiosity / JEPA World Model)* | **8.4 / 10**<br>(Weighted: **16.8**) | **7.8 / 10**<br>(Weighted: **7.8**) | **8.8 / 10**<br>(Weighted: **8.8**) | **33.4 / 40.0** | **8.35 / 10** | **Pruning Donor** *(Latent Pruning & ELP)* |

---

### 2.2 Analytical Breakdown of Score Distribution

1. **The Decisive Factor in Lens 1 (RSI Credibility)**:
   The divergence in Lens 1 separates designs with true test-time amplification mechanisms from those reliant purely on policy-gradient sampling. `proposal_search_v2.md` (9.4) and `proposal_library_v2.md` (9.3) lead decisively because both introduce structural mechanisms that expand capability beyond the fixed neural parameter manifold. Search uses lookahead trees with exact verifier pruning to discover out-of-distribution trajectories; Library accumulates external symbolic code that permanently lowers AST search depth. Conversely, `proposal_minimal_v2.md` (8.3) was penalized because pure RLVR on small sample sizes ($K=4$) contracts policy entropy, lacking a robust test-time search amplifier.
2. **The Realities of Lens 2 (Feasibility on Free Hardware)**:
   `proposal_minimal_v2.md` achieved the highest feasibility score (9.6) due to its minimal software surface area, clean PyTorch/JAX formulation, and lightning-fast 6.7-minute cycle time. `proposal_search_v2.md` followed closely (9.1) by rightsizing its generation batch to 5,000 tasks and deploying Snell adaptive search, achieving a clean 1.0–2.0 hour turnaround. In contrast, `proposal_depth_v2.md` (8.2) suffered due to an unwieldy 18-hour GPU rollout schedule that allows only ~1 generation per week, while `proposal_curiosity_v2.md` (7.8) incurred heavy implementation risk from its 13-module topology, multi-objective JEPA loss, and latent variance collapse hazards.
3. **The Precision of Lens 3 (Boss Ladder Rigor)**:
   All five v2 revisions successfully restored missing scoreboard rows, disaggregated MuSR subtasks, and respected the G-rule margin. `proposal_search_v2.md` scored highest (9.4) by providing exhaustive binomial confidence intervals, verified baselines from `reason_baseline.md`, and a grounded ARC-AGI-1 projection (32.0% base, 40.5% with CA adapter and test-time training). `proposal_library_v2.md` (9.2) and `proposal_depth_v2.md` (8.9) also presented compelling scoreboard routes, whereas `proposal_minimal_v2.md` (8.6) showed lower credibility on advanced rungs (MATH > 70%, LiveCodeBench > 40%) due to its lack of deep search.

---

## 3. Exhaustive Qualitative Breakdown of All Five Proposals

---

### 3.1 Evaluation of Proposal 1: ALEMBIC-LIB v2.0 (Library-Growing RSI)

**Document**: `/home/user/hip/agy-blueprint/work/proposal_library_v2.md`  
**Philosophy**: Neuro-Symbolic Compilation over 0-Parameter Executors with Stitch Symbolic Abstraction.  
**Core Parameter Footprint**: 73,772,009 (~73.77M parameters; 26.23M safety margin).  
**Hardware Profile**: Cycle time = 103.3 min (TPU) / 133.7 min (T4); Throughput = 12–14 generations/week.  

```
+----------------------------------------------------------------------------------------------------+
| EVALUATION SUMMARY: ALEMBIC-LIB v2.0                                                               |
+------------------------------------+----------------------------------+----------------------------+
| Lens 1: RSI Credibility: 9.3 / 10  | Lens 2: Feasibility: 8.6 / 10   | Lens 3: Boss Ladder: 9.2   |
+------------------------------------+----------------------------------+----------------------------+
| STRENGTHS:                         | WEAKNESSES:                      | VERDICT:                   |
| - Non-saturating symbolic growth   | - Stitch e-graph runtime latency | High-value donor for Phase |
| - Zero parameter library transfer  | - Complex AST lowering bridge    | E synthesis; primary L3    |
| - Robust Ψ saturation trigger      | - Moderate solo engineering load | capability layer.          |
+------------------------------------+----------------------------------+----------------------------+
```

#### Detailed Evaluation Across Lenses:

- **Lens 1: RSI Credibility (Score: 9.3 / 10 — Excellent)**:
  `ALEMBIC-LIB v2.0` offers one of the most theoretically profound answers to the tiny-model capacity paradox. While neural weights hit a hard memorization ceiling (~2 bits per parameter), an externalized symbolic library stored in pure Python/SymPy functions has an infinite parameter ceiling and costs zero neural weights. The DreamCoder wake-sleep loop, modernized via the Stitch functional S-expression anti-unification engine, compresses complex 50-step reasoning trajectories into 3-step macro-invocations ($\Delta \text{DL}$). This compresses effective search depth, allowing a fixed 73.8M core to solve increasingly difficult problems without parameter growth. 
  
  The cord-cut criteria are exceptionally thorough: requiring a 20% solve rate on teacher-failure problems, consecutive self-improvement deltas, description-length compression ratio $\text{CR} \ge 1.30$, and a $+5.0\%$ margin over a parallel random-reward control arm. The domain expansion mechanism is grounded in an empirical saturation metric:
  $$\Psi_D(g) = \frac{\text{pass@1}(D)}{\text{pass@64}(D)} \times \left(1 - \frac{\Delta \text{DL}(\mathcal{P}_g)}{\text{DL}(\mathcal{P}_g)}\right) \ge 0.95$$
  This provides a concrete, mathematical trigger for unlocking the next verifier interface from the frozen registry.

- **Lens 2: Feasibility (Score: 8.6 / 10 — Very Good)**:
  The v2 revision thoroughly resolved its v1 flaws. Parameters were rigorously recounted to 73,772,009, properly accounting for formal embeddings (1.05M), causal convs, DeltaNet output gates (3.67M), and pointer heads. The FLOP arithmetic honestly accounts for recurrent unrolling ($r_{\text{avg}} = 4.2$ rollout, $r_{\text{train}} = 4$). Latency was grounded in Turing T4 memory bandwidth ($300\text{ GB/s}$), correcting single-pass latency to $0.742\text{ s}$ and 32-rollout search to $3.80\text{ s}$.
  
  The primary feasibility concern centers on the engineering surface area for a solo builder in 6–12 months. Operating Stitch requires compiling a functional S-expression Intermediate Representation (Symbolic IR), maintaining an e-graph anti-unification binary in Rust, and executing a deterministic 0-parameter transpiler lowering IR into Python, SymPy, and Z3. While completely feasible, it introduces compiler debugging friction that the winning search spine avoids.

- **Lens 3: Boss Ladder (Score: 9.2 / 10 — Excellent)**:
  The proposal demonstrates mastery of the benchmark landscape. All Tier A and Tier B rows are accounted for with verified baseline numbers from `reason_baseline.md`. Word problems are mapped to SymPy equations (GSM8K at 89.5% vs $G=88.1$; GSM-Plus at 84.8% vs $G=74.1$), logic puzzles are autoformalized to Z3 (FOLIO at 85.0%, PrOntoQA at 96.5%), and algorithmic puzzles compile to Python AST (Game of 24 at 100%, Blocksworld at 88.0%). ARC-AGI-1 was honestly recalibrated to 31.5% [추정] (respecting attention-free limits), augmented by 2D coordinate embeddings and depthwise conv patch tokenizers. Subsequent rungs are framed through qualitative confidence tiers with explicit empirical falsification criteria.

---

### 3.2 Evaluation of Proposal 2: APHASIC-DEPTH-0 (Depth-Distillation RSI)

**Document**: `/home/user/hip/agy-blueprint/work/proposal_depth_v2.md`  
**Philosophy**: Latent Recursive Deliberation via Weight-Tied Gated DeltaNet with Amplify-then-Distill Ratchet.  
**Core Parameter Footprint**: 75,928,490 (~75.93M parameters; 24.07M safety margin).  
**Hardware Profile**: Cycle time = Massive batch schedule (18h T4 rollout + 1.5h TPU train); Throughput = ~1 generation/week.  

```
+----------------------------------------------------------------------------------------------------+
| EVALUATION SUMMARY: APHASIC-DEPTH-0                                                                |
+------------------------------------+----------------------------------+----------------------------+
| Lens 1: RSI Credibility: 8.9 / 10  | Lens 2: Feasibility: 8.2 / 10   | Lens 3: Boss Ladder: 8.9   |
+------------------------------------+----------------------------------+----------------------------+
| STRENGTHS:                         | WEAKNESSES:                      | VERDICT:                   |
| - Brilliant depth-regret metric    | - Very slow generation cadence   | Essential grafting donor;  |
| - Elegant System 1/2 crystallization| - High T4 quota concentration    | provides depth regret and  |
| - Solved v1 collapse paradox       | - Eventual neural capacity wall  | adaptive halting modules.  |
+------------------------------------+----------------------------------+----------------------------+
```

#### Detailed Evaluation Across Lenses:

- **Lens 1: RSI Credibility (Score: 8.9 / 10 — Very Good)**:
  `APHASIC-DEPTH-0` introduces one of the most compelling biological analogies: System 2 deliberation (deep test-time unrolling at $r=16..32$ with MCTS) solving hard frontier tasks, which are subsequently distilled into System 1 shallow intuition ($r=2..4$) via token-level policy distillation on verified ASTs. This directly resolves the fatal v1 collapse paradox (where latent cosine matching forced the recurrent cell into an identity map).
  
  Its formulation of **Depth Regret** is an outstanding contribution:
  $$R_{\text{depth}} = \max\left(0, \text{Score}_{\text{deep}} - \text{Score}_{\text{shallow}}\right)$$
  This parameter-free filter cleanly isolates the model's zone of proximal development, discarding trivial problems ($r \le 2$) and impossible problems ($r=32$), completely dismantling the own-level prison. However, because all capability is compressed into the fixed 75.9M neural weights, it will inevitably hit a capacity ceiling earlier than external library designs when climbing toward frontier multi-domain rungs.

- **Lens 2: Feasibility (Score: 8.2 / 10 — Good)**:
  Parameter accounting is airtight at 75,928,490 weights ($d=576, d_{\text{ffn}}=1,325$), with 24.07M headroom. The 4-layer reader (2 bidirectional passes + 2 causal passes) and 8-layer recurrent trunk are correctly formulated. Stage 0 training was wisely shifted from T4 to TPU v5e-8 ($0.89\text{ h}$ at $1.26 \times 10^{18}$ FLOPs), avoiding T4 session timeouts.
  
  The fatal weakness in feasibility is its **operational cadence**. By organizing rollouts into a massive single batch of 70.4 million formal tokens taking 18 GPU-hours on 2× T4, a single generation consumes 80% of the entire weekly T4 quota. Running only ~1 generation per week creates severe risk for a solo builder: an undetected bug, preemption failure, or distribution shift ruins an entire week of progress. Agile RSI requires tighter, micro-batch generation cycles.

- **Lens 3: Boss Ladder (Score: 8.9 / 10 — Very Good)**:
  The scoreboard is comprehensive, cleanly restoring PrOntoQA ($G=95.0$), StrategyQA ($G=80.3$), and decoupling MuSR into Murder Mysteries ($G=80.0$), Object Placement ($G=72.1$), and Team Allocation ($G=88.8$). Baseline values are honestly labeled, and ARC-AGI-1 is grounded at 32.5% [추정] (with 42.0% [가설] requiring 2D multi-directional Hilbert scan inductive bias). The Boss Ladder progression is clearly articulated, though its climb beyond Rung 2 will be constrained by the lack of an external symbolic library.

---

### 3.3 Evaluation of Proposal 3: EXPERT-ITERATION-MCTS v2.0 (Search-Driven RSI) — THE WINNER

**Document**: `/home/user/hip/agy-blueprint/work/proposal_search_v2.md`  
**Philosophy**: Policy-Value MCTS over Formal Reasoning Steps with Exact Verifiers and Latent Depth Distillation.  
**Core Parameter Footprint**: 81,639,425 (~81.64M parameters; 18.36M safety margin).  
**Hardware Profile**: Cycle time = 1.0 to 2.0 hours (0.69h T4 rollout + 0.01h TPU train); Throughput = 10–12 generations/week.  

```
+----------------------------------------------------------------------------------------------------+
| EVALUATION SUMMARY: EXPERT-ITERATION-MCTS v2.0 (WINNING SPINE)                                     |
+------------------------------------+----------------------------------+----------------------------+
| Lens 1: RSI Credibility: 9.4 / 10  | Lens 2: Feasibility: 9.1 / 10   | Lens 3: Boss Ladder: 9.4   |
+------------------------------------+----------------------------------+----------------------------+
| STRENGTHS:                         | WEAKNESSES:                      | VERDICT:                   |
| - AlphaZero-proven compounding     | - Slightly higher parameter count| OFFICIAL WINNING SPINE;    |
| - O(1) state MCTS node expansion   |   (81.6M vs 64.9M)               | perfect balance of rigor,  |
| - Optimal 1-2 hour cycle time      | - Tree search tuning overhead    | throughput, and hardware.  |
+------------------------------------+----------------------------------+----------------------------+
```

#### Detailed Evaluation Across Lenses:

- **Lens 1: RSI Credibility (Score: 9.4 / 10 — Outstanding / Category Best)**:
  `EXPERT-ITERATION-MCTS v2.0` represents the gold standard of autonomous self-improvement. By framing reasoning as Monte Carlo Tree Search over discrete formal tokens guided by policy priors $\pi_\theta$, value evaluations $V_\phi$, and terminal/intermediate mechanical verification $z \in \{-1, +1\}$, the search distribution $\pi_{\text{search}}$ provably acts as an amplified, superior policy operator:
  $$\mathbb{E}_{a \sim \pi_{\text{search}}}[Q(s, a)] \ge \mathbb{E}_{a \sim \pi_\theta}[Q(s, a)]$$
  
  Crucially, this addresses the fundamental finding of arXiv:2504.13837: while pure RLVR contracts policy entropy and narrows reasoning boundaries, lookahead tree search actively expands exploration into unvisited branches. Retraining the network on $\pi_{\text{search}}$ via Expert Iteration forms a genuine compounding ratchet: yesterday's multi-step search becomes today's shallow intuition. 
  
  The proposal features an indispensable metric: the **Search Internalization Metric ($E_g$)**:
  $$E_g = \frac{\text{Rollouts}_{g-1}(80\%)}{\text{Rollouts}_g(80\%)} > 1.0$$
  This directly measures whether the policy is internalizing reasoning capability. Combined with a 3-way paraphrase reading harness, counterfactual perturbation gating, and a strict cord-cut gate, its RSI credibility is unmatched.

- **Lens 2: Feasibility (Score: 9.1 / 10 — Outstanding)**:
  `proposal_search_v2.md` achieved a decisive breakthrough in hardware feasibility by discovering the **Non-Transformer MCTS Architectural Moat**:
  - In standard Transformers, MCTS is crippled by the KV-cache, which must be allocated, copied, or recomputed for every node in the search tree, causing severe memory fragmentation and latency stalls.
  - In Gated DeltaNet, the hidden state is a **constant-size matrix ($S_t \in \mathbb{R}^{d \times d}$, exactly 320 KB per layer)**. Node states are cached directly in the tree with $O(1)$ memory, accelerating tree expansion by **8× to 10× on NVIDIA T4 GPUs**.
  
  By rightsizing its generation batch to 5,000 tasks and deploying Snell adaptive search (mean 18 simulations/task), rollout time on 2× T4 is just 41.6 minutes ($0.694\text{ h}$), verification takes 45 seconds, and training takes 27 seconds on TPU (or 6.6 minutes on T4). The resulting **1.0 to 2.0 hour generation cycle** delivers **10 to 12 generations per week** while consuming only 24 of the 30 available T4 GPU-hours and ~1 TPU-hour. This leaves comfortable safety buffers against Kaggle session preemption. Parameter accounting is rigorous at 81,639,425 weights ($<100\text{M}$ with 18.36M headroom).

- **Lens 3: Boss Ladder (Score: 9.4 / 10 — Outstanding / Category Best)**:
  The proposal provides the most rigorous, decontaminated scoreboard in the cohort. Every target score is derived from verified baselines in `reason_baseline.md` and accompanied by exact 95% confidence interval half-widths ($\text{CI}_{95}$). The winning mechanisms are concrete: Game of 24 (98.0% vs $G=46.0$), Blocksworld (68.5% vs $G=37.4$), GSM8K (89.2% vs $G=88.1$), MATH (64.8% vs $G=56.1$), LiveCodeBench (51.2% vs $G=48.4$), and PrOntoQA (96.5% vs $G=95.0$). ARC-AGI-1 is realistically estimated at 32.0% [추정] (with 40.5% requiring a cellular-automaton adapter and test-time training). The subsequent ladder progression through o1/R1 and frontier tin cans (Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra) is backed by concrete operational prerequisites.

---

### 3.4 Evaluation of Proposal 4: ORRERY-Curiosity v2.0 (Curiosity & World-Model RSI)

**Document**: `/home/user/hip/agy-blueprint/work/proposal_curiosity_v2.md`  
**Philosophy**: Attention-Free Non-Transformer Core + JEPA Latent World Model + Verifier-Grounded Epistemic Search.  
**Core Parameter Footprint**: 68,328,193 (~68.33M parameters; 31.67M safety margin).  
**Hardware Profile**: Cycle time = 15.5 min (TPU) / 37.1 min (T4); Throughput = 18–22 generations/week (T4) / >40 (TPU).  

```
+----------------------------------------------------------------------------------------------------+
| EVALUATION SUMMARY: ORRERY-Curiosity v2.0                                                          |
+------------------------------------+----------------------------------+----------------------------+
| Lens 1: RSI Credibility: 8.4 / 10  | Lens 2: Feasibility: 7.8 / 10   | Lens 3: Boss Ladder: 8.8   |
+------------------------------------+----------------------------------+----------------------------+
| STRENGTHS:                         | WEAKNESSES:                      | VERDICT:                   |
| - Solved "Noisy TV" via ELP        | - High implementation complexity | Critical donor for Phase E |
| - Value-head latent pruning (>60%) | - Latent JEPA stability risks    | latent pruning and domain  |
| - Strong North Star vision         | - Fragile multi-objective loss   | transition triggers.       |
+------------------------------------+----------------------------------+----------------------------+
```

#### Detailed Evaluation Across Lenses:

- **Lens 1: RSI Credibility (Score: 8.4 / 10 — Very Good)**:
  `ORRERY-Curiosity v2.0` tackled its Phase B critique with exceptional intellectual vigor. Its primary conceptual breakthrough is solving the notorious **"Noisy TV" problem** (where curiosity engines get hypnotized by unlearnable stochastic noise):
  $$\text{ELP}_g(x) = \left| \mathcal{E}_g(x) - \mathcal{E}_{g-1}(x) \right| \cdot \mathbb{I}(v(x) \text{ is deterministically verifiable})$$
  By weighting latent prediction error deltas exclusively by deterministic verifiability, unlearnable noise yields $\Delta \mathcal{E} \to 0$, driving curiosity strictly to the reducible error frontier.
  
  Furthermore, its **Value-Head Latent Pruning** mechanism (pruning branches with predicted outcome $\hat{v} < 0.2$ in latent space before sandbox dispatch) cuts expensive external execution by $>60\%$. However, learning a stable JEPA representation space for discrete formal programs remains theoretically precarious. In continuous vision/robotics, JEPA excels; in discrete deductive logic, a single flipped token radically alters semantics, making smooth latent transitions prone to false interpolation.

- **Lens 2: Feasibility (Score: 7.8 / 10 — Acceptable / Lowest in Cohort)**:
  The proposal successfully verified its parameter table at 68,328,193 weights (including the 2D conv adapter, output gates, and dense bi-encoder) and corrected its FLOP math for $r=16$ ($4.19 \times 10^{15}$ rollout FLOPs; $4.58 \times 10^{16}$ training FLOPs). Cycle time is impressive: 15.5 min on TPU and 37.1 min on T4.
  
  However, feasibility scored lowest due to **severe architectural surface area and multi-objective optimization risk**. The system attempts to simultaneously optimize: (1) policy cross-entropy, (2) smooth L1 JEPA transition loss with stop-gradients, (3) value BCE loss, (4) VICReg representation variance penalties, (5) dynamic halting, and (6) a 3-head curiosity ensemble across 13 distinct modules. For a solo builder with 0 KRW in 6–12 months, balancing these gradients without experiencing latent representation collapse is a daunting engineering hazard.

- **Lens 3: Boss Ladder (Score: 8.8 / 10 — Very Good)**:
  The scoreboard is comprehensive, restoring PrOntoQA ($G=95.0$), ReClor ($G=71.6$), and unbundling MuSR subtasks. ARC-AGI-1 is targeted at 41.2% [추정] ($G=40.4$), grounded in the 2D spatial convolution adapter. Its North Star mapping to oncology therapeutics (using RDKit, OpenMM, and AutoDock Vina as physical verifiers guided by Bayesian active learning) is the most visionary and detailed in the cohort.

---

### 3.5 Evaluation of Proposal 5: Project ORRERY-0 v2.0 (Minimal Fastest-Cycle RSI)

**Document**: `/home/user/hip/agy-blueprint/work/proposal_minimal_v2.md`  
**Philosophy**: Minimal Fastest-Cycle RSI Engine (Attention-Free Core + Verifiers + Entropy-Regularized GRPO + Configuration-Program Self-Play).  
**Core Parameter Footprint**: 64,890,883 (~64.89M parameters; 35.11M safety margin).  
**Hardware Profile**: Cycle time = 6.69 min (TPU) / 10.68 min (T4); Throughput = 100 to 220 generations/week.  

```
+----------------------------------------------------------------------------------------------------+
| EVALUATION SUMMARY: Project ORRERY-0 v2.0                                                          |
+------------------------------------+----------------------------------+----------------------------+
| Lens 1: RSI Credibility: 8.3 / 10  | Lens 2: Feasibility: 9.6 / 10   | Lens 3: Boss Ladder: 8.6   |
+------------------------------------+----------------------------------+----------------------------+
| STRENGTHS:                         | WEAKNESSES:                      | VERDICT:                   |
| - Highest feasibility in cohort    | - Lacks deep test-time search    | Essential operational and  |
| - Extreme 6.7-min cycle velocity   | - Susceptible to RLVR boundary   | control arm donor for the  |
| - Pioneered random-reward gate     |   contraction (arXiv 2504.13837) | final synthesized engine.  |
+------------------------------------+----------------------------------+----------------------------+
```

#### Detailed Evaluation Across Lenses:

- **Lens 1: RSI Credibility (Score: 8.3 / 10 — Good / Lowest in Cohort)**:
  `ORRERY-0 v2.0` represents an extraordinary benchmark for execution speed. By stripping away all non-essential modules, it achieves a generation cycle of 6.69 minutes on TPU (10.68 min on T4), allowing 100 to 220 generations per week. It also made a permanent contribution to project integrity by pioneering the **Random-Reward Control Arm Gate** (requiring verified self-improvement to beat a random Bernoulli control by $\ge 5.0\%$).
  
  However, its RSI credibility scored lowest because of an inescapable algorithmic limitation: **it relies on shallow rollouts ($K=4$) trained via entropy-regularized GRPO**. As established by arXiv:2504.13837, pure policy-gradient RLVR contracts policy entropy and narrows reasoning boundaries. Without an explicit lookahead tree search (like MCTS in Search) or deep recursive amplification (like Depth), the model's exploratory horizon is strictly bounded by what its current policy can sample at $K=4$. On easy domains, rapid iterations compound nicely; on complex frontier tasks (MATH, LiveCodeBench, Lean 4), sampling probability drops to zero, stalling the engine in an algorithmic dead-end.

- **Lens 2: Feasibility (Score: 9.6 / 10 — Outstanding / Category Best)**:
  In feasibility, `ORRERY-0` is unmatched. With only 64,890,883 parameters, it has the leanest memory footprint (35.11M headroom). Its software surface area is exceptionally small: standard Gated DeltaNet, pure PyTorch/JAX, zero exotic e-graph compilers, and zero unstable latent world models. A solo builder could build, debug, and operationalize this entire codebase within 6 to 8 weeks. It also correctly identified that large-batch decoding ($B=128$) amortizes Turing T4 memory bandwidth, reaching $\approx 10,600\text{ tokens/s}$ across 2× T4 GPUs.

- **Lens 3: Boss Ladder (Score: 8.6 / 10 — Very Good)**:
  The scoreboard is complete and rigorous, featuring exact binomial 95% confidence intervals for every row. It correctly grounds ARC-AGI-1 at 34.5% [추정] (using a 2D Multi-Axis Associative Scan). However, its probability of conquering subsequent rungs (Rung 2 GPT-4 MATH 75%, Rung 3 o1/R1 AIME 50%) is low without an explicit test-time compute scaling mechanism like MCTS.

---

## 4. The Grafting Matrix & Synthesis Blueprint

The true power of this comparative audit lies in systematic architectural grafting. Rather than building a monoculture, the Phase E Chief Architect Blueprint must assemble a hybrid system that anchors to the winning **Search-Driven MCTS Spine** while grafting the highest-value mechanisms from all four competing designs.

### 4.1 Master Grafting Matrix

Cross-referenced with `AUDIT_FROM_CLAUDE.md` §3 (Graftable Ideas):

| Graft ID | Source Proposal | Grafted Innovation / Mechanism | Cross-Ref in AUDIT §3 | Architectural Function & Compounding Rationale | Parameters Added | FLOP Impact | Phase E Blueprint Role |
|:---:|---|---|:---:|---|:---:|:---:|---|
| **G-01** | `minimal` | **Random-Reward Control Arm Gate** | Item 1 | Parallel control arm trained on shuffled Bernoulli rewards; must outperform control by $\ge 5.0\%$ to validate self-improvement. | **0 params** | +10% training FLOPs | **Mandatory Cord-Cut Gate** (Falsifies fake self-improvement). |
| **G-02** | `depth` | **Depth Regret Task Selection** | Item 2 | $R = \max(0, S_{\text{deep}} - S_{\text{shallow}})$; targets tasks at the boundary of competence solvable only with search. | **0 params** | 0 FLOPs | **Proposer Reward Signal** (Neutralizes own-level prison). |
| **G-03** | `depth` | **Depth-Distillation Ratchet** | Item 3 | Depth modulation embeddings $e_r$ allow policy to distill deep search solutions into shallow passes ($r=2..4$). | **8.2k params** | 0 FLOPs | **Policy Ratchet** (Frees deep search for harder frontiers). |
| **G-04** | `library` | **Zero-Parameter Symbolic Skill Library** | Item 4 | Discovered subroutines compressed into external Python/SymPy library; carries across scale with zero weights. | **0 params** | Cuts rollout FLOPs by 30% | **Non-Parametric Compounding** (L3 Skill Accumulation). |
| **G-05** | `io_split` / All | **Counterfactual Swap Test for Renderer** | Item 5 | Corrupt core's output; renderer must repeat corrupted value $\ge 99.0\%$ of time, proving it does not solve tasks. | **0 params** | 0 FLOPs | **Renderer Firewall CI Gate** (Guarantees fair play). |
| **G-06** | `library` | **Domain Saturation Coefficient ($\Psi$)** | Item 6 | $\Psi = \frac{\text{pass@1}}{\text{pass@64}} \times (1 - \frac{\Delta DL}{DL}) \ge 0.85$; triggers transition to next verifier domain. | **0 params** | 0 FLOPs | **S-Curve Domain Trigger** (Automates domain ascension). |
| **G-07** | `curiosity` | **Empirical Domain Learnability Index (EDLI)** | Item 7 | $\text{EDLI} = \Delta \text{Accuracy} / \text{GPU-Hour}$; compute-aware allocation directing rollouts to highest-velocity domains. | **0 params** | 0 FLOPs | **Compute Allocation Governor** (Prevents compute waste). |
| **G-08** | `search` | **Search Internalization Metric ($E_g$)** | Item 8 | $E_g = \text{Rollouts}_{g-1}(80\%) / \text{Rollouts}_g(80\%) > 1.0$; tracks crystallization of search into policy. | **0 params** | 0 FLOPs | **Core RSI Metric** (Primary compounding health monitor). |
| **G-09** | `library` / `depth` | **Description Length Compression Ratio ($CR$)** | Item 9 | $CR = \text{DL}_{\text{raw}} / \text{DL}_{\text{compressed}}$; measures syntactic compaction of solutions over generations. | **0 params** | 0 FLOPs | **Macro-Induction Monitor** (Validates library efficiency). |
| **G-10** | `search` | **Constant-Size Recurrent State per MCTS Node** | Item 10 | DeltaNet recurrent state ($S_t \in \mathbb{R}^{d \times d}$) cached per tree node with zero KV-cache bloat. | **0 params** | Accelerates MCTS by 8x | **Foundational Search Moat** (Core algorithmic advantage). |
| **G-11** | `curiosity` | **Value-Head Latent Pruning** | Item 11 | Value critic prunes search branches with $\hat{v} < 0.2$ in latent space before sandbox dispatch. | **0 params** | Cuts sandbox calls by >60% | **Sandbox Latency Shield** (Eliminates execution bottlenecks). |
| **G-12** | `curiosity` | **Verifier-Grounded Learning Progress (ELP)** | Item 12 | Error delta multiplied by verifier determinism; completely neutralizes the "Noisy TV" stochastic trap. | **0 params** | 0 FLOPs | **Noise Filter for Self-Play** (Guarantees focus on valid tasks). |
| **G-13** | All | **4-Layer Symmetric Reading Firewall** | AUDIT §1.4 | Reverse procedural generation + 3-way paraphrase gating + counterfactual twin perturbation + robust twin gates. | **0 params** | +15% rollout FLOPs | **Reading Hole Solution** (Eliminates misreading exploits). |

---

### 4.2 Detailed Operational Mechanics of Key Grafted Innovations

```
====================================================================================================
               PHASE E ARCHITECTURAL SYNTHESIS BLUEPRINT (THE GRAFTED ENGINE)
====================================================================================================

      +---------------------------------------------------------------------------------------+
      | 1. DUAL CURRICULUM GENERATION                                                         |
      | - 50% Stepped Procedural Curricula (Reasoning Gym, re-ARC, SymPy, PlanBench)          |
      | - 50% Asymmetric Proposer rewarded by DEPTH REGRET: R = max(0, S_deep - S_shallow)     |
      | - G-12 Verifier-Grounded ELP filters out unlearnable stochastic noise                 |
      +-------------------------------------------┬-------------------------------------------+
                                                  │
                                                  v
      +---------------------------------------------------------------------------------------+
      | 2. G-13 4-LAYER SYMMETRIC READING FIREWALL                                            |
      | - Reverse Procedural Generation: constructs gold AST P* before rendering NL text X    |
      | - Three-Way Paraphrase Invariance: asserts AST equivalence across 3 surface forms     |
      | - Counterfactual Perturbation Twins: verifies exact predicted output shifts           |
      +-------------------------------------------┬-------------------------------------------+
                                                  │
                                                  v
      +---------------------------------------------------------------------------------------+
      | 3. APHASIC REASONING CORE (78M Gated DeltaNet, r in [1, 8])                           |
      | - Bidirectional Reader (d=576) + ARC 2D Multi-Axis Spatial Adapter                    |
      | - G-10 Constant-Size Recurrent State (S_t in R^{576x576}, O(1) Memory, NO KV-CACHE)   |
      | - G-03 Depth-Embedding Modulation (e_r) supporting shallow and deep operational modes |
      +-------------------------------------------┬-------------------------------------------+
                                                  │
                                                  v
      +---------------------------------------------------------------------------------------+
      | 4. SNELL ADAPTIVE MCTS SEARCH & LATENT PRUNING                                        |
      | - Dynamic simulation budget (8 to 64 sims based on root entropy; mean 18 sims/task)   |
      | - G-11 Value-Head Latent Pruning: discards dead-end branches (v < 0.2) in latent space|
      | - G-04 Zero-Parameter External Skill Library (Stitch DSL macros shorten search trees) |
      +-------------------------------------------┬-------------------------------------------+
                                                  │
                                                  v
      +---------------------------------------------------------------------------------------+
      | 5. IMMUTABLE SANDBOXED MECHANICAL VERIFICATION                                        |
      | - Isolated Docker / nsjail Sandboxes (Python 3.12, SymPy 1.13, Z3 SMT, Lean 4)        |
      | - Execution correctness + formal AST isomorphism check: SymPy(P_core - P*) == 0       |
      +-------------------------------------------┬-------------------------------------------+
                                                  │
                                                  v
      +---------------------------------------------------------------------------------------+
      | 6. EXPERT ITERATION POLICY UPDATE & CORD-CUT AUDITING                                 |
      | - Core weights updated via ExIt cross-entropy on verified search traces pi_search     |
      | - G-08 Search Internalization Metric (E_g > 1.0) confirms distillation success        |
      | - G-01 Random-Reward Control Arm Gate: must outperform random rewards by >= +5.0%     |
      | - G-06 Saturation Metric (Psi >= 0.85) triggers S-curve ascension via G-07 EDLI index |
      +---------------------------------------------------------------------------------------+
====================================================================================================
```

---

## 5. Catalog of Remaining Risks & Structural Weaknesses

Every candidate proposal contains residual technical risks that the Phase E Chief Architect must explicitly mitigate:

| Proposal Identity | Primary Technical Risk | Severity | Concrete Failure Mechanism | Phase E Mandatory Defensive Countermeasure |
|---|---|:---:|---|---|
| **`proposal_library_v2.md`** | **Symbolic Compression Latency Bottleneck** | High | As verified traces accumulate into tens of thousands, Stitch e-graph anti-unification CPU time scales super-linearly, threatening to bottleneck generation cycle times on 4-vCPU runners. | Mandate bounded sample clustering: run Stitch over randomly partitioned shards of 2,000 verified traces rather than the full historical archive; enforce a strict 15-minute CPU watchdog timeout. |
| **`proposal_depth_v2.md`** | **Neural Deliberation Capacity Wall** | High | Attempting to compress all historical multi-domain knowledge into weight-tied recurrent parameters ($75.9\text{M}$) will cause catastrophic forgetting or representational interference as domains expand. | Couple the recurrent trunk with the non-parametric external skill library (G-04) and enforce an immutable 25% replay buffer of historical seed data. |
| **`proposal_search_v2.md`** | **MCTS Sandbox IPC Latency Spike** | Medium | Dispatching 18 simulations $\times$ 5,000 tasks ($90,000$ executions) through subprocess pipes could create severe operating system context-switching overhead on Kaggle environments. | Deploy pre-warmed persistent daemon pools inside `nsjail` communicating over shared Unix memory pipes; leverage G-11 latent value pruning to discard $>60\%$ of branches before IPC dispatch. |
| **`proposal_curiosity_v2.md`** | **JEPA Latent Representation Collapse** | Critical | Joint optimization of latent transition predictors on discrete formal tokens risks mapping all states to a trivial invariant manifold, corrupting value estimates. | Avoid using JEPA for policy updates; restrict JEPA strictly to a secondary pruning filter; enforce stop-gradients and VICReg covariance regularization. |
| **`proposal_minimal_v2.md`** | **Reasoning Boundary Contraction** | Critical | Pure policy-gradient optimization (GRPO) on small sample sizes ($K=4$) narrows policy entropy, failing to discover novel reasoning paths on hard Olympiad-level math. | Completely replace the $K=4$ GRPO rollout mechanism with Snell Adaptive MCTS (G-10), ensuring search lookahead actively expands exploration boundaries. |

---

## 6. Conclusive Winner Selection & Phase E Architecture Directive

### 6.1 Formal Winner Declaration
The Phase D Evaluation Board conclusively declares **`proposal_search_v2.md` (Expert-Iteration RSI / AlphaZero-Style MCTS)** as the **WINNING RSI SPINE** for the Aphasic Genius Engine.

### 6.2 Technical Justification for the Chief Architect
The selection of the Search-Driven Expert-Iteration spine is justified by four unassailable architectural realities:

1. **The Mathematical Compounding Imperative**:
   The central mandate of `BRIEF.md` is that the engine must continuously compound in intelligence without hitting a ceiling. As established by recent empirical literature (arXiv:2504.13837), pure reinforcement learning from verifiable rewards (RLVR) contracts policy support, while distillation caps the student at the teacher. Monte Carlo Tree Search is the *only* mathematically certified operator that expands exploration beyond the model's current sampling distribution. When combined with mechanical verifiers, MCTS generates verified reasoning paths that raw policy rollouts cannot sample, providing pristine, out-of-distribution training targets for Expert Iteration.
2. **The Hardware-Topology Synergy (The DeltaNet Search Moat)**:
   In standard Transformer models, tree search is economically impractical on free compute due to explosive KV-cache allocation. In Gated DeltaNet, memory state update is an $O(1)$ associative recurrence. Caching a 320 KB state matrix at each search node eliminates prefix recomputation entirely, unlocking high-throughput tree search on zero-cost Turing T4 hardware. This synergy makes search a natural, unfair hardware advantage for non-Transformers.
3. **Operational Robustness on 0 KRW Budget**:
   With a generation cycle time of 1.0 to 2.0 hours (yielding 10–12 generations per week), the Search spine operates in the optimal Goldilocks zone: fast enough to provide continuous learning feedback and prevent stale rollouts, yet deliberate enough to ensure that every generation reflects genuine, deep search amplification. It consumes only 24 of the 30 available weekly T4 GPU-hours and ~1 TPU-hour, leaving generous margins against cloud preemption.
4. **Flawless Scoreboard Clearance**:
   `proposal_search_v2.md` establishes the most rigorous, decontaminated path to conquering the GPT-3.5 scoreboard by the G-rule margin, backing every projection with binomial confidence intervals and verified baselines.

---

### 6.3 Directive for Phase E Chief Architect Blueprint
The Chief Architect is directed to immediately author `/home/user/hip/agy-blueprint/work/blueprint.md` using the following binding architectural parameters:

- **Codename**: `ARCHIMEDES-SEARCH` (or `ORRERY-SEARCH`).
- **Core Architecture**: ~78M Parameter Aphasic Reasoner consisting of:
  1. Compact BPE Input Tokenizer ($V_{\text{in}} = 16,384$) + 2D Multi-Axis Spatial Grid Adapter ($d=576$).
  2. 4-Layer Bidirectional Gated DeltaNet Reader ($d=576, d_{\text{ffn}}=1,376$).
  3. 8-Layer Recurrent Gated DeltaNet Reasoning Trunk (Weight-Tied, $r \in [1, 8]$ with learned adaptive halting and depth modulation embeddings $e_r$).
  4. Decision Heads: MCTS Action Policy Head ($V_{\text{formal}} = 2,048$), Shared Value Critic Head, PAIRED Regret Proposer Head, and DROP Span Pointer Head.
  5. Total Trainable Parameters executed at inference: strictly **$< 85,000,000$** (preserving $>15\text{M}$ safety margin below 100M cap).
- **Core RSI Loop**:
  1. Proposer-Solver curriculum guided by **Depth Regret ($R = \max(0, S_{\text{deep}} - S_{\text{shallow}})$)** and **Verifier-Grounded ELP**.
  2. **4-Layer Symmetric Reading Firewall** (Reverse procedural generation, 3-way paraphrase gating, counterfactual twin perturbation).
  3. **Snell Adaptive MCTS Search** (mean 18 simulations/task) accelerated by **G-10 Constant-Size Node State Caching** and **G-11 Latent Value Pruning**.
  4. External **Zero-Parameter Symbolic Skill Library (Stitch DSL)** compressing verified solutions into reusable macros.
  5. Policy update via **Expert Iteration (ExIt)** with depth-modulated distillation ($e_r$).
  6. Mandatory **Random-Reward Control Arm Gate** ($\ge +5.0\%$ margin) and **Domain Saturation Metric ($\Psi \ge 0.85$)** triggering domain ascension via **EDLI**.
- **Free Compute Allocation**:
  - Rollouts: 5,000 tasks $\times$ 18 mean sims on Kaggle 2× T4 GPUs ($\approx 42\text{ min}$).
  - Verification: Parallel execution in read-only `nsjail` sandboxes ($\approx 1\text{ min}$).
  - Training: 10M verified tokens on Kaggle TPU v5e-8 ($\approx 30\text{ s}$) or 2× T4 fallback ($\approx 7\text{ min}$).
  - Total generation cycle: **$\approx 1.2\text{ hours}$**, yielding **10–12 verified generations per week** at **0 KRW spend**.
- **Absolute Rule**: Maintain strictly **NO CODE** in all blueprint documentation.

---
*End of Phase D Judging & Comparative Evaluation Report (`work/judging.md`).*
