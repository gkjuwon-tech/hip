# PROJECT HEPHAESTUS: THE APHASIC GENIUS RSI ENGINE
## Master Technical Architecture Blueprint (FINAL, after Phase F audits: `audit_numbers.md`, `audit_completeness.md`)

**Codename**: `PROJECT HEPHAESTUS` (`HEPHAESTUS-RSI`)  
**Document Classification**: Architectural Standard & Master Technical Specification  
**System Target**: 81.65M-Parameter Attention-Free Non-Transformer Reasoning Core  
**Governing Documents**: `BRIEF.md`, `AUDIT_FROM_CLAUDE.md`, `work/judging.md`, `reason_baseline.md`  
**Execution Environment**: Zero-Capital (0 KRW) / Kaggle Free Tier (2× NVIDIA T4, TPU v5e-8) & RTX 3060 Laptop  
**Constraint Mandate**: Strict Prohibition on Executable Code / Pseudocode Scripts; All Formal Specifications via Equations, Schemas, Tables, and Plain-Language Procedures  

---

# CANONICAL NUMBERS SHEET

This Canonical Numbers Sheet serves as the unyielding, authoritative mathematical baseline for Project Hephaestus. Every architectural dimension, parameter tally, computational budget, empirical threshold, and benchmark target documented across this blueprint derives directly from the parameters certified in this master registry.

### Table 0.1: Master System Parameter Breakdown (< 100M Ceiling)

| Module Identifier | Architectural Specification & Layer Dimensions | Algebraic Parameter Formula | Executed Trainable Parameters | Status |
|---|---|---|---|:---:|
| **Input Token Embeddings** | Vocabulary $V_{\text{in}} = 16,384$, Hidden Dimension $d = 640$ | $V_{\text{in}} \times d$ | **10,485,760** | Fixed |
| **Formal IR Embeddings** | Formal Vocabulary $V_{\text{formal}} = 2,048$, Hidden Dimension $d = 640$ | $V_{\text{formal}} \times d$ | **1,310,720** | Fixed |
| **2D Spatial Grid Adapter** | Cellular-Automaton spatial conv ($3 \times 3$, 8 channels) + coord proj | $3 \times 3 \times 8 \times 8 + 8 \times d + 2d$ | **6,976** | Fixed |
| **Bidirectional Input Reader** | 2 Bi-Layers (4 directional Gated DeltaNet blocks, $d=640, d_{\text{ff}}=1600$) | $4 \times [(3+2+1)d^2 + 3d \cdot d_{\text{ff}} + 3 \times 4 \times d + 3d]$ | **22,156,800** | Fixed |
| **Recurrent Reasoning Core** | 8 Layers Weight-Tied Gated DeltaNet ($d=640, d_{\text{ff}}=1600$, depth $r \in [1, 8]$) | $8 \times [(3+2+1)d^2 + 3d \cdot d_{\text{ff}} + 3 \times 4 \times d + 3d]$ | **44,313,600** | Fixed |
| **Depth Modulation ($e_r$)** | 8 Learned Depth Vector Embeddings ($r \in [1, 8]$, dimension $d=640$) | $8 \times d$ | **5,120** | Fixed |
| **Policy Head ($\pi_\theta$)** | MCTS Action Projection: LayerNorm + Linear ($d \to V_{\text{formal}}$) | $d \times V_{\text{formal}} + V_{\text{formal}} + 2d$ | **1,314,048** | Fixed |
| **Value Head ($V_\phi$)** | Latent Critic MLP: LayerNorm + $d \to 256 \to 1$ with GELU activation | $d \times 256 + 256 + 256 \times 1 + 1 + 2d$ | **165,633** | Fixed |
| **Proposer Head ($\mathcal{P}$)** | Task Synthesis Head: LayerNorm + Linear ($d \to V_{\text{formal}}$) | $d \times V_{\text{formal}} + V_{\text{formal}} + 2d$ | **1,314,048** | Fixed |
| **DROP Pointer Head** | Dual-Span Boundary Pointer Projections: Start & End Affine Projections | $d \times d + 2d$ | **410,880** | Fixed |
| **Retriever Key Projection** | Dense Alignment Projection for External Knowledge Store: $d \to 256$ | $d \times 256 + 256 + 2d$ | **165,376** | Fixed |
| **External Memory / Tools** | BM25 Lexical Store, Python 3.12, SymPy 1.13, Z3 SMT, Lean 4, Stitch DSL | 0-Parameter Deterministic Mechanical Engines | **0** | Non-parametric |
| **TOTAL PARAMETERS** | **Sum of all trainable weights executed during inference** | **Verified Sum of All Module Rows** | **81,648,961** | **< 100M (18.35M Headroom)** |

### Table 0.2: Computational Envelope, Cycle Time & Quota Allocation (Per Generation)

| Metric Description | Exact Mathematical Derivation / Resource Limit | Value | Allocation Margin |
|---|---|---|:---:|
| **Active Parameters at Median Depth ($r=4$)** | $N_{\text{eff}} = N_{\text{non\_core}} + 4 \times N_{\text{core}} = 37.34\text{M} + 4 \times 44.31\text{M}$ | **214.59M active parameters** | — |
| **Inference FLOPs per Token ($r=4$)** | $2 \times N_{\text{eff}} = 2 \times 2.1462 \times 10^8$ | **$4.292 \times 10^8$ FLOPs/token** | Exact |
| **Training FLOPs per Token ($r=4$)** | $6 \times N_{\text{eff}} = 6 \times 2.1462 \times 10^8$ | **$1.288 \times 10^9$ FLOPs/token** | Exact |
| **MCTS Rollout Batch Size** | 5,000 tasks $\times$ 18 mean Snell simulations (search actions = program steps) $\times$ 100 formal tokens | **$9.0 \times 10^6$ tokens/generation** | Sized to T4 |
| **MCTS Rollout Compute** | $9.0 \times 10^6 \text{ tokens} \times 4.292 \times 10^8 \text{ FLOPs/token}$ | **$3.863 \times 10^{15}$ FLOPs** | Exact |
| **Rollout Wall-Clock on 2× T4** | 5,000 tasks / 2.0 tasks/sec sustained (batched memory streaming) | **0.694 hours (41.6 minutes)** | Within session |
| **Sandbox Tri-Verification Time** | 45,000 checks at ~40–150 sandboxed checks/sec across 4 Kaggle host vCPUs [추정] | **5–20 minutes (0.1–0.3 hours)** | Bounded |
| **Expert Iteration (ExIt) Training** | ≈3.0M verified tokens × 3 epochs = 9.0M token-passes $\times 1.288 \times 10^9 \text{ FLOPs/token}$ | **$1.16 \times 10^{16}$ FLOPs** | Exact |
| **ExIt Training Time on TPU v5e-8** | Used only for Stage 0 and occasional large retrains, not inside each generation (separate sessions, XLA compile) | **—** | Not in loop |
| **ExIt Training Time on 2× T4**| $1.16 \times 10^{16} \text{ FLOPs} / (3.250 \times 10^{13} \text{ FLOP/s sustained @ 25\% MFU})$ | **357 seconds (≈6 min)** | In loop |
| **Total Generation Cycle Wall-Clock (all on 2× T4)** | Rollout ($0.69\text{h}$) + Verify ($0.1$–$0.3\text{h}$) + Train ($0.1\text{h}$) + Buffer I/O ($0.2\text{h}$) | **$\approx 1.1$–$1.3 \text{ hours}$** | Fast Turnaround |
| **Weekly Operating Cadence** | ≈24 T4 session-hours for generations + ≈6 for evals per week | **10 to 12 generations / week [추정]** | 20% Buffer |
| **Stage 0 Bootloader Pretraining** | 5.0 Billion tokens $\times 1.288 \times 10^9 \text{ FLOPs/token} = 6.44 \times 10^{18} \text{ FLOPs}$ | **3.8–5.7 hours on Kaggle TPU v5e-8 (30%–20% MFU)** | One-Time Boot |

### Table 0.3: Official Reasoning Scoreboard Targets & Margins vs. GPT-3.5 ($G = T + \max(10, \frac{100-T}{3})$)

| Benchmark Identifier | Benchmark Domain & Protocol | Max GPT-3.5 ($T$) | Target Margin ($G$) | Expected Gen | Target Score | Projected Margin ($\Delta$) | 95% CI Lower Bound |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **A1: GSM8K** | Grade School Math (0-shot CoT) | 78.1 | **88.1** | Gen 4 [추정] | **89.2 [추정]** | +11.1 [추정] | 87.5 ($N=1,319$) |
| **A2: GSM-Plus** | Adversarial Number/Clause Perturbation | 61.2 | **74.1** | Gen 4 [추정] | **76.5 [추정]** | +15.3 [추정] | 75.7 ($N=10,552$) |
| **A3: MATH** | Full 5,000 Olympiad/High-School Problems | 43.1 [UNVERIFIED max; verified 34.1] | **62.1** | Gen 7 [추정] | **64.8 [추정]** | +30.7 [추정] | 63.5 ($N=5,000$) |
| **A4: SVAMP** | Word Problems with Varied Structures | 83.0 | **93.0** | Gen 3 [추정] | **94.2 [추정]** | +11.2 [추정] | 92.6 ($N=1,000$) |
| **A5: BBH** | BIG-bench Hard (27 Algorithmic Tasks) | 70.7 | **80.7** | Gen 6 [추정] | **82.3 [추정]** | +11.6 [추정] | 81.3 ($N=6,511$) |
| **A6: ARC-AGI-1** | Abstraction & Reasoning Corpus (pass@2) | 10.6 | **40.4** | Gen 6 [가설] | **40.5 [가설]** | +29.9 [가설] | 35.7 ($N=400$) |
| **A7: HumanEval+** | Synthesized Python Code (EvalPlus Test) | 70.7 | **80.7** | Gen 5 [추정] | **82.5 [추정]** | +11.8 [추정] | 76.5 ($N=164$) |
| **A8: MBPP+** | Basic Python Programming (EvalPlus Test) | 69.7 | **79.8** | Gen 5 [추정] | **81.2 [추정]** | +11.5 [추정] | 77.2 ($N=378$) |
| **A9: LiveCodeBench**| Fresh Contest Code (Post-Freeze Split) | 22.6 | **48.4** | Gen 8 [추정] | **51.2 [추정]** | +28.6 [추정] | 45.5 ($N=300$) |
| **A10: CRUXEval-O** | Python Code Output Execution Prediction | 63.3 | **75.5** | Gen 6 [추정] | **77.4 [추정]** | +14.1 [추정] | 74.4 ($N=800$) |
| **A11: DROP** | Discrete Reasoning over Paragraphs (F1) | 70.2 [UNVERIFIED max; verified 64.1] | **80.2** | Gen 6 [추정] | **81.8 [추정]** | +17.7 [추정] | 81.0 ($N=9,536$) |
| **A12a: FOLIO** | First-Order Logic Inferences | 62.7 | **75.2** | Gen 3 [추정] | **78.0 [추정]** | +15.3 [추정] | 72.1 ($N=204$) |
| **A12b: ProofWriter**| Multi-Step Logical Deduction (Depth-5) | 71.5 | **81.5** | Gen 3 [추정] | **84.5 [추정]** | +13.0 [추정] | 81.6 ($N=600$) |
| **A12c: PrOntoQA** | Fictional Ontology Deduction (Hardest) | 85.0 | **95.0** | Gen 4 [추정] | **96.5 [추정]** | +11.5 [추정] | 94.8 ($N=500$) |
| **A13a: LogiQA 2.0**| Reading Comprehension Logic Puzzles | 52.4 | **68.2** | Gen 7 [추정] | **70.5 [추정]** | +18.1 [추정] | 66.8 ($N=1,000$) |
| **A13b: ReClor** | Logical Reasoning Aptitude Test (dev) | 57.4 | **71.6** | Gen 7 [추정] | **73.8 [추정]** | +16.4 [추정] | 69.9 ($N=500$) |
| **A14a: MuSR MM** | Murder Mysteries Complex Context Mystery | 70.0 | **80.0** | Gen 7 [추정] | **81.5 [추정]** | +11.5 [추정] | 76.6 ($N=250$) |
| **A14b: MuSR OP** | Object Placement Spatial Tracking | 58.2 | **72.1** | Gen 7 [추정] | **74.0 [추정]** | +15.8 [추정] | 68.6 ($N=256$) |
| **A14c: MuSR TA** | Team Allocation Combinatorial Constraint | 78.8 | **88.8** | Gen 7 [추정] | **90.2 [추정]** | +11.4 [추정] | 86.3 ($N=250$) |
| **A15: Game of 24** | Combinatorial Arithmetic Tree Search | 19.0 | **46.0** | Gen 2 [추정] | **98.0 [추정]** | +79.0 [추정] | 95.2 ($N=100$) |
| **A16: Blocksworld**| PDDL Classical Planning from English | 6.1 | **37.4** | Gen 3 [추정] | **68.5 [추정]** | +62.4 [추정] | 64.6 ($N=600$) |
| **A17: KMMLU-Math**| Korean CSAT/University Level Math | 30.3 | **53.5** | Gen 8 [추정] | **56.0 [추정]** | +25.7 [추정] | 51.5 ($N=700$) |
| **B1: GPQA-Diamond**| Graduate-Level Science (Offline Datastore)| 30.6 | **53.7** | Gen 10 [추정] | **55.6 [추정]** | +25.0 [추정] | 48.6 ($N=198$) |
| **B2: MedQA** | USMLE Medical Board Exams (4-choice) | 63.4 | **75.6** | Gen 9 [추정] | **77.2 [추정]** | +13.8 [추정] | 74.9 ($N=1,273$) |
| **B3: StrategyQA** | Multi-hop Implicit Question Answering | 70.3 [UNVERIFIED max; verified 51.5] | **80.3** | Gen 7 [추정] | **82.0 [추정]** | +30.5 [추정] | 79.8 ($N=2,290$) |

### Table 0.4: Primary Operational & Safety Thresholds

| System Metric | Operational Threshold Condition | Mandatory System Action Upon Threshold Breach |
|---|---|---|
| **Rocket Stage 1 Cord-Cut** | $\text{TE}_g \ge 0.25 \land \text{TE}_g > \text{TE}_{g-1} > \text{TE}_{g-2} \land \Delta_{\text{rand}} \ge +5.0\%$ | Permanently sever teacher training data stream; transition to 100% self-generated data. |
| **Random-Reward Gate** | $\Delta_{\text{rand}} = \text{Score}_{\text{verifier}}(g) - \text{Score}_{\text{random\_reward}}(g) < +5.0\%$ | Invalidate generation; abort weight update; purge contaminated replay buffer shard. |
| **In-Loop / Held-Out Gap ($H_g$)** | $H_g = S_g(\mathcal{D}_{\text{in-loop}}) - S_g(\mathcal{D}_{\text{held-out}}) > 5.0 \text{ points}$ | Emergency circuit breaker: halt training pipeline; assert verifier SHA-256 hashes; rollback to $g-1$. |
| **Search Internalization ($E_g$)** | $E_g = \text{Rollouts}_{g-1}(80\%) / \text{Rollouts}_g(80\%) \le 1.0$ | Adjust policy distillation temperature; inject $+50\%$ Dirichlet root exploration noise. |
| **Domain Saturation Metric ($\Psi$)** | $\Psi_D = \frac{\text{pass@1}(D)}{\text{pass@64}(D)} \times \left(1 - \frac{\Delta\text{DL}}{\text{DL}}\right) \ge 0.85$ (for 2 generations) | Trigger autonomous domain ascension; activate next verifier environment via EDLI prioritization. |
| **T4 Memory Bandwidth Latency** | Streaming $429.2\text{ MB}$ weights (no on-chip reuse assumed) at $230\text{ GB/s}$ sustained throughput | $\mathbf{0.202 \text{ s}}$ greedy (100 tokens, $r=4$); $\mathbf{0.90\text{--}1.25 \text{ s}}$ Snell MCTS (18 sims, $B=16$). |
| **Decoupled Interpreter Renderer** | Blindfolded **Qwen3-1.7B** (`enable_thinking=False`, zero task prompt) | Must achieve $\ge 99.0\%$ adherence on Counterfactual Swap Test; $0\text{ KRW}$ scoring influence. |

---

# 1. Executive Summary & System Philosophy

### 1.1 The Aphasic Genius: Decoupling Pure Deliberation from Conversational Rhetoric
Contemporary artificial intelligence research is dominated by a pervasive, computationally wasteful conflation: the assumption that reasoning capability is inseparable from natural language fluency. Modern autoregressive Large Language Models expend massive fractions of their parametric capacity and operational compute budgets on conversational rhetoric—generating polite conversational padding, navigating syntactic agreement, handling multilingual conversational nuances, and projecting hidden activations across vast token vocabularies ($V > 128,000$, frequently consuming 20% to 50% of total weights). 

Project Hephaestus executes a radical, uncompromising pivot: **the absolute elimination of natural language generation from the core learner**. The reasoning core is formulated as an **"Aphasic Genius"**—an intelligence possessing dense, acute natural language comprehension, but rendered entirely mute in natural prose. It reads and comprehends complex textual problem descriptions, extracts formal invariants, and deliberates entirely in structured representations:
1. The core model allocates **100% of its trainable parameter capacity into pure deductive reasoning**, mathematical synthesis, and symbolic verification.
2. It emits **formal representations only**: discrete AST programs, symbolic expressions, Lean 4 tactic sequences, First-Order Logic clauses, SAT/SMT assertions, and pointer-copy spans.
3. Conversational human interpretation is delegated entirely to an external, off-the-shelf, open-weights model: the **"Interpreter" (통역사)**. Analagous to Stephen Hawking’s speech synthesizer or a chess grandmaster’s official scribe, the interpreter translates the core's formal execution traces into human prose behind a strict, cryptographically auditable firewall.
4. Evaluation of reasoning is conducted **strictly on the core's formal output**. The interpreter is never permitted to participate in benchmark scoring, eliminating conversational hallucination from the scientific measurement of intelligence.

### 1.2 Rocket Stage Separation: The Teacher Ceiling vs. Mechanical Verifiers
Supervised fine-tuning and distillation from frontier foundation models represent a fundamental theoretical trap. Minimizing Kullback-Leibler divergence or token-level cross-entropy against a teacher distribution $P_{\text{teacher}}$ confines the student within the teacher's cognitive support manifold:
$$\text{supp}(P_{\text{student}}) \subseteq \text{supp}(P_{\text{teacher}})$$
Distillation copies heuristics, biases, and hallucinations. Crucially, contemporary frontier models—including **Claude Opus 5.5, Gemini 4 Argon, and GPT-6 Astra**—remain brittle "tin cans" (깡통) when measured against the true North Star of Artificial Superintelligence:
- They hallucinate false mathematical steps when pushed across long-horizon compositional deductions.
- They experience catastrophic collapse under minor adversarial perturbations: inserting a single irrelevant, distractor clause into an arithmetic word problem degrades frontier LLM accuracy by up to 65% [검증됨: https://arxiv.org/abs/2410.05229].
- They cannot certify the formal validity of their own internal reasoning steps.

In practice, a student trained only to imitate a teacher stays close to that teacher's distribution [가설: the strict support-containment formula above is an idealization; arXiv 2504.13837 reports that distillation can add new reasoning patterns while RLVR narrows the reasoning boundary at large pass@k, https://arxiv.org/abs/2504.13837v4]. That second finding is why Hephaestus improves by expert iteration with MCTS over exact verifiers, where search expands coverage, rather than by plain RLVR sharpening. Therefore, Project Hephaestus treats teacher distillation strictly as **Rocket Stage 1**: a temporary, finite booster utilized exclusively to endow the 81.6M core with basic formal syntax and non-zero ground-truth sampling probability. 

Once the vehicle achieves aerodynamic liftoff, **Stage 1 is permanently detached and discarded**. Improvement beyond the bootloader is fueled exclusively by **0-parameter exact mechanical verifiers**:
- **Python 3.12 Sandboxes**: Executing unit tests and checking programmatic invariants.
- **SymPy Computer Algebra System**: Certifying algebraic equivalence via symbolic reduction ($\text{simplify}(A - B) == 0$).
- **Z3 SMT Solver**: Checking First-Order Logic satisfiability and emitting formal countermodels.
- **Lean 4 Interactive Theorem Prover**: Mechanically verifying mathematical proofs through dependent type theory.
- **VAL Planning Validator**: Checking classical domain planning states against PDDL axioms.

A mechanical verifier does not fatigue, flatter, or hallucinate. The rules of Go possess zero conversational eloquence, yet pure self-play guided solely by Go’s exact rules enabled AlphaGo Zero to defeat human-data-trained AlphaGo 100–0 [검증됨: https://deepmind.google/blog/alphago-zero-starting-scratch/]. Exact verifiers possess an infinite ceiling. By grounding self-improvement in mechanical checkers rather than human or LLM approval, Project Hephaestus establishes an unceilinged, compounding trajectory toward superintelligence.

### 1.3 RSI as the Definition of True Scalability: The S-Curve Staircase
True scalability is not measured by the static benchmark score of a frozen checkpoint; it is defined by the velocity and durability of the engine that manufactures intelligence:
$$\text{Intelligence Velocity} = \frac{d(\text{Capability})}{d(\text{Generation})}, \quad \text{Acceleration} = \frac{d^2(\text{Capability})}{d(\text{Generation})^2}$$
Conquering GPT-3.5 on reasoning benchmarks is merely the **first slime encountered outside the tutorial village**—a mandatory operational milestone, but conceptually trivial compared to the broader horizon. The Hephaestus ladder ascends monotonically:
$$\text{GPT-3.5 Slime} \longrightarrow \text{GPT-4 Level} \longrightarrow \text{o1/R1 Parity} \longrightarrow \text{Frontier Tin Cans} \longrightarrow \text{Top Human Scientists} \longrightarrow \text{ASI}$$

To sustain compounding progress without stalling against the finite limits of any single formal domain, the architecture implements an **S-Curve Domain Staircase** (analogous to the successive lithographic wavelengths that sustained Moore's Law for decades). When learning velocity within an existing verifier domain saturates, the engine autonomously transitions compute to higher-order formal environments:
$$\text{Arithmetic \& CAS} \longrightarrow \text{Unit-Tested Code} \longrightarrow \text{SMT Logic} \longrightarrow \text{Axiomatic Lean 4} \longrightarrow \text{Molecular Simulators}$$
By coupling Monte Carlo Tree Search, an external non-parametric skill library, and autonomous domain ascension under strict 0-KRW compute constraints, Project Hephaestus provides the complete blueprint for an autonomous, self-compounding reasoning engine.

---

# 2. The Synthesized RSI Engine

Project Hephaestus synthesizes the winning **Search-Driven MCTS Spine** (`proposal_search_v2.md`) with all 13 breakthrough mechanisms identified across the competing architectural candidates (`proposal_library_v2.md`, `proposal_depth_v2.md`, `proposal_curiosity_v2.md`, `proposal_minimal_v2.md`) per the Master Grafting Matrix in `work/judging.md`.

```
====================================================================================================
               PROJECT HEPHAESTUS: AUTONOMOUS RSI PIPELINE DATAFLOW
====================================================================================================

      +---------------------------------------------------------------------------------------+
      | 1. DUAL-CURRICULUM TASK GENERATION                                                    |
      | - 50% Stepped Procedural Generator Stream (Reasoning Gym, re-ARC, SymPy, PlanBench)   |
      | - 50% Autonomous Proposer Regret Head (PAIRED): Rewarded by Depth Regret              |
      |   [G-02] R_depth(x) = max(0, Score_deep(x) - Score_shallow(x))                        |
      | - [G-12] Verifier-Grounded ELP: Multiplies prediction delta by deterministic flag     |
      |   ELP_g(x) = |E_g(x) - E_{g-1}(x)| * I(v(x) is verifiable) -> Discards Noisy TV      |
      +-------------------------------------------┬-------------------------------------------+
                                                  │ Proposed Task x = (x_text, Spec)
                                                  v
      +---------------------------------------------------------------------------------------+
      | 2. G-13 FOUR-LAYER SYMMETRIC READING FIREWALL                                         |
      | - Reverse Procedural Generation: Ground-truth AST P* synthesized before NL text       |
      | - 3-Way Paraphrase Invariance: Core must yield isomorphic AST across x^(1), x^(2), x^(3)|
      | - Counterfactual Perturbation Twins: Verifies invariant & controlled output shifts    |
      | - Robust Twin Gating: Drops candidate if sensitive to cosmetic variable renaming     |
      +-------------------------------------------┬-------------------------------------------+
                                                  │ Certified Comprehension Stream
                                                  v
      +---------------------------------------------------------------------------------------+
      | 3. CORE REASONER DELIBERATION (81.6M Gated DeltaNet)                                  |
      | - 2-Layer Bidirectional Reader (d=640) + ARC 2D CA Spatial Adapter                    |
      | - [G-10] Constant-Size Recurrent State: S_t in R^{640x640} (O(1) Memory, NO KV-CACHE) |
      | - [G-03] Depth-Embedding Modulation: Policy conditioned on depth vector e_r           |
      | - [G-04] Zero-Parameter External Skill Library: Pre-indexed Stitch DSL macros loaded  |
      +-------------------------------------------┬-------------------------------------------+
                                                  │ Latent States S_t & Action Priors P(a|s)
                                                  v
      +---------------------------------------------------------------------------------------+
      | 4. SNELL ADAPTIVE MCTS SEARCH & VALUE LATENT PRUNING                                  |
      | - Snell Compute-Optimal Budget: 8 to 64 sims/task based on root entropy (mean 18 sims)|
      | - [G-11] Value-Head Latent Pruning: Discards branches with v_hat < 0.2 in latent space |
      |   Bypasses sandbox dispatch for most non-viable trajectories [가설]              |
      | - PUCT Tree Traversal over Formal Token Vocabulary (V_formal = 2,048)                 |
      +-------------------------------------------┬-------------------------------------------+
                                                  │ Candidate Formal Programs P_cand
                                                  v
      +---------------------------------------------------------------------------------------+
      | 5. IMMUTABLE SANDBOXED MECHANICAL VERIFICATION                                        |
      | - Isolated Read-Only Containers (Python 3.12, SymPy 1.13, Z3 SMT, Lean 4)             |
      | - Dual Verification: Runtime Output Execution + Formal AST Equivalence                |
      |   SymPy(P_cand - P*) == 0  or  Z3(P_cand <=> P*) == SAT                               |
      | - Emits Binary Ground-Truth Reward z in {-1, +1}                                      |
      +-------------------------------------------┬-------------------------------------------+
                                                  │ Verified Traces (s, pi_search, z=+1)
                                                  v
      +---------------------------------------------------------------------------------------+
      | 6. EXPERT ITERATION (ExIt) WEIGHT UPDATE & RATCHET                                    |
      | - Policy Update: Distill MCTS visit distribution pi_search into shallow policy        |
      | - [G-08] Search Internalization Metric: E_g = Rollouts_{g-1}(80%) / Rollouts_g(80%)   |
      | - [G-09] Description Length Compression: CR = DL_raw / DL_compressed                  |
      | - [G-01] Random-Reward Control Arm Gate: Margin Delta_rand >= +5.0% enforced          |
      | - [G-06] Domain Saturation Trigger: Psi_D >= 0.85 triggers ascension via [G-07] EDLI  |
      +---------------------------------------------------------------------------------------+
====================================================================================================
```

### 2.1 The Generation Loop Step-by-Step

Each recursive generation $g \to g+1$ operates across six strictly pipelined stages, designed to execute continuously on Kaggle free quotas without manual intervention:

#### Stage 1: Dual-Curriculum Generation & Noise Filtering
To eliminate both the human curriculum bottleneck and the fatal "Own-Level Prison" (where an autonomous proposer only synthesizes problems it can already solve trivially):
1. **50% Stepped Procedural Ingestion**: Half the task volume is generated by deterministic external procedural libraries (Reasoning Gym, re-ARC, SymPy equation generators, CLRS algorithmic graph generators, and PlanBench planning generators). Task difficulty parameters (e.g., graph vertex count, equation degree, planning horizon) are stepped upward by a deterministic governor whenever the core's pass@1 solve rate exceeds 80.0%.
2. **50% Autonomous Proposer Regret Curriculum (PAIRED)**: The core's proposer head $\mathcal{P}$ emits task specifications $(x_{\text{spec}}, \text{constraints})$. In compliance with **[G-02] Depth Regret**, tasks are evaluated under shallow policy execution ($r=1$, greedy) and deep MCTS search ($r=8$, 64 simulations):
   $$R_{\text{depth}}(x) = \max\left(0, \text{Score}_{\text{MCTS\_deep}}(x) - \text{Score}_{\text{Policy\_shallow}}(x)\right)$$
   Tasks that the shallow policy already solves yield $R=0$ (trivial waste). Tasks that deep MCTS cannot solve yield $R=0$ (impossible noise). The proposer is rewarded strictly for maximizing $R_{\text{depth}}$, forcing it to generate problems exclusively within the system's **Zone of Proximal Development**.
3. **[G-12] Verifier-Grounded Empirical Learning Progress (ELP)**: To resolve the classic "Noisy TV" problem (where curiosity engines chase irreducible stochastic noise):
   $$\text{ELP}_g(x) = \left| \mathcal{E}_g(x) - \mathcal{E}_{g-1}(x) \right| \cdot \mathbb{I}(v(x) \text{ is deterministically verifiable})$$
   Any task whose outcome is non-deterministic, noisy, or unverified has its progress weight multiplied by 0, forcing exploration strictly toward reducible deductive error.

#### Stage 2: The Four-Layer Symmetric Reading Firewall [G-13]
The Achilles' heel of an aphasic reasoner is reading comprehension. A program execution sandbox verifies only that an emitted code block runs and outputs a value; it does *not* verify that the program accurately reflects the natural language problem statement. A corrupt core can learn to misread the question into a simpler problem and still achieve an execution pass. Hephaestus permanently closes this hole via a 4-layer reading harness:
1. *Program-First Reverse Procedural Generation*: For synthetic tasks, the generator first samples a verified AST $P^*$ and target solution $y$, then renders them into natural language text $x_{\text{text}}$ via certified attribute grammars. The core's emitted AST $P_{\text{core}}$ is verified not only by execution, but by formal symbolic isomorphism against $P^*$ using SymPy or Z3:
   $$\text{Verification} = \left(\text{Exec}(P_{\text{core}}) == y\right) \land \left(\text{Simplify}_{\text{SymPy}}(P_{\text{core}} - P^*) == 0\right)$$
2. *Three-Way Paraphrase Invariance*: Natural language prompts are presented in three linguistically divergent surface forms $(x^{(1)}, x^{(2)}, x^{(3)})$, generated via rule-based syntactic transformations (active/passive inversion, clause re-ordering, vocabulary swapping). The core must emit semantically identical ASTs across all three variants:
   $$\text{Paraphrase\_Pass} = \mathbb{I}\left(P_{\text{core}}(x^{(1)}) \equiv P_{\text{core}}(x^{(2)}) \equiv P_{\text{core}}(x^{(3)})\right)$$
3. *Counterfactual Perturbation Twin Gating*: Numbers, entity names, and conditions are systematically mutated (GSM-Symbolic / GSM-Plus protocol). The core's emitted program must exhibit the exact mathematically predicted functional mutation.
4. *Robust Twin Gating*: Insertion of irrelevant distractor clauses (NoOp operations) must produce zero variance in the emitted AST logic.

#### Stage 3: Constant-State Core Processing & Depth Modulation [G-03, G-10]
The sanitized problem tokens are processed by the 2-layer Bidirectional Input Reader ($d=640$), generating contextual representations $H_{\text{ctx}}$. During token-by-token formal program synthesis, the 8-layer weight-tied Gated DeltaNet reasoning trunk unrolls over the sequence.
- **[G-10] Constant-Size Recurrent State**: Unlike Transformers that accumulate quadratic KV-caches requiring dynamic memory allocation per search node, Gated DeltaNet maintains a strictly constant-size recurrent associative matrix $S_t \in \mathbb{R}^{d \times d}$ ($\approx 160\text{ KB}$ per layer in fp16, multi-head). MCTS node expansion caches $S_t$ directly in tree memory with $\mathcal{O}(1)$ footprint, eliminating prefix recomputation and which should speed up tree search on Turing T4 GPUs (size of the gain unmeasured [가설]).
- **[G-03] Depth-Embedding Modulation ($e_r$)**: The recurrent core is conditioned on learned depth vectors $e_r$ ($r \in [1, 8]$). This enables the policy to operate flexibly in shallow mode ($r=2$) during rapid filtering, or deep deliberation mode ($r=8$) during high-entropy search.

#### Stage 4: Snell Adaptive MCTS & Latent Value Pruning [G-11]
Search actions are **program steps** (one Python line, one SymPy rewrite, one Lean tactic, one SMT assertion); token decoding inside a step is done by the policy, so 18–64 simulations explore meaningful alternatives rather than single tokens [audit fix]. Tree policy is PUCT (Predictor + UCT):
$$a_t = \arg\max_a \left( Q(s, a) + c_{\text{puct}} \cdot P(a|s) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right)$$
where $c_{\text{puct}} = 1.25$, and Dirichlet exploration noise ($\alpha = 0.3, \epsilon = 0.25$) is injected at the root.
- **Snell Compute-Optimal Simulation Sizing**: Rather than allocating a rigid simulation count, search budget scales with root value entropy $H(V)$. High-confidence tasks ($H(V) < 0.2$) exit early at 8 simulations; high-entropy, difficult tasks ($H(V) \ge 0.5$) receive up to 64 simulations. Population mean search budget is locked at **18 simulations per task**.
- **[G-11] Value-Head Latent Pruning**: Before dispatching candidate ASTs to the external operating system sandbox for execution, the latent value critic evaluates candidate state $s$:
   $$\text{If } \hat{V}(s) < 0.20 \implies \text{Prune branch immediately (bypass sandbox execution)}$$
   This latent filter is expected to discard a majority of doomed trajectories inside GPU memory [가설; measured in Experiment 1], eliminating IPC subprocess bottlenecks.
- **[G-04] Integration of Zero-Parameter Skill Library**: Discovered Stitch DSL macro combinators are injected directly into the formal grammar. Calling a pre-compiled macro collapses an entire 20-step reasoning subtree into a single tree edge, exponentially shortening search depth.

#### Stage 5: Immutable Mechanical Sandbox Verification
Candidate trajectories reaching terminal formal states are dispatched to isolated, read-only `nsjail` execution sandboxes (Python 3.12, SymPy 1.13, Z3 SMT, Lean 4). The verifier evaluates correctness, returning terminal reward $z \in \{-1, +1\}$. Process execution is capped at a 2.0-second timeout and 256 MB RAM limit per task.

#### Stage 6: Expert Iteration Weight Update & Metrics Auditing [G-01, G-08, G-09]
Verified successful trajectories ($z = +1$) are staged in the Expert Replay Buffer $\mathcal{D}_g^*$.
1. **Network Retraining**: The core policy $\pi_\theta$ and value critic $V_\phi$ are updated via joint cross-entropy and MSE loss over $\mathcal{D}_g^*$:
   $$\mathcal{L}(\theta, \phi) = \mathbb{E}_{(s, \pi, z) \sim \mathcal{D}_g^*} \left[ -\sum_{a \in V_{\text{formal}}} \pi(a) \log P_\theta(a|s) + \alpha (z - V_\phi(s))^2 \right] + \lambda_{\text{prop}} \mathcal{L}_{\text{proposer}}$$
2. **[G-08] Search Internalization Verification**: The system evaluates the internalization ratio:
   $$E_g = \frac{\text{Rollouts}_{g-1}(80\%)}{\text{Rollouts}_g(80\%)} > 1.0$$
   A value $E_g > 1.0$ certifies that the policy has internalized search trajectories, allowing shallow greedy passes in generation $g+1$ to achieve the accuracy previously reachable only via deep MCTS.
3. **[G-09] Description Length Compression**: The Stitch engine factors recurring AST subgraphs across $\mathcal{D}_g^*$, measuring compression ratio $CR = \text{DL}_{\text{raw}} / \text{DL}_{\text{compressed}}$. If $CR \ge 1.30$, extracted macros are committed to the immutable external library.
4. **[G-01] Control Arm Enforcement**: The updated checkpoint must outperform a parallel random-reward control arm by $\ge +5.0\%$ on held-out evals.

---

### 2.2 RSI Levels L1 Through L5

Project Hephaestus defines five discrete evolutionary tiers, establishing clear boundaries between autonomous weight updates and human-gated meta-modifications:

| RSI Level | Operational Scope | Autonomous Mechanisms | Human-in-the-Loop Governance | Parameter Footprint |
|---|---|---|---|:---:|
| **Level 1 (L1)** | **Policy & Value Weight Evolution** | Expert Iteration retraining on verified self-generated search buffer $\mathcal{D}_g^*$; AdamW optimization. | Fully autonomous; zero human intervention; bounded by automated circuit breakers. | 81.6M (Fixed) |
| **Level 2 (L2)** | **Curriculum Co-Evolution** | Asymmetric Proposer-Solver self-play (PAIRED); Depth Regret optimization ($R_{\text{depth}}$); procedural generator parameter stepping. | Fully autonomous; bounded by solve-rate band $[35\%, 65\%]$ and held-out sanity checks. | 81.6M (Fixed) |
| **Level 3 (L3)** | **Skill Library Accumulation** | Stitch functional AST abstraction; e-graph anti-unification; zero-parameter Python/SymPy macro indexing. | Fully autonomous; library verified against static test assertions; syntax-checked by AST parser. | 81.6M Neural + $\infty$ Symbolic |
| **Level 4 (L4)** | **Procedural Meta-Optimization** | Evolutionary search over MCTS hyperparameters ($c_{\text{puct}}$, Dirichlet $\alpha$), procedural generator grammars, and scan kernels. | **MANDATORY HUMAN APPROVAL**: Proposed diffs presented as Git patches; human review required before merge. | 81.6M (Fixed) |
| **Level 5 (L5)** | **Architectural Self-Redesign** | Non-Transformer Neural Architecture Search (NAS); dynamic channel/layer expansion; custom CUDA scan synthesis. | **LOCKED**: research-only under human governance, air-gapped sandbox, never automatic. | Variable ($<100\text{M} \to 1\text{B}$) |

---

### 2.3 Teacher Bootloader and the Cord-Cut Criterion (Rocket Stage Separation)

The Stage 0 bootloader endows the 81.6M core with the capability to parse natural language tokens into valid AST syntax. To preserve scientific integrity and adhere to licensing laws, **zero tokens derived from OpenAI models (GPT-3.5, GPT-4, GPT-4o) are permitted in the training corpus**. All pretraining data originates from permissively licensed open models (Llama-3.1-405B, Mixtral-8x7B) and procedural engines.

#### The Cord-Cut Reference Teacher
In compliance with `AUDIT_FROM_CLAUDE.md` §1.5, the teacher baseline used to measure exceedance must be physically runnable on free compute or evaluated from an immutable failure manifest. We establish **Qwen2.5-Math-7B-Instruct** (runnable on Kaggle 2× T4 in 4-bit AWQ) as the official Cord-Cut Reference Teacher. A static test manifest of 2,000 problems failed by Qwen2.5-Math-7B-Instruct across 8 independent greedy sampling runs ($\mathcal{D}_{\text{held-out}}^{\text{teacher\_fail}}$) is frozen before self-improvement begins.

#### The Triple-Lock Cord-Cut Criterion
Rocket Stage 1 separation occurs permanently when the engine satisfies three simultaneous conditions across two consecutive generations:
$$\text{Condition 1 (Teacher Exceedance)}: \quad \Delta_{\text{TE}}(g) = \frac{|\{x \in \mathcal{D}_{\text{held-out}}^{\text{teacher\_fail}} : \text{Solve}(x) = 1\}|}{|\mathcal{D}_{\text{held-out}}^{\text{teacher\_fail}}|} \ge 0.25$$
$$\text{Condition 2 (Monotonic Acceleration)}: \quad \Delta_{\text{TE}}(g) > \Delta_{\text{TE}}(g-1) > \Delta_{\text{TE}}(g-2)$$
$$\text{Condition 3 (Random-Reward Control Gap)}: \quad \text{Score}_{\text{verifier}}(g) - \text{Score}_{\text{random\_reward}}(g) \ge +5.0\%$$

When all three conditions hold, the teacher cord is irreversibly severed:
1. The Stage 0 pretraining corpus and teacher distillation loss functions are permanently unmounted.
2. The model enters 100% autonomous operation, sustained exclusively by procedural generators, proposer regret loops, MCTS tree search, and exact verifiers.
3. A cryptographic attestation (recording the git commit, model weights SHA-256, and evaluation logs) is published to the project ledger.

---

### 2.4 The S-Curve Staircase of Verifier Domains

Autonomous progress compounds within a specific verifier domain until the model captures the full deductive entropy of that domain. Continuing to train on a saturated domain results in compute stagnation. Hephaestus navigates this via an autonomous **S-Curve Staircase**:

```
Deductive
Capability
    ^                                                          [Domain 5: Molecular Simulators]
    |                                                  +---/  (RDKit, OpenMM, AutoDock Vina)
    |                                          +---/   [Domain 4: Interactive Proofs]
    |                                  +---/   (Lean 4 Axiomatic Theorem Proving)
    |                          +---/   [Domain 3: First-Order SMT Logic]
    |                  +---/   (Z3 SMT, FOLIO, ProofWriter, Satisfiability)
    |          +---/   [Domain 2: Algorithmic Code Synthesis]
    |  +---/   (Python 3.12, Unit Tests, LiveCodeBench, CruxEval)
    | /        [Domain 1: Combinatorial Arithmetic & CAS]
    |          (Game of 24, SymPy, GSM-Symbolic, SVAMP)
    +-----------------------------------------------------------------------------------> Generation
```

#### Autonomous Domain Ascension Protocol [G-06, G-07]
To eliminate human bias while respecting BRIEF.md §2.7 (the model can never write or modify its own verifiers):
1. **The Isolated Verifier Registry**: All five domain sandboxes reside pre-installed in read-only container partitions with immutable execution APIs.
2. **[G-06] Domain Saturation Metric ($\Psi_D$)**: Learning saturation in active domain $D$ is declared when:
   $$\Psi_D(g) = \frac{\text{pass@1}(D)}{\text{pass@64}(D)} \times \left(1 - \frac{\Delta\text{DL}(\mathcal{P}_g)}{\text{DL}(\mathcal{P}_g)}\right) \ge 0.85$$
   holds across two consecutive generations. This signifies that greedy policy execution has captured $\ge 85\%$ of amplified search capacity, and symbolic macro extraction ($\Delta\text{DL}$) has plateaued.
3. **[G-07] Empirical Domain Learnability Index (EDLI)**: When domain $D$ saturates, rollout compute allocation shifts dynamically to candidate domains $D'$ prioritized by learning velocity:
   $$\text{EDLI}(D') = \frac{\Delta\text{Accuracy}(D')}{\text{GPU-Hour}} \quad [\text{AUDIT §3.7}]$$
4. **Autonomous Domain Reduction Compilers**: The core explores the next domain by synthesizing typed reduction compilers that lower problems in $D_{k+1}$ into proven base domains $D_1 \dots D_k$ (e.g., reducing planning problems in Domain 2 into propositional SAT instances in Domain 3, or reducing kinematic physics in Domain 5 into symbolic differential equations in Domain 1).

---

### 2.5 Generation Cycle Time & Explicit FLOP Accounting on Free Quota

All computational budgets are rigorously derived using true recurrent depth multipliers and physical hardware bandwidth limits, directly resolving the undercounting errors identified in `AUDIT_FROM_CLAUDE.md` §1.1 and §1.2.

#### Real Recurrent FLOP Formulations
The reasoning core executes $N_{\text{core}} = 44,313,600$ recurrent weights unrolled across recurrent depth $r \in [1, 8]$, alongside $N_{\text{non\_core}} = 37,335,361$ non-recurrent weights (embeddings, reader, heads).
- **Inference FLOPs per Token**:
  $$\text{FLOPs}_{\text{inf}}(r) = 2 \times \left(N_{\text{non\_core}} + r \times N_{\text{core}}\right)$$
  $$\text{At median recursion } (r=4): \quad 2 \times (37.38\text{M} + 4 \times 44.31\text{M}) = 2 \times 2.1462 \times 10^8 = \mathbf{4.292 \times 10^8 \text{ FLOPs/token}}$$
- **Training FLOPs per Token**:
  $$\text{FLOPs}_{\text{train}}(r) = 6 \times \left(N_{\text{non\_core}} + r \times N_{\text{core}}\right)$$
  $$\text{At median recursion } (r=4): \quad 6 \times (37.38\text{M} + 4 \times 44.31\text{M}) = 6 \times 2.1462 \times 10^8 = \mathbf{1.288 \times 10^9 \text{ FLOPs/token}}$$

#### Per-Generation Computational Execution (5,000 Tasks)
1. **Snell Adaptive MCTS Rollouts (Kaggle 2× T4 GPUs)**:
   - Batch size: 5,000 tasks $\times 18$ mean Snell simulations $\times 100$ formal tokens $= 9.0 \times 10^6$ tokens.
   - Total Rollout Compute: $9.0 \times 10^6 \times 4.292 \times 10^8 = 3.863 \times 10^{15} \text{ FLOPs}$.
   - **T4 Physical Latency Bounds**: NVIDIA T4 features $300\text{ GB/s}$ peak bandwidth ($230\text{ GB/s}$ sustained). Sequential token generation streaming $429.2\text{ MB}$ of active weights at $r=4$ requires $1.87\text{ ms/token}$. With batched action expansion ($B=16$), weight streaming is amortized across beams. Sustained rollout throughput is $2.0\text{ tasks/second}$ across 2× T4 GPUs:
     $$\text{Time}_{\text{rollout}} = \frac{5,000 \text{ tasks}}{2.0 \text{ tasks/s}} = 2,500 \text{ seconds} \approx \mathbf{0.694 \text{ hours (41.6 minutes)}}$$
2. **Sandboxed Mechanical Verification (Kaggle 4 vCPUs)**:
   - 45,000 sandbox executions (accounting for AST cache hits) at roughly 40–150 sandboxed checks/sec [추정]:
     $$\text{Time}_{\text{verify}} \approx 5\text{–}20 \text{ minutes} \; (0.1\text{–}0.3 \text{ hours})$$
3. **Expert Iteration Retraining (on the same 2× T4 session)**:
   - ≈3.0M verified formal tokens × 3 epochs = 9.0M token-passes $\times 1.288 \times 10^9 \text{ FLOPs/token} = 1.16 \times 10^{16} \text{ FLOPs}$.
   - On 2× T4 GPUs ($32.5\text{ TFLOPS}$ sustained @ 25% MFU):
     $$\text{Time}_{\text{train\_T4}} = \frac{1.16 \times 10^{16}}{3.250 \times 10^{13}} \approx \mathbf{357 \text{ seconds (≈6 min)}}$$
   - The TPU v5e-8 is reserved for Stage 0 and occasional large retrains; mixing T4 and TPU sessions inside one generation is impractical (separate sessions, XLA compile time).
4. **Total Cycle & Weekly Cadence**:
   - Total generation turnaround is **≈1.1–1.3 hours** on 2× T4.
   - About 24 T4 session-hours per week for generations plus about 6 for evals gives **10 to 12 generations per week [추정]**, within the ≈30 h/week GPU quota; the TPU quota stays free for Stage 0 and retrains.

---

# 3. Curve-Killer Defenses

Self-improvement engines that lack formal circuit breakers inevitably succumb to systematic degeneration. Project Hephaestus integrates concrete defensive mechanisms, empirical detectors, and automated rollback rules for each known curve-killer:

### Table 3.1: Master Curve-Killer Defensive Matrix

| Curve-Killer Threat | Concrete Architectural Mechanism | Empirical Real-Time Detector | Automated Circuit Breaker & Rollback Rule |
|---|---|---|---|
| **1. Plateau (Exploration Stagnation)** | MCTS search explores outside policy support; Depth Regret ($R_{\text{depth}}$) continuously targets learning frontier; Stitch DSL compression shrinks effective tree depth. | Fresh commit-revealed benchmark delta $\Delta_g \le 0$ across 2 consecutive generations; pass@1024 slope $\le 0$ [검증됨: https://arxiv.org/abs/2504.13837v4]. | Inject $+50\%$ Dirichlet exploration noise at MCTS root ($\alpha=0.45$); if stagnant for 3 generations, rollback weights to $g-1$ and trigger autonomous domain ascension. |
| **2. Own-Level Prison (Proposer Trivialization)** | Replace solver regret with parameter-free **[G-02] Depth Regret**: $R = \max(0, S_{\text{deep}} - S_{\text{shallow}})$; inject $50\%$ stepped procedural tasks from frozen generators. | Proposer task solve-rate drifts outside $[35\%, 65\%]$ band; proposer formal vocabulary entropy drops $>25\%$. | Hard-clamp curriculum: force $70\%$ external procedural task injection; completely re-initialize proposer head weights with fresh random initialization. |
| **3. Reward Hacking (The Fake Exponential)** | **[G-13] Four-Layer Reading Firewall**: Reverse procedural generation, 3-way paraphrase gating, and counterfactual perturbation. Verifiers execute inside read-only namespaces. | **In-Loop vs. Held-Out Gap ($H_g$)**: $H_g = S_{\text{in-loop}} - S_{\text{held-out}} > 5.0$ points while held-out remains flat; random-reward control delta $\Delta_{\text{rand}} < 0$. | **EMERGENCY CIRCUIT BREAKER**: Immediately terminate pipeline; quarantine generation $g$; purge corrupted buffer shard; assert verifier SHA-256 hashes. |
| **4. Model Collapse (Entropy Death)** | Immutable historical replay archive ($\ge 25\%$ buffer reserved for seed data); policy logit entropy regularization penalty ($\lambda_{\text{ent}} = 0.01$). | Emitted formal token entropy drops $>20\%$; semantic AST diversity collapses; duplicate program ratio exceeds $15\%$ across batch. | Restore replay buffer historical mix to $50\%$; rollback policy head to generation $g-1$; increase MCTS sampling temperature to $T=1.2$. |
| **5. Compute Wall (Latency Explosion)** | **Snell Compute-Optimal Search** ($H(V)$ adaptive allocation); **[G-10] Constant-Size Recurrent State** ($S_t \in \mathbb{R}^{d \times d}$) eliminates KV-cache memory bloat. | Mean inference latency exceeds 2.5 seconds per task; MCTS tree node expansion timeout rate exceeds $5\%$. | Clamp maximum MCTS simulations to 32; prune actions with prior $P(a|s) < 0.03$; restrict tree search depth to $D_{\max} = 16$. |
| **6. Real-Experiment / Simulation Latency** | **[G-11] Value-Head Latent Pruning**: Discards candidate hypotheses with $\hat{V} < 0.2$ in latent space; active learning selects hypotheses via Expected Information Gain (EIG). | Simulator job queue latency exceeds 24 hours; simulation compute quota exceeds allocated budget by $>10\%$. | Throttle physical simulation requests; activate coarse-grained proxy filters (e.g., molecular mechanics force-fields prior to ab initio DFT). |

---

# 4. RSI Metrics Dashboard

All self-improvement metrics are calculated in an isolated, read-only audit container evaluated against frozen, commit-revealed evaluation splits:

```
====================================================================================================
                        HEPHAESTUS TELEMETRY & RSI METRICS DASHBOARD
====================================================================================================

  [CORD-CUT METRICS]
  - Teacher-Exceedance (TE_g):        TE_g = |{x in D_fail : Solve(x)=1}| / |D_fail|  [Target: >= 25%]
  - Random-Reward Control Delta:      Delta_rand = Score_true(g) - Score_rand(g)      [Target: >= +5.0%]
  - Autonomous Generations (G_0):     Monotonic counter of generations since Stage 1  [Target: Monotonic]

  [COMPOUNDING & ACCELERATION METRICS]
  - Held-Out Improvement Rate:        Delta_g = S_g(D_heldout) - S_{g-1}(D_heldout)   [Target: Delta_g > 0]
  - Second Derivative (Acceleration): a_g = Delta_g - Delta_{g-1}                     [Target: a_g >= 0]
  - Search Internalization (E_g):     E_g = Rollouts_{g-1}(80%) / Rollouts_g(80%)     [Target: E_g > 1.0]
  - Description Length Compression:   CR = DL_raw / DL_compressed                     [Target: CR >= 1.30]

  [INTEGRITY & EFFICIENCY MONITORS]
  - In-Loop vs Held-Out Gap (H_g):    H_g = S_g(D_inloop) - S_g(D_heldout)            [Tripwire: H_g > 5.0]
  - Empirical Learnability (EDLI):    EDLI = Delta_Accuracy / GPU-Hour                [Resource Governor]
  - Domain Saturation Coeff (Psi_D):  Psi_D = (pass@1 / pass@64) * (1 - DeltaDL/DL)   [Ascension: >= 0.85]

====================================================================================================
```

### 4.1 Detailed Metric Formulations

1. **Teacher-Exceedance ($\text{TE}_g$)**: The primary metric measuring escape from the teacher ceiling. Evaluated against the frozen manifest of 2,000 problems failed by the 7B teacher ($\mathcal{D}_{\text{held-out}}^{\text{teacher\_fail}}$):
   $$\text{TE}_g = \frac{|\{x \in \mathcal{D}_{\text{held-out}}^{\text{teacher\_fail}} : \text{Solve}(x) = 1\}|}{|\mathcal{D}_{\text{held-out}}^{\text{teacher\_fail}}|}$$
   Requirement: $\text{TE} \ge 25\%$ at Stage 1 cord-cut; $\ge 60\%$ at the GPT-4 parity rung [가설].
2. **Search Internalization Metric ($E_g$) [G-08]**: Measures the degree to which deep lookahead search is crystallized into the core policy's shallow intuition:
   $$E_g = \frac{\text{Rollouts}_{g-1}(80\%)}{\text{Rollouts}_g(80\%)} > 1.0$$
   An $E_g > 1.0$ proves that the network requires fewer MCTS rollouts in generation $g$ to achieve an 80% solve rate than it did in generation $g-1$.
3. **Random-Reward Control Arm Delta ($\Delta_{\text{rand}}$) [G-01]**:
   $$\Delta_{\text{rand}} = \text{Score}_{\text{true\_reward}}(g) - \text{Score}_{\text{random\_reward}}(g) \ge +5.0 \text{ points}$$
   Guarantees that measured self-improvement is causally driven by exact verifiers rather than spurious dataset memorization or policy drift.
4. **In-Loop vs. Held-Out Gap ($H_g$)**:
   $$H_g = S_g(\mathcal{D}_{\text{in-loop}}) - S_g(\mathcal{D}_{\text{held-out}})$$
   Acts as the primary tripwire against reward hacking. If $H_g > 5.0$ percentage points while held-out performance is stagnant, training is halted immediately.
5. **Description Length Compression Ratio ($CR$) [G-09]**:
   $$CR = \frac{\sum_{P \in \mathcal{D}^*} \text{AST\_Node\_Count}_{\text{raw}}(P)}{\sum_{P \in \mathcal{D}^*} \text{AST\_Node\_Count}_{\text{compressed}}(P)} \ge 1.30$$
   Quantifies the macro-induction efficiency of the external Stitch skill library.

---

# 5. The Core Learner Architecture

### 5.1 Architecture Overview & Dataflow

The Hephaestus reasoning core is an attention-free **Gated DeltaNet** [검증됨: https://arxiv.org/abs/2412.06464v3], explicitly engineered to eliminate sequence-length softmax and provide constant-size associative memory recurrence for high-throughput MCTS node expansion.

```
                                  [Problem Prompt / Specification]
                                                 |
                                                 v
                              +-------------------------------------+
                              |   16K BPE Input Embedding Table     | (10.49M params)
                              +-------------------------------------+
                                                 |
                                                 v
                              +-------------------------------------+
                              |     Bidirectional Input Reader      | (22.16M params)
                              |  (2 Bi-Layers = 4 Blocks, d=640)    |
                              +-------------------------------------+
                                                 |
                                         [Latent Context H_ctx]
                                                 v
             +-----------------------------------------------------------------------+
             |                 Recurrent Reasoning Core Trunk (Weight-Tied)          | (44.31M params)
             |           (8 Layers Gated DeltaNet, d=640, Recurrent Depth r=1..8)    |
             |           Conditioned on Learned Depth Embeddings e_r [G-03]          |
             +-----------------------------------------------------------------------+
                   |                     |                    |                  |
                   v                     v                    v                  v
     +------------------------+ +-----------------+ +------------------+ +---------------+
     |   Policy Head (pi_theta)| | Value Head (V)  | |  Proposer Head   | | DROP Pointer  |
     | (d=640 -> 2,048 Tokens)| | (Critic, MLP)   | | (d=640 -> 2,048) | | (Start / End) |
     |     (1.31M params)     | | (0.17M params)  | |  (1.31M params)  | | (0.41M params)|
     +------------------------+ +-----------------+ +------------------+ +---------------+
                   |                     |
                   +----------+----------+
                              |
                              v
              +-------------------------------+
              |   Snell MCTS Expansion Loop   | <---> [0-Param Mechanical Sandboxes]
              | [G-10] State S_t in R^{640x640}|       (Python, SymPy, Z3, Lean 4)
              | [G-11] Latent Value Pruning   |
              +-------------------------------+
                              |
                              v
                   [Structured JSON Output] ===> [Renderer Firewall (Qwen3-1.7B)]
```

### 5.2 Mathematical Formulation of Non-Transformer Recurrence [G-10]

Standard Transformer architectures compute sequence attention via $A = \text{softmax}(QK^T / \sqrt{d})V$, requiring $\mathcal{O}(N^2)$ sequence compute and an ever-expanding KV-cache. In MCTS, where a search tree branches across hundreds of partial sequences, maintaining and cloning KV-caches causes memory fragmentation and severe latency bottlenecks.

Gated DeltaNet replaces sequence softmax with an **associative delta-rule matrix recurrence** [검증됨: https://arxiv.org/abs/2412.06464v3]:
$$S_t = \alpha_t S_{t-1} \left(I - \beta_t k_t k_t^T\right) + \beta_t v_t k_t^T$$
$$y_t = S_t q_t$$
where:
- $S_t \in \mathbb{R}^{d \times d}$ is the constant-size recurrent memory state matrix (implemented as 5 heads of $128 \times 128$, i.e. $81,920$ floats $\approx 160\text{ KB}$ per layer in fp16).
- $\alpha_t = \sigma(W_\alpha x_t) \in (0, 1)$ is a data-dependent continuous decay gate.
- $\beta_t = \sigma(W_\beta x_t) \in (0, 1)$ is an adaptive learning-rate gate governing memory write intensity.
- $k_t, v_t, q_t \in \mathbb{R}^d$ are projected key, value, and query vectors normalized such that $\|k_t\|_2 = 1$.

**The MCTS Non-Transformer Moat**: Because $S_t$ has fixed dimension $\mathbb{R}^{640 \times 640}$ regardless of token sequence length $t$, caching node states in the MCTS tree requires strictly $\mathcal{O}(1)$ memory. A search tree of 256 nodes stores $256 \times 8 \times 160\text{ KB} \approx 320\text{ MB}$ of states, independent of prefix length, whereas a Transformer's per-node KV-cache grows with every token. The resulting MCTS speedup is **unmeasured [가설]** and is one of the measurements of Experiment 1.

### 5.3 Detailed Parameter Breakdown Table (< 100M Cap)

Every single trainable parameter executed at inference is explicitly enumerated below:

| Component | Layer Specifications & Dimensions | Arithmetic Derivation | Parameter Count |
|---|---|---|---|
| **Input Token Embeddings** | $V_{\text{in}} = 16,384, d = 640$ | $16,384 \times 640$ | **10,485,760** |
| **Formal IR Embeddings** | $V_{\text{formal}} = 2,048, d = 640$ | $2,048 \times 640$ | **1,310,720** |
| **2D CA Spatial Adapter** | $3 \times 3$ Conv (8ch) + Coord Embeddings | $3 \times 3 \times 8 \times 8 + 8 \times 640 + 2 \times 640$ | **6,976** |
| **Bidirectional Reader** | 4 Directional Blocks ($d=640, d_{\text{ff}}=1600, k=4$) | $4 \times [6 \times 640^2 + 3 \times 640 \times 1600 + 12 \times 640 + 3 \times 640]$ | **22,156,800** |
| **Recurrent Reasoning Core** | 8 Weight-Tied Blocks ($d=640, d_{\text{ff}}=1600, k=4$) | $8 \times [6 \times 640^2 + 3 \times 640 \times 1600 + 12 \times 640 + 3 \times 640]$ | **44,313,600** |
| **Depth Embeddings ($e_r$)** | 8 Learned Depth Vectors ($r \in [1, 8], d=640$) | $8 \times 640$ | **5,120** |
| **Policy Head ($\pi_\theta$)** | LayerNorm + Linear ($640 \to 2,048$) | $640 \times 2,048 + 2,048 + 2 \times 640$ | **1,314,048** |
| **Value Head ($V_\phi$)** | LayerNorm + MLP ($640 \to 256 \to 1$) | $640 \times 256 + 256 + 256 \times 1 + 1 + 2 \times 640$ | **165,633** |
| **Proposer Head ($\mathcal{P}$)** | LayerNorm + Linear ($640 \to 2,048$) | $640 \times 2,048 + 2,048 + 2 \times 640$ | **1,314,048** |
| **DROP Pointer Head** | Start / End Affine Projections | $640 \times 640 + 2 \times 640$ | **410,880** |
| **Retriever Key Projection** | Dense Datastore Alignment ($640 \to 256$) | $640 \times 256 + 256 + 2 \times 640$ | **165,376** |
| **TOTAL PARAMETERS** | **Sum of all inference-executed trainable weights** | **Sum of rows above** | **81,648,961** |

*Verification*: $81,648,961 < 100,000,000$. The architecture provides **18,351,039 parameters (18.35M) of headroom**, fully compliant with `BRIEF.md` §2.1.

---

# 6. Renderer Firewall & I/O Protocol

### 6.1 Structured JSON Output Schema

The core learner is strictly prohibited from emitting conversational natural language. It outputs an immutable, strictly validated JSON object conforming to the Draft 2020-12 specification:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AphasicCoreStructuredOutput",
  "type": "object",
  "required": [
    "problem_id",
    "formal_ir",
    "execution_trace",
    "answer",
    "verifier_status",
    "confidence"
  ],
  "properties": {
    "problem_id": { "type": "string" },
    "formal_ir": {
      "type": "object",
      "required": ["language", "code_ast"],
      "properties": {
        "language": {
          "type": "string",
          "enum": ["python_ast", "sympy_expr", "lean4_tactic", "z3_smt", "pddl_plan"]
        },
        "code_ast": { "type": "string" }
      }
    },
    "execution_trace": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["step_idx", "opcode", "stdout"],
        "properties": {
          "step_idx": { "type": "integer" },
          "opcode": { "type": "string" },
          "stdout": { "type": "string" }
        }
      }
    },
    "answer": {
      "type": "object",
      "required": ["type", "value"],
      "properties": {
        "type": {
          "type": "string",
          "enum": ["integer", "float", "latex", "choice_letter", "pointer_span", "grid_array"]
        },
        "value": {
          "type": ["string", "number", "array"]
        }
      }
    },
    "verifier_status": {
      "type": "string",
      "enum": ["verified_success", "execution_error", "timeout", "unverified"]
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    }
  },
  "additionalProperties": false
}
```

### 6.2 The Blindfold Protocol & Firewall Rules

To prevent any autoformalization or conversational leakage from smuggling reasoning into the evaluation:
1. **The Blindfold Constraint**: The renderer receives **only** the `AphasicCoreStructuredOutput` JSON payload. It is **never provided with the original problem text**. A static 8-gram intersection filter asserts zero token overlap between the renderer's input context and the raw benchmark question prompt.
2. **Core-Only Scoring**: All official benchmark scores are parsed directly from the JSON payload (`answer.value`, `formal_ir.code_ast`). The renderer is completely absent from the evaluation loop.
3. **Renderer-Alone Controls**: For every evaluated benchmark row, we publish renderer performance under three control regimes: (a) raw problem prompt directly, (b) empty or randomly shuffled core trace, (c) core trace with answer masked. Pass condition: Scores under Controls (b) and (c) must be $\le \text{random chance} + 2\sigma$.
4. **Counterfactual Swap Test [G-05]**: In 500 test trials, core answers are deliberately corrupted. If renderer output matches the corrupted answer in $<99\%$ of trials, deployment is immediately halted and output falls back to Tier 0 verbatim templating [가설, AUDIT §3.5].
5. **Direct Core Input Reading**: The core's 2-layer Bidirectional Input Reader ingests raw natural language text directly. No intermediary LLM pre-parser, auto-formalizer, or prompt rewriter is permitted.

### 6.3 Complete End-to-End Round-Trip Execution Trace

To illustrate the operational firewall, consider a sample problem from the GSM-Symbolic benchmark:

#### Input Text (Ingested Directly by Core Reader)
*"A farmer has 14 apple trees. Each tree yields 45 apples. If he sells 320 apples at the market and packages the remainder into bags of 5 apples each, how many bags can he fill?"*

#### Formal JSON Emitted by Aphasic Core (Scored Directly)
```json
{
  "problem_id": "gsm_sym_0842",
  "formal_ir": {
    "language": "python_ast",
    "code_ast": "total_apples = 14 * 45\nremaining_apples = total_apples - 320\nbags = remaining_apples // 5\nans = bags"
  },
  "execution_trace": [
    { "step_idx": 1, "opcode": "MULT", "stdout": "total_apples = 630" },
    { "step_idx": 2, "opcode": "SUB",  "stdout": "remaining_apples = 310" },
    { "step_idx": 3, "opcode": "DIV",  "stdout": "bags = 62" }
  ],
  "answer": {
    "type": "integer",
    "value": 62
  },
  "verifier_status": "verified_success",
  "confidence": 1.00
}
```

#### Blindfolded Interpreter Output (Generated by Qwen3-1.7B)
*"Based on the verified calculations, the farmer produces an initial harvest of 630 apples across the 14 trees. After selling 320 apples at the market, 310 apples remain. Dividing these into 5-apple containers yields exactly 62 completed bags."*

---

# 7. External Tools & Skill Library

### 7.1 Zero-Parameter Mechanical Tool Bridge

In accordance with BRIEF.md §2.3, world knowledge and procedural computation are externalized to zero-parameter mechanical tools, avoiding parametric bloating:
1. **Python 3.12 Sandbox**: Executes arbitrary deterministic algorithmic code, evaluating test assertions with strict timeouts ($2.0\text{ s}$) and memory ceilings ($256\text{ MB}$).
2. **SymPy 1.13 Computer Algebra System**: Certifies algebraic identities, calculates limits, solves linear systems, and verifies symbolic equivalence via zero-parameter reduction ($\text{simplify}(A - B) == 0$).
3. **Z3 SMT Solver**: Translates propositional and first-order predicates into bit-vector and integer arithmetic constraints, solving for satisfiability or emitting formal refutation proofs.
4. **Lean 4 Theorem Prover**: Mechanically checks interactive proofs, ensuring that mathematical steps adhere strictly to axiomatic foundations.
5. **VAL PDDL Validator**: Verifies classical domain planning states against STRIPS/PDDL domain definitions.

### 7.2 Stitch-Style Macro Extraction & Library Growth [G-04, G-09]

As verified solutions accumulate in the Expert Replay Buffer $\mathcal{D}_g^*$, Project Hephaestus applies the **Stitch** library-learning compressor, as used in LILO [검증됨: https://arxiv.org/html/2310.19791v2] to compress repeated reasoning subroutines:
1. **E-Graph Subgraph Induction**: ASTs of verified programs are converted into S-expressions. An e-graph matching algorithm identifies recurring functional motifs across distinct problems.
2. **Macro Synthesis**: Recurring subgraphs are abstracted into parameterized DSL combinators (e.g., factoring repeated prime-factorization or graph-traversal logic into a single primitive `fn_macro_14`).
3. **Grammar Expansion**: Discovered macros are added to the core's formal token vocabulary ($V_{\text{formal}}$). Calling `fn_macro_14` allows MCTS to execute a 20-step reasoning subtree in a single search transition.
4. **Description Length Compression [G-09]**: The library accepts a new macro only if it improves global compression ratio:
   $$CR = \frac{\text{DL}_{\text{raw}}}{\text{DL}_{\text{compressed}}} \ge 1.30$$
   This permanently contracts effective search depth without adding a single neural parameter.

---

# 8. Bootloader Training & 0-KRW Compute Plan

### 8.1 Stage 0 Bootloader Recipe & Dataset Composition

Stage 0 pretraining is strictly restricted to open-license, decontaminated datasets. In compliance with BRIEF.md §2.6, **zero tokens derived from GPT-3.5 or GPT-4 are included** (TinyGSM and MetaMathQA are completely banned).

| Dataset Identifier | Token Volume | License / Provenance | Educational & Domain Focus |
|---|---|---|---|
| **OpenMathInstruct-2** | 1.50B tokens | CC-BY-4.0 (Llama-3.1-405B generated, SymPy verified) [검증됨: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2] | High-school & Olympiad algebra, geometry, calculus |
| **Python-Edu & The Stack v2** | 1.50B tokens | Permissive Open Source (SmolLM-Corpus cleaned subset) [검증됨: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus] | Cleaned algorithmic Python scripts and unit tests |
| **Cosmopedia v2** | 1.00B tokens | ODC-BY (Mixtral-8x7B synthetic academic textbooks) [검증됨: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus] | Synthetic textbooks covering formal STEM curricula |
| **Reasoning Gym & CLRS Traces**| 0.60B tokens | MIT / Apache-2.0 (Procedural algorithmic traces) [검증됨: https://arxiv.org/pdf/2505.24760, https://arxiv.org/html/2209.11142] | Sorting, dynamic programming, logic, graph algorithms |
| **FineWeb-2 Korean split (ODC-By), filtered by our own STEM/math classifier** | 0.40B tokens | Open Web, ODC-By [검증됨: https://arxiv.org/abs/2506.20920v1] | Korean language math problems for reader tuning |
| **TOTAL BOOTLOADER DATA** | **5.00B tokens** | **100% Permissive Open Licenses (0% OpenAI Data)** | **Arithmetic Check: $1.5+1.5+1.0+0.6+0.4 = 5.0\text{B}$** |

#### Mandatory No-Teacher Control Arm (Track 0-NT)
To rigorously demonstrate that self-improvement compounds independently of teacher distillation (BRIEF.md §2.6), a parallel control arm is pretrained:
- **Track 0-NT**: Trained on 5.0B tokens composed exclusively of procedural algorithmic traces (CLRS, Reasoning Gym, SymPy derivations, re-ARC; zero tokens from open LLM teachers).
- Track 0 and Track 0-NT are advanced through 5 generations of MCTS Expert Iteration. We report the empirical boost $\Delta_{\text{teacher\_boost}}(g) = \text{Score}_{\text{boot}}(g) - \text{Score}_{\text{no\_teacher}}(g)$ to quantify the exact generation where Track 0-NT surpasses the static teacher.

### 8.2 Explicit FLOP Accounting Against Free Quotas

- **Model Parameters**: $N_{\text{total}} = 81.65\text{M}$ ($N_{\text{core}} = 44.31\text{M}, N_{\text{non\_core}} = 37.34\text{M}$).
- **Effective Training Parameters ($r=4$)**: $N_{\text{eff}} = 37.34\text{M} + 4 \times 44.31\text{M} = 214.59\text{M}$.
- **Training FLOPs per Token**: $6 \times N_{\text{eff}} = 6 \times 2.1462 \times 10^8 = 1.288 \times 10^9 \text{ FLOPs/token}$.
- **Stage 0 Pretraining Compute (5.0B Tokens)**:
  $$\text{Total FLOPs} = 5.0 \times 10^9 \text{ tokens} \times 1.288 \times 10^9 \text{ FLOPs/tok} = \mathbf{6.44 \times 10^{18} \text{ FLOPs}}$$
- **Execution on Kaggle TPU v5e-8**:
  $$\text{Sustained Throughput (@ 30\% MFU)} = 1,576 \text{ TFLOPS} \times 0.30 = 4.728 \times 10^{14} \text{ FLOP/s}$$
  $$\text{Pretraining Time} = \frac{6.44 \times 10^{18}}{4.728 \times 10^{14}} = 13,621 \text{ seconds} \approx \mathbf{3.78 \text{ hours}}$$
  At 20% MFU (small JAX models measured 16–27% in the reference tutorial cited by compute.md) this is ≈5.7 hours. Either way it fits one ≤9-hour TPU session.

### 8.3 Hardware Kernel & Device Engineering Notes

1. **NVIDIA T4 (Turing sm_75, 16 GB GDDR6)**:
   - T4 does not support bfloat16; fp16 tensor cores only [검증됨: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf].
   - **Triton Pinning**: Pin environment strictly to **Triton 3.2.0** [검증됨: https://raw.githubusercontent.com/triton-lang/triton/v3.2.0/README.md] (Triton $\ge 3.3$ dropped sm_75 support).
   - Maintain pure-PyTorch parallel chunked scan CUDA kernels with fp32 accumulators to prevent numerical gate underflow during recurrent state updates.
2. **Kaggle TPU v5e-8 (JAX/XLA)**:
   - XLA compilation mandates strictly static tensor shapes. Sequences are padded to static powers of two ($L \in \{256, 512, 1024\}$).
   - Recurrent scan implemented via native `jax.lax.associative_scan`.
3. **Local Laptop RTX 3060 (6 GB GDDR6)**:
   - Ampere architecture natively supports bfloat16 and modern Triton kernels. Utilized for local kernel prototyping, unit test debugging, and offline evaluation before pushing to Kaggle runners.

---

# 9. The Boss Ladder & Scoreboard

### 9.1 Scoreboard vs. GPT-3.5 and the "아득히" Margin Rule

To beat GPT-3.5 "아득히" (by a decisive margin), the engine must hit or exceed $G = T + \max(10, \frac{100-T}{3})$ on all gated rows, where $T$ is the **maximum** reported GPT-3.5 score in `reason_baseline.md`. **Honesty note (audit):** the "Target Score" and "95% CI lower bound" columns are targets derived from G, not forecasts or measurements; they will be replaced by measured numbers. Confidence by row family [추정]: high for search-friendly rows (Game of 24, Blocksworld, FOLIO, ProofWriter, PrOntoQA, SVAMP); medium for math and algorithmic rows (GSM8K, GSM-Plus, MATH, BBH, CRUXEval, DROP, ARC-AGI-1); low for reading-heavy rows (LogiQA, ReClor, MuSR), code generation (HumanEval+, MBPP+, LiveCodeBench) and knowledge-heavy Tier B rows (GPQA, MedQA, StrategyQA).

#### Tier A — Gated, Closed-World Reasoning (No Retrieval Permitted)

| # | Benchmark Identifier | Max GPT-3.5 ($T$) | Target Margin ($G$) | Target Score | Projected Margin ($\Delta$) | 95% CI Lower Bound | Expected Gen ($g$) | Winning Mechanism & Architecture Moat |
|---|---|---|---|---|---|---|:---:|---|
| **A1** | **GSM8K** (0-shot) | 78.1 | **88.1** | **89.2 [추정]** | +11.1 [추정] | 87.5 ($N=1,319$) | Gen 4 | Program-of-Thought Python execution + reverse generation reading firewall. |
| **A2** | **GSM-Plus** (Robust Twin) | 61.2 | **74.1** | **76.5 [추정]** | +15.3 [추정] | 75.7 ($N=10,552$) | Gen 4 | Counterfactual perturbation invariance; immune to the 65% frontier drop. |
| **A3** | **MATH** (Full 5,000) | 43.1 (verified 34.1) | **62.1** | **64.8 [추정]** | +30.7 [추정] | 63.5 ($N=5,000$) | Gen 7 | SymPy symbolic execution + MCTS proof tree expansion over Olympiad lemmas. |
| **A4** | **SVAMP** | 83.0 | **93.0** | **94.2 [추정]** | +11.2 [추정] | 92.6 ($N=1,000$) | Gen 3 | Structural semantic parsing into executable algebraic AST. |
| **A5** | **BBH** (27 Tasks) | 70.7 | **80.7** | **82.3 [추정]** | +11.6 [추정] | 81.3 ($N=6,511$) | Gen 6 | Algorithmic trace induction over Boolean, Dyck, and tracking tasks. |
| **A6** | **ARC-AGI-1** (Public Eval, pass@2)| 10.6 | **40.4** | **40.5 [가설]** | +29.9 [가설] | 35.7 ($N=400$) | Gen 6 | 2D CA spatial conv adapter + test-time training (TTT) over re-ARC grids. |
| **A7** | **HumanEval+** (EvalPlus) | 70.7 | **80.7** | **82.5 [추정]** | +11.8 [추정] | 76.5 ($N=164$) | Gen 5 | Unit-tested self-repair; MCTS tree search over AST program mutations. |
| **A8** | **MBPP+** (EvalPlus) | 69.7 | **79.8** | **81.2 [추정]** | +11.5 [추정] | 77.2 ($N=378$) | Gen 5 | Stitch macro combinator injection shortening standard algorithm search. |
| **A9** | **LiveCodeBench** (Post-Freeze) | 22.6 | **48.4** | **51.2 [추정]** | +28.6 [추정] | 45.5 ($N=300$) | Gen 8 | Contamination-free fresh contest code execution and test-case repair. |
| **A10**| **CRUXEval-O** | 63.3 | **75.5** | **77.4 [추정]** | +14.1 [추정] | 74.4 ($N=800$) | Gen 6 | Recurrent mental execution of Python state transitions. |
| **A11**| **DROP** (F1 Score) | 70.2 (verified 64.1) | **80.2** | **81.8 [추정]** | +17.7 [추정] | 81.0 ($N=9,536$) | Gen 6 | DROP Pointer Head extracting numeric spans without conversational text. |
| **A12a**| **FOLIO** | 62.7 | **75.2** | **78.0 [추정]** | +15.3 [추정] | 72.1 ($N=204$) | Gen 3 | First-Order Logic autoformalization into Z3 SMT solver assertions. |
| **A12b**| **ProofWriter** (Depth-5) | 71.5 | **81.5** | **84.5 [추정]** | +13.0 [추정] | 81.6 ($N=600$) | Gen 3 | Recursive backward chaining through factual rule ontologies. |
| **A12c**| **PrOntoQA** (Hardest) | 85.0 | **95.0** | **96.5 [추정]** | +11.5 [추정] | 94.8 ($N=500$) | Gen 4 | Strict deductive resolution over fictional entity taxonomies. |
| **A13a**| **LogiQA 2.0** | 52.4 | **68.2** | **70.5 [추정]** | +18.1 [추정] | 66.8 ($N=1,000$) | Gen 7 | Bidirectional reader parsing dense formal arguments into truth tables. |
| **A13b**| **ReClor** (dev) | 57.4 | **71.6** | **73.8 [추정]** | +16.4 [추정] | 69.9 ($N=500$) | Gen 7 | Logical fallacy and counterexample detection via constraint satisfaction. |
| **A14a**| **MuSR Murder Mysteries** | 70.0 | **80.0** | **81.5 [추정]** | +11.5 [추정] | 76.6 ($N=250$) | Gen 7 | Long-context clue tracking; reader extracts timelines into relational tuples. |
| **A14b**| **MuSR Object Placement** | 58.2 | **72.1** | **74.0 [추정]** | +15.8 [추정] | 68.6 ($N=256$) | Gen 7 | Recurrent state tracking spatial coordinates across multi-hop narrative shifts. |
| **A14c**| **MuSR Team Allocation** | 78.8 | **88.8** | **90.2 [추정]** | +11.4 [추정] | 86.3 ($N=250$) | Gen 7 | Exact bipartite graph matching compiled into Z3 linear constraints. |
| **A15**| **Game of 24** (Hard Games) | 19.0 | **46.0** | **98.0 [추정]** | +79.0 [추정] | 95.2 ($N=100$) | Gen 2 | Exhaustive Snell MCTS search with exact arithmetic verification. |
| **A16**| **Blocksworld** (PlanBench) | 6.1 | **37.4** | **68.5 [추정]** | +62.4 [추정] | 64.6 ($N=600$) | Gen 3 | PDDL action planning validated by VAL validator. |
| **A17**| **KMMLU-Math** | 30.3 | **53.5** | **56.0 [추정]** | +25.7 [추정] | 51.5 ($N=700$) | Gen 8 | Korean BPE reader tokenization + SymPy symbolic computation. |

*\*Note on ARC-AGI-1 Realism*: In accordance with `AUDIT_FROM_CLAUDE.md` §1.8, attention-free recurrent cores reach 29.6% on ARC-AGI-1 without inductive priors. Hephaestus reaches 40.5% exclusively by deploying the 2D Cellular-Automaton Spatial Convolutional Adapter ($3 \times 3$, 8 channels, 0.05M params) paired with Test-Time Training (TTT) over re-ARC augmented grids per test puzzle. Without TTT, the baseline expectation is **32.0% [추정]**.

#### Tier B — Gated, Retrieval Allowed (Frozen Offline Datastore)

| # | Benchmark Identifier | Max GPT-3.5 ($T$) | Target Margin ($G$) | Target Score | Projected Margin ($\Delta$) | 95% CI Lower Bound | Expected Gen ($g$) |
|---|---|---|---|---|---|---|:---:|
| **B1** | **GPQA-Diamond** (0-shot) | 30.6 | **53.7** | **55.6 [추정]** | +25.0 [추정] | 48.6 ($N=198$) | Gen 10 [추정] |
| **B2** | **MedQA** (USMLE 4-Option) | 63.4 | **75.6** | **77.2 [추정]** | +13.8 [추정] | 74.9 ($N=1,273$) | Gen 9 [추정] |
| **B3** | **StrategyQA** | 70.3 (verified 51.5) | **80.3** | **82.0 [추정]** | +30.5 [추정] | 79.8 ($N=2,290$) | Gen 7 [추정] |

---

### 9.2 The Subsequent Ladder Rungs

Surpassing GPT-3.5 is merely conquering the first rung. In compliance with AUDIT §1.6, subsequent rungs are evaluated through qualitative feasibility ratings and concrete operational prerequisites:

```
[Rung 1: GPT-3.5 Slime] ===> [Rung 2: GPT-4 Level] ===> [Rung 3: o1/R1 Parity] ===> [Rung 4: Frontier Tin Cans] ===> [Rung 5: ASI North Star]
  - Clear G-margin (Gen 8)     - MATH > 85%, HumanEval > 85%  - AIME 2024 > 75%            - Outperform Claude Opus 5.5,   - Open Math Conjectures
  - 100% Free Kaggle Tier      - Lean 4 tactic integration    - Codeforces >= 1800         - Gemini 4 Argon, GPT-6 Astra   - Oncology Lead Discovery
  - Feasibility: VERY HIGH     - Feasibility: MODERATE-HIGH   - Feasibility: MODERATE      - Feasibility: LOW-MODERATE     - Feasibility: THEORETICAL
```

1. **Rung 2: GPT-4 Level Reasoning (Expected Gen 15–20)**:
   - *Target Capability*: MATH $>85\%$, HumanEval $>85\%$, AIME $2024 >30\%$.
   - *Feasibility*: **Moderate-High [추정]**.
   - *Prerequisites*: Full operational integration of Domain 4 (Lean 4 tactic synthesis), automated execution repair loops, and Stage 2 domain expansion.
2. **Rung 3: o1 / DeepSeek-R1 Parity (Expected Gen 30–40)**:
   - *Target Capability*: AIME $2024 >75\%$, Codeforces Elo $\ge 1800$.
   - *Feasibility*: **Moderate [추정]**.
   - *Prerequisites*: Deep test-time compute scaling ($k \ge 512$ simulations), massive synthetic Lean 4 lemma extraction, and Stitch DSL compression exceeding $CR \ge 2.5$.
3. **Rung 4: Frontier "Tin Can" Parity (Expected Gen 50–60)**:
   - *Target Capability*: Decisively outperforming **Claude Opus 5.5, Gemini 4 Argon, and GPT-6 Astra** on formal mathematics, verified code generation, and complex planning.
   - *Feasibility*: **Low-Moderate [가설]**.
   - *Prerequisites*: Autonomous L4 meta-procedural optimization (evolutionary discovery of custom scan kernels and search heuristics); zero human intervention in task curation.
4. **Rung 5: North Star / Superhuman ASI (Scientific Discovery Milestone)**:
   - *Target Capability*: Proving open Erdős conjectures; synthesizing novel, non-toxic small-molecule oncology therapeutics with sub-nanomolar binding affinity.
   - *Feasibility*: **Long-term Theoretical Bet [가설]**.
   - *Prerequisites*: Seamless active-learning coupling with molecular dynamics simulators (OpenMM, AutoDock Vina, RDKit) operating as verified closed-loop oracles.

---

# 10. Scaling & North Star

### 10.1 Transfer Across Scales: 0.1B $\to$ 1B $\to$ 10B

A critical property of the Hephaestus architecture is that capability compounds in **non-parametric artifacts** that transfer losslessly across neural scales:
1. **Zero-Parameter Library Transfer**: The L3 program library (thousands of verified Stitch DSL functions, Lean 4 lemmas, and Python macros) transfers instantly to a 1B or 10B core on day 1 with zero parameter retraining.
2. **Curriculum Replay Distillation**: Millions of verified $(x, \pi_{\text{search}}, z=+1)$ trajectories synthesized by the 81.6M core form an ultra-dense, mathematically certified pretraining dataset for scaling larger models.
3. **Depth-to-Width Distillation**: MCTS search latency scales inversely with parameter scale (a 1B model would require 15–20s per MCTS step on T4). Scaling to 1B and 10B utilizes **Depth-to-Width Distillation**: using the 81.6M core + MCTS to generate dense data that trains the 1B/10B core to execute complex deductions in a single, shallow forward pass ($r=1$).

### 10.2 Compute Grant Acquisition Strategy (0-KRW Bootstrap)

The 0.1B core operates 100% within free Kaggle quotas. Once the 81.6M core officially conquers GPT-3.5 by the G-margin under commit-revealed fair-play audit, the project bootstraps scaling via verified institutional research grants:
1. **Google TPU Research Cloud (TRC)**: Free Cloud TPU quota for open research; VMs and storage are paid and no per-project size is guaranteed [검증됨: https://sites.research.google/trc/faq/].
2. **Modal Academic / Open-Source Compute Grants**: Up to $10,000+ in serverless GPU compute for high-velocity evaluation runs [검증됨: https://gpuperhour.com/blog/free-cloud-gpus-and-credits].
3. **Hugging Face Open Science Grants**: High-capacity dataset and checkpoint hosting for certified open weights [검증됨: https://huggingface.co/docs/hub/en/storage-limits].

### 10.3 The North Star: Automated Oncology Drug Discovery

The ultimate realization of Project Hephaestus is the autonomous discovery of novel therapeutic candidates:
1. **Molecular Physics Engines as Verifiers**: Open-source computational chemistry engines serve as zero-parameter ground-truth environments:
   - **RDKit**: Validates chemical valence, drug-likeness (Lipinski Rule of 5), and synthetic accessibility.
   - **AutoDock Vina**: Calculates binding free energy ($\Delta G_{\text{bind}}$) against oncogenic protein targets (e.g., KRAS G12D, EGFR).
   - **OpenMM**: Executes explicit-solvent molecular dynamics simulations to certify ligand-receptor conformational stability.
2. **MCTS in Chemical Combinatorial Space**: The core explores discrete molecular transformations (SMILES AST mutations, retrosynthetic disconnection cuts). The policy proposes chemical modifications; physics engines verify free energy.
3. **Bayesian Active Learning via Expected Information Gain (EIG)**: The engine prioritizes molecular hypotheses that maximize expected information gain, proposing wet-lab synthesis only when simulation epistemic uncertainty is maximal.

---

# 11. Stage-Gated Safety Mechanisms

Safety mechanisms scale strictly with system capability ("faster cars require better brakes"):

```
[Day 1: Sandbox & Kill Switches] ===> [L4+: Mandatory Human Approval] ===> [Bio/Chem: Dual-Use Review & Gate]
  - Immutable verifier hashes           - No autonomous code edits            - Zero pathogen/toxin data
  - Seccomp BPF syscall traps           - Human signs off on training diffs   - Wet-lab actions blocked
  - Automatic divergence cut-off        - Hyperparameter changes locked       - Mandatory expert review
```

### 11.1 Day 1 Mandatory Safeguards
- **Verifier Immutability**: All verifiers (Python sandbox, SymPy CAS, Z3 SMT, Lean 4) and benchmark evaluation splits reside on read-only filesystems. SHA-256 checksums are asserted at kernel boot.
- **Process & Network Isolation**: Training and rollout environments execute inside secure containers with network access permanently disabled (`--network none`).
- **Seccomp BPF Syscall Filtering**: Linux seccomp filters trap and immediately terminate any process attempting socket syscalls, unauthorized subprocess execution, or directory traversal outside `/tmp/eval_sandbox`.
- **Tamper-Proof Audit Logging**: Every generation writes an immutable cryptographic ledger containing: random seeds, code commit hashes, verifier hashes, training loss logs, and exact MCTS trace samples.
- **Automatic Circuit Breakers**: If In-Loop vs. Held-Out gap exceeds $5.0$ points ($H_g > 5.0$), or any unauthorized syscall occurs, execution terminates instantly.

### 11.2 Level 4 (L4) and Above Safeguards
- **Mandatory Human-in-the-Loop Sign-off**: Any proposed modification to the training pipeline, MCTS algorithm, loss functions, learning rate schedules, or data generators synthesized by the meta-optimizer must be formatted as a human-readable Git diff and explicitly approved by the human operator before being merged into generation $g+1$.

### 11.3 Biology & Chemistry Safeguards
- **Strict Data and Tool Quarantine**: All biological and chemical simulation modules remain disabled by default. No biological data shards or tools may be loaded without prior external dual-use review.
- **Non-Learned Toxin/Pathogen Blocklist**: A hardcoded string/substructure filter automatically rejects any chemical query or SMILES structure related to controlled chemical weapons, biological toxins, or pandemic-potential pathogens.
- **Wet-Lab Air-Gap**: The model is strictly forbidden from executing or triggering external robotic wet-lab actions. Any synthesized molecule or protocol requires independent human expert verification.
- **The Golden Rule**: *If at any point there is any plausible risk of harm to humans, the system must immediately and unconditionally halt.*

---

# 12. Initial Experiments & Risk Register

All initial experiments are specified strictly as **7-part plain-language scientific plans with zero executable code or pseudocode scripts**, in full compliance with `AUDIT_FROM_CLAUDE.md` §0.2 and §2.

### 12.1 Experiment 1 (Weeks 1–2): Demonstrating 1 Measurable RSI Generation in Game of 24

#### 1. Objective & Core Hypothesis
- **Objective**: Execute exactly one complete autonomous recursive self-improvement generation cycle ($g_0 \to g_1$) on a toy arithmetic verification domain with zero human demonstrations, verifying that amplified tree search distills into raw policy weights.
- **Core Hypothesis**: Distilling Snell MCTS search distributions $\pi_{\text{search}}$ and terminal verifier rewards $z \in \{-1, +1\}$ into the 81.6M Gated DeltaNet core produces a measurable greedy pass@1 improvement on held-out tasks ($\Delta_{g1} = S_{g1} - S_{g0} \ge +10.0$ percentage points), and beats a random-reward control arm by $\ge 8.0$ percentage points.

#### 2. Domain & Input Specification
- **Domain**: Game of 24 (given 4 integers in $[1, 9]$, synthesize a valid arithmetic expression using $+,-,\times,/$ that evaluates exactly to 24.0).
- **Data Partitions**:
  - Training Partition: 400 procedurally generated 4-tuples with known combinatorial solution depth.
  - Held-out Evaluation Partition: 100 fixed, unseen 4-tuples drawn from the standard hard benchmark set [검증됨: https://arxiv.org/pdf/2305.10601].
- **0-Parameter Mechanical Verifier**: Isolated Python 3.12 arithmetic evaluator checking digit conservation and exact numeric equality $| \text{eval}(P) - 24.0 | < 10^{-6}$.

#### 3. Step-by-Step Procedure
- **Step 1 (Base Policy Initialization & Baseline Evaluation)**: Initialize the 81.6M Gated DeltaNet policy $\pi_{\theta_0}$ and value critic $V_{\phi_0}$ from Stage 0 arithmetic pretraining. Evaluate greedy pass@1 accuracy on the 100 held-out evaluation tuples ($g_0$).
- **Step 2 (MCTS Search Rollout)**: For each of the 400 training tuples, execute MCTS search (64 simulations per tuple, PUCT exploration constant $c_{\text{puct}} = 1.25$, Dirichlet root noise $\alpha = 0.3, \epsilon = 0.25$). Terminate each simulation at formal leaf states and query the Python verifier. Record normalized visit count distributions $\pi_{\text{search}}(a|s)$ and binary terminal outcomes $z \in \{-1, +1\}$. Store successful trajectories in Expert Replay Buffer $\mathcal{D}_0^*$.
- **Step 3 (Expert Iteration Training)**: On Kaggle 2× T4 GPUs, train the core network on $\mathcal{D}_0^*$ for 3 epochs using the AdamW optimizer (learning rate $2 \times 10^{-4}$, weight decay $0.01$, batch size 32, cosine schedule). The policy head minimizes cross-entropy against $\pi_{\text{search}}$; the value head minimizes mean squared error against $z$. This produces generation checkpoint $g_1$ ($\theta_1, \phi_1$).
- **Step 4 (Distilled Policy Evaluation)**: Evaluate generation $g_1$ greedy pass@1 accuracy on the 100 held-out evaluation tuples without search ($r=1$). Compute performance delta $\Delta_{g1} = S_{g1} - S_{g0}$.
- **Step 5 (Internalization Measurement)**: Measure search internalization $E_1 = \text{Rollouts}_{g0}(80\%) / \text{Rollouts}_{g1}(80\%)$.

#### 4. Measurement Methodology & Metrics
- Primary Metric: Greedy pass@1 accuracy on the 100 held-out tuples evaluated under deterministic greedy decoding.
- Search Baseline: Pass@1 accuracy achieved by MCTS search on the held-out set at $g_0$.
- Internalization Ratio: $E_1$ measuring reduction in search simulations required to reach an 80% solve rate.
- Engineering measurements (audit): fraction of branches pruned by the value head before sandbox dispatch, MCTS nodes/second with cached recurrent states vs. recomputing prefixes, and sandbox checks/second on Kaggle vCPUs.

#### 5. Control Arm (Negative Control)
- **Random-Reward Arm ($g_1^{\text{ctrl}}$)**: An identical parallel training run where terminal verifier outcomes $z$ are replaced with random Bernoulli noise $z_{\text{rand}} \sim \text{Bernoulli}(0.5)$.
- The control arm isolates genuine verifier guidance from spurious optimization artifacts.

#### 6. Pass & Kill Criteria
- **Pass Criteria**:
  1. Distilled policy improvement: $\Delta_{g1} = S_{g1} - S_{g0} \ge +10.0$ percentage points.
  2. Control arm separation: $S_{g1} - S_{g1}^{\text{ctrl}} \ge +8.0$ percentage points.
  3. Search internalization: $E_1 \ge 1.5$.
- **Kill Criteria**:
  1. Distilled policy fails to improve: $\Delta_{g1} \le 0.0$ points.
  2. Random-reward control matches true reward within 3.0 points ($S_{g1} - S_{g1}^{\text{ctrl}} < 3.0$), indicating reward-independent memorization.
  3. MCTS search fails to exceed raw policy prior ($S_{\text{search}} \le S_{g0}$).

#### 7. Compute & Wall-Clock Budget
- Rollouts: 400 tasks $\times 64$ sims $\times 40$ tokens $\approx 1.02 \times 10^6$ tokens. On 2× T4 GPUs: $\approx 8\text{ minutes}$.
- Training: 3 epochs over buffer $\approx 2\text{ minutes}$.
- Held-out Evaluation: 100 tasks greedy pass $\approx 25\text{ seconds}$.
- **Total Compute & Wall-Clock**: $\approx \mathbf{15\text{ to }25\text{ minutes}}$ on 2× T4 GPUs ($\approx 0.4$ GPU-hours), well within a single Kaggle session.

---

### 12.2 Experiment 2 (Week 3): Procedural GSM-Symbolic Reading & Perturbation Invariance

#### 1. Objective & Core Hypothesis
- **Objective**: Validate that the 2-layer Bidirectional Input Reader correctly parses natural-language arithmetic into formal Python expressions without conversational leakage, evaluating invariance against number swapping and distractor clauses.
- **Core Hypothesis**: Reverse procedural generation training enables the 81.6M core to achieve $\ge 50.0\%$ execution accuracy on GSM-Symbolic with a performance drop of $\le 5.0\%$ under distractor clause insertion (GSM-Symbolic reports drops of up to 65% for the models it tested; it has no GPT-3.5 row).

#### 2. Domain & Input Specification
- **Domain**: Procedural GSM-Symbolic problems generated via attribute grammars with randomized numerical values and distractor clauses [검증됨: https://arxiv.org/abs/2410.05229].
- **Partitions**: 1,000 clean problems, 1,000 perturbed problems with numerical mutation, and 1,000 problems with injected NoOp distractor clauses.

#### 3. Step-by-Step Procedure
- **Step 1**: Ingest clean and perturbed problems through the Bidirectional Input Reader.
- **Step 2**: Generate formal Python ASTs via greedy policy execution ($r=4$).
- **Step 3**: Execute generated programs in the read-only sandbox.
- **Step 4**: Measure output accuracy and AST semantic invariance across clean vs. perturbed splits.

#### 4. Measurement Methodology
- Exact numerical match of execution output against ground-truth labels.
- Relative Performance Drop (RPD): $(\text{Score}_{\text{clean}} - \text{Score}_{\text{distractor}}) / \text{Score}_{\text{clean}}$.

#### 5. Control Arm
- Evaluate off-the-shelf Qwen2.5-Math-1.5B on the identical clean vs. distractor split to benchmark standard Transformer autoregressive degradation.

#### 6. Pass & Kill Criteria
- *Pass*: Parse-and-execute accuracy $\ge 50.0\%$; RPD under distractor insertion $\le 5.0\%$.
- *Kill*: Parse-and-execute accuracy $<35.0\%$; RPD $>25.0\%$, indicating vulnerability to prompt distraction.

#### 7. Compute & Wall-Clock Budget
- Evaluation of 3,000 items on 2× T4 GPUs: $\approx 12\text{ minutes}$ wall-clock.

---

### 12.3 Experiment 3 (Week 4): Absolute-Zero Proposer Regret Loop & Stage 1 Cord Cutting

#### 1. Objective & Core Hypothesis
- **Objective**: Operationalize the autonomous task proposer head, validating that Depth Regret ($R_{\text{depth}}$) maintains curriculum solvability within $[35\%, 65\%]$, and execute the Rocket Stage 1 cord-cutting protocol.
- **Core Hypothesis**: Proposer-solver self-play autonomously increases problem difficulty while sustaining rising Teacher Exceedance, triggering the triple-lock cord-cutting criteria within 5 autonomous generations [가설: the timing is optimistic; a miss is information, not project failure].

#### 2. Domain & Input Specification
- Mixed procedural domain (Game of 24, SVAMP, and Reasoning Gym algorithmic traces). 2,000 frozen teacher-failed evaluation problems ($\mathcal{D}_{\text{held-out}}^{\text{teacher\_fail}}$).

#### 3. Step-by-Step Procedure
- **Step 1**: Run 5 continuous autonomous self-improvement generations ($g_1 \to g_5$) without external human data.
- **Step 2**: Proposer head emits 2,500 tasks per generation, filtered by Depth Regret ($R_{\text{depth}}$).
- **Step 3**: Solver executes Snell MCTS; verified traces update core weights via Expert Iteration.
- **Step 4**: Audit Teacher Exceedance ($\Delta_{\text{TE}}$) and Random-Reward Control Delta ($\Delta_{\text{rand}}$) at each generation boundary.

#### 4. Measurement Methodology
- Proposer task solve rate; Teacher Exceedance $\Delta_{\text{TE}}(g)$; held-out benchmark score progression.

#### 5. Control Arm
- Parallel control track with proposer head rewarded by raw solver failure (standard adversarial self-play), demonstrating the Own-Level Prison failure mode.

#### 6. Pass & Kill Criteria
- *Pass*: Proposer solve-rate stays within $[35\%, 65\%]$; $\Delta_{\text{TE}} \ge 0.25$ with monotonic growth for 2 consecutive generations; Stage 1 cord cut successfully executed.
- *Kill*: Proposer collapses into unsolvable or trivial tasks; $\Delta_{\text{TE}}$ stalls below 0.10.

#### 7. Compute & Wall-Clock Budget
- 5 generations $\times 1.0\text{ hour} \approx 5.0\text{ GPU-hours}$ on Kaggle 2× T4 GPUs.

---

### 12.4 Top 5 Existential Risks & Mitigation Register

| # | Risk Description | Severity | Feasibility / Probability | Mitigation & Defense Strategy | Contingency / Fallback Plan |
|---|---|---|---|---|---|
| **R1** | **Comprehension Wall**: A $<100\text{M}$ non-Transformer reader fails to parse dense natural language (e.g., MuSR narratives or DROP passages) into formal IR [추정; see io_split.md research note]. | High | Moderate Risk [추정] | Port ELECTRA-style replaced token detection (RTD) pretraining [검증됨: https://arxiv.org/abs/2003.10555] into the 2-layer Bi-Gated DeltaNet reader; utilize pointer-generator copy mechanisms. | Fall back to hybrid retrieval-augmented reading, allocating 10M additional parameters to the input reader. |
| **R2** | **MCTS Search Latency Wall**: Python/SymPy sandbox verification overhead on Kaggle 4 vCPU creates an I/O bottleneck during MCTS rollouts. | Medium | Low-Moderate Risk [추정] | Batch verification calls via multiprocessing; compile frequent SymPy checks into vectorized C/Rust shared libraries; cache verified subtrees. | Limit MCTS budget to Snell compute-optimal allocation ($k=18$ mean sims); use greedy beam search for unambiguous queries. |
| **R3** | **Turing (T4) Triton Kernel Deprecation**: Kaggle T4 GPUs fail on modern Triton 3.3+ scan kernels [검증됨: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md]. | Medium | Low Risk [추정] | Pin environment strictly to Triton 3.2.0; maintain pure-PyTorch parallel chunked scan CUDA kernels with fp32 accumulators. | Execute heavy training runs on Kaggle TPU v5e-8 using native JAX `associative_scan`. |
| **R4** | **Subtle Reward Hacking via Verifier Edge Cases**: Core discovers zero-division or memory-overflow exceptions that return exit code 0 without solving the problem. | Fatal | Low Risk [추정] | Tri-Verification Reading Harness: check formal AST equivalence against ground-truth templates; assert non-empty typed output schema. | Immediate circuit breaker trigger; roll back to previous generation checkpoint; add AST static type validator. |
| **R5** | **Curriculum Stagnation (Own-Level Prison)**: Proposer head overfits to a narrow family of solvable algebraic structures, refusing to explore novel reasoning modes. | High | Moderate Risk [추정] | Enforce parameter-free **Depth Regret** rewards; mandate $50\%$ procedural tasks from external stepped generators into every generation batch. | Re-seed proposer with novel formal grammar templates; re-initialize proposer head weights. |

---
*End of Master Technical Architecture Blueprint (`work/blueprint_final.md`).*
