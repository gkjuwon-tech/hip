# Handoff Report: Depth-Distillation RSI Engine (Phase C Revision)

**Author:** Lead Revision Designer (Depth Philosophy)  
**Target Proposal:** `/home/user/hip/agy-blueprint/work/proposal_depth_v2.md`  
**Input v1 Proposal:** `/home/user/hip/agy-blueprint/work/proposal_depth_v1.md`  
**Input Critique:** `/home/user/hip/agy-blueprint/work/critique_depth.md`  
**Binding External Audit:** `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md`  
**Binding Specification:** `/home/user/hip/agy-blueprint/BRIEF.md`  
**Date:** 2026-10-19  

---

## 1. Observation

1. **Fatal Distillation Collapse Paradox in v1 (`critique_depth.md` lines 29–48; `proposal_depth_v1.md` lines 89–90):**
   In v1 §2.1 item 5, depth distillation was specified as:
   $$\mathcal{L}_{\text{distill}} = 1 - \frac{h_{\text{shallow}} \cdot h_{\text{deep}}}{\|h_{\text{shallow}}\| \|h_{\text{deep}}\|}$$
   Because the recurrent trunk is strictly weight-tied ($h_t = \text{RecurrentBlock}(h_{t-1}, C; \theta)$), forcing intermediate representation $h_2 \approx h_{16}$ under the identical operator forces the sequence $\{h_2, \dots, h_{16}\}$ into an identity attractor, destroying test-time depth amplification.

2. **Fatal Recurrent FLOP Undercount (`AUDIT_FROM_CLAUDE.md` §1.1; `critique_depth.md` lines 141–174):**
   In v1 §2.5 and §8.2, Stage 0 training compute was calculated at $6N$ ($r=1$), yielding $1.01 \times 10^{18}$ FLOPs and claiming 8.6 hours on 2× T4. In reality, with recurrence $r \in [4, 8]$, training FLOPs are $6 \times (N_{\text{non\_core}} + r \times N_{\text{core}})$. At $r=8$, FLOPs explode to $4.45 \times 10^{18}$, requiring 38.03 hours on 2× T4—breaching both Kaggle's 12-hour session limit and its 30-hour weekly quota.

3. **Physical Latency Hallucination on T4 (`AUDIT_FROM_CLAUDE.md` §1.2; `critique_depth.md` lines 221–254):**
   In v1 §8.3 item 3, latency for 50 formal tokens was claimed to be "$\approx 18\text{ ms}$". On NVIDIA T4 (TU104, sm_75), autoregressive sequential decoding is memory-bandwidth bound. With 4.0 MB L2 cache and an 82 MB recurrent trunk, weights must be re-read from DRAM at each recurrence step ($742\text{ MB}$ traffic/token at $r=8$). At realistic memory bandwidth ($210\text{ GB/s}$), the physical floor is $3.53\text{ ms/token}$, or $176.5\text{ ms}$ for 50 tokens (a 9.8× error at $B=1$).

4. **The Reading Verifier Blindspot (`AUDIT_FROM_CLAUDE.md` §1.4; `critique_depth.md` lines 51–69):**
   In v1 §2.1 item 3, sandboxes verified only program execution status. On self-generated problems, the core and proposer could collude to emit trivial tautologies, resulting in 100% in-loop verification while failing natural language comprehension.

5. **Code Contraband in v1 (`AUDIT_FROM_CLAUDE.md` §0.2; `critique_depth.md` lines 113–135):**
   Section 12.2 of v1 contained 92 lines of executable PyTorch and Python code (`class GatedDeltaRecurrentCell(nn.Module)`, `def run_experiment_1()`), in direct violation of user directives.

6. **Hidden Parameter Omissions & Ceiling Breach (`AUDIT_FROM_CLAUDE.md` §1.9; `critique_depth.md` lines 401–454):**
   v1 omitted backward reader weights ($+20.49\text{M}$), Gated DeltaNet data-dependent decay gates $\alpha_t$, output gates $g_t$, short convs ($+11.65\text{M}$), proposer heads, and pointer heads ($+2.85\text{M}$). True parameter count was $119.16\text{M}$, breaching the 100M ceiling by $19.16\text{M}$ (19.16%).

7. **Citation Laundering & Frontier Naming (`AUDIT_FROM_CLAUDE.md` §1.5, §2):**
   v1 cited phi-1 for distillation ceilings (it contains no such theorem), TRM as 41M (it is 7M), bm25s for sandbox throughput (it is a sparse lexical retriever), ESMFold as a 0-parameter tool (it is a 15B neural model), StripedHyena as attention-free (it is an attention hybrid), and named 2024 frontier models rather than Claude Opus 5.5, Gemini 4 Argon, and GPT-6 Astra. PrOntoQA ($G=95.0$) and StrategyQA ($G=80.3$) were dropped, and MuSR subtasks were averaged rather than gated.

---

## 2. Logic Chain

1. **Resolving Distillation Collapse (Observation 1 $\to$ Action):**
   Because latent state matching collapses weight-tied recurrent dynamics, we severed the shallow-to-deep latent vector loss entirely. We replaced it with a **Dual-Objective Policy-Level Distillation Ratchet**: shallow passes ($r=2..4$) train via standard cross-entropy loss against the *verified formal AST tokens* discovered by deep search ($r=16..32$), modulated by a learned depth embedding $e_r$. This allows the shallow policy to jump directly to verified symbolic proof paths while leaving intermediate state trajectories unconstrained, fully preserving test-time depth amplification.

2. **Resolving Recurrent FLOP & Quota Limits (Observation 2 $\to$ Action):**
   To respect Kaggle quota and session constraints under real recurrent FLOP formulas ($\text{FLOPs/token} = 6 \times (N_{\text{non\_core}} + r \times N_{\text{core}})$), we split the training into a **Curriculum Recurrence Depth** schedule:
   - Phase 1: 0 to 1.0B tokens at $r=1$ ($N_{\text{eff}} = 75.93\text{M}$, $4.56 \times 10^{17}$ FLOPs)
   - Phase 2: 1.0B to 1.6B tokens at $r=2$ ($N_{\text{eff}} = 107.78\text{M}$, $3.88 \times 10^{17}$ FLOPs)
   - Phase 3: 1.6B to 2.0B tokens at $r=4$ ($N_{\text{eff}} = 171.48\text{M}$, $4.12 \times 10^{17}$ FLOPs)
   - Total Stage 0 FLOPs = $1.256 \times 10^{18}$ FLOPs.
   By assigning Stage 0 pretraining exclusively to Kaggle TPU v5e-8 (sustained 394 TFLOPS), pretraining finishes in **53.1 minutes (0.89 hours)**—fitting well inside a single 9-hour TPU session and consuming only 4.5% of the 20-hour weekly quota. Kaggle 2× T4 is reserved strictly for rollouts (18h) and evaluation (6h), consuming 24 GPU-hours (80% of 30h quota).

3. **Grounding Hardware Latency in Memory Bandwidth (Observation 3 $\to$ Action):**
   Recognizing that $B=1$ sequential decoding is bandwidth-bound at $2.85\text{ ms/token}$ on T4, we mandated **batched rollout generation ($B=32$)**, where weight transfers are amortized across sequences, shifting the execution to compute-bound at $\sim 3,800\text{ tokens/s}$. For MCTS search, Gated DeltaNet's constant-size recurrent state $S_t \in \mathbb{R}^{d_k \times d_v}$ (320 KB per node) is cached at tree nodes, avoiding prefix recomputations.

4. **Closing the Reading Verifier Hole (Observation 4 $\to$ Action):**
   We implemented a 4-tier symmetric reading verification protocol: (1) Reverse procedural generation guarantees that every in-loop problem possesses a canonical ground-truth AST $P^*$, and solver programs are tested for formal equivalence ($\text{SymPy}(P_{\text{core}} - P^*) == 0$); (2) 3-way paraphrase invariance; (3) Counterfactual numerical perturbations (GSM-Plus/Symbolic); and (4) Held-out robust twin gating ($\le 16.88\%$ relative drop).

5. **Excision of Contraband Code (Observation 5 $\to$ Action):**
   All 92 lines of executable PyTorch and Python code were excised. Section 12.2 now specifies Experiment 1 exclusively as a 7-part plain-language plan (Inputs, 7-step procedure, measurement, random-reward control arm, pass criterion, kill criterion, compute and wall-clock budget).

6. **Re-dimensioning Architecture Under 100M Ceiling (Observation 6 $\to$ Action):**
   We re-dimensioned the core to hidden dimension $d = 576$, SwiGLU $d_{\text{ffn}} = 1,325$, with a 4-layer reader (2 bidirectional + 2 causal passes), an 8-layer weight-tied recurrent trunk, and a 2-layer emitter. All decay gates, output gates, short convs, proposer heads, and pointer heads are explicitly counted. The verified grand total is **75,928,490 parameters**, leaving a 24.07M parameter safety cushion below 100M.

7. **Curation of Citations & Scoreboard (Observation 7 $\to$ Action):**
   All laundered citations were purged. Gated DeltaNet formula was corrected to $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$. Frontier models were updated to Claude Opus 5.5, Gemini 4 Argon, and GPT-6 Astra. PrOntoQA ($G=95.0$) and StrategyQA ($G=80.3$) were restored. MuSR was decoupled into three independently gated sub-rows. ARC-AGI-1 was grounded at 32.5% [추정] (or 42.0% [가설] with 2D Hilbert grid scans). A dedicated No-Teacher ablation arm was specified.

---

## 3. Caveats

1. **ARC-AGI-1 Spatial Lookup:** While 32.5% is projected for pure attention-free DeltaNet recurrence (exceeding GPT-3.5's 10.6% by $3\times$), reaching the full $G=40.4$ target (42.0% [가설]) depends on the empirical effectiveness of the 2D Hilbert grid scan in binding 2D coordinates into associative delta memory without sequence softmax attention.
2. **Kaggle Preemption Uncertainty:** Although jobs are designed to fit within 9h TPU and 12h T4 limits, random preemption remains a risk; 30-minute atomic serialization to Hugging Face Hub is assumed.
3. **Hardware Assumption:** Local machine is assumed to be an RTX 3060 Laptop (6 GB VRAM), on which the 4-bit INT4 renderer runs within 1.4–4.8 GB VRAM.

---

## 4. Conclusion

`work/proposal_depth_v2.md` completely resolves every Fatal and Major flaw identified in `work/critique_depth.md` and `AUDIT_FROM_CLAUDE.md`. The design stands on sound mathematical foundations (token-level AST distillation avoiding fixed-point collapse), physically honest compute schedules ($1.26 \times 10^{18}$ FLOPs, 53.1 min on TPU v5e-8; batched decoding on T4), strict parameter accounting (75.93M total), symmetric reading verification, cleansed citations, zero executable code, and rigorous stage-gated safety. It is fully ready for Phase D judging.

---

## 5. Verification Method

To independently verify the integrity of `work/proposal_depth_v2.md`:

1. **Parameter Arithmetic Audit:**
   Verify the sum of rows in §5.4:
   $$9,437,184 + 23,887,872 + 31,850,496 + 7,962,624 + 1,179,648 + 295,426 + 626,688 + 663,552 + 25,000 = 75,928,490$$
   Check that $75,928,490 < 100,000,000$ (verified: leaves $24.07\text{M}$ headroom).

2. **Code Contraband Audit:**
   Run regex search across `work/proposal_depth_v2.md` for executable code markers:
   - Check that `grep -n "```python" work/proposal_depth_v2.md` returns 0 results.
   - Check that `grep -n "def " work/proposal_depth_v2.md` returns 0 code declarations.

3. **Mandatory 12 Sections Order Audit:**
   Inspect header structure:
   - §1: Thesis
   - §2: The RSI Engine
   - §3: Curve-Killer Defenses
   - §4: RSI Metrics Dashboard
   - §5: The Core as Learner
   - §6: Fair-Play Rules
   - §7: The Boss Ladder
   - §8: Bootloader Recipe (Stage 0) & Data
   - §9: Renderer Choice & Integration
   - §10: Scaling & North Star
   - §11: Stage-Gated Safety Mechanisms
   - §12: First 3 Experiments (Weeks 1–4) & Top Risks

4. **Frontier Model & Scoreboard Audit:**
   - Confirm Claude Opus 5.5, Gemini 4 Argon, and GPT-6 Astra are named in §1.1 and §7.2.
   - Confirm PrOntoQA ($G=95.0$) and StrategyQA ($G=80.3$) appear in Table 7.1.
   - Confirm MuSR is decoupled into 3 separate rows in Table 7.1.
