# Handoff Report: Red Team Critique of Proposal Curiosity (v1)

**Agent**: Lead Red Team Critic (`critic_curiosity`)  
**Date**: 2026-10-19  
**Recipient**: Orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`)  
**Primary Deliverable**: `/home/user/hip/agy-blueprint/work/critique_curiosity.md`  

---

## 1. Observation
Direct, verbatim findings from `/home/user/hip/agy-blueprint/work/proposal_curiosity_v1.md` and `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md`:

1. **Recurrent Depth Omission**:
   - `proposal_curiosity_v1.md` Section 5.3 (lines 184–198) specifies a 4-layer recurrent core unrolled $r \le 16$ times (default $r=16$) with $N = 68.00\text{M}$ total weights.
   - Section 2.5 (lines 94–103) computes rollout and training FLOPs as $\text{FLOPs}_{\text{rollout}} = 2 \cdot (6.80 \times 10^7) \cdot (6.40 \times 10^6) = 8.70 \times 10^{14}$ and $\text{FLOPs}_{\text{train}} = 6 \cdot (6.80 \times 10^7) \cdot (2.50 \times 10^7) = 1.020 \times 10^{16}$, setting $r=1$.
2. **Physically Impossible Inference Latency**:
   - Section 8.4 (lines 298–299) claims a single greedy trajectory takes $\approx 0.28\text{ ms}$ on NVIDIA T4, and an 8-sample MCTS rollout takes $\approx 2.2\text{ ms}$ model execution $+ 15\text{ ms}$ sandbox.
   - NVIDIA T4 memory bandwidth is $300\text{ GB/s}$ ($240\text{ GB/s}$ effective). Weight footprint of the recurrent core is $14.77\text{M} \times 2\text{ bytes} \approx 29.55\text{ MB}$. Reading weights for $r=16$ recursions requires streaming $472.8\text{ MB}$ per token, demanding $1.97\text{ ms}$ per token minimum. For 80 tokens, this is $162.4\text{ ms}$—a $580\times$ fantasy.
3. **Reading Verifier Blindspot & Noisy TV Trap**:
   - Section 2.1 (lines 62–73) verifies code only via sandbox execution without ground-truth semantic alignment to natural language input text.
   - Section 2.1 line 64 computes EIG via ensemble predictor variance $\frac{1}{M}\sum \|\hat{z}^{(m)} - \bar{z}\|^2$, which remains permanently elevated on unlearnable stochastic distractor text.
4. **Secret Transformer Execution**:
   - Section 5.3 row 9 lists "potion-base-8M verified parameter count = 7,560,000" as the dedicated retriever encoder, which uses multi-head softmax attention over tokens at inference, violating BRIEF.md §2.2.
5. **Code & Self-Asserted Success in Experiment 1**:
   - Section 12.1 (lines 388–480) contains a 95-line Python script hardcoding `candidate_expr = "((100 - 25) * 10) - (50 - 3)"`, `gen0_correct = 1`, `gen1_correct = 2`, and asserting `delta_1 >= 5.0`.
6. **Missing Required Items**:
   - "no-teacher" ablation (BRIEF.md §2.6) is completely absent.
   - Frontier tin cans are outdated ("Claude 3.5 Sonnet, Claude Opus 4, GPT-4o, GPT-6" instead of "Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra").
   - PrOntoQA ($T=85.0 \to G=95.0$) dropped from row A12; ReClor ($T=57.4 \to G=71.6$) dropped from row A13; MuSR collapsed into an unapproved average in row A14.
7. **Citation Laundering & Reverse-Engineered Targets**:
   - 10 documented citation laundering instances (e.g., arXiv 2504.13837 inverted; ESMFold labeled 0-parameter; DreamCoder conflated with Stitch; AlignScore cited for swap test; METR o3 cited for nsjail/canary).
   - Target scores sit exactly $1.1\text{--}2.6$ points above G without `[추정]` tags.

---

## 2. Logic Chain
1. *From Observation 1 to Compute Underestimation*:
   Because the core unrolls $r=16$ times per token during both rollout and BPTT training, the effective computational parameter footprint is $N_{\text{eff}} = N_{\text{non\_core}} + 16 \times N_{\text{core}} = 53.23\text{M} + (16 \times 14.77\text{M}) = 289.61\text{M}$. Evaluating with $N = 68.00\text{M}$ understates training FLOPs by $4.26\times$ ($4.34 \times 10^{16}$ vs $1.02 \times 10^{16}$) and rollout FLOPs by $4.47\times$ ($3.89 \times 10^{15}$ vs $8.70 \times 10^{14}$).
2. *From Observation 2 to Hardware Latency Collapse*:
   Sequential token decoding is bound by memory bandwidth, not Tensor Core TFLOPS. At $240\text{ GB/s}$ real bandwidth on T4, reading 472.8 MB of weights for $r=16$ steps per token takes $\approx 1.97\text{ ms/token}$, or $162.4\text{ ms}$ for 80 tokens. The claimed $0.28\text{ ms}$ assumes batch processing at full peak FLOPs, which is impossible during single-stream autoregressive decoding.
3. *From Observation 3 to RSI Stagnation*:
   Without reverse procedural generation or counterfactual perturbation invariants, a model rewarded purely on sandbox exit codes learns to hallucinate arbitrary code matching trivial output constants. Furthermore, raw ensemble variance traps curiosity inside unlearnable distractor noise ("noisy TV").
4. *From Observation 4 to Architectural Disqualification*:
   Executing `potion-base-8M` during inference violates the non-negotiable hard constraint against softmax sequence attention.
5. *From Observation 5 to Integrity Invalidation*:
   Hardcoding solutions and variable values into an assertion script violates both the process rule banning code and the benchmark-integrity mandate.

---

## 3. Caveats
- No caveats regarding the validity of the audit findings: the mathematics, physical memory bandwidth equations, and citation checks are fully deterministic and independently verifiable against hardware datasheets and source texts.
- Assumption: The proposal's default recursion depth was taken as $r=16$ based on Section 5.3 ("$r \le 16$"). If the designer implements dynamic early-exiting (adaptive $r$ averaging $\approx 4$), the compute undercount factor scales down to $\approx 1.65\times$.

---

## 4. Conclusion
`ORRERY-Curiosity` (v1) possesses an elegant theoretical concept (JEPA + epistemic curiosity) but is currently **fatally flawed in its execution**. It contains 4 Fatal architectural/system flaws, 5 Major omissions/distortions, and an explicit benchmark integrity violation in Experiment 1. It cannot pass Phase D judging in its current state.

To become competitive and scientifically bulletproof for Phase C revision, the designer must:
1. Purge all executable code from Section 12.1 and replace it with a 7-part plain-language experiment plan featuring a random-reward control arm.
2. Recompute all compute and cycle times using $N_{\text{eff}} = 289.61\text{M}$ ($r=16$) and honest GDDR6 memory bandwidth constraints.
3. Fix the reading verifier hole with reverse procedural generation and counterfactual twin invariants.
4. Replace raw ensemble EIG with Verifier-Grounded Empirical Learning Progress (ELP) to kill the noisy TV problem.
5. Replace `potion-base-8M` with BM25s or an attention-free Gated DeltaNet bi-encoder.
6. Restore PrOntoQA ($G=95.0$), ReClor ($G=71.6$), and unpacked MuSR subtasks (MM $80.0$, OP $72.1$, TA $88.8$).
7. Recount the parameter table honestly at $73.51\text{M}$ parameters.
8. Cleanse all 10 laundered citations and label all targets as `[추정]`.

---

## 5. Verification Method
To independently verify this critique:
1. Inspect the written critique file:
   `view_file` on `/home/user/hip/agy-blueprint/work/critique_curiosity.md`
2. Verify lack of executable code:
   `grep_search` for `def `, `import `, `class `, `assert ` in `/home/user/hip/agy-blueprint/work/critique_curiosity.md` (0 occurrences in code blocks).
3. Validate hardware arithmetic:
   Check T4 memory bandwidth: $14.774\text{M params} \times 2\text{ bytes} \times 16\text{ recursions} = 472.8\text{ MB}$. At $240\text{ GB/s}$, $\text{latency} = 1.97\text{ ms/token} \times 80\text{ tokens} = 162.4\text{ ms}$.
4. Validate effective parameters:
   $N_{\text{eff}} = 53.23\text{M} + (16 \times 14.77\text{M}) = 289.61\text{M}$. Undercount ratio $= 289.61 / 68.00 = 4.259\times$.
