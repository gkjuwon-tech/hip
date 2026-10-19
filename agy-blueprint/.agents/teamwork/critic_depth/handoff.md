# Phase B Red Team Handoff Report: Depth-Distillation RSI Engine Critique

**Target Proposal:** `/home/user/hip/agy-blueprint/work/proposal_depth_v1.md`  
**Output Critique:** `/home/user/hip/agy-blueprint/work/critique_depth.md`  
**Agent:** Lead Red Team Critic (`critic_depth`)  
**Timestamp:** 2026-10-19T09:49:00Z  

---

## 1. Observation

Direct textual and mathematical observations from `work/proposal_depth_v1.md`, verified against `AUDIT_FROM_CLAUDE.md`, `BRIEF.md`, and research baselines:

1. **Latent Distillation Equation (§2.1, lines 89–90):**
   > *"Latent Distillation: Cosine loss aligns shallow states $h_{t=2}$ with deep states $h_{t=16}$: $\mathcal{L}_{\text{distill}} = 1 - \frac{h_{\text{shallow}} \cdot h_{\text{deep}}}{\|h_{\text{shallow}}\| \|h_{\text{deep}}\|}$"*
   The recurrent reasoning trunk is weight-tied (§5.4 Row 3: *"8 Causal Gated DeltaNet layers... iterated $r \in [1, 32]$ times... executed recurrently with weight tying"*).

2. **FLOP Math & Stage 0 Training Time (§2.5 & §8.2, lines 123, 133–135, 321–329):**
   > *"Training FLOPs/token: $6N \approx 5.05 \times 10^8$."*  
   > *"Total Stage 0 FLOPs: $\text{FLOPs}_{\text{boot}} = 6 \times (84.18 \times 10^6) \times (2.0 \times 10^9) \approx 1.01 \times 10^{18} \text{ FLOPs}$."*  
   > *"Kaggle 2x T4 (30 h/wk, sustained 32.5 TFLOPS): $T_{\text{T4}} = \frac{1.01 \times 10^{18}}{32.5 \times 10^{12}} \approx 31,076 \text{ s} \approx \mathbf{8.6 \text{ hours}}$."*  
   Recurrent depth factor $r$ is completely omitted from the training calculation ($r=1$ used).

3. **Inference Latency Claim (§8.3, lines 334–335):**
   > *"Inference Latency: At depth $r=8$, generating 50 formal tokens requires $\approx 6.7 \times 10^8$ FLOPs. Latency on single T4 is $\approx 18\text{ ms}$; cost is $0.00."*  
   NVIDIA T4 memory bandwidth is $300\text{ GB/s}$ peak, with L2 cache of $4\text{ MB}$ (`compute.md` §2).

4. **Reading Verifier Specification (§2.1, lines 85–86):**
   > *"Candidate formal programs are dispatched to an isolated, unprivileged Docker sandbox executing Python 3.12, SymPy 1.13, Z3 SMT, Lean 4, or VAL... The sandbox returns a strict boolean $\{0, 1\}$."*  
   No autoformalization verifier, paraphrase invariance check, or counterfactual twin gate is specified inside the self-play loop.

5. **Self-Synthesized Verification Wrappers (§2.4, lines 117–118):**
   > *"2. DSL Interface Synthesis: The core synthesizes Python interface wrappers for new verification engines (e.g., importing `sympy.solvers`, `z3.Solver`, or `lean4` REPL)... promoted to Active Training Fuel."*

6. **Executable Code Contraband (§12.2, lines 437–528):**
   Contains 92 lines of executable PyTorch and Python code (`class GatedDeltaRecurrentCell(nn.Module)`, `class LatentDepthReasoner(nn.Module)`, `def run_experiment_1()`).

7. **Parameter Table Omissions (§5.4, lines 235–250):**
   Table lists 84,177,411 parameters.
   - Row 2 counts 4 bidirectional layers at $4 \times 5,122,560 = 20,490,240$, omitting backward weights ($20.49\text{M}$).
   - Omits Gated DeltaNet $\alpha$-decay projections ($W_\alpha \in \mathbb{R}^{d \times d}$), output gates ($W_g \in \mathbb{R}^{d \times d}$), and short 1D depthwise convolutions across 14 physical layers ($11.65\text{M}$).
   - Omits proposer network ($2.5\text{M}$) and DROP pointer query weights ($0.66\text{M}$).
   True parameter count: $119.16\text{M}$ (violates $<100\text{M}$ ceiling).

8. **Citation Distortions (§1.1, §1.2, §2.5, §5.2, §10.3):**
   - Cites arXiv:2306.11644 (phi-1) for distillation ceiling.
   - Cites TRM (arXiv:2510.04871) as "executing exactly 41M recurrent parameters" (TRM is 7M).
   - Cites bm25s (arXiv:2407.03618) for "~800 verifier checks/s".
   - Cites ESMFold as a "0-parameter physical verifier" (ESMFold is 15B neural network).
   - Inverts Gated DeltaNet state update equation in §5.2 line 209.

9. **Benchmark Gaps & Outdated References (§1.1, §7.1, §7.2):**
   - Omitted PrOntoQA ($T=85.0 \to G=95.0$) and StrategyQA ($T=70.3 \to G=80.3$).
   - MuSR subtasks not independently gated.
   - Frontier models named as Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro instead of Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra.
   - Arbitrary probabilities listed: 75%, 45%, 25%, 10% in §7.2.

---

## 2. Logic Chain

1. **Distillation Collapse (Ref: Observation 1):**
   Because the recurrent trunk is weight-tied across all recursion steps $t \in [1, r]$, minimizing cosine distance between $h_{t=2}$ and $h_{t=16}$ forces the recurrent operator $\text{RecurrentBlock}(\cdot, C; \theta)$ to behave as an identity mapping between steps 2 and 16. If $h_2 \approx h_{16}$, the intermediate steps cannot evolve the latent representation. Consequently, when the model attempts test-time search at $r=16$ or $r=32$, it generates the same latent state as $r=2$. The "amplify-then-distill" premise implodes.

2. **Compute Arithmetic Collapse (Ref: Observation 2):**
   Recurrent depth is executed iteratively per token. In Stage 0, training the core with recurrence ($r=8$) scales the effective parameter count from 84.18M to $371.04\text{M}$. Training $2.0\text{B}$ tokens requires $6 \times 371.04\text{M} \times 2.0\text{B} = 4.45 \times 10^{18}\text{ FLOPs}$. On Kaggle 2× T4 (sustained 32.5 TFLOPS), this requires $38.03\text{ hours}$. Because Kaggle enforces a strict 12.0-hour session limit and a 30-hour weekly quota, this training job will fail to complete and exhausts the user's entire compute allocation before a single generation finishes.

3. **Inference Latency Impossibility (Ref: Observation 3):**
   Autoregressive decoding at batch size 1 requires reading all active weights from DRAM for every token. At $r=8$, reading 86.4 MB non-core + $8 \times 81.96\text{ MB}$ recurrent core equals $742.08\text{ MB}$ per token. The 4 MB L2 cache on a T4 cannot hold the 82 MB core. At 70% of 300 GB/s bandwidth ($210\text{ GB/s}$), each token takes $\ge 3.53\text{ ms}$. Generating 50 tokens requires at least $176.5\text{ ms}$ (a 9.8× error vs. the 18 ms claim). For MCTS with 64 rollouts at $r=16$, query latency exceeds 21 seconds.

4. **Reading Hole & Alignment Breach (Ref: Observation 4 & 5):**
   Executing candidate programs in Python/SymPy confirms syntax and runtime validity, but does not confirm that the code models the natural language input. In an autonomous loop with a self-play proposer, the core optimizes for code that returns ground-truth numbers via trivial shortcuts without parsing natural language. Furthermore, allowing the core to synthesize its own verification wrappers directly violates BRIEF.md §2.7 and enables trivial `return True` exploits.

5. **Parameter Ceiling Breach (Ref: Observation 7):**
   A bidirectional recurrent layer contains both forward and backward weights ($2\times$). Gated DeltaNet requires $\alpha_t$ projection, $W_g$ output gating, and depthwise conv1d layers. Proposing tasks and copying spans requires additional pointer and MLP heads. Adding these omitted modules pushes the true parameter count to $119.16\text{M}$, violating the 100M ceiling.

---

## 3. Caveats

1. **JAX vs. PyTorch TPU Kernel Efficiency:** The TPU v5e-8 throughput numbers assume standard 25% MFU using `jax.lax.associative_scan`. If the designer uses PyTorch/XLA instead of JAX, MFU may drop to 12–15%, requiring further batch size tuning.
2. **Kaggle TPU Availability Fluctuation:** The audit relies on Kaggle's stated TPU v5e-8 quota of ~20 h/week. During competition peaks, preemptions may necessitate executing rollouts entirely on the laptop RTX 3060.
3. **No Hidden Attention Detected:** Inspection of the Gated DeltaNet equations confirmed that the backbone does not secretly hide sequence softmax or Hopfield attention. The architecture is genuinely attention-free.

---

## 4. Conclusion

`proposal_depth_v1.md` contains fundamentally brilliant design elements—specifically, depth regret ($R = \max(0, \text{Score}_{\text{deep}} - \text{Score}_{\text{shallow}})$) and the depth-distillation ratchet—which are highlighted in `AUDIT_FROM_CLAUDE.md` §3 as top-tier innovations worth preserving.

However, the v1 proposal suffers from **seven fatal flaws** that disqualify it from production:
- Latent state-matching distillation collapses recurrent capacity.
- Training FLOPs are undercounted by 4.4×, crashing the Kaggle 12-hour session limit.
- Inference latency is physically understated by 9.8× due to memory bandwidth ignorance.
- Natural language reading comprehension is completely unverified in self-play.
- Model-synthesized verifiers violate BRIEF §2.7.
- Executable Python code in §12.2 violates the NO CODE directive.
- True parameters (119.2M) breach the 100M hard limit.

All seven flaws have been mapped to **rigorous, concrete architectural fixes** in `work/critique_depth.md` that preserve the model's high-ambition recursive self-improvement goals while establishing mathematical, physical, and hardware feasibility.

---

## 5. Verification Method

To independently verify this critique:
1. **Parameter Recount:** Inspect §5.4 in `work/proposal_depth_v1.md` and verify that 4 bidirectional layers are calculated as $4 \times 5.12\text{M}$ instead of $4 \times 2 \times 5.12\text{M}$. Verify that $W_\alpha$, $W_g$, and depthwise convs from arXiv:2412.06464 are absent from the arithmetic.
2. **FLOP Recomputation:** In §8.2, calculate $6 \times (N_{\text{non\_core}} + 8 \times N_{\text{core}}) \times 2.0\times 10^9$; confirm it equals $4.45 \times 10^{18}$ FLOPs, which at 32.5 TFLOPS takes 38.0 hours.
3. **Memory Bandwidth Verification:** Multiply $742.08\text{ MB}$ by 50 tokens and divide by 210 GB/s (70% of T4 300 GB/s peak); verify that the minimum sequential decoding latency is $176.5\text{ ms}$.
4. **Code Inspection:** Check §12.2 of `work/proposal_depth_v1.md` to confirm the presence of 92 lines of prohibited executable Python script.
5. **Critique Verification:** Inspect `/home/user/hip/agy-blueprint/work/critique_depth.md` to confirm full coverage of AUDIT §1 and §2 with zero executable code.
