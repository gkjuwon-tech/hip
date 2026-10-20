# Handoff Report: Red Team Audit of Proposal 'Library' (ALEMBIC-LIB)

## 1. Observation
- Target File: `/home/user/hip/agy-blueprint/work/proposal_library_v1.md` (728 lines, 70,765 bytes).
- Binding Audit: `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md` (§1 & §2).
- Key Verbatim Quotes & Observations:
  1. **Executable Code in Proposal**: Section 12.1 lines 568–703 contains 135 lines of executable Python code (`def run_experiment_1()`, `class GameOf24Verifier`, `class SymbolicPolicyCore`, `class StitchCompressor`). Line 637 hardcodes regexes: `r"\d+\s*\/\s*\(\s*\d+\s*-\s*\(\s*\d+\s*\/\s*\d+\s*\)\s*\)"` and asserts `assert delta >= 15.0`.
  2. **Recurrent Depth FLOP Math**: Section 2.6 line 180 calculates inference FLOPs as $\text{FLOP}_{\text{rollout}} = 2 \times (6.871 \times 10^7) \times (4.096 \times 10^8) = 5.628 \times 10^{16}$ FLOPs, ignoring recurrent depth $r \in [1, 16]$ specified in Section 5.2 line 302 and Section 5.4 line 272.
  3. **T4 Latency Claims**: Section 8.3 line 467 claims: "Single-pass generation ($L=256$ tokens): $3.52 \times 10^{10}$ FLOPs ($\approx 0.9\text{ms}$ on T4). Full MCTS search query ($K=32$ rollouts): $1.13 \times 10^{12}$ FLOPs ($\approx 28.9\text{ms}$ on T4)."
  4. **Impossible 70B Teacher**: Section 2.3 line 129 specifies "Llama-3.1-70B-Instruct or Qwen2.5-72B" on a platform with 2x T4 16GB.
  5. **Missing Modules in Parameter Table**: Section 5.4 Table lines 339–352 omits formal token input embeddings ($2,048 \times 512 = 1.05\text{M}$), DeltaNet causal Conv1D ($86\text{k}$), DeltaNet output gates $Z$-projection ($3.67\text{M}$), and span pointer head ($262\text{k}$), undercounting parameters by 5.07M.
  6. **ARC-AGI Target**: Section 7.1 Table line 394 claims ARC-AGI-1 Expected Score = 47.5% without attention.
  7. **Citation Laundering**: Section 1.1 line 15 cites `arXiv:2504.13837v4` for distillation ceiling; Section 1.3 line 31 cites `arXiv:2006.08381v1` for Stitch; Section 5.5 line 360 omits $\alpha_t$ gate in Gated DeltaNet; Section 9.3 line 497 cites AlignScore for suffix-automaton masking; Section 10.2 line 511 claims 110+ TPUs from TRC.

## 2. Logic Chain
1. *From Observation 1 to Fatal Code Ruling*: The proposal violated the strict directive prohibiting executable code in proposals, and the included code hardcoded its own success using pre-selected regexes and test cases, invalidating the empirical validity of Experiment 1.
2. *From Observation 2 to Compute Explosion*: Per AUDIT §1.1, per-token compute with recurrent depth is $2 \times (N_{\text{non\_core}} + r \times N_{\text{core}})$. With $N_{\text{non\_core}} = 22.23\text{M}$ and $N_{\text{core}} = 51.54\text{M}$, at $r = 8$ the effective parameter count is $434.55\text{M}$ ($6.32\times$ the proposal's $68.71\text{M}$). Generation rollout compute increases to $3.56 \times 10^{17}$ FLOPs, blowing up 2x T4 rollout time to 2.54 hours per generation and dropping weekly throughput from 47 to 8.5 generations.
3. *From Observation 3 to Latency Refutation*: On Turing T4 ($300\text{ GB/s}$ memory bandwidth), autoregressive decoding at batch size 1 is memory-bandwidth bound at operational intensity $I = 1.0\text{ FLOP/Byte}$. Fetching $869.1\text{ MB}$ of weights per token ($r=8$) takes $2.897\text{ ms}$ per token. A 256-token pass takes $0.742\text{ seconds}$ ($824\times$ longer than the claimed $0.9\text{ ms}$). MCTS with $K=32$ rollouts and sandbox executions takes $3.8\text{ seconds}$ ($130\times$ longer than the claimed $28.9\text{ ms}$).
4. *From Observation 4 to Hardware Infeasibility*: A 70B model requires 140 GB fp16 or 38–42 GB in 4-bit quantization, exceeding the 32 GB PCIe split memory of 2x T4 and unable to run within Kaggle free quotas.
5. *From Observation 5 to Parameter Compliance*: Restoring the omitted functional modules increases the parameter total from $68.71\text{M}$ to $73.77\text{M}$, which remains safely below the $100\text{M}$ hard ceiling ($26.23\text{M}$ headroom).
6. *From Observation 6 to Benchmark Reality*: Attention-free TRM-MLP achieves only 29.6% on ARC-AGI-1 and 0.0% on Maze-Hard; claiming 47.5% on an attention-free Gated DeltaNet backbone without an explicit 2D spatial inductive bias exceeds empirical evidence.

## 3. Caveats
- No caveats regarding the mathematical derivations of memory bandwidth, operational intensity, or recurrent FLOP scaling; they are exact hardware roofline bounds.
- Regarding Stitch compression: benchmark performance of Stitch was evaluated based on the literature on typed lambda calculus; the exact runtime of Stitch on Python ASTs was not physically timed in a subprocess because writing executable benchmark code was forbidden.

## 4. Conclusion
Proposal ALEMBIC-LIB v1.0 has a fundamentally powerful concept: a zero-parameter external symbolic skill library that compounds capability without expanding neural weights. However, the proposal cannot be accepted in its current form due to 4 fatal flaws: (1) reading verification void, (2) 6.3x recurrent FLOP undercount, (3) 800x T4 latency fantasy, and (4) forbidden self-fulfilling executable code. It requires conditional revision in Phase C following the 11-point remediation matrix provided in `work/critique_library.md`.

## 5. Verification Method
1. Inspect critique file: `view_file` on `/home/user/hip/agy-blueprint/work/critique_library.md`.
2. Verify absence of executable code:
   `grep -nE "^(def |class |import |from |    return |print\()" /home/user/hip/agy-blueprint/work/critique_library.md` (must return 0 matches).
3. Verify parameter arithmetic:
   Embeddings ($8.39\text{M} + 1.05\text{M}$) + Bi-SSM Reader ($17.33\text{M}$) + DeltaNet Solver ($51.54\text{M}$) + Heads ($1.74\text{M}$) = $73.77\text{M} < 100\text{M}$.
4. Verify memory bandwidth arithmetic:
   $869.1 \times 10^6 \text{ bytes} / (300 \times 10^9 \text{ bytes/s}) = 2.897\text{ ms/token}$; $256 \times 2.897\text{ ms} = 741.6\text{ ms} \approx 0.742\text{ s}$.
