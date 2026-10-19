# Research: Non-Transformer Architectures (topic "archs")

Scope: backbones and reasoning mechanisms for a <100M-trainable-parameter, non-Transformer, knowledge-externalized system that has to scale. Labels used throughout: [VERIFIED: url] means checked against a primary source or official repo; [ESTIMATE] means an informed number that was not verified; [HYPOTHESIS] means an untested design claim.

## 0. Bottom line

1. **No single non-Transformer backbone beats GPT-3.5 at 0.1B.** Every published recurrent or SSM language model reaches parity with Transformers only at matched parameters and tokens. RWKV-7 2.9B matches 3B Transformers [VERIFIED: https://arxiv.org/abs/2503.14456] and xLSTM 7B sits mid-pack among 7B models [VERIFIED: https://arxiv.org/pdf/2503.13427]. At 135M even a strong Transformer gets about 1.4% on GSM8K and about 31.5 on MMLU (cloze) [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M]. GPT-3.5's scores are 70.0 MMLU, 57.1 GSM8K (5-shot CoT) and 48.1 HumanEval [VERIFIED: https://arxiv.org/pdf/2303.08774]. Changing the backbone buys efficiency, not a 30x jump in capability.
2. The levers with evidence of **params-for-something trades** are these:
   - **Retrieval / external datastore.** RETRO matched GPT-3 on the Pile with 25x fewer parameters [VERIFIED: https://arxiv.org/abs/2112.04426v1]. Atlas-11B beat PaLM-540B on 64-shot NaturalQuestions (about 42.4 vs 39.6) [VERIFIED headline; exact numbers from secondary sources: https://alphaxiv.org/overview/2208.03299v3].
   - **Recursion / test-time depth.** TRM at 5–7M parameters gets 87.4% on Sudoku-Extreme and 44.6% on ARC-AGI-1 [VERIFIED: https://arxiv.org/html/2510.04871v1]. Huginn 3.5B reaches compute "equivalent to 50B parameters" on reasoning [VERIFIED: https://arxiv.org/abs/2502.05171].
   - **Sparse memory parameters.** Memory layers beat dense models with more than 2x the compute, mostly on factual tasks [VERIFIED: https://arxiv.org/abs/2412.09764]. **Under the user's fair-play rule, memory-layer values are trainable weights and count toward the 100M budget.**
3. **Recommended backbone family:** delta-rule recurrences (Gated DeltaNet / RWKV-7) or Mamba-2, with no softmax attention. Add (a) a retrieval read path over a non-parametric datastore, (b) a weight-tied recurrent reasoning core (TRM / Huginn style) for test-time compute, and (c) tool calls (interpreter, solvers). [HYPOTHESIS]
4. **T4 reality check:**
   - Recent Triton dropped Turing. Triton ≤3.2 lists "Compute Capability 7.0+" and Triton 3.3+ lists "8.0+" [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.2.0/README.md, https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md].
   - Every Triton-based kernel stack (flash-linear-attention, Mamba-2 SSD) therefore needs a **pinned Triton 3.2-era toolchain** on T4, or a fallback to pure-PyTorch / CUDA kernels.
   - The CUDA kernels in mamba-ssm and causal-conv1d do compile for sm_75 [VERIFIED: https://raw.githubusercontent.com/state-spaces/mamba/main/setup.py, https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/main/setup.py].

## 1. Compute envelope (Kaggle and the laptop)

- **Kaggle sessions:** 2x T4 (32GB total) or 1x P100. Sessions are capped at 12h for GPU and 9h for TPU [VERIFIED secondary, citing Kaggle docs: https://gpuperhour.com/blog/free-cloud-gpus-and-credits].
- **Weekly GPU quota:** about 30h. This number is user-reported; Kaggle's docs don't state it [same source; https://aimultiple.com/free-cloud-gpu].
- **Free TPU:** reported as v5e-8 (replacing v3-8 in 2025) by one source [https://aimultiple.com/free-cloud-gpu] and as v3-8 at 20h/week by another [https://pypi.org/p/kgz]. **Unresolved: check the Kaggle accelerator menu.** TPU means JAX/XLA. Custom CUDA and Triton scan kernels don't run there, but `jax.lax.associative_scan` / chunked matmul formulations of linear RNNs do [ESTIMATE].
- **T4 throughput:** about 65 TFLOPS fp16 tensor-core peak, 16GB, about 320 GB/s, no bf16 [ESTIMATE: NVIDIA spec sheet, not fetched]. Realistic MFU for small recurrent models is 25–40% [ESTIMATE].
- **Weekly token budget:**
  - Budget: 30 h × 2 GPUs × 65e12 × 0.3 ≈ 4.2e18 FLOP/week [ESTIMATE].
  - Training cost is about 6·N·D. A 100M model therefore trains on about 7B tokens/week.
  - Twelve weeks gives about 80B tokens [ESTIMATE].
  - For comparison, SmolLM2-135M trained on trillions of tokens [model card above]. **We cannot out-pretrain anyone, so data quality, distillation-free synthetic data, and test-time compute must carry the load.**
- **RTX 3060 laptop (6GB, Ampere sm_86):** has bf16, FlashAttention-2 and current Triton [ESTIMATE that sm_86 is supported by FA2 and Triton ≥3.3, consistent with the "Ampere" and "8.0+" statements above]. Use it as the **kernel-development and ablation box**. It is the only machine where modern FLA kernels run unpatched.

## 2. Architecture survey

| Arch | Core mechanism | Small-scale / headline evidence | Scaling evidence | T4 (sm_75) friendliness | Transformer-rule status |
|---|---|---|---|---|---|
| **Mamba (S6)** | Input-selective diagonal SSM, hardware-aware scan | MambaByte: competitive with subword Transformers on raw bytes [VERIFIED: https://arxiv.org/abs/2401.13660v1] | Mamba-1 paper claims Transformer parity to about 3B [ESTIMATE, paper not fetched] | Selective-scan CUDA kernel builds for sm_75 [VERIFIED: mamba setup.py]; fp16 only | Pure-Mamba OK |
| **Mamba-2 (SSD)** | Scalar-identity-times-A SSM ≡ structured masked linear attention; much larger state | 2–8x faster than Mamba-1 at parity with Transformers [VERIFIED: https://arxiv.org/abs/2405.21060v1] | Used in many hybrids [ESTIMATE] | SSD chunk kernels are **Triton** → need Triton ≤3.2 on T4 [VERIFIED README facts; kernel behaviour on sm_75 untested = HYPOTHESIS] | OK if no attention layers |
| **RWKV-7 "Goose"** | Generalized delta rule, vector gating, in-context learning rates; O(1) memory per token | 0.19B–2.9B models; 2.9B is 3B SOTA on multilingual and matches English 3B SOTA on far fewer tokens; recognizes all regular languages (beyond TC⁰) [VERIFIED: https://arxiv.org/abs/2503.14456] | 4 sizes, 3.1T-token corpus, Apache-2.0 [same] | Has a pure-CUDA kernel path (RWKV-LM) and FLA Triton path [ESTIMATE: repo not fetched] | OK, attention-free |
| **Gated DeltaNet** | Gating (fast erase) + delta rule (targeted overwrite) | Beats Mamba2 and DeltaNet on LM, commonsense, in-context retrieval, length extrapolation; hybrids with SWA/Mamba2 even better [VERIFIED: https://arxiv.org/abs/2412.06464v3] | ICLR 2025, 1.3B-scale experiments [ESTIMATE: scale not re-checked] | FLA (Triton) and NVlabs repo [VERIFIED: https://github.com/NVlabs/GatedDeltaNet] → Triton pin needed | Pure GDN OK; **GDN+SWA hybrid = attention, flag** |
| **GLA / DeltaNet / RetNet** | Linear attention with data-dependent decay / delta rule / fixed decay | RetNet: parallel, recurrent O(1) and chunkwise forms [VERIFIED: https://arxiv.org/abs/2307.08621v4] | RetNet claims mostly superseded by GDN/RWKV-7 [ESTIMATE] | FLA Triton | OK (no softmax over sequence) |
| **xLSTM (mLSTM/sLSTM)** | Exponential gating + matrix memory | xLSTM 7B: mid-range on Open LLM Leaderboard v2 among 7B; fastest 7B inference per authors [VERIFIED: https://arxiv.org/pdf/2503.13427] | 7B exists; data quality blamed for gap [same] | Has Triton kernels (mlstm_kernels) [ESTIMATE] → Triton pin | OK |
| **Hyena / StripedHyena** | Long implicit convolutions + gating | SH-7B: 50% smaller cache than GQA Transformer; 10/20/50% faster training at 32k/64k/131k [VERIFIED: https://hessian.ai/architectures-for-longer-sequences-and-efficient-inference-stripedhyena/] | Evo genomic models used this line [ESTIMATE] | FFT convs run anywhere | **StripedHyena includes GQA attention → hybrid, violates**; pure Hyena OK |
| **Liquid LFM2** | Gated short convs + a few GQA blocks, hardware-in-loop search | 350M–8.3B family [VERIFIED: https://arxiv.org/abs/2511.23404] | Dense and MoE variants [same] | Short convs are trivial on T4 | **Contains GQA attention → violates**; conv-only variant would pass |
| **Titans** | Neural long-term memory MLP updated by gradient ("surprise") at test time, with momentum and forgetting | Beats recurrent models and Transformers at the same window; >2M context [VERIFIED: https://arxiv.org/abs/2501.00663] | Small-scale only in paper [ESTIMATE] | Per-chunk inner-loop gradient; fp16 inner updates risky → keep fp32 [ESTIMATE] | **Core branch is (windowed) attention → violates as published**; memory module itself is attention-free and can sit on a GDN core |
| **Huginn (recurrent depth)** | Prelude → weight-tied recurrent block iterated r times → coda | 3.5B, 800B tokens; gains up to "50B-equivalent" compute; adaptive exit [VERIFIED: https://arxiv.org/abs/2502.05171] | One scale; test-time scaling shown | Any backbone | **Huginn's block is a Transformer → violates**; the *idea* transplants to a recurrent block |
| **HRM** | Two coupled recurrent modules (slow planner, fast worker) | 27M, about 1000 examples, ARC-AGI-1 40.3%, ARC-AGI-2 5.0%, Sudoku-Extreme 55.0% [VERIFIED: https://arxiv.org/html/2510.04871v1] | Puzzle-only | Small | **Modules are 4-layer Transformers → violates** |
| **TRM** | Single 2-layer net recursed over (x, y, z) with deep supervision | TRM-Att 7M: ARC-1 44.6%, ARC-2 7.8%, Maze-Hard 85.3%, Sudoku 74.7%. **TRM-MLP (attention-free) 5M: Sudoku 87.4%**, but ARC-1 29.6% and Maze 0.0% [VERIFIED: https://arxiv.org/html/2510.04871v1] | Trains on 1 L40S <36h (Sudoku) to 4 H100 about 3 days (ARC) [same] | ARC-scale run ≈ 4 H100 × 72h ≫ one Kaggle week [ESTIMATE] | TRM-MLP passes; TRM-Att uses self-attention → violates |
| **Diffusion LMs (LLaDA, MDLM, Mercury)** | Masked/discrete diffusion, parallel denoising | LLaDA 8B on 2.3T tokens ≈ LLaMA3-8B on ICL; fixes reversal curse [VERIFIED: https://arxiv.org/abs/2502.09992v3]. Mercury Coder Mini/Small 1109 / 737 tok/s on H100 [VERIFIED: https://arxiv.org/abs/2506.17298] | LLaDA 8B; 0.13M H800-hours | Denoiser cost = steps × forward | **LLaDA and Mercury denoisers are Transformers → violate**. Diffusion *objective* on a bidirectional RNN/SSM denoiser is allowed but unproven [HYPOTHESIS] |
| **JEPA (LLM-JEPA)** | Predict embedding of view B from view A, added to next-token loss | Beats standard objectives on NL-RX, GSM8K, Spider, RottenTomatoes across Llama3/Gemma2/OLMo [VERIFIED: https://arxiv.org/abs/2509.14252] | Fine-tune scale, NeurIPS 2025 | Objective only, backbone-agnostic | Objective OK on any backbone |
| **Energy-Based Transformers** | Learn energy E(x, ŷ); predict by gradient descent on ŷ (System-2) | Up to 35% higher scaling rate than Transformer++ [secondary: https://liner.com/en/review/energybased-transformers-are-scalable-learners-and-thinkers; paper: https://arxiv.org/abs/2507.02092] | Small scale | Inner-loop gradients double memory [ESTIMATE] | **EBT backbone is a Transformer → violates**; energy objective on GDN is allowed [HYPOTHESIS] |
| **Memory layers / product keys** | Trainable sparse KV lookup, product-quantized keys; params without FLOPs | Beats dense models with >2x compute, especially factual QA; scaled to 128B memory params, 1T tokens, base ≤8B [VERIFIED: https://arxiv.org/abs/2412.09764] | Strongest params-vs-FLOP scaling evidence here | Bandwidth-bound gathers; T4 bandwidth is low [ESTIMATE] | Attention-free; **but memory values are trainable params → count in 100M** |
| **Small MoE** | Sparse experts | OLMoE 6.9B total / 1.3B active, 5.1T tokens, beats Llama2-13B-Chat [VERIFIED: https://arxiv.org/abs/2409.02060] | Strong | Many-expert routing is inefficient on 2x T4 [ESTIMATE] | Neutral; **total params count**, so MoE does not help under a 100M *total* cap |
| **BLT / MambaByte** | Tokenizer-free; entropy-based byte patches | BLT matches tokenized LLMs FLOP-controlled to 8B / 4T bytes; better long-tail and robustness [VERIFIED: https://arxiv.org/abs/2412.09871v1] | Patch size and model size scale together | Byte sequences are about 4x longer → recurrent backbones benefit | **BLT's latent model is a Transformer → violates**; MambaByte passes |
| **Modern Hopfield** | Continuous associative memory, exponential capacity in dimension | Update rule ≡ attention [VERIFIED: https://arxiv.org/abs/2008.02217] | Theoretical | Same as attention | **Equivalent to softmax attention → treat as attention for rule purposes** |

## 3. What counts as "Transformer": proposed rule text

Proposed: **"No softmax (or Hopfield-equivalent) attention over the token sequence in the backbone, at any layer."** Under that rule:

- **Allowed:** SSMs, linear RNNs, delta-rule/linear-attention recurrences (constant-size state, no N×N score matrix), convolutions, MLPs, test-time-trained memories, and hash/product-key lookups.
- **Gray zone (user must decide):**
  - **Cross-attention from a recurrent state to K retrieved chunks** (RETRO-style chunked cross-attention).
  - **Softmax over a top-k product-key memory** (as in memory layers).
  - Both are softmax over *external items*, not over the sequence. [HYPOTHESIS] They should be allowed with K ≤ 64 and reported separately. Otherwise retrieval fusion must be done by appending retrieved text into the recurrent stream, which is pure-RNN and costs about K×chunk tokens of scan.
- **Violations as published:** StripedHyena, LFM2, Titans (MAC/MAG/MAL with attention core), Huginn, HRM, TRM-Att, LLaDA, Mercury, EBT, BLT, and GDN+SWA hybrids. In each case **the mechanism can be ported**; the reference model cannot be used.

## 4. Scaling: what grows from 0.1B to 1B to 10B

| Axis | Evidence | Knob |
|---|---|---|
| Backbone params | RWKV-7 0.19→2.9B monotone [VERIFIED: https://arxiv.org/abs/2503.14456]; xLSTM to 7B | Width/depth of GDN/RWKV-7 core |
| Non-parametric datastore | RETRO: 2T-token DB ≈ 25x param savings [VERIFIED: https://arxiv.org/abs/2112.04426v1] | DB size (free: Wikipedia, arXiv, code, multilingual incl. Korean); index quality |
| Sparse memory params | Memory layers to 128B memory params [VERIFIED: https://arxiv.org/abs/2412.09764] | Only above the 0.1B milestone (counts against budget) |
| Test-time depth | Huginn up to 50B-equivalent [VERIFIED: https://arxiv.org/abs/2502.05171]; TRM recursion [VERIFIED: https://arxiv.org/html/2510.04871v1] | Iterations r, adaptive halting |
| Test-time learning | Titans memory to >2M context [VERIFIED: https://arxiv.org/abs/2501.00663] | Inner-loop LR and steps |
| Tools / verifiers | Not surveyed here [ESTIMATE: separate topic] | Interpreter, CAS, SAT/SMT, proof checkers |
| Self-generated data | Not surveyed here | Verifier-filtered synthetic traces |

[HYPOTHESIS] The central bet is that capability ≈ f(backbone params, datastore tokens, test-time FLOPs, tool coverage) and that the last three are nearly free on a 0 KRW budget. No paper shows this composite scaling law for a non-Transformer system. **It must be measured: a 3x3 grid of {20M, 50M, 100M} params × {r = 1, 4, 16} recursions, with and without retrieval.**

## 5. Hard walls

1. **Open-ended chat and instruction following at 100M.**
   - Retrieval helps facts, and recursion helps puzzles. Neither has evidence of improving fluent long-form generation quality at 100M.
   - TRM/HRM evidence is all on fixed-size grid puzzles trained per task. It doesn't transfer to language by default: TRM-MLP collapses on large grids (Maze 0.0%) [VERIFIED: https://arxiv.org/html/2510.04871v1].
   - **This is the most likely category to lose to GPT-3.5.** [ESTIMATE]
2. **Code (HumanEval 48.1).** An interpreter allows execute-and-repair loops, but generating syntactically valid code needs a decent prior. At 135M, Transformer baselines are near zero on reasoning-heavy benchmarks (GSM8K 1.4) [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M].
3. **Recurrent models are weak at exact in-context recall.** A fixed state can't hold arbitrary context [ESTIMATE: well known; the GDN paper targets exactly this, https://arxiv.org/abs/2412.06464v3]. Mitigations: delta-rule overwrites, re-reading via retrieval, and Titans-style test-time memory.
4. **T4 numerics.** There is no bf16, so fp16 recurrences with exponential gates (xLSTM, RWKV-7 decay) can overflow. Keep the state and scan accumulators in fp32 and use fp16 matmuls [ESTIMATE].
5. **Toolchain.** Triton ≥3.3 does not target sm_75 [VERIFIED READMEs above]. Pin torch/Triton to a version that ships Triton 3.2 [ESTIMATE: torch 2.6 pairs with Triton 3.2; verify]. Also maintain a pure-PyTorch chunked fallback for every kernel.
6. **FlashAttention-2 is Ampere+.** Turing has a separate partial fork [VERIFIED: https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md]. This is irrelevant if the design is attention-free, which is one more argument for being attention-free.

## 6. Recommended stack for this project [HYPOTHESIS, to be ablated]

- **Backbone (about 60–80M):** byte-level or small-vocab (≤32k, Korean-aware) **Gated DeltaNet or RWKV-7** core, about 12–16 layers × d=512–768. Zero softmax attention.
- **Reasoning core:** the middle K layers are weight-tied and iterated r ∈ [1, 32] times with adaptive halting (Huginn/TRM transplant), trained with truncated backprop through the recursion and deep supervision.
- **Memory:** (a) a Titans-style test-time-trained MLP memory (a few M params) per episode; (b) a **non-parametric datastore** (public text and code, embedded by an in-budget ≤10M encoder that counts in the 100M); (c) no product-key memory at the milestone, kept as the 1B+ scaling knob.
- **Objectives:** next-byte/token + LLM-JEPA embedding loss [VERIFIED effect at fine-tune scale: https://arxiv.org/abs/2509.14252] + optional energy/verifier head for reranking candidates.
- **Hardware split:** 3060 for kernel development and bf16 ablations; Kaggle 2x T4 fp16 for main training with pinned Triton; TPU only if it is verified available and the core is ported to JAX scans.

## Sources (all fetched or returned by search in this session)

https://arxiv.org/abs/2503.14456 · https://arxiv.org/abs/2412.09764 · https://arxiv.org/html/2510.04871v1 · https://arxiv.org/abs/2502.05171 · https://arxiv.org/abs/2412.06464v3 · https://github.com/NVlabs/GatedDeltaNet · https://raw.githubusercontent.com/state-spaces/mamba/main/setup.py · https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/main/setup.py · https://raw.githubusercontent.com/triton-lang/triton/v3.2.0/README.md · https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md · https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md · https://arxiv.org/abs/2501.00663 · https://arxiv.org/abs/2502.09992v3 · https://arxiv.org/abs/2506.17298 · https://arxiv.org/abs/2412.09871v1 · https://arxiv.org/pdf/2503.13427 · https://arxiv.org/abs/2405.21060v1 · https://arxiv.org/abs/2401.13660v1 · https://arxiv.org/abs/2307.08621v4 · https://hessian.ai/architectures-for-longer-sequences-and-efficient-inference-stripedhyena/ · https://arxiv.org/abs/2511.23404 · https://arxiv.org/abs/2507.02092 · https://arxiv.org/abs/2509.14252 · https://arxiv.org/abs/2409.02060 · https://arxiv.org/abs/2008.02217 · https://arxiv.org/abs/2112.04426v1 · https://alphaxiv.org/overview/2208.03299v3 · https://arxiv.org/pdf/2303.08774 · https://huggingface.co/HuggingFaceTB/SmolLM2-135M · https://gpuperhour.com/blog/free-cloud-gpus-and-credits · https://aimultiple.com/free-cloud-gpu · https://pypi.org/p/kgz
