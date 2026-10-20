# CORTEX-v1: A Neuro-Inspired Recursive Reasoner (86.8M trainable parameters)

Design philosophy: a hierarchical recurrent core that thinks in latent space, a hippocampus-like fast episodic memory, and a cortex-like external retrieval memory. Language is decoded only at the end.
Labels: [VERIFIED: url] = taken from the research digest, which checked the primary source. [ESTIMATE] = my arithmetic or judgement. [HYPOTHESIS] = untested design claim.

---

## 1. Thesis: where big models spend their parameters, and how to get those abilities without the parameters

A dense LLM uses its parameters for four jobs at once:

1. **A fact store.** Knowledge capacity is about 2 bits per parameter [VERIFIED: https://arxiv.org/html/2404.05405].
2. **Serial compute depth.** Each layer is one step of thought, and depth costs parameters.
3. **Working memory.** The KV cache, plus the circuits that read it.
4. **A fluency prior.** The surface statistics of human text.

A 100M model loses all four contests at once, which is why SmolLM2-135M scores 4.1 on TriviaQA and 1.4 on GSM8K [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M]. The brain does not solve these problems with more cortex. It handles them with **time, state and the outside world**. CORTEX moves three of the four jobs out of the parameter budget, and hands a fifth (exact computation) to tools:

- **Facts** go to an external, re-indexable datastore (the "cortex").
- **Depth** goes to time: a small, weight-tied recurrent core is iterated up to hundreds of effective layers with adaptive halting. This is the TRM/HRM lesson: 7M parameters scored 44.6% on ARC-AGI-1 [VERIFIED: https://arxiv.org/html/2510.04871v1].
- **Working memory** goes to *state, not weights*: a delta-rule fast-weight matrix plus a test-time-trained memory MLP (the "hippocampus"), written per query.
- **Exact computation** goes to zero-parameter tools: Python, SymPy, Z3 and a unit-test runner.

What remains in the weights is a **library of procedures** (decompose, query, bind, compute, verify, express) and a **fluency prior**. The honest core of this proposal is that fluency cannot be borrowed cheaply. It is the one job that stays in the parameters, and it is where this design is most likely to lose (§5).

---

## 2. Fair-play rules (binding on this proposal)

1. **Parameters** = every trainable weight executed at inference. That includes embeddings, encoder, recurrent core, hippocampal projections, the *initial weights* of the test-time memory MLP, decoder, verifier, halting, router and copy heads. The cap is a total under 100M; CORTEX-v1 uses **86.85M**.
2. **Not parameters, but disclosed** with size, hash and build script: tokenizer, BM25 statistics, IVF centroids, binary-quantization thresholds, and the datastore text. Per-query fast weights are state, not parameters. They are created from scratch for each query and discarded afterwards.
3. **Allowed at inference:** a frozen, dated, offline datastore snapshot; Python, SymPy, Z3 and Lean in a sandbox; a deterministic unit-test runner for tests *the model wrote itself*.
4. **Banned at inference:** any neural network above 100M parameters, any LLM API, live web access, eval items or their solutions in the datastore, hidden benchmark tests during generation, and routing or prompts keyed on dataset identity.
5. **Strict non-Transformer rule (stricter than required):** no softmax, and nothing Hopfield-equivalent, over the token sequence **or over retrieved items**. All content reads are delta-rule / linear-recurrence reads or MLP mixing. The gray zone named in the archs digest (cross-attention to K retrieved chunks) is therefore avoided completely, so nobody can say "it's secretly a Transformer".
6. **Training-time teachers:** open-weight models (Apache/MIT/Llama-licensed) may generate training data. Every teacher and its size is logged. **No GPT-3.5 or GPT-4 output is used anywhere**, which excludes TinyGSM, MetaMathQA, UltraChat and TinyStories from weights training. This follows OpenAI's terms [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/] and keeps the head-to-head comparison clean.
7. **Decontamination:** 13-gram plus MinHash plus an embedding check (cosine > 0.9) against every eval set, for both training data and the datastore. The overlap report is published. Every answer gets a retrieval-audit log.
8. **Targets** are the maximum reported GPT-3.5 score per row (baseline digest §4). One checkpoint and one configuration are used for all rows.

---

## 3. Architecture

### 3.1 Component diagram

```
                         ┌───────────────────────────── CORTEX (non-parametric) ─────────────────────────────┐
                         │  20B→100B tokens, 128-tok chunks; BM25 (bm25s) + 256-bit binary dense (IVF, mmap)   │
                         │  shards: EN-edu | KO | code | math | dialog-exemplars | KG triples                  │
                         └───────────▲───────────────────────────────────────────────┬──────────────────────┘
                                     │ query (sparse terms + 256-d vector)            │ top-50 → rerank → top-8
                                     │                                                ▼
 prompt ──► [Tokenizer 32k ko/en/code] ──► SENSORY ENCODER (6× Gated-DeltaNet, fwd+rev scan, shared weights)
                                     │             │ token states h_t                    │ chunk states
                                     │             ▼                                     ▼
                                     │    ┌──────── HIPPOCAMPUS (per-query state) ─────────────────┐
                                     │    │ (a) delta-rule fast-weight matrix M ∈ 8×(64×64)         │
                                     │    │     M ← M − β (M k − v) kᵀ   (targeted overwrite)       │
                                     │    │ (b) Titans-style memory MLP, few SGD steps on "surprise" │
                                     │    └──────────▲──────────────┬───────────────────────────────┘
                                     │      write   │              │ read  r = M q + MLP_mem(q)
                                     │   (tool out) │              ▼
                                     │    ┌──── PREFRONTAL LOOP: latent workspace (no tokens) ─────┐
                                     │    │  x = pooled problem code, y = 64 answer slots,          │
                                     │    │  z = 64 scratch slots (512-d)                           │
                                     │    │  L-module (2 Mixer blocks): z ← f_L(x, y, z, read)  ×6  │
                                     │    │  H-module (2 Mixer blocks): y ← f_H(y, z, read)     ×1  │
                                     │    │  = 1 cycle; 3 cycles = 1 supervision step; ≤16 steps    │
                                     │    │  heads: HALT(Q) | ROUTE{answer, retrieve, tool}         │
                                     │    │         | VERIFY(value) | QUERY-vector                  │
                                     │    └──────┬──────────────┬─────────────────┬─────────────────┘
                                     └───────────┘ re-query     │ tool request    │ answer slots y*
                                                                ▼                 ▼
                                             [Python/SymPy/Z3 sandbox]   BROCA DECODER (10× GDN, causal)
                                             result → encoder → write    prefix = y* slots; per-layer
                                                                         hippocampal read; COPY head
                                                                         points into retrieved spans
                                                                                  ▼
                                                                                 text
```

### 3.2 Why each piece exists

- **Gated DeltaNet everywhere** for sequence processing. It beat Mamba2 and DeltaNet on language modeling, commonsense and in-context retrieval [VERIFIED: https://arxiv.org/abs/2412.06464v3], and it is attention-free.
- **The core is a slot Mixer.** It uses MLP mixing across slots plus channel MLPs, in the style of TRM-MLP. TRM-MLP is the attention-free variant that reached 87.4% on Sudoku-Extreme [VERIFIED: https://arxiv.org/html/2510.04871v1]. TRM-MLP collapsed on large grids (Maze 0.0%, ARC 29.6%) because it has no content-addressable read. **Our fix [HYPOTHESIS]:** every core block reads the hippocampal fast-weight memory with a learned query. This brings back content addressing without softmax.
- **Two timescales (H/L)**, borrowed from HRM. The ARC Prize analysis found that *outer-loop refinement*, not the hierarchy, carries most of the gain [VERIFIED: https://arcprize.org/blog/hrm-analysis]. So we keep H/L cheap (the same block type), and we put our trust in deep supervision plus refinement.
- **Language only at the end.** The decoder is conditioned on 64 answer slots (a prefix the recurrence scans first) and on hippocampal reads. It acts as a renderer, not a thinker. A copy head lets it lift spans out of retrieved text, so surface fluency is partly borrowed from the datastore. This is in the spirit of CoG [VERIFIED: https://arxiv.org/abs/2307.06962].

### 3.3 Inference dataflow, step by step

1. **Perceive.** Tokenize the prompt (32k unigram vocabulary trained on a ko/en/code mix). Run the encoder forward and backward, with shared weights.
2. **Recall (cortex).** The encoder produces (a) a 256-d query that is binarized for IVF Hamming search and (b) a sparse term set for BM25. Merge with reciprocal-rank fusion to get the top 50. Rerank by encoding each chunk with the same encoder and taking the cosine against the workspace query. Keep the top 8.
3. **Bind (hippocampus).** Scan the prompt and the 8 chunks with the encoder. Write key/value pairs into M with delta-rule updates. Take 2–4 SGD steps on the memory MLP, using the reconstruction loss on these states (Titans-style surprise writing [VERIFIED: https://arxiv.org/abs/2501.00663]).
4. **Think (prefrontal loop).** Initialize y and z from learned slot embeddings. Run L×6 / H×1 cycles. After each supervision step, the HALT head decides whether to stop, as in TRM/HRM Q-halting. The ROUTE head can:
   - emit a **new query vector**, which loops back to step 2 for iterative retrieval with fresh chunks written into M;
   - request a **tool**: the decoder writes a short program from the current y, the sandbox runs it, and the output is encoded and written into M. This repeats for up to 8 tool calls.
5. **Deliberate.** For hard items, run K latent rollouts with different slot-noise seeds (K = 1 for easy, up to 16 for hard). The VERIFY head scores each final y and the best one is kept. Search in latent space is cheap, because nothing is decoded until the end.
6. **Express.** The decoder renders y\*. For code, it generates N candidates, runs the model's own tests, and repairs. For math, it renders a program-of-thought, executes it, and checks the answer against the verifier. For multiple choice, it generates the answer letter (generative scoring, per baseline §5).

---

## 4. Parameter budget, exact (d = 512, vocab 32,768)

Per-block formulas:
- **GDN block** = q, k, v, o and output-gate projections 5·d² + α/β gates 2·d·8 + short conv 3·d·4 + SwiGLU MLP 3·d·1408 + norms 2d = **3,488,768**.
- **Core Mixer block** = slot-mixing MLP 2·64·256 + SwiGLU 3·d·1408 + read projections (q, out) 2·d² + norms = **2,720,768**.

| # | Module | Composition | Trainable params |
|---|---|---|---|
| 1 | Token embedding (tied input/output softmax) | 32,768 × 512 | 16,777,216 |
| 2 | Sensory encoder (also the dense retriever and reranker) | 6 GDN blocks | 20,932,608 |
| 3 | Prefrontal core | L-module 2 + H-module 2 Mixer blocks | 10,883,072 |
| 4 | Hippocampus | delta-rule writer k/v 2·d² + gates 8,192 + memory MLP init 512→1024→512 (1,048,576) + memory q/k/v 3·d² | 2,367,488 |
| 5 | Broca decoder | 10 GDN blocks | 34,887,680 |
| 6 | Heads | verifier/value 512→1024→1 (525,312) + halt (1,026) + retrieval projection 512→256 (131,072) + copy gate d²+d (262,656) + slot inits 2·64·512 (65,536) + segment embeddings 16·512 (8,192) + router 512·8 (4,096) | 997,890 |
| | **Total** | | **86,845,954 (86.85M)** |
| | Headroom to the cap | Reserved for ablations (E3: +2 decoder blocks for fluency) | 13.15M |

Parameter-free: BM25, IVF centroids, the tokenizer, and the tools. The 32k vocabulary keeps embeddings at 19% of the budget. Gemma-3-270M spends 63% on embeddings [VERIFIED: https://developers.googleblog.com/en/introducing-gemma-3-270m/], which is a trap we avoid.

---

## 5. Scoreboard: estimated score, confidence and reasoning (all estimates are [ESTIMATE])

"P(beat)" is my subjective probability that v1, after the full recipe, clears the target plus its CI.

| Row | Target | CORTEX-v1 estimate | P(beat) | Why |
|---|---|---|---|---|
| ARC-AGI-1 (pass@2) | 5.0 | 25–40 | 0.90 | The latent recursion is TRM-like, and the hippocampal read fixes the TRM-MLP weakness [HYPOTHESIS]. Test-time training on demos. |
| NQ-open EM | 38.6 | 36–46 | 0.55 | FiD with 440M scored 48.2 [VERIFIED: https://arxiv.org/abs/2007.01282]. Our reader is about 5× smaller, with a BM25+binary retriever. |
| DROP F1 | 64.1 | 55–68 | 0.40 | Span copy plus executed arithmetic. No retrieval needed. |
| KoBEST COPA / HellaSwag (Ko, generative) | 57.88 / 40.00 | 60–70 / 40–55 | 0.6 / 0.5 | GPT-3.5's generative scores are unusually low. The BoolQ (82.34) and SentiNeg (91.69) rows are harder: P ≈ 0.25. |
| HAE-RAE | ~49.5–55.4 | 45–55 | 0.35 | Lexical subsets become dictionary/kowiki lookup. General knowledge and history depend on Korean datastore coverage. |
| KMMLU | 42.47 | 38–48 | 0.35 | Retrieve-then-reason over a Korean exam/textbook store. Random guessing is 25. |
| MATH (4-shot) | 34.1 | 25–38 | 0.35 | SymPy plus verifier search. Qwen2.5-0.5B-Instruct scores 34.4 [VERIFIED: https://qwenlm.github.io/blog/qwen2.5-llm/] with about 5× our parameters. |
| GSM8K | 78.1 (0-shot) / 57.1 (5-shot) | 55–70 | 0.10 / 0.55 | Program-of-thought plus execution plus verifier. TinyGSM-125M reached 63.1 [VERIFIED: https://arxiv.org/html/2312.09241], but on GPT-3.5 data that we ban. We use OpenMathInstruct-2 (CC-BY-4.0) instead. |
| BBH | 68.3 | 50–65 | 0.25 | Dyck, boolean, sorting and arithmetic tasks are executable. Disambiguation and causal-judgement tasks are fluency-bound. |
| HumanEval / HumanEval+ | 76.8 / 70.7 | 35–50 | < 0.05 | Generate–test–repair helps, but phi-1-small at 350M only reached 45 [VERIFIED: https://arxiv.org/abs/2306.11644]. EvalPlus's GPT-3.5 numbers are very strong. |
| MBPP / MBPP+ | 82.5 / 69.7 | 40–55 | < 0.05 | Same reasons. |
| MMLU | 71.4 | 45–58 | < 0.10 | Atlas-770M scored 56.3 [VERIFIED: https://arxiv.org/abs/2208.03299]. MassiveDS shows retrieval adds only +3.5 on MMLU [VERIFIED: https://arxiv.org/html/2407.12854]. |
| ARC-Challenge | 87.4 | 50–65 | < 0.05 | Textbook retrieval helps. The rest is reasoning over everyday physics. |
| TriviaQA | 85.8 | 62–75 | 0.10 | FiD reached 65 EM. GPT-3.5 is at its strongest here. |
| HellaSwag | 85.5 | 45–60 | < 0.03 | Pure continuation plausibility. Neither retrieval nor tools apply. |
| WinoGrande | 81.6 | 55–65 | < 0.03 | Everything at or below 270M scores 51–53 [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M]. |
| MT-Bench (local judge) | 8.39 | 3.5–5.0 | < 0.02 | MobileLLM-125M scored 2.33 [VERIFIED: https://arxiv.org/html/2402.14905]. |
| AlpacaEval 2 LC | 22.35 | 3–10 | < 0.05 | Breadth, style and length are all bound by fluency. |

### Honest verdict

**CORTEX-v1 will not beat GPT-3.5 on every row.** P(all Tier A+B rows) ≈ 0 for v1. It should win the abstraction and lookup rows (ARC-AGI, NQ, likely DROP and the weak KoBEST rows). It may win Korean knowledge and MATH. It loses commonsense, broad exams, code and chat.

That is the "F quarter" from the baseline digest: about 25% of the score mass, where no published system under 1B comes close. The smallest model to match GPT-3.5 in the Arena is about 2.6B [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard].

**Why the wall exists.**
- Fluency and implicit world plausibility live in high-dimensional statistics that need either parameters or a very large number of training tokens.
- These rows have no exact verifier, so test-time search has nothing to optimize against.
- Retrieval keys don't exist for "which ending is plausible".

**How CORTEX attacks it (each attack is a [HYPOTHESIS] with a kill test).**

1. **Plausibility as energy, not generation.** HellaSwag, WinoGrande and COPA are scored by generating a letter. Internally, the latent core runs each option as a hypothesis through the workspace and the VERIFY head ranks them. That head is trained contrastively on large numbers of synthetic minimal-pair negatives. Open teachers generate the pairs, and they are decontaminated against WikiHow and ActivityNet mirrors. This is discrimination, which is much cheaper in parameters than generation. Evidence that data moves this row: phi-1.5's WinoGrande went from 60.4 to 73.4 from data alone at 1.3B [VERIFIED: https://arxiv.org/html/2309.05463].
2. **Event-memory shard.** A cortex shard of short situation→outcome narratives, so plausibility gets analogical neighbours. This is a kNN prior, not a lookup of the answer.
3. **Chat as retrieve–plan–render.**
   - A dialog-exemplar shard holds about 1M high-quality open-licensed responses.
   - The core plans in slots: intent, outline, constraints.
   - The decoder renders, copying structure from exemplars.
   - A local judge on T4 then runs a critique→revise loop at training time only; the critiques are distilled into the verifier. **I expect this to get MT-Bench to about 5, not 8.4.** Winning chat probably needs the 1B+ tier (§9). I will not claim otherwise.
4. **Korean fluency.** The Korean share of pretraining is raised to 25%. Morphology-aware tokenization brings Korean tokens/char close to English. Expect a gap similar to the English chat gap.

---

## 6. Training recipe

Token counts are [ESTIMATE]. Datasets come from the data digest.

| Stage | Objective | Data | Volume |
|---|---|---|---|
| S0 Build | Tokenizer; datastore v0; decontamination | FineWeb-Edu-dedup sample, en/ko Wikipedia, FineWeb-2 Korean, FineMath-4+, Python-Edu, Cosmopedia v2 | Store v0: 20B tokens (156M chunks × 32 B binary ≈ 5 GB, plus BM25 ≈ 3–6 GB) |
| S1 Language + reading | Encoder: span denoising. Decoder: next-token with retrieved neighbours written into M (RETRO-style, but without attention). LLM-JEPA latent-prediction loss between views [VERIFIED effect: https://arxiv.org/abs/2509.14252]. Retriever: in-batch contrastive, refreshed once per epoch. | 40% FineWeb-Edu, 25% Korean (FineWeb-2-ko, kowiki), 15% Python-Edu, 10% FineMath, 10% Cosmopedia | 40B tokens |
| S2 Latent reasoning | Deep supervision on y at every supervision step. Gradient flows only through the last cycle (TRM one-step gradient). Q-learning for HALT. | Reasoning Gym (100+ generators) [VERIFIED: https://arxiv.org/pdf/2505.24760], re-ARC train split [VERIFIED: https://arxiv.org/pdf/2404.07353], SymPy/Z3 generators, code-execution traces | 30M problems, curriculum held at a 50–70% solve rate |
| S3 Tool + retrieval SFT | Routing and query supervision; program-of-thought; copy head | KILT provenance triples [VERIFIED: https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/]; OpenMathInstruct-2 rewritten to execute in tool form, keeping only items whose execution matches the answer [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2]; SmolTalk; Tulu-3 minus NC subsets; Magpie-style prompts regenerated with open teachers on T4; Korean translations of templates | ~4B tokens |
| S4 RLVR | GRPO with executor, test and SymPy rewards. The verifier head is trained on our own rollouts: TinyGSM found the verifier matters more than generator size [VERIFIED: https://arxiv.org/html/2312.09241] | Math, code (self-written tests plus a held-out training test pool), executable BBH-style tasks, KILT exact-match answers | ~2M episodes |
| S5 Self-play | Proposer → solver → verifier. Keep only problems checked by execution. Expert iteration. | Self-generated | Grows from 10% to 50% of the mix |
| S6 Preference / chat | Pairwise preferences from a local open judge (allowed at training time, logged), DPO-style; critique distillation into VERIFY | Our own samples on open prompts | ~200k pairs |

---

## 7. Compute plan (free quotas)

Quota figures: TPU v5e-8, about 20 h/week, 9 h sessions; T4×2, about 30 h/week. Both are reported by third parties and must be checked in the account.
- [VERIFIED: https://www.kaggle.com/discussions/product-announcements/607202]
- [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits]
- v5e chip: 197 TFLOPS bf16 [VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e]
- T4: 65 TFLOPS fp16 [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf]

Effective throughput [ESTIMATE]:
- TPU at 30% MFU: 1,576e12 × 0.3 = 4.7e14 FLOP/s, so 20 h ≈ **3.4e19 FLOP/week**.
- T4×2 at 20%: 2.6e13 FLOP/s, so 30 h ≈ **2.8e18 FLOP/week**.

FLOPs per stage:
- **S1 (6ND):** N = 86.85M, D = 40B → 6 × 8.685e7 × 4e10 = **2.08e19 FLOP**, about 12.3 TPU-hours, or about 0.6 TPU-week.
- **S2 (core, per supervision step):**
  - One L-application = 2 × 5.44M × 128 slots ≈ 1.39e9 FLOP forward.
  - One cycle = 7 applications ≈ 9.7e9. Three cycles ≈ 2.9e10 forward.
  - The one-step backward adds about 2 × 9.7e9.
  - Total ≈ 4.9e10 per supervision step, plus the encoder over a 512-token problem (2 × 2 × 21M × 512 ≈ 4.3e10 forward + backward) ≈ **~1.1e11 per step**.
  - 30M problems × about 6 steps on average → **2.0e19 FLOP**, about 12 TPU-hours.
- **S3:** 6 × 8.685e7 × 4e9 = 2.1e18, about 1.2 TPU-hours.
- **S4 RLVR:** 2M episodes × 8 samples × about 2e12 FLOP per rollout (§8) ≈ 3.2e19 FLOP forward, plus training. This takes about 1.5 TPU-weeks, or it is spread across T4 sessions.
- **Index build:** 20B tokens × 2 × 21M ≈ 8.4e17, about 30 minutes on TPU. Re-index after each encoder refresh.
- **Teacher data:** 7B AWQ on T4 gives about 65–170M tokens/week [ESTIMATE, compute digest]. Over 8 weeks that is about 0.5–1.4B tokens, which is enough for S3 and S6.

**Total: about 1.1e20 FLOP, about 4 TPU-weeks plus 8 T4-weeks running in parallel. That is roughly a 10–12 week calendar, including failed runs.** As the digests concluded, compute is not the bottleneck; data quality and evaluation are.

### Engineering
- **TPU:** JAX is the primary codebase. The chunkwise delta rule is written as WY-form matmuls in `lax.scan` over chunks of 64, with static shapes. Use a fixed maximum number of supervision steps and mask after halting, so the XLA graph stays static.
- **T4** (rollouts, evaluation, vLLM teachers):
  - PyTorch fp16 matmuls, with **fp32 state and accumulators for the delta-rule recurrence and the gates**, plus loss scaling.
  - Triton is pinned to ≤3.2, the last line that supports sm_75 [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.2.0/README.md].
  - A pure-PyTorch chunked fallback is kept and checked for parity against JAX in CI on the 3060.
  - DDP across the 2 GPUs over PCIe; no tensor parallelism.
- **3060 laptop (bf16):** kernel development, 10–30M ablations, offline demo.
- **Persistence:** checkpoint every 45 minutes to the HF Hub (100 GB private [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits]). The datastore lives in Kaggle Datasets and is mounted read-only.
- **Day 1:** apply to TRC [VERIFIED: https://sites.research.google/trc/faq/].

---

## 8. Inference cost per query [ESTIMATE]

| Phase | FLOPs (hard query) | Notes |
|---|---|---|
| Encode prompt + 8 chunks (1.5k tokens, forward + reverse) | 1.5e3 × 2 × 2 × 21M ≈ 1.3e11 | Repeated for up to 3 re-retrievals: ≈ 4e11 |
| Retrieval | CPU: IVF-binary 10–50 ms; BM25 20–100 ms | ~0 GPU |
| Latent loop, K = 16 rollouts × 16 steps × 2.9e10 | 7.4e12 | Easy queries: K = 1, ~4 steps ≈ 1.2e11 |
| Decode 400 tokens × 2 × 52M | 4.2e10 | Code: × 10 candidates ≈ 4e11 |
| Tools | 0.1–2 s CPU per call, ≤ 8 calls | Dominates latency on math and code |
| **Total** | **~1e11 (easy) to ~8e12 (hard)** | Recurrent decoding is O(1) memory per token |

Latency on a single T4 at about 10 TFLOPS effective: about 0.3 s for easy queries, about 2–10 s for hard ones, most of it tool and serial-recurrence time. A full Tier A run (about 70k items × about 2e12 on average ≈ 1.4e17 FLOP) takes about 2–4 h on T4×2 plus CPU tool time.

All test-time compute is reported per row, following Snell et al.'s FLOPs-matched framing [VERIFIED: https://arxiv.org/abs/2408.03314].

---

## 9. Scaling story: what grows, and how

The core bet [HYPOTHESIS]: capability ≈ f(procedure parameters, datastore tokens, recursion depth × verifier quality, hippocampal capacity, tool coverage, verified self-generated data). Every axis except the first grows without retraining the core.

| Axis | 0.1B (v1) | 1B (v2) | 10B (v3) | Evidence |
|---|---|---|---|---|
| Procedure parameters | d = 512 | d = 1536, decoder 16 blocks, core 4 → 8 blocks | d = 4096 | RWKV-7 improves monotonically from 0.19B to 2.9B [VERIFIED: https://arxiv.org/abs/2503.14456] |
| Fluency | Weak (MT-Bench ~5) | Gemma-2-2.6B-class Arena parity is plausible near 1–3B [ESTIMATE] | Strong | Arena table above |
| Datastore | 20B → 100B tokens | 1T (MassiveDS saw no saturation at 1.4T [VERIFIED: https://arxiv.org/html/2407.12854]) | Multi-T + PubMed / UniProt / ChEMBL shards | RETRO: 25× parameter savings [VERIFIED: https://arxiv.org/abs/2112.04426v1] |
| Sparse parametric memory | none | Product-key memory layers (count against budget) | 10B+ memory parameters | Memory layers beat dense models with >2× the compute [VERIFIED: https://arxiv.org/abs/2412.09764] |
| Recursion depth | ≤ 16 × 21 block-applications | ≤ 64 steps | Unbounded, budgeted per query | Huginn: "50B-equivalent" from depth [VERIFIED: https://arxiv.org/abs/2502.05171] |
| Hippocampus | 8 × 64 × 64 fast weights | Larger heads; multi-episode consolidation into the datastore ("sleep") | Lifelong episodic store | Titans: context beyond 2M [VERIFIED: https://arxiv.org/abs/2501.00663] |
| Tools | Python, SymPy, Z3 | Lean, simulators | Docking, molecular dynamics, retrosynthesis, lab protocols (with human gating) | AlphaGeometry / AlphaProof [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/] |
| Self-generated data | 10–50% of the mix | Majority | Science-environment rollouts | DeepSeek-R1-Zero RL [VERIFIED: https://arxiv.org/html/2501.12948] |

**Compute wall.** 1B at Chinchilla scale is about 1.2e20 FLOP, about 3 TPU-weeks, so it still fits the free tier. 10B is about 1.2e22, about 7 years of free quota [ESTIMATE, compute digest]. 10B therefore needs TRC or grants, earned by publishing v1 and v2 openly.

**North-star logic.** Scientific discovery is a verifier-rich environment (simulations, proofs, assays). In that environment, a latent recursive reasoner plus a verifier plus tools improves with compute rather than with human data. That is why CORTEX's architecture is a bet aimed at the north star, not only at benchmarks [HYPOTHESIS].

**Biosafety hard stop.**
- The datastore and tool registry exclude pathogen-enhancement and toxin-synthesis content.
- The router has a refusal class for hazardous chemistry and biology.
- Any wet-lab-actionable output requires human expert review.
- Bio shards and weights go through dual-use review before release.
- Latent thinking is opaque, so a "thought probe" decoder that renders y and z at each step is mandatory for safety audits on bio tasks.

---

## 10. First three experiments: go/kill criteria

**E1: Does latent recursion transfer to language reasoning?** (3060 plus about 1 TPU-week)
- Setup: a 20M CORTEX core at r ∈ {1, 4, 16} supervision steps, against a 50M plain GDN baseline that uses token chain-of-thought. Both get equal training FLOPs on Reasoning Gym held-out generators, plus GSM8K run as program-of-thought on the training split and validated on the test split.
- **Go:** the 20M model at r = 16 matches or beats the 50M baseline, and accuracy rises monotonically with r.
- **Kill:** gain from r = 1 to 16 under 3 points, or the model fails to beat the baseline. Then drop latent recursion, fall back to token chain-of-thought, and spend the core's parameters on the decoder.

**E2: Can the hippocampus plus cortex reader do exact recall without attention?** (about 1 TPU-week)
- Setup: a 50M model plus a 4B-token Wikipedia store. Measure NQ-open EM and recall@8, and run a synthetic key→value recall probe at 1–8k tokens.
- **Go:** NQ EM ≥ 33, recall@8 ≥ 70%, probe accuracy ≥ 90% at 4k tokens.
- **Kill:** NQ < 25 or probe < 60%. That means the fixed-state recall wall from the archs digest (§5.3) holds, and the read path must be redesigned (larger state, multiple fast-weight banks, re-reading via the encoder).

**E3: Is there a fluency floor?** (full 86.85M through S3; about 3 TPU-weeks)
- Setup: compare against the stored GPT-3.5 MT-Bench answers [VERIFIED exists: https://huggingface.co/spaces/lmsys/mt-bench] and the stored AlpacaEval outputs [VERIFIED exists: https://github.com/tatsu-lab/alpaca_eval]. Use a local open judge with both answer orders, calibrated on the stored outputs of at least 5 reference models.
- **Go:** pairwise win rate ≥ 15% against GPT-3.5 and calibrated MT-Bench ≥ 4.0. Then spend the 13M headroom on 2 more decoder blocks and the exemplar shard.
- **Kill:** win rate < 5%. Publicly declare chat as a v2 (1B) objective, keep v1 as a reasoning and knowledge system, and do not quietly redefine "win".

---

## 11. Top risks

1. **Latent recursion may not transfer from grids to language** [HYPOTHESIS]. All TRM/HRM evidence is on fixed-size puzzles, and TRM-MLP collapsed to 0% on Maze. Covered by E1.
2. **Fixed-state recall.** Recurrent models lose at exact in-context recall, and the hippocampus may not be enough. Covered by E2.
3. **The fluency/commonsense quarter is probably unwinnable at 0.1B.** HellaSwag, WinoGrande and chat are expected losses for v1. This is the main reason "all domains at 0.1B" is not credible today.
4. **Datastore contamination fakes wins.** Retrieval copies verbatim (RETRO exploits leakage more [VERIFIED: https://arxiv.org/abs/2112.04426]). Mandatory scrub plus audit logs, with a private post-cutoff set as the headline.
5. **Local-judge validity.** Its agreement with the GPT-4 judge is unmeasured. Calibrate it before reporting any chat number.
6. **Quota volatility.** v5e-8 is reported as oversubscribed at times [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab]. A T4-only fallback costs about 8× in throughput, which is still above Chinchilla for 100M.
7. **Toolchain.** Turing support was dropped in Triton 3.3+, and the JAX and PyTorch paths may drift. Mitigation: a parity CI.
8. **Korean data licensing.** 35.3% of Korean datasets on HF have unknown licenses [VERIFIED: https://arxiv.org/pdf/2507.04329], and AI Hub data is not redistributable.
9. **Opacity of latent thought.** Latent reasoning is harder to audit, so the thought probe is mandatory for the north-star bio work.
10. **Benchmark gravity.** The temptation to tune toward the GPT-3.5 rows. Countermeasure: Tier C held-out sets and ARC-AGI-2 are reported every release.

**One-line summary:** CORTEX moves facts to the datastore, depth to time, working memory to state and exact computation to tools. It should beat GPT-3.5 on abstraction and lookup, is roughly even on Korean knowledge and MATH, and will lose on commonsense and chat at 0.1B. It is built so that each added axis (parameters, store, recursion, tools, self-play) feeds the north star rather than one benchmark.
