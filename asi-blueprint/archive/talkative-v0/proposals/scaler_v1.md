# OUROBOROS-0: A Pragmatic, Scaling-First Hybrid (proposal `scaler_v1`)

Chief-architect proposal. Philosophy: **use proven linear-time recurrent parts, add recurrent depth for test-time compute and sparse memory, and put knowledge in a datastore and tools. Every axis must have a measurable scaling law from day one.** Date: 2026-10-19.
Labels: [VERIFIED: url] = taken from the research digest's primary sources; [ESTIMATE] = my arithmetic or judgment; [HYPOTHESIS] = untested design bet.

---

## 1. Thesis

A dense LLM spends parameters on three things: **(a) memorized facts**, **(b) depth per token**, and **(c) procedures**, meaning the reusable skills of reading, decomposing, computing and writing. Facts are the most expensive. Knowledge capacity is about 2 bits per parameter [VERIFIED: https://arxiv.org/html/2404.05405], and a 135M model trained on 2T tokens still gets only 4.1% on TriviaQA [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct]. Depth is the second cost, because a fixed-depth network is a fixed-size circuit.

The bet is to **steal (a) with a datastore**, which costs 0 parameters and is worth about 10–25x parameters on knowledge tasks [VERIFIED: https://arxiv.org/abs/2112.04426]. We **steal (b) with weight-tied recurrent depth**: a 7M recursive network beats frontier LLMs on ARC-AGI-1 [VERIFIED: https://arxiv.org/abs/2510.04871]. We **steal the "being right" part of (c) with exact verifiers and tools**: PAL beats PaLM-540B CoT by 15 points [VERIFIED: https://arxiv.org/abs/2211.10435], and a 0.5B model plus search beats GPT-4o on MATH-500 [VERIFIED: https://arxiv.org/abs/2502.06703]. That leaves the 100M weights one job: **a compact library of procedures**, plus the small amount of fluency needed to drive them.

This splits the system into five independent knobs: core params, sparse-memory params, datastore tokens, recurrence iterations r, and tool coverage. Each one must show its own power law before we scale it.

**Honesty up front.** With today's evidence, this design is likely to **lose** to GPT-3.5 on HellaSwag, WinoGrande, MMLU and open-ended chat at the first milestone (§5). The plan treats those rows as named, scheduled wall-breaking campaigns rather than hiding them.

---

## 2. Fair-play rules

1. **Parameters** means every trainable weight that runs at inference: embeddings, backbone, recurrent core, sparse product-key memory, the initial weights of the test-time memory, retrieval projections, critic/verifier, halting and copy heads. Weights used only in training (the JEPA predictor) are **counted anyway**, to be conservative. Hard cap: **< 100M**. This design uses **86.96M**.
2. **Not parameters (disclosed data):** tokenizer, BM25 statistics, FAISS IVF centroids and PQ/binary codebooks (learned, so their KB/MB sizes are reported), and the datastore text and its codes. Each datastore snapshot is dated and hashed.
3. **Allowed at inference:** the frozen offline datastore, a Python sandbox, SymPy, Z3, Lean 4, and a unit-test runner the model writes tests for. All run on CPU with 0 parameters.
4. **Banned at inference:** any neural network over 100M parameters, any LLM API, live web search, routing or prompts keyed on dataset identity, and running the hidden benchmark tests during generation.
5. **Training time:** open-licensed teacher models may generate data. Each teacher and its size are logged, and a "no-teacher" ablation track is reported. **No GPT-3.5/4 outputs anywhere**: that breaks OpenAI ToS [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/] and contaminates the head-to-head. This excludes TinyGSM, MetaMathQA, UltraChat and TinyStories from weight training.
6. **Architecture rule:** no softmax (or Hopfield-equivalent) attention over the token sequence, at any layer. **Declared gray zone:** the product-key memory uses top-k (k=32) normalized weights over *parameter slots*, not tokens. A ReLU-normalized variant is kept as a fully compliant fallback, and both are reported.
7. **Decontamination:** 13-gram plus MinHash plus embedding-cosine (>0.9) scrubs of the training data **and the datastore** against every eval set and its mirrors. Every eval answer logs its top-k retrieved passages, with the % that contain the gold answer string. A private post-cutoff test set is the headline sanity check.
8. **Scoring:** generative answers everywhere, shot counts matched to the target row, one checkpoint and one config for all rows, 3 seeds for anything sampled. Per-query FLOPs and wall-clock are published.

---

## 3. Architecture

### 3.1 Component diagram

```
                       ┌──────────────── OFFLINE (0 params) ─────────────────┐
                       │ DATASTORE v0: 30B tok (en/ko wiki, FineWeb-Edu,     │
                       │ FineWeb-2-ko, Cosmopedia, Python-Edu, OpenWebMath)  │
                       │  ├─ BM25 (+ Korean char-n-gram BM25)                │
                       │  ├─ binary 256-bit codes (IVF) + int8 rescoring     │
                       │  └─ phrase index (CoG-style spans)                  │
                       │ TOOLS: Python · SymPy · Z3 · Lean4 · test-runner    │
                       └───────────▲───────────────────────▲────────────────┘
                                   │ SEARCH(q)/READ        │ PY/SOLVE/CHECK
 tokens ─► [Factorized Emb 6.4M] ─►┌──────────────┐        │
                                   │ PRELUDE      │ 5× Gated-DeltaNet (L1–2 frozen
                                   │ 21.7M        │ after Stage 1 = KEY ENCODER)
                                   └──────┬───────┘
          retrieved chunks ─► prelude+core(r=1) ─► per-layer delta states S_i
                                          │            │ STATE MERGE: S0 = Σ g_i·S_i
                                          ▼            ▼
                        ┌────────────────────────────────────────┐
                        │ RECURRENT CORE (weight-tied) 21.7M     │◄─ input injection
                        │ 5× GDN, iterate r = 1..32               │   adapter 0.66M
                        │  + PKM sparse memory 10.06M (mid-core)  │
                        │  + Test-time memory MLP 2.32M           │
                        │  + halting head                         │
                        └───────┬───────────────────┬────────────┘
                                ▼                   ▼
                     ┌─────────────────┐  ┌───────────────────────┐
                     │ CODA 3× GDN 13M │  │ CRITIC branch 2×GDN    │
                     │ → tied LM head  │  │ + value head 9.34M     │
                     │ + phrase-copy   │  │ (verifier / PRM /      │
                     │   head 0.15M    │  │  reward / difficulty)  │
                     └────────┬────────┘  └──────────┬────────────┘
                              ▼                      ▼
                     ┌──────────────────────────────────────────┐
                     │ CONTROLLER (code, 0 params): compute-     │
                     │ optimal budget, sample N, verify, vote    │
                     └──────────────────────────────────────────┘
```

**Backbone choice:** Gated DeltaNet (GDN). It beats Mamba-2 and DeltaNet on language modeling, commonsense and in-context retrieval [VERIFIED: https://arxiv.org/abs/2412.06464v3]. RWKV-7 is the pre-registered challenger in Experiment 1 [VERIFIED: https://arxiv.org/abs/2503.14456]. Width d=576 follows MobileLLM's deep-thin finding for this size class [VERIFIED: https://arxiv.org/html/2402.14905]. Unrolled at r=8 the network is 48 layers deep.

**Original component: retrieval as state priming** [HYPOTHESIS]. A delta-rule state is a sum of key-value outer products. Each retrieved chunk is therefore run once through prelude + core (r=1), with all chunks batched in parallel. Its per-layer states are merged into the query's initial core state with learned per-head gates: S₀ = Σ g_i·S_i. This gives RETRO-like fusion with no cross-attention and no K×chunk-length serial scan. With gated decay the merge is only approximate, which is why Experiment 2 tests it against plain text-appending (the guaranteed-compliant fallback).

### 3.2 Inference dataflow

1. **Encode** the prompt through the embedding and prelude.
2. **Plan.** The core emits tool tokens: `<search>q</search>`, `<py>`, `<sympy>`, `<z3>`, `<lean>`, `<check>`, `<answer>`.
3. **Retrieve.** BM25 and binary-dense IVF lists are fused by reciprocal rank fusion. The binary codes come from frozen prelude layers 1–2 plus a 256-d projection. Top-K=8 chunks of 128 tokens, rescored with int8.
4. **Prime state.** The K chunks run through prelude+core in parallel, then merge into S₀. Long contexts (documents, ARC demos) are also written into the test-time memory MLP with a few inner gradient steps (Titans-style; that paper's attention core is not used) [VERIFIED mechanism: https://arxiv.org/abs/2501.00663].
5. **Think.** The core iterates r times per token. The halting head picks r adaptively (cap 32) [VERIFIED mechanism: https://arxiv.org/abs/2502.05171].
6. **Act.** Tool calls run, and the outputs are spliced back as tokens.
7. **Write.** The coda produces token logits, mixed with the phrase-copy distribution so the model can copy fluent spans verbatim from retrieved text [VERIFIED mechanism: https://arxiv.org/abs/2307.06962].
8. **Verify and select.** The controller samples N candidates where the critic's uncertainty is high (Snell compute-optimal allocation [VERIFIED: https://arxiv.org/abs/2408.03314]). Exact checks come first (execution, SymPy equality, self-written tests), the critic score breaks ties, and the answer is a majority vote or best-of-N.

---

## 4. Exact parameter budget

GDN layer at d=576 (9 heads × 64):
- q, k, v, o projections: 4d² = 1,327,104
- output gate: d² = 331,776
- α/β gates: 10,368
- short conv (k=4) on q, k, v: 6,912
- SwiGLU MLP, hidden 1536: 3·576·1536 = 2,654,208
- norms ≈ 1,728
- **Total ≈ 4.34M per layer** [ESTIMATE, arithmetic].

| # | Module | Shape / formula | Params |
|---|---|---|---|
| 1 | Factorized embedding (tied output) | 32,768×192 + 192×576 | 6.40M |
| 2 | Prelude: 5 GDN layers (L1–2 frozen after Stage 1) | 5 × 4.34M | 21.70M |
| 3 | Recurrent core: 5 GDN layers, weight-tied, iterated r times | 5 × 4.34M | 21.70M |
| 4 | Coda: 3 GDN layers | 3 × 4.34M | 13.02M |
| 5 | Input-injection adapter | Linear(2d→d) | 0.66M |
| 6 | Retrieval query/key projection | 576×256 | 0.15M |
| 7 | State-merge gates | 5 layers × 576×9 | 0.03M |
| 8 | Test-time memory (MLP init + q/k/v projections) | 2·576·1152 + 3·576² | 2.32M |
| 9 | Product-key memory | 2×512 sub-keys×288 + 16,384 values×576 + query proj 576² | 10.06M |
| 10 | Critic branch: 2 GDN + value MLP (576→1152→1) | 8.68M + 0.66M | 9.34M |
| 11 | Halting head | 576→1 | 0.0006M |
| 12 | Phrase-copy head | 576×256 | 0.15M |
| 13 | JEPA predictor (training only, counted anyway) | 576→1152→576 | 1.33M |
| 14 | Final norms and misc biases | — | 0.10M |
| | **TOTAL** | | **86.96M** |
| | Headroom under the cap | reserved; allocated only by Experiment 1/2 results | 13.04M |

Active parameters per token (the PKM touches only k=32 values): about 59M at r=1, 124M at r=4 and 384M at r=16 [ESTIMATE].

---

## 5. Beating the scoreboard, row by row

Targets are the max reported GPT-3.5 scores from the baseline digest. Estimates are for milestone M1, 12 months in. "P(win)" is my probability of beating target + CI. **These are judgment calls, not measurements** [ESTIMATE throughout].

| Row (target) | Mechanism | Est. M1 score | P(win) | Why |
|---|---|---|---|---|
| NQ-open (38.6) | retrieve + reader + phrase copy | 40–50 | 0.65 | A 220M FiD reached 48.2 [VERIFIED: https://arxiv.org/abs/2007.01282]; the risk is our smaller reader |
| TriviaQA (85.8) | same | 68–80 | 0.15 | FiD-base gets 65 EM; Llama-2-7B plus a 1.4T store reaches 77.0 [VERIFIED: https://arxiv.org/html/2407.12854]. This is a hidden wall |
| ARC-AGI-1 (≈5) | recurrent core + test-time training on demos | 15–35 | 0.85 | TRM-class recursion; our core is attention-free, so expect to be closer to TRM-MLP's 29.6 than to 44.6 [VERIFIED: https://arxiv.org/abs/2510.04871] |
| GSM8K (78.1) | program-of-thought + Python + critic + 64-sample vote | 62–80 | 0.35 | TinyGSM 125M scores 63.1 alone and 76.1 with a 1.3B verifier [VERIFIED: https://arxiv.org/html/2312.09241]; our verifier is 9M, not 1.3B |
| MATH (34.1) | SymPy tool + search | 25–40 | 0.35 | Qwen2.5-0.5B-Instruct scores 34.4 [VERIFIED: https://qwenlm.github.io/blog/qwen2.5-llm/]; the tool takes over the algebra |
| BBH (68.3) | executable subtasks go to Python; the rest to the recurrent core | 55–70 | 0.35 | About 30% of BBH is exactly executable [ESTIMATE] |
| DROP F1 (64.1) | span pointer + calculator | 55–70 | 0.40 | No retrieval needed; arithmetic is offloaded |
| KMMLU (42.47) | Korean store + reader | 38–48 | 0.40 | The target is low (chance is 25) |
| HAE-RAE (~49–55) | Korean dictionary/wiki store | 45–55 | 0.35 | Vocabulary and loanword subsets are lookups; history is hard |
| KoBEST (79.3 En-prompt avg) | reader + negation curriculum | 62–75 | 0.20 | BoolQ and SentiNeg are fine; COPA and HellaSwag inherit the commonsense wall |
| MMLU (71.4) | exam-explanation store + option elimination | 50–60 | 0.08 | Retrieval adds only +3–4 points on reasoning-heavy knowledge [VERIFIED: https://arxiv.org/html/2407.12854]; Atlas-770M reached 56.3 [VERIFIED: https://arxiv.org/abs/2208.03299] |
| ARC-Challenge (87.4) | textbook store + critic | 55–70 | 0.07 | Qwen2.5-0.5B gets 35.6; this is a big gap |
| HumanEval+ / HE (70.7 / 76.8) | generate, write self-tests, execute, repair (8 rounds) | 40–58 | 0.08 | phi-1-small (350M) reached 45 [VERIFIED: https://arxiv.org/abs/2306.11644]; the code prior is capacity-bound |
| MBPP+ / MBPP (69.7 / 82.5) | same + API-doc retrieval | 50–65 | 0.08 | Same reasoning |
| HellaSwag (85.5) | per-option critic + retrieved analog narratives | 45–62 | 0.02 | Nothing ≤0.5B exceeds 52.1; F90 row |
| WinoGrande (81.6) | multi-hypothesis critic + synthetic commonsense | 55–68 | 0.02 | ≤270M models are at chance; synthetic data moved phi-1.5 from 60.4 to 73.4, but at 1.3B [VERIFIED: https://arxiv.org/html/2309.05463] |
| MT-Bench pairwise vs stored GPT-3.5 answers (>50%) | retrieve exemplar answers, plan, realize, critic best-of-8 | 10–25% wins | 0.03 | Every ~130M model scores 2.0–2.5 on MT-Bench vs 8.39 [VERIFIED: https://arxiv.org/html/2402.14905] |
| AlpacaEval 2 LC (22.35) | same | 5–15 | 0.03 | Same reasoning |
| Creative writing / Korean chat fluency | phrase copy + style store | weak | <0.02 | No sub-1B route is demonstrated; the smallest Arena model to beat GPT-3.5 is about 2.6B [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard] |

**Plain verdict.** P(sweeping every Tier A+B row at M1) is roughly 0 [ESTIMATE]. Realistic M1 wins are NQ, ARC-AGI, some of GSM8K/MATH/BBH/DROP/KMMLU, and possibly HAE-RAE. The losses are commonsense, MMLU/ARC-C, code and chat. Each one has a funded campaign:

- **Commonsense campaign** [HYPOTHESIS]. Commonsense is cast as a **verification problem**. Each option becomes a hypothesis, and the critic is trained contrastively on tens of millions of open-teacher-generated situation/consequence pairs with hard negatives; phi-1.5 is the evidence that this kind of data works. Retrieved analog narratives come from a store decontaminated against WikiHow and ActivityNet mirrors. The 13M headroom goes to the critic if Experiment 3b shows a slope.
- **Chat campaign** [HYPOTHESIS]. The generator's job shrinks to composing: retrieve 3–5 high-quality permissive human answers to similar instructions, write an outline, realize it with phrase copying, and pick from 8 drafts with a critic. The critic is **distilled from an open local judge at training time** (allowed: the judge is not run at inference). Its preferences feed DPO/GRPO.
- **Code campaign.** Test-time compute is unbounded where a checker exists: self-written tests, execution and repair, plus pass-rate-weighted voting. The binding limit is base pass@k, so code-trace data gets 15% of Stage 2.
- **The honest fallback.** If M1 leaves rows lost, the milestone moves: the same architecture, still under 100M, with a larger datastore and more test-time compute. Or we report a loss. **We do not tune per benchmark.**

---

## 6. Training recipe

Training FLOPs per token = 6·N_active at mean r̄=4, using truncated backprop through the last 4 iterations. N_active is 128M including amortized critic/JEPA, and stop-grad neighbor encoding adds about 15%. That gives **≈ 9e8 FLOPs/token** [ESTIMATE].

| Stage | Tokens | Data | Objectives |
|---|---|---|---|
| 0. Infra (CPU/T4) | — | 32k BPE over en/ko/code (complete Korean syllable coverage, byte fallback); datastore v0, 30B tokens; decontamination pipeline | — |
| 1. Fluency + retrieval pretraining | 150B | FineWeb-Edu-dedup 45%, FineWeb-2-ko + kowiki 15%, Cosmopedia 10%, Python-Edu/Stack-permissive 15%, FineMath-4+/OpenWebMath 10%, KILT-style QA 5% [VERIFIED sources: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus, https://arxiv.org/abs/2506.20920v1]. 50% of batches retrieval-on. The store is held-out text, so the model **learns to rely on it** | next-token loss, r ~ log-normal(mean 4), + LLM-JEPA embedding loss [VERIFIED effect: https://arxiv.org/abs/2509.14252]; freeze L1–2 and the embedding at the end of the stage, then build the dense index |
| 2. Procedure mid-training | 60B (≈12B unique × 5 epochs) | Reasoning Gym 100+ generators [VERIFIED: https://arxiv.org/pdf/2505.24760], re-ARC on the train split only [VERIFIED: https://arxiv.org/pdf/2404.07353], SymPy/Z3 generators, program→exact-output traces, tool-call traces, OpenMathInstruct-2 rewritten to program-of-thought (CC-BY-4.0) [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2], Korean templated procedures. Only **verified** samples are kept | next-token loss on calls/results (tool outputs masked) + deep supervision on every iteration + critic loss (correct/incorrect labels from executors) |
| 3. SFT | 3B | SmolTalk (Apache-2.0) [VERIFIED: https://www.huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/blob/main/README.md], Tulu-3 minus NC subsets [VERIFIED: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/main/README.md], Magpie-style data regenerated from an open teacher, Korean instructions regenerated (never ChatGPT-made) | SFT; critic learns from preferences distilled from the open judge |
| 4. RLVR | ~0.5B rollout tokens | math, code, logic, retrieval-QA with exact-match rewards | GRPO [VERIFIED method: https://arxiv.org/html/2501.12948]; rewards for correctness, for using fewer iterations r, and for calling a tool where the answer is checkable |
| 5. Self-play flywheel (continuous) | grows 10%→50% of the mix | the model plus a problem-proposer prompt generates frontier problems (50–70% solve rate); an exact checker gates every accepted solution | expert iteration (STaR-style) [VERIFIED: https://arxiv.org/html/2203.14465] |

Overtraining ratio ≈ 2,400 tokens/param. SmolLM2-135M used about 15,000 [VERIFIED: https://arxiv.org/abs/2502.02737], so we deliberately trade raw tokens for data density.

---

## 7. Compute plan

Quotas (third-party sources, verify in-account on day 1): **TPU v5e-8 ≈ 20 h/week, 9 h sessions**; T4×2 ≈ 30 h/week [VERIFIED: https://www.kaggle.com/discussions/product-announcements/607202, https://gpuperhour.com/blog/free-cloud-gpus-and-credits].

| Platform | Peak | MFU | FLOPs/week | Tokens/week at 9e8 FLOPs/token |
|---|---|---|---|---|
| TPU v5e-8 | 8×197 = 1,576 TFLOPS bf16 [VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e] | 20–30% (JAX tutorial measured 16–27% on a tiny GPT [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html]) | 1576e12×0.25×72,000 ≈ 2.8e19 | **≈ 25–38B** |
| T4×2 | 2×65 TFLOPS fp16 [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf] | 15–25% | 130e12×0.2×108,000 ≈ 2.8e18 | ≈ 3–4B |
| 3060 laptop | ~25 TFLOPS effective [ESTIMATE] | 35% | dev only | ablations at ≤20M params |

**Budget** [ESTIMATE]:
- Stages 1+2: 210B tokens × 9e8 ≈ **1.9e20 FLOPs ≈ 6–8 TPU-weeks**.
- Ablations: about 4 TPU-weeks.
- RL: about 1 TPU-week plus T4 rollouts.
- Total: about **13 TPU-weeks**.

The T4s run in parallel, every week:
- Open-teacher data generation at **0.1–0.6B tokens/week** (vLLM, fp16/AWQ) [ESTIMATE from the compute digest]. Over 20 weeks that is about 2–12B tokens before filtering. **This is the real bottleneck**, so procedural CPU generators carry most of Stage 2.
- The eval harness, index builds and RL rollouts.

**If the TPU is unavailable:** cut Stage 1 to 60B tokens and run on T4s, which takes about 20–25 weeks and still gives about 700 tokens/param.

**T4 engineering (sm_75, no bf16):**
- **Numerics:** fp16 matmuls with dynamic loss scaling. Delta-rule states, decay gates (in log space) and scan accumulators stay in **fp32**. Gradient clipping is applied per recurrence iteration.
- **Kernels:** Triton ≥3.3 dropped sm_75 [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md]. Either pin a Triton 3.2 toolchain (torch 2.6 [ESTIMATE, verify]) for flash-linear-attention, or use our **pure-PyTorch chunkwise GDN** (chunk size 64, matmul form). The pure-PyTorch path is mandatory as a reference anyway.
- **Parallelism:** DDP over PCIe; no tensor parallelism.
- **Memory:** activation checkpointing at each core iteration, so memory does not grow with r.
- **One codebase on TPU:** the same chunkwise algorithm in pure JAX (`lax.scan` over chunks, matmul inside each chunk). Static shapes: sequence length 2048, and r drawn from compiled buckets {1, 2, 4, 8, 16}. 4-way data parallel × 2 if 8-way has poor MFU, per the tutorial's 13% vs 27% observation.
- **Persistence:** checkpoint every 45 minutes to /kaggle/working, then push to the HF Hub (100 GB private) [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits].

**Datastore v0** [ESTIMATE]:
- 30B tokens in 128-token chunks gives about 234M chunks.
- 256-bit binary codes take about 7.5 GB, which fits Kaggle's ~30 GB RAM.
- int8 rescoring vectors stay on disk (mmap). Compressed text is about 35 GB.
- Shards ship as public HF and Kaggle datasets.
- Building the index costs about 1e18 FLOPs, under an hour of TPU time.

---

## 8. Inference cost per query

Forward FLOPs/token ≈ 2·N_active: 1.2e8 at r=1, 2.5e8 at r=4, 4.2e8 at r=8 [ESTIMATE].

| Query type | Budget | FLOPs | T4 latency [ESTIMATE] |
|---|---|---|---|
| Factoid QA | 1 search (~100 ms CPU), state priming of 8×128 tokens (9e10), 20 output tokens, r=4 | ~1e11 | ~0.3 s |
| GSM8K | 64 samples × 300 tokens × r=8, plus 64 Python runs | ~8e12 | ~2–4 s (batched; the recurrent state is constant-size, so 64-way batching is cheap) |
| HumanEval | 16 samples × 400 tokens × up to 8 repair rounds | ~2e13 | ~10–30 s |
| Chat turn | 8 drafts × 500 tokens × r=4, plus critic | ~1e12 | ~3–5 s. Sequential depth (5+5r+3 layer calls per token) dominates: about 5 ms/token at r=8 [ESTIMATE] |
| ARC-AGI task | test-time training (≈200 inner steps) + 32 samples | ~1e15 | ~1–3 min |

The full Tier A eval suite is about 15–30 T4-hours, so a full scoreboard run fits about once per week. Test-time compute is not free, and we publish FLOPs per query next to every score.

---

## 9. Scaling story

| Axis | 0.1B (M1) | 1B | 10B | Measured by |
|---|---|---|---|---|
| Core params | 87M | ~700M dense core (d=1536) | ~4B dense core | loss and pass@k vs N |
| Sparse memory params | 10M PKM | ~300M PKM | ~6B PKM. Memory layers scale to 128B memory params [VERIFIED: https://arxiv.org/abs/2412.09764] | factual QA vs PKM size at fixed FLOPs |
| Datastore | 30B tokens | 300B | 2T+ (no saturation seen up to 1.4T [VERIFIED: https://arxiv.org/html/2407.12854]) | QA accuracy vs log(store) |
| Test-time depth r | ≤32 | ≤64 | ≤128 | accuracy vs log(r), Huginn-style |
| Samples × verifier | 64 | 256 | 1,024+ | accuracy vs log(N) |
| Tools | Py/SymPy/Z3/Lean | + geometry DDAR engine, theorem libraries | + docking, molecular dynamics, protein-structure tools, lab-protocol simulators (gated) | coverage |
| Self-play data | 10–50% of the mix | majority | dominant | verified-tokens/week |

**Why the scaling should be predictable** [HYPOTHESIS]:
- Each axis is separable and gets its own power-law fit from the 0.1B grid (Experiment 1).
- We fit a composite law Acc(N, M, D_store, r, k) at {20M, 50M, 87M} and extrapolate. Scaling up is allowed only when the fit predicts the next point within the CI.
- Bigger cores raise base pass@k. That *multiplies* the value of search, since test-time compute needs non-trivial base success [VERIFIED: https://arxiv.org/abs/2408.03314].

**Compute wall** [ESTIMATE]:
- Free Kaggle compute stops at about 1B: 1B × 20B tokens = 1.2e20 FLOPs, about 4 TPU-weeks.
- 10B at Chinchilla scale needs about 1.2e22 FLOPs, about 7+ years of free quota.
- Route: apply to TPU Research Cloud in week 1. It is free, but VM/storage costs are "minimal" rather than zero, and results must be published [VERIFIED: https://sites.research.google/trc/faq/].
- Most growth (store, r, tools, self-play) needs **no core retraining**, so capability keeps rising inside the free envelope even with no TRC access.

**Toward the north star** [HYPOTHESIS]:
- Science is a verifier-rich environment: proofs, simulations, assays. The same loop (propose, act with tools, verify, keep verified data) is the engine, and that is the design's main structural bet.
- **Biosafety is a hard stop, not a knob:**
  - no pathogen-enhancement, toxin-synthesis or dual-use protocol content in the store or the training data;
  - the tool registry is allow-listed;
  - the controller enforces a refusal policy;
  - every wet-lab-adjacent output needs qualified human review;
  - bio models and data get dual-use review before release.

---

## 10. First three experiments (go/kill)

**E1: Backbone × recurrence grid (weeks 1–4; 3060 + T4 + TPU).**
- **Setup:** GDN vs RWKV-7 vs Mamba-2 at 20M/50M, 5B tokens each, plus GDN {20M, 50M, 87M} × r ∈ {1, 4, 16}. Metrics: validation loss, Reasoning Gym accuracy, needle recall, measured MFU.
- **GO:** the winning backbone reaches ≥15% MFU on T4 and ≥20% on TPU; r=16 beats r=1 by ≥8 points on procedural tasks; and 50M@r=16 ≥ 87M@r=1 on them (recurrence is worth ≥1.7× params).
- **KILL:** a gain under 3 points beyond r=4. Then drop recurrence above 4 and spend the core's tied capacity on width and critic. If MFU is under 10%, fall back to the backbone with the best kernel path.

**E2: Retrieval as state priming (weeks 4–7).**
- **Setup:** 50M model, 10B-token store. Compare no retrieval, text-append (K=8) and state-merge (K=8) on NQ, TriviaQA and MMLU-dev, with leakage audits.
- **GO:** state-merge recovers ≥80% of text-append's gain at ≤30% of its FLOPs, and NQ ≥ 30 EM at 50M.
- **KILL:** under 50% of the append gain. Then switch to text-append (costlier but proven RETRO/FiD pattern). If even text-append gives under +10 NQ points over closed-book at 50M, the reader is too small: reassign the headroom to the prelude.

**E3: Procedure pipeline on GSM8K, plus a commonsense probe (weeks 7–11).**
- **(a) Setup:** 87M full stack, program-of-thought + Python + critic + 64 votes, on a decontaminated train/dev split.
  - **GO:** ≥55% by week 11, on the trajectory to beat TinyGSM-125M's 63.1 by M1.
  - **KILL:** <30%. Then the data supply is the bottleneck, so redirect T4 time to verified PoT generation and pause the other stages.
- **(b) Setup:** a 50M critic trained on 2M synthetic commonsense contrast pairs, measured on the WinoGrande dev set.
  - **GO:** ≥60 (clearly above chance).
  - **KILL:** ≤54. Then publicly downgrade the commonsense rows to "expected loss at M1" and re-plan the milestone. Honesty beats hope.

---

## 11. Top risks

1. **The fluency/chat wall** (highest). No evidence exists that a model under 1B reaches GPT-3.5 chat quality. Mitigations: compose-not-generate, phrase copying, critic distilled from a judge. It may still lose.
2. **The commonsense wall** (HellaSwag/WinoGrande). Retrieval has no key for these rows, and the verifier route is untested at this scale.
3. **State priming fails** (E2). The fallback costs about 8x more serial scan at inference, but stays compliant.
4. **Synthetic-data throughput.** 0.1–0.6B tokens/week of open-teacher data is thin; procedural generators must carry the load.
5. **TPU availability.** v5e-8 is sometimes oversubscribed or disabled [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab]. On T4-only, token budgets drop about 8×.
6. **fp16 recurrence instability on T4.** Mitigated by fp32 state, log-space gates, and per-iteration clipping.
7. **Datastore contamination creates a fake win.** RETRO-style models exploit leakage more strongly than baselines [VERIFIED: https://arxiv.org/abs/2112.04426]. Mitigations: the mandatory audit plus a private post-cutoff set.
8. **Protocol ambiguity** around GPT-3.5 snapshots and judges. Mitigations: max-reported targets, stored-answer pairwise judging, and publishing the local judge's correlation with known rankings.
9. **Licensing.** Mitigations: NC/BY-SA manifests; BY-SA data stays in the store, not in weights; never closed-API outputs.
10. **Scaling past 1B depends on TRC or grants.** The plan stays productive without them by growing the store, r, tools and self-play.
11. **Biosafety at the north-star stage.** Hard gate, as described in §9.

---

*Summary for the orchestrator:*
- **Design:** an 86.96M-parameter, attention-free Gated-DeltaNet stack. Components: a weight-tied recurrent core, a 10M product-key memory, a test-time memory, a 9M critic, retrieval as state priming, and exact tools.
- **Likely M1 outcome:** wins on retrieval QA, ARC-AGI and parts of math/reasoning/Korean; **probable losses** on commonsense, MMLU, code and chat, each with a pre-registered wall-breaking campaign.
- **Scaling:** five separable axes, each with a measured power law before it is scaled.
