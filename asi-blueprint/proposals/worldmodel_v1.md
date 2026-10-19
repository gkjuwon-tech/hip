# Proposal WM-1 "ORRERY": Latent World Model + Energy-Based Reasoning

Author role: chief architect for the "latent world model / energy" philosophy. Date: 2026-10-19.
Labels: [VERIFIED: url] = taken from the research digest or a primary source it cites; [ESTIMATE] = my arithmetic or judgement; [HYPOTHESIS] = untested design claim.

---

## 1. Thesis

A big LLM spends its parameters on three jobs at once: (a) **storing facts** (about 2 bits per parameter at best [VERIFIED: https://arxiv.org/html/2404.05405]), (b) **amortizing reasoning** into one forward pass per token, because an autoregressive model cannot "think again" about its answer except by writing more tokens, and (c) **surface fluency**, i.e. modelling every token of the output distribution, including all the tokens that carry no decision. ORRERY steals each of the three without paying for it in weights:

- Facts go to a **frozen, decontaminated datastore** (0 parameters, scales with disk).
- Reasoning is **not amortized**. The answer is a *point in latent space* that is found by **iterative optimization**: denoising plus descent on a learned energy E(context, z). Test-time FLOPs, not parameters, buy depth. This is the TRM/Huginn lesson (7M recursive nets beating frontier LLMs on Sudoku and ARC [VERIFIED: https://arxiv.org/html/2510.04871v1]; 3.5B recurrent depth reaching "50B-equivalent" compute [VERIFIED: https://arxiv.org/abs/2502.05171]) moved from grids to language.
- Fluency is **delegated to a small renderer** that verbalizes an already-decided latent plan, and that copies spans from retrieved human text (CoG-style [VERIFIED: https://arxiv.org/abs/2307.06962]) instead of inventing every token.

One scalar, the energy, is the system's universal currency: it ranks multiple-choice options, chooses between candidate plans, decides when to stop thinking, and is trained to agree with exact verifiers (Python, SymPy, unit tests) wherever those exist. The bet [HYPOTHESIS]: **a discriminative energy over latents is far more parameter-efficient than a generative token likelihood**, because judging is easier than generating, and the research shows that verifiers matter more than generator size at small scale (TinyGSM: "the verifier seems to play a bigger role than generation model size" [VERIFIED: https://arxiv.org/html/2312.09241]).

The honest part, stated up front: no published system shows this works for open-domain language at under 100M. Section 5 gives row-by-row odds, and several rows are red.

---

## 2. Fair-play rules ORRERY operates under

1. **Parameters** = every trainable weight that runs at inference: embeddings, encoder, retrieval projection, fusion, reasoning core, energy head, test-time-memory initialization, decoder, pointer and halting heads. Total must be under 100,000,000. ORRERY uses **91,368,970** (Section 4).
2. **Training-only weights are disclosed, not counted**: the EMA target encoder used by the JEPA loss never runs at inference. Teacher models used to generate training data are named with their sizes and licenses.
3. **Data, not parameters**: tokenizer, BM25 statistics, FAISS IVF centroids and PQ codebooks (disclosed with sizes, since they are learned), the text datastore and its index vectors.
4. **Allowed at inference**: the frozen, dated, hashed local datastore; an offline Python sandbox, SymPy, Z3 and Lean; deterministic solvers. All CPU and 0 parameters.
5. **Banned**: any neural net over 100M at inference, any LLM API, live web search, eval items or their solutions in training data or datastore, dataset-identity routing, benchmark-specific prompts or heads, running hidden tests during generation, and any GPT-3.5/GPT-4 output in training data (OpenAI ToS forbids it [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/], and it would contaminate the head-to-head).
6. **Training splits** of public benchmarks may appear inside the general mixture (standard practice: FiD fine-tuned per dataset [VERIFIED: https://arxiv.org/abs/2007.01282]), but with no per-dataset heads, and every headline row is also reported on a "no-train-split" ablation and the private post-cutoff Tier C set.
7. **Transformer rule**: no softmax (or Hopfield-equivalent) attention over the token sequence anywhere. ORRERY has exactly one gray-zone component, the copy pointer (a normalized score over retrieved token positions, i.e. over external items, not the input sequence). It is reported separately and there is an ablation mode, "pure-recurrent copy", where the pointer is replaced by appending spans to the decoder stream.
8. **Disclosure per query**: FLOPs, wall-clock, number of candidates, refinement steps, tool calls. Test-time-compute wins without FLOP disclosure do not count (Snell et al. standard [VERIFIED: https://arxiv.org/abs/2408.03314]).
9. **Scoring** is generative everywhere, as the baseline protocol requires: for multiple choice the system internally scores options by energy, then the decoder *emits* the answer letter, which is exact-matched.

---

## 3. Architecture

### 3.1 Component diagram

```
                         +------------------------------------------------+
   user prompt --------->|  TOKENIZER  (32k BPE, Korean-aware, 0 params)  |
                         +------------------------+-----------------------+
                                                  |
             +------------------------------------v-------------------------------+
             |  E_ctx  CONTEXT ENCODER  (Gated DeltaNet, 10 layers, d=512)        |
             |  -> per-token states h_1..n   + final fast-weight states S (per head)|
             +---------+-------------------------------------+-------------------+
                       | pooled query (512->256)             |
                       v                                     |
   +-------------------------------------+                   |
   |  RETRIEVER (0 params beyond proj)   |                   |
   |  BM25 (bm25s) + binary dense index  |                   |
   |  over 50-300B decontaminated tokens |                   |
   |  RRF fusion -> top-K chunks (K=8-32)|                   |
   +------------------+------------------+                   |
                      | chunk text                            |
                      v                                       |
   +-------------------------------------+                   |
   |  E_ctx (shared weights) encodes     |                   |
   |  each chunk -> chunk states          |                   |
   |  READ-FUSION (2 GDN layers): scans  |-----> S' = S + delta-rule writes of
   |  chunks, delta-rule writes into S   |       retrieved evidence (no softmax)
   +-------------------------------------+                   |
                                                              v
   +--------------------------------------------------------------------------+
   |  LATENT REASONING CORE  R  (2 weight-tied layers, iterated t = 1..T)     |
   |  state: Z = 64 "thought slots" x 512                                     |
   |  each step: slot-mix MLP -> linear read from S' (q.S', no softmax)       |
   |             -> SwiGLU ; conditioned on noise level sigma_t / step t       |
   |  role 1 (System 1): amortized denoiser  Z_noise -> Z_hat in 1-4 steps     |
   |  role 2 (System 2): refinement; Z <- Z - eta * dE/dZ + denoise(Z)        |
   +------------+-----------------------------------------+-------------------+
                |                                         ^
                v                                         | gradient of E
   +-----------------------------+             +----------+-----------------+
   |  ENERGY HEAD  E(S', Z)      |------------>|  SEARCH CONTROLLER (0 par.)|
   |  1 core-type layer + MLP    |  scalar     |  N candidates, beam/MCTS,  |
   |  = universal verifier/value |             |  halting head (33k params) |
   +-----------------------------+             +----------+-----------------+
                                                          | best plan Z*
                                                          v
   +--------------------------------------------------------------------------+
   |  RENDERER / DECODER  (GDN, 6 layers, tied embeddings)                    |
   |  FiLM-conditioned on Z*, copy-pointer over retrieved spans (gray zone)   |
   |  emits: text | answer letter | PY(...) / SOLVE(...) / SEARCH(...) calls  |
   +------------------+-------------------------------------------------------+
                      | tool call                         ^ tool result re-encoded
                      v                                   |  by E_ctx, written into S'
   +-------------------------------------+               |
   |  TOOLS: Python sandbox, SymPy, Z3,  |---------------+
   |  Lean, unit-test runner (0 params)  |
   +-------------------------------------+
   +-------------------------------------+
   |  TEST-TIME MEMORY M (Titans-style   |  fast MLP, learned init counted;
   |  MLP, updated by gradient on surprise|  reset per episode
   +-------------------------------------+
```

### 3.2 Inference dataflow, step by step

1. **Encode.** The prompt goes through E_ctx (a Gated DeltaNet: gating plus delta-rule overwrite, the strongest attention-free mixer on in-context retrieval at its scale [VERIFIED: https://arxiv.org/abs/2412.06464v3]). Output: token states plus the per-head fast-weight matrices S, which are a compressed associative memory of the context.
2. **Retrieve.** A pooled query (512 to 256 projection) hits a binary dense index (32x smaller than fp32, 92.5% raw / 96.45% rescored quality [VERIFIED: https://huggingface.co/blog/embedding-quantization]) and BM25 (0 params [VERIFIED: https://arxiv.org/abs/2407.03618]); results are fused by reciprocal rank fusion. Korean uses a rule-based morphological or character-n-gram BM25 tokenizer.
3. **Read.** Each retrieved chunk is encoded by the *same* E_ctx (so retrieval costs no extra encoder parameters), then two READ-FUSION layers scan the chunk states and write evidence into S with delta-rule updates. No N-by-N score matrix exists anywhere.
4. **Think (System 1).** The core starts from N noise samples of the 64-slot latent Z and denoises them in 1 to 4 steps into candidate answer plans. Easy prompts stop here if the energy is low and the halting head agrees.
5. **Think (System 2).** For uncertain prompts the controller runs refinement: each step mixes a denoising update with an energy-gradient step (Langevin-style, with annealed noise). Candidates are pruned by energy; survivors branch. The halting head decides when the energy improvement per step has flattened. This is where test-time compute buys accuracy.
6. **Act (latent world model).** When the plan contains a tool action, the core first *predicts the latent of the tool result* (JEPA-style). The decoder renders the call, the tool runs, the actual result is encoded and written into S'. The prediction error ("surprise") is used two ways: it updates the test-time memory M (Titans-style [VERIFIED: https://arxiv.org/abs/2501.00663], here on an attention-free core), and a large surprise triggers re-planning. When the prediction is confident and the tool is expensive (e.g. a long search), the controller may skip the call; this is a learned imagination-versus-action trade [HYPOTHESIS].
7. **Verify.** Where an exact verifier exists (code executes against self-written tests, SymPy checks an equation, Z3 checks constraints), its verdict overrides the energy. Where none exists (chat, commonsense), the energy is the verdict.
8. **Render.** The decoder verbalizes Z*, with FiLM conditioning on the plan and a copy pointer that lifts phrases from retrieved human-written text. For multiple choice, the option with lowest energy is selected and the decoder emits its letter.

### 3.3 Why "world model"

The core is trained to predict *latent consequences*: the latent of the next paragraph, of a masked span, of a tool's output, of a program's execution result, of a solution given a problem. That is a world model of text-and-tools. The energy is its compatibility function. This is the LeCun-style JEPA program (LLM-JEPA already improves GSM8K and Spider when added to next-token loss [VERIFIED: https://arxiv.org/abs/2509.14252]) combined with energy-based System-2 inference (Energy-Based Transformers report up to 35% higher scaling rate [secondary: https://liner.com/en/review/energybased-transformers-are-scalable-learners-and-thinkers; paper https://arxiv.org/abs/2507.02092]), ported to a non-Transformer backbone. The port itself is [HYPOTHESIS].

---

## 4. Exact parameter budget

Shared definitions: d = 512, SwiGLU hidden 1408, 8 heads.
- One GDN layer = q,k,v,o (4d^2) + output gate (d^2) + alpha/beta gates (2·d·8) + short convs (4·3d) + SwiGLU (3·d·1408) + 2 norms = **3,488,768**.
- One core-type layer = q-read, out, gate (3d^2) + slot-mix MLP (64->256->64, 33,088) + SwiGLU + 3 norms = **2,983,744**.

| # | Module | Composition | Trainable params |
|---|---|---|---|
| 1 | Token embedding (tied to decoder output) | 32,768 × 512 | 16,777,216 |
| 2 | Context encoder E_ctx (also encodes chunks, tool outputs, MC options) | 10 GDN layers | 34,887,680 |
| 3 | Retrieval query projection | 512 × 256 | 131,072 |
| 4 | Read-fusion | 2 GDN layers | 6,977,536 |
| 5 | Latent reasoning core R (weight-tied, iterated) | 2 core layers (5,967,488) + noise-level MLP 512×512+512 + slot init 64×512 + step embedding 32×512 | 6,279,296 |
| 6 | Energy head E | 1 core layer + MLP 512->1024->1 | 3,510,081 |
| 7 | Test-time memory M (learned initialization) | MLP 512->1024->512 | 1,050,112 |
| 8 | Plan-to-decoder FiLM | 512 -> 1024 | 525,312 |
| 9 | Decoder / renderer | 6 GDN layers | 20,932,608 |
| 10 | Copy pointer (gray zone) | query proj 512×512 + gate | 262,657 |
| 11 | Halting / tool-routing head | 512->64->8 | 33,352 |
| 12 | Final norms | 4 × 512 | 2,048 |
| | **Total trainable at inference** | | **91,368,970** |
| | Headroom under 100M | | 8,631,030 |
| — | EMA target encoder (training only, not run at inference) | copy of 1+2 | disclosed, not counted |
| — | BM25, IVF centroids, PQ/binary codes, tokenizer | data | disclosed, 0 |

Headroom use (decided by Experiment 1): either 2 more decoder layers (+6.98M, if fluency is the binding constraint) or widening the energy head (+3-8M, if verification is). Note that 18% of the budget is embeddings; a 16k vocabulary would free about 8.4M, at a cost in Korean coverage [ESTIMATE].

---

## 5. Beating the scoreboard, row by row (honest odds)

Targets are the max reported GPT-3.5 numbers from the baseline digest. "P(win)" = my probability that the v1 system at 91M beats target plus CI on the gated protocol [ESTIMATE throughout].

| Row | Target | ORRERY v1 estimate | P(win) | Mechanism and reasoning |
|---|---|---|---|---|
| NQ-open (EM) | 38.6 | 40–48 | 0.55 | Retrieval-first. FiD-base got 48.2 with 440M total [VERIFIED: https://arxiv.org/abs/2007.01282]; we have a weaker reader but a larger datastore. |
| TriviaQA | 85.8 | 62–75 | 0.10 | FiD-base 65.0 EM. GPT-3.5's closed-book TriviaQA is extreme (82.9 EM 0-shot [VERIFIED: https://arxiv.org/abs/2407.02485]). Likely red. |
| MMLU | 71.4 | 45–58 | 0.07 | Atlas-770M reached 56.3 with retrieval [VERIFIED: https://arxiv.org/abs/2208.03299]. MassiveDS shows only +3–4 on MMLU from retrieval [VERIFIED: https://arxiv.org/html/2407.12854]. Option-energy ranking plus an exam-explanation datastore is the bet. Red. |
| KMMLU | 42.47 | 38–48 | 0.35 | Low GPT-3.5 bar; Korean exam datastore + energy ranking. |
| HAE-RAE (mean of 5) | 49.45 | 45–60 | 0.40 | Loan words, rare words and nomenclature are dictionary lookups; history and general knowledge are retrieval. |
| KoBEST | 79.3 (avg) | 68–80 | 0.25 | Energy MC scoring is natural here; Korean fluency is the constraint. |
| GSM8K | 78.1 | 55–72 | 0.20 | Program-of-thought + execution + energy-reranked candidates. TinyGSM needed 125M generator + 1.3B verifier to get 76.1 [VERIFIED: https://arxiv.org/html/2312.09241], and its data is GPT-3.5-made, which we cannot use. |
| MATH | 34.1 | 20–35 | 0.25 | SymPy offload; Qwen2.5-0.5B-Instruct hits 34.4 at 5x params [VERIFIED: https://qwenlm.github.io/blog/qwen2.5-llm/]. |
| HumanEval / HumanEval+ | 76.8 / 70.7 | 30–50 | 0.03 | The max-reported target is brutal; phi-1-small (350M) got 45 on HumanEval [VERIFIED: https://arxiv.org/abs/2306.11644]. Generate, self-test, repair. Red. |
| MBPP / MBPP+ | 82.5 / 69.7 | 40–58 | 0.03 | Same. Red. |
| BBH | 68.3 | 50–70 | 0.30 | Many BBH tasks are exactly executable (Dyck, boolean, sorting); others need verbal reasoning. |
| DROP (F1) | 64.1 | 55–70 | 0.40 | Span copy + calculator; no world knowledge needed. |
| HellaSwag | 85.5 | 55–70 | 0.04 | This is where the energy bet is sharpest: a bidirectional discriminative scorer of continuation plausibility, trained contrastively, should do much better per parameter than generative likelihood (SmolLM2-135M: 42.1 [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M]). From memory and **not verified this session**: fine-tuned discriminative encoders in the ~300M class were reported in the 80s on HellaSwag; must be checked before relying on it. At 91M, still red. |
| WinoGrande | 81.6 | 60–72 | 0.05 | Same argument; from memory and unverified, ~350M discriminative encoders were reported near 79. Everything ≤270M generative sits at chance (51–53) [VERIFIED: tiny_sota digest]. Red. |
| ARC-Challenge | 87.4 | 55–70 | 0.05 | Textbook retrieval + option energy. Red. |
| ARC-AGI-1 (pass@2) | 5.0 | 15–35 | 0.85 | Latent refinement is TRM-shaped; TRM-MLP (attention-free) got 29.6 [VERIFIED: https://arxiv.org/html/2510.04871v1]. Add test-time training on each task's demos. Green. |
| MT-Bench (pairwise vs stored GPT-3.5 answers) | 8.39 / >50% wins | 10–25% win rate | 0.02 | See below. Red. |
| AlpacaEval 2.0 LC | 22.35 | far below | 0.02 | Red. |
| Arena-Hard-Auto | 24.8 | far below | 0.01 | Red. |

**Expected v1 outcome: about 3 to 6 green rows out of about 19. Probability of winning every Tier A and Tier B row at v1: under 1%** [ESTIMATE]. I would rather say this now than discover it in month 6. The design is chosen because it has the most knobs for turning red rows green *without* adding parameters (Section 9), not because v1 wins.

### Open-ended chat, creative writing and Korean fluency: the honest wall

- Tiny models score MT-Bench 1.98–2.52 [VERIFIED: tiny_sota digest, https://arxiv.org/html/2402.14905]; GPT-3.5 scores 7.94–8.39. The smallest Arena match for GPT-3.5 is about 2.6B (Gemma-2-2B) [VERIFIED: baseline digest]. Nothing under 1B is close.
- ORRERY's attack on chat [HYPOTHESIS]: (1) the latent plan decides *content and structure* (what to say, in what order) with full test-time search under the energy; (2) the renderer only verbalizes; (3) the copy pointer borrows fluent phrasing from a **response datastore** of open-licensed human-written answers (StackExchange-style explanations, Wikipedia, open instruction data minus NC parts); (4) the energy head is trained on preference pairs, so best-of-N over plans acts as a built-in reward model. TinyStories shows sub-10M fluency is possible in a narrow register [VERIFIED: https://arxiv.org/abs/2305.07759]; the open question is breadth.
- **Creative writing** is the worst case: retrieval and copying help little with originality, and the energy has no ground truth. Expect a clear loss at v1.
- **Korean fluency**: the renderer shares 21M params across English, Korean and code. Korean open data is scarcer and 35.3% of Korean HF datasets have unknown licenses [VERIFIED: https://arxiv.org/pdf/2507.04329]. Arena Korean target is 1049.6, lower than overall, but still out of reach at v1.
- What it would take to turn chat green (Section 9): roughly a 1B-class renderer, or a decisive positive result from Experiment 3's chat probe. I will not pretend otherwise.

---

## 6. Training recipe

Token budget assumes the TPU v5e-8 is available (Section 7). "Target tokens" = tokens the decoder/predictor is supervised on; the retrieval context adds about 1.6x compute per target token.

| Stage | Purpose | Data | Objectives | Target tokens |
|---|---|---|---|---|
| S0 | Datastore and hygiene | FineWeb-Edu-dedup 220B, Wikipedia en+ko, FineWeb-2 ko, OpenWebMath/arXiv, Python-Edu, Cosmopedia (licenses: ODC-BY / CC-BY / Apache [VERIFIED: data digest]) | 13-gram + MinHash + embedding-cosine > 0.9 scrub against every eval set; URL blocklist; overlap report | — (index 50B tokens first, grow to 300B) |
| S1 | Retrieval-aware world-model pretraining | 15% fluency (curated edu en+ko), 20% retrieval-reading (KILT provenance [VERIFIED: https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/] + synthetic QA over the store), 35% procedural (Reasoning Gym 100+ generators [VERIFIED: https://arxiv.org/pdf/2505.24760], re-ARC training split [VERIFIED: https://arxiv.org/pdf/2404.07353], own SymPy/graph/execution-trace generators), 20% open-teacher distilled, 10% code | (a) next-token on decoder with retrieved context; (b) **JEPA**: predict EMA-target latents of masked spans, of next paragraph, and of tool outputs; (c) retrieval contrastive loss (query vs KILT provenance chunk); (d) VICReg-style variance/covariance terms on latents to prevent collapse | 60B |
| S2 | Latent denoising and energy | Problem-answer pairs from all S1 sources; MC train splits; candidate answers sampled from the model itself | (a) **latent diffusion**: Z* = EMA-encoder(answer) pooled into 64 slots; train R to denoise from sigma-noised Z* given S'; (b) **energy**: InfoNCE over {true, hard negatives: retrieved distractors, self-samples, corruptions}, plus a score-matching consistency term so that -dE/dZ agrees with the denoiser direction; (c) decoder reconstruction from Z* | 20B |
| S3 | Tool use and instruction format | OpenMathInstruct-2 (CC-BY-4.0) rewritten into PY/SOLVE calls, kept only if execution reproduces the answer; SmolTalk (Apache-2.0), Tulu 3 minus NC subsets [VERIFIED: data digest]; Magpie-style data regenerated from Apache-licensed open teachers | SFT on calls and on use of results; JEPA predicts tool-result latents | 5B |
| S4 | RL with verifiable rewards (GRPO) + energy calibration | Reasoning Gym, re-ARC, math with checkable answers, code with self-generated + reference tests (training-set problems only) | GRPO on the full think-act-render loop; energy head regressed onto execution outcomes (energy as a calibrated value function); a proposer head writes new problems near 50–70% solve rate | 3–5B rollout tokens |
| S5 | Preference / chat energy | Open-license preference pairs; pairs judged by a local open-weight judge on T4 (training-time only, disclosed) | Energy head trained as reward model on plans; best-of-N plan selection distilled back into System-1 denoiser | 1B |
| S6 | Self-play flywheel (continuous) | Model + tools generate problems and solutions; only verifier-accepted traces kept | STaR/expert iteration [VERIFIED: https://arxiv.org/html/2203.14465] | grows 10% -> 50% of mix |

Rules: no GPT-3.5/4 data anywhere; every teacher is named with size and license; NC data lives in a separate manifest; Wikipedia (BY-SA) lives in the datastore, not in the weights' training text where avoidable.

---

## 7. Compute plan

Verified envelope: TPU v5e chip 197 TFLOPS bf16 [VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e]; Kaggle offers v5e-8 (replacing v3-8) [VERIFIED: https://www.kaggle.com/discussions/product-announcements/607202] at about 20 h/week (third-party); T4 65 TFLOPS fp16, no bf16 [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf]; about 30 GPU-h/week (user-reported, not in Kaggle docs).

**FLOPs per target token** [ESTIMATE], for a training sample of 512 target tokens with K=8 retrieved chunks × 128 tokens:
- Encoder on 512 + 1024 tokens: 34.9M × 1536 = 53.6 GFLOP (forward multiply-adds counted as 1 param-token)
- Read-fusion on 1024: 7.0M × 1024 = 7.1
- Decoder on 512: 20.9M × 512 = 10.7
- Core, T=8 unrolled steps × 64 slots: 6.3M × 512 = 3.2; energy head on 64 slots × 4 candidates: 3.5M × 256 = 0.9
- Sum ≈ 75.5 G param-tokens; × 6 = **453 GFLOP per sample = 0.89 GFLOP per target token**, about 1.6x a plain 91M LM (6N = 0.55 GFLOP).

**Throughput** (MFU 30% on v5e-8 = 473 TFLOPS effective; the JAX tutorial measured only 16–27% for tiny models [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html], so 30% is optimistic):
- 473e12 / 0.89e9 ≈ 530k target tokens/s → × 72,000 s (20 h) ≈ **38B target tokens/week** on TPU [ESTIMATE]. At 20% MFU: 25B.
- T4x2 at 25% MFU: 32.5 TFLOPS → 36.5k tok/s → × 108,000 s ≈ **3.9B/week** [ESTIMATE].

**Schedule** (12 weeks, TPU-primary):
| Weeks | TPU v5e-8 | T4x2 | RTX 3060 6GB |
|---|---|---|---|
| 1–2 | Experiment 1/2 ablations at 20–50M | Datastore build: BM25 + embed 50B tokens with E_ctx v0 (2N ≈ 7e7 FLOP/token → 3.5e18 FLOP ≈ 30 T4-hours at 25% MFU; or about 2 TPU hours) | Kernel development, bf16 reference runs, unit tests |
| 3–4 | S1 pretraining 60B target tokens (≈ 2 weeks at 30%, ≈ 2.5 at 20%) | Teacher generation (Qwen2.5-class Apache models, ~0.1–0.6B tokens/week [ESTIMATE: compute digest]); decontamination | Eval harness development |
| 5–6 | S2 latent diffusion + energy, 20B | Re-index datastore with trained encoder; grow to 150B tokens | Small ablations |
| 7 | S3 SFT 5B | RL rollouts setup | — |
| 8–10 | S4 GRPO (rollouts on T4, updates on TPU or T4) | GRPO rollouts with tool sandbox | — |
| 11 | S5 preference energy | Local judge scoring | — |
| 12 | Experiment 3 scaling grid | Full eval suite, 3 seeds | Offline demo |

Total ≈ 90B target tokens ≈ 8e19 FLOPs, about 4% of a year of free quota (≈2e21 FLOPs/year [ESTIMATE: compute digest]). Pretraining FLOPs are not the bottleneck; **data quality, verified synthetic data and wall-clock of serial RL rollouts are**.

**If the TPU is unavailable** (reported as sometimes oversubscribed): T4-only gives ≈4B/week, so the plan stretches to about 22 weeks with S1 cut to 30B target tokens. Still above Chinchilla (about 2B for 100M) by 15x.

**T4 and TPU engineering:**
- One codebase, **JAX first**: delta-rule recurrence written as chunked matmuls (chunk 64) plus `jax.lax.associative_scan`/`lax.scan` across chunks, which XLA compiles on TPU and on CUDA GPUs [ESTIMATE that the JAX CUDA build supports sm_75; verify day 1]. This avoids Triton entirely: Triton 3.3+ dropped compute capability below 8.0 [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md].
- A PyTorch port for inference on T4/3060, with pure-PyTorch chunked kernels; optional FLA kernels pinned to the Triton 3.2 era on T4 [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.2.0/README.md].
- T4 numerics: fp16 matmuls, **fp32 recurrent state, fp32 scan accumulators, fp32 energy-gradient inner loop**, dynamic loss scaling; decay gates parameterized in log space and clamped. The energy-descent inner loop is computed in fp32 because small gradient steps underflow in fp16 [ESTIMATE].
- Static shapes on TPU: fixed K chunks, fixed 64 slots, fixed unroll T during training with random truncation (truncated BPTT through the last 2 refinement steps, TRM-style deep supervision on every step [VERIFIED: https://arxiv.org/html/2510.04871v1]).
- 2x T4 over PCIe: pure data parallel. Checkpoint every 45 minutes to `/kaggle/working`, then push to the HF Hub (100 GB private [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits]).
- Datastore: binary 256-bit codes for 1.2B chunks ≈ 38 GB, mmap on laptop SSD; on Kaggle, a 50B-token shard (≈0.4B chunks, ≈13 GB binary) fits the ~30 GB RAM [ESTIMATE].

---

## 8. Inference cost and latency per query

Multiply-adds ×2 = FLOPs. Assume a 512-token prompt, K=16 retrieved chunks, 300 output tokens [ESTIMATE throughout].

| Mode | Candidates × steps | Breakdown | FLOPs/query | T4 latency (≈10 TFLOPS effective, serial overhead included) |
|---|---|---|---|---|
| Fast (System 1) | 4 × 2 | encode 2,560 tok × 70M ≈ 0.36 TF; core 0.007 TF; render 300 × 42M ≈ 0.013 TF | ≈ 0.4 TFLOP | 0.3–0.8 s |
| Standard | 16 × 16 | + core 16·16·64 × 2·6.3M ≈ 0.21 TF; energy 0.11 TF; 2 renders | ≈ 0.75 TFLOP | 1–3 s (recurrence latency, not FLOPs, dominates) |
| Hard / verifiable | 64 × 32 + 8 tool rounds | + core 1.65 TF; energy 0.9 TF; 8 re-encodes of tool output; sandbox CPU time | ≈ 5–10 TFLOP | 10–60 s; code repair loops up to minutes |

For reference, a dense model of the GPT-3.5 class (parameter count unconfirmed; if 20B–175B) spends about 2 × N × 800 tokens ≈ 30–280 TFLOP per such query [ESTIMATE]. So ORRERY's hard mode is still 3–50x cheaper in FLOPs, but **far slower in wall-clock** on a T4 because refinement is serial. The retrieval itself (BM25 + binary Hamming search on CPU) adds about 50–300 ms [ESTIMATE]. All reported scores carry the mode used.

---

## 9. Scaling story

ORRERY has six independent axes. Only the first one costs retraining.

| Axis | 0.1B milestone | 1B | 10B | Evidence |
|---|---|---|---|---|
| Core/renderer params | 91M | encoder 400M, renderer 400M, core+energy 200M | Same split ×10; add product-key memory layers (memory params scale to 128B [VERIFIED: https://arxiv.org/abs/2412.09764]) | RWKV-7 0.19→2.9B monotone [VERIFIED: https://arxiv.org/abs/2503.14456] |
| Datastore tokens | 50–300B | 1–2T | multi-T + domain stores (PubMed, patents, code) | RETRO: no diminishing gains 150M–7B, about 10x param-equivalent [VERIFIED: https://arxiv.org/html/2112.04426]; MassiveDS no saturation to 1.4T [VERIFIED: https://arxiv.org/html/2407.12854] |
| Test-time refinement (T, N) | up to 64 × 32 | same knob, higher base pass@k | same | Compute-optimal TTS beats 14x larger models when base success is non-trivial [VERIFIED: https://arxiv.org/abs/2408.03314] |
| Energy-head calibration data | executions + preferences | + formal proofs, simulators | + lab-validated assays (human-gated) | TinyGSM verifier effect [VERIFIED: https://arxiv.org/html/2312.09241] |
| Tool registry | Python, SymPy, Z3, Lean | + geometry engine, theorem libraries | + docking, MD, retrosynthesis (biosafety-gated) | AlphaGeometry 25/30 with symbolic engine [VERIFIED: https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/] |
| Self-generated verified data | 10% → 50% of mix | dominant | dominant | R1-Zero AIME 15.6→77.9 by RL [VERIFIED: https://arxiv.org/html/2501.12948] |

Why the world-model framing scales better than an LLM [HYPOTHESIS]: parameters buy a better *energy landscape* (sharper discrimination, smoother gradients) and a better *denoiser prior*, which raises the base success rate that test-time search multiplies. Datastore and tools grow knowledge and computation linearly with disk and CPU. The energy is domain-agnostic, so a new verifier (a simulator, a proof checker) plugs in as a new calibration signal without architectural change.

**The compute wall:** 1B at Chinchilla scale ≈ 1.2e20 FLOPs, about 3 TPU-weeks, feasible on the free tier. 10B ≈ 1.2e22 FLOPs ≈ 7 years of free quota [ESTIMATE: compute digest]. The 10B step therefore requires TPU Research Cloud (free TPU quota, small VM/GCS costs, publish openly [VERIFIED: https://sites.research.google/trc/faq/]). Apply in week 1.

**Toward the north star.** Science is the regime where this design shines: verifier-rich, simulator-rich, and fundamentally about *searching* hypothesis space under a scoring function. The long-run system plans experiments in latent space, predicts outcomes (world model), runs cheap simulators to calibrate the energy, and proposes only the highest-value expensive actions to humans. **Biosafety is a hard constraint, not a knob**: the datastore and tool registry exclude pathogen-enhancement and toxin-synthesis content; the controller refuses hazardous chemistry and biology; any wet-lab-actionable output requires human expert review; bio models and datasets go through dual-use review before release.

---

## 10. First three experiments (go/kill)

**E1: "Does latent refinement buy anything on language-shaped reasoning?" (weeks 1–2, 20M model, 3060 + T4)**
- Setup: 20M ORRERY-mini on 6 Reasoning Gym families + arithmetic word problems + a small retrieval-QA set. Compare (a) decoder-only single pass, (b) System 1 denoiser, (c) refinement T = 4, 16, 32 with energy selection over N = 16.
- **Go** if: accuracy rises at least +10 points from T=1 to T=16 on held-out harder difficulty levels in at least 4 of 7 families, **and** the energy ranks the correct candidate first with AUROC ≥ 0.80 against execution truth.
- **Kill** if: gain < +3 points or AUROC < 0.65. Fallback: drop latent descent; keep the backbone and use sample-then-exact-verify (a different proposal's design).

**E2: "Does energy scoring beat likelihood on commonsense, and does JEPA help?" (weeks 2–4, 50M, TPU)**
- Setup: same compute, three arms: next-token only; next-token + JEPA; next-token + JEPA + S2 energy. Evaluate HellaSwag, WinoGrande, KoBEST-COPA (generative protocol), NQ with retrieval, and a 1k-item MMLU dev slice.
- **Go** if: energy MC scoring beats decoder-likelihood scoring by ≥ 5 points on HellaSwag and ≥ 3 on WinoGrande at equal params, and JEPA adds ≥ +1.5 points on the average.
- **Kill** if: energy scoring is within 1 point of likelihood (the discriminative-efficiency bet is false); JEPA loss is removed if it does not help.

**E3: "Is it scalable?" (weeks 10–12)**
- Setup: grid {20M, 50M, 100M} × {T = 1, 4, 16} × {retrieval on/off} [as proposed in the archs digest], plus a **chat probe**: pairwise win rate against stored GPT-3.5 MT-Bench answers judged by a local open judge, both orders.
- **Go** if: score is log-linear in params *and* in test-time compute with no saturation at T=16, the retrieval gap does not shrink with size, and the chat win rate improves at least +5 points per 2.5x params.
- **Kill / pivot** if: the test-time axis saturates by T=4 (then this is not a test-time-scaling architecture), or the chat win rate is flat across 20M→100M (then chat needs a different renderer strategy, likely a larger renderer that only becomes possible at the 1B tier).

---

## 11. Top risks

1. **Chat and creative writing loss** (near-certain at v1). Mitigation: response datastore + copy pointer + plan search; honest reporting; the main reason to reach 1B quickly.
2. **Energy hacking.** Gradient descent on Z finds adversarial latents with low energy but wrong answers (the latent-space version of reward hacking). Mitigation: noise-annealed Langevin rather than pure descent, denoiser-prior regularization, exact verifiers override energy, adversarial negatives mined from descent trajectories during S2/S4.
3. **Latent collapse / representation drift** in JEPA training. Mitigation: EMA targets, VICReg terms, monitoring latent rank every 1k steps.
4. **Plan-to-text gap.** The renderer may not faithfully verbalize Z*. Mitigation: reconstruction loss, cycle-consistency (re-encode rendered text, compare to Z*), reject renders whose re-encoded energy is high.
5. **Recurrent recall limits.** A fixed fast-weight state cannot hold arbitrary context [ESTIMATE, well known; https://arxiv.org/abs/2412.06464v3]. Mitigation: delta-rule overwrite, retrieval re-reads, test-time memory M.
6. **Datastore contamination makes a "win" fake.** RETRO exploits leakage more than baselines [VERIFIED: https://arxiv.org/abs/2112.04426]. Mitigation: full scrub, retrieval-audit logs of gold-answer strings in top-k, private post-cutoff Tier C as headline.
7. **Compute availability.** The TPU v5e-8 is reported as sometimes oversubscribed; T4-only stretches the plan to about 22 weeks.
8. **Toolchain.** JAX-on-T4 and Triton-on-sm_75 both unverified; day-1 check, pure-PyTorch fallback.
9. **Data licensing.** Korean licenses are often unknown; teacher licenses vary. Manifest every source; no closed-API outputs.
10. **Serial latency.** Refinement steps are sequential; hard mode may take a minute per query on T4, making Arena-style human evaluation impractical.
11. **Biosafety** (north star): hard stop on dual-use content; human review gate; this never gets traded off for capability.

**Bottom line:** ORRERY is the design where test-time compute, an external datastore and a learned universal verifier do the work that GPT-3.5 does with parameters. At v1 it should win retrieval QA, abstraction (ARC-AGI) and possibly several Korean, DROP and BBH rows, and it will likely lose code (against max-reported targets), MMLU-class knowledge reasoning, HellaSwag/WinoGrande and chat. Its value is that every red row has a knob that does not require more parameters, and E1–E3 tell us within 12 weeks whether those knobs actually turn.
