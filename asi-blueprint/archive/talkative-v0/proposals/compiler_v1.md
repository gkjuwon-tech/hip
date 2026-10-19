# Proposal: COMPILER-0, a neuro-symbolic compiler under 100M parameters

Author role: chief architect, "Neuro-Symbolic Compiler" track. Date: 2026-10-19.
Labels: [VERIFIED: url] = taken from the research digest, which was checked against the source. [ESTIMATE] = my arithmetic or judgement. [HYPOTHESIS] = untested design bet.

---

## 1. Thesis

A frontier LLM spends its parameters on three jobs. (1) **Storing facts**: about 2 bits per parameter, and the only demonstrated fix at small sizes is retrieval [VERIFIED: https://arxiv.org/html/2404.05405]. (2) **Emulating computation**: carrying digits, tracking variables, searching options, all done imprecisely in the weights. (3) **Producing fluent surface text**. A Python interpreter does job 2 exactly, for free. A 50–300B-token datastore does job 1 with no params counted and without ever forgetting. Wikipedia and human-written answers already contain fluent prose for job 3, ready to copy and recombine. What is left, and what has to live in the weights, is a **compiler**: a model that turns a messy natural-language task into a typed program over tools, reads what comes back, and decides when to stop. Our bet is that **the compiler is much smaller than the thing it replaces**. Evidence:
- A 125M generator writing Python reaches 63.1% on GSM8K, which beats GPT-3.5's 57.1 (5-shot) [VERIFIED: https://arxiv.org/html/2312.09241].
- PAL with Python beat PaLM-540B CoT by 15 points [VERIFIED: https://arxiv.org/abs/2211.10435].
- A 7M recursive net gets 44.6% on ARC-AGI-1 [VERIFIED: https://arxiv.org/html/2510.04871v1].
- Retrieval stands in for about 25x the parameters [VERIFIED: https://arxiv.org/abs/2112.04426v1].

Language is the I/O layer. The program trace is where the thinking happens. The program trace is also where the north star lives: science is a verifier-rich environment (simulators, proofs, assays), and a compiler is the kind of agent that can use one.

The honest corollary: the compiler thesis is strong wherever a program, a lookup or a verifier exists. It is weakest where the skill *is* the fluency: HellaSwag, WinoGrande, open chat, creative writing. This proposal attacks those walls with original mechanisms. It also states plainly that, by my own estimates, **those rows are more likely to be lost than won at v1**. A real kill/pivot gate is set for them (section 10).

---

## 2. Fair-play rules (binding)

1. **Parameters**: every trainable weight executed at inference counts. That includes the token embeddings, every backbone layer, the mode adapters, the verifier, the retrieval encoder, the static index embedder, the fast-weight memory's initial weights and all heads. Total must be < 100,000,000.
2. **Data, not params (disclosed)**: the datastore text, BM25 statistics, FAISS IVF centroids and PQ codebooks (learned, kilobytes to MB, reported), the tokenizer, the rule-based Korean morphological splitter, and the program-library source code (human- or model-written Python stored as text).
3. **Allowed at inference**: a frozen, dated, hashed offline datastore; a Python 3 sandbox with the standard library, NumPy, SymPy, Z3 and NetworkX; Lean 4 with Mathlib; SQLite over a public knowledge graph dump; and our own program library. Everything is CPU-only and offline.
4. **Banned at inference**: any neural network with >100M params, all LLM APIs, live web, and any eval item or its solution in the datastore or the program library.
5. **Training time**: open-licensed teacher models may generate data. Every teacher and its size is disclosed. A "no-teacher" ablation track is reported. **No GPT-3.5/4 outputs** in training data (OpenAI ToS [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/]). This excludes TinyGSM and MetaMathQA as training text, so TinyGSM is a design reference only.
6. **No routing by dataset identity.** The planner sees only the prompt. Hidden unit tests are never run. Every eval answer logs its retrieval and tool trace for audit.
7. **Decontamination**: 13-gram filter plus MinHash, plus an embedding scrub at cosine > 0.9, applied to the training data, the datastore AND the program library. Benchmark hosting URLs are blocklisted. The overlap report is published.
8. **Scoring**: generative exact match. The target is the max reported GPT-3.5 score per row (the baseline scoreboard). Win = score ≥ target + 95% CI half-width.

---

## 3. Architecture

### 3.1 Component diagram

```
                         ┌───────────────────────────── INPUT (en/ko text, code, grid) ─────────────┐
                         │                                                                           │
                 ┌───────▼────────┐   rule-based Korean jamo/morph split (0 params)                   │
                 │  Tokenizer 24k │   byte-fallback, ko+en+code balanced                              │
                 └───────┬────────┘                                                                   │
                         │                                                                            │
   ┌─────────────────────▼───────────────────────────────────────────────────────────────┐           │
   │  CORE: Gated-DeltaNet trunk, d=640, ZERO softmax attention                          │           │
   │   Prelude (3 layers) ──► Recurrent Reasoning Core (4 tied layers × r loops, adaptive│           │
   │                          halting, r∈[1,32]) ──► Coda (3 layers)                     │           │
   │   + Fast-weight episodic memory (Titans-style MLP, updated at test time)            │           │
   │   + 4 MODE adapters (low-rank): PLAN | READ | VERIFY | REALIZE                      │           │
   └───┬──────────────┬──────────────┬───────────────┬──────────────────┬────────────────┘           │
       │ plan tokens   │ query vec    │ score         │ copy pointer     │ halt logit                │
       ▼               ▼              ▼               ▼                  ▼                           │
 ┌────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────────┐ ┌───────────────┐               │
 │ Program IR │ │  RETRIEVER    │ │ VERIFIER     │ │ PHRASE-COPY    │ │ SEARCH        │               │
 │ (typed DSL │ │ BM25 (0p) +   │ │ tower (+1    │ │ realizer: copy │ │ CONTROLLER    │               │
 │ + Python)  │ │ binary dense  │ │ unique layer │ │ spans from     │ │ (beam/MCTS    │               │
 │ grammar-   │ │ (static 6.3M  │ │ + value head)│ │ retrieved text │ │ over programs,│               │
 │ constrained│ │ + trunk proj) │ │              │ │ + exemplar     │ │ compute-      │               │
 └─────┬──────┘ │ RRF fusion    │ └──────▲───────┘ │ answers)       │ │ optimal alloc)│               │
       │        └──────┬────────┘        │         └────────────────┘ └───────────────┘               │
       ▼               ▼                 │                                                           │
 ┌─────────────────────────────────────────────────────────────────────────┐                         │
 │ EXECUTORS (0 params, CPU): Python sandbox · SymPy · Z3 · Lean4 ·        │                         │
 │ SQLite KG · NetworkX · unit-test runner · ARC grid DSL · LIBRARY (text) │──── results ───────────┘
 └─────────────────────────────────────────────────────────────────────────┘     (spliced back into the stream)
       ▲
 ┌─────┴───────────────────────────────────────────────────────────┐
 │ DATASTORE (data, not params): 50B→300B tokens, en/ko/code/math,  │
 │ 128–256-tok chunks, binary IVF + int8 rescoring, BM25, SQLite KG,│
 │ exemplar bank of open-licensed human answers (chat realizer)     │
 └──────────────────────────────────────────────────────────────────┘
```

### 3.2 The Program IR (the key design choice)

The core does not emit free-form chain-of-thought. It emits a **typed intermediate representation** that is a strict subset of Python plus tool primitives:

```python
facts = SEARCH("ko+en", "boiling point of water at 2000 m", k=8)   # retrieval
ev    = READ(facts, ask="temperature in C")                          # reader mode, returns span+cite
x     = SOLVE(sympy_eq)  |  PROVE(lean_stmt)  |  SAT(z3_formula)
rows  = KG("SELECT ... FROM triples WHERE ...")
ok    = CHECK(pred, x)                                                # executable assertion
opts  = SCORE([c1, c2, c3, c4], by=VERIFY)                            # learned plausibility
ANSWER(x, cite=ev.cite) | RENDER(plan, style="concise")               # I/O layer
```

Decoding is **grammar-constrained**: illegal tokens are masked, so every sample parses (0 params). This matters because a tiny model's raw failure is often syntax, not logic [ESTIMATE].

### 3.3 Inference dataflow, step by step

1. **Tokenize** the prompt. Korean goes through a rule-based morph split, then the 24k vocab.
2. **Prelude encode** in PLAN mode. The trunk's state also yields a 256-d query vector. The first retrieval (BM25 plus binary dense, reciprocal rank fusion, top-32) runs for *every* query. It is cheap, and it gives the planner context about what kind of task this is.
3. **Retrieved chunks are scanned** by the trunk in READ mode. This is plain recurrent ingestion, with no cross-attention. Their key facts are written into the **fast-weight episodic memory** with a delta-rule update, so later recall is exact-ish and does not depend on the fixed recurrent state [HYPOTHESIS].
4. **Plan**: the recurrent core loops r times (adaptive halting) and then samples N candidate programs in the IR. N and r are set by the controller from the halting/uncertainty signal: easy prompts get N=1, r≈2, and hard ones get N up to 64, r up to 32. This follows compute-optimal allocation [VERIFIED: https://arxiv.org/abs/2408.03314].
5. **Execute** each program in the sandbox with a timeout of 2 s per call and 10 tool calls max per program. Each tool result is spliced back into the stream, and the program may branch on results (ReAct-style loops inside the IR).
6. **Verify**, in three tiers:
   - Exact checks: `CHECK` asserts, self-written unit tests, SymPy substitution, Lean acceptance.
   - Agreement: majority vote over executed answers.
   - The learned VERIFY tower scores (prompt, program, trace, answer).
   - The final score is a product with a calibrated temperature [ESTIMATE].
7. **Search**: if the best verified score is below threshold, the controller expands. It repairs failing programs using the error message, re-retrieves with a reformulated query, or deepens r. Budget caps are in section 8.
8. **Realize**: REALIZE mode turns the answer plus the plan into text. A pointer head copies spans from retrieved evidence and from the **exemplar bank** of human-written answers. The model writes only the glue.
9. **Emit** the answer with citations and the trace hash (audit log).

---

## 4. Exact parameter budget

GDN layer ≈ 13·d² params at d=640. That breaks down as q, k, v and o projections plus the output gate (≈5d²), and a SwiGLU FFN at hidden 1707 (≈8d²). Gates, short convs and norms add <1%.

| # | Module | Shape / formula | Params |
|---|---|---|---|
| 1 | Token embedding (tied with LM head) | 24,576 × 640 | 15,728,640 |
| 2 | Prelude: 3 GDN layers | 3 × 13 × 640² | 15,974,400 |
| 3 | Recurrent core: 4 weight-tied GDN layers (looped r times, counted once) | 4 × 13 × 640² | 21,299,200 |
| 4 | Coda: 3 GDN layers | 3 × 13 × 640² | 15,974,400 |
| 5 | Gates, short convs (k=4), RMSNorms, loop-input injection | ≈1% of 2–4, + 2×640² injector | 1,351,000 |
| 6 | Mode adapters: 4 modes × 10 layers × 4 proj × rank-32 LoRA (2×640×32) | 4×10×4×40,960 | 6,553,600 |
| 7 | Verifier tower: 1 extra unique GDN layer + 2-layer value MLP (640→1024→1) | 5,324,800 + 656,385 | 5,981,185 |
| 8 | Fast-weight episodic memory, initial weights (640→1280→640) | 2 × 640 × 1280 | 1,638,400 |
| 9 | Retrieval query projection (trunk → 256-d) | 640 × 256 | 163,840 |
| 10 | Static index embedder for the datastore side (model2vec-style, distilled from the trunk) | 24,576 × 256 | 6,291,456 |
| 11 | Phrase-copy pointer head (bilinear) | 640 × 640 | 409,600 |
| 12 | Halting + controller head (r, N, abstain) | 640 × 64 + 64 × 4 | 41,216 |
| 13 | Numeric-slot head (marks spans as variables for the IR) | 640 × 128 | 81,920 |
| | **Total counted** | | **91,488,857** |
| | Headroom under 100M | | 8,511,143 |

Notes:
- **Effective depth per token** with r loops is 3 + 4r + 3 layers. At r=8 that is 38 layer-passes from 10 unique layers. This is the Huginn/TRM trick on an attention-free block [VERIFIED idea: https://arxiv.org/abs/2502.05171].
- The headroom is deliberately reserved for the one thing E3 may force: a larger embedding or an extra REALIZE layer for fluency.
- The static index embedder (row 10) exists so the 0.8B-chunk datastore can be embedded on CPU. Running the full trunk over 300B tokens is the alternative. It is affordable on TPU (2N ≈ 0.11 GFLOP/token × 3e11 ≈ 3.3e19 FLOPs ≈ 1 TPU-week [ESTIMATE]), but it would have to be redone after every core retrain. So the datastore side uses the static table, and the query side uses the trunk projection, aligned contrastively.

---

## 5. Benchmark-by-benchmark plan vs the GPT-3.5 scoreboard

Confidence is the probability that v1 (week 24) beats target + CI [ESTIMATE, all rows]. Scores are point estimates with wide error bars.

| Row | Target | Mechanism | Est. score | Conf. |
|---|---|---|---|---|
| GSM8K 0-shot | 78.1 | Program-of-thought IR + Python + maj@32 + verifier. TinyGSM's 125M gen + 1.3B verifier reached 76.1 [VERIFIED: https://arxiv.org/html/2312.09241]. Ours: a 6M-extra verifier on a shared trunk, but executed answers and unlimited procedural data | 72–82 | 0.45 |
| MATH 4-shot | 34.1 | SymPy/`SOLVE`, case enumeration, search. Qwen2.5-0.5B-Instruct = 34.4 [VERIFIED: https://qwenlm.github.io/blog/qwen2.5-llm/] | 30–42 | 0.5 |
| BBH | 68.3 | Most tasks are exactly executable (Dyck, boolean, sorting, dates, logical deduction → Z3) | 65–78 | 0.55 |
| DROP F1 | 64.1 | READ spans → Python arithmetic. Passage is given, no retrieval needed | 60–72 | 0.5 |
| ARC-AGI-1 pass@2 | 5.0 | Grid DSL program search + test-time training of fast weights. TRM 7M: 44.6 (MLP: 29.6) [VERIFIED: https://arxiv.org/html/2510.04871v1] | 15–35 | 0.9 |
| NQ-open | 38.6 | Retrieve + READ + exact span. FiD 440M total: 48.2 [VERIFIED: https://arxiv.org/abs/2007.01282] | 38–47 | 0.6 |
| TriviaQA | 85.8 | Same. 85.8 is a 5-shot accuracy-like score; FiD-base EM 65.0 [VERIFIED: https://arxiv.org/abs/2007.01282]. Main risk is protocol (EM vs alias matching) | 68–80 | 0.15 |
| HumanEval+ / MBPP+ | 70.7 / 69.7 | Generate code + self-tests + execute-repair loop + API-doc retrieval. phi-1-small 350M: 45 [VERIFIED: https://arxiv.org/abs/2306.11644] | 45–65 / 50–68 | 0.15 |
| MMLU 5-shot | 71.4 | Retrieve textbook/explanation chunks + per-option evidence scoring + STEM questions compiled to SymPy + option elimination. Atlas-770M: 56.3 [VERIFIED: https://arxiv.org/abs/2208.03299] | 52–64 | 0.1 |
| ARC-Challenge | 87.4 | Science-textbook retrieval + option scoring | 65–80 | 0.1 |
| KMMLU | 42.47 | Korean exam/law/wiki store + cross-lingual retrieval; target is close to the chance-adjusted midrange | 40–50 | 0.4 |
| HAE-RAE (per subset) | ~49–62 | LW/RW/SN subsets are dictionary lookups (KG + Korean dictionary datastore); HI/GK are retrieval; RC is the hard one | 50–65 | 0.35 |
| KoBEST | 79.3 avg | BoolQ via READ; SentiNeg via verifier; COPA/HellaSwag as below | 65–78 | 0.2 |
| HellaSwag 10-shot | 85.5 | See "F-wall" below | 45–65 | 0.03 |
| WinoGrande | 81.6 | See "F-wall" below | 55–68 | 0.03 |
| MT-Bench | 8.39 | See "Chat" below | 4.0–5.5 | <0.02 |
| AlpacaEval 2 LC | 22.35 | Pairwise vs stored GPT-3.5 outputs, local judge | 5–15 | <0.05 |
| Arena-Hard | 24.8 | Same | 2–8 | <0.02 |

### The F-wall: HellaSwag and WinoGrande (compiled commonsense)

Why tiny models fail here: plausibility of an everyday continuation is a smooth statistical judgement over a huge space of situations. It is exactly what massive pretraining buys. Best ≤150M: HellaSwag 42.1, WinoGrande 51.3 [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M]. There is no exact verifier. My original attacks [HYPOTHESIS, all]:

1. **Situation retrieval.** Index procedural and narrative text (WikiHow-like how-to text from FineWeb-Edu, with ActivityNet captions and WikiHow test articles scrubbed) as event chains. For each candidate ending, retrieve the nearest situations and compute *evidence agreement*. That is a kNN plausibility vote, not a learned one.
2. **Contrastive plausibility energy.** Train the VERIFY tower on hundreds of millions of self-made (context, true continuation, corrupted continuation) triples from the datastore: shuffled steps, swapped entities, negated outcomes. This reuses the parameters already budgeted for verification.
3. **WinoGrande as constraint solving.** Compile "The trophy doesn't fit in the suitcase because *it* is too big" into two candidate bindings. Query the KG and the store for property relations (size/contain, cause/effect), and pick the binding consistent with the retrieved relation.

Honest read: these push toward 60s, not 85. phi-1.5 at 1.3B with synthetic data reached only 73.4 on WinoGrande and 47.6 on HellaSwag [VERIFIED: https://arxiv.org/html/2309.05463]. **The compiler thesis does not have a credible path to 85.5 HellaSwag at 0.1B, and I will not pretend it does.** Experiment E3 decides whether to spend the 8.5M headroom, or whether the project must renegotiate this row.

### Chat, creative writing, Korean fluency

GPT-3.5's chat quality comes from RLHF over massive preference data plus 100B+ params of fluency. MT-Bench at ~125–135M is 1.98–2.52 [VERIFIED: https://arxiv.org/html/2402.14905]. The compiler attack:
- **Plan-then-realize.** PLAN mode emits an outline program: intents, sections, constraints from the instruction, e.g. "3 bullets, formal, Korean".
- Each section is grounded by retrieval from an **exemplar bank** of open-licensed human-written answers (StackExchange, Wikipedia, SmolTalk/Tulu non-NC subsets [VERIFIED: https://www.huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/blob/main/README.md]).
- REALIZE copies phrases (CoG-style copy LM, which improves by enlarging the index without retraining [VERIFIED: https://arxiv.org/abs/2307.06962]).
- Format constraints are **executed as checks**: word count, bullet count, language-ID, JSON validity. That makes IFEval-style instruction following partly verifiable. It is the one chat-adjacent area where I expect real gains.
- Math, code and reasoning MT-Bench turns go to the compiler and should score well. Writing, roleplay and humanities turns are where we lose.

Creative writing (poems, stories with novel combinations) is the worst case. Copying from a store produces pastiche, and a 100M realizer produces bland glue. Korean fluency is worse still: open Korean data is scarcer, and 35.3% of Korean HF datasets have unknown licenses [VERIFIED: https://arxiv.org/pdf/2507.04329]. **Estimated MT-Bench 4–5.5 vs 8.39. This row is lost at v1 with ~98% probability.** The chat path is a scaling problem (section 9), not a v1 promise.

**Bottom line of section 5**: likely wins are ARC-AGI, BBH, MATH, NQ, DROP, maybe GSM8K, KMMLU and HAE-RAE. Contested: code, TriviaQA. Likely losses: MMLU, ARC-C, HellaSwag, WinoGrande, KoBEST, all chat rows. The overall win condition (all of Tier A + B) is **not** expected at v1 [ESTIMATE]. Saying otherwise would be a lie. The plan is built so each loss becomes a measured gap with a named scaling lever.

---

## 6. Training recipe

All counts are token-passes [ESTIMATE].

| Stage | Weeks | Tokens | Data | Objective |
|---|---|---|---|---|
| S0 Tokenizer + datastore v0 | 1–3 | — | FineWeb-Edu-dedup subset, en+ko Wikipedia, FineWeb-2-ko, OpenWebMath, Python-Edu → 50B tokens indexed; decontamination pass | BM25 + static-embed build (CPU, T4) |
| S1 Base language + reading | 3–5 | 40B | 15% fluency (edu en/ko, Cosmopedia v1 Apache-2.0 [VERIFIED: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia]), 35% retrieval-conditioned text (chunk + retrieved neighbours prepended, RETRO-style but in-stream), 30% code, 20% math web | Next-token + LLM-JEPA embedding loss [VERIFIED: https://arxiv.org/abs/2509.14252], recurrence r~Poisson(4) with truncated BPTT over the last 2 loops |
| S2 Procedure + tool traces | 5–9 | 30B | Reasoning Gym 100+ generators [VERIFIED: https://arxiv.org/pdf/2505.24760], re-ARC train split [VERIFIED: https://arxiv.org/pdf/2404.07353], own SymPy/Z3/graph/date generators, KILT retrieval-provenance traces [VERIFIED: https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/], OpenMathInstruct-2 (CC-BY-4.0) rewritten to IR and **kept only if execution reproduces the answer** [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2], Stack v2 permissive functions + generated tests | Next-token on IR with tool outputs masked from the loss; deep supervision at every loop |
| S3 Verifier + plausibility | 8–10 | 5B | (problem, program, trace, correct?) from S2 rollouts; contrastive commonsense triples; KoBEST/HellaSwag-*style* triples from the store (never the benchmark) | BCE/ranking on VERIFY mode + verifier tower |
| S4 Chat/realize SFT | 10–11 | 3B | SmolTalk, Tulu-3 minus NC [VERIFIED: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/main/README.md], Magpie-style data regenerated from an Apache-licensed teacher on T4, rewritten as outline→copy-realize traces | Next-token + pointer loss + format-check rewards |
| S5 RLVR + self-play | 11–24 | 20B+ rollouts | Problem proposer (PLAN mode) writes problems near the 50–70% solve frontier; verifiable-only acceptance | GRPO with exact rewards [VERIFIED approach: https://arxiv.org/html/2501.12948]; STaR-style rationalization; expert iteration into S2 data |

Teachers: open-weight 1.5B–7B models on T4 via vLLM, about 0.1–0.6B synthetic tokens/week [ESTIMATE from the compute digest]. The no-teacher track drops OpenMathInstruct-2, Magpie and the teacher regenerations and uses only generators, the store and self-play.

**Library learning** [HYPOTHESIS]: after each S5 round, frequently re-used program fragments that verify correctly are refactored into named library functions, stored as text (data, not params). The compiler then learns to call `lib.solve_rate_problem(...)` instead of re-deriving it. This lets capability grow **without growing parameters**: the library is a second datastore, of skills.

---

## 7. Compute plan

Effective FLOP-params per training token at mean r=4: 3+16+3 = 22 layer-passes × 5.32M = 117M, plus the 15.7M LM head, ≈ 133M. That gives **6 × 1.33e8 ≈ 8.0e8 FLOPs/token** [ESTIMATE].

| Platform | Sustained FLOP/s | Hours/week | FLOPs/week | Tokens/week |
|---|---|---|---|---|
| TPU v5e-8 (1,576 TFLOPS bf16 peak [VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e]; v5e-8 on Kaggle [VERIFIED: https://www.kaggle.com/discussions/product-announcements/607202]), MFU 30% | 4.7e14 | 20 | 3.4e19 | **~42B** |
| T4×2 (65 TFLOPS fp16 each [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf]), MFU 25% | 3.25e13 | 30 | 3.5e18 | ~4.4B |
| RTX 3060 laptop, ~25 TFLOPS, 35% | 8.7e12 | ~40 usable | 1.3e18 | ~1.6B |

The stage totals S1–S4 sum to about 78B tokens, about 2 TPU-weeks. S5 rollouts are generation, ~2N FLOPs per generated token with r loops. They are CPU-bound by the sandbox, not GPU-bound: Kaggle gives about 4 vCPU [ESTIMATE], at about 20–50 program executions/s. So **CPU, not FLOPs, is the self-play bottleneck**. Mitigations:
- Batch-execute pure-arithmetic programs in a vectorized NumPy evaluator.
- Run the sandbox on the laptop CPU overnight.
- Cache results by program hash.

If the TPU is unavailable (users report oversubscription [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab]), the T4-only plan is about 10x slower: ~18 weeks for S1–S4. That is still feasible, because 100M × 20 tokens = 2B tokens is Chinchilla-optimal and we are far past it.

**Engineering:**
- **Primary implementation in JAX** for TPU. The chunked delta-rule is written as matmuls inside `lax.scan` over chunks of 64, with static shapes and sequence length bucketed to 2048. The recurrent core uses `lax.fori_loop` with r sampled per batch, not per sample, to keep shapes static.
- **T4 path (PyTorch)**:
  - fp16 matmuls; the recurrent state, decay products and scan accumulators in **fp32**. T4 has no bf16 [VERIFIED: https://discuss.vllm.ai/t/no-throughput-improvement-for-quantized-qwen-2-5-7b-instruct/1901].
  - Dynamic loss scaling.
  - Pin a Triton 3.2-era toolchain: Triton 3.3+ requires compute capability 8.0+ [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md].
  - Maintain a pure-PyTorch chunked fallback for every kernel.
  - DDP over PCIe; no tensor parallelism.
- **3060**: current Triton/FLA kernels for ablations at 10–30M params, plus offline demo serving.
- Checkpoint every 45 min to `/kaggle/working`, push to the HF Hub (100 GB private [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits]), and chain sessions.
- The datastore ships as Kaggle Datasets shards, mounted read-only. Binary 256-bit vectors × 400M chunks = 12.8 GB, which fits under ~30 GB RAM with mmap [ESTIMATE].

---

## 8. Inference cost and latency per query

Per generated token: 2 × (10 + 4(r−1)) × 5.32M ≈ 0.11 GFLOP at r=1 and 0.33 GFLOP at r=8, plus the head [ESTIMATE].

| Query type | N programs × tokens | Loops r | Tool calls | GPU FLOPs | Wall-clock (3060 laptop) [ESTIMATE] |
|---|---|---|---|---|---|
| Factoid (NQ) | 1 × 60 + reading 32×200 tok | 2 | 1 search | ~1.5e12 | 0.3–0.8 s (retrieval 50 ms) |
| GSM8K | 32 × 250 | 4 | 32–64 exec | ~2e12 | 2–6 s (CPU exec dominates) |
| MATH hard | 64 × 600 + repair rounds | 8–16 | ~200 | ~3e13 | 20–90 s |
| HumanEval | 16 × 400 + 3 repair rounds | 8 | ~100 test runs | ~1e13 | 15–60 s |
| ARC-AGI task | DSL search 1e4–1e5 candidates + TTT fast weights | 16–32 | 1e5 grid evals | ~1e15 | 2–10 min |
| Chat turn | 1 plan + realize 400 tok | 2 | 2–4 searches | ~3e11 | 1–3 s |

Full Tier A eval (~80k items) is about 1–3 days on T4×2 plus CPU [ESTIMATE]. That is roughly one week of quota, so evals run on subsets weekly and on full sets monthly. **FLOPs per query are reported alongside every score**, as the fair-play rules require.

---

## 9. Scaling story

Five axes scale independently [HYPOTHESIS, to be measured]:

1. **Core params (0.1B → 1B → 10B)**: better translation of NL into programs, a higher base pass@k (which multiplies the value of search [VERIFIED trend: https://arxiv.org/abs/2408.03314]), and a stronger REALIZE mode. **This is the axis that closes the F-wall and chat**, which are the parts that really are parametric. At 1B, the phi-1.5 data point (WinoGrande 73.4 with synthetic data) and Gemma-2-2B's Arena score above GPT-3.5 [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard] suggest the F rows become winnable at roughly 1–3B with our data and retrieval. 1B needs ~1.2e20 FLOPs at Chinchilla, about 3–4 TPU-weeks, so it is feasible on free quota. **10B needs ~1.2e22, about 7 years of free quota, so it requires TPU Research Cloud** [VERIFIED: https://sites.research.google/trc/faq/]. Apply at month 3 with the public v1 results.
2. **Datastore (50B → 2T tokens)**: knowledge rows. No saturation up to 1.4T in MassiveDS [VERIFIED: https://arxiv.org/html/2407.12854]. Swappable without retraining.
3. **Skill library (text)**: grows with every self-play round. It is a compounding asset that transfers to a bigger core unchanged.
4. **Tools**: each new verifier or simulator is a new domain at 0 params. North-star tools: cheminformatics toolkits, docking, molecular dynamics, protein-structure predictors run as tools. A >100M structure predictor would be a disclosed **north-star track** tool, not part of the 0.1B milestone.
5. **Test-time compute**: N, r and search depth grow with hardware, not params.

Predictability: I commit to fitting a measured surface. The grid is {20M, 50M, 100M} × r ∈ {1, 4, 16} × datastore {5B, 50B} × N ∈ {1, 16}, each cell scored on a fixed dev suite. The surface is score ≈ f(params, r, store, N). The 1B decision is made only if the 20M→100M slope extrapolates to beating the F-wall rows by 1–3B.

**North star**: a scientific-discovery agent is a compiler over experiments. Hypothesis = program; simulator/assay = executor; result = verifier; lab notebook = datastore; validated protocols = library. The architecture does not change, only the tool registry. **Biosafety (hard stop)**:
- The tool registry and datastore exclude pathogen-enhancement, toxin-synthesis and agent-weaponization content.
- PLAN mode carries a refusal policy, and it is checked by a non-learned classifier list plus a review gate.
- Any wet-lab-actionable output needs human expert review.
- Bio models and data release only after dual-use review.

---

## 10. First three experiments (go/kill)

**E1: "Does compiling beat memorizing at 30M?"** (weeks 2–5, ~1 TPU-week)
- Train a 30M GDN compiler (S1+S2 mini, 8B tokens) on the GSM8K-style IR, plus a 30M pure-CoT twin on identical data with the same budget.
- **GO** if, on a decontaminated GSM8K-test-like dev split: IR maj@16 ≥ 45% and ≥ 2x the CoT twin, AND recurrence r=8 beats r=1 by ≥ 5 points.
- **KILL/PIVOT** if IR maj@16 < 25%, or recurrence adds < 2 points. The pivot is to drop the looped core and spend those params on width.

**E2: "Retrieval-reading at 100M"** (weeks 4–8)
- 100M trunk with the 50B store, evaluated on NQ-open dev and KMMLU dev.
- **GO** if NQ EM ≥ 38 and KMMLU ≥ 40.
- **KILL** the "static embedder + BM25" retriever if recall@32 for the gold passage is < 70% on NQ dev. Pivot to trunk-embedded datastore shards, paid with 1 TPU-week.

**E3: "The F-wall probe"** (weeks 6–12)
- 100M model with situation retrieval + contrastive VERIFY + KG binding, evaluated generatively on HellaSwag and WinoGrande validation, plus a 300-prompt Korean/English chat set judged pairwise against stored GPT-3.5 MT-Bench answers by a local judge.
- **GO** (keep the compiler-only design for v1) if HellaSwag ≥ 60, WinoGrande ≥ 62, and chat win rate ≥ 25%.
- **PIVOT** if HellaSwag < 50 or WinoGrande < 56. Spend the 8.5M headroom on REALIZE plus 20B more fluency tokens, and re-run.
- **ESCALATE** if the pivot is still < 60. This is the signal that the F rows need ~1B params. Tell the user explicitly that the "all rows at 0.1B" goal is not reachable by this design, and propose a 1B branch on TRC.

---

## 11. Top risks

1. **The F-wall is real** (HellaSwag, WinoGrande, chat, creative, Korean fluency). Probability of losing at v1 is >90% [ESTIMATE]. Mitigation: E3 gate, headroom, and the 1B branch. This risk alone decides whether the "never lose" constraint is achievable at 0.1B.
2. **Autoformalization ceiling**: if the compiler cannot translate NL into correct programs, tools are useless. AlphaProof failed both combinatorics problems [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/]. Mitigation: grammar-constrained decoding, repair loops, library learning.
3. **Contamination makes the win fake**: retrieval systems copy verbatim [VERIFIED: https://arxiv.org/html/2112.04426], and paraphrase leaks evade n-gram filters [VERIFIED: https://arxiv.org/pdf/2311.04850v1]. Mitigation: a three-layer scrub, retrieval-audit logs, and a private post-cutoff test set as the headline.
4. **Recurrent recall weakness**: a fixed state loses exact facts from long retrieved context. Mitigation: fast-weight memory, re-retrieval, and short chunks.
5. **TPU availability and JAX port cost**: the T4-only fallback is 10x slower. Triton on sm_75 needs a pinned toolchain.
6. **CPU-bound self-play**: S5 may stall at ~4 vCPU. Mitigation: vectorized evaluators, laptop CPU, caching.
7. **Verifier gaming in RLVR**: programs that "pass" weak self-tests. Mitigation: held-out property tests, never using benchmark hidden tests, and adversarial test generation.
8. **Licensing/ToS drift**: teacher licenses, BY-SA datastore content, and Korean "unknown" licenses. Mitigation: per-sample provenance manifest and the no-teacher track.
9. **Protocol variance**: GPT-3.5 targets differ by 20 points across sources (GSM8K 57.1 vs 78.1). We gate on the max, so a small protocol mismatch can flip a row.
10. **Biosafety** on the north-star path: a hard stop, governed as in section 9, never traded off for capability.

---

*Summary of the bet:* procedures in weights, facts in the store, truth from executors, fluency borrowed from humans. COMPILER-0 has a credible path to beating GPT-3.5 on roughly 8–10 of ~20 rows at 91.5M params, on zero budget. It has **no demonstrated path** on the commonsense-continuation and open-chat rows at that size, and its first job is to measure, within 12 weeks, exactly how far off it is.
