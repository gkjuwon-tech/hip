# Externalizing Knowledge and Reasoning: Research Report ("externalize")

Scope: what parts of "intelligence" can move out of a <100M-parameter core into (a) a datastore, (b) tools, solvers and verifiers, (c) test-time search, and (d) self-generated training data. How much each one substitutes for parameters, where it breaks, and what fits on Kaggle free tier or a 6 GB laptop.

Labels: [VERIFIED: url] means checked against the primary source in this session. [ESTIMATE] means my arithmetic or judgment. [HYPOTHESIS] means untested.

---

## 0. The target: GPT-3.5 numbers to beat

From the GPT-4 Technical Report, Table 2 [VERIFIED: https://cdn.openai.com/papers/gpt-4.pdf]:

| Benchmark | GPT-3.5 |
|---|---|
| MMLU (5-shot) | 70.0% |
| HellaSwag (10-shot) | 85.5% |
| ARC-Challenge (25-shot) | 85.2% |
| WinoGrande (5-shot) | 81.6% |
| HumanEval (0-shot) | 48.1% |
| DROP (F1, 3-shot) | 64.1 |
| GSM8K (5-shot CoT) | 57.1% |

Note: GPT-3.5 is not uniformly strong. Its math (57.1%) and code (48.1%) scores are well within reach of tool-augmented systems. MMLU, HellaSwag, WinoGrande and open chat are the hard walls, because they test broad parametric world-modeling and fluency, not lookup.

---

## 1. Retrieval: how much a datastore substitutes for parameters

### 1.1 Primary evidence

| System | Core size | Datastore | Result | Source |
|---|---|---|---|---|
| RETRO | 7.5B | 2T tokens | Comparable to GPT-3 and Jurassic-1 on the Pile with about **25x fewer params**. Beats Jurassic-1 (178B) on most Pile subsets but **not dm_mathematics or ubuntu_irc**. | [VERIFIED: https://arxiv.org/abs/2112.04426], [VERIFIED: https://arxiv.org/html/2112.04426] |
| RETRO (scaling) | 172M to 7.5B | 2T | Retrieval gain is about **10x parametric size** on several datasets | [VERIFIED: https://arxiv.org/html/2112.04426] |
| RETRO on NQ | 7.5B | Wikipedia + DPR | **45.5 EM vs 30.4** for the 7B closed-book baseline | [VERIFIED: https://arxiv.org/html/2112.04426] |
| Atlas | 11B | Wikipedia + CC | **>42% NQ with 64 examples**, beats a 540B model by 3 points with **50x fewer params** | [VERIFIED: https://arxiv.org/abs/2208.03299] |
| kNN-LM | LM + token-level kNN | WT-103 train | PPL **15.79**, **-2.9** with no extra training | [VERIFIED: https://arxiv.org/abs/1911.00172] |
| MassiveDS | Llama-2 7B | **1.4T tokens** | TriviaQA 64.1 to **77.0**, NQ 26.6 to **34.6**, MMLU 45.8 to **49.3**, MedQA 36.6 to **39.4**. The 7B model beats 13B LM-only on PPL, TQA and NQ. **It does not beat the larger model on MMLU or MedQA.** No saturation observed in the range tested. | [VERIFIED: https://arxiv.org/html/2407.12854] |
| CoG ("Copy is All You Need") | phrase-copy LM | phrase index | Better generation quality than token LMs on WT-103. Improves by **swapping or enlarging the index with no retraining**. | [VERIFIED: https://arxiv.org/abs/2307.06962] |
| Self-RAG | 7B/13B | Wikipedia | Beats ChatGPT and retrieval-augmented Llama2-chat on open QA, reasoning and fact verification. Better citation accuracy. | [VERIFIED: https://arxiv.org/abs/2310.11511] |

### 1.2 Takeaways

1. **Knowledge-lookup tasks** (NQ, TriviaQA, factual PPL): retrieval is worth about **10x to 50x parameters** [VERIFIED above].
2. **Reasoning-heavy knowledge tasks** (MMLU, MedQA): gains are only about **+3 to 4 points**, and the smaller model **does not** overtake the larger one [VERIFIED: https://arxiv.org/html/2407.12854]. The MassiveDS authors attribute this partly to missing textbook and biomedical sources. That points to a lever: **datastore composition**, i.e. textbooks, worked solutions and Q&A-style data rather than raw web.
3. **Leakage caveat**: RETRO exploits train/test overlap more strongly than the baseline; the authors attribute only a "modest fraction" of the gain to it [VERIFIED: https://arxiv.org/html/2112.04426]. Our fair-play rules must decontaminate the datastore against every eval set (13-gram overlap filtering is the common practice [ESTIMATE]).
4. **Scaling extrapolation to <100M** [ESTIMATE]: no paper shows a <100M retrieval LM at GPT-3.5 level on any broad benchmark. RETRO 172M is the smallest data point, with gains equal to about 10x params. So a 100M core plus retrieval plausibly behaves like about a 1B LM-only model on knowledge tasks. That is far below GPT-3.5 on MMLU. **Retrieval alone cannot win MMLU at 0.1B.** MMLU needs retrieval plus a reasoning/reading module plus a datastore of exam-style explanations [HYPOTHESIS].

### 1.3 Parameter accounting trap

MassiveDS uses Contriever-MSMARCO, which has **177M params** [VERIFIED: https://arxiv.org/html/2407.12854]. Under our rules that **alone breaks the 100M budget**. Options that fit the budget:

| Retriever | Params | Quality | Source |
|---|---|---|---|
| BM25 (bm25s) | **0** | Lexical. Up to 500x faster than the most popular Python BM25. NumPy/SciPy only. | [VERIFIED: https://arxiv.org/abs/2407.03618] |
| potion-base-8M (model2vec, static) | **7.56M** | MTEB 51.32, **92% of MiniLM-L6** | [VERIFIED: https://huggingface.co/minishlab/potion-base-8M] |
| all-MiniLM-L6-v2 | **22.7M**, 384-d, 256 tok | MTEB about 55.8 (avg-all, per potion card) | [VERIFIED: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2] |
| bge-small-en-v1.5 | **33.4M**, 384-d | MTEB 62.17 (56 datasets) | [VERIFIED: https://huggingface.co/BAAI/bge-small-en-v1.5] |
| multilingual-e5-small | 384-d | C-MTEB avg 55.38 (listed on bge card). Param count not verified here. | [VERIFIED: https://huggingface.co/BAAI/bge-small-en-v1.5] |

**Recommendation** [ESTIMATE]: BM25 (0 params) plus a static embedding of about 8M params, fused by reciprocal rank fusion. Add a tiny cross-encoder reranker (<10M) only if the budget allows. **Share the core's own encoder** as the dense retriever (RETRO-style frozen keys, or Atlas-style joint training) so that retrieval costs no extra params. Korean BM25 needs a non-neural morphological tokenizer or character n-grams; rule-based tokenizers count as 0 params [ESTIMATE].

---

## 2. Index engineering on zero budget

Kaggle quota (not verified from official docs this session): about **30 GPU-h/week** (sometimes higher, "floating"). Sessions are 9h (tips page) or 12h (forum, quoting docs). TPU is a separate quota, historically **20h/week, 9h/session**. Current TPU type is uncertain: v3-8 historically, with third-party reports of v5e-8 [VERIFIED (non-official/older): https://www.kaggle.com/docs/efficient-gpu-usage, https://www.kaggle.com/product-feedback/173129, https://www.kaggle.com/docs/tpu, https://aimultiple.com/free-cloud-gpu]. The builder must check the live notebook settings page. Kaggle RAM (about 29-30 GB) and disk (about 20 GB in /kaggle/working) are [ESTIMATE, unverified].

**Embedding compression** (mxbai-large, 1024-d, 250M vectors) [VERIFIED: https://huggingface.co/blog/embedding-quantization]:
- int8: **4x** smaller, 97% of fp32 NDCG (about 99% with rescoring), 3.66x faster.
- binary: **32x** smaller, **92.5% raw / 96.45% with rescoring**, **24.76x** mean speedup. 250M vectors take **29.8 GB** binary vs 953.67 GB fp32.

**Sizing for our stack** [ESTIMATE, own arithmetic]:
- English Wikipedia is about **4B tokens** [VERIFIED: https://arxiv.org/html/2112.04426]. RETRO's own Wikipedia index is about 215 GB. A token-level kNN-LM store would be about 15 TB, so **token-level kNN-LM is infeasible; use chunk-level** [VERIFIED same source].
- 4B tokens in 128-token chunks gives about 31M chunks.
  - 384-d fp32: 31M × 1536 B ≈ **48 GB** (does not fit).
  - Binary 384-bit: 31M × 48 B ≈ **1.5 GB** (fits in RAM, on a laptop too).
  - PQ at 64 B/vector: ≈ 2 GB.
  - BM25 sparse index ≈ 5-15 GB [ESTIMATE]. Raw text ≈ 16 GB, about 5 GB zstd-compressed.
- A 50-100B-token datastore (FineWeb-Edu subset + Wikipedia + arXiv + StackExchange + Korean Wiki/Namu-like CC sources + code) has about 0.4-0.8B chunks. Binary that is about **20-40 GB**, which fits on a laptop SSD with mmap and CPU FAISS `IndexBinaryIVF` plus int8 rescoring. Encoding 0.8B chunks with an 8M static model on CPU is feasible in days. A 33M transformer encoder on 2x T4 would need about weeks of quota [ESTIMATE]. Static embeddings plus BM25 are the only route that scales the datastore on 0 KRW.
- MassiveDS ran exact `IndexFlatIP` [VERIFIED: https://arxiv.org/html/2407.12854]. We must use IVF + PQ/binary, and accept roughly a 3-8% recall loss [ESTIMATE, consistent with the 92.5-97% figures above].

---

## 3. Tools, code execution and symbolic solvers: externalizing computation

| Evidence | Result | Source |
|---|---|---|
| PAL (program-aided) | Codex + Python beats PaLM-540B CoT on GSM8K by **15 absolute points** | [VERIFIED: https://arxiv.org/abs/2211.10435] |
| Toolformer | Self-supervised API calls (calculator, QA, 2 search engines, MT, calendar). "Often competitive with much larger models." | [VERIFIED: https://arxiv.org/abs/2302.04761] |
| AlphaGeometry | **25/30** IMO geometry problems vs 10 for the prior SOTA (Wu's method) and **25.9** for the average gold medalist. Trained on **100M synthetic examples**, no human demos. | [VERIFIED: https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/] |
| AlphaProof + AG2 (IMO 2024) | **28/42 (silver)**, 4/6 problems, in Lean. Some problems took **up to 3 days**. Trained by proving or disproving millions of problems. Failed both combinatorics problems. | [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/] |

**Takeaways:**
- Arithmetic, algebra and formal logic can be offloaded almost entirely. The core only has to **translate the problem into a program or formal statement**. GSM8K 57.1% is the GPT-3.5 bar. A small seq2program model plus a Python sandbox plus self-consistency is the cheapest win on this list [HYPOTHESIS, strongly suggested by PAL].
- Recommended solver stack: Python, SymPy, Z3 and Lean (all 0 params, CPU). Add a deductive-database geometry engine on the AlphaGeometry pattern.
- **Wall:** combinatorics and open-ended problems resist formalization (AlphaProof failed both [VERIFIED above]). Autoformalization quality limits everything downstream, and it is itself a language-understanding task the tiny core must do.

---

## 4. Test-time compute: substituting FLOPs at inference for parameters

| Evidence | Result | Source |
|---|---|---|
| Snell et al. 2024 (PaLM 2-S*, MATH) | Compute-optimal allocation is **>4x more efficient than best-of-N**. FLOPs-matched, it **beats a 14x larger model**, but only where the base model has "somewhat non-trivial" success. On the hardest questions there is "very little benefit"; pretraining is better there and when the inference/pretrain token ratio R >> 1. | [VERIFIED: https://arxiv.org/abs/2408.03314], [VERIFIED: https://arxiv.org/html/2408.03314] |
| Liu et al. 2025 ("Can 1B surpass 405B?") | With compute-optimal TTS: **1B > 405B on MATH-500**, **0.5B > GPT-4o** on MATH-500 and AIME24, 3B > 405B, 7B > o1 and DeepSeek-R1 | [VERIFIED: https://arxiv.org/abs/2502.06703] |
| Let's Verify Step by Step | A PRM trained on **800K step labels** solves **78%** of a MATH subset. Process supervision beats outcome supervision. | [VERIFIED: https://arxiv.org/abs/2305.20050] |

**Takeaways:**
- In verifiable domains (math, code with tests, logic, formal proof), a sub-1B policy plus a verifier plus search has **already beaten GPT-4o-class models** [VERIFIED: https://arxiv.org/abs/2502.06703]. These results use a PRM that is often **larger than the policy**. Under our rules the PRM counts toward the 100M. Use the core as both policy and verifier (shared weights), or use **exact verifiers** (unit tests, SymPy, Lean, Z3), which cost 0 params [ESTIMATE].
- **Wall:** test-time compute needs a base success rate above zero. The 0.1B core must reach a usable pass@k before search helps. **Wall 2:** no exact verifier exists for open chat, commonsense or HellaSwag-style continuation. Only learned verifiers apply there, and they cost params [ESTIMATE].
- **Scaling law to exploit**: test-time compute grows with hardware, not params. That makes it a scaling axis for the north star [VERIFIED for the trend: https://arxiv.org/abs/2408.03314].

---

## 5. Self-improvement: externalizing data creation

| Evidence | Result | Source |
|---|---|---|
| STaR (GPT-J 6B) | CommonsenseQA **72.5%** vs 73.0% for fine-tuned GPT-3 (30x larger) vs 60.0% for fine-tuned GPT-J. GSM8K only **10.7%**. | [VERIFIED: https://arxiv.org/html/2203.14465] |
| DeepSeek-R1-Zero (pure RL, GRPO) | AIME24 pass@1 **15.6% to 77.9%**, **86.7%** with self-consistency | [VERIFIED: https://arxiv.org/html/2501.12948] |
| R1-Distill-Qwen-1.5B | AIME24 **28.9** vs 9.3 for GPT-4o, MATH-500 **83.9** vs 74.6. But **GPQA 33.8 vs 49.9** and **LiveCodeBench 16.9 vs 32.9**, i.e. it **loses** outside math. Distillation beats RL run directly on small models. | [VERIFIED: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B] |
| TinyStories | **<10M** models produce fluent multi-paragraph English **in a narrow-vocabulary domain**. 125M GPT-2 is incoherent on general text after heavy training. | [VERIFIED: https://arxiv.org/abs/2305.07759] |

**Takeaways:**
- The 1.5B distill **beats GPT-4o on math and loses on knowledge (GPQA) and code** [VERIFIED above]. This is the clearest real-world demonstration that a tiny reasoner is lopsided. Knowledge has to come from retrieval, and code from execution plus tests.
- RL with verifiable rewards (GRPO) is compute-cheap per step and needs no human labels. It is the right self-improvement loop for Kaggle. But direct RL on small models underperforms distillation [VERIFIED above]. **Fair-play issue:** distilling from a larger open model's outputs at *training* time is permitted under rules that only ban >100M nets at inference. That is a policy decision the rules section must state explicitly. Recommended: allow open-licensed teacher data at training, log it, and also report a "no-teacher" track [ESTIMATE].
- TinyStories shows that **data curriculum substitutes for params on fluency**, but only in a narrow domain. General open-domain fluency at <100M is unproven, and it is the biggest single risk for the "chat / instruction following" category [HYPOTHESIS: shrink the generator's job to composing and copying retrieved spans (CoG-style) and render through templates, so fluency comes from the datastore].

---

## 6. Substitution ledger (what each axis buys)

| Axis | Substitutes for params on | Measured multiplier | Fails on |
|---|---|---|---|
| Retrieval (chunk/phrase) | Facts, QA, PPL, citations | **10x to 50x** (RETRO, Atlas) | MMLU/MedQA reasoning (+3-4 pts only), math |
| Code/solver tools | Arithmetic, algebra, formal logic, geometry | PAL **+15 pts** vs a model about 3x larger; AG matches gold medalists | Combinatorics, informal problems, autoformalization |
| Test-time search + verifier | Verifiable math/code | **14x** FLOPs-matched; 0.5B > GPT-4o on MATH | Hardest problems; anything without a verifier |
| Self-generated data (STaR/RLVR) | Reasoning style, rationales | **30x** (STaR CQA) | Knowledge breadth; small-model RL < distillation |
| Curriculum data | Fluency | <10M fluent in TinyStories | Open-domain breadth (unproven) |

**Domain-by-domain feasibility vs GPT-3.5 at <100M** [ESTIMATE]:
- **GSM8K 57.1 / HumanEval 48.1**: plausible with seq2program + execution + self-consistency + RLVR. MBPP/HumanEval need code retrieval (a copy-from-corpus approach, decontaminated).
- **NQ/TriviaQA, DROP 64.1**: plausible with retrieval plus a reader. DROP arithmetic goes to the tool.
- **MMLU 70.0**: hard. Needs an exam-style explanation datastore plus retrieve-then-reason plus option-elimination search. No published evidence at this scale.
- **HellaSwag 85.5 / WinoGrande 81.6**: hardest. They test implicit world-model plausibility, and retrieval helps little [HYPOTHESIS]. Possible route: kNN over situation/event corpora (ATOMIC-like), plus learned plausibility scoring trained with contrastive negatives.
- **Open chat / instruction following / Korean**: highest risk. No exact verifier exists, and generator fluency is capped by params.

---

## 7. Architecture implications (non-Transformer, externalized, scalable)

[HYPOTHESIS, the design proposal for the blueprint]

1. **Core (≤100M total, every weight counted)**: a recurrent/SSM-style backbone (e.g. a Mamba-like or linear-RNN mixer; no self-attention blocks), with **cross-reading over retrieved chunks via chunked cross-memory**. Note that RETRO's chunked cross-attention is attention; a non-attention alternative is to recurrently scan retrieved chunks, or use kNN-memory slots. One shared trunk serves as encoder (retrieval keys), policy (generator/program writer) and verifier/value head.
2. **Memory**: BM25 + static/shared dense binary index over 50-100B decontaminated tokens. Phrase-level copy table (CoG) so that surface fluency is borrowed from text. Per-language shards (KO/EN/code).
3. **Executors**: Python sandbox, SymPy, Z3, Lean 4, unit-test runner. All 0 params, CPU.
4. **Controller**: adaptive test-time compute, following Snell's compute-optimal rule: easy prompts get one pass, medium get search, very hard get abstain/refer [VERIFIED rationale: https://arxiv.org/html/2408.03314].
5. **Learning loop**: RLVR (GRPO) on verifiable tasks. STaR-style rationalization. Synthetic problem generation on the AlphaGeometry pattern (symbolic engine generates problems and solutions for free).

**What scales toward the north star** (each axis is independent):
- **Params** (0.1B to 10B): better translation into programs/queries, higher base pass@k, which multiplies search value.
- **Datastore** (tokens): no saturation up to 1.4T [VERIFIED: https://arxiv.org/abs/2407.12854]. Updatable with no retraining [VERIFIED: https://arxiv.org/abs/2307.06962].
- **Test-time compute**: 4x efficiency from allocation alone [VERIFIED: https://arxiv.org/abs/2408.03314].
- **Tools**: adding a solver (e.g. docking/MD simulators for the drug north star) adds capability with no new params.
- **Self-generated data**: grows with compute and verifier coverage.

Scientific discovery is exactly this regime: a verifier-rich environment (simulators, experiments, proofs). That is why this design lines up with the north star [HYPOTHESIS].

---

## 8. Fair-play rules specific to externalization

- **Counted params**: core + any encoder, reranker, embedding table (including static model2vec tables), PRM and router. BM25 statistics, FAISS centroids/PQ codebooks and tokenizers are counted as **data, not params**, and are disclosed with their sizes [ESTIMATE, proposed rule]. Note that PQ codebooks are learned, so they must be reported; they are kilobytes to megabytes.
- **Allowed at inference**: frozen local datastore snapshot (dated, hashed), offline Python/SymPy/Z3/Lean. No live web, so results are reproducible. Live web search could be a separate disclosed track.
- **Banned**: any neural net >100M params at inference (including hosted APIs), and any eval-set text in the datastore. Decontaminate with n-gram overlap against all benchmark test sets and publish the overlap report (RETRO's leakage analysis is the template [VERIFIED: https://arxiv.org/html/2112.04426]). Benchmark-specific hand rules are banned.
- **Report** wall-clock and FLOPs per query. Test-time-compute wins must be FLOPs-disclosed, as in Snell et al.

---

## 9. Biosafety note for the drug-discovery north star

Externalized tools (synthesis planners, toxicity predictors) are dual-use. The datastore and tool registry must exclude pathogen enhancement and toxin-synthesis content. The controller needs a refusal policy for hazardous chemistry and biology. Any wet-lab-adjacent outputs require human expert review. This is a hard constraint, not an optimization target [ESTIMATE/policy].

---

## 10. Unverified items to check next
- Official current Kaggle GPU/TPU quotas, RAM and disk (the docs page returned 404 or no content this session).
- GPT-3.5 Korean benchmark scores (e.g. KMMLU). None verified here.
- The smallest multilingual embedding model's param count and Korean retrieval quality.
- Any published <100M retrieval-augmented model scoring above 50% on MMLU (none found; assumed none).
