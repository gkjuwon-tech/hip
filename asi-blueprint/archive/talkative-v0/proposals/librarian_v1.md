# LIBRARIAN v1: A Retrieval-Native Generator

*Chief-architect proposal, philosophy: "the weights are the librarian, not the library."* Date: 2026-10-19.
Labels: [VERIFIED: url] = figure from a primary source in the research digest; [ESTIMATE] = my arithmetic or judgement; [HYPOTHESIS] = untested design claim.

---

## 1. Thesis

A dense LLM spends most of its parameters as a **lossy compressed copy of its training corpus**. Physics-of-LMs puts the ceiling at about 2 bits of knowledge per parameter [VERIFIED: https://arxiv.org/html/2404.05405]. A 100M network can therefore hold about 25 MB of facts [ESTIMATE], and the evidence matches: SmolLM2-135M gets 4.1% on TriviaQA after 2T tokens [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct]. The big model's other parameters buy three things: (a) **procedures** (read, compare, decompose, compute), (b) **surface fluency**, which is mostly memorized phrasing, and (c) **calibrated plausibility**, a sense of what usually happens next in the world.

The first-principles bet is that (a) needs few parameters, while (b) and (c) are **statistics of text**. Statistics of text can be stored as text plus an index, which costs disk instead of parameters. Disk is free; we have about 100 GB of Hugging Face storage plus a laptop SSD.

So LIBRARIAN keeps **all knowledge, most phrasing and much of the plausibility signal in a compressed public datastore**. A small attention-free recurrent core (Gated DeltaNet) acts as glue: it writes queries, reads evidence by scanning it, copies spans, calls tools and checks its own work. Knowledge then scales with **datastore tokens**, which RETRO showed still helping at 2T tokens, with about 25x parameter savings [VERIFIED: https://arxiv.org/abs/2112.04426v1]. Quality never depends on cramming.

I also state up front what this philosophy **cannot** do on its own (§5). It is the knowledge-axis champion, not a complete answer to the mid-boss.

---

## 2. Fair-play rules

1. **Params** = every trainable weight executed at inference: token embeddings, the static retrieval-key table, the core, every head and adapter, and the meta-learned initialization of the test-time memory. **Cap: < 100,000,000.** Tokenizer merges, BM25 statistics, FAISS IVF centroids and PQ codebooks count as *data*. They are disclosed with their sizes because PQ codebooks are learned (kilobytes to megabytes).
2. **Datastore:** a frozen, dated, hashed offline snapshot of public text and code. It is excluded from params but fully disclosed: sources, token count, build script, licence manifest.
3. **Two datastore tracks**, which close a loophole that matters a lot for this philosophy:
   - **CLEAN (headline, gating).** The datastore holds only human-written text (Wikipedia, FineWeb-Edu/FineWeb-2 web text, OpenWebMath, arXiv, permissive code). No model-generated text is in the store. Otherwise "retrieval" could mean retrieving a cached LLM answer to everything.
   - **OPEN (reported).** Up to 10% of store tokens may be synthetic text from open-licensed models (e.g. Cosmopedia, Apache-2.0 [VERIFIED: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia]). The text must be generated without seeing eval prompts, and the fraction is disclosed.
4. **Allowed at inference:** Python sandbox, SymPy, Z3, unit-test runner on *self-written* tests, deterministic solvers. All are CPU-only with 0 params.
5. **Banned:** any neural net > 100M, any LLM API, live web, eval items or their solutions in the store or the training data, dataset-identity routing, and running hidden test cases during generation.
6. **Decontamination:** 13-gram plus MinHash near-duplicate filtering of the store and training data against every eval set and its mirrors. Add an embedding scrub at cosine > 0.9 for paraphrases, since n-grams miss rephrased test sets [VERIFIED: https://arxiv.org/pdf/2311.04850v1], and a URL blocklist of benchmark-hosting sites. **Retrieval-audit log:** for every eval item, publish the top-k passages and the percentage containing the gold string verbatim. RETRO "exploit[s] leakage more strongly" [VERIFIED: https://arxiv.org/abs/2112.04426], so for this design the audit is mandatory, not optional.
7. **Architecture rule:** no softmax attention over the token sequence at any layer. One disclosed gray-zone item: the copy head scores at most 64 candidate phrases taken from retrieved chunks, as in "softmax over external items, K ≤ 64" from the archs survey.
8. **Training-time teachers:** open-licensed models only, logged with name and size. **No GPT-3.5/4 outputs**, because of the OpenAI ToS "compete" clause [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/]; this excludes TinyGSM, MetaMathQA, UltraChat and TinyStories from training. A no-teacher ablation track is also reported.

---

## 3. Architecture

### 3.1 Component diagram

```
                         ┌────────────────────────── LIBRARY (disk, 0 params) ───────────────────────────┐
                         │  Text shards (zstd)   EN/KO/CODE/MATH   ~20B tok (v0) → 100B+ (v1)            │
                         │  CATALOG: BM25 (bm25s) + binary 256-bit keys in IVF + PQ-32B rescoring          │
                         │  PHRASE TABLE: n-gram→chunk postings for copy candidates + corpus counts       │
                         └───────────▲───────────────────────────────────▲─────────────────────────────────┘
                                     │ top-k chunk ids                    │ phrase candidates / counts
 user ──► TOKENIZER (32k, ko-aware) ─┼──► CLERK (static key enc.) ──────┘
                │                    │         ▲ query vector
                ▼                    │         │
        ┌───────────────────────── SCRIBE: Gated DeltaNet core, d=576, no attention ─────────────────────┐
        │ PRELUDE (4 layers) ─► RECURRENT BLOCK (4 layers, weight-tied, iterate r=1..16) ─► CODA (4 layers) │
        │      ▲  reads [query | retrieved chunks | tool outputs] by scanning them as one stream           │
        │      │  SCRATCH: test-time-trained MLP memory (per-episode delta updates)                        │
        └──────┼───────────────┬───────────────────────┬──────────────────────┬─────────────────────────┘
               │               ▼                       ▼                      ▼
               │        LM head (tied)         COPYIST head (phrase       DISPATCH head (SEARCH / READ /
               │        free tokens            pointer over ≤64 cands)    PY / SOLVE / CHECK / ANSWER / r)
               │               └──────────┬────────────┘                      │
               │                          ▼                                   ▼
               │                     candidate drafts ◄──── WORKBENCH: Python, SymPy, Z3, test runner
               │                          ▼
               └──────────────── CRITIC (low-rank adapter on SCRIBE + value head) → rank / accept / retry
```

### 3.2 Inference dataflow, step by step

1. **Tokenize** with a 32,768-entry byte-fallback BPE trained on a 45/35/20 EN/KO/code mix, so Korean is not shredded into bytes.
2. **CLERK** builds the query vector. It mean-pools a static 256-d table over the tokens (model2vec-style; such tables reach about 92% of MiniLM quality at 7.6M params [VERIFIED: https://huggingface.co/minishlab/potion-base-8M]). In parallel, BM25 runs on the raw query, with Korean handled by character bigrams (0 params).
3. **CATALOG** search: Hamming search over binary keys in IVF, then PQ rescoring. Binary keeps 92.5% of retrieval quality raw and 96.45% with rescoring [VERIFIED: https://huggingface.co/blog/embedding-quantization]. Dense and BM25 lists are fused by reciprocal rank to the top 32.
4. **PRELUDE rerank:** the first 4 core layers encode query and candidates, a small head scores them, and the top K = 8 chunks (256 tokens each) are kept.
5. **READ:** the stream `[task | chunk_1 … chunk_8 | task again]` is scanned. The **task is repeated after the evidence**, the standard trick for recurrent models whose fixed state can't look backwards. The delta rule overwrites slots selectively, which is exactly what Gated DeltaNet improves for in-context recall [VERIFIED: https://arxiv.org/abs/2412.06464v3]. SCRATCH writes key facts into a fast MLP memory by gradient steps, Titans-style [VERIFIED: https://arxiv.org/abs/2501.00663], so they survive the long scan.
6. **THINK:** the recurrent block iterates r times. DISPATCH picks r ∈ {1, 2, 4, 8, 16} and can emit a tool call (SEARCH again with a rewritten query, PY, SOLVE, CHECK). Tool output is spliced into the stream and scanned.
7. **WRITE:** at every step the decoder mixes p_LM(token) with p_COPY(phrase). COPYIST proposes up to 64 phrases (1–16 tokens) drawn from the retrieved chunks and the phrase table, scored as a dot product against the core state. This follows "Copy is All You Need", which improves when the index is swapped or enlarged with no retraining [VERIFIED: https://arxiv.org/abs/2307.06962]. Fluency is borrowed from human text.
8. **CHECK:** generate N drafts (N = 1 easy, 4–16 medium, 32 hard). Exact verifiers run where they exist (execution, SymPy, self-written tests). CRITIC scores the rest. The best draft, or a majority-voted answer, is returned together with **citations to chunk ids**.

### 3.3 Exact parameter budget

Gated DeltaNet layer at d = 576, 8 heads: q, k, v, o and output-gate projections are 5d² = 1,658,880. The α/β gate projections are 2·576·8 = 9,216. Short convs (k = 4) on q, k, v are 6,912. Two RMSNorms are 1,152. That gives 1,676,160 for the mixer. The SwiGLU FFN with hidden 1,536 is 3·576·1,536 = 2,654,208. **Per layer: 4,330,368.**

| Module | Shape | Trainable params |
|---|---|---|
| Token embedding (tied with LM head) | 32,768 × 576 | 18,874,368 |
| SCRIBE prelude | 4 unique GDN layers | 17,321,472 |
| SCRIBE recurrent block (weight-tied, iterated) | 4 unique GDN layers | 17,321,472 |
| Recurrence input injection (concat state + prelude out → d) | 1,152 × 576 + 576 | 664,128 |
| SCRIBE coda | 4 unique GDN layers | 17,321,472 |
| Final norm | 576 | 576 |
| CLERK static key table | 32,768 × 256 | 8,388,608 |
| Rerank head (prelude pooled → 256, bilinear) | 576×256 + 256×256 | 212,992 |
| COPYIST (phrase start/end encoders + query proj) | 3 × 576 × 576 | 995,328 |
| Phrase-length embedding (1–16) + source-type embedding (8) | 24 × 576 | 13,824 |
| DISPATCH head (12 actions + 5 r-levels) | 576 × 17 + 17 | 9,809 |
| SCRATCH test-time memory init (576→1,152→576 MLP) + LR/decay gates | 1,327,680 + 1,154 | 1,328,834 |
| CRITIC low-rank adapters (rank 32, on q, v, o, FFN-down of all 12 layers) | 12 × 4 × 2 × 576 × 32 | 1,769,472 |
| CRITIC value head (576→576→1) | 331,776 + 576 + 577 | 332,929 |
| Retrieval-provenance / segment embeddings (task, chunk, tool, answer) | 4 × 576 | 2,304 |
| **Total** | | **84,557,588** |
| Reserve (Korean vocab growth, rerank cross-scorer) | | 15,442,412 (unallocated; total must stay < 100,000,000) |

**84.56M counted params**, all trainable and all executed at inference. Unique-layer depth is 12. Effective depth is 4 + 4r + 4, which is 72 layers at r = 16. Test-time compute buys depth without parameters, as in Huginn [VERIFIED: https://arxiv.org/abs/2502.05171], here transplanted onto an attention-free block.

---

## 4. Why a datastore can stand in for parameters, quantified

| Lever | Evidence | What it buys LIBRARIAN |
|---|---|---|
| Chunk retrieval | RETRO: about 25x fewer params at GPT-3-level Pile perplexity [VERIFIED: https://arxiv.org/abs/2112.04426v1]; NQ 45.5 vs 30.4 closed-book [VERIFIED: https://arxiv.org/html/2112.04426] | Factual QA |
| Small reader + retriever | FiD-base: 48.2 EM on NQ with a 220M reader [VERIFIED: https://arxiv.org/abs/2007.01282] | Shows a sub-0.5B system beats GPT-3.5's 38.6 |
| Datastore scaling | MassiveDS: no saturation up to 1.4T tokens; MMLU gains only +3.5 [VERIFIED: https://arxiv.org/html/2407.12854] | Knowledge grows with disk; reasoning does not |
| Phrase copying | CoG: better generation quality than token LMs [VERIFIED: https://arxiv.org/abs/2307.06962] | Fluency from text, not weights |
| Exec + verifier | TinyGSM 125M: 63.1 GSM8K; verifier "plays a bigger role" [VERIFIED: https://arxiv.org/html/2312.09241] | Math and code |

---

## 5. Scoreboard: honest per-row forecast

Targets are the max-reported GPT-3.5 values from the baseline digest. Forecasts are for the CLEAN track at 84.6M params after the §6 recipe. **All forecasts are [ESTIMATE].** Confidence is the probability of beating the target plus its CI.

| Row | Target | LIBRARIAN v1 forecast | P(win) | Why |
|---|---|---|---|---|
| NQ-open | 38.6 | 42–48 | 0.70 | Retrieval-native home turf; FiD did 48.2 at 440M. Risk: recall@8 of binary keys. |
| TriviaQA 5-shot | 85.8 | 68–78 | 0.10 | FiD-base got 65.0 EM [VERIFIED: https://arxiv.org/abs/2007.01282]; GPT-3.5's 85.8 is very high. Honest loss. |
| MMLU | 71.4 | 48–58 | <0.05 | Atlas-770M reached only 56.3 [VERIFIED: https://arxiv.org/abs/2208.03299]; MMLU needs reasoning over knowledge. Loss. |
| KMMLU | 42.47 | 38–47 | 0.40 | Low bar (random is 25), and ko-Wikipedia plus FineWeb-2-ko store. Coin flip. |
| HAE-RAE (LW/RW/SN/GK/HI) | 49.45 mean | 50–62 | 0.55 | Loanwords, rare words and nomenclature are dictionary lookups; history and general knowledge are retrieval. |
| KoBEST avg | 79.3 | 60–72 | 0.10 | BoolQ/SentiNeg need reading quality; HellaSwag-ko is fluency. |
| GSM8K 0-shot | 78.1 (57.1 5-shot) | 62–74 | 0.15 vs 78.1; 0.85 vs 57.1 | Program-of-thought + exec + self-consistency + CRITIC; TinyGSM's 76.1 needed a 1.3B verifier. |
| MATH | 34.1 | 22–36 | 0.30 | SymPy offload helps algebra; geometry and counting fail on formalization. |
| HumanEval+ / HumanEval | 70.7 / 76.8 | 30–48 | <0.05 | phi-1-small got 45 at 350M [VERIFIED: https://arxiv.org/abs/2306.11644]. Code retrieval + self-tests + repair help, but the target is EvalPlus's best snapshot. Loss. |
| MBPP+ / MBPP | 69.7 / 82.5 | 40–55 | <0.05 | Same reasoning. Loss. |
| BBH | 68.3 | 48–60 | 0.10 | Dyck, boolean and sorting tasks are executable; the language-heavy tasks are not. |
| DROP F1 | 64.1 | 60–72 | 0.50 | Span copy is literally COPYIST's job; arithmetic goes to PY. Strongest reading row. |
| ARC-Challenge | 87.4 | 55–68 | <0.05 | Textbook retrieval helps, but distractor science reasoning remains. Loss. |
| HellaSwag | 85.5 | 45–58 | <0.02 | See §5.1. Loss. |
| WinoGrande | 81.6 | 52–58 | <0.02 | Adversarially built so co-occurrence statistics cancel; retrieval has no key. Loss. |
| ARC-AGI-1 pass@2 | 5.0 | 6–15 | 0.60 | Python program search plus test-time training of SCRATCH; TRM shows tiny models can (44.6) [VERIFIED: https://arxiv.org/html/2510.04871v1]. |
| MT-Bench | 8.39 | 3.5–5.0 | <0.01 | See §5.2. Loss. |
| AlpacaEval 2 LC | 22.35 | 3–10 | <0.02 | Loss. |
| Arena-Hard | 24.8 | 1–5 | <0.01 | Loss. |

**Bottom line, stated plainly: LIBRARIAN v1 is forecast to win about 4–6 of roughly 19 gated rows** (NQ, HAE-RAE, DROP, ARC-AGI-1, possibly KMMLU and MATH). P(all Tier A+B rows) is below 1% [ESTIMATE]. Retrieval-native generation is necessary for the mid-boss, because nothing else fixes the knowledge-capacity wall, but **it is not sufficient**. Pure "copy + glue" loses chat, commonsense and code. This proposal is offered as the knowledge and grounding subsystem of the final blueprint, and the forecasts above should be treated as the bar a hybrid must clear.

### 5.1 Commonsense (HellaSwag, WinoGrande): the daring attempt

[HYPOTHESIS] Plausibility is partly a corpus statistic. For each candidate ending, the PHRASE TABLE supplies **corpus support features**: counts of the context's key n-grams followed by the ending's key n-grams, plus the best retrieved analogous narrative. These go to CRITIC, which is trained contrastively on self-generated negatives built from FineWeb-Edu.

- HellaSwag is drawn from ActivityNet/WikiHow, so WikiHow-like how-to text is the right store shard. It must be decontaminated against HellaSwag contexts by 13-gram and embedding scrub, and audited.
- Expected gain: +5–15 points over SmolLM2-135M's 42.1 [ESTIMATE]. That is still 25+ points short.
- WinoGrande is built to defeat exactly this kind of statistic, so I expect it near chance (+2–6).

**Why current tech can't close this:** both tasks reward an implicit world model plus fluent continuation scoring. The only published sub-1.5B jump on WinoGrande came from synthetic textbook data at 1.3B (60.4 → 73.4) [VERIFIED: https://arxiv.org/html/2309.05463]. That is 13x our budget.

### 5.2 Open chat, creative writing, Korean fluency: where tiny models die

- At 125–135M, MT-Bench is 1.98–2.52 [VERIFIED: https://arxiv.org/html/2402.14905, https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct], against 8.39 for GPT-3.5. The smallest model on record that beats GPT-3.5 in Arena is Gemma-2-2B (about 2.6B) [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard].
- LIBRARIAN's attack is **plan → retrieve exemplars → copy-compose → critique**. Fluency is borrowed phrase by phrase from human prose, and every factual sentence is grounded. That should lift factual-explanation prompts (writing, STEM and humanities categories) to roughly 5–6 [ESTIMATE].
- It does nothing for multi-turn instruction tracking, role-play, persuasive or creative writing, or formatting constraints. Copying human phrases does not produce *new* coherent long-form structure. Creative writing is the worst case: copy-heavy text reads as a collage, and judges penalize it.
- Korean is harder still: the open Korean pool is smaller, 35.3% of Korean HF datasets have unknown licences [VERIFIED: https://arxiv.org/pdf/2507.04329], and GPT-3.5 itself only manages Arena-Korean 1049.6. I forecast Korean chat roughly 1 MT-Bench point below English.

**Precise wall:** chat quality needs knowledge, fluency and reasoning *simultaneously at every token*. Copying supplies the first two locally but not the global plan. Getting around it requires a much stronger planner (more of the budget in the recurrent block), on-policy preference learning against a local judge, or more parameters. I do not have evidence that any of these works at 85M.

---

## 6. Training recipe

All token counts are [ESTIMATE]. "RA" means a retrieval-augmented sequence: each training example carries its K retrieved chunks, so every supervised token costs about 3–5 tokens scanned.

| Stage | Goal | Data | Tokens scanned | Objective |
|---|---|---|---|---|
| S0 Tokenizer + CLERK | Vocabulary; static keys | 2B-token EN/KO/code sample | — | BPE; CLERK distilled from the PRELUDE after S1, then **frozen forever** so the index never needs re-encoding |
| S1 Fluency pretrain | Grammar, Korean, code syntax; deliberately NOT facts | FineWeb-Edu (int_score ≥ 4), FineWeb-2-ko, Python-Edu, OpenWebMath: 40/30/15/15 | 20B | Next-token, plus LLM-JEPA embedding loss [VERIFIED effect at fine-tune scale: https://arxiv.org/abs/2509.14252], with r sampled from 1–4 and truncated backprop |
| S2 RA pretrain ("learn to lean") | Use evidence instead of memory | Same corpora, each 1,024-token target paired with 8 retrieved neighbours from a store that **excludes its own document** | 30B (8B targets) | Next-token + copy-head loss on spans present in the neighbours. **Fact-dropout:** 30% of targets have entities masked unless they are retrievable, which punishes memorization |
| S3 Reader/retriever SFT | Query writing, rerank, cite | KILT provenance triples [VERIFIED: https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/], synthetic QA over the store from open teachers | 3B | Rerank InfoNCE, answer cross-entropy, citation accuracy |
| S4 Tool + procedure curriculum | Program-of-thought, SEARCH/PY/SOLVE/CHECK grammar | Reasoning Gym (100+ generators) [VERIFIED: https://arxiv.org/pdf/2505.24760], re-ARC on train split [VERIFIED: https://arxiv.org/pdf/2404.07353], OpenMathInstruct-2 rewritten as Python and kept only if execution agrees [VERIFIED licence CC-BY-4.0: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2], Python-Edu exec traces | 8B | Cross-entropy on calls and result use; adaptive difficulty at about 60% solve rate |
| S5 Chat SFT | Format, multi-turn | SmolTalk (Apache-2.0) and Tulu-3 minus NC subsets, every answer re-grounded with retrieved citations | 1B | Cross-entropy; CRITIC adapter frozen |
| S6 RLVR + self-play | Raise pass@k, train CRITIC | Verifiable tasks (math, code with self-written tests, Reasoning Gym, retrieval QA with exact match); a proposer generates near-frontier problems | 4B rollout tokens | GRPO on the policy; CRITIC trained on accepted vs rejected rollouts (outcome plus step labels where exact) |
| S7 Local-judge preference | Chat style | Prompts only; judged by an open-weight judge on T4 (training-time only) | 0.5B | DPO on judged pairs; **report the judge**; never trained on MT-Bench/AlpacaEval prompts |

The training-token mix is roughly Layer-B "procedures" 60%, fluency 30% and chat 10%. Facts get 0% by design; they live in the store.

**Datastore build (v0, CLEAN):**
- Sources: en+ko Wikipedia (BY-SA, kept in the store rather than the weights), a FineWeb-Edu-dedup subset, FineWeb-2-ko, OpenWebMath, arXiv, Python-Edu.
- About 20B tokens, giving about 78M chunks of 256 tokens.
- Index size: binary keys 2.5 GB, PQ-32 2.5 GB, BM25 about 8 GB, text about 25 GB zstd [ESTIMATE].
- This fits a Kaggle dataset, HF private (100 GB [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits]) and the laptop SSD.
- v1 grows the store to 100B+ tokens on the laptop SSD and as a public HF dataset with a dataset card.

---

## 7. Compute plan (6ND)

The core trains as 12 unique layers. With mean r = 3 on the 4-layer recurrent block, effective layers are 4 + 12 + 4 = 20, so per-token compute is about 1.67x a plain 12-layer pass. Embedding and CLERK lookups add negligible FLOPs. **Effective N ≈ 1.67 × 57M (non-embedding core + heads) + LM-head matmul (18.9M) ≈ 114M FLOP-params** [ESTIMATE], so 6N ≈ 6.8e8 FLOPs per token.

| Stage | Tokens | FLOPs (6·N_eff·D) |
|---|---|---|
| S1 | 20B | 1.37e19 |
| S2 | 30B | 2.05e19 |
| S3–S5 | 12B | 8.2e18 |
| S6 RL (generation ≈ 2N plus training on 50% kept) | 4B | about 1.8e19 |
| S7 | 0.5B | 3.4e17 |
| **Total** | | **≈ 6.1e19** |

**Platforms:**
- **TPU v5e-8** (the main engine if the menu shows it): 1,576 TFLOPS bf16 peak, 8 × 197 [VERIFIED per chip: https://docs.cloud.google.com/tpu/docs/v5e]. At 25% MFU that is 3.9e17 FLOP/s, so the whole recipe takes about 43 TPU-hours, **about 2.5 weeks of the ~20 h/week quota** (third-party figure). The official JAX tutorial measured only 13–27% MFU on tiny models [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html], so I plan for 4 weeks.
- **T4 x2 only (fallback):** 130 TFLOPS fp16 peak [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf]. At 20% MFU that is 2.6e13 FLOP/s, so about 650 GPU-pair-hours, or **about 22 weeks at 30 h/week**. Painful but finite. Cut S1+S2 to 25B tokens to get about 12 weeks.
- **Teacher generation (S3–S5 synthetic data, S7 judging):** runs on T4s with vLLM fp16 (Qwen2.5-1.5B at about 0.3–0.65B tokens/week [ESTIMATE]), in parallel with TPU training.
- **Index build:** CLERK is a lookup plus mean-pool, about 0 FLOPs and CPU-bound. 78M chunks take hours on Kaggle's CPUs. Encoding never needs a GPU, and that is the decisive engineering win of a **frozen static key table**.
- **Ablations:** the RTX 3060 laptop (6 GB, bf16, current Triton) runs 10–30M variants of every idea first. At 100M, training state is about 1.6 GB, so the full model fits on the laptop too.

**T4 engineering:**
- Use fp16 matmuls with dynamic loss scaling. Keep the **delta-rule state, gate cumulative sums and SCRATCH inner updates in fp32**, since fp16 decay products underflow [ESTIMATE].
- Kernels: Triton ≥ 3.3 targets compute capability 8.0+ [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md]. Pin a Triton-3.2-era torch, which I believe is torch 2.6 [ESTIMATE, verify on day 1], for flash-linear-attention GDN kernels. Keep a pure-PyTorch chunked (chunk = 64) GDN fallback, plus a JAX `associative_scan` / chunked-matmul port for the TPU.
- Parallelism: DDP over PCIe across the two T4s, no tensor parallel. Use static shapes everywhere: retrieved chunks padded to exactly 8 × 256, which XLA needs anyway.
- Operations: checkpoint to HF Hub every 45 minutes and chain sessions within the 12 h GPU and 9 h TPU limits [VERIFIED secondary: https://gpuperhour.com/blog/free-cloud-gpus-and-credits].

---

## 8. Inference cost and latency per query (target: laptop RTX 3060, store on SSD)

All numbers are [ESTIMATE].

| Step | Work | Latency |
|---|---|---|
| CLERK + BM25 + IVF-binary search over 78M chunks | CPU popcount on about 64 probed lists | 30–120 ms |
| Fetch 32 chunks, rerank via PRELUDE | 32 × 256 tok × 2 × 17M FLOPs ≈ 2.8e11 FLOPs | 20–40 ms |
| Scan 8 chunks + task (about 2.3k tok) with r = 1 | 2 × 57M × 2.3k ≈ 2.6e11 FLOPs | 30–60 ms |
| Decode 300 tokens, r = 4, N = 1 | Bandwidth-bound: about 170 MB fp16 weights / 336 GB/s ≈ 0.5 ms floor per pass, about 3–5 ms real | 1–2 s |
| Hard query: N = 16 drafts batched, r = 8, 2 tool rounds, exec | About 16 × 300 × 2 × 114M × (8/3) ≈ 3e12 FLOPs + sandbox | 5–15 s |
| CRITIC scoring of 16 drafts | One adapter pass each | 0.2–0.5 s |

- An easy factual question takes about 2 s end to end.
- A hard math or code question takes 5–20 s and about 5e12 FLOPs, which is still roughly 100x less than one GPT-3.5-scale forward pass over the same tokens, if GPT-3.5 is about 175B [ESTIMATE].
- Every eval report will state the mean FLOPs and wall-clock per item, per Snell-style disclosure [VERIFIED rationale: https://arxiv.org/abs/2408.03314].

---

## 9. Scaling story

Four axes, each with its own knob. Only the first needs retraining.

| Axis | 0.1B (now) | 1B | 10B | North star |
|---|---|---|---|---|
| Core params | 85M GDN | 1B GDN (d = 1,536, 24 unique layers), still attention-free | 10B, plus product-key memory layers as a params-without-FLOPs knob [VERIFIED: https://arxiv.org/abs/2412.09764] | Planner quality |
| Datastore | 20B → 100B tokens | 0.5–1T (MassiveDS saw no saturation to 1.4T [VERIFIED: https://arxiv.org/html/2407.12854]) | Multi-T plus structured stores (tables, KGs, proofs) | All of science: PubMed, UniProt, ChEMBL, theorem libraries |
| Test-time compute | r ≤ 16, N ≤ 32 | Adaptive r ≤ 64, tree search | Compute-optimal allocation, about 4x over best-of-N [VERIFIED: https://arxiv.org/abs/2408.03314] | Days per problem, as with AlphaProof [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/] |
| Tools / verifiers | Python, SymPy, Z3 | Lean | Simulators | Docking, MD, lab-in-the-loop, with biosafety gates |

Why it should scale *predictably* [HYPOTHESIS, to be measured in E2]:
- **Params raise base pass@k and query and plan quality.** Test-time search multiplies a non-trivial base rate and does nothing for a zero one [VERIFIED: https://arxiv.org/abs/2408.03314].
- **Datastore** gains are log-linear in tokens, per RETRO and MassiveDS.
- The retrieval gain at 172M was worth about 10x parametric size [VERIFIED: https://arxiv.org/html/2112.04426], and RETRO's gains "do not diminish" with model size [VERIFIED: https://arxiv.org/abs/2112.04426]. The axes compound rather than substitute.
- **Compute wall:** a 1B Chinchilla-scale run is about 1.2e20 FLOPs, roughly 3 TPU-weeks, so it fits. A 10B run is about 1.2e22 FLOPs, roughly 7 years of free quota, so it needs TPU Research Cloud [VERIFIED free-quota terms: https://sites.research.google/trc/faq/]. Apply in week 1.

For the north star, scientific discovery is the ideal regime for this design: the knowledge is in the literature (the store), and the truth comes from experiments and simulators (verifiers). The weights only have to be a good scientist-librarian, not an encyclopedia.

**Biosafety (hard stop):**
- The bio and chem store shards exclude pathogen enhancement, toxin synthesis and other dual-use protocols, enforced with a blocklist plus classifier audit.
- DISPATCH carries a refusal policy for hazardous chemistry and biology.
- No wet-lab-actionable output leaves the system without human expert review.
- Any release of bio data or models goes through dual-use review first.

---

## 10. First three experiments (go/kill)

**E1 – "Does the librarian read?"** (laptop plus 1 Kaggle week)
- Setup: 20M and 85M cores, S1 + S2 on 2B tokens, Wikipedia-only store (en). Evaluate NQ-open and TriviaQA, decontaminated and audited.
- **GO:** 85M reaches NQ EM ≥ 35, and the RA model beats the same core closed-book by ≥ 20 EM.
- **KILL (pivot the reader to explicit span extraction):** NQ < 25, or recall@8 of the frozen CLERK + BM25 is < 60%.

**E2 – "Does knowledge scale with disk, not params?"** (2 TPU weeks)
- Setup: grid of {20M, 85M} params × {1B, 5B, 20B} store tokens × {r = 1, 4}. Evaluate generative MMLU, KMMLU, NQ and HAE-RAE.
- **GO:** at least +2 points per 4x store on MMLU and KMMLU at 85M, and 20M + 20B-store ≥ 85M + 1B-store on NQ (the substitution).
- **KILL:** flat after 5B tokens. That would mean the reader, not the store, is the bottleneck, and the parameter budget moves to the core.

**E3 – "Can copying buy fluency?"** (1 T4 week)
- Setup: S5 chat SFT with COPYIST on vs off, same params. Pairwise on MT-Bench turn-1 against the stored `gpt-3.5-turbo.jsonl` answers and against SmolLM2-135M-Instruct, both orders, local open judge. Calibrate the judge first on the stored reference-model answers.
- **GO:** ≥ 70% win vs SmolLM2-135M-Instruct, ≥ 15% win-or-tie vs GPT-3.5, and copy-on beats copy-off by ≥ 10 points.
- **KILL:** copy-on ≤ copy-off, i.e. the collage effect dominates. Then chat moves to a different subsystem and LIBRARIAN keeps only the grounding role.

---

## 11. Top risks

1. **Chat and commonsense walls** (§5.1–5.2). Most likely failure, with P(loss) above 0.95 [ESTIMATE]. Mitigation: hybridize with whichever proposal wins the fluency axis. Honesty requires saying retrieval does not fix these.
2. **Fake wins from contamination.** A retrieval system copies leaks verbatim. Mitigation: the CLEAN track, the audit log, post-cutoff private evals as the headline number, and paraphrase scrubbing.
3. **Recurrent recall bottleneck.** A fixed-size state may not hold 8 × 256 tokens of evidence. Mitigation: task re-statement after evidence, SCRATCH memory, re-SEARCH loops. Fallback: lower K and use a multi-pass FiD-style "read each chunk separately, then merge states".
4. **Frozen-CLERK recall ceiling.** Static keys are about 92% of MiniLM quality, and binary search loses another 3–8%. Mitigation: BM25 fusion, PRELUDE rerank, query rewriting via DISPATCH.
5. **Korean data and licences.** The pool is small and many licences are unknown; AI Hub data is non-redistributable, so it is eval-only. Mitigation: a licence manifest, BY-SA text kept in the store only.
6. **Quota availability.** TPU v5e-8 is sometimes oversubscribed (third-party report). The T4-only fallback is about 12–22 weeks for the whole recipe.
7. **Toolchain on sm_75.** Triton pinning breaks easily. Mitigation: keep a pure-PyTorch fallback and test it in CI on the laptop.
8. **Latency of test-time search** makes the eval suite slow: about 30k items at a mean of 3 s is about 25 GPU-hours. Budget eval as a first-class weekly cost.
9. **Dual-use leakage** once science stores are added. Hard gate, not a tunable.

**Verdict:** LIBRARIAN is the right answer to *where knowledge lives*, and it is the axis that scales most cheaply toward the north star. It is the wrong answer to *where judgement and voice come from*. Build it first (E1 and E2 are cheap and decisive), and make the final blueprint a hybrid that keeps the library and replaces the glue with something smarter.
