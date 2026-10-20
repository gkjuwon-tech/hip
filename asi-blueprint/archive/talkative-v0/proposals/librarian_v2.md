# LIBRARIAN v2: A Retrieval-Native Generator (revised after red team)

*Chief-architect proposal. Philosophy unchanged: "the weights are the librarian, not the library."* Date: 2026-10-19.
Labels: [VERIFIED: url] = figure from a primary source in the research digest or red-team inputs; [ESTIMATE] = my arithmetic or judgement; [HYPOTHESIS] = untested design claim.
All derived numbers (parameter counts, FLOPs, hours, index sizes) come from one script: `/home/user/hip/asi-blueprint/proposals/calc/librarian_v2_budget.py`. If a number in this file disagrees with the script, the script wins.

---

## 1. Thesis

A dense LLM uses most of its parameters as a **lossy compressed copy of its training corpus**. Physics-of-LMs puts the ceiling at about 2 bits of knowledge per parameter [VERIFIED: https://arxiv.org/html/2404.05405]. A 100M network can therefore hold about 25 MB of facts [ESTIMATE], and the evidence agrees: SmolLM2-135M scores 4.1% on TriviaQA after 2T tokens [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct]. A big model's other parameters buy three things:

- (a) **procedures**: read, compare, decompose, compute;
- (b) **surface fluency**: mostly memorized phrasing;
- (c) **calibrated plausibility**: a sense of what usually happens next.

The first-principles bet is that (a) needs few parameters, while (b) and (c) are largely **statistics of text**. Text statistics can be stored as text plus an index, which costs disk rather than parameters.

v2 sharpens this with a lesson from the red team: **"no facts in weights" was too strong.** A librarian who has never heard of chemistry cannot shelve a chemistry question. v2 keeps a **concept skeleton** in the weights (frequent entities, taxonomy, what kind of thing each thing is). Only the **long tail** goes to the store: specific dates, numbers, rare entities, verbatim text. The skeleton is what lets the core write good queries. The tail is what the store is for.

So LIBRARIAN keeps long-tail knowledge, most phrasing and part of the plausibility signal in a compressed public datastore. A small attention-free Gated DeltaNet core is the glue. It writes queries, reads evidence by scanning it, points at spans to copy, calls tools and checks its own work. Knowledge scales with **datastore tokens**: RETRO was still improving at 2T tokens, with about 25x parameter savings [VERIFIED: https://arxiv.org/abs/2112.04426v1].

What this philosophy **cannot** do on its own is stated in §5, and it has not changed since v1. **LIBRARIAN is the knowledge-axis champion, not a complete answer to the mid-boss.** Every forecast below is the bar a hybrid blueprint must clear.

---

## 2. Fair-play rules

1. **Params** are every trainable weight executed at inference: token embeddings, the static retrieval-key table (CLERK), the core, every head and adapter, and, when enabled, the meta-learned initialization of test-time memory. **Cap: < 100,000,000.**
2. **Data vs params (new, precise text):**
   - Statistics produced only by **counting or indexing** the frozen store, with no gradient fitting, are *data*: BM25 statistics, n-gram counts, inverted indexes, tokenizer merges. They are disclosed in bytes.
   - Anything **fitted** to data is *reported*. PQ codebooks and IVF centroids are reported in bytes and are tiny (kilobytes to megabytes). CLERK is fitted by distillation, so it **counts toward the 100M**.
   - Rationale: count-based statistics are a classical unbounded n-gram model. That is allowed in spirit, because the store is disclosed public text, but it must be visible.
3. **Datastore:** a frozen, dated, hashed offline snapshot of public text and code. It is excluded from params and fully disclosed: sources, token count, build script and licence manifest.
4. **Two store tracks:**
   - **CLEAN (headline, gating):** human-written text only. No model-generated text, so "retrieval" cannot mean fetching a cached LLM answer.
   - **OPEN (reported, never the headline):** up to 10% synthetic text from open-licensed models, such as Cosmopedia, which is Mixtral-generated [VERIFIED: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia]. Every OPEN result is published next to its per-row CLEAN delta.
5. **Allowed at inference:** Python sandbox, SymPy, Z3, a unit-test runner on *self-written* tests (including tests extracted from I/O examples in the prompt itself), and deterministic solvers. All are CPU-only with 0 params.
6. **Banned:** any neural net over 100M at inference, any LLM API, the live web, eval items or their solutions in the store or the training data, dataset-identity routing, and running hidden test cases during generation.
7. **Architecture rule: pointer-not-mixer (new).**
   - No softmax (or Hopfield-equivalent [VERIFIED: https://arxiv.org/abs/2008.02217]) attention over the token sequence at any layer.
   - A softmax may appear **only as an output distribution over discrete choices**: vocabulary items, ≤64 candidate phrases, tools, r-levels, retrieved-chunk rerank.
   - **A softmax-weighted sum of vectors must never feed back into the hidden state.** Choices re-enter the model only as discrete tokens, exactly like sampling.
   - Candidate representations must come from a **separate phrase encoder over token embeddings**, never from cached per-position core states.
   - Merging states across retrieved chunks uses **unnormalized sigmoid gates** (a linear-RNN state sum), not a softmax mixture. This is disclosed as the one place where external items are combined.
8. **Decontamination** applies to the store, all training data and every synthetic set:
   - 13-gram plus MinHash near-duplicate filtering against every eval set and its mirrors.
   - An embedding scrub at cosine > 0.9 with CLERK, because n-grams miss rephrased test sets [VERIFIED: https://arxiv.org/pdf/2311.04850v1].
   - A per-benchmark URL and repository blocklist (§6.3).
   - **Whole-source removal** for benchmarks built from real documents: drop the source *document*, not just the matching context (§6.3).
   - **Retrieval-audit log:** for every eval item, publish the top-k passages, the percentage that contain the gold answer string, and, for continuation tasks, the percentage where **the gold ending ⊂ top-k**. RETRO "exploit[s] leakage more strongly" [VERIFIED: https://arxiv.org/abs/2112.04426], so this audit is mandatory.
9. **Training-time teachers:**
   - Open-licensed models only, logged with name, size and licence.
   - **No GPT-3.5/4 outputs**, because of the OpenAI "compete" clause [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/]. This excludes TinyGSM, MetaMathQA, UltraChat and TinyStories from training.
   - A no-teacher ablation track is reported.
   - Teachers used for **CLERK distillation** and for **training-time judging** never run at inference.
10. **Judges:**
    - The training-time preference judge and the evaluation judge come from **different model families**.
    - Before use, the eval judge is calibrated on stored answers from at least 5 reference models.
    - The judge identities are published.
11. **Compute disclosure:** mean FLOPs and wall-clock per item for every row [VERIFIED rationale: https://arxiv.org/abs/2408.03314].

---

## 3. Architecture

### 3.1 Component diagram

```
                     ┌──────────────────────────── LIBRARY (disk, 0 params) ─────────────────────────────┐
                     │ TEXT: uint16 token memmap shards, EN/KO/CODE/MATH, ~20B tok (v0) → 100B+ (v1)       │
                     │ CATALOG-D: binary 256-bit CLERK keys, IVF + int8/PQ rescoring   (all shards)          │
                     │ CATALOG-S: on-disk compressed BM25   (en+ko Wikipedia + ko-Wiktionary only, v0)     │
                     │ [v1+] PHRASE COUNTS over a 1–2B-token subset (suffix-array style)                     │
                     └──────────▲──────────────────────────────────────────────────────────────────────────┘
                                │ top-32 chunk ids (RRF fusion)
 user ─► TOKENIZER (32k, ko-aware) ─► CLERK (static keys, distilled from strong open embedder) ─┘
            │
            ▼
   ┌─────────────────────── SCRIBE: Gated DeltaNet core, d=640, 5 heads × 128, no attention ──────────────────────┐
   │  PRELUDE (3 layers) ─► RECURRENT BLOCK (8 tied layers, iterated r ∈ {1,2,4,8,16,32}, r fixed per episode) ─► CODA (3) │
   │                                                                                                             │
   │  READ mode "fan-out": K chunks scanned IN PARALLEL as [q | chunk_i]  → states S_i                            │
   │                       merged S = Σ σ(g_i)·S_i  (sigmoid gates, no softmax) → scan [q] again from S           │
   └──────┬───────────────────────┬──────────────────────────────┬─────────────────────────────┬─────────────────┘
          ▼                       ▼                              ▼                             ▼
   LM head (factorized,    COPYIST pointer: scores ≤64     DISPATCH head:              CRITIC: rank-32 LoRA on
   tied with embedding)    phrases from the K chunks;      SEARCH/READ/PY/SOLVE/       SCRIBE + value head
                           phrase vectors from a           CHECK/ANSWER/PLAN + r, N    → rank / accept / retry
                           SEPARATE encoder over token
                           embeddings; chosen phrase re-enters as tokens
                                   └──────────► drafts ◄──── WORKBENCH: Python, SymPy, Z3, self-test runner
```

### 3.2 Inference dataflow, step by step

1. **Tokenize** with a 32,768-entry byte-fallback BPE, trained on a 45/35/20 EN/KO/code mix so Korean is not shredded into bytes. 32,768 fits uint16, so the store is a flat token memmap.
2. **CLERK query vector.** CLERK mean-pools a static 256-d table over the query tokens. The table is **distilled at training time from a strong open multilingual embedder**, the model2vec recipe; potion-base-8M is itself distilled from a sentence model and reaches about 92% of MiniLM-L6 at 7.6M params [VERIFIED: https://huggingface.co/minishlab/potion-base-8M]. Teacher candidates are the multilingual-e5-small and bge families [VERIFIED listing: https://huggingface.co/BAAI/bge-small-en-v1.5]. The final choice depends on Korean retrieval quality and licence [ESTIMATE: param count and Korean quality not verified]. In parallel, **BM25** runs over the Wikipedia and Wiktionary shards only, with Korean handled by character bigrams (0 params).
3. **CATALOG search.**
   - Hamming search over binary keys in IVF, then int8/PQ rescoring. Binary keeps 92.5% of retrieval quality raw and 96.45% with rescoring [VERIFIED: https://huggingface.co/blog/embedding-quantization].
   - Reciprocal-rank fusion with BM25 gives the top 32.
   - **DISPATCH may rewrite the query** and search again, up to 2 rounds.
   - **Multiple-choice mode:** one query per answer option (stem + option), so each option's evidence is retrieved separately.
4. **PRELUDE rerank.** The 3 prelude layers encode [query | candidate] pairs. A bilinear head scores them and keeps K: K = 32 in fan-out mode, K = 8 in sequential mode.
5. **READ.**
   - **Fan-out mode (default, FiD-style, new in v2):** each kept chunk is scanned *in parallel* as `[q | chunk_i]` through the full core at the episode's r, producing per-head states S_i. The merged state is S = Σ_i σ(g_i)·S_i, with g_i from a tiny gate head over (rerank score, pooled state).
     - For plain linear attention, the state is an additive sum of key-value outer products, so summing per-chunk states equals scanning the concatenation without decay [ESTIMATE: algebraic identity].
     - For gated delta-rule states the merge is only approximate. The core is trained in this mode, so it learns to work with the approximation [HYPOTHESIS].
     - The core then scans `[q]` again from S and decodes. Cost is linear in K and fully batched.
   - **Sequential mode (fallback):** `[task | chunk_1 … chunk_8 | task again]`, i.e. v1's task re-statement. Delta-rule overwrites help in-context recall here [VERIFIED: https://arxiv.org/abs/2412.06464v3].
6. **THINK.** r is fixed per **episode**, not per token (fix for the fatal flaw, see §3.4).
   - A cheap probe pass at r = 1 over the query lets DISPATCH choose r* ∈ {1, 2, 4, 8, 16, 32} and the draft count N.
   - If r* ≠ 1, the prefill is re-run at r* (linear cost), and decoding continues at r*.
   - Tool calls (SEARCH, PY, SOLVE, CHECK) splice their output into the stream, which is scanned at the same r*.
7. **PLAN, then WRITE (new).** For open-ended answers, DISPATCH first emits a short PLAN: an outline of 3–8 lines, each a sentence intent plus the chunk ids to use. Then each sentence is realized in turn. At every step the decoder mixes p_LM(token) with p_COPY(phrase):
   - COPYIST scores at most 64 phrases (1–16 tokens), taken **only from the K retrieved chunks** in v0. This is the CoG locality idea; CoG improves when the index is swapped or enlarged with no retraining [VERIFIED: https://arxiv.org/abs/2307.06962].
   - Phrase vectors come from a **separate phrase encoder**: an MLP over the [start, end, mean] token embeddings plus length and source embeddings. They never come from cached core states (pointer-not-mixer).
   - A selected phrase re-enters the stream as ordinary input tokens.
8. **CHECK.** N drafts are generated: N = 1 for easy items, 4–16 for medium, 32–100 for hard verifiable ones. Different drafts may use different r.
   - Exact verifiers run where they exist: execution, SymPy, and self-tests built from the prompt's own I/O examples.
   - CRITIC scores the rest.
   - The best draft or the majority-voted answer is returned with **citations to chunk ids**.

### 3.3 Exact parameter budget (from `librarian_v2_budget.py`)

**Changes from v1:**
- d = 640 with **5 heads of dim 128**, aligned to 128-wide matrix units.
- **Factorized embedding** (32768×128 + 128×640, tied with the LM head), which frees about 14.6M.
- The freed params and the v1 reserve go to the **recurrent planner**: 8 tied layers instead of 4.

**Per Gated DeltaNet layer:**
- q, k, v, o and output-gate projections: 5d² = 2,048,000.
- α/β gates: 2·640·5 = 6,400.
- Short convs (k = 4) on q, k, v: 7,680.
- Two RMSNorms: 1,280.
- Mixer total: 2,063,360.
- SwiGLU FFN, hidden 1,728: 3·640·1,728 = 3,317,760.
- **Per layer: 5,381,120.**

| Module | Shape | Trainable params |
|---|---|---|
| Token embedding (factorized, tied with LM head) | 32,768×128 + 128×640 | 4,276,224 |
| SCRIBE prelude | 3 unique GDN layers | 16,143,360 |
| SCRIBE recurrent block (tied, iterated r times) | 8 unique GDN layers | 43,048,960 |
| Recurrence input injection (concat state + prelude out → d) | 1,280×640 + 640 | 819,840 |
| SCRIBE coda | 3 unique GDN layers | 16,143,360 |
| Final norm | 640 | 640 |
| CLERK static key table (distilled) | 32,768 × 256 | 8,388,608 |
| Rerank head | 640×256 + 256×256 | 229,376 |
| Fan-out merge gates (per chunk, per head, sigmoid) | 640×5 + 5 | 3,205 |
| COPYIST phrase encoder (over token embeddings) | 384×640 + 640 + 640×640 | 656,000 |
| COPYIST query projection | 640 × 640 | 409,600 |
| Phrase-length (16) + source-type (8) embeddings | 24 × 640 | 15,360 |
| DISPATCH head (12 actions + 6 r-levels) | 640×18 + 18 | 11,538 |
| CRITIC LoRA (rank 32 on q, v, o, FFN-down of the 14 unique layers) | 14 × [3·(1,280·32) + (2,368·32)] | 2,781,184 |
| CRITIC value head | 640→640→1 | 410,881 |
| Segment embeddings (task, chunk, tool, answer) | 4 × 640 | 2,560 |
| **Total, v0 (counted, all executed at inference)** | | **93,340,696** |
| SCRATCH test-time memory init (640→1,280→640 MLP + gates), **disabled in v0** | | 1,640,322 |
| **Total with SCRATCH (v1+)** | | **94,981,018** |
| Headroom under 100,000,000 | | 5,018,982 |

- Unique depth is 14 layers. Effective depth is 3 + 8r + 3, which is 70 at r = 8 and 262 at r = 32.
- Huginn showed test-time depth buying up to "50B-equivalent" compute [VERIFIED: https://arxiv.org/abs/2502.05171]. Huginn's block is a Transformer; here the idea is transplanted onto an attention-free block, so whether it transfers is [HYPOTHESIS].
- The recurrent block is now 46% of the budget (planner), against 9% for the embeddings plus CLERK lookup tables. In v1 the split was 20% vs 32%.

### 3.4 Recursion in a stateful backbone (fix for the fatal flaw)

**The problem**, as the red team correctly identified: in a recurrent backbone, iteration i of the tied block at token t needs iteration i's state from token t−1. If r varies per token, deeper iterations have missing states.

**The v2 definition:** unrolled, the tied block applied r times is r stacked layers *in depth*, and each one carries its own time-recurrent state S_{i,t}. Inside one episode r is constant, so every state is always defined.

| Option | Rule | Cost | Status |
|---|---|---|---|
| **A (default)** | r fixed per episode, chosen by DISPATCH after an r=1 probe. Re-prefill at r* if needed. Different drafts may use different r. | Probe + re-prefill ≈ (1 + r*)× a single prefill. Prefill is linear and cheap next to decode with N drafts. | Well-defined; adopted |
| B | r per segment (per sentence or tool round). When r grows from r_a to r_b at a boundary, the new iterations' states are initialized by copying S_{r_a} (the deepest existing state). | No re-prefill | [HYPOTHESIS], ablate on the 3060 |
| C | One shared state per tied layer, written once per token after the last iteration; earlier iterations read it without writing | Cheapest; r can vary per token | [HYPOTHESIS], ablate on the 3060 |

**Training to match inference (train/test r mismatch fix):**
- r is sampled per sequence from a heavy-tailed distribution over {1, …, 32} with mean ≈ 2.
- **Deep supervision:** the coda and LM loss are applied after iteration r/2 and after iteration r.
- Truncated backprop through the last 4 iterations.
- **Report accuracy vs r for r = 1–32 on every row.**

All three items above are [HYPOTHESIS]. The compute in §7 counts full 6N for every iteration, which is conservative because truncated iterations are forward-only.

---

## 4. Why a datastore can stand in for parameters, quantified

| Lever | Evidence | What it buys LIBRARIAN | What it does not buy |
|---|---|---|---|
| Chunk retrieval | RETRO: about 25x fewer params at GPT-3-level Pile perplexity [VERIFIED: https://arxiv.org/abs/2112.04426v1]; NQ 45.5 vs 30.4 closed-book [VERIFIED: https://arxiv.org/html/2112.04426] | Factual QA | dm_mathematics: RETRO did not beat Jurassic-1 there [VERIFIED: same] |
| Small reader + many passages | FiD-base: 48.2 EM on NQ with a 220M reader, 2×110M DPR and 100 passages [VERIFIED: https://arxiv.org/abs/2007.01282] | The reason for fan-out READ with K = 32 | Our reader is 2.6x smaller and the retriever is static |
| Retrieval at small scale | Atlas-770M MMLU: 56.3 full/transfer, but **38.9 at 5-shot** vs 29.2 closed-book [VERIFIED: https://arxiv.org/abs/2208.03299, via tiny_sota digest] | +10 points from retrieval at 5-shot | **v1 anchored on the wrong Atlas number; corrected here** |
| Datastore scaling | MassiveDS: no saturation up to 1.4T tokens; MMLU only +3.5 [VERIFIED: https://arxiv.org/html/2407.12854] | Knowledge grows with disk | Reasoning does not |
| Phrase copying | CoG beats token LMs on generation quality [VERIFIED: https://arxiv.org/abs/2307.06962] | Fluency borrowed from text | Global structure (§5.2) |
| Exec + verifier | TinyGSM: verifier "plays a bigger role" [VERIFIED: https://arxiv.org/html/2312.09241] | Math and code selection | TinyGSM's 63.1 used GPT-3.5 data, which we exclude; a clean 140M reaches 16.3 [VERIFIED: https://arxiv.org/html/2509.24945v3] |

---

## 5. Scoreboard: honest per-row forecast (revised)

Targets are the max-reported GPT-3.5 values. Forecasts are for the CLEAN track at 93.3M params, after the §6 recipe and the red-team fixes. **All forecasts are [ESTIMATE].** P(win) is the probability of beating the target plus its CI.

| Row | Target | v1 forecast | **v2 forecast** | **P(win)** | Reason for change |
|---|---|---|---|---|---|
| NQ-open | 38.6 | 42–48 (0.70) | 35–46 | 0.45 | v1 was overconfident vs FiD's 100 passages and a 220M reader. Now K = 32 fan-out plus NQ/TQA *train*-split SFT (legal), which pulls it back up |
| TriviaQA 5-shot | 85.8 | 68–78 | 62–75 | 0.05 | FiD-base 65.0 EM [VERIFIED: https://arxiv.org/abs/2007.01282]. Honest loss |
| MMLU | 71.4 | 48–58 | **38–50** | <0.02 | Re-anchored to Atlas 5-shot 38.9. Gate 1 is beating 38.9 |
| KMMLU | 42.47 | 38–47 | 36–46 | 0.35 | Per-option retrieval helps; the Korean exam-site blocklist removes leaked items |
| HAE-RAE (5 subsets) | 49.45 mean | 50–62 | 48–60 | 0.50 | ko-Wiktionary in the store targets LW/RW; blocklisting cost about −2 |
| KoBEST avg | 79.3 | 60–72 | 58–70 | 0.05 | Reading quality plus Korean HellaSwag (fluency) |
| GSM8K 0-shot | 78.1 (5-shot 57.1) | 62–74 | **45–65** | 0.03 vs 78.1; 0.45 vs 57.1 | TinyGSM data excluded. OpenMathInstruct-2 + SymPy generators instead |
| MATH | 34.1 | 22–36 | 18–32 | 0.20 | Same data correction |
| HumanEval+ / HumanEval | 70.7 / 76.8 | 30–48 | 30–48 | <0.03 | Self-tests from docstring examples and N = 32–100 with repair add a little; the target is EvalPlus's best snapshot |
| MBPP+ / MBPP | 69.7 / 82.5 | 40–55 | 40–55 | <0.03 | Same |
| BBH | 68.3 | 48–60 | 45–58 | 0.05 | Executable subtasks win; language-heavy subtasks lose |
| DROP F1 | 64.1 | 60–72 | 56–68 | 0.40 | Span pointing + PY arithmetic; the red team's discount accepted |
| ARC-Challenge | 87.4 | 55–68 | 50–64 | <0.02 | Per-option retrieval; distractor reasoning remains |
| HellaSwag | 85.5 | 45–58 | **42–52** | <0.01 | Leak path removed (§6.3); what remains is honest plausibility |
| WinoGrande | 81.6 | 52–58 | 51–57 | <0.01 | No retrieval key |
| ARC-AGI-1 pass@2 | 5.0 | 6–15 | 6–15 | 0.50 | Python/DSL program search over re-ARC-trained proposals; TRM shows tiny nets can [VERIFIED: https://arxiv.org/html/2510.04871v1]. TRM's ARC run cost about 4 H100 × 3 days, far beyond our budget [VERIFIED: same] |
| MT-Bench | 8.39 | 3.5–5.0 | **2.5–4.0** | <0.01 | v1 had no support for 3.5–5.0; 125–135M models score 1.98–2.52 [VERIFIED: https://arxiv.org/html/2402.14905] |
| AlpacaEval 2 LC | 22.35 | 3–10 | 2–8 | <0.01 | |
| Arena-Hard | 24.8 | 1–5 | 1–4 | <0.01 | |
| Korean chat | none measurable | — | **unmeasured head-to-head** | n/a | No stored GPT-3.5 Korean chat outputs found in any input; the only reference is Arena-Korean 1049.6 [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard] |

**Bottom line, stated plainly:**
- LIBRARIAN v2 is forecast to win about **3–5 of roughly 19 gated rows**: NQ, HAE-RAE, ARC-AGI-1, and possibly DROP and KMMLU.
- **P(winning all Tier A+B rows) < 0.1%** [ESTIMATE].
- The red team made the forecasts *worse*, and that is correct: v1 was anchored on one wrong number (Atlas) and one excluded dataset (TinyGSM).
- Retrieval is **necessary**, because nothing else fixes the knowledge-capacity wall. It is **not sufficient**.

### 5.1 Commonsense (HellaSwag, WinoGrande)

[HYPOTHESIS] Plausibility is partly a corpus statistic. In v0, CRITIC scores each candidate ending using:
- the best retrieved *analogous* narrative (dense search on the context);
- n-gram support computed **only over the K retrieved chunks**. Corpus-wide phrase counts move to v1, built over a 1–2B-token subset; see §6.4 for sizing;
- contrastive training on self-generated negatives from FineWeb-Edu.

**The v1 leak fix is mandatory and stronger than the red team asked:**
- **WikiHow and ActivityNet-derived text are excluded from the CLEAN store entirely**, by domain blocklist and by context match. v1's "WikiHow-like shard" is gone; how-to text comes only from other FineWeb-Edu domains.
- The audit metric "gold ending ⊂ top-k" must be ≈ 0% before any HellaSwag number is reported.

Expected result: +0–10 over SmolLM2-135M's 42.1 [ESTIMATE]. WinoGrande stays near chance.

**Why current tech can't close this:** the only published sub-1.5B jump on WinoGrande came from synthetic textbook data at 1.3B (60.4 → 73.4) [VERIFIED: https://arxiv.org/html/2309.05463], which is 14x our budget.

### 5.2 Open chat, creative writing, Korean fluency: the ladder

v1 had honesty but no ladder. v2's attack is **plan → retrieve exemplars → point/copy → realize sentence by sentence → critique**, with three new ingredients:
1. **8 tied planner layers**, funded by the embedding factorization and the reserve.
2. **A human-written Q&A shard** in the CLEAN store for retrieve-and-edit answers:
   - StackExchange dumps (CC BY-SA, store-only like Wikipedia) [ESTIMATE: licence not re-verified this session].
   - Open-licence ELI5-style explanations, **only where the licence is verified**; Reddit-derived sets are excluded by default.
3. **Explicit PLAN tokens:** the global structure that copying cannot supply comes from a short plan the planner writes first.

**Rungs, with the same judge protocol for all of them:**

| Rung | Target | Reference |
|---|---|---|
| R1 | MT-Bench > 1.98 | SmolLM2-135M-Instruct [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct] |
| R2 | > 2.52 | MobileLLM-LS-125M [VERIFIED: https://arxiv.org/html/2402.14905] |
| R3 | ≥ 5.0 | Unclaimed territory for < 1B |
| R4 | > 8.39 | GPT-3.5-0613 [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard] |

- Each rung is measured at 10M, 20M, 40M and 85M (§10 E2/E3). Fit score vs log(params) and vs log(r·N).
- **If the extrapolated crossing point with 8.39 is above 100M, the report says so, with the slope.** That is a scientific result, not a failure to hide.
- **Korean chat:** labelled "unmeasured head-to-head" until stored GPT-3.5 Korean outputs are found; finding them is an open task. Korean SFT data comes from open-licensed teachers (licence checked per model card) and is filtered by back-translation consistency. ko-Wiktionary is added to the store (BY-SA, store-only) [ESTIMATE on licence].

**Precise wall (unchanged):** chat needs knowledge, fluency and reasoning *simultaneously at every token*. Copying gives the first two locally, and PLAN tokens are a bet on the third. I have no evidence that this works at 93M.

---

## 6. Training recipe

All token counts are [ESTIMATE]. "Tokens" means tokens **scanned**, including retrieved neighbours.

| Stage | Goal | Data | Tokens | Objective |
|---|---|---|---|---|
| S0 Tokenizer + CLERK | Vocabulary; static keys | 2B-token EN/KO/code sample | — | BPE. **CLERK distilled from a strong open multilingual embedder (model2vec recipe), then frozen *within this scale*.** Re-distilled at each scale-up; re-indexing static keys takes CPU-hours |
| S1 Fluency + concept skeleton | Grammar, Korean, code syntax, frequent concepts | FineWeb-Edu (int_score ≥ 4) [VERIFIED: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu], FineWeb-2-ko [VERIFIED: https://arxiv.org/abs/2506.20920v1], Python-Edu, OpenWebMath: 40/30/15/15 | 15B | Next-token + LLM-JEPA embedding loss [VERIFIED at fine-tune scale: https://arxiv.org/abs/2509.14252]; heavy-tailed r, deep supervision |
| S2 RA pretrain ("learn to lean") | Use evidence for the long tail | 1,024-token targets, each with precomputed neighbours from a 10B-token neighbour store. Neighbours **exclude the target's own document and its MinHash near-duplicate cluster** | 25B (about 6.1M targets) | Next-token + pointer loss on spans present in the neighbours. **Long-tail fact-dropout:** only entities below a corpus-frequency threshold are masked when they are not retrievable. The threshold is ablated in E2 |
| S3 Reader/retriever SFT | Query writing, per-option queries, rerank, cite, fan-out merge | KILT provenance triples [VERIFIED: https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/] **plus the NQ and TriviaQA train splits (train only)**, synthetic QA over the store from open teachers | 3B | Rerank InfoNCE, answer cross-entropy, citation accuracy |
| S4 Tool + procedure curriculum | Program-of-thought, tool grammar | Reasoning Gym (100+ generators) [VERIFIED: https://arxiv.org/pdf/2505.24760]; re-ARC on the train split [VERIFIED: https://arxiv.org/pdf/2404.07353]; **OpenMathInstruct-2** (about 14M items, CC-BY-4.0, Llama-3.1-405B solutions) [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2], 2M items rewritten to Python by an open teacher and kept only if execution matches the reference answer; SymPy-generated algebra and arithmetic; Python-Edu execution traces | 8B | Cross-entropy on calls and result use; adaptive difficulty at about 60% solve rate |
| S5 Chat SFT | Format, multi-turn, PLAN tokens | SmolTalk (Apache-2.0) [VERIFIED: https://www.huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/blob/main/README.md] and Tulu-3 minus NC subsets [VERIFIED: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/main/README.md]; every answer re-grounded with retrieved citations and given a PLAN prefix; Korean SFT from open teachers | 1B | Cross-entropy |
| S6 RLVR + self-play | Raise pass@k; train CRITIC | Math, code with self-written tests, Reasoning Gym, retrieval QA with exact match; a proposer for near-frontier problems | 4B rollout tokens at r = 4, 50% kept for training | GRPO; CRITIC trained on accepted vs rejected rollouts |
| S7 Preference | Chat style | Prompts only, **never** MT-Bench/AlpacaEval/Arena-Hard prompts. **Judge from family A; the eval judge is from family B** | 0.5B | DPO on judged pairs |

**Teacher-token budget (new):**
- Rewriting 2M OpenMathInstruct-2 items at about 300 tokens each is about 0.6B tokens.
- With Qwen2.5-1.5B fp16 on T4x2 at about 0.3–0.65B tokens/week, that is **about 1–2 T4-weeks** [ESTIMATE, from the compute digest].
- Korean SFT and synthetic QA add another 1–2 T4-weeks.
- Optional: if vllm-tpu holds up, the TPU path gives about 0.74B tokens/week of 27B-class text [VERIFIED third-party, not reproduced: https://github.com/ARahim3/kaggle-tpu-lab]. It competes with training hours.

### 6.1 Datastore build (v0, CLEAN)

- **Sources:** en+ko Wikipedia and ko-Wiktionary (BY-SA, kept in the store and not the weights [VERIFIED licence: https://download.wikipedia.com/legal.html]), a FineWeb-Edu-dedup subset, FineWeb-2-ko, OpenWebMath, arXiv, Python-Edu (permissive), StackExchange (BY-SA) [ESTIMATE on licence], and open-textbook exam explanations from human-written open textbooks (CC BY, e.g. OpenStax) [ESTIMATE: licence not verified this session].
- **Size:** about 20B tokens, i.e. 78M chunks of 256 tokens.
- **Index footprint** (from the script, [ESTIMATE] until E0 measures it):
  - Text as a uint16 memmap: 40 GB raw (about 20–25 GB zstd).
  - Binary keys: 2.5 GB. int8 rescoring keys: about 20 GB, kept on SSD and memory-mapped.
  - **BM25 over Wikipedia only** (about 6B tokens, 23M chunks, about 3.5e9 postings): about 28 GB uncompressed, **about 5 GB as an on-disk compressed inverted index**, e.g. a Lucene-class engine [ESTIMATE: tool choice not verified this session].
  - The red team's 94 GB figure for BM25 over all 20B tokens is confirmed by the script, which is why BM25 is restricted to Wikipedia.

### 6.2 S2 data pipeline (fix for the unbudgeted join)

- Neighbour **IDs** are precomputed on the laptop CPU using CLERK binary search. That is 6.1M targets × 8 IDs × 4 B ≈ **195 MB**.
- The 10B-token neighbour store is one uint16 memmap (about **20 GB**), shipped as a Kaggle Dataset or HF dataset. It is joined on the fly in the data loader, so there are no pre-joined sequences.
- Batched binary IVF search runs at many queries/s per core, far faster than the red team's 100 ms/query assumption, but I have not measured it [ESTIMATE].
- Run `df -h` on the TPU VM on day 1. If disk is below 30 GB free, shrink the neighbour store to 5B tokens.

### 6.3 Contamination: per-benchmark leak paths

| Row | Leak path | Fix |
|---|---|---|
| HellaSwag | Gold endings are real WikiHow/ActivityNet continuations | Exclude WikiHow and ActivityNet-caption sources from the store and training entirely; audit "gold ending ⊂ top-k" ≈ 0% as a gate |
| KMMLU / HAE-RAE | Korean exam sites crawled into FineWeb-2-ko | Domain blocklist of Korean exam/quiz sites; embedding scrub at cosine > 0.9; per-row audit |
| HumanEval / MBPP | Solutions copied across many GitHub repos in Stack-v2 / Python-Edu | Repository blocklist + function-level MinHash against canonical solutions + embedding scrub |
| MMLU / ARC / GSM8K / MATH | Mirrors everywhere; teacher regurgitation | URL blocklist + 13-gram + embedding scrub on the store **and** on every synthetic set |
| All | Paraphrases | Embedding scrub; **headline = private post-cutoff set** |

### 6.4 What is deliberately NOT in v0 (solo-dev scope, red-team §7)

These are deferred: SCRATCH memory, corpus-wide phrase counts, sequential READ mode (fan-out only), PLAN tokens (enter at S5), CRITIC LoRA (enters at S6) and GRPO (enters at S6).

**v0 MVP for E1:** GDN core + CLERK + Wikipedia BM25 + binary dense search + PY tool + fixed r per episode. Nothing else.

---

## 7. Compute plan (6ND, from `librarian_v2_budget.py`)

- **FLOP-params per token** (matmul params touched, counting the LM-head matmul): N_eff(r) = (3 + 3 + 8r)·5.38M + injection + head. At r = 1, 2, 4, 8 and 16 this gives 80.8M, 124.7M, 212.4M, 387.9M and 738.9M.
- Training uses mean r ≈ 2, so N_eff ≈ 124.7M and **6N ≈ 7.5e8 FLOPs/token**.
- This is conservative: truncated-backprop iterations really cost 2N, not 6N.

| Stage | Tokens | FLOPs |
|---|---|---|
| S1 | 15B | 1.12e19 |
| S2 | 25B | 1.87e19 |
| S3–S5 | 12B | 8.98e18 |
| S6 (4B rollout tokens at 2N, r = 4, + training on 2B at 6N) | — | 4.25e18 |
| S7 | 0.5B | 3.74e17 |
| **Total** | | **≈ 4.35e19** |

**Platforms (v1's unit error corrected):**
- **TPU v5e-8:** 1,576 TFLOPS bf16 peak (8 × 197 [VERIFIED per chip: https://docs.cloud.google.com/tpu/docs/v5e]).
  - At the red team's realistic **13.5% MFU** (the JAX tutorial measured 13–27% on tiny GPTs [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html]), the rate is **2.13e14 FLOP/s** (v1 wrongly wrote 3.9e17).
  - The full recipe takes **about 57 TPU-hours, about 2.8 weeks** of the ~20 h/week quota (third-party figure [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab]).
  - Plan for 5 weeks, to absorb oversubscription and restarts.
- **T4x2 only (fallback):** 130 TFLOPS fp16 peak [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf].
  - At 15% MFU that is 1.95e13 FLOP/s, so **about 620 GPU-pair-hours, about 21 weeks** at 30 h/week.
  - Cutting S1+S2 to 20B tokens brings it to about 12 weeks. S1 and S2 token counts are config knobs.
- **E2 scaling grid** (four core sizes, 3B tokens each): about 4.1e18 FLOPs, **about 5 TPU-hours**. This is cheap because store size and r are *inference-time* knobs: one training run per parameter size, evaluated at every (store, r) cell.
- **Evaluation is a first-class cost:**
  - Full Tier A is about **103k items**. At a mean of 3 s that is about **86 GPU-hours**, roughly 3 weeks of T4 quota (the red team's correction to v1's 30k).
  - So the weekly eval runs on **fixed stratified subsets of 300 items per row** (about 6k items, about 5 GPU-hours) and reports CIs.
  - The full suite runs only at milestones.
  - Early exit: r = 1, N = 1 whenever CRITIC confidence is high.

**Toolchain (red-team §7 adopted: no Triton at all):**
- **Training:** JAX on the TPU, with a chunked (chunk = 64) GDN formulation and `jax.lax` scans, static shapes, and fan-out batches padded to exactly 32 × (q + 256).
- **Reference, inference and T4 fallback training:** pure PyTorch. Chunked prefill plus an O(1) recurrent decode step. fp16 matmuls; **delta-rule state, gate cumulative sums and the fan-out merge kept in fp32**, because T4 has no bf16 [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf] and fp16 decay products underflow [ESTIMATE].
- **Why no Triton:** Triton ≥ 3.3 targets compute capability 8.0+ [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md], so dropping it removes the sm_75 pin risk entirely.
- **CI on the 3060:** every JAX kernel is cross-checked against the PyTorch reference (tolerance tests in bf16 and fp32).
- **Operations:**
  - DDP over PCIe on T4x2.
  - Checkpoint to the HF Hub every 45 minutes (100 GB private [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits]).
  - Chain sessions within the 12 h GPU and 9 h TPU limits [VERIFIED secondary: https://gpuperhour.com/blog/free-cloud-gpus-and-credits].
  - Do not launch TPU jobs via `kaggle kernels push` without an in-run device check [VERIFIED: https://github.com/Kaggle/kaggle-cli/issues/1197].

---

## 8. Inference cost and latency per query (target: laptop RTX 3060 6 GB, store on SSD)

All numbers are [ESTIMATE].

| Step | Work | Latency |
|---|---|---|
| CLERK + Wikipedia BM25 + IVF-binary search over 78M chunks | CPU popcount, about 64 probed lists; 1–4 queries (rewrite or per-option) | 30–200 ms |
| Rerank 32 candidates through the prelude | 32 × 300 tok × 2 × 16M ≈ 3e11 FLOPs | 20–40 ms |
| Fan-out READ, K = 32, r = 1 | 32 × 300 tok × 2 × 80.8M ≈ 1.6e12 FLOPs, batched | 0.1–0.3 s |
| Decode 300 tokens, r = 4, N = 1 | Weights re-read per iteration: (6 + 32) × 10.8 MB ≈ 0.41 GB/token at 336 GB/s ≈ 1.2 ms floor, 4–6 ms real | 1.5–2 s |
| Hard query: N = 32 drafts batched, r = 8, 2 tool rounds, execution | 32 × 300 × 2 × 388M ≈ 7.5e12 FLOPs + sandbox | 5–20 s |
| CRITIC scoring | One adapter pass per draft | 0.2–0.5 s |

- **Laptop serving footprint:** 0.19 GB fp16 weights; 2.5 GB binary keys in RAM; int8 keys (20 GB), text (40 GB) and Wikipedia BM25 (about 5 GB) memory-mapped from SSD. That is about **70 GB of SSD and 4–6 GB of RAM** [ESTIMATE]. Check the laptop on day 1, and shrink the int8 rescoring tier to PQ-32 (2.5 GB) if the SSD is short.
- SCRATCH, when enabled in v1+, runs only when the scan exceeds about 4k tokens, because its serial gradient steps break draft batching.

---

## 9. Scaling story

Four axes. Only the first one needs retraining; CLERK re-distillation is CPU-hours.

| Axis | 0.1B (now) | 1B | 10B | North star |
|---|---|---|---|---|
| Core params | 93M GDN, 14 unique layers, 8 tied | 1B GDN (d = 1,536, head dim 128), still attention-free | 10B plus product-key memory as a params-without-FLOPs knob [VERIFIED: https://arxiv.org/abs/2412.09764] | Planner quality |
| Retriever | CLERK distilled, 8.4M | Re-distilled, wider table; optional in-budget cross-scorer | Learned dense encoder sharing the core | Structured scientific search |
| Datastore | 20B → 100B tokens | 0.5–1T (MassiveDS: no saturation to 1.4T [VERIFIED: https://arxiv.org/html/2407.12854]) | Multi-T, plus tables, KGs and proofs | All of science |
| Test-time compute | r ≤ 32, N ≤ 100 | Tree search | Compute-optimal allocation, about 4x over best-of-N [VERIFIED: https://arxiv.org/abs/2408.03314] | Days per problem, as with AlphaProof [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/] |
| Tools / verifiers | Python, SymPy, Z3 | Lean | Simulators | Docking, MD, lab-in-the-loop with biosafety gates |

**Designed-for predictability (new, red-team 5.4):**
- "It should scale" is not a claim; it is a measurement.
- E2 trains {10, 20, 40, 85}M cores with an identical recipe and evaluates each at {1, 5, 20}B store tokens × r ∈ {1, 4, 16}.
- The fitted metric is a **smooth** one: retrieval-conditioned NLL on held-out, post-cutoff text, plus answer NLL on NQ/MMLU. Noisy accuracies are not used for the fit.
- The goal is a fit of the form L(N, D_store, r) with residuals small enough that the 85M point is *predicted* from 10–40M before it is run.
- **A law that predicts 85M from 10–40M is the first real evidence for 1B and 10B.** If no law predicts it, the scaling story is downgraded to [HYPOTHESIS] in the blueprint.

**Why the axes should compound** [HYPOTHESIS, to be measured]:
- Params raise base pass@k and query quality, and search multiplies a non-trivial base rate [VERIFIED: https://arxiv.org/abs/2408.03314].
- RETRO's retrieval gains "do not diminish" with size [VERIFIED: https://arxiv.org/abs/2112.04426].

**Compute wall:**
- A 1B Chinchilla run is about 1.2e20 FLOPs, about 3 TPU-weeks: it fits.
- A 10B run is about 1.2e22, about 7 years of free quota [ESTIMATE, compute digest]: it does not.
- **Apply to TPU Research Cloud in week 1** [VERIFIED: https://sites.research.google/trc/faq/]. Honest caveat: TRC users pay for VMs and storage, so 10B is *not strictly 0 KRW*. The plan says so.

**Biosafety (hard stop, unchanged):**
- Bio and chem store shards exclude pathogen enhancement, toxin synthesis and other dual-use protocols, enforced by a blocklist plus a classifier audit.
- DISPATCH carries a refusal policy for hazardous chemistry and biology.
- No wet-lab-actionable output leaves the system without human expert review.
- Releases of bio data or models go through dual-use review first.

---

## 10. First experiments (go/kill)

**E0 – "Measure before believing"** (day 1–3, laptop + 1 Kaggle session each)
- Run `nvidia-smi`, `df -h`, `free -g` and `nproc` on T4x2; `jax.devices()` on the TPU; check remaining quota in account settings.
- Measure GDN MFU at 40M/93M on v5e (bf16) and T4 (fp16).
- Build the BM25, binary and int8 indexes on a **1B-token shard** and measure bytes and query latency.
- Replace every [ESTIMATE] in §6.1, §7 and §8 with measured values.
- **KILL condition:** v5e MFU < 8% *and* T4 MFU < 10%. Then cut S1+S2 to 10B tokens and drop r_max to 8.

**E1 – "Does the librarian read?"** (MVP only; laptop + 1 Kaggle week)
- Setup: 20M and 93M cores, S1 + S2 on 2B tokens, Wikipedia-only store, fan-out K = 32, NQ/TQA train-split SFT. Evaluate NQ-open and TriviaQA, decontaminated and audited.
- **Recursion ablation:** options A/B/C from §3.4 on the 20M core, on the 3060.
- **GO:** 93M NQ EM ≥ 35; RA beats closed-book by ≥ 20 EM; CLERK + BM25 recall@32 ≥ 70%; option A shows monotone accuracy vs r for r = 1–8.
- **KILL (pivot the reader to explicit span extraction):** NQ < 25, or recall@32 < 60%.

**E2 – "Does knowledge scale with disk, and predictably with params?"** (about 1 TPU week including evals)
- Setup: {10, 20, 40, 85}M cores × {1, 5, 20}B store × r ∈ {1, 4, 16}, plus the long-tail fact-dropout threshold ablation at 40M.
- Metrics: generative MMLU (per-option retrieval), KMMLU, NQ, HAE-RAE, and the smooth NLL from §9.
- **GO:**
  - at least +2 MMLU/KMMLU points per 4x store at 85M;
  - 20M + 20B store ≥ 85M + 1B store on NQ (substitution);
  - 85M MMLU 5-shot > 38.9 (the Atlas-770M 5-shot number);
  - the fit from 10–40M predicts 85M NLL within its CI.
- **KILL:** flat after 5B store tokens, which means the reader is the bottleneck: move budget from CLERK to the core. Also kill the "predictable" claim if the fit fails.

**E3 – "Can pointing + planning buy fluency?"** (1 T4 week)
- Setup: S5 chat SFT on 20M and 93M cores, COPYIST on/off × PLAN on/off.
- Pairwise on MT-Bench turn 1 against the stored `gpt-3.5-turbo.jsonl` answers [VERIFIED exists: https://huggingface.co/spaces/lmsys/mt-bench] and against SmolLM2-135M-Instruct, both orders.
- The **eval judge is from a different model family than the S7 training judge**, calibrated first on at least 5 reference models' stored answers.
- **GO:** clears rung R1 (> 1.98 single-score) at 93M; ≥ 70% win vs SmolLM2-135M-Instruct; ≥ 15% win-or-tie vs GPT-3.5; copy-on + plan-on beats copy-off + plan-off by ≥ 10 points of win rate.
- **KILL:** copy-on ≤ copy-off (collage effect dominates). Then chat moves to another subsystem and LIBRARIAN keeps only grounding. **Publish the rung slope either way.**

---

## 11. Top risks

1. **Chat and commonsense walls** (§5.1–5.2). The most likely failure; P(loss) > 0.95 [ESTIMATE]. Mitigation: the hybrid blueprint, plus the rung ladder that measures the slope honestly.
2. **Fake wins from contamination.** Mitigations: CLEAN track, whole-source removal (WikiHow/ActivityNet excluded), per-benchmark blocklists, the "gold ⊂ top-k" audit, and a private post-cutoff headline set.
3. **Fan-out merge is approximate for gated delta-rule states** [HYPOTHESIS]. Fallback: sequential mode with task re-statement and K = 8; then SCRATCH memory.
4. **Recursion option A costs a re-prefill;** options B/C are untested. E1 ablates all three. If none shows monotone gains with r, the tied block becomes 8 plain layers and test-time compute shifts to N (drafts) only.
5. **Static CLERK recall ceiling.** About 92% of MiniLM, minus 3–8% for binary. Mitigations: distillation from a strong teacher, BM25 fusion on Wikipedia, query rewriting, per-option queries, re-distillation per scale.
6. **Long-tail threshold misset.** Too low, and facts are crammed into weights; too high, and the core can't write queries. E2 ablates it.
7. **Korean data and licences.** 35.3% of Korean HF datasets have unknown licences [VERIFIED: https://arxiv.org/pdf/2507.04329]; AI Hub is eval-only; Korean chat is unmeasured head-to-head.
8. **Quota availability.** v5e-8 is sometimes oversubscribed [VERIFIED third-party: https://github.com/ARahim3/kaggle-tpu-lab]. The T4 fallback takes 12–21 weeks.
9. **Eval cost** (86 GPU-hours per full Tier A). Use stratified weekly subsets.
10. **Judge hacking.** Separate judge families, calibrated, published.
11. **Solo-developer scope.** Two implementations (JAX train, PyTorch reference/inference), not three; the MVP gates every subsystem.
12. **Dual-use leakage** once science stores are added. A hard gate, not a tunable.

**Verdict:** LIBRARIAN v2 is the right answer to *where knowledge lives*, and it is the cheapest axis to scale toward the north star. It is still the wrong answer to *where judgement and voice come from*. The planner now holds 46% of the budget, and E3 will measure whether that helps.
- Build E0–E1 first: about two weeks, and decisive.
- Make the final blueprint a hybrid that keeps the library, the fair-play rules and the pointer-not-mixer rule, and that competes on the planner.

---

## Changelog v1 → v2

**Fatal**
- **F1 (per-token r undefined; red-team 5.1):** r is now fixed per episode (option A: r=1 probe, then re-prefill at r*). Options B (state-copy per segment) and C (shared state) are ablated in E1. Training uses heavy-tailed r with deep supervision, and accuracy is reported vs r (5.2).
- **F2 (HellaSwag leak; 3.1):** WikiHow and ActivityNet sources are excluded from the store and training entirely, which goes further than the requested article-level removal. "Gold ending ⊂ top-k ≈ 0%" is a reporting gate. A per-benchmark leak table is added (§6.3; covers 3.2).
- **F3 (COPYIST = attention; §4):** the pointer-not-mixer rule is in the rulebook. Phrase vectors come from a separate encoder over token embeddings. Chosen phrases re-enter only as tokens. The fan-out merge uses sigmoid, not softmax.

**Major**
- 1.2: head dim 128 (d=640, 5 heads); MFU planned at 13.5%.
- 1.3: BM25 restricted to Wikipedia, on-disk compressed (about 5 GB vs 94 GB).
- 1.4: copy candidates come from retrieved chunks only in v0.
- 1.5: precomputed neighbour IDs (195 MB) plus a 20 GB uint16 memmap joined on the fly.
- 1.6: about 103k Tier A items, 86 GPU-hours per full eval, so weekly stratified 300-item subsets.
- 2.1: MMLU re-anchored to Atlas 5-shot 38.9; forecast 38–50; per-option retrieval plus an open-textbook explanation shard.
- 2.2: fan-out K=32 FiD-style read plus NQ/TQA train-split SFT; NQ P(win) is now 0.45.
- 2.3: TinyGSM-based forecast removed; OpenMathInstruct-2 with execution filtering, with the teacher budget stated; GSM8K forecast 45–65.
- 2.5: embedding factorized (−14.6M), budget moved into an 8-layer tied planner, human Q&A shard, PLAN tokens, rung ladder with slope reporting; MT-Bench forecast 2.5–4.0.
- 2.6: Korean chat labelled unmeasured head-to-head; Korean SFT from open teachers with back-translation filtering; ko-Wiktionary added.
- 2.7: concept skeleton kept in weights; fact-dropout applies to the long tail only, with the threshold ablated.
- 3.3: separate judge families for training and eval, calibrated.
- 5.3: CLERK distilled from a strong open embedder and re-distilled per scale.
- 5.4: E2 is a 4-size × 3-store × 3-r grid with a smooth-NLL fit and a predict-85M-from-10–40M test, costing about 5 TPU-hours.
- §6 SCRATCH dropped from v0.
- §7 no Triton; JAX for training, PyTorch for reference and inference; v0 MVP scope defined.

**Minor**
- 1.1: unit error fixed (2.13e14 FLOP/s at 13.5% MFU); all arithmetic moved to one script.
- 2.4: self-tests from docstring examples.
- 3.4: count-vs-fit rule text.
- 3.5: OPEN track never the headline.
- 3.6: decontamination of every synthetic set.
- 5.5: TRC "not strictly 0 KRW" stated.
- Laptop footprint estimated at about 70 GB SSD.

**Where I partly disagree with the red team:**
- 1.5: 100 ms/query is pessimistic for *batched* binary IVF search. I still adopted precomputation because it is free [ESTIMATE].
- 3.1: article-level removal is not enough when one source site supplies the benchmark. I excluded the whole source instead.

**Budget:** 84,557,588 → **93,340,696** (94,981,018 with SCRATCH). Compute 6.1e19 → **4.35e19** FLOPs, with conservative N_eff and smaller S1/S2. Forecasts got worse: 4–6 → **3–5** rows expected to win.
