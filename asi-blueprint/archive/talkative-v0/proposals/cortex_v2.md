# CORTEX-v2: A Neuro-Inspired Recursive Reasoner (90.61M trainable parameters)

Saved to: /home/user/hip/asi-blueprint/proposals/cortex_v2.md. Parameter and FLOP figures were recomputed with a script (`budget.py`, kept in the session scratchpad); every total below matches its output.

Design philosophy, unchanged from v1: a recurrent core that thinks in latent space, a hippocampus-like fast episodic memory, and a cortex-like external retrieval memory, with language decoded at the end. What changed: v2 aims its parameters at the rows v1 was going to lose (fluency, commonsense, code, broad exams). It closes a fair-play loophole and a breach of its own non-Transformer rule, corrects the compute and storage budgets, and adds a scaling experiment that can falsify the core claim before anything large is built.

Labels: [VERIFIED: url] = taken from the research digests or the red-team critique, which checked the primary source. [ESTIMATE] = my arithmetic or judgement. [HYPOTHESIS] = untested design claim.

---

## 1. Thesis: where big models spend their parameters, and how to get those abilities without the parameters

A dense LLM uses its parameters for four jobs at once:

1. **A fact store.** Knowledge capacity is about 2 bits per parameter [VERIFIED: https://arxiv.org/html/2404.05405].
2. **Serial compute depth.** Each layer is one step of thought, and depth costs parameters.
3. **Working memory.** The KV cache, plus the circuits that read it.
4. **A fluency and plausibility prior.** The surface statistics of human text, plus an implicit model of what usually happens in the world.

A 100M model loses all four contests at once. SmolLM2-135M scores 4.1 on TriviaQA and 1.4 on GSM8K [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M]. CORTEX moves jobs out of the weights one at a time:

- **Facts** go to an external, re-indexable datastore (the "cortex"), plus a zero-parameter structured fact store (entity triples).
- **Depth** goes to time: a small weight-tied core iterated up to hundreds of effective layers with adaptive halting. This is the TRM lesson: 7M parameters, 44.6% on ARC-AGI-1 [VERIFIED: https://arxiv.org/html/2510.04871v1].
- **Working memory** goes to *state, not weights*: a delta-rule fast-weight memory plus a test-time-trained memory MLP (the "hippocampus"), written fresh for each query.
- **Exact computation** goes to zero-parameter tools: Python, SymPy, Z3 and a unit-test runner.

**What v1 got wrong.** v1 treated job 4 as something that "stays in the parameters" and then gave it the least design effort. v2 splits job 4 in two and attacks each part:

- **Surface fluency becomes a datastore axis.** The generator emits a mix of tokens and *whole human-written phrases* fetched from a phrase index, in the style of CoG. CoG improved generation quality and got better when the index was enlarged or swapped, with no retraining [VERIFIED: https://arxiv.org/abs/2307.06962]. Fluency then scales with the index, not only with parameters [HYPOTHESIS].
- **Plausibility becomes discrimination, not generation.** A dedicated 12M verifier/reward model ranks candidates: option endings, code candidates, chat drafts. TinyGSM found the verifier contributes more than generator size: a 125M→1.3B verifier added +7.2 points, against +5.1 for the same jump in generator size [VERIFIED: https://arxiv.org/html/2312.09241]. Ranking a handful of candidates is a much smaller job than producing the most plausible continuation token by token [HYPOTHESIS].

The weights then hold a **library of procedures** (decompose, query, bind, compute, verify, express) and a **judge**. They do not hold a library of facts.

**The honest frame.** The milestone is a *campaign* held under 100M parameters, and v2 is the second shot in it, not the final one. v2's estimated P(winning every gated row) is still well under 5% (§5). The campaign does not end in a loss: it ends when every gated row is won with one checkpoint, under the rules in §2, at under 100M. Between attempts, everything except parameter count may grow: datastore, test-time compute (within a published cap), tools, and verified self-generated data. That is the north-star scaling story applied to the mid-boss.

---

## 2. Fair-play rules (binding on this proposal)

1. **Parameters** = every trainable weight executed at inference: embeddings, the shared trunk, core, hippocampal projections, the *initial weights* of the test-time memory MLP, the verifier, every head. The cap is a total under 100,000,000; CORTEX-v2 uses **90,607,378**.
2. **Not parameters, but disclosed** with size, hash and build script: tokenizer, BM25 statistics, IVF centroids, binarization thresholds, datastore text, phrase-index vectors, entity-alias dictionary, Wikidata triples. Every index vector must be produced by the in-budget encoder. Per-query fast weights are state, not parameters: they are created from scratch for each query and discarded afterwards.
3. **Gated-track datastore rule (closes the "LLM cache" loophole).** In the gated track, every datastore item, phrase-index entry and fact triple must be (a) pre-existing human-authored public text or human-curated structured data, or (b) produced by the in-budget model itself. **Content generated by any model above 100M parameters may be used for training only, never stored for retrieval at inference.**
   - This excludes Cosmopedia, any LLM-written exemplar shard and any LLM-extracted triples from the gated store.
   - Web crawls that may contain LLM text are limited in the gated store to Common Crawl snapshots dated **2022 or earlier**. FineWeb-Edu ships per-crawl subsets from CC-MAIN-2013-20 onward, so this cut is mechanical [VERIFIED: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu]. I am assuming pre-2023 crawls hold little LLM text [ESTIMATE].
   - A separate, clearly labelled **"synthetic-store" track** may use LLM-generated store content. It is reported but never counted as the win.
4. **Allowed at inference:** the frozen, dated, offline datastore snapshot; Python, SymPy, Z3 and Lean in a sandbox; a deterministic unit-test runner for tests *the model wrote itself*.
5. **Banned at inference:** any neural network above 100M parameters, any LLM API, live web access, eval items or their solutions in the datastore, hidden benchmark tests during generation, and routing, prompts or heads keyed on dataset identity.
6. **Strict non-Transformer rule, with the family named honestly.**
   - The backbone is Gated DeltaNet: **linear-attention recurrences with a constant-size state, wrapped in the usual block layout** (q/k/v/o projections plus an MLP).
   - The rule is: **no softmax (or Hopfield-equivalent) mixing over token positions or over retrieved items, anywhere in the forward pass.**
   - Allowed:
     - Delta-rule and linear-recurrence reads.
     - MLP and convolution mixing.
     - *Hard* top-k retrieval: a non-differentiable lookup whose results are scanned into the recurrent stream as tokens, never softmax-blended into hidden states.
     - The ordinary output-vocabulary softmax of a generator, which every language model has and which mixes nothing.
   - Contrastive losses over candidates during training are losses, not token mixers, and are allowed.
   - v1's pointer-softmax copy head broke this rule. v2 removes it (§3.2).
7. **Inference-compute cap.**
   - Every query is capped at **2e13 FLOP** of neural compute, plus at most 8 tool calls and 30 s of CPU tool time. FLOPs and wall-clock time are reported per row, following Snell et al.'s FLOPs-matched framing [VERIFIED: https://arxiv.org/abs/2408.03314].
   - For scale: a single pass of a hypothetical 175B dense model over 500 tokens is about 1.75e14 FLOP [ESTIMATE; GPT-3.5's size is unconfirmed]. So even at our cap we spend less per query than that comparison point.
8. **Training-time teachers.**
   - Open-weight models (Apache/MIT/Llama-licensed) may generate *training* data. Every teacher and its size is logged.
   - **No GPT-3.5 or GPT-4 output is used anywhere.** This excludes TinyGSM, MetaMathQA, UltraChat, TinyStories and any ShareGPT/Alpaca-derived data from weights training. It follows OpenAI's terms [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/] and keeps the head-to-head comparison clean.
   - A "no-teacher" ablation is reported alongside.
9. **Public benchmark *train* splits** (WinoGrande-train, HellaSwag-train, ARC-train, GSM8K-train, MBPP's train/prompt split, re-ARC on the ARC training split) are **allowed in the single multi-task mix, and disclosed**.
   - A **no-train-split variant** is trained and reported for every row.
   - I am assuming GPT-3.5's training mix probably included these public sets too [ESTIMATE].
10. **Decontamination**, applied to both training data and the datastore:
    - 13-gram plus MinHash near-duplicate checks, plus an embedding check (cosine > 0.9 with our own encoder).
    - **Character 20-grams for Korean**, because word 13-grams miss short multiple-choice stems.
    - **Function signature + docstring matching for code**, because HumanEval and MBPP are copied all over GitHub.
    - A **domain blocklist**: benchmark hosting sites, Korean license-exam practice sites (the KMMLU/HAE-RAE sources), and WikiHow and ActivityNet, which are dropped entirely because HellaSwag is built from them.
    - **Per-answer retrieval audit:** an item is flagged contaminated if any retrieved chunk overlaps its stem by more than 50% of characters. Scores are reported with and without flagged items.
    - The **private post-cutoff set** (Tier C) is the headline sanity number.
11. **Targets** are the **maximum reported GPT-3.5 score per row across snapshots, sources and prompt languages: max(Ko, En)**. One checkpoint and one configuration are used for all rows.
12. **Scoring is generative everywhere.** Multiple-choice answers are generated letters. Log-likelihood numbers are secondary only (baseline digest §5).

---

## 3. Architecture

### 3.1 Component diagram

```
             ┌──────────────────── CORTEX (non-parametric, human-authored in gated track) ────────────────────┐
             │ DENSE: 20B tok (pre-2023 FineWeb-Edu, en/ko Wikipedia, FineWeb-2-ko, OpenWebMath/arXiv,         │
             │        Python-Edu/Stack-v2 permissive), 128-tok chunks, 256-bit binary codes, FAISS binary IVF  │
             │ BM25 : 2–3B-tok high-value shard only (en/ko Wikipedia, dictionaries, human textbooks), mmap    │
             │ FACTS: Wikidata triples + entity-alias dictionary (from Wikipedia anchor text)  ← symbolic      │
             │ PHRASES: ~200M human-written phrases (en/ko/code), 256-bit codes  ← fluency datastore           │
             └───────▲───────────────────────────────▲─────────────────────────────────▲──────────────────────┘
                     │ query vec + terms              │ entity lookup                    │ phrase query (decode)
                     │ hard top-24 → re-scan top-8    │                                  │ hard nearest neighbour
 prompt ─► [32k ko/en/code unigram tokenizer] ─► SHARED GDN TRUNK (14 blocks, d=512)
                     │      modes: ENCODE = forward + reverse scan (same weights, direction embedding)
                     │             DECODE = causal scan; prompt's causal pass is reused, not recomputed
                     │      span tagger (sigmoid BIO) marks copyable spans; spans get in-stream IDs [S1]…[S64]
                     ▼
     ┌──────── HIPPOCAMPUS (per-query state; projections are parameters) ─────────┐
     │ (a) delta-rule fast weights: 16 heads × (128×128), ONE KEY PER SPAN         │
     │     M_h ← M_h − β_h (M_h k − v) kᵀ                                          │
     │ (b) Titans-style memory MLP 512→1024→512, 2–4 SGD steps on "surprise"       │
     │ read r = Σ_h M_h q_h + MLP_mem(q)  (shared read projections, 5 read sites)  │
     └──────────▲───────────────────────┬──────────────────────────────────────────┘
       write    │                       │ read
     (tool out, │                       ▼
      chunks)   │   ┌──── PREFRONTAL LOOP: latent workspace (variable size) ─────────────┐
                │   │ text: y = 64 answer slots, z = 64 scratch slots                     │
                │   │ grid: one y slot and one z slot per output cell (≤ 900 + 900)       │
                │   │ ONE network, 2 core blocks (TRM-style): bidirectional GDN scan over │
                │   │ slots (axial row/column scans on grids) + SwiGLU + hippocampal read │
                │   │ z ← f(x, y, z, r) ×6 ; y ← f(y, z, r) ×1 = 1 cycle; 3 cycles = 1 step│
                │   │ ≤16 steps; heads: HALT | ROUTE{answer, retrieve, read, tool} | QUERY│
                │   │ | latent VALUE (cheap pre-filter for K latent rollouts)             │
                │   └────────┬───────────────────┬────────────────────┬───────────────────┘
                └────────────┘ re-query / READ   │ tool request       │ y* (top-K latents)
                                                 ▼                    ▼
                                [Python/SymPy/Z3 sandbox pool]  TRUNK in DECODE mode
                                result → ENCODE → hippocampus   emits: vocab tokens | [S_k] span IDs
                                                                | PHRASE(query) → hard NN in phrase index
                                                                      ▼ N candidates
                                                     ┌──── VERIFIER / REWARD MODEL (12.2M, separate) ────┐
                                                     │ 6 bidirectional GDN blocks, d=384, own weights    │
                                                     │ heads: plausibility | code-rank | chat-reward |   │
                                                     │        process-value                              │
                                                     └───────────────────────┬───────────────────────────┘
                                                                             ▼ best candidate → text
```

### 3.2 Why each piece exists (and what changed)

**Shared trunk (critique B1).** v1 had a separate 6-block encoder and a 10-block decoder (55.8M). v2 has one 14-block Gated DeltaNet stack (48.8M) run in two modes:
- **Encode** runs a forward and a reverse scan with the same weights, plus a direction embedding. Its outputs serve as the dense retriever keys, reranker states and hippocampal write keys.
- **Decode** runs a causal scan. The prompt's causal pass from encoding is reused as the decoder's prefix state, so decoding does not re-read the prompt.
- Every token now passes through 14 blocks when generating, against 10 in v1, so the generator is *deeper*.
- Gated DeltaNet beat Mamba2 and DeltaNet on language modelling, commonsense and in-context retrieval [VERIFIED: https://arxiv.org/abs/2412.06464v3].

**Dedicated verifier, 12.2M (critique B1, B2, B7, B8).**
- It is a separate, bidirectional GDN stack at d=384 that shares only the token embedding table.
- It is kept separate so the generator cannot learn to grade its own mistakes as correct.
- Its four heads cover the four losing families:
  - option plausibility (HellaSwag, WinoGrande, COPA, MMLU elimination);
  - code-candidate ranking;
  - chat reward;
  - process value on math steps.
- The TinyGSM result above is why it gets its own parameters.

**Core, rebuilt for variable size (critique B5).**
- v1's 64-slot workspace could not represent a 30×30 ARC output, and its slot-mixing MLP had a fixed width.
- v2 mixes slots with a **bidirectional GDN scan over slots**. On grids it runs **axial scans** along rows and then columns, with shared weights and learned row/column embeddings. This handles any size, has no softmax, and gives every cell a content-dependent path to every other cell in two scans.
- The core is a single 2-block network used for both the z and y updates. TRM showed one small network beats HRM's two modules [VERIFIED: https://arxiv.org/html/2510.04871v1], and the ARC Prize analysis found outer-loop refinement carries most of HRM's gain [VERIFIED: https://arcprize.org/blog/hrm-analysis].
- TRM-MLP, the attention-free variant, collapsed on large grids (ARC-AGI-1 29.6, Maze-Hard 0.0) [VERIFIED: https://arxiv.org/html/2510.04871v1].
- Our two fixes are axial scans (long-range mixing) plus hippocampal reads (content addressing) [HYPOTHESIS; tested in E1b].

**Hippocampus, 4× capacity (critique S2).**
- v1 had 8 heads of 64×64, about 512 interference-free associations, against about 1,500 token writes per query. That was overloaded by a factor of about 3 [ESTIMATE].
- v2 has three fixes:
  1. **One key per tagged span.** The span tagger cuts about 1,500 tokens to about 150–250 spans [ESTIMATE].
  2. **16 heads of 128×128**, about 2,048 associations [ESTIMATE, linear algebra].
  3. A **`READ(chunk_i)` action** that re-encodes [question; chunk] as a fresh trunk pass. The recurrence then knows what to keep while it scans the chunk [HYPOTHESIS].
- The Titans-style memory MLP is kept as the "fuzzy" memory [VERIFIED mechanism: https://arxiv.org/abs/2501.00663].

**Copying without a pointer (critique N1).**
- When retrieved chunks are scanned into the stream, the span tagger's spans are wrapped with reserved in-stream ID tokens, `[S17] … [/S17]`.
- To copy, the decoder emits the *vocabulary token* `[S17]`. A deterministic lookup then expands it to the exact span text.
- The decoder must have stored the ID↔content binding in its recurrent state and hippocampus. Nothing does a content-based softmax over positions. This is pure-RNN, and it also gives DROP exact spans.

**Phrase-level generation (critique B2).**
- At each decode step, the output layer can emit a vocabulary token, a span-ID token, or the special token `PHRASE`.
- After `PHRASE`, a 512→256 projection forms a query and does a **hard nearest-neighbour lookup** in the human-written phrase index. The phrase text is appended to the stream as ordinary tokens.
- Training uses an in-batch contrastive loss over phrase candidates (CoG style) [VERIFIED mechanism: https://arxiv.org/abs/2307.06962]. That is a loss, permitted by rule 6; inference is a hard lookup.
- **Why this matters:** fluency and register (Korean honorifics, idiom, technical phrasing) come from a store of real human sentences that can grow from 200M to billions of phrases without retraining [HYPOTHESIS].

**Structured fact store (critique B6).**
- Wikidata triples, plus an entity-alias dictionary built from Wikipedia anchor text, give zero-parameter entity linking and symbolic lookup for TriviaQA and NQ.
- Wikidata is reported as CC0 [ESTIMATE; the license must be verified on day 1 before inclusion].
- These are human-curated, so they are legal in the gated store.

### 3.3 Inference dataflow, step by step

1. **Perceive.** Tokenize. The trunk ENCODE pass runs forward and reverse, and the causal half is reused later as the decoder prefix.
2. **Link.** The span tagger finds entity mentions. The alias dictionary maps them to Wikidata IDs, and the matching triples are written into the hippocampus as (subject+relation → object) spans.
3. **Recall.**
   - The trunk emits a 256-d query, binarized for IVF Hamming search over the dense store, plus a term set for BM25 over the high-value shard.
   - Reciprocal-rank fusion gives the top 24, reranked by cosine on cached chunk codes (no re-encoding).
   - The top 8 are re-scanned with the trunk and written into the hippocampus, one key per span.
   - For MMLU-style items, retrieval runs **once per option**: the query is [question; option].
4. **Think.**
   - Initialize the workspace (64+64 slots for text, one slot per cell for grids).
   - Run cycles of 6 z-updates and 1 y-update; 3 cycles make one supervision step.
   - After each step, HALT can stop the loop, and ROUTE can trigger a re-query, a `READ`, or a tool call. A tool call means the trunk writes a short program from y, the sandbox runs it, and the output is encoded and written into the hippocampus.
   - Up to 8 tool calls per query.
5. **Deliberate.**
   - K latent rollouts with different slot-noise seeds: K=1 for easy items, up to 16 for hard ones.
   - The cheap latent VALUE head prunes them to the top 4.
6. **Express and judge.**
   - Each surviving y\* is decoded, giving N candidates.
   - **Multiple choice:** the verifier's plausibility head scores every option. Its features include each option's per-option evidence, and the decoder's own letter preference is one more input. The model then generates the winning letter.
   - **Math:** execute the rendered program, vote over executed answers, and break ties with process value.
   - **Code:** sample 20–50 candidates. Generate inputs with the model, run every candidate on them, and cluster candidates by behaviour. Pick the largest consistent cluster, with the verifier's code-rank head breaking ties. Model-written tests also gate the cluster [HYPOTHESIS].
   - **Chat:** sample N=4–8 drafts and pick the best by chat reward. One optional critique→revise pass is allowed, using the same weights.

---

## 4. Parameter budget, exact (d = 512, vocab 32,768; script-checked)

Per-block formulas:
- **GDN block** (d=512, MLP 1408) = q, k, v, o and output-gate projections 5·d² + α/β gates 2·d·8 + short conv 3·d·4 + SwiGLU MLP 3·d·1408 + norms 2d = **3,488,768**.
- **Core block** = bidirectional GDN slot-mixer (5·d² + 2·d·8 + 3·d·4) + SwiGLU 3·d·1408 + norms = **3,488,768**. It uses the same shape as a trunk block so the kernels can be shared; hippocampal reads go through shared projections, counted in row 4.
- **Verifier GDN block** (d=384, MLP 1024) = **1,928,448**.

| # | Module | Composition | Trainable params |
|---|---|---|---|
| 1 | Token embedding (tied input/output) | 32,768 × 512 | 16,777,216 |
| 2 | Shared GDN trunk (encoder + retriever + reranker + decoder) | 14 blocks + direction embedding (2·d) + mode embedding (4·d) | 48,845,824 |
| 3 | Prefrontal core | 2 core blocks + row/column embeddings 2·32·d + y/z init 2·d | 7,011,328 |
| 4 | Hippocampus | writer k/v d→2048 ×2 (2,097,152) + per-head α/β gates d·16·2 (16,384) + shared read q d→2048 and out 2048→d (2,097,152) + 5 read-site gain/bias/norm (10,240) + memory-MLP init 512→1024→512 (1,048,576) | 5,269,504 |
| 5 | Verifier / reward model | 6 GDN blocks d=384 (11,570,688) + input projection 512→384 (196,608) + 4 heads 384→256→1 (394,500) + final norm (768) | 12,163,332 |
| 6 | Small heads | halt (1,026) + router (4,104) + retrieval query 512→256 (131,072) + phrase query 512→256 (131,072) + span tagger 512→3 (1,539) + segment embeddings 16·512 (8,192) + latent value 512→512→1 (263,169) | 540,174 |
| | **Total** | | **90,607,378 (90.61M)** |
| | Headroom to the cap | Reserved: +2 trunk blocks (+6.98M) if E3 shows a fluency floor, or verifier width if E4 says so | 9,392,622 |

Where the budget went, compared with v1:

| Module group | v1 | v2 |
|---|---|---|
| Generator path (trunk) | 34.9M | 48.8M |
| Judge (verifier) | 0.5M | 12.2M |
| Puzzle core | 10.9M | 7.0M |
| Hippocampus | 2.4M | 5.3M |
| Embedding share of total | 19% | 19% |

Gemma-3-270M puts 63% of its parameters in embeddings [VERIFIED: https://developers.googleblog.com/en/introducing-gemma-3-270m/]. Before the 32k vocabulary is frozen, Korean tokenizer fertility is measured on kowiki on day 1 (critique B3). If Korean needs more than 1.6× English tokens per character, the vocabulary goes to 40k: about +4.2M parameters, paid from headroom [ESTIMATE].

Parameter-free: BM25, IVF centroids, the tokenizer, the alias dictionary, the triples, the phrase-index codes, and the tools.

---

## 5. Scoreboard: estimated score, confidence and reasoning (all estimates are [ESTIMATE])

All targets are max(snapshot, source, prompt language). "P(beat)" is my subjective probability that v2, after the full recipe, clears the target plus its 95% CI.

| Row | Target (max) | v1 est. | v2 est. | P(beat) | What changed and why |
|---|---|---|---|---|---|
| ARC-AGI-1 (pass@2) | 5.0 | 25–40 | 15–35 | 0.85 | Variable workspace and axial scans fix v1's representational bug. TTT capped at ≤60 s per task, about 7 T4-hours for 400 tasks. The estimate is lowered because one multi-task checkpoint dilutes TRM's dedicated 4 H100 × 3-day ARC run. **Not a headline.** |
| ARC-AGI-2 (reported, Tier C) | ≈0 | — | 2–6 | — | The meaningful abstraction row. TRM gets 7.8 [VERIFIED: https://arxiv.org/html/2510.04871v1]. |
| NQ-open EM | 38.6 | 36–46 | 40–50 | 0.65 | Entity linking plus Wikidata lookup handles the many single-entity NQ questions symbolically. FiD-base with 440M total reached 48.2 [VERIFIED: https://arxiv.org/abs/2007.01282]. |
| TriviaQA | 85.8 | 62–75 | 68–80 | 0.12 | Triples plus per-entity reads help. GPT-3.5 is strongest here, and Llama-2-7B with a 1.4T store reaches only 77.0 [VERIFIED: https://arxiv.org/html/2407.12854]. |
| DROP F1 | 64.1 | 55–68 | 60–72 | 0.50 | Span-ID copy gives exact spans; arithmetic is executed. |
| KoBEST (avg, max of Ko/En) | 79.3 | (gated on 57.88/40.00) | 65–78 | 0.20 | **Target corrected** from the Ko-prompt subsets to the En-prompt average [VERIFIED: https://arxiv.org/abs/2309.02706]. Per-subset En targets are to be transcribed from that paper's Table 13 before any claim. The v1 claim of an easy win was protocol arbitrage. |
| HAE-RAE | 55.4 | 45–55 | 48–58 | 0.35 | **Target corrected** to 55.4. Lexical subsets become dictionary/kowiki lookup; history and general knowledge depend on Korean store coverage minus the blocklisted exam sites. |
| KMMLU | 42.47 | 38–48 | 40–50 | 0.40 | Per-option retrieval plus verifier elimination. Random guessing is 25. Korean exam-site blocklisting may cost us 2–4 points of honest-but-removed coverage. |
| MATH (4-shot) | 34.1 | 25–38 | 28–40 | 0.40 | SymPy plus process-value search. Qwen2.5-0.5B-Instruct scores 34.4 [VERIFIED: https://qwenlm.github.io/blog/qwen2.5-llm/]. |
| GSM8K | **78.1** | 55–70 | 60–75 | 0.15 | **Target corrected** to 78.1 (0-shot, 1106). Program-of-thought, execution, voting and verifier. TinyGSM-125M reached 63.1 with a banned data source [VERIFIED: https://arxiv.org/html/2312.09241]; we substitute OpenMathInstruct-2 (CC-BY-4.0) [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2]. |
| BBH | 68.3 | 50–65 | 55–68 | 0.30 | Executable subtasks go to tools; disambiguation subtasks go to the verifier. |
| HumanEval / HumanEval+ | 76.8 / 70.7 | 35–50 | 45–62 | 0.05 | Behavioural clustering over 20–50 candidates, API-doc retrieval and code-only RLVR. phi-1-small (350M) scored 45 [VERIFIED: https://arxiv.org/abs/2306.11644]. EvalPlus's GPT-3.5 numbers are very strong. |
| MBPP / MBPP+ | 82.5 / 69.7 | 40–55 | 50–65 | 0.05 | Same mechanism. |
| Natural2Code-style fresh code (Tier C) | 62.3 | — | 40–55 | — | The honest code row, held out with no web leakage [VERIFIED target: https://arxiv.org/abs/2312.11805]. |
| MMLU | 71.4 | 45–58 | 50–62 | 0.08 | Per-option evidence plus verifier elimination; STEM routed to tools. Atlas-770M scored 56.3 [VERIFIED: https://arxiv.org/abs/2208.03299]. **Stated openly: P < 0.2 even with every fix.** |
| ARC-Challenge | 87.4 | 50–65 | 60–75 | 0.05 | ARC-train is allowed (rule 9), plus textbook retrieval and the verifier. |
| HellaSwag | 85.5 | 45–60 | 55–72 | 0.04 | HellaSwag-train plus the plausibility head, trained contrastively on open-teacher minimal pairs. WikiHow and ActivityNet are dropped from the store. |
| WinoGrande | 81.6 | 55–65 | 60–72 | 0.05 | WinoGrande-train plus synthetic minimal pairs plus the verifier. Data alone moved phi-1.5 from 60.4 to 73.4 at 1.3B [VERIFIED: https://arxiv.org/html/2309.05463]. At 100M, no evidence exists. |
| MT-Bench (calibrated local judge) | 8.39 | 3.5–5.0 | 4.5–6.0 | 0.03 | Phrase generation, best-of-N under chat reward, and one critique→revise pass. MobileLLM-125M scored 2.33 [VERIFIED: https://arxiv.org/html/2402.14905]. |
| AlpacaEval 2 LC vs stored GPT-3.5 outputs | 22.35 | 3–10 | 6–14 | 0.04 | Length-controlled only, so a small model cannot game a weak judge with verbosity. |
| **EN chat proxy:** pairwise vs Gemma-2-2B-it, fresh prompts | > 50% win | — | 15–30% | 0.03 | Gemma-2-2B-it (Arena 1163.8) is above gpt-3.5-0125 (1132.3) [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard]. Beating it pairwise suggests GPT-3.5-level chat by approximate Bradley-Terry transitivity [HYPOTHESIS]. |
| **KO chat gate:** Korean MT-Bench vs an open model with Arena-Korean ≥ 1049.6 | > 50% win | — | 10–25% | 0.02 | GPT-3.5-0125 Korean rating is 1049.6 [VERIFIED: same leaderboard]. The open proxy model is read off the pickle on day 1. Korean volunteer raters, blind and both answer orders. |

### Honest verdict

- **P(all Tier A + B rows) for v2 is about 0.1–1%.** v1 was ≈0. Rows that become likely or coin-flip wins: ARC-AGI-1, NQ, DROP, and possibly KMMLU, MATH and HAE-RAE.
- **The F quarter, code and MMLU still lose at 0.1B** unless E3/E4 surprise us.
- **v2 does not claim the milestone.** What v2 changes is that every losing row now has a dedicated mechanism with a kill test (§10), and each mechanism rides an axis that grows without parameters: phrase index, verifier-guided test-time compute up to the cap, and verified self-play.

**Why the remaining walls exist.**

1. **No exact verifier for chat or commonsense.** The learned verifier is itself a 12M network with its own errors. Best-of-N under a weak reward can overfit to the reward model and not improve real quality [ESTIMATE]. Rows affected: chat, HellaSwag, WinoGrande.
2. **Code generation needs a prior strong enough that one of 50 samples is correct.** Test-time compute helps only where the base model has "somewhat non-trivial" success rates [VERIFIED: https://arxiv.org/abs/2408.03314]. Rows affected: HumanEval/MBPP.
3. **Exam knowledge needs reasoning over evidence, not just evidence.** Retrieval adds only about +3.5 on MMLU even at 7B and 1.4T tokens [VERIFIED: https://arxiv.org/html/2407.12854]. Row affected: MMLU.

**How the campaign gets around them, after v2.**

- **(i) Grow the phrase index and the verifier's training data.** Both are free axes.
- **(ii) Self-play on verifiable code until pass@50 is high enough for clustering to work.**
- **(iii) If E0 shows the composite scaling law holds, use the fitted law to decide whether 100M can reach the walls at all.** If the law says no, we will say so in public, and v3 stays < 100M while the walls go to the 1B track. We will not quietly redefine "win" or move the cap.

---

## 6. Training recipe

Order: a **walking skeleton first** (critique E1). The skeleton is the trunk LM, binary retrieval, the Python tool and an eval-harness wrapper. The core, hippocampus, TTT memory, span copy, phrase generation, verifier and self-play are then added **one at a time, each behind an ablation gate**: a component stays only if it improves its target rows at matched FLOPs. Token counts are [ESTIMATE].

| Stage | Objective | Data (store vs train separation enforced by rule 3) | Volume |
|---|---|---|---|
| S0 Build | Tokenizer (fertility test on kowiki), datastore v0, BM25 shard, phrase index, entity aliases, decontamination | Store: pre-2023 FineWeb-Edu-dedup subset, en/ko Wikipedia, FineWeb-2-ko (pre-2023 snapshots), OpenWebMath/arXiv, Python-Edu, Stack-v2 permissive. **No Cosmopedia in the store.** | Dense: 20B tokens → 156M chunks × 32 B ≈ 5 GB codes. BM25: 2–3B-token shard ≈ 15 GB. Text ≈ 80 GB raw / 25–30 GB zstd. Phrases: ~200M × 32 B ≈ 6.4 GB |
| S1 Language + reading | Trunk: causal next-token, plus span denoising in ENCODE mode, plus LLM-JEPA latent loss [VERIFIED effect at fine-tune scale: https://arxiv.org/abs/2509.14252]. **Retrieval augmentation on 20% of documents**, once per 1k-token document (8 chunks scanned into the stream and the hippocampus). Retriever: in-batch contrastive, re-indexed twice. Phrase head: contrastive phrase selection. | 40% FineWeb-Edu, 25% Korean, 15% Python-Edu, 10% FineMath, 10% Cosmopedia (training only) | 25B tokens |
| S2 Latent reasoning | Deep supervision on y at every step; one-step gradient through the last cycle (TRM); Q-learning for HALT. Grid tasks use the variable workspace. | Reasoning Gym [VERIFIED: https://arxiv.org/pdf/2505.24760], re-ARC train split [VERIFIED: https://arxiv.org/pdf/2404.07353], SymPy/Z3 generators, code-execution traces | 12M problems at a 50–70% solve rate |
| S3 Tool + retrieval SFT | Routing, query and `READ` supervision; program-of-thought; span-ID copy; entity linking | KILT provenance triples [VERIFIED: https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/]; OpenMathInstruct-2 rewritten into tool form, keeping only execution-matching items; SmolTalk; Tulu-3 minus NC subsets [VERIFIED: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/main/README.md]; Magpie-style prompts regenerated with open teachers on T4; Korean template translations; disclosed benchmark train splits | ~4B tokens |
| S3v Verifier | Contrastive and ranking losses on four heads: (a) gold vs open-teacher minimal-pair negatives for plausibility (decontaminated; no WikiHow/ActivityNet); (b) executed-pass vs executed-fail for code; (c) local-judge pairwise preferences for chat (judge used at training time only, logged); (d) step correctness from executed math | Our own rollouts plus teacher negatives | ~3M pairs |
| S4 RLVR | GRPO with executor, test and in-process numeric answer rewards; SymPy only on ties | Math, code (self-written tests plus a held-out training-test pool), executable BBH-style tasks, KILT exact-match | **300k curriculum episodes** × 8 samples (was 2M) |
| S5 Self-play | Proposer → solver → verifier. Keep only problems checked by execution. Expert iteration. | Self-generated (in-budget model; also legal for the store under rule 3b) | 10% → 50% of the mix over time |
| S6 Preference / chat | DPO-style on local-judge pairs; critique distillation into the chat-reward head; Korean prompts at 30% | Our own samples on open prompts | ~200k pairs |

---

## 7. Compute plan (free quotas; corrected per critique C1–C5)

Quota figures, both reported by third parties and to be checked in the account:
- TPU v5e-8, about 20 h/week, 9 h sessions [VERIFIED: https://www.kaggle.com/discussions/product-announcements/607202].
- T4×2, about 30 h/week [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits].
- Hardware peaks: v5e chip 197 TFLOPS bf16 [VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e]; T4 65 TFLOPS fp16 [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf].

**MFU is now planned at 12%, not 30%.**
- The JAX tutorial measured 16–27% for a tiny GPT, and 13% with 8-way data parallelism [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html].
- One TPU-week = 1,576e12 × 0.12 × 72,000 s ≈ **1.36e19 FLOP** [ESTIMATE].
- Day-1 tuning targets more: chunk 128, head dim 128, 4-way data parallelism with a larger per-device batch.

| Item | FLOP | TPU-weeks @12% | Notes |
|---|---|---|---|
| S1: 25B tokens × 5.11e8 FLOP/token | 1.28e19 | 0.94 | 6 × (trunk 48.8M + output embedding 16.8M), plus 20% of documents with 1:1 neighbour tokens through ENCODE (fwd + rev) |
| S2: 12M problems × 6 steps × 2.0e11 | 1.44e19 | 1.06 | Step = 3 cycles × 7 core applications over 128 slots, + one-step backward, + trunk ENCODE of a 512-token problem |
| S3: 4B tokens, trunk + verifier | 1.9e18 | 0.14 | |
| S3v + S6 | ~1.5e18 | 0.11 | |
| S4: 300k × 8 rollouts × 5.7e11 × 3 | 4.1e18 | 0.30 | Rollout = ENCODE 1.5k tokens, 8 latent steps, 300-token decode; ×3 for training |
| **E0 scaling grid** (§10) | ~1.4e19 | ~1.0 | 27 small runs |
| E1–E4 experiments | ~2e19 | ~1.5 | |
| **Subtotal** | ~7e19 | **~5.1** | |
| Failed runs, restarts, reindexing (×1.8) | | **~9** | |

**Calendar:** about **20 weeks (≈5 months)**. TPU quota is the critical path. T4×2 runs in parallel for teacher data, evals, index builds and rollouts. Volatility in v5e availability [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab] could stretch this; a T4-only fallback costs about 8× throughput [ESTIMATE].

**CPU walls, fixed:**
- **S1 retrieval:** retrieval runs once per 1k-token document and only on the 20% of documents that are augmented. That is about 25M docs × 0.2 = 5M batched queries against the binary IVF only, with no BM25 in training. At 1–5 ms per batched query that is about 1.5–7 CPU-hours [ESTIMATE], against v1's implicit 1,700–8,700.
- **S4 tools:** in-process numeric checks; a pre-forked sandbox pool on the TPU host's vCPUs (count checked on day 1); 300k episodes. That is about 2.4M rollouts × ~0.5 tool calls × 0.2 s ≈ 70 CPU-hours [ESTIMATE].
- **Eval:** a full Tier A run takes about 70k items. About 23k of them are tool-heavy, at ~3 CPU-s each, × 3 seeds for sampled rows. **That is 5–15 h per full eval** [ESTIMATE]. It runs on Kaggle CPU sessions and the laptop through an asynchronous tool queue. A 10% stratified "dev panel" is used between full runs.

**Storage, corrected (critique C3):**

| Item | Size | Location |
|---|---|---|
| Dense binary codes, 20B tokens | 5 GB | Kaggle dataset (read-only mount) |
| BM25 over the 2–3B-token high-value shard | ~15 GB | Kaggle dataset, mmap |
| Chunk text | 25–30 GB zstd | Kaggle dataset |
| Phrase index | 6.4 GB | Kaggle dataset |
| Wikidata subset | ~5–10 GB [ESTIMATE] | Kaggle dataset |
| **Total** | **≈60–65 GB** | Within the 100 GB private dataset quota (unconfirmed [VERIFIED claim varies: https://subscription.packtpub.com/book/data/9781835083208/3/ch03lvl1sec16/gathering-the-data]) |

Stores beyond 20B tokens (the campaign's free scaling axis) live on the laptop SSD and HF public datasets, and are used for **inference only**. HF free private storage is 100 GB [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits].

**Engineering (critique E1, E2):**
- **One framework: JAX.**
  - The delta rule is written as pure `jnp` chunkwise WY-form matmuls in `lax.scan`.
  - Training loops use a fixed maximum number of supervision steps with halting masks, so XLA shapes stay static.
  - Tools run host-side only, between jitted steps.
  - This drops Triton entirely, and with it the Turing toolchain problem (Triton 3.3+ requires compute capability 8.0+ [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md]).
  - JAX's CUDA support for sm_75 (T4) must be confirmed on day 1 [ESTIMATE]. If it fails, the T4 side falls back to *eager* pure-PyTorch chunked scans with no Triton, used only for evaluation and rollouts, and checked for parity against JAX on the 3060.
- **T4 numerics:** fp16 matmuls; fp32 recurrent state, gates and accumulators; loss scaling.
- **Kaggle operations:**
  - Assert the device type at the start of every run, because CLI pushes may silently land on a non-TPU image [VERIFIED: https://github.com/Kaggle/kaggle-cli/issues/1197].
  - Checkpoint every 45 minutes to the HF Hub.
  - Chain sessions from the last checkpoint.
  - One account only.
- **Day 1:** apply to TPU Research Cloud [VERIFIED: https://sites.research.google/trc/faq/].

---

## 8. Inference cost per query [ESTIMATE, script-checked]

| Phase | Hard query | Easy query | Notes |
|---|---|---|---|
| Trunk ENCODE: prompt + 8 chunks, ×3 retrieval rounds | 8.8e11 | ~1e11 | fwd + rev |
| Rerank top-24 | ~0 neural | ~0 | Cosine on **cached** chunk codes; the v1 plan re-encoded 50 chunks (≈3.8e12) |
| `READ` re-scans (≤4 × 1k tokens) | 7.8e11 | 0 | |
| Latent loop, K=16 × 16 steps × 3.0e10 | 7.7e12 | 1.2e11 (K=1, 4 steps) | Batched as one 2,048-row matmul per application |
| Decode: 10 candidates × 400 tokens | 5.3e11 | 4e10 | Code uses up to 50 short candidates, about the same total |
| Verifier on 10 candidates | 1.5e11 | 1.5e10 | |
| **Neural total** | **~1.0e13** | **~2.6e11** | Under the 2e13 cap (rule 7) |
| ARC grid task: 1,800 slots, 16 steps, pass@2 | ~1.4e13 per rollout pair | | TTT ≤60 s per task, outside this budget but wall-clock-reported |
| Tools | ≤8 calls, ≤30 s CPU | 0–1 call | Dominates math/code latency |

**Latency (critique L1, accepted):**
- Hard queries take **10–20 s** on one T4: serial tool calls, retrieval rounds, and kernel-launch overhead at batch 1.
- Mitigations: CUDA-graph or `jit` capture of the core step, K rollouts batched into one matmul, and a hard tool budget.
- Wall-clock time is reported per row next to the score.

---

## 9. Scaling story: what grows, and how

The core bet [HYPOTHESIS] is that capability ≈ f(procedure parameters, datastore tokens, phrase-index size, recursion depth × verifier quality, hippocampal capacity, tool coverage, verified self-generated data). Every axis except the first grows **without retraining the core**. v2 adds what v1 lacked: **a measured fit of this function (E0) before any scale-up**, so "scalable" becomes a falsifiable prediction.

| Axis | 0.1B (v2) | 1B | 10B | Evidence |
|---|---|---|---|---|
| Procedure parameters | d=512, 14-block trunk | d=1536, 24-block trunk, 4-block core | d=4096 | RWKV-7 improves monotonically from 0.19B to 2.9B [VERIFIED: https://arxiv.org/abs/2503.14456] |
| Verifier | 12M | 150–250M (a fixed ~15–25% share) | ~1.5B | TinyGSM: verifier size beats generator size [VERIFIED: https://arxiv.org/html/2312.09241] |
| Fluency (phrase index) | 200M phrases | 2B | 20B+ | CoG improves when the index is enlarged, with no retraining [VERIFIED: https://arxiv.org/abs/2307.06962] |
| Datastore | 20B tokens trained against; more at inference | 1T | Multi-T + PubMed / UniProt / ChEMBL shards (after dual-use review) | RETRO: 25× parameter savings, gains do not diminish 150M→7B [VERIFIED: https://arxiv.org/abs/2112.04426]; MassiveDS: no saturation at 1.4T [VERIFIED: https://arxiv.org/html/2407.12854] |
| Structured facts | Wikidata subset | Full Wikidata + domain KGs | Science KGs | [HYPOTHESIS] |
| Sparse parametric memory | none | Product-key memory layers (count against the budget) | 10B+ memory parameters | Memory layers beat dense models with >2× the compute [VERIFIED: https://arxiv.org/abs/2412.09764] |
| Recursion depth | ≤16 steps × 21 applications | ≤64 steps | Budgeted per query | Huginn: "50B-equivalent" from depth [VERIFIED: https://arxiv.org/abs/2502.05171] |
| Hippocampus | 16 × 128×128, span keys | Larger heads; consolidation into the store ("sleep") | Lifelong episodic store | Titans: context beyond 2M [VERIFIED: https://arxiv.org/abs/2501.00663] |
| Tools | Python, SymPy, Z3 | Lean, simulators | Docking, molecular dynamics, retrosynthesis (human-gated) | AlphaProof/AG2 [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/] |
| Self-generated data | 10–50% of the mix | Majority | Science-environment rollouts | DeepSeek-R1-Zero RL [VERIFIED: https://arxiv.org/html/2501.12948] |

**Fallback path (critique S3).** Every exotic module (latent core, TTT memory, phrase generation) is optional behind its ablation gate. If any of them fails its kill test, what remains is a GDN language model plus retrieval, tools and a verifier. That combination rests only on evidenced scaling axes: monotone recurrent-LM scaling [VERIFIED: https://arxiv.org/abs/2503.14456] and retrieval gains that do not diminish [VERIFIED: https://arxiv.org/abs/2112.04426].

**Compute wall.**
- 1B at Chinchilla scale is about 1.2e20 FLOP, about **9 TPU-weeks at 12% MFU**.
- 10B is about 1.2e22, or **about 7 years of free quota** at the compute digest's assumptions, and longer at 12% MFU [ESTIMATE].
- 10B therefore needs TRC or grants, earned by publishing v2 and the E0 law openly.

**North-star logic.** Scientific discovery is a verifier-rich environment (simulations, proofs, assays). There, a latent recursive reasoner plus a learned and exact verifier plus tools improves with compute rather than with human data [HYPOTHESIS]. v2's dedicated verifier is the piece that transfers most directly: the north-star system is a proposer–verifier loop over hypotheses.

**Biosafety hard stop.**
- **v1 and v2 ship no bio shards and no bio tools.**
- Adding them is gated behind an **external dual-use review**, not self-review.
- When they arrive:
  - The store and the tool registry exclude pathogen-enhancement and toxin-synthesis content.
  - The router has a refusal class for hazardous chemistry and biology.
  - Any wet-lab-actionable output needs human expert review.
  - The "thought probe" decoder, which renders y and z at every step, is mandatory for auditing latent reasoning on bio tasks.

---

## 10. First experiments: go/kill criteria

Order: E0 runs on the walking skeleton plus minimal modules. E1–E4 can run in any order once E0's grid is training.

**E0: Measure the composite scaling law before building anything big** (~1 TPU-week; critique S1)
- Grid: {10M, 25M, 50M} parameters × store {0, 1B, 10B tokens} × recursion r ∈ {1, 4, 16}. That is 27 runs at about 20 tokens per parameter, plus a fixed S2-mini reasoning curriculum.
- Fixed panel: NQ dev, a GSM8K-train holdout, Reasoning Gym held-out generators, a HellaSwag-val subset, a KMMLU dev subset and an MT-Bench-style local-judge set.
- Fit per-row power laws in each axis, plus their interactions.
- **Publish the predictions for 90M (v2) and 1B before training them.**
- **Go:** each axis gives a monotone gain with a fit residual < 2 points, and the 90M prediction is later confirmed within its interval.
- **Kill / redirect:** if store size gives no gain on the reasoning panel, or r gives no gain on language rows, those axes are declared non-scaling for those rows, in public, and their parameters are reassigned.

**E1: Does latent recursion transfer to language reasoning, and does the variable workspace fix grids?** (3060 + ~0.7 TPU-week)
- E1a: a 20M core at r ∈ {1, 4, 16} against a 50M plain GDN using token chain-of-thought, at equal training FLOPs. Panel: Reasoning Gym held-out, GSM8K program-of-thought (train → test).
- E1b: Maze-Hard and ARC-AGI-1 eval with axial scans and hippocampal reads, against an axial-only ablation.
- **Go:** the 20M model at r=16 is at least as good as the 50M baseline, accuracy rises monotonically in r, and E1b Maze-Hard is above 50 (TRM-MLP: 0.0).
- **Kill:** an r=1→16 gain under 3 points, or Maze-Hard under 20. Then drop latent recursion, use token chain-of-thought, and give the core's 7M to the trunk.

**E2: Exact recall without attention** (~0.7 TPU-week)
- A 50M model plus a 4B-token Wikipedia store, with span keys, 16×128² heads and `READ` re-scans.
- Measures: NQ-open EM, recall@8, and a synthetic key→value recall probe at 1–8k tokens.
- **Go:** NQ EM ≥ 33, recall@8 ≥ 70%, probe ≥ 90% at 4k.
- **Kill:** NQ < 25 or probe < 60%. Then the hippocampus becomes multi-bank (4 banks, routed by span type) and all reading goes through `READ` re-scans. If that also fails, the reader design is declared the blocker for knowledge rows.

**E3: Fluency floor, and does the phrase index move it?** (full 90.6M through S3 + S6; ~2 TPU-weeks, shared with the main run)
- Arms: token-only decoding vs token+phrase decoding, each at phrase-index sizes {20M, 200M}.
- Judges: pairwise against the stored GPT-3.5 MT-Bench answers [VERIFIED exists: https://huggingface.co/spaces/lmsys/mt-bench] and AlpacaEval outputs [VERIFIED exists: https://github.com/tatsu-lab/alpaca_eval]. Also pairwise against **Gemma-2-2B-it on fresh prompts**, and Korean MT-Bench against the Korean proxy model.
- The local judge is calibrated first: rank correlation with the published MT-Bench scores of at least 5 models whose stored answers exist must reach Spearman ρ ≥ 0.8, or no chat number is reported.
- **Go:** phrase decoding beats token decoding by at least 5 points of win rate; growing the index 10× gives at least +3; win rate against Gemma-2-2B-it is at least 20%. Then spend the headroom on 2 trunk blocks.
- **Kill:** the proxy win rate stays under 20%, or the index-size gain is under 1 point. Chat is then declared the v-next (1B) blocker, in public.

**E4: Does a 12M discriminator move commonsense at all?** (~0.3 TPU-week)
- Verifier-only training on WinoGrande-train, HellaSwag-train and synthetic minimal pairs. Eval on dev splits with generative-letter scoring, plus a no-train-split arm.
- **Go:** WinoGrande dev ≥ 62 and HellaSwag dev ≥ 55. Every published model ≤270M scores 51–53 on WinoGrande [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M]. Then test 12M vs 21M of verifier width from headroom.
- **Kill:** WinoGrande < 56. Then plausibility at 100M is declared parameter-bound by our data, and those rows go to the 1B track in public.

---

## 11. Top risks

1. **The F quarter, code and MMLU probably remain unwinnable at 0.1B.** v2 attacks each with a mechanism and a kill test, but no published evidence says 100M can get there. This is the main reason the milestone is a campaign, not a v2 promise. E0, E3 and E4 settle it with data.
2. **Latent recursion may not transfer from grids to language** [HYPOTHESIS]. All TRM/HRM evidence is on puzzles. Covered by E1, with the fallback path in §9.
3. **Reward hacking of the learned verifier.** Best-of-N under a 12M reward model can raise reward without raising quality [ESTIMATE]. Mitigations: train the verifier on executed outcomes wherever possible, and report the calibrated-judge correlation alongside every number.
4. **Fixed-state recall.** Covered by E2, with the multi-bank fallback.
5. **Contamination fakes wins.** RETRO exploits leakage more than its baseline [VERIFIED: https://arxiv.org/abs/2112.04426]. v2 adds Korean char-20-grams, code signature matching, the domain blocklist, the 50% stem-overlap audit, and reporting with and without flagged items. The private post-cutoff set is the headline.
6. **Phrase-index licensing.** Human-authored web text, Wikipedia (CC BY-SA [VERIFIED: https://download.wikipedia.com/legal.html]) and code have mixed licenses. Phrase-level reuse of BY-SA text in outputs may need attribution [HYPOTHESIS]. Keep source IDs on every phrase so outputs can be attributed.
7. **Local-judge validity.** Gated on ρ ≥ 0.8 calibration (E3), length-controlled variants only.
8. **Quota volatility and the 5-month calendar.** v5e is sometimes oversubscribed [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab]. T4-only fallback; TRC application on day 1.
9. **Toolchain.** JAX on T4 is unverified [ESTIMATE]; fallback is eager PyTorch with no Triton, plus a parity test.
10. **Korean data.** 35.3% of Korean datasets on HF have unknown licenses [VERIFIED: https://arxiv.org/pdf/2507.04329]. AI Hub data is not redistributable, so it is eval-only or skipped. Korean volunteer raters are a real-world bottleneck for a solo builder at 0 KRW.
11. **Opacity of latent thought.** The thought probe is mandatory for any north-star bio work.
12. **Benchmark gravity.** The temptation to tune toward the rows. Countermeasures: Tier C held-out sets, ARC-AGI-2 and Natural2Code-style fresh code reported every release, and the E0 predictions published in advance.

**One-line summary:** CORTEX-v2 keeps facts in the store, depth in time, working memory in state and computation in tools. It now also moves *fluency* into a human-written phrase index and *plausibility* into a dedicated judge. It closes the LLM-cache loophole, deletes its own hidden attention, budgets compute honestly (about 9 TPU-weeks, about 5 months), and measures its scaling law before betting on it. It is still expected to lose the commonsense, chat, code and MMLU rows at 0.1B, and it says exactly which experiment will prove or kill each attack.

---

## Changelog (v1 → v2)

| Critique item | Severity | Resolution |
|---|---|---|
| P1 LLM-cache loophole | FATAL | Rule 3: the gated store is human-authored or self-generated only, with pre-2023 crawls; Cosmopedia and LLM-written exemplars are training-only; a separate labelled synthetic-store track. All index vectors come from the in-budget encoder. |
| B1 budget aimed at already-won rows | FATAL | Encoder and decoder merged into a 14-block shared trunk (55.8M → 48.8M; generation depth up from 10 to 14 blocks). New 12.2M dedicated verifier with 4 heads. Core cut to 7.0M. Total 90.61M, 9.39M headroom. |
| N1 copy head was attention | MAJOR | Replaced by a sigmoid span tagger, in-stream span-ID tokens emitted from the ordinary vocabulary, and deterministic expansion. Rule 6 is restated to ban softmax mixing over positions and retrieved items while allowing hard top-k lookup. |
| N2 family naming | MINOR | GDN is described as a linear-attention recurrence in Transformer block layout, with no softmax token mixing and a constant-size state. |
| B2 no fluency mechanism | MAJOR | Phrase-level generation over a human-written phrase index (hard NN, contrastive training); best-of-N under the chat-reward head; E3 tests index-size scaling. |
| B3 no Korean chat gate | MAJOR | Korean MT-Bench gate against an open proxy with Arena-Korean ≥ 1049.6, with blind volunteer raters; day-1 tokenizer fertility test with a 40k-vocabulary contingency. |
| B4 cherry-picked targets | MAJOR | max(Ko, En) everywhere: KoBEST 79.3, HAE-RAE 55.4, GSM8K 78.1. P(beat) revised down where needed. |
| B5 ARC workspace mismatch | MAJOR | Variable workspace (one slot per cell), axial bidirectional GDN scans, TTT ≤60 s per task, ARC-AGI-2 as the meaningful row, E1b Maze-Hard gate. |
| B6 knowledge rests on retrieval alone | MAJOR | Wikidata triples + alias-dictionary entity linking (license to verify); per-option evidence plus verifier elimination for MMLU; MMLU P < 0.2 stated. |
| B7 code near-certain loss | MAJOR | 20–50 candidates, behavioural clustering on model-generated inputs, API-doc retrieval, code-only RLVR, Natural2Code-style fresh row in Tier C. |
| B8 thin commonsense data plan | MAJOR | Rule 9: public train splits allowed and disclosed, with a no-train-split variant; E4 discriminator kill test. |
| P2 contamination gaps | MAJOR | Korean char-20-grams, code signature+docstring matching, blocklist (exam sites, WikiHow, ActivityNet, benchmark hosts), 50% stem-overlap audit, with/without reporting. |
| P3 judge validity | MINOR | Calibration on stored answers with ρ ≥ 0.8 gate; length-controlled only. |
| C1 MFU and compute understated | MAJOR | 12% MFU, FLOP/token recomputed (5.11e8), explicit E0–E4 budget, ~5.1 TPU-weeks nominal and ~9 with failures, ~20-week calendar. |
| C2 S1 retrieval CPU wall | MAJOR | Once per 1k-token document, 20% of documents, binary IVF only: ~5M queries, ~1.5–7 CPU-hours. |
| C3 BM25 storage wrong by >10× | MAJOR | BM25 restricted to a 2–3B-token shard (~15 GB); total store ≈60–65 GB; >20B tokens for inference only on SSD/HF. |
| C4 RLVR CPU wall | MAJOR | S4 cut to 300k episodes, in-process checks, pre-forked sandbox pool; ~70 CPU-hours. |
| C5 eval undercounted | MINOR | 5–15 h per full eval, CPU sessions, async tool queue, 10% dev panel; reranking moved to cached codes. |
| S1 scaling law never measured | MAJOR | E0: 27-run grid, power-law fit, predictions for 90M and 1B published before training. |
| S2 hippocampus too small | MAJOR | Span keys, 16 heads × 128×128, `READ` re-scan action; multi-bank fallback in E2. |
| S3 10B wall | MINOR | TRC on day 1; every exotic module optional; fallback path on evidenced axes; 1B ≈ 9 TPU-weeks stated. |
| L1 latency | MINOR | 10–20 s for hard queries accepted; jit/CUDA-graph capture, batched rollouts, a per-query 2e13-FLOP cap, wall-clock reported. |
| E1 too many moving parts | MAJOR | JAX only (Triton dropped), walking skeleton first, each module behind an ablation gate; eager-PyTorch T4 fallback. |
| E2 Kaggle operations | MINOR | Device assert per run, HF checkpoints, T4-only fallback schedule. |
| Biosafety | MINOR | No bio shards or tools in v1/v2; external dual-use review required before adding them. |

Disagreements with the critique: none on substance.
- **B1 merge.** I adopted it, but kept a separate verifier stack rather than putting the verifier on the shared trunk, so the generator cannot grade its own errors as correct. This costs 0.2M for an input projection.
- **Phrase lookup and rule 6.** The critique did not raise this. The phrase lookup is a hard nearest-neighbour retrieval at inference, the same operation as datastore retrieval; it is not softmax mixing. I flag it explicitly in rule 6 so a reviewer can disagree on the record.
