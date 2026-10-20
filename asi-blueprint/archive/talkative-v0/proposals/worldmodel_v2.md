# Proposal WM-2 "ORRERY v2": Latent World Model + Energy-Based Reasoning (red-team revision)

Author role: chief architect for the "latent world model / energy" philosophy. Date: 2026-10-19.
Labels: [VERIFIED: url] = taken from the research digest or a primary source it cites; [ESTIMATE] = my arithmetic or judgement; [HYPOTHESIS] = untested design claim.

Saved to: /home/user/hip/asi-blueprint/proposals/worldmodel_v2.md (supersedes worldmodel_v1.md)

**What changed in one paragraph.** v1 hid ~35M parameters in a stale shared retriever, choked its latent plan through a 512-number FiLM, and served LLM-written text through a softmax copy pointer. v2 merges encoder and decoder into **one causal Gated DeltaNet trunk** (d=640) whose frozen lower stack *is* the retriever, feeds the plan in as **soft prefix tokens** (0 extra params), restricts the headline datastore to **human-authored text**, deletes the pointer, adds a **4M attention-free grid reasoner**, recomputes FLOPs honestly (about 2x v1's throughput claim was wrong), and inverts the build order: **ORRERY-0, a boring strong baseline, is built first, and every latent component must beat it at equal FLOPs or be cut.** Total: **95,813,058** trainable parameters.

---

## 1. Thesis

A big LLM spends its parameters on three jobs at once:

- (a) **Storing facts.** At best about 2 bits per parameter [VERIFIED: https://arxiv.org/html/2404.05405].
- (b) **Amortizing reasoning** into one forward pass per token. An autoregressive model cannot "think again" about its answer except by writing more tokens.
- (c) **Surface fluency.** It models every output token, including the many that carry no decision.

ORRERY takes each job out of the weights:

- **Facts** go to a frozen, decontaminated, **human-authored** datastore (0 parameters, grows with disk).
- **Reasoning is not amortized.** The answer is a point in latent space found by iterative optimization: denoising plus descent on a learned energy E(context, z). Test-time FLOPs, not parameters, buy depth. This is the TRM/Huginn lesson, moved from grids to language: 7M recursive nets beat frontier LLMs on Sudoku and ARC [VERIFIED: https://arxiv.org/html/2510.04871v1]; a 3.5B recurrent-depth model reaches "50B-equivalent" compute [VERIFIED: https://arxiv.org/abs/2502.05171].
- **Fluency** is the job of the same trunk running in generation mode, steered token-by-token by the plan (now actually wired in, see F2 fix), and reading retrieved human text that is already in its stream.

One scalar, the energy, is the system's universal currency. It ranks multiple-choice options, chooses between candidate plans and decides when to stop thinking. It is trained to agree with exact verifiers (Python, SymPy, Z3, unit tests) wherever those exist.

The bet [HYPOTHESIS]: **a discriminative energy over latents uses parameters more efficiently than generative token likelihood.** Judging is easier than generating, and at small scale the verifier matters more than generator size (TinyGSM: "the verifier seems to play a bigger role than generation model size" [VERIFIED: https://arxiv.org/html/2312.09241]).

**v2 adds a discipline to the bet.** The bet is only worth anything if it beats the obvious alternative at equal compute: a recurrent LM with retrieval, sampling and exact verification. v2 therefore builds that alternative first (ORRERY-0, Section 3.0) and treats every latent component as an *add-on that must earn its FLOPs*. If latent reasoning loses to ORRERY-0, the project keeps ORRERY-0 and says so publicly.

**Campaign framing (red-team strategic note, accepted).** The user's rule is "losing is not acceptable." The honest response is not to claim v1 wins; it is to treat the mid-boss as a **campaign of rounds**, where every red row has (1) a named knob that does not add parameters, (2) a measured slope on that knob, and (3) a forecast round in which it turns green, falsifiable by the slope. Section 5 contains that table. If a row's slope is flat after the last sub-100M round, we publish the loss and the slope. We do not redefine the boss.

---

## 2. Fair-play rules ORRERY operates under

1. **Parameters** = every trainable weight that runs at inference: embeddings, trunk (including its frozen lower layers, counted once), retrieval projection, reasoning core, energy head, plan adapter, initial weights of the test-time memory, grid reasoner, halting/tool head and norms. Total must be < 100,000,000. ORRERY v2 uses **95,813,058** (Section 4). **Frozen weights still count.** There is no "frozen copy" anywhere: the retriever is the same tensors as trunk layers 1–4.
2. **Training-only weights are disclosed but not counted.** The EMA target encoder for the JEPA loss never runs at inference. Every teacher and judge model used at training time is named with size, family and license.
3. **Data, not parameters:** the tokenizer, BM25 statistics, IVF centroids, PQ codebooks, binary codes, the text datastore. Learned codebooks are disclosed with their sizes.
4. **Allowed at inference:** the frozen, dated, hashed local datastore; offline Python sandbox, SymPy, Z3, Lean and other deterministic solvers on CPU (0 parameters).
5. **Headline datastore = human-authored text only (red-team F3, accepted).** Wikipedia, FineWeb-Edu / FineWeb-2 web text, StackExchange-style human Q&A, arXiv, permissively licensed human code. **No LLM-generated text** (Cosmopedia, SmolTalk, Magpie, OpenMathInstruct-2 solutions, our own synthetic data) may be retrievable at inference in the gated track, because retrieving a 47B model's prose at inference is a cache of a >100M network [VERIFIED that Cosmopedia is Mixtral-8x7B-generated: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia; Magpie is Llama-3-generated: https://arxiv.org/abs/2406.08464v1]. A disclosed "synthetic-store" track may be reported but never gates. Every chunk carries a provenance tag; every reported score includes the share of output tokens that are verbatim copies (≥8-gram) from each source class.
   - Note on FineWeb-Edu: the *text* is human web text; only the *quality label* came from an LLM classifier. That counts as human-authored text selected by a filter, not generated text [ESTIMATE, flagged for the referee].
6. **Banned:** any neural net >100M at inference or any LLM API; live web search; eval items or their solutions in training data or datastore; routing on dataset identity; benchmark-specific prompts or heads; running hidden tests during generation; **any GPT-3.5/GPT-4 output in training data** (OpenAI terms forbid it [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/], and it would contaminate the head-to-head).
7. **Benchmark training splits (red-team M6, accepted and extended).** The **gated checkpoint is trained with zero train splits of any multiple-choice benchmark on the scoreboard** (HellaSwag, WinoGrande, ARC, KoBEST, MMLU auxiliary-train, KMMLU-style exam data). GPT-3.5's numbers are few-shot; a discriminative scorer fine-tuned on ~40k in-distribution items would be a different contest. Train splits of *generative procedure* benchmarks (NQ/TriviaQA via KILT, GSM8K, MATH, MBPP-train) are allowed in the gated checkpoint because they teach transferable procedures, and this is disclosed. A second, fully split-free ablation is also reported for every row. One checkpoint, one config, for every gated row.
8. **Input-type routing is allowed, dataset routing is not.** A deterministic detector sends inputs that are 2-D grids of small integers to the grid reasoner (Section 3.4). The detector looks at the input's *shape*, never at a dataset name or prompt template. Its code is published, and its routing decisions on every eval row are logged (expected: 100% on ARC-AGI, 0% elsewhere).
9. **Transformer rule (red-team M10, made explicit):** *No softmax (or Hopfield-equivalent [VERIFIED equivalence: https://arxiv.org/abs/2008.02217]) normalization over token positions, whether those positions come from the input, the retrieved chunks or generated text, anywhere in the inference path.* **Explicitly permitted:** linear-attention / delta-rule / state-space recurrences with constant-size state (Gated DeltaNet, DeltaNet, RWKV-7, Mamba), convolutions, MLPs (including token-mixing MLPs over fixed grids), test-time-trained memories, BM25 and Hamming/IVF search, argmin over a small set of candidate answers. v1's copy pointer (softmax over 2,048 retrieved positions) **is deleted**: the red team was right that "external items" is a thin defense when those items already went through the encoder. Softmax inside *training losses* (InfoNCE over candidates, cross-entropy over the vocabulary) is a loss, not a sequence mixer, and is allowed.
10. **Disclosure per query:** FLOPs, wall-clock, candidates, refinement steps, tool calls, retrieval provenance and gold-answer-in-top-k flag (Section 6, S0). A test-time-compute win without FLOP disclosure does not count (the Snell et al. standard [VERIFIED: https://arxiv.org/abs/2408.03314]).
11. **Scoring is generative everywhere.** For multiple choice, options are scored internally by energy, then the trunk writes the answer letter, which is exact-matched.
12. **Judges.** The preference judge used at training time (S5) and the evaluation judge come from **disjoint model families** (e.g., if S5 uses a Qwen-family judge, evaluation uses a Llama- or Gemma-family judge) [ESTIMATE: specific checkpoints chosen by license and T4 fit in week 1]. Plus a blind hand spot-check of ~100 pairs by the author (Section 5).

---

## 3. Architecture

### 3.0 Build order: ORRERY-0 first

ORRERY-0 is the baseline every latent component must beat at equal FLOPs (red-team engineering-risk fix, accepted in full). It is not a fallback; it is the *control arm*.

| Component | ORRERY-0 | ORRERY-L (full v2) |
|---|---|---|
| Trunk | causal GDN, d=640, 10 layers, 32k vocab | same tensors |
| Retrieval | BM25 + frozen-trunk dense binary index, RRF | same |
| Tools | Python, SymPy, Z3 sandbox; tool calls in stream | same + latent prediction of tool results |
| Answer selection | sample N text candidates, exact verifier where it exists, else **energy reranker over decoded text** (energy head reads trunk state after each candidate) | latent plans: denoise + energy descent, render best plan |
| Grid inputs | grid reasoner (it is cheap and independent) | same |
| Extra modules | none | latent core R, plan adapter, test-time memory M, JEPA loss, diffusion loss |

Both arms share S0, S1, S3 and S4 infrastructure. They branch at S2. The comparison is scored at equal **inference FLOPs per query** and equal **training FLOPs after S1**.

### 3.1 Component diagram

```
 user prompt
     |
     v
+------------------------------------------------------------------------------+
| TOKENIZER  32k BPE, Korean-aware (0 params)                                  |
+--------------------------------------+---------------------------------------+
                                       |
          (A) query pass: embed + trunk layers 1-4 (FROZEN after S1a)
                                       |  mean-pool -> proj 640->512 -> sign()
                                       v
+------------------------------------------------------------------------------+
| RETRIEVER (0 params beyond the 512-proj)                                     |
|  BM25: on-disk compressed inverted index (mmap), Korean char-n-gram/morph    |
|  Dense: 512-bit binary codes in RAM, IVF, PQ/int8 rescoring of top-1000      |
|  Keys were produced by THE SAME frozen layers 1-4 + proj  -> no drift         |
|  RRF fusion -> top-K human-authored chunks (K = 4-16, 128 tok, provenance)   |
+--------------------------------------+---------------------------------------+
                                       |
          (B) main stream:  [chunk_1 ... chunk_K] [prompt] [<plan> Z* slots] [answer ...]
                                       v
+------------------------------------------------------------------------------+
| TRUNK  causal Gated DeltaNet, 10 layers, d=640, 10 heads x 64                |
|  - reads retrieved text in-stream (this IS the copy mechanism; no pointer)   |
|  - final fast-weight states S' = compressed memory of chunks + prompt        |
|  - recurrent state can be FORKED: encode context once, branch per option     |
|  - generation mode: writes text, answer letters, PY()/SOLVE()/SEARCH() calls |
+-----------+-------------------------------------------+----------------------+
            | S' (per-head fast weights) + pooled h     ^ plan prefix: 64 soft tokens
            v                                           |   (LN + 640x640 adapter)
+------------------------------------------+            |
| LATENT REASONING CORE R  (ORRERY-L only) |            |
|  2 weight-tied core layers, iterated T   |            |
|  Z = P plan-chunks x 64 slots x 640      |------------+
|  step: slot-mix MLP -> linear read q.S'  |
|        (no softmax) -> SwiGLU, sigma/t   |<-----------+  -dE/dZ
|  System 1: denoise noise -> Z_hat, 1-4   |            |
|  System 2: Z <- denoise(Z) - eta dE/dZ   |            |
|            + annealed noise (Langevin)   |            |
+-------------------+----------------------+            |
                    v                                   |
+------------------------------------------+    +-------+--------------------+
| ENERGY HEAD E(S', Z) or E(S', h_text)    |--->| SEARCH CONTROLLER (0 par.) |
|  1 core layer + MLP 640->1024->1         |    | N candidates, beam, prune, |
|  = universal verifier / value / reward   |    | halting+tool head (42k)    |
+------------------------------------------+    +----------------------------+
+------------------------------------------+    +----------------------------+
| TOOLS: Python, SymPy, Z3, Lean, tests    |    | TEST-TIME MEMORY M (Titans)|
| result text appended to trunk stream     |    | MLP 640-1024-640, learned  |
+------------------------------------------+    | init counted, reset/episode|
+------------------------------------------+    +----------------------------+
| GRID REASONER (input-type routed, 4.0M)  |
| TRM-MLP-style recursive mixer on 30x30   |
| + test-time training on demo pairs       |
+------------------------------------------+
```

### 3.2 Inference dataflow

1. **Query pass.** The prompt goes through the embedding and trunk layers 1–4 only. These, plus the 640→512 projection, were frozen at the end of stage S1a (Section 6). Mean-pool, project, take the sign: a 512-bit query code.
2. **Retrieve.** Hybrid search over the human-authored datastore:
   - BM25 (0 params [VERIFIED bm25s: https://arxiv.org/abs/2407.03618]) on an on-disk compressed inverted index, memory-mapped. Korean uses a rule-based morphological or character-n-gram analyzer.
   - Dense: Hamming search over 512-bit binary codes with IVF, then rescoring of a top-1,000 shortlist with finer codes (PQ in RAM on Kaggle, int8 on the laptop SSD).
   - **Why the keys never go stale:** the datastore keys were produced by exactly these frozen tensors. Later stages fine-tune trunk layers 5–10 and everything else, but never layers 1–4, the embedding or the projection. The index is built once (after S1a) and stays valid.
   - Merge by reciprocal rank fusion → top-K chunks with provenance tags.
3. **Read (in-stream).** The trunk scans `[chunks][prompt]`. Retrieved evidence enters the fast-weight states through ordinary delta-rule writes. There is no read-fusion module and no pointer: copying a phrase from a chunk is just the trunk reproducing text it has recently read, which delta-rule models are specifically good at [VERIFIED: GDN beats Mamba2/DeltaNet on in-context retrieval, https://arxiv.org/abs/2412.06464v3]. The output is S' plus a pooled state h.
4. **Think, System 1 (ORRERY-L).** The core starts from N noise samples of Z and denoises them into candidate plans in 1–4 steps, reading S' through linear (non-softmax) reads q·S'. Easy prompts stop here if the energy is low and the halting head agrees.
5. **Think, System 2.** For uncertain prompts the controller mixes denoising updates with energy-gradient steps (Langevin-style, annealed noise), prunes by energy, branches survivors, and halts when energy gain per step flattens. Each energy-gradient step needs a backward pass through the energy head (~3x the cost of a forward step; counted in Section 8).
6. **Act.** If the plan contains a tool action, the core first predicts the latent of the tool result (JEPA-style), then the trunk writes the call, the tool runs, and the result text is appended to the stream. The prediction error ("surprise") updates the test-time memory M (Titans mechanism [VERIFIED: https://arxiv.org/abs/2501.00663], here on an attention-free core). Large surprise triggers re-planning. Skipping expensive calls when confident is a [HYPOTHESIS] and is off by default in the gated track.
7. **Verify.** Exact verifiers override the energy: self-written tests for code, SymPy for equations, Z3 for constraints. Where none exists (chat, commonsense) the energy decides.
8. **Render (F2 fix).** Z* enters the trunk as **64 soft prefix tokens per plan chunk** after a `<plan>` marker, through a LayerNorm + 640×640 adapter. The trunk scans them like any other tokens, so the plan is written into **every layer's fast-weight state with full 64×640 resolution**, and every subsequent output token can read it. v1's design squeezed the plan to ~512 numbers through one FiLM; v2 passes 40,960 numbers per plan chunk at the cost of 64 extra scan tokens.
   - **Plans grow with the problem (red-team M11).** A plan is a sequence of P plan-chunks (P = 1 to 8). The halting head decides P. The trunk renders chunk p's section after its prefix, then the core produces chunk p+1 conditioned on what was rendered. Answer complexity is no longer capped by a fixed 64-slot latent.
   - Multiple choice: the context state is computed once and **forked** per option (a recurrent model's state is O(1) to copy); the option with the lowest energy wins; the trunk writes its letter.

### 3.3 Why "world model"

The core is trained to predict the latent of what comes next or what is missing: the next paragraph, a masked span, a tool's output, a program's execution result, a solution given the problem. That makes it a world model of text and tools, and the energy is its compatibility score. This is the JEPA program, which already improves GSM8K and Spider when added to next-token loss [VERIFIED: https://arxiv.org/abs/2509.14252], combined with energy-based System-2 inference (Energy-Based Transformers report up to 35% higher scaling rate [secondary: https://liner.com/en/review/energybased-transformers-are-scalable-learners-and-thinkers; paper https://arxiv.org/abs/2507.02092]). Both are moved onto a non-Transformer backbone; whether that port works is a [HYPOTHESIS], tested by E1/E2 against ORRERY-0.

### 3.4 Grid reasoner (red-team M9 fix)

v1 asked a BPE text model with 64 slots to solve 900-cell grids. v2 adds a dedicated, attention-free recursive module in the TRM-MLP style: TRM-MLP (5M, no attention) reached 87.4% on Sudoku-Extreme and 29.6% on ARC-AGI-1 [VERIFIED: https://arxiv.org/html/2510.04871v1].

- Input: grid padded to 30×30 = 900 cells, 16 color/padding tokens, d_g = 384.
- Two weight-tied mixer layers: token-mixing linear 900→900 + channel SwiGLU (384→1024→384). Recursed over (x, y, z) with deep supervision, TRM-style.
- Test-time training: a few hundred gradient steps on each task's demonstration pairs (with augmentations), weights reset per task. CompressARC shows per-puzzle training alone reaches about 20% with 76K params in ~20 min on one RTX 4070 [VERIFIED: https://arcprize.org/blog/arc-prize-2025-results-analysis].
- Training data: re-ARC generators on the training split only [VERIFIED: https://arxiv.org/pdf/2404.07353], Reasoning Gym grid tasks [VERIFIED: https://arxiv.org/pdf/2505.24760], Sudoku and mazes.
- Honesty: TRM-MLP's 29.6 took ~4 H100 × ~3 days of ARC-only training [VERIFIED: https://arxiv.org/html/2510.04871v1], and HRM's claimed 40.3 dropped to 32 when verified [VERIFIED: https://arcprize.org/blog/hrm-analysis]. Our budget is a fraction of that. P(win) is revised to 0.5 (Section 5).

---

## 4. Exact parameter budget

Shared settings: d = 640, 10 heads × 64, SwiGLU hidden 1728.

- **One GDN layer** = q,k,v,o (4d² = 1,638,400) + output gate (d² = 409,600) + α/β gates (2·d·10 = 12,800) + short convs (kernel 4 on 3d channels = 7,680) + SwiGLU (3·640·1728 = 3,317,760) + 2 norms (1,280) = **5,387,520**.
- **One core-type layer** = q-read, out, gate (3d² = 1,228,800) + slot-mix MLP 64→256→64 (33,088) + SwiGLU (3,317,760) + 3 norms (1,920) = **4,581,568**.
- **Grid mixer layer** = token-mix 900×900+900 (810,900) + SwiGLU 3·384·1024 (1,179,648) + 2 norms (768) = 1,991,316.

| # | Module | Composition | Trainable params |
|---|---|---|---|
| 1 | Token embedding (tied to output; **frozen after S1a**) | 32,768 × 640 | 20,971,520 |
| 2 | Trunk (layers 1–4 **frozen after S1a**, serve as retriever) | 10 GDN layers | 53,875,200 |
| 3 | Retrieval projection (frozen after S1a) | 640 × 512 | 327,680 |
| 4 | Latent reasoning core R (weight-tied, iterated) | 2 core layers (9,163,136) + noise-level MLP 640×640+640 (410,240) + slot init 64×640 (40,960) + step embedding 32×640 (20,480) | 9,634,816 |
| 5 | Energy head E | 1 core layer (4,581,568) + MLP 640→1024→1 (657,409) | 5,238,977 |
| 6 | Plan adapter (soft-prefix) | LayerNorm (1,280) + 640×640+640 (410,240) | 411,520 |
| 7 | Test-time memory M (learned initial weights) | MLP 640→1024→640 | 1,312,384 |
| 8 | Halting / tool-routing / plan-length head | 640→64→8 | 41,544 |
| 9 | Grid reasoner | 2 mixer layers (3,982,632) + color embed 16×384 + out head 384×16+16 + y/z init 2×384 + halt 385 + demo-flag 2×384 | 3,996,857 |
| 10 | Final norms | 4 × 640 | 2,560 |
| | **Total trainable at inference** | | **95,813,058** |
| | Headroom under 100M | | 4,186,942 |
| — | EMA target encoder (training only) | copy of rows 1+2 | 74,846,720, disclosed, not counted |
| — | BM25 index, IVF centroids, PQ codebooks, binary codes, tokenizer | data | disclosed, 0 |

ORRERY-0 uses rows 1, 2, 3, 5, 8, 9, 10 = **84,454,338** and may spend the difference (11.36M) on 2 extra trunk layers (+10.78M) **for the equal-parameter comparison** [ESTIMATE: this is the fair control; an ORRERY-0 that is smaller than ORRERY-L would flatter the latent arm].

Arithmetic checked by script on 2026-10-19 (sum = 95,813,058).

**Where v1's parameters went.**
- Read-fusion (6.98M) deleted: retrieval is read in-stream by the trunk.
- Separate decoder (20.93M) merged into the trunk. Encoder + decoder (16 GDN layers at d=512, 55.8M) became one 10-layer trunk at d=640 (53.9M). The trunk is wider where fluency is decided and is used for every role, as the red team proposed.
- FiLM (0.53M) → soft-prefix adapter (0.41M).
- Copy pointer (0.26M) deleted.
- Grid reasoner (+4.0M) added.

**Known trade-offs [ESTIMATE].**
- Freezing the tied embedding after S1a limits later adaptation of the output layer. E0 measures the cost at 20M (frozen vs unfrozen embedding after 8B-equivalent tokens). If the loss gap exceeds 2%, the fallback is to **rebuild the index after each later stage instead of freezing**; at the 10B-token headline tier that costs ~1–3 TPU-hours per rebuild (Section 7), which is affordable. This partly answers the red team's cost argument: the 3-week rebuild figure assumed 150B tokens on T4s; at 10B tokens on the TPU it is hours.
- 10 trunk layers is shallower than v1's 16. Width 640 vs 512 compensates partly. MobileLLM found deep-and-thin better at 125M [VERIFIED: https://arxiv.org/html/2402.14905]; E0 tests 10×640 vs 14×544 at equal params.

---

## 5. Beating the scoreboard, row by row (honest odds + campaign forecast)

Targets are the highest reported GPT-3.5 numbers from the baseline research. "P(win R1)" is my probability that the round-1 system (week ~16, 95.8M) beats target + CI under the gated protocol of Section 2 (no MC train splits, human-only datastore). All probabilities and estimates are [ESTIMATE].

### 5.1 Round-1 odds

| Row | Target | v2 R1 estimate | P(win R1) | Change vs v1 and reason |
|---|---|---|---|---|
| NQ-open (EM) | 38.6 | 40–48 | 0.55 | Unchanged. FiD-base got 48.2 with 440M total [VERIFIED: https://arxiv.org/abs/2007.01282]. Human-only store costs nothing here (answers live in Wikipedia). KILT train split allowed. |
| TriviaQA | 85.8 | 62–75 | 0.08 | GPT-3.5 closed-book is 82.9 EM 0-shot [VERIFIED: https://arxiv.org/abs/2407.02485]. Red. |
| MMLU | 71.4 | 42–55 | 0.04 | Lower: no MMLU aux-train, and no LLM-written explanation store. Atlas-770M with retrieval 56.3 [VERIFIED: https://arxiv.org/abs/2208.03299]; retrieval gives only +3–4 on MMLU [VERIFIED: https://arxiv.org/html/2407.12854]. Red. |
| KMMLU | 42.47 | 34–45 | 0.25 | Lower: Korean exam sites and exam-format documents are now blocked (M7). |
| HAE-RAE (mean of 5) | 49.45 | 44–58 | 0.35 | Dictionary/nomenclature lookups from ko-Wikipedia; history via retrieval. |
| KoBEST | 79.3 (avg) | 62–76 | 0.15 | Lower: no KoBEST train split in gated model. |
| GSM8K | 78.1 | 55–72 | 0.20 | Program-of-thought + execution + energy reranking; GSM8K train allowed; no GPT-3.5 data (TinyGSM excluded). |
| MATH | 34.1 | 20–35 | 0.25 | Offload to SymPy. Qwen2.5-0.5B-Instruct: 34.4 at 5x our size [VERIFIED: https://qwenlm.github.io/blog/qwen2.5-llm/]. |
| HumanEval / + | 76.8 / 70.7 | 30–50 | 0.03 | phi-1-small (350M): 45 [VERIFIED: https://arxiv.org/abs/2306.11644]. Red. |
| MBPP / + | 82.5 / 69.7 | 40–58 | 0.03 | Red. |
| BBH | 68.3 | 50–70 | 0.30 | Many tasks exactly executable. |
| DROP (F1) | 64.1 | 55–70 | 0.35 | Slightly lower: the copy pointer is gone; in-stream copy is untested at this scale. |
| HellaSwag | 85.5 | 45–62 | 0.02 | Lower: no train split, and WikiHow/ActivityNet documents removed from store. v1 cited "discriminative encoders ~300M in the 80s" from memory, unverified, and those were fine-tuned on the train split anyway; **that claim is withdrawn.** SmolLM2-135M: 42.1 [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M]. Red. |
| WinoGrande | 81.6 | 52–65 | 0.02 | v1's unverified ~79 recollection withdrawn for the same reason. Every generative model ≤270M sits at 51–53 [VERIFIED: tiny_sota digest]. Red. |
| ARC-Challenge | 87.4 | 50–65 | 0.03 | No ARC train split. Red. |
| ARC-AGI-1 (pass@2) | 5.0 | 10–30 | 0.50 | Revised from 0.85 per M9. The bar is low (GPT-4o 5% [VERIFIED: https://arcprize.org/blog/oai-o3-pub-breakthrough]), but our training compute is a fraction of TRM's. |
| Chat pooled (MT-Bench + AlpacaEval + Arena-Hard, pairwise vs stored GPT-3.5) | > 50% + CI (≈ 52.6%) | 8–25% wins | 0.02 | Red. See 5.3. |

**Expected R1 outcome:** 2–5 green rows out of ~17. P(all gated rows at R1) < 0.5%. This is lower than v1 because v2 closed the loopholes (train splits, LLM-written store, softmax pointer) that would have made some v1 wins fake.

### 5.2 Campaign table: every red row gets a knob, a slope and a forecast

Rounds: **R1** = week 16 (first full eval). **R2** = months 5–9, still <100M: datastore 10B→25B tokens, self-play data 10%→50%, test-time compute up to the "hard" mode, more RL. **R3** = months 9–15, last sub-100M attempt: everything that worked, retrained once from scratch with the best mix. **1B tier** (needs TPU Research Cloud) is a north-star step and does **not** count for the mid-boss.

| Red row | Knob (no new params) | Slope we must measure (in E3 and between rounds) | Forecast green round [HYPOTHESIS] | Falsified if |
|---|---|---|---|---|
| TriviaQA | datastore coverage + reader RL on KILT | EM per doubling of store and per K | R2 | < +3 EM per store doubling after R1 |
| MMLU | curated human textbook/StackExchange store, option-elimination search under energy | accuracy vs test-time T and vs store composition | R3 (low confidence, P ≈ 0.15) | slope vs T flat by T=4 |
| KoBEST, KMMLU | ko store share, cross-lingual retrieval curriculum, Korean self-play | per-subset accuracy vs ko token share | R2 | no gain from doubling ko data |
| GSM8K | self-play verified traces, N candidates × executor | pass@1 vs N and vs self-play rounds | R2 | < +2 per self-play round after round 2 |
| HumanEval / MBPP | generate–self-test–repair loop, permissive human code store | pass@1 vs repair rounds and vs N | R3 (P ≈ 0.2) | flat after 4 repair rounds |
| HellaSwag | energy scoring with contrastive negatives mined from the model's own continuations (no benchmark data), retrieval of similar narratives (WikiHow removed) | accuracy vs energy-head training tokens and vs params (5-point curve) | **No sub-100M forecast.** Honest: extrapolation needed | 5-point param curve extrapolates below 85.5 at 100M |
| WinoGrande | same + multi-hypothesis verification (substitute each option, score the whole sentence by energy) | same | **No sub-100M forecast** | same |
| ARC-Challenge | science textbook store + option energy | accuracy vs T and store | R3 (P ≈ 0.15) | flat in T |
| Chat | plan search under preference energy; human-answer store read in-stream; chunked plans | pooled win rate vs params (5 points) and vs N plans | **No sub-100M forecast** | win rate < +3 points per 2.5x params |

The three "no forecast" rows are the honest center of this proposal. The smallest model on record that matches GPT-3.5 in Arena is ~2.6B [VERIFIED: baseline digest], and nothing ≤270M rises above chance on WinoGrande. If ORRERY's slopes on those rows are not dramatically better than the LM baseline's, the mid-boss is not beatable under 100M with this design, and we will say so with the curves attached.

### 5.3 Open-ended chat, creative writing and Korean fluency

- **Where we start.** Tiny models: MT-Bench 1.98–2.52 [VERIFIED: https://arxiv.org/html/2402.14905, SmolLM2 card]; GPT-3.5: 7.94–8.39.
- **Statistical power (M8, accepted).** MT-Bench alone has ~80 questions × 2 turns ≈ 160 items: the 95% CI is about ±7.7 points, so a win needs ≥58% [ESTIMATE]. v2 **pools** MT-Bench (stored `gpt-3.5-turbo.jsonl` answers [VERIFIED exists: https://huggingface.co/spaces/lmsys/mt-bench]), AlpacaEval 2.0 (805 prompts, stored 0613 outputs [VERIFIED exists: https://github.com/tatsu-lab/alpaca_eval]) and Arena-Hard (500 prompts; whether GPT-3.5 answers ship is [ESTIMATE]): N ≈ 1,465, CI ≈ ±2.6 points, gate = pooled pairwise win rate ≥ 52.6%, length-controlled, both answer orders. Per-benchmark numbers are reported too.
- **Judge hygiene.** Training-time preference judge and eval judge from disjoint families (Rule 12). The author hand-labels ~100 pooled pairs blind; if agreement with the eval judge is < 70% [ESTIMATE threshold], the judge result is reported but marked unreliable.
- **ORRERY's attack [HYPOTHESIS].** (1) Chunked latent plans decide content and order under energy search. (2) The trunk renders with the plan in its state at full resolution (F2 fix). (3) Fluent phrasing is borrowed by reading **human-written** answers retrieved into the stream (StackExchange-style, Wikipedia), never LLM-written ones. (4) The energy head, trained on preference pairs, makes best-of-N over plans a built-in reward model.
- **Creative writing** is the worst case: retrieval helps little and there is no verifier. Expected clear loss.
- **Korean.** 21M of embeddings plus the shared trunk serve en, ko and code. 35.3% of Korean HF datasets have unknown licenses [VERIFIED: https://arxiv.org/pdf/2507.04329]. GPT-3.5's Korean Arena score (1049.6) is below its overall score, but still out of reach at R1.

---

## 6. Training recipe

"Target tokens" are tokens the trunk or predictor is supervised on. FLOPs per target token are computed in Section 7.

| Stage | Purpose | Data | Objectives | Target tokens |
|---|---|---|---|---|
| **S0** | Harness, hygiene, datastore (week 1, *before any training*) | Headline store **T1 = 10B human-authored tokens**: English Wikipedia (~4B tokens [VERIFIED: https://arxiv.org/html/2112.04426]), ko-Wikipedia, FineWeb-Edu subset, FineWeb-2 ko, StackExchange-style Q&A, arXiv, permissive human code (Python-Edu subset of Stack v2 [VERIFIED: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus]). T2 = 25B (laptop only, R2) | (a) 13-gram + MinHash + embedding-cosine > 0.9 scrub of **store and training data** against every eval set and mirrors; (b) **URL-level removal** of WikiHow and ActivityNet source documents (HellaSwag contexts come from them), benchmark-hosting sites, GitHub copies of HumanEval/MBPP, and **Korean exam sites**; (c) **exam-format filter**: drop documents with ≥3 numbered-option blocks (①②③④ / (A)–(D) patterns) followed by an answer key [ESTIMATE heuristic]; (d) published overlap report; (e) retrieval-audit logger: per eval item, store top-k chunk IDs and whether the gold answer string appears in them, reported per row | — |
| **S1a** | Trunk pretraining, BM25-only retrieval (stated honestly: no dense retrieval exists yet) | Mix: 15% fluency (curated edu en+ko); 20% retrieval-reading (KILT with provenance [VERIFIED: https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/] + synthetic QA over the store); 35% procedural (Reasoning Gym [VERIFIED: https://arxiv.org/pdf/2505.24760], own SymPy/graph/execution-trace generators); 20% open-teacher distilled; 10% human code | next-token loss (with BM25-retrieved context on 40% of samples); **contrastive retrieval loss** on the layer-4 pooled projection (query vs KILT source passage, in-batch + BM25 hard negatives) so the soon-to-be-frozen keys are good | 8B |
| **Freeze + index** | Build the dense index once | T1 store | freeze embedding, layers 1–4, projection; embed 78M chunks (10B / 128) → 512-bit codes + PQ | ~1–3 TPU-h (Section 7) |
| **S1b** | Trunk pretraining with hybrid retrieval | same mix | next-token (trainable layers 5–10); **JEPA** on the EMA target (masked spans, next paragraph, tool outputs); VICReg variance/covariance on latents | 17B |
| *Branch point* | ORRERY-0 vs ORRERY-L | | | |
| **S2-0** (ORRERY-0) | Text-candidate energy reranker | problem–answer pairs (non-MC), the model's own sampled answers, verifier labels | InfoNCE: true answer vs own samples + corruptions; regression onto execution outcomes | 10B (equal FLOPs to S2-L) |
| **S2-L** (ORRERY-L) | Latent denoising and energy | same | (a) latent diffusion: Z* = EMA-encoder(answer) pooled into 64-slot plan chunks; R denoises σ-noised Z* given S'. (b) energy InfoNCE over true vs hard negatives (own samples, corruptions, **adversarial latents mined from descent trajectories**) + score matching so −dE/dZ aligns with the denoiser. (c) trunk reconstructs text from the soft-prefix plan; **cycle consistency**: re-encode rendered text, match Z* | 10B |
| **S3** | Tool use and instruction format | OpenMathInstruct-2 (CC-BY-4.0 [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2]) rewritten into PY/SOLVE calls, kept only if execution reproduces the answer; SmolTalk (Apache-2.0); Tulu 3 minus NC subsets [VERIFIED: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/main/README.md]; Magpie-style data regenerated from Apache-licensed teachers. **These are training data only; none enter the datastore.** | SFT on calls and on using results; JEPA on tool-result latents | 3B |
| **S4** | RL with verifiable rewards (GRPO), **co-located on the TPU** | Reasoning Gym, re-ARC (train split), math with checkable answers, code with self-generated + reference tests (training problems only) | GRPO on the full loop; rollouts and updates in the same JAX program on the TPU, with the sandbox on the TPU host CPU, so the policy is never more than one update stale (M5). Energy head regressed onto outcomes. Proposer head writes problems at a 50–70% solve rate | 3–5B rollout tokens |
| **S5** | Preference energy for chat | open-license preference pairs + pairs labeled by a local judge of family A (training only, disclosed) | energy as reward model on plans; best-of-N distilled into System 1 | 0.5B |
| **S6** | Self-play flywheel (R2 onward) | model + tools generate problems and solutions; only verifier-accepted traces kept | STaR / expert iteration [VERIFIED: https://arxiv.org/html/2203.14465] | 10% → 50% of mix |

Grid reasoner: trained separately (S-G) on re-ARC, Reasoning Gym grids, Sudoku, mazes; ~1 TPU-week across R1 [ESTIMATE].

Data rules: no GPT-3.5/4 data anywhere; every teacher named with size and license; NC data in a separate manifest; Wikipedia (CC BY-SA) in the datastore, kept out of weight-training text where avoidable [VERIFIED license: https://download.wikipedia.com/legal.html].

---

## 7. Compute plan (recomputed, red-team M1 accepted)

**Envelope.** TPU v5e chip 197 TFLOPS bf16 [VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e]; Kaggle v5e-8 replacing v3-8 [VERIFIED: https://www.kaggle.com/discussions/product-announcements/607202], ~20 h/week (third-party), 9 h sessions. T4 65 TFLOPS fp16, no bf16 [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf], ~30 GPU-h/week (user-reported). MFU for tiny models on Kaggle TPUs was measured at 16–27%, and 13% at 8-way data parallelism [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html]. **v2 plans at 15–20% MFU.**

**FLOPs per training sample** [ESTIMATE]. Sample = 512 target tokens; with retrieval, K=8 chunks × 128 = 1,024 context tokens; 64 plan-prefix tokens. Parameter × token products:

| Term | Retrieval sample | No-retrieval sample |
|---|---|---|
| Trunk (53.9M non-embedding) on 1,536+64 / 512+64 tokens | 86.2 G | 31.0 G |
| Tied output projection, 21.0M × 512 targets (v1 forgot this) | 10.7 G | 10.7 G |
| Core, T=8 unrolled × 64 slots, 9.63M | 4.9 G | 4.9 G |
| Energy head, 5.24M × 64 slots × 4 candidates | 1.3 G | 1.3 G |
| Hard-negative encoding, ~256 tokens × 53.9M (v1 forgot this) | 13.8 G | 13.8 G |
| Subtotal × 6 (fwd + bwd) | 702 GFLOP | 370 GFLOP |
| EMA target forward, 2 × 53.9M × 512 (v1 forgot this) | +55 GFLOP | +55 GFLOP |
| **Per sample / per target token** | **757 GFLOP / 1.48 GFLOP** | **425 GFLOP / 0.83 GFLOP** |

With 40% retrieval samples: **≈ 1.09 GFLOP per target token**, 1.2x v1's 0.89 and about 2x a plain 96M LM (6N ≈ 0.57). Self-sampled candidates in S2 add serial decoding wall-clock that is not in this table; S2 budgets 25% extra wall-clock for it [ESTIMATE]. After S1a, frozen layers 1–4 skip weight-gradient computation, saving ~10% [ESTIMATE]; not counted, it is margin.

**Throughput** [ESTIMATE].
- TPU v5e-8 at 15–20% MFU = 236–315 TFLOPS → 217k–289k target tokens/s → × 72,000 s = 15.6–20.8B/week; minus ~10% for JIT compilation and session restarts = **≈ 14–19B target tokens/week**.
- T4x2 at 20% MFU = 26 TFLOPS → 24k tokens/s × 108,000 s ≈ **2.6B/week**.
- **Day-1 rule:** measure tokens/s for the real model on both platforms; every schedule row below is re-planned from the measurement.

**Delta-rule kernel (M2, accepted).** `associative_scan` over dense d_h × d_h transition matrices would cost O(d_h³) per combine, ~30x the sequential cost [ESTIMATE]. v2 uses the **chunkwise WY/UT formulation**: within each 64-token chunk, the delta-rule updates are expressed as dense matmuls; across chunks, `lax.scan` carries the state. This is the form the GDN line of work uses [ESTIMATE: check against https://github.com/NVlabs/GatedDeltaNet]. Chunk-internal overhead ≈ 5% of parameter matmul FLOPs [ESTIMATE], within the MFU margin. Pure JAX first; Pallas only if profiling demands it.

**Storage (M3, M4 accepted)** [ESTIMATE unless marked].
- T1 store: 10B tokens ≈ 40 GB raw, ≈ 10–14 GB zstd. Fits HF 100 GB private [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits] and Kaggle dataset quota.
- Chunks: 10B / 128 ≈ 78M. Dense: 512-bit binary codes = 64 B → **5.0 GB RAM**; IVF over it; rescoring with 64-B PQ codes (another 5.0 GB RAM) on a top-1,000 shortlist. ~10 GB total, inside Kaggle's ~30 GB RAM.
- BM25: on-disk compressed inverted index (Lucene/Tantivy-style), memory-mapped; compressed postings ≈ 20–30% of raw → ~8–12 GB on disk, not in RAM.
- T2 store (25B tokens, R2): laptop only, int8 512-d rescoring vectors (≈ 100 GB) on SSD. **Precondition: check the laptop's free SSD space before planning T2.** If it lacks ~200 GB free, T2 is PQ-only (≈ 25 GB) and the 300B tier from v1 is cancelled.
- **Recall check before trusting dense retrieval:** on a 1M-chunk sample, measure recall@20 of 512-bit binary + PQ rescoring vs exact fp32 search on KILT queries. Go if the gap is ≤ 5 points; otherwise move to 1024-bit codes (10 GB RAM). The 92.5%/96.45% figures in the research were measured on 1024-d mxbai vectors [VERIFIED: https://huggingface.co/blog/embedding-quantization] and are **not assumed** to transfer.

**Index build cost.** Embedding + 4 trunk layers ≈ 21.6M params × 2 = 43 MFLOP/token. 10B tokens → 4.3e17 FLOP ≈ 0.5 h of TPU at 236 TFLOPS; with I/O and session overhead ~1–3 TPU-h. 25B → ~3–6 h. Cheap enough that "rebuild after each stage" is a real fallback if freezing the embedding hurts (Section 4).

**Schedule** (16 weeks to R1; TPU first; every row re-planned after day-1 measurements).

| Weeks | TPU v5e-8 | T4x2 | RTX 3060 6 GB |
|---|---|---|---|
| 1 | Device/MFU check (`jax.devices()`), JAX GDN kernel smoke test | **S0: eval harness, decontamination, retrieval-audit logger**; BM25 build over T1 | Kernel dev in bf16, unit tests |
| 2–4 | **E0** (ORRERY-0 at 20M, frozen-vs-unfrozen embedding, 10×640 vs 14×544), **E1/E2** at 20–50M | Teacher data from Apache-licensed open models (~0.1–0.6B tokens/week [ESTIMATE: compute digest]); verifier filtering | Grid reasoner prototype |
| 5 | S1a, 8B (~0.5 week) → freeze → dense index (1–3 h) → recall check | Teacher data; KILT recall eval | — |
| 6–7 | S1b, 17B (~1 week) | Teacher data; decontam of teacher data | — |
| 8 | S2-0 and S2-L, 10B each (~1.2 weeks with sampling overhead) | Eval of S1 checkpoint | — |
| 9 | S3, 3B; S-G grid training | Local judge setup (family B), stored-answer alignment | — |
| 10–12 | S4 GRPO, co-located on TPU | Eval of ORRERY-0 vs ORRERY-L at equal FLOPs | — |
| 13 | S5, 0.5B | Preference labels from judge family A | — |
| 14–15 | **E3 scaling grid** (Section 10) | Full eval suite on pinned modes | Offline demo |
| 16 | **R1 eval, 3 seeds on sampled rows** | Overflow evals | — |

**Totals.** S1–S5 ≈ 8 + 17 + 20 + 3 + 0.5 ≈ 48.5B target tokens + 3–5B rollouts + ~18.5B for E3 ≈ 7e19 FLOPs ≈ 3.5% of the ~2e21 FLOPs/year envelope [ESTIMATE: compute digest]. Pretraining FLOPs are still not the bottleneck; wall-clock of serial RL, eval, and verified-data generation are.

**If the TPU is unavailable** (reported as sometimes oversubscribed [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab, third-party]): T4 alone gives ~2.6B/week. S1 is cut to 12B total (still ~6x Chinchilla-optimal for 100M), S2 to 4B each arm, E3 to 4 sizes; R1 moves to ~week 30.

**Engineering details.**
- **One JAX codebase** for training, sampling and RL. The PyTorch port is dropped for R1 (red-team simplification). Whether JAX's CUDA build supports sm_75 is [ESTIMATE]: verify in week 1; if not, T4s run only BM25, data generation (vLLM) and CPU-side work, and evaluation moves to the TPU.
- Triton is avoided entirely: Triton 3.3+ requires compute capability 8.0+ [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md].
- **T4/fp16 numerics** (if JAX-on-T4 works): fp16 matmuls, fp32 recurrent state and scan accumulators, fp32 energy-descent loop, dynamic loss scaling, decay gates in log space and clamped.
- **TPU static shapes.** Fixed K buckets {0, 4, 8, 16}, fixed 64 slots per plan chunk, plan-chunk counts bucketed {1, 2, 4, 8}, unroll T bucketed {1, 4, 8, 16}. Compiled programs cached to `/kaggle/working` per bucket so the 9-h session cap is not eaten by recompilation (red-team minor 4). Backprop through the last 2 refinement steps with TRM-style deep supervision on every step [VERIFIED: https://arxiv.org/html/2510.04871v1].
- **TPU via CLI pitfall:** `kaggle kernels push` can silently give a non-TPU image [VERIFIED: https://github.com/Kaggle/kaggle-cli/issues/1197]; every run asserts the device in its first cell.
- Checkpoint every 45 min to `/kaggle/working`, push to the HF Hub.

---

## 8. Inference cost and latency per query

FLOPs ≈ 2 × parameter × token products; energy-gradient steps cost ~3x a forward step (forward + backward w.r.t. Z). Assumptions: 512-token prompt, K = 16 chunks × 128, 300 output tokens. All [ESTIMATE].

| Mode | Candidates × steps | Breakdown | FLOPs/query | T4 latency (≈10 TFLOPS effective) |
|---|---|---|---|---|
| Fast (System 1) | 4 × 2 | trunk 2,560 tok ≈ 0.28 TF; render 300 tok (trunk + output proj) ≈ 0.045 TF; core 0.01 TF; energy fwd 0.003 TF | ≈ 0.34 TF | 0.3–0.8 s |
| Standard | 16 × 16 | + core 16·16·64 × 2·9.63M ≈ 0.32 TF; energy descent 16·16·64 × 5.24M × 6 ≈ 0.52 TF; 2 renders 0.09 TF | ≈ 1.25 TF | 1.5–4 s |
| Hard / verifiable | 64 × 32, + 8 tool rounds | + core ≈ 2.5 TF; energy descent ≈ 4.1 TF; tool re-reads 8 × 256 tok ≈ 0.22 TF; 8 renders 0.36 TF | ≈ 7.5 TF | 15–90 s; code-repair loops up to minutes |
| ORRERY-0 (equal-FLOP control) | text candidates sized to match each mode | sampling + energy over decoded text | matched | similar |

**Eval-suite cost (red-team minor 2, accepted).** Each row is **pinned to one mode** before seeing test results: MC rows → Standard; generative math/code/BBH → Hard; QA/DROP → Standard; chat → Standard with N=16 plans. Roughly 8e4 Standard items × 1.25 TF + 1.5e4 Hard items × 7.5 TF ≈ 2.1e17 FLOPs per seed. Evaluation runs on the TPU (batched across items, since serial recurrence latency hides behind batching), ≈ 3–6 TPU-hours per seed including overhead. Three seeds only on rows that sample; deterministic rows run once.

Comparison: a dense GPT-3.5-class model of unconfirmed size (20B–175B) would cost about 30–280 TFLOP for the same query, so hard mode remains 4–40x cheaper in FLOPs but much slower in wall-clock on a T4 because refinement is serial. Retrieval (BM25 mmap + Hamming search on CPU) adds ~50–300 ms [ESTIMATE]. Every reported score states its mode.

---

## 9. Scaling story

Six axes; only the first needs retraining. Each axis gets a **separately fitted curve** in E3 (Section 10), not a story.

| Axis | 0.1B milestone | 1B | 10B | Evidence |
|---|---|---|---|---|
| Trunk/core/energy params | 95.8M | trunk 750M, core+energy 200M, grid 50M | ×10, + product-key memory layers (memory params scale to 128B [VERIFIED: https://arxiv.org/abs/2412.09764]) | RWKV-7 improves monotonically 0.19B→2.9B [VERIFIED: https://arxiv.org/abs/2503.14456] |
| Datastore tokens (human-authored) | 10B (T1) → 25B (T2) | 0.2–1T | multi-T + domain stores (PubMed, patents, code) | RETRO gains don't diminish 150M→7B, ~10x params [VERIFIED: https://arxiv.org/html/2112.04426]; MassiveDS no saturation to 1.4T [VERIFIED: https://arxiv.org/html/2407.12854] |
| Test-time refinement (N, T, plan chunks P) | up to 64 × 32 × 8 | same knob, higher base pass@k | same | compute-optimal TTS beats 14x larger models when base success is non-trivial [VERIFIED: https://arxiv.org/abs/2408.03314] |
| Energy-head calibration signals | executions, preferences | + formal proofs, simulators | + lab-validated assays (human-gated) | TinyGSM verifier effect [VERIFIED: https://arxiv.org/html/2312.09241] |
| Tool registry | Python, SymPy, Z3, Lean | + geometry engine, theorem libraries | + docking, MD, retrosynthesis (biosafety-gated) | AlphaGeometry 25/30 with a symbolic engine [VERIFIED: https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/] |
| Self-generated verified data | 10% → 50% | dominant | dominant | R1-Zero AIME 15.6 → 77.9 from RL [VERIFIED: https://arxiv.org/html/2501.12948] |

**Why it should scale better than an LLM [HYPOTHESIS], and how we would know it doesn't.** More parameters buy a sharper energy landscape and a better denoiser prior, both of which raise the base success rate that search multiplies. The datastore and tools grow knowledge and computation with disk and CPU. The energy is domain-agnostic: a new verifier plugs in as a new calibration signal. **The kill condition is concrete:** if ORRERY-L's fitted slope (score vs log total FLOPs, training + inference) is not better than ORRERY-0's on the same 5-point grid, the latent world model does not scale better, and the 1B tier is built as ORRERY-0.

**What the 1B step changes.** Freezing the embedding and lower trunk stays the retrieval strategy (keys stable across stages). The grid reasoner becomes a general structured-input reasoner (grids, graphs, molecules as graphs, under biosafety rules).

**The compute wall** [ESTIMATE: compute digest]. 1B at Chinchilla scale ≈ 1.2e20 FLOPs ≈ 3 TPU-weeks: feasible on the free tier. 10B ≈ 1.2e22 ≈ 7 years of free quota: needs TPU Research Cloud (free TPU quota; small VM/GCS costs; open publication required [VERIFIED: https://sites.research.google/trc/faq/]). **Apply in week 1.**

**Toward the north star.** Science is a verifier-rich environment: simulators, proofs, assays. Discovery is search through hypothesis space under a scoring function, which is ORRERY's native loop: plan experiments in latent space, predict outcomes (world model), calibrate the energy on cheap simulators, propose only the highest-value expensive actions to humans.

**Biosafety is a hard constraint, not a tuning knob.**
- The datastore and tool registry exclude pathogen-enhancement and toxin-synthesis content.
- The controller refuses hazardous chemistry and biology requests; refusal is an exact rule, not an energy trade-off.
- Any output a wet lab could act on requires review by a qualified human expert.
- Biology models, datasets and tools go through dual-use review before release.
- These rules are never traded against capability, in any round.

---

## 10. First experiments (go/kill), now against ORRERY-0

**E0: "Is the baseline real and are the plumbing choices sound?"** (weeks 2–3, 20M, TPU + 3060)
- ORRERY-0 at 20M on the S1 mix (scaled down), with BM25 then frozen dense retrieval.
- Measure: tokens/s and MFU on TPU (and T4 if JAX supports it); frozen-vs-unfrozen embedding loss gap; 10×640 vs 14×544 shape (scaled); KILT recall@20 for binary+PQ vs exact.
- **Go** if MFU ≥ 12% on TPU and recall gap ≤ 5 points. **Else** re-plan compute (Section 7) and move to 1024-bit codes.

**E1: "Does latent refinement beat sampling + verification at equal FLOPs?"** (weeks 3–4, 20M)
- Arms: (a) ORRERY-0 single pass; (b) ORRERY-0 best-of-N with exact verifier/energy-over-text; (c) ORRERY-L System 1; (d) ORRERY-L refinement T = 4, 16, 32 with N = 16. Arms (b) and (d) are matched in inference FLOPs.
- Tasks: 6 Reasoning Gym families + arithmetic word problems, held-out harder difficulty levels.
- **Plan-to-text fidelity probes (F2):** (i) cycle consistency: cosine between Z* and the EMA re-encoding of the rendered text, reported per task; (ii) **plan-swap test**: render prompt A with prompt B's plan and measure whether the answer follows the plan (≥ 80% for go [ESTIMATE threshold]). A renderer that ignores the plan fails here.
- **Go** if (d) beats (b) by ≥ 5 points at equal FLOPs in ≥ 4 of 7 families, accuracy rises ≥ 10 points from T=1 to T=16, energy AUROC vs execution ≥ 0.80, and the plan-swap test passes.
- **Kill** latent descent if (d) ≤ (b) + 1 point or AUROC < 0.65. ORRERY-0 becomes the system; the energy head survives only as a text reranker.

**E2: "Does energy scoring beat likelihood on commonsense without train splits, and does JEPA help?"** (weeks 3–5, 50M, TPU)
- Arms at equal compute: next-token only; + JEPA; + JEPA + energy training. **No benchmark train splits** (Rule 7); negatives mined from the model's own continuations of human text.
- Eval: HellaSwag, WinoGrande, KoBEST-COPA (generative protocol), NQ with retrieval, 1k-item MMLU dev slice.
- **Go** if energy scoring beats likelihood scoring by ≥ 5 on HellaSwag and ≥ 3 on WinoGrande, and JEPA adds ≥ 1.5 on average.
- **Kill** the discriminative-efficiency bet if energy is within 1 point of likelihood; drop JEPA if it does not help.

**E3: "Is it scalable, and does it scale better than the baseline?"** (weeks 14–15)
- **5 sizes** (5, 10, 20, 50, 100M) at a fixed 100 tokens per parameter (0.5 + 1 + 2 + 5 + 10 = 18.5B target tokens) for **both** ORRERY-0 and ORRERY-L (ORRERY-0 grid at reduced eval set if compute is short).
- **Separate curves:** score vs params (fixed T, store); vs test-time T and N (fixed params); vs datastore size 1B / 3B / 10B tokens (fixed params).
- Chat probe on the pooled set (N ≈ 1,465), both answer orders, disjoint-family judge.
- **Go** if: ORRERY-L's slope vs log total FLOPs beats ORRERY-0's; the test-time axis does not saturate by T=16; the retrieval-on/off gap does not shrink with size; pooled chat win rate rises ≥ 3 points per 2.5x params (detectable at ±2.6 CI across the 5-point fit).
- **Kill / pivot** if: the test-time axis saturates by T=4 (not a test-time-scaling architecture); ORRERY-L's slope ≤ ORRERY-0's (build the 1B tier as ORRERY-0); chat flat from 5M to 100M (chat needs a different renderer strategy, only possible at 1B, and the chat row is declared unwinnable under 100M with this design).

---

## 11. Top risks

1. **Losing on chat, HellaSwag, WinoGrande.** Near-certain at R1; no sub-100M forecast exists. Mitigation: chunked plans, human-answer store, preference energy, measured slopes, public honest reporting. This is also the main reason to reach the 1B tier quickly (TRC).
2. **Latent arm loses to ORRERY-0.** Now an explicit outcome, not a hidden failure. Mitigation: E1 at week 4 decides cheaply; the project survives either way.
3. **Energy hacking.** Descent finds low-energy wrong latents. Mitigations: annealed Langevin steps, denoiser-prior regularization, exact-verifier override, adversarial negatives mined from descent trajectories (S2-L, S4).
4. **Latent collapse / drift** in JEPA. EMA targets, VICReg terms, latent-rank check every 1k steps.
5. **Plan-to-text gap.** Now probed directly (cycle consistency, plan-swap test in E1). Soft-prefix conditioning gives the trunk full-resolution access to the plan.
6. **Frozen-embedding cost.** Measured in E0; fallback is per-stage index rebuild (~1–3 TPU-h at T1).
7. **Retrieval quality at 512 bits.** Measured on KILT before use; 1024-bit fallback.
8. **Recurrent recall limits.** A fixed fast-weight state cannot hold arbitrary context [ESTIMATE, well known; https://arxiv.org/abs/2412.06464v3]. With the pointer deleted, in-stream copying carries the load. Mitigations: delta-rule overwrite, K ≤ 16, re-retrieval mid-generation (SEARCH calls), test-time memory M. Allowed ablation: a delta-rule copy memory (key = prefix state, value = next span), which is linear attention and permitted by Rule 9.
9. **Contamination makes a win fake.** RETRO exploits leakage more than baselines [VERIFIED: https://arxiv.org/abs/2112.04426]; n-grams miss paraphrases [VERIFIED: https://arxiv.org/pdf/2311.04850v1]. Mitigations: S0 is week 1; URL-level WikiHow/ActivityNet/exam-site removal; exam-format filter; gold-in-top-k logging per row; private post-cutoff Tier C set (including Korean) as the headline sanity check.
10. **Compute availability and toolchain.** TPU oversubscription (T4-only plan: R1 at ~week 30); JAX on sm_75 unverified (eval moves to TPU); recompilation eats session time (shape buckets + cache).
11. **Solo-developer complexity.** ORRERY-0 first; each of JEPA, diffusion, energy descent, test-time memory and the grid reasoner is added only after it beats the control at equal FLOPs. The PyTorch port is dropped.
12. **Data licensing.** Korean licenses often unknown; teacher licenses vary; manifest for every source; no closed-API outputs.
13. **Judge gaming.** Disjoint judge families, length control, both orders, 100-pair hand check.
14. **Biosafety** (north star). Hard stop on dual-use content and a human review gate; never traded for capability.

**Bottom line.** ORRERY v2 is the same bet (test-time compute, an external human-authored datastore and a learned universal verifier do the work GPT-3.5 does with parameters) with the loopholes closed and a control arm attached. At R1 it should win NQ and has even odds on ARC-AGI, with live chances on BBH, DROP, HAE-RAE, MATH and KMMLU. It will likely lose code, MMLU-class knowledge, commonsense and chat. Every red row now has a knob, a slope and a falsifiable forecast, and three of them carry no sub-100M forecast at all, which is the most important honest sentence in this document. Weeks 1–16 will show whether the knobs move the rows.

---

## Changelog v1 → v2

**Fatal**
- **F1 (hidden retriever params / stale index):** encoder and decoder merged into one trunk; embedding + trunk layers 1–4 + projection frozen after S1a and used as the retriever (keys and queries from the same tensors, counted once); S1a explicitly BM25-only; index built once; per-stage rebuild kept as a measured fallback (cheap at the 10B-token tier, which partly rebuts the 3-week cost estimate that assumed 150B tokens on T4s).
- **F2 (FiLM bottleneck):** FiLM removed; plan enters as 64 soft prefix tokens per plan chunk (full 64×640 resolution, in every layer's state); variable plan chunks; cycle-consistency and plan-swap probes added to E1.
- **F3 (LLM-generated datastore):** headline datastore human-authored only; synthetic data training-only; provenance tags and copy-share reporting; synthetic-store track non-gating.

**Major**
- **M1:** FLOPs recomputed with output projection, EMA forward and hard negatives (1.09 GFLOP/target token average); MFU planned at 15–20%; throughput 14–19B/week; S1 cut to 25B; day-1 measurement rule.
- **M2:** chunkwise WY/UT delta-rule form + `lax.scan` instead of `associative_scan`.
- **M3:** headline store capped at 10B curated tokens (T1), 25B on laptop (T2) conditional on SSD space; 300B tier cancelled; on-disk compressed BM25 via mmap; storage arithmetic includes raw text.
- **M4:** 512-bit codes + IVF + PQ/int8 rescoring; recall@20 measured on KILT before use; 1024-bit fallback; mxbai quality figures no longer assumed.
- **M5:** GRPO rollouts and updates co-located on the TPU.
- **M6:** gated checkpoint trained without any MC-benchmark train splits; fully split-free ablation also reported.
- **M7:** URL-level removal of WikiHow/ActivityNet sources, Korean exam sites and benchmark mirrors; exam-format filter; gold-in-top-k logging; "Korean exam datastore" removed.
- **M8:** pooled chat eval (N ≈ 1,465, CI ≈ ±2.6), disjoint judge families, length control, both orders, 100-pair hand check.
- **M9:** 4.0M attention-free TRM-MLP-style grid reasoner with test-time training, input-type routing disclosed; ARC-AGI P lowered 0.85 → 0.5.
- **M10:** copy pointer deleted; Transformer rule extended to retrieved and generated positions; linear attention / GDN explicitly permitted.
- **M11:** E3 uses 5 sizes at fixed tokens/param, separate curves per axis, variable plan length, kill condition against an equal-compute ORRERY-0 slope.
- **Chat/renderer capacity:** single trunk at d=640 used for all roles (red-team "one trunk, several roles" fix).
- **Solo-developer risk:** ORRERY-0 built first as control arm; latent components gated by E1/E2; PyTorch port dropped.

**Minor**
- Inference table now counts energy-gradient backward passes (~3x per step); eval modes pinned per row; seeds only on sampled rows; eval on TPU; JIT shape buckets and compile cache; in-run device assertion.

**Withdrawn claims**
- v1's unverified recollections of ~300M discriminative encoders scoring in the 80s on HellaSwag and ~79 on WinoGrande are removed.

**Parameter count:** 91,368,970 → **95,813,058** (headroom 4,186,942), recomputed by script.

**Strategic:** campaign framing (R1/R2/R3 under 100M; 1B tier does not count for the mid-boss); every red row has a knob, slope and falsifiable forecast; three rows (HellaSwag, WinoGrande, chat) explicitly carry no sub-100M forecast. TRC application in week 1. Biosafety gate unchanged.
