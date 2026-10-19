# ALEMBIC-0: Blueprint v1

**A sub-100M, attention-free, neuro-symbolic compiler that distills tasks into programs, facts into a library, and judgement into a bidirectional judge.**

Chief-architect synthesis of five revised proposals (CORTEX-v2, COMPILER-0 v2, LIBRARIAN v2, ORRERY v2, OUROBOROS-0 v2) and three judge lenses. Date: 2026-10-19.

Labels: **[VERIFIED: url]** = a number or fact taken from a primary source cited in the research digests or proposals. **[ESTIMATE]** = my arithmetic or judgement. **[HYPOTHESIS]** = an untested design bet. Every derived number (parameters, FLOPs, hours, bytes, per-query cost) comes from one script, `/home/user/hip/asi-blueprint/calc/alembic_budget.py`. **If this document and the script disagree, the script wins**, and CI re-runs the script on every config change.

**Why the name.** An alembic is the vessel alchemists used to distill. The system distills a messy prompt into a typed program, distills facts into an external library, distills open teachers into *rankings* rather than text, and distills its own verified self-play into its next training set. The "-0" marks the first round of a campaign, not a finished system.

**Lineage, so the reader knows what was grafted and why:**
- **Spine: COMPILER-0.** Typed program IR, executors, bidirectional judge tower, skill library, and a north-star mapping that changes only the tool registry. It had the highest summed score across the three lenses (feasibility 7.0, win 2.4, north star 8.2).
- **Engineering skeleton: OUROBOROS-0 (scaler).** d=640 with 5 heads × 128, per-query recurrence r, `<think>` tokens, suffix-automaton copy, pre-registered parameter pools, the scope fence, and the "win-or-extend" rule.
- **Admission law: ORRERY (worldmodel).** ORRERY-0 control-arm discipline, frozen-lower-layer retriever keys, state forking per multiple-choice option, and the pooled chat gate.
- **Scaling methodology: CORTEX + LIBRARIAN.** A published-before-training scaling grid fit on smooth retrieval-conditioned NLL, a concept skeleton with long-tail fact dropout, the pointer-not-mixer rule, the count-vs-fit parameter rule, and the human-only gated store.

---

## 0. CANONICAL NUMBERS SHEET

Every later section uses exactly these numbers.

### 0.1 Parameter budget (script output)

Shared shape: d = 640, 5 heads × head_dim 128, SwiGLU hidden 1536, vocab 32,768, embedding bottleneck 256, GDN chunk 128.

One Gated DeltaNet (GDN) layer has **5,012,480** params:
- q, k, v, o projections, 4d² = 1,638,400
- output gate, d² = 409,600
- α/β gates, 2·d·H = 6,400
- short conv (k=4) on q, k, v, 3·d·4 = 7,680
- SwiGLU, 3·640·1536 = 2,949,120
- two RMSNorms = 1,280

| # | Module | Composition | Trainable params |
|---|---|---|---|
| 1 | Unigram tokenizer scores (counted, rule F2) | 32,768 scalars | 32,768 |
| 2 | Factorized token embedding, tied to the LM head | 32,768×256 + 256×640 | 8,552,448 |
| 3 | Prelude | 3 GDN layers (L1–L2 + proj = retriever key encoder) | 15,037,440 |
| 4 | Recurrent core (weight-tied, iterated r times per query) | 4 GDN layers, counted once | 20,049,920 |
| 5 | Loop-input injector | Linear(2d→d) + bias | 819,840 |
| 6 | Coda | 4 GDN layers | 20,049,920 |
| 7 | Final norm | 640 | 640 |
| 8 | Judge tower (right-to-left GDN scan over trunk states) | 3 GDN layers | 15,037,440 |
| 9 | Judge input norm | 2·640 | 1,280 |
| 10 | Judge fusion MLP, 4 heads (energy/verify, chat reward, process value, difficulty) | 1280→1024→4 | 1,315,844 |
| 11 | Retrieval query/key projection | 640×256 | 163,840 |
| 12 | Controller head (r-level 6, N-level 4, route 8) | 640→64→18 | 42,194 |
| 13 | Segment embeddings (prompt / chunk / tool / IR / answer / judge / think / pad) | 8×640 | 5,120 |
| | **TOTAL, M1 base** | | **81,108,694** |
| | Headroom under 100,000,000 | | 18,891,306 |

**Pre-registered pools.** These are released only by a named experiment outcome:

| Pool | Content | Params | Released by |
|---|---|---|---|
| G (generator) | +2 coda layers | 10,024,960 | X1 chat slope positive (§10) |
| J (judge) | +1 judge layer | 5,012,480 | X1 commonsense slope positive |
| R (phrase query) | 640→256 projection for admission candidate A3 | 163,840 | A3 admitted |
| | **Worst case with all pools** | | **96,309,974** |

The 40k-vocabulary contingency for Korean costs +1.85M and also comes from headroom (§6.6).

Active parameters per generated token, and effective depth (3 + 4r + 4):

| r | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| N_active | 64.5M | 85.4M | 127.1M | 210.6M | 377.6M | 711.5M |
| Effective depth (layers) | 11 | 15 | 23 | 39 | 71 | 135 |
| Forward FLOP/token | 1.29e8 | 1.71e8 | 2.54e8 | 4.21e8 | 7.55e8 | 1.42e9 |

The judge adds 16.35M params and 3.27e7 FLOP per scored token.

### 0.2 Datastore (gated track: human-authored or self-generated only)

| Item | M1 value |
|---|---|
| Store tokens | **20B** (laptop SSD). Kaggle mounts a 10B subset |
| Chunks | 156M × 128 tokens |
| Dense keys | 256-bit binary codes, **5.0 GB** RAM, IVF |
| Rescoring | int8 256-d **40 GB** mmap on the laptop; PQ-32 **5.0 GB** on Kaggle |
| BM25 | compressed on-disk inverted index over a **~6B-token core**: en+ko Wikipedia, ko-Wiktionary, open textbooks, Python docs. ≈ 5–10 GB |
| Text | uint16 memmap, 40 GB raw, ≈ 20–25 GB zstd |
| Kaggle mount (10B subset) | ≈ 24 GB |
| Laptop footprint (20B) | ≈ 100–110 GB. **Day-1 check: free SSD ≥ 150 GB, else the store tier drops to 10B** |
| Key-encoder rebuild | 3.75e7 FLOP/token, 20B tokens = 7.5e17 FLOP = **1.1 TPU-h** (+ I/O) |

### 0.3 Training tokens and compute per stage

Planning MFU is 12% on TPU v5e-8, which gives 6.81e17 FLOP per TPU-hour and **1.36e19 FLOP per 20-hour TPU-week**. T4×2 at 20% MFU gives **2.81e18 FLOP per 30-hour week**.

| Stage | Tokens | FLOP | TPU-h |
|---|---|---|---|
| S1 Language + reading pretrain (r̄=2) | 30B scanned | 1.54e19 | 22.6 |
| S2 Procedures + tool/IR traces (r̄=4) | 15B | 1.14e19 | 16.8 |
| S3 Judge (trunk forward + judge train) | 3B | 8.1e17 | 1.2 |
| S4 Instruction/chat SFT | 2B | 1.0e18 | 1.5 |
| S5 RLVR (GRPO) | 2B rollout + 1B train | 1.27e18 | 1.9 |
| S6 Preference | 0.3B pairs (0.6B tokens) | 3.1e17 | 0.5 |
| **Main subtotal** | 52.3B + rollouts | **3.02e19** | **44.4** (2.2 TPU-weeks) |
| Experiments X1–X5 | — | 1.81e19 | 26.6 |
| **All × 1.8 (failures, restarts)** | | **8.7e19** | **127.8 (6.4 TPU-weeks)** |

- **Quota in the 12-week window:** about 12 TPU-weeks (240 h), so about 47% slack.
- **T4×2 window:** 11 weeks × 30 h = 330 h.
  - Teacher data: **≈ 135 h**.
  - Index builds, RL rollouts, evals: the rest.
- **T4-only fallback:** S1 drops to 12B and S2 to 8B, giving ≈ 4.8e19 with the multiplier, or **≈ 17 T4-weeks**.

### 0.4 Targets (max reported GPT-3.5) and gates (target + 95% CI half-width)

| Row | Target | Gate | M1 estimate | P(win at M1) |
|---|---|---|---|---|
| NQ-open EM | 38.6 | 40.2 | 38–48 | 0.55 |
| TriviaQA (harness EM, 5-shot) | 85.8 | 86.4 | 65–78 | 0.05 |
| MMLU 5-shot | 71.4 | 72.1 | 42–55 | 0.02 |
| KMMLU 5-shot direct | 42.47 | 43.0 | 38–48 | 0.35 |
| HAE-RAE (mean of harness subsets) | 55.4 | 57.9 | 46–58 | 0.20 |
| KoBEST avg | 79.3 | 80.8 | 62–76 | 0.10 |
| GSM8K 0-shot | 78.1 | 80.3 | 60–76 | 0.15 |
| MATH 4-shot | 34.1 | 35.4 | 22–36 | 0.25 |
| BBH 0-shot | 68.3 | 69.4 | 55–70 | 0.30 |
| DROP F1 | 64.1 | 65.1 | 56–70 | 0.40 |
| HumanEval+ / HumanEval | 70.7 / 76.8 | 77.7 / 83.8 | 35–58 | 0.03 |
| MBPP+ / MBPP | 69.7 / 82.5 | 74.3 / 87.1 | 45–63 | 0.03 |
| ARC-Challenge | 87.4 | 89.3 | 50–68 | 0.03 |
| HellaSwag 10-shot | 85.5 | 86.2 | 48–68 | 0.03 |
| WinoGrande 5-shot | 81.6 | 83.7 | 56–70 | 0.02 |
| ARC-AGI-1 pass@2 (400 eval) | 10.6 | 13.6 | 10–25 | 0.55 |
| Chat pooled pairwise vs stored GPT-3.5 (N≈1,465) | 50% | 52.6% | 10–28% | 0.02 |
| Korean MT-Bench pairwise vs open proxy (Arena-Ko ≥ 1049.6) | 50% | 50% + CI | 8–25% | 0.02 |

**P(every gated row at M1) < 0.1% [ESTIMATE].** That is the honest headline. The plan exists to turn it into a measured campaign (§1.9, §8, §10).

### 0.5 Inference cost per query (script output; cap 2e13 neural FLOP per query, ARC-AGI excepted and disclosed)

| Query type | Work | FLOP | Wall-clock [ESTIMATE] |
|---|---|---|---|
| Factoid (NQ) | 1,088-token prefill (8 chunks + prompt) at r=2, 20 decode tokens | 1.9e11 | 0.3–1 s |
| MC (MMLU-style), 4 options | per-option retrieval, 4 × 712 tokens at r=2, plus the judge | 5.8e11 | 0.5–1.5 s |
| GSM8K | 32 IR programs × 250 tokens at r=4, judge verify, ≤64 executions | 2.6e12 | 3–8 s (CPU-bound) |
| Chat turn | plan + 8 drafts × 400 tokens at r=4, judge, one revise pass | 1.3e12 | 3–8 s |
| HumanEval | 16 candidates × 400 tokens × 4 rounds at r=8, plus judge | 1.2e13 | 15–60 s |
| MATH hard | 64 × 600 tokens at r=8 | 1.6e13 | 20–90 s |
| ARC-AGI-1 task | DSL program search + verification on demo pairs | ~1e15 | 2–10 min (separate monthly laptop job) |

Tool limits: ≤10 tool calls per program, ≤200 executions per query, ≤60 s CPU per query (ARC excepted). FLOPs and wall-clock are reported per row.

---

## 1. Rules of the game

These are binding and pre-registered. They change only by a dated, public amendment made **before** any affected eval is run.

**F1. The cap.** Under 100,000,000 trainable parameters executed at inference, counted on the *instantiated* JAX module by a CI assertion (`sum(p.size for p in leaves(params)) < 100e6`; it currently returns 81,108,694). The count includes:
- embeddings, every trunk layer, the tied core counted once, the judge, every head and projection;
- the initial weights of any test-time memory, if ever admitted;
- the learned scores of the unigram tokenizer.

**F2. Count vs fit** (from LIBRARIAN):
- Anything produced only by **counting or indexing** a frozen store is *data*, disclosed in bytes: BM25 statistics, inverted indexes, suffix automata, the alias dictionary.
- Anything **fitted by gradient or EM** either counts toward the cap (tokenizer scores, learned projections) or is reported in bytes (IVF centroids, PQ codebooks, binarization thresholds; these are kilobytes to megabytes).
- Training-only weights (any EMA target, auxiliary probes) and all teacher models are **not counted but disclosed**, with name, size, family and licence.

**F3. Gated store = human-authored or self-generated.** Every inference store holds only two kinds of content. The stores are the datastore, BM25 core, exemplar bank, skill library and any later phrase index.
- **(a)** Pre-existing human-authored public text, code or human-curated structured data. Web crawls are limited to Common Crawl snapshots from **2022 or earlier**, which is mechanical with FineWeb-Edu's per-crawl subsets [VERIFIED: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu]. The assumption that pre-2023 crawls hold little LLM text is an [ESTIMATE].
- **(b)** Artifacts produced by ALEMBIC itself that passed an executable check.
- **Teacher text never enters any inference store.** That covers Cosmopedia (Mixtral-generated [VERIFIED: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia]), OpenMathInstruct-2 solutions, Magpie-style data and regenerated chat. Retrieving it would be a lookup-table copy of a >100M network.
- Teacher-written code is banned from the skill library.
- Every vector in every index is produced by the in-budget encoder.
- A labelled "synthetic-store" track may be reported. It never gates.

**F4. Allowed at inference:**
- the frozen, dated, hashed offline store;
- an **allow-listed sandbox image**: Python stdlib, NumPy, SymPy, NetworkX, plus Z3 once admitted. No ML libraries, no weight files, no network, image hash published;
- a deterministic test runner for tests the model wrote itself, including tests extracted from I/O examples in the prompt;
- Hodel's ARC DSL as library text [VERIFIED: https://arxiv.org/pdf/2404.07353].

Everything runs on CPU with 0 parameters.

**F5. Banned at inference:**
- any neural network over 100M, or any network outside the counted budget;
- LLM APIs and live web search;
- eval items or their solutions in any store;
- running hidden benchmark tests during generation;
- routing, prompts or heads keyed on dataset identity;
- "abstain" in eval mode.

*Input-type* routing (for example, "the input is a 2-D integer grid") is allowed. The detector's code is published and its decisions are logged per row.

**F6. Architecture rule, pointer-not-mixer:**
- No softmax (or Hopfield-equivalent [VERIFIED: https://arxiv.org/abs/2008.02217]) normalization over token positions or over retrieved items, anywhere in the inference forward pass.
- A softmax may appear **only as an output distribution over discrete choices**: vocabulary, r-level, route, or an argmax over a few candidate answers. A chosen item re-enters the model only as discrete tokens.
- Linear-attention / delta-rule recurrences with a constant-size state are explicitly permitted.
- Softmax inside *training losses* (cross-entropy, InfoNCE over candidates, listwise ranking) is a loss, not a token mixer, and is permitted.

**F7. Training data:**
- Open-licensed teachers only, each logged.
- **No GPT-3.5/4 output anywhere.** OpenAI's terms forbid using output "to develop models that compete" [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/], and it would contaminate the head-to-head. This excludes TinyGSM, MetaMathQA, original UltraChat and TinyStories from weights training; they are cited as evidence only.
- Every SmolTalk and Tulu-3 subset card is audited, and subsets with OpenAI-generated or unclear responses are dropped.
- NC subsets (for example No Robots in Tulu-3 [VERIFIED: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/main/README.md]) go in a manifest so a commercial fork can drop them.
- A **no-teacher track** is reported.

**F8. Benchmark train splits:**
- Official *train* splits (NQ, TriviaQA, GSM8K, MATH, MBPP-train, DROP, HellaSwag, WinoGrande, ARC, KoBEST, KILT) may enter **one shared training mix** for **one checkpoint**. Their use is disclosed. This is the **Open track**, which gates.
- A **Strict track** with zero multiple-choice benchmark train splits is trained and reported for every row (ORRERY's discipline).
- Any row whose win holds on Open but not on Strict is flagged "win depends on train split" in every publication.
- Test and validation splits are never trained on.

**F9. Decontamination** of training data **and every store**, against every eval set and its mirrors:
1. 13-gram + MinHash near-duplicate filtering.
2. An embedding scrub at cosine > 0.9 with our own key encoder, because n-grams miss rephrasings [VERIFIED: https://arxiv.org/pdf/2311.04850v1].
3. **Korean character 20-grams**, because word 13-grams miss short Korean multiple-choice stems.
4. **Function signature + docstring matching** for code.
5. **Whole-source blocklists:** wikihow.com and ActivityNet captions (HellaSwag's sources), Korean license-exam and quiz sites (KMMLU/HAE-RAE sources), benchmark-hosting sites, and GitHub mirrors of HumanEval/MBPP.
6. An **exam-format filter**: drop documents with ≥3 numbered option blocks (①②③④ or (A)–(D)) followed by an answer key [ESTIMATE heuristic].
7. A **per-item retrieval audit**:
   - generative QA: the % of top-k chunks containing the gold string;
   - multiple choice: max n-gram overlap of the stem and of each option with the retrieved chunks;
   - continuation tasks: "gold ending ⊂ top-k" must be ≈ 0% before any HellaSwag number is published;
   - any item whose stem overlaps a retrieved chunk by >50% of characters is flagged, and scores are reported with and without flagged items.
8. A **private post-cutoff set** (fresh math, code, Korean and chat prompts written after the store snapshot) is the headline sanity number.
9. Every store-dependent row is reported **with and without the store**.
10. The procedural-generator manifest is frozen and hashed before generator design. **Five BBH task families are drawn by a seeded random draw and committed as held out**; no generator may target them.

**F10. Scoring:**
- **Generative everywhere.** Multiple choice is answered by an emitted letter, which the system's own judge may choose, because the judge is part of the system. Log-likelihood numbers are secondary.
- Shot counts match the target row.
- One checkpoint and one config serve every row.
- **Each row's inference mode (r, N, K, tool budget) is pinned before any test result is seen.**
- 3 seeds for sampled rows.
- **Win = score ≥ target + 95% CI half-width**, with matched metrics (EM vs EM). Paired bootstrap is used where stored GPT-3.5 per-item outputs exist, otherwise a binomial CI.

**F11. Judging chat:**
- The training-time judge (family A) and the eval panel (≥2 judges, families B and C) come from **disjoint model families**.
- Both answer orders, length-controlled variants only.
- The eval panel is calibrated first: Spearman ρ ≥ 0.8 against published GPT-4-judge MT-Bench scores of ≥5 models whose stored answers exist [VERIFIED stored answers: https://huggingface.co/spaces/lmsys/mt-bench]. **If the panel misses ρ ≥ 0.8, no chat number is published.**
- Plus a 100-pair blind hand check by the author, and Korean volunteer raters for the Korean gate.
- **Eval judges are never used as a reward or a training filter.**

**F12. Compute disclosure.**
- Per row: mean FLOPs, wall-clock, candidates, r, tool calls, retrieval provenance, following the FLOPs-matched framing of Snell et al. [VERIFIED: https://arxiv.org/abs/2408.03314].
- Per-query neural cap is 2e13 FLOP. ARC-AGI is disclosed separately at ~1e15 per task.

**F13. Win-or-extend** (from OUROBOROS):
- The cap, the rules and the one-checkpoint requirement never move.
- No victory is declared until **every** gated row (Tier A + B) clears its gate on one checkpoint.
- Each round grows only the axes that are free under the cap: store, test-time compute within the cap, tools, verified self-play data.
- Every round publishes the full scoreboard, losses included.
- If the measured slopes say a row cannot be reached under 100M, that is published with the curve, and the decision about the 1B track goes to the user. **The boss is never quietly redefined.**

**F14. Biosafety hard stop** (§9.5). It applies in every round and is never traded for capability.

---

## 2. The GPT-3.5 scoreboard

"GPT-3.5" is at least four API snapshots (0301/0314, 0613, 1106, 0125), and published numbers vary by up to 20 points by protocol. HumanEval is 48.1 in the GPT-4 report but 76.8 on EvalPlus; GSM8K is 57.1 vs 78.1 [VERIFIED: https://arxiv.org/abs/2303.08774, https://evalplus.github.io/leaderboard.html, https://arxiv.org/abs/2404.14219]. **Rule: the target is the maximum reported score per row, across snapshots, sources and prompt languages.** We fight the strongest GPT-3.5 on record. We cannot re-measure it (0 KRW), so targets are published numbers, and chat is compared head-to-head against stored public GPT-3.5 outputs.

### Tier A: gated academic rows (EleutherAI lm-evaluation-harness task names [VERIFIED: https://github.com/EleutherAI/lm-evaluation-harness])

| Domain | Row (harness task) | Target | Source of target | N | Gate |
|---|---|---|---|---|---|
| Knowledge | MMLU 5-shot (`mmlu_generative`) | 71.4 (1106) | [VERIFIED: https://arxiv.org/abs/2404.14219] | 14,042 | 72.1 |
| Korean knowledge | KMMLU 5-shot direct (`kmmlu_direct`) | 42.47 | [VERIFIED: https://arxiv.org/abs/2402.11548] | ~35k | 43.0 |
| Korean culture | HAE-RAE 0-shot (`haerae`) | 55.4 (En prompt; Ko-prompt subset mean 49.45 [ESTIMATE]) | [VERIFIED: https://arxiv.org/abs/2309.02706] | ~1.5k [ESTIMATE] | 57.9 |
| Korean NLU | KoBEST 0-shot (`kobest_*`) | 79.3 (En-prompt avg) | [VERIFIED: https://arxiv.org/abs/2309.02706] | — | 80.8 [ESTIMATE CI] |
| Math | GSM8K 0-shot CoT | 78.1 (1106) | [VERIFIED: https://arxiv.org/abs/2404.14219] | 1,319 | 80.3 |
| Math | MATH 4-shot (`minerva_math_*`) | 34.1 | [VERIFIED: https://arxiv.org/abs/2312.11805] | 5,000 | 35.4 |
| Code | HumanEval+ / HumanEval | 70.7 / 76.8 | [VERIFIED: https://evalplus.github.io/leaderboard.html] | 164 | 77.7 / 83.8 |
| Code | MBPP+ / MBPP | 69.7 / 82.5 | same | 378 | 74.3 / 87.1 |
| Reasoning | BBH 0-shot | 68.3 | [VERIFIED: https://arxiv.org/abs/2404.14219] | 6,511 | 69.4 |
| Reading + arithmetic | DROP 3-shot F1 | 64.1 | [VERIFIED: https://arxiv.org/abs/2303.08774] | 9,536 | 65.1 |
| Commonsense | HellaSwag 10-shot | 85.5 | same | 10,042 | 86.2 |
| Commonsense | WinoGrande 5-shot | 81.6 | same | 1,267 | 83.7 |
| Science | ARC-Challenge 10-shot | 87.4 (1106) | [VERIFIED: https://arxiv.org/abs/2404.14219] | 1,172 | 89.3 |
| Open-domain QA | TriviaQA 5-shot | 85.8 (1106) | same | ~11k [ESTIMATE] | 86.4 |
| Open-domain QA | NQ-open 0-shot | 38.6 (1106) | [VERIFIED: https://arxiv.org/abs/2407.02485] | 3,610 | 40.2 |
| Abstraction | ARC-AGI-1 public eval, pass@2 | 10.6 (text-davinci-003, a GPT-3.5-series model, 85/800 tasks) | [VERIFIED: https://arxiv.org/abs/2307.04721] | 400 | 13.6 |

The ARC-AGI target uses the strictest GPT-3.5-family number on record, per the max rule. The GPT-4o context figure of 5% [VERIFIED: https://arcprize.org/blog/oai-o3-pub-breakthrough] is lower.

### Tier B: gated chat (zero-budget protocol)

| Row | GPT-3.5 reference | ALEMBIC protocol |
|---|---|---|
| **Pooled chat gate** | MT-Bench 8.39 (0613) [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard]; AlpacaEval 2 LC 22.35 (0613) [VERIFIED: https://github.com/tatsu-lab/alpaca_eval]; Arena-Hard v0.1 24.8 (0613) [VERIFIED: https://github.com/lmarena/arena-hard-auto/blob/main/misc/past_leaderboards.md] | Pairwise vs stored GPT-3.5 answers on MT-Bench (160 turns) + AlpacaEval (805) + Arena-Hard (500). N ≈ 1,465, CI ≈ ±2.6. **Gate: pooled length-controlled win rate ≥ 52.6%**, both orders, F11 panel. Per-benchmark numbers are also reported |
| **Korean chat gate** | Arena Korean 1049.6 (0125) [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard] | Stored GPT-3.5 Korean chat outputs have not been found. Korean MT-Bench-style prompts are compared pairwise against an **open proxy model whose Arena-Korean rating is ≥ 1049.6** (read off the leaderboard on day 1), with blind volunteer Korean raters, both orders. Gate: win rate > 50% + CI |
| English proxy (secondary) | Gemma-2-2B-it Arena 1163.8 > gpt-3.5-0125's 1132.3 [VERIFIED: same leaderboard] | Pairwise on fresh prompts. Transitivity is a [HYPOTHESIS], so this is reported but not gated |
| Chatbot Arena listing | 1132 overall / 1226 style-controlled | Stretch goal: an LMArena listing as the real arbiter. A model this small runs on CPU [ESTIMATE] |

### Tier C: reported, not gating (anti-overfit and north star)

- GPQA 29.0 and MedQA 63.4 [VERIFIED: https://arxiv.org/abs/2404.14219].
- Natural2Code-style fresh code, GPT-3.5 62.3 [VERIFIED: https://arxiv.org/abs/2312.11805].
- ARC-AGI-2: GPT-3.5 ≈ 0 [ESTIMATE]; TRM gets 7.8 [VERIFIED: https://arxiv.org/abs/2510.04871].
- The private post-cutoff set.
- The no-store, no-teacher and Strict (no-split) tracks.
- KMMLU-Hard is reported only, because it was built from items GPT-3.5 failed [VERIFIED: https://arxiv.org/abs/2402.11548].

---

## 3. First-principles decomposition: where a dense LLM spends parameters, and where ALEMBIC moves each job

| Job | How a dense LLM pays for it | Evidence that it is expensive | ALEMBIC's move | What still lives in weights |
|---|---|---|---|---|
| **1. Long-tail facts** | ~2 bits/param of knowledge [VERIFIED: https://arxiv.org/html/2404.05405]; 100M ≈ 25 MB of facts [ESTIMATE] | SmolLM2-135M gets 4.1 on TriviaQA after 2T tokens [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct] | **Human-authored 20B-token store** + BM25 + binary dense keys + per-option retrieval + suffix-automaton copy. RETRO matched GPT-3 on the Pile with ~25× fewer params [VERIFIED: https://arxiv.org/abs/2112.04426]; FiD with 440M total reached 48.2 NQ EM [VERIFIED: https://arxiv.org/abs/2007.01282] | A **concept skeleton**: frequent entities, taxonomy, "what kind of thing is this", so the trunk can write good queries (LIBRARIAN). Long-tail fact dropout keeps rare facts out of the weights (§6.3) |
| **2. Exact computation** | Emulated digit carrying and variable tracking | GSM8K 1.4 at 135M [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M] | **Typed program IR** → Python/SymPy/NetworkX sandbox. PAL beat PaLM-540B CoT by 15 points [VERIFIED: https://arxiv.org/abs/2211.10435] | The *translation* from language to program (autoformalization). This is the irreducible core skill |
| **3. Serial reasoning depth** | One layer = one step; depth costs params | TRM 7M scores 44.6% on ARC-AGI-1 by recursion [VERIFIED: https://arxiv.org/abs/2510.04871]; Huginn reaches "50B-equivalent" from depth [VERIFIED: https://arxiv.org/abs/2502.05171] | **Weight-tied 4-layer core iterated r ∈ {1…32} per query** + learned `<think>` tokens. Effective depth goes from 11 to 135 layers with no new params | The 4 tied layers: a reusable "step" procedure |
| **4. "Being right"** | Implicit in the likelihood | TinyGSM: "the verifier seems to play a bigger role than generation model size" [VERIFIED: https://arxiv.org/html/2312.09241] | **Exact verifiers first** (execution, SymPy, self-tests, ARC demo pairs); then the **judge tower** where none exists; sampling N under a compute-optimal controller [VERIFIED: https://arxiv.org/abs/2408.03314] | The 16.35M judge, which reads the whole trunk (§5.6) |
| **5. Plausibility / commonsense** | An implicit world model inside the generator | WinoGrande 51–53 at every size ≤ 270M [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M] | **Discrimination, not generation**: per-option state forks scored by a bidirectional judge trained by *teacher-ranking distillation* (COMPILER) | Judge + trunk. **Still parameter-bound**; the hardest wall (§8.3) |
| **6. Surface fluency** | Memorized phrasing, ~every output token | MT-Bench 1.98–2.52 at 125–135M [VERIFIED: https://arxiv.org/html/2402.14905] | Shrink the generator's job: **plan → section-by-section realize → copy exact spans via suffix automaton → critique → best-of-N under judge reward**. Top-k logit distillation from an open teacher during training (Qwen3 strong-to-weak recipe [VERIFIED: https://arxiv.org/html/2505.09388]). The phrase index is an admission candidate (A3) | Trunk generator capacity, plus Pool G if earned. **Also parameter-bound; the other hardest wall** |
| **7. Instruction format** | Learned from RLHF data | IFEval rises 29.9 → 51.2 → 59.2 from 135M to 0.6B [VERIFIED: tiny_sota digest sources] | **Format constraints compiled into executable checks** (word count, bullet count, language-ID, JSON validity), enforced at decode time with repair | Small |

**What remains in the 81.1M weights:** a compiler (language to typed program and back), a reader (scan evidence, copy spans), a reusable recurrent step, and a judge. Everything else is a disk, CPU or FLOP problem. Disk, CPU and FLOPs are the resources a 0-KRW builder can grow.

The judges agreed on one honest corollary: **jobs 5 and 6 cannot be fully externalized**, because no exact verifier exists for "is this ending plausible" or "is this answer good". They are where the fight is decided. §8.3 states what is known, what is bet, and how each bet is killed early.

---

## 4. Architecture overview

### 4.1 Diagram

```
                ┌────────────────────── OFFLINE, 0 params (data, disclosed, hashed) ─────────────────────┐
                │ STORE 20B tok human-authored (pre-2023 crawl cut): en/ko Wikipedia, ko-Wiktionary,       │
                │   FineWeb-Edu(≤2022), FineWeb-2-ko(≤2022), OpenWebMath/arXiv, Python-Edu/Stack-permissive,│
                │   StackExchange-style human Q&A, open textbooks                                          │
                │   dense: 156M chunks × 256-bit codes (IVF, 5.0 GB RAM) → int8 / PQ-32 rescore             │
                │   BM25: compressed on-disk index over ~6B-token core (Wikipedia, Wiktionary, textbooks)   │
                │ SKILL LIBRARY (text): human OSS code + self-play verified IR fragments, frozen per eval   │
                │ SANDBOX (allow-listed, hashed): stdlib · numpy · sympy · networkx · test runner · ARC DSL │
                └──────▲────────────────────────────────────────────▲─────────────────────▲───────────────┘
                       │ SEARCH(q) → RRF top-K (K=4..32)             │ PY/SOLVE/CHECK       │ suffix automaton
                       │                                              │ async CPU pool       │ over retrieved
  prompt ─► TOKENIZER (32k unigram, ko pre-split rules) ─► EMB (32768×256 → 640, tied)    │ text + prompt
                       │                                              │                      │
            ┌──────────┴──────────── TRUNK: causal Gated DeltaNet, d=640, 5 heads×128 ──────┴────────────┐
            │ PRELUDE 3 layers  (L1–L2 + 640→256 proj = retriever key/query encoder, frozen after S1)   │
            │ CORE 4 weight-tied layers × r   (r ∈ {1,2,4,8,16,32} fixed per query; injector each loop) │
            │ CODA 4 layers → tied LM head                                                              │
            │ mode tokens (0 params): <PLAN> <READ> <IR> <REALIZE> <JUDGE> <think> <copy> <search> ...  │
            │ retrieved chunks are read IN-STREAM (prefix tokens); state is FORKED per MC option          │
            └───────┬──────────────────────┬─────────────────────────────┬───────────────────────────────┘
                    │ IR program tokens     │ answer / prose tokens        │ per-token states h_1..h_T
                    ▼ (grammar-constrained) ▼                              ▼
          ┌──────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────────────────────┐
          │ EXECUTORS (CPU)  │  │ CONTROLLER (code, 0 par.)│  │ JUDGE TOWER: 3 GDN layers scanning h     │
          │ run whole program│  │ budget (r,N,K) from head │  │ RIGHT-TO-LEFT + fusion MLP → 4 heads:    │
          │ result → tokens  │  │ sample N, exact-verify,  │  │ energy/verify · chat reward ·            │
          │ spliced in stream│  │ judge-rank, vote, repair │  │ process value · difficulty               │
          └──────────────────┘  └──────────────────────────┘  └──────────────────────────────────────────┘
```

### 4.2 Dataflow for one query

1. **Tokenize.** Korean goes through a rule-based josa/eomi pre-splitter (0 learned params) before the unigram model.
2. **Probe** (r=1 over the prompt). The controller head reads the pooled prelude state and the judge's difficulty output, and chooses the budget (r, N, K). The budget is fixed for the whole episode, or for one answer segment.
3. **Retrieve.**
   - The query vector is the mean-pooled L2 state through the 640→256 projection, then sign().
   - IVF Hamming search over 156M binary codes, rescoring the top 1,000 with int8 (laptop) or PQ-32 (Kaggle).
   - In parallel, BM25 over the ~6B-token core, with Korean character bigrams.
   - Reciprocal-rank fusion produces the top K.
   - **Multiple choice: one query per option** (stem + option_i).
   - `<search>` may be re-issued up to 3 hops with rewritten queries.
4. **Read.** The top-K chunks are scanned **in-stream** as prefix tokens before the prompt, so evidence enters the recurrent state through ordinary delta-rule writes. Gated DeltaNet beats Mamba-2 and DeltaNet on in-context retrieval [VERIFIED: https://arxiv.org/abs/2412.06464v3]. There is no read-fusion module.
5. **Compile.** Under `<IR>`, the trunk samples N whole programs in the typed IR (§5.4). Grammar-constrained decoding masks illegal tokens, so every sample parses.
6. **Execute.** Programs run asynchronously in a pre-forked sandbox pool, with a 2 s timeout per call and results cached by program hash. Results are spliced back as tokens. Resuming a GDN after a tool result costs only the new tokens, because the state is constant-size [ESTIMATE].
7. **Verify and select.**
   - Exact checks first: CHECK asserts, self-written tests, SymPy substitution, ARC demo pairs.
   - Then majority vote over executed answers.
   - Then judge energy.
   - The three are combined by a logistic calibrator fit on dev rollouts (data, disclosed).
   - Below threshold, the controller repairs using the error message, re-retrieves, or raises r, within the cap.
8. **Express.**
   - **Multiple choice:** the context state is computed once and **forked per option** (an O(1) state copy, from ORRERY). The judge scores each [context; option] fork. The trunk then writes the winning letter.
   - **Prose:** `<PLAN>` emits an outline (3–8 lines: intent, constraints, evidence chunk ids). Each section is realized under `<REALIZE>`. `<copy>` triggers suffix-automaton extension of an exact span. Executable format checks run. N drafts are ranked by chat reward, and one critique → revise pass with the same weights is allowed.
9. **Emit** with chunk-id citations (which also satisfies BY-SA attribution for copied Wikipedia/StackExchange spans [VERIFIED: https://download.wikipedia.com/legal.html]) and a trace hash.

### 4.3 Why this is not a Transformer, stated honestly

- **Mixer.** There is no softmax attention over token positions, anywhere. Every sequence mixer is a gated delta-rule linear recurrence with a constant-size state of 5 heads × 128 × 128 per layer. There is no KV cache, and memory per token is O(1).
- **Judge.** The judge is a right-to-left recurrence over the trunk's states, not attention.
- **Copying** is a 0-parameter suffix-automaton operation: no pointer, no softmax over positions.
- **Multiple choice** uses state forking, not cross-attention.
- **Retrieval** is a hard, non-differentiable top-K whose results enter only as tokens.
- **The honest caveat:** a GDN block keeps the projection-plus-MLP macro-structure of a Transformer block, and a reviewer may call it a *linear* Transformer. ALEMBIC's claim to novelty does not rest on block anatomy. It rests on system properties: constant-state recurrence, weight-tied test-time depth, externalized human-written memory, program-mediated reasoning with exact verification, and a discriminative judge. Rule F6 is the operational definition, and it is published so a referee can disagree on the record.

---

## 5. Module-by-module specification

### 5.1 Tokenizer (32,768 counted scores)

- Unigram LM, byte fallback, trained on a 2B-token sample with mix 45% English, 35% Korean, 20% code.
- Korean handling:
  - The 2,350 most common Hangul syllables are seeded.
  - About 7k entries go to Korean syllables and morphemes.
  - A rule-based pre-splitter separates common josa/eomi suffixes, from a fixed list, so no learned morphological-analyzer costs enter the count.
  - Jamo splitting is rejected, because it lengthens Korean sequences on a fixed-state recurrence.
- 32,768 fits uint16, so the store is a flat token memmap.
- **Day-1 gate:** measure fertility on kowiki. If Korean needs more than 1.6× English tokens per character, go to 40k vocabulary (+7,232 × 256 = 1.85M params, from headroom) [ESTIMATE].

### 5.2 Factorized tied embedding (8,552,448)

- 32,768 × 256 table plus a 256 → 640 up-projection, transposed for the LM head.
- Embeddings are 10.5% of the budget. For comparison, Gemma-3-270M spends 63% on embeddings [VERIFIED: https://developers.googleblog.com/en/introducing-gemma-3-270m/].
- X1 ablates a 384-d bottleneck (+4.2M from headroom).

### 5.3 Trunk: prelude (3), core (4 tied × r), coda (4)

**GDN layer, per head, with S ∈ ℝ^{128×128}:**
- S_t = α_t · S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ, then o_t = S_t q_t.
- k and q are L2-normalized.
- α_t ∈ (0,1) is a log-space-parameterized decay gate.
- β_t = sigmoid(·).
- The output goes through an output gate and RMSNorm.
- A short depthwise conv (k=4) runs on q, k, v.
- SwiGLU 640 → 1536 → 640.
- Mechanism per GDN [VERIFIED: https://arxiv.org/abs/2412.06464v3]. Our exact gate parameterization follows the NVlabs reference, to be checked line by line against https://github.com/NVlabs/GatedDeltaNet [ESTIMATE].

**State size:** 5 × 128 × 128 = 81,920 floats per layer. In fp32 that is 327 KB per layer, and 11 unique layers × r copies of the core give the per-sequence recurrent memory. At r=8 that is (3 + 32 + 4) × 327 KB ≈ 12.8 MB per sequence. That is what makes 64-way best-of-N batching cheap [ESTIMATE].

**Recurrence definition** (the fix every judge demanded):
- Unrolled, iteration i of the tied core is its own depth-layer with its own time-recurrent state S_{i,t}.
- **r is fixed per query (or per answer segment).** Every state is therefore always defined.
- The injector re-concatenates the prelude output at every iteration: x_i = W[h_{i−1}; h_prelude].
- **Per-position extra compute** comes from learned `<think>` tokens. Each has a fixed input embedding, so the expanded sequence stays chunkwise-parallel in training.
- Changing r between segments (OUROBOROS option B / LIBRARIAN option A re-prefill) is an ablation in X2.

**Training the recurrence:**
- r is sampled per sequence, log-uniform over {1, 2, 4} in S1 (mean ≈ 2) and over {1, …, 16} in S2 (mean ≈ 4).
- TRM-style detached deep supervision: the coda and LM loss apply after iteration ⌈r/2⌉ and after iteration r.
- Gradients flow through only the last 2 iterations plus the injector path to the prelude, so there is no full BPTT [VERIFIED mechanism: https://arxiv.org/abs/2510.04871].
- The budget script counts the full 6N for every iteration, which is conservative.
- Accuracy vs r for r ∈ {1, …, 32} is reported on every row.

**Mode tokens** (0 params, reserved vocabulary): `<PLAN> <READ> <IR> <REALIZE> <JUDGE> <think> <copy> <search> </search> <py> <answer> <cite>`. They replace COMPILER-v1's LoRA adapters.

### 5.4 Program IR and executors (0 params)

A strict typed subset of Python plus tool primitives, decoded under a grammar mask:

```python
ev   = SEARCH(lang="ko+en", q="...", k=8)        # BM25 + dense RRF, returns chunk ids
sp   = READ(ev, ask="...")                       # span + citation (via <copy> suffix match)
x    = SOLVE(sympy_expr) ; g = GRAPH(edges)      # SymPy / NetworkX; SAT(z3) after admission A2
ok   = CHECK(predicate, x)                       # executable assertion
best = JUDGE([c1, c2, c3, c4], evidence=sp)      # judge tower energy ranking
ANSWER(x, cite=sp.cite) | RENDER(plan, style="concise", lang="ko")
```

- **Execution:** whole programs are generated, then executed asynchronously in a pre-forked pool (laptop CPU + Kaggle host CPUs).
- **Caching:** results are cached by program hash.
- **Pure-arithmetic programs** go to a vectorized NumPy evaluator.
- **Limits:** ≤10 tool calls per program, ≤200 executions per query, ≤60 s CPU per query.
- **ARC-AGI:** programs are written in Hodel's DSL, and the demo pairs are an *exact* verifier. This is the cheapest abstraction row: enumerative + trunk-proposed program search, with no separate grid network at M1.

### 5.5 Copy: suffix automaton (0 params; from OUROBOROS)

1. The model emits `<copy>` followed by 1–3 ordinary tokens.
2. The controller looks up that prefix in a suffix automaton built over [retrieved chunks; prompt; tool outputs].
3. It extends the match greedily, scoring each continuation token with the model's **own** vocabulary distribution in one teacher-forced chunkwise pass.
4. It stops at a sentence boundary or when p(token) < τ (τ tuned on dev).

Copied text re-enters the stream as tokens. This gives DROP and NQ exact spans and gives prose verbatim human phrasing, with no gray-zone softmax.

### 5.6 Judge tower (16,353,284 including fusion and norm)

- **Input:** the trunk's per-token final states h_1..h_T, from a forward pass over [evidence; context; `<JUDGE>`; candidate] at the query's r (r=2 in multiple-choice mode).
- **Body:** 3 GDN layers scan h_T → h_1 (right to left). Every judged position thus sees the whole candidate and its context. This is a recurrence, not attention.
- **Fusion:** [h_fwd_last; h_bwd_first] (1280-d), RMSNorm, then 1280 → 1024 → 4 outputs:
  - (a) **energy / verify**: lower is better; used for MC options, programs and traces;
  - (b) **chat reward**;
  - (c) **process value** on IR steps;
  - (d) **difficulty**, which feeds the controller budget.
- **Why one tower, not CORTEX's separate 12M d=384 stack:** the tower reads features produced by the full trunk, so judging draws on about 81M of representational capacity while adding only 16.35M (the win-lens graft).
- **Self-grading:** the generator is prevented from grading itself as correct by training the judge **only** on executed outcomes, teacher rankings, contrastive pairs and benchmark train labels (Open track), never on the generator's own preference.
- **Training:** losses in §6 and §7.
- **Pool J (+1 layer)** is released only if X1 shows the commonsense dev slope rising with judge depth.

### 5.7 Retriever (163,840 counted; everything else is data)

- **Keys and queries:** embedding + prelude L1–L2 + 640→256 projection, mean-pooled, sign() → 256-bit codes. These tensors are **frozen after S1** (from ORRERY), so the 156M-chunk index never goes stale and nothing is double-counted.
- **Fallback:** if X3 shows that freezing costs more than 2% loss, the index is rebuilt after each stage instead. That costs 1.1 TPU-h of encoding plus I/O at 20B tokens.
- **Tuning knobs:** K is adaptive in {4, 8, 16, 32}; up to 3 hops; reciprocal-rank fusion of dense and BM25.
- **Recall gate:** recall@32 on KILT/NQ dev must be within 5 points of exact fp32 search on a 1M-chunk sample. Otherwise move to 512-bit codes (10 GB of RAM). The 92.5% / 96.45% binary-quantization figures were measured on 1024-d vectors [VERIFIED: https://huggingface.co/blog/embedding-quantization] and are **not assumed** to transfer.
- **Alternative arm in X3:** LIBRARIAN's static distilled key table (model2vec-style; potion-base-8M reaches about 92% of MiniLM at 7.56M params [VERIFIED: https://huggingface.co/minishlab/potion-base-8M]). It would cost 8.4M from headroom. It is kept only if it beats the frozen-trunk keys on recall@32 per rebuild-hour.

### 5.8 Controller head (42,194) and controller code (0 params)

- The head outputs an r-level (6 classes), N-level (4) and route (8), trained by regression on which budget solved the training problem cheapest.
- The controller code implements Snell-style compute-optimal allocation [VERIFIED: https://arxiv.org/abs/2408.03314]:
  - easy prompts: r=1–2, N=1;
  - medium: N up to 16;
  - hard verifiable: N up to 64 and r up to 16, under the 2e13 cap.
- Row modes are pinned before eval (F10).

### 5.9 Explicitly out of M1 (scope fence)

These are allowed back only through admission (§10.3):
- Titans/hippocampal test-time memory;
- CORTEX's latent workspace and axial grid scans;
- ORRERY's latent diffusion and energy descent;
- product-key memory (Pool M, 1B+ track);
- Lean 4;
- Wikidata triple store;
- phrase index;
- a JEPA loss;
- a separate grid reasoner;
- fan-out state merge.

---

## 6. Dataset design: "procedures in weights, facts in the store, truth from verifiers"

### 6.1 Store (data; §0.2)

| Source | Role | Licence note |
|---|---|---|
| en + ko Wikipedia, ko-Wiktionary | facts, dictionary lookups (HAE-RAE LW/RW/SN subsets) | CC BY-SA, store-only, cite on copy [VERIFIED: https://download.wikipedia.com/legal.html] |
| FineWeb-Edu-dedup, ≤2022 crawls | general explanations | ODC-BY [VERIFIED: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus] |
| FineWeb-2 Korean, ≤2022 snapshots | Korean breadth | ODC-By [VERIFIED: https://arxiv.org/abs/2506.20920v1] |
| OpenWebMath / arXiv (Proof-Pile-2 components) | math explanations | licence to verify per component [ESTIMATE] |
| Python-Edu / Stack-v2 permissive | API usage, code idioms | per-file permissive [VERIFIED: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus] |
| StackExchange-style human Q&A, open textbooks (CC BY) | human exemplar answers for chat; exam-style explanations for MMLU/ARC-C | licences to verify on day 1 [ESTIMATE] |

**Not in the gated store:** Cosmopedia, any teacher output, WikiHow, ActivityNet, Korean exam sites, benchmark mirrors (F3, F9).

### 6.2 Training data by what it teaches

| Bucket | Share of S1+S2 gradient | Sources |
|---|---|---|
| Language fluency + concept skeleton (en/ko/code) | 30% | FineWeb-Edu (int_score ≥ 4), FineWeb-2-ko, kowiki, OpenLab corpus-ko-23m (Apache-2.0 [VERIFIED: https://huggingface.co/datasets/OpenLab-NLP/corpus-ko-23m/blob/main/README.md]), Cosmopedia **as training text only**, Python-Edu, FineMath-4+ |
| Retrieval-reading | 20% | 30% of S1 documents with in-stream neighbours; KILT provenance triples [VERIFIED: https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/]; NQ/TQA train (Open) |
| Procedural, verifiable | 30% | Reasoning Gym, 100+ generators [VERIFIED: https://arxiv.org/pdf/2505.24760], with 5 BBH families held out; re-ARC train split [VERIFIED: https://arxiv.org/pdf/2404.07353]; own SymPy/graph/date/unit/SAT generators; program → exact-output traces |
| Open-teacher distilled, execution-filtered | 15% | OpenMathInstruct-2 (CC-BY-4.0, Llama-3.1-405B solutions [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2]), rewritten into IR and kept **only if execution reproduces the answer**; code with tests |
| Self-play (grows after week 12) | 5% → 50% | §6.5 |

### 6.3 Long-tail fact dropout (from LIBRARIAN) [HYPOTHESIS]

- In S1 retrieval-augmented documents, tokens inside entity spans whose corpus frequency is below θ get **zero LM loss unless the span appears in the in-stream neighbours**, in which case they become a copy target.
- Frequent entities are trained normally (the concept skeleton).
- θ is ablated in X3 at {1e-7, 1e-6, 1e-5} relative frequency [ESTIMATE grid].

### 6.4 Judge data (S3): the main commonsense lever

1. **Teacher-ranking distillation.**
   - **1.5M** synthetic multiple-choice items, cut from COMPILER's 20–50M per the feasibility lens.
   - Each is a context from human text (FineWeb-Edu, Wikipedia, StackExchange; WikiHow/ActivityNet blocked) with 1 true continuation and 3 corruptions (shuffled steps, swapped entities, negated outcomes, wrong pronoun referent).
   - An Apache-licensed open teacher on T4 scores all options.
   - The judge learns the teacher's **ranking distribution** (listwise KL), never its text.
   - Cost: ~0.9B prefill tokens (§7.3).
2. **Self-made contrastive pairs:** 20M, CPU-generated, free.
3. **Benchmark train splits** (Open track only): HellaSwag-train, WinoGrande-XL, ARC-train, KoBEST-train.
4. **Verification labels** from S2/S5 rollouts: trace → executed correct/incorrect.
5. **Chat preference pairs** labelled by the family-A training judge (§6.6, S6).

### 6.5 Self-play flywheel

1. A proposer prompt (`<PLAN>` + "write a new problem") generates problems near the frontier (50–70% solve rate).
2. The solver samples N IR programs.
3. **Only problems and solutions accepted by an executor are kept**, as in STaR / expert iteration [VERIFIED: https://arxiv.org/html/2203.14465].
4. Verified IR fragments that recur are refactored into named functions and added to the **skill library** (text).
5. The library is frozen, hashed and decontaminated before every eval. It is retrieved, never inlined wholesale.

The data supply then grows with compute, not money.

### 6.6 Korean

- **Store:** kowiki, ko-Wiktionary, FineWeb-2-ko (≤2022).
- **Training:** FineWeb-2-ko, corpus-ko-23m, translated *templates* of the procedural generators (translate procedures, not facts), and Korean instruction data regenerated with an open teacher, filtered by back-translation consistency.
- **Cross-lingual retrieval curriculum:** Korean questions retrieve from both the Korean and English stores.
- **Licence caution:** 35.3% of Korean HF datasets have unknown licences [VERIFIED: https://arxiv.org/pdf/2507.04329]. KORMo's report maps open Korean sources [VERIFIED: https://arxiv.org/pdf/2510.09426]. AI Hub data is eval-only or skipped, never redistributed. KoAlpaca's provenance is checked before use (possibly ChatGPT-made [HYPOTHESIS]).
- **Decontamination:** character 20-grams, exam-site blocklist, exam-format filter.

### 6.7 Contamination protocol

The protocol is F9, applied to everything above, including every synthetic and self-play set. Tooling:
- lm-eval-harness decontamination;
- Tulu-3 decontamination tools [VERIFIED: https://arxiv.org/pdf/2411.15124];
- the OpenMathInstruct-2 contamination explorer;
- longest-contaminated-substring checks [VERIFIED: https://arxiv.org/pdf/2411.03923v1].

The overlap report is published.

---

## 7. Training recipe and compute plan

### 7.1 Stages (tokens and FLOPs from §0.3)

| Stage | Weeks | Tokens | Data | Objectives |
|---|---|---|---|---|
| S0 Build | 1–6 (CPU/laptop) | — | Tokenizer, store, BM25 core, decontamination, sandbox image, eval harness | — |
| S1 Language + reading | 7–8 | **30B** scanned | §6.2 fluency + retrieval buckets; 30% of documents with in-stream neighbours (neighbour IDs precomputed on CPU, joined on the fly from a uint16 memmap, from LIBRARIAN); **top-16 logit distillation** on a 0.3B-token subset | Next-token; long-tail fact dropout; copy targets; r ~ {1,2,4}; contrastive retrieval loss on the L2 pooled projection (query vs KILT gold passage, in-batch + BM25 hard negatives); **freeze key encoder → build index (1.1 TPU-h)** |
| S2 Procedures + IR | 9–10 | **15B** | Procedural generators, execution-filtered OMI-2 IR, code with tests, tool traces with `<think>`/`<copy>`/`<search>`, Korean procedure templates, Open-track train splits | Next-token on calls and result use (tool outputs masked from loss); deep supervision; r ~ {1..16}, mean 4 |
| S3 Judge | 10 | **3B** | §6.4 | Listwise KL to teacher rankings; InfoNCE over options; BCE on executed-outcome labels; regression of difficulty. Trunk frozen, run forward at r=2 |
| S4 Instruction / chat SFT | 11 | **2B** | Audited SmolTalk (Apache-2.0 [VERIFIED: https://www.huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/blob/main/README.md]) and Tulu-3 minus NC, rewritten into plan → section → copy-realize traces with executable format checks; teacher-regenerated en/ko instructions | Next-token |
| S5 RLVR | 11–12 + ongoing | **2B rollout + 1B train** | Math, code (self-written tests + training-test pool), executable BBH-like tasks, retrieval QA EM, ARC train | GRPO [VERIFIED method: https://arxiv.org/html/2501.12948]; rewards: correctness, fewer `<think>`, lower r; judge regressed on outcomes |
| S6 Preference | 12 | **0.3B pairs** | Our samples on open prompts (never MT-Bench/AlpacaEval/Arena-Hard prompts), labelled by the family-A judge | DPO; chat-reward head training |

**Rollout placement** (feasibility graft):
- RL rollouts run on the T4×2 and the 3060, with an asynchronous CPU tool pool.
- Policy updates run on the TPU, at most one update stale.
- The sandbox is **never** inside the jitted TPU loop.

### 7.2 Compute envelope (all quotas re-verified in-account on day 1)

**Quotas:**
- TPU v5e-8 replaces v3-8 on Kaggle [VERIFIED: https://www.kaggle.com/discussions/product-announcements/607202]. About 20 h/week with 9 h sessions; third-party figure [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab].
- T4×2: about 30 h/week, 12 h sessions; third-party [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits].

**Peaks:**
- v5e: 197 TFLOPS bf16 per chip, 1,576 for 8 chips [VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e].
- T4: 65 TFLOPS fp16, no bf16 [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf].

**MFU:**
- Planning value 12% on TPU. The JAX tutorial measured 16–27% for a tiny GPT and 13% at 8-way data parallelism [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html].
- Planning value 20% on T4.

**Throughput:**
- S1 runs at about **26.6B tokens per TPU-week** at r̄=2, so 30B is about 1.1 TPU-weeks (22.6 h).
- Main stages total 44.4 TPU-h. Experiments total 26.6 TPU-h.
- With the ×1.8 multiplier: **127.8 TPU-h = 6.4 TPU-weeks of the ~12 available in the window.**
- The 120B-token token budget is not the bottleneck. 52B training tokens for 81M params is about 640 tokens/param, about 32× Chinchilla [ESTIMATE]. Data density, CPU-bound tools and evaluation are the bottlenecks.

**TPU unavailable** (v5e is sometimes oversubscribed [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab]):
- S1 drops to 12B and S2 to 8B.
- About 4.8e19 FLOP with the multiplier, roughly **17 T4-weeks**.
- R1 slips from week ~20 to week ~30, and teacher generation moves to the 3060.
- **Apply to TPU Research Cloud in week 1** [VERIFIED: https://sites.research.google/trc/faq/]. It is free TPU quota, but VM/GCS costs are "generally minimal", not zero.

### 7.3 Teacher budget (T4×2, planned at measured throughput; vllm-tpu is upside, not plan)

- **Decode throughput:** projected at about 0.1–0.6B tokens/week on T4×2 for a 1.5B fp16 teacher [ESTIMATE, compute digest]. Planning value is 0.3B/week.
- **Prefill** is compute-bound: about 2·1.5e9 FLOP/token.

| Job | Tokens | Kind | T4×2-weeks |
|---|---|---|---|
| OMI-2 → IR rewrite (1M items × 300) | 0.3B | decode | 1.0 |
| Chat/instruction regeneration (en/ko) | 0.2B | decode | 0.7 |
| Korean procedures/instructions | 0.1B | decode | 0.3 |
| Judge ranking: 1.5M items × 4 options × 150 | 0.9B | prefill | 0.35 |
| Top-16 logit distillation subset | 0.3B | prefill | 0.15 |
| **Total** | | | **≈ 2.5, budgeted 4.5 (≈ 135 h) for retries** |

- **Logit storage:** 0.3B tokens × 16 × (uint16 id + fp16) = 19.2 GB, kept on a Kaggle dataset.
- **Teacher licence:** checked per model card for permission to train on outputs. Teacher family A is never an eval judge.

### 7.4 One codebase; T4 engineering; Kaggle operations

**Single training codebase: JAX.**
- The GDN delta rule is pure `jnp`: chunkwise WY/UT form with C=128 inside `lax.scan` over chunks, fp32 state and accumulators.
- No Triton and no Pallas-GPU. Triton 3.3+ targets compute capability 8.0+, so Turing T4s are excluded [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md].
- JAX supports SM 7.5+ under CUDA 13 [VERIFIED: https://docs.jax.dev/en/latest/installation.html], which covers the T4 (sm_75) and the 3060 (sm_86). **Day-1 gate:** confirm by running the parity test on both.
- **If JAX-CUDA fails on the T4:** T4s run only vLLM teacher generation, BM25/index jobs and evals in an eager-PyTorch *inference-only* port, with a numerical-parity test against JAX on every checkpoint. There is never a second training codebase.
- **Throwaway harness:** the backbone challenger test (GDN vs Mamba-2 [VERIFIED: https://arxiv.org/abs/2405.21060v1]) runs on the 3060 with existing library kernels and is thrown away. Only the winner exists in JAX.

**Recurrence and shapes:**
- One XLA `while_loop` with static r_max = 32 and masking, so there are no per-r recompiles.
- Shape buckets for sequence length {512, 1024, 2048, 4096} and K ∈ {0, 4, 8, 16, 32}.
- The compile cache lives in `/kaggle/working`.

**T4 numerics (no bf16):**
- fp16 matmuls with dynamic loss scaling.
- Delta-rule state, log-space decay cumulative products and scan accumulators in fp32. Decay gates are clamped.
- Gradient clipping per core iteration.
- Activation checkpointing per core iteration, so memory stays flat in r.

**Multi-GPU:**
- Data-parallel over PCIe on T4×2; no tensor parallelism.
- On v5e-8, the week-1 benchmark compares a 4-way data × 2-way model mesh against 8-way data parallelism.

**Memory:** 81M params × 16 B (fp32 master + Adam moments + grads) ≈ 1.3 GB. It fits a 6 GB 3060 for ablations, evals and rollouts.

**Kaggle operations:**
- The first cell asserts the device type and quota. `kaggle kernels push` can silently land on a non-TPU image [VERIFIED: https://github.com/Kaggle/kaggle-cli/issues/1197].
- Training runs as committed "Save & Run All" sessions.
- Checkpoint every 45 minutes to `/kaggle/working`, then push to the HF Hub (100 GB private [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits]).
- Sessions are chained from the last checkpoint.
- Static data (store subset, codes, logits) is mounted as private Kaggle datasets.
- One account only.

### 7.5 Evaluation operations

- **Weekly:** a stratified 10% dev panel on the 3060.
- **Monthly:** full Tier A + B, about 15–30 T4-h plus CPU tool time. A full MATH run alone is about 1M tool calls, 5–14 h [ESTIMATE].
- **ARC-AGI-1:** a separate monthly job on the laptop, 400 tasks at 2–10 minutes each.
- **Seeds:** 3 for sampled rows; deterministic rows run once.

---

## 8. Benchmark-by-benchmark attack plan

Probabilities are my subjective P(clearing the gate at M1, month 12) on the Open track [ESTIMATE]. Strict-track numbers are reported alongside.

### 8.1 Likely or coin-flip rows

- **ARC-AGI-1 (gate 13.6, est. 10–25, P 0.55).**
  - Method: the trunk proposes DSL programs, enumerative search fills them in, and demo pairs verify exactly; the judge ranks surviving programs; re-ARC-train self-play adds data.
  - Evidence that small systems can do this: TRM 44.6 at 7M [VERIFIED: https://arxiv.org/abs/2510.04871] and CompressARC about 20% at 76K params [VERIFIED: https://arcprize.org/blog/arc-prize-2025-results-analysis].
  - Risk: TRM's ARC run cost about 4 H100 × 3 days, and HRM's claimed 40.3 was verified at 32 [VERIFIED: https://arcprize.org/blog/hrm-analysis]. The estimate is not a headline.
- **NQ-open (gate 40.2, est. 38–48, P 0.55).**
  - Method: retrieve with K up to 32, read in-stream, suffix-copy the span, NQ train on the Open track.
  - Anchor: FiD-base reached 48.2 with a 220M reader and 100 passages [VERIFIED: https://arxiv.org/abs/2007.01282]. Our reader is smaller, so recall@32 is the lever (X3 gate).
- **DROP (gate 65.1, est. 56–70, P 0.40).** Spans are copied exactly and arithmetic runs in Python; no retrieval is needed.
- **KMMLU (gate 43.0, est. 38–48, P 0.35).**
  - GPT-3.5 is weak here (chance is 25).
  - Method: per-option Korean and English retrieval, judge elimination, STEM items routed to SymPy.
  - Blocking Korean exam sites costs honest coverage.
- **BBH (gate 69.4, est. 55–70, P 0.30).** About half the tasks are exactly executable; the language-heavy half goes to the judge. The 5 held-out families are reported separately.
- **MATH (gate 35.4, est. 22–36, P 0.25).**
  - Method: SymPy, brute-force small-case enumeration, process-value search.
  - Anchors: Qwen2.5-0.5B-Instruct scores 34.4 [VERIFIED: https://qwenlm.github.io/blog/qwen2.5-llm/]; the best ≤150M model scores 4.6 on MATH500 [VERIFIED: https://arxiv.org/html/2509.24945v3].
- **HAE-RAE (gate 57.9, est. 46–58, P 0.20).** Lexical subsets become dictionary lookups; history and general knowledge depend on store coverage.

### 8.2 Long shots

- **GSM8K (gate 80.3, est. 60–76, P 0.15).**
  - Method: IR + Python + maj@32 + judge verify.
  - TinyGSM's 63.1 (125M) and 76.1 (with a 1.3B verifier) used banned GPT-3.5 data [VERIFIED: https://arxiv.org/html/2312.09241]. We rebuild from execution-filtered OMI-2 plus generators.
  - X4 tells us by week 6 whether the slope gets there.
- **KoBEST (gate 80.8, est. 62–76, P 0.10).** COPA and HellaSwag inherit the commonsense wall.
- **TriviaQA (gate 86.4, est. 65–78, P 0.05).**
  - GPT-3.5 is strongest here. Llama-2-7B with a 1.4T store reaches only 77.0 [VERIFIED: https://arxiv.org/html/2407.12854].
  - The levers are store growth and multi-hop search. Wikidata linking is admission candidate A4.

### 8.3 The four bosses: honest walls and precise plans

**Boss 1. Open-ended chat and writing (pooled gate 52.6%; est. 10–28% wins; P 0.02).**

- **Why current technology can't do it.**
  - Every model of about 135M or fewer scores 1.98–2.52 on MT-Bench [VERIFIED: https://arxiv.org/html/2402.14905]. The smallest Arena model above GPT-3.5 is about 2.6B (Gemma-2-2B-it) [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard].
  - Chat needs knowledge, fluency and reasoning simultaneously at every token, judged without an exact verifier.
  - Arithmetic on MT-Bench [ESTIMATE]: even a perfect 9 on math, code, reasoning and extraction turns, plus about 3 on writing, roleplay and humanities, averages about 6.
- **The attack, in order of evidence:**
  1. Top-k logit distillation from an open teacher on the generator. The Qwen3 strong-to-weak recipe is the best-evidenced sub-1B fluency lever [VERIFIED: https://arxiv.org/html/2505.09388].
  2. Shrink the generator's unit of work: plan → 2–4 sentence sections → copy-realize from **human-written** exemplar answers retrieved from the store. Fluency is borrowed; it is not generated. TinyStories showed <10M models are fluent *in a narrow register* [VERIFIED: https://arxiv.org/abs/2305.07759]. Sections are a narrow register [HYPOTHESIS].
  3. Executable format checks, the IFEval-like part, which is the cheapest win inside chat.
  4. Best-of-8 under the chat-reward head, plus one critique → revise pass.
  5. Math, code, reasoning and extraction turns route through the compiler.
  6. Pool G (+2 coda layers) if X1 shows a positive slope.
- **Kill test X1 (weeks 2–5):** detailed in §10.2.
- **Pre-registered escalation:** at week 8, if the 5→20M slope extrapolates below the gate at 100M, I tell the user plainly that this row is not reachable at 0.1B by any design known to me, attach the curve, and propose the 1B track. Under F13, the row stays open and unclaimed.

**Boss 2. HellaSwag and WinoGrande (gates 86.2 / 83.7; est. 48–68 / 56–70; P 0.03 / 0.02).**

- **Why it is hard.**
  - There is no retrieval key.
  - WinoGrande sits at chance (51–53) at every size up to 270M [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M].
  - The 2019 discriminative SOTA ceiling was 59.4–79.1 depending on training size [VERIFIED: https://arxiv.org/abs/1907.10641], below our 83.7 gate.
  - Synthetic textbook data moved phi-1.5 from 60.4 to 73.4, but at 1.3B [VERIFIED: https://arxiv.org/html/2309.05463].
- **The attack:**
  - Discrimination, not generation: per-option state forks, each scored by the judge tower reading the full trunk.
  - Teacher-ranking distillation on 1.5M human-text items, 20M self-made minimal pairs, and the Open-track train splits.
  - For WinoGrande specifically, **multi-hypothesis verification**: substitute each option into the sentence and score the whole sentence by energy.
- **Kill test:** in X1 (§10.2). If the judge does not beat chance clearly at 20M, these rows are published as parameter-bound, with the slope.
- These are the rows most likely to depend on train splits. The Strict track exposes that.

**Boss 3. Korean (chat gate; KoBEST; HAE-RAE).**

- Data is scarce and licences are murky.
- Arena-Korean for GPT-3.5 is 1049.6, which is lower than its overall rating, so Korean is a relatively weak GPT-3.5 skill.
- Plan: Korean is 35% of the tokenizer mix and about 25% of fluency tokens; cross-lingual retrieval; procedure templates translated, not facts; a 30% Korean share in S6 preference prompts; volunteer raters for the gate.
- Kill test: Korean has its own pairwise slope in X1. The fertility gate is on day 1.

**Boss 4. MMLU and ARC-Challenge, knowledge plus reasoning (gates 72.1 / 89.3; est. 42–55 / 50–68; P 0.02 / 0.03).**

- **Why it is hard.** Retrieval adds only about +3.5 on MMLU even at 7B with a 1.4T store [VERIFIED: https://arxiv.org/html/2407.12854]. Atlas-770M scores **38.9 at 5-shot** (56.3 is the full/transfer setting) [VERIFIED: https://arxiv.org/abs/2208.03299].
- **The attack:** an exam-explanation shard (human open textbooks), per-option evidence retrieval, judge elimination, STEM routed to SymPy, and the ARC train split on the Open track.
- **Stated openly:** P < 0.05 at M1. The store-composition slope (X3, X5) decides whether "more store" can ever carry these rows inside the cap.

**Code (HumanEval+/MBPP+, P 0.03).**

- Test-time compute helps only when base pass@k is non-trivial [VERIFIED: https://arxiv.org/abs/2408.03314], and EvalPlus's GPT-3.5 numbers are strong.
- phi-1-small (350M) reached 45 on HumanEval [VERIFIED: https://arxiv.org/abs/2306.11644].
- Method:
  - 16 candidates;
  - model-written tests, plus tests extracted from docstring examples;
  - behavioural clustering on model-generated inputs;
  - up to 4 repair rounds;
  - API-documentation retrieval;
  - code-only RLVR.
- The honest code row is the held-out Natural2Code-style fresh set (Tier C).

### 8.4 Verdict

- Expected at M1: **3–6 green rows** (ARC-AGI-1, NQ, DROP, KMMLU, possibly BBH/MATH).
- **P(all gated rows) < 0.1%.**
- What ALEMBIC changes is that every red row has a mechanism, a knob that adds no parameters, and a slope measured by week 6–12 that can falsify its forecast.
- The user's "losing is not acceptable" is honoured the only honest way: no loss is ever reported as a win, and no wall is hidden past week 8.

---

## 9. Scaling plan: 0.1B → 1B → 10B → north star

### 9.1 Axes

Only the first two need retraining.

| Axis | M1 (0.1B) | 1B | 10B | Evidence for the axis |
|---|---|---|---|---|
| Trunk params | 81.1M (+Pools ≤ 96.3M) | d=1536 trunk | d≈4096 | RWKV-7 improves monotonically from 0.19 to 2.9B [VERIFIED: https://arxiv.org/abs/2503.14456] |
| Judge share | 16.35M (20%) | 150–250M (a fixed ~15–25% share) | ~1.5–2.5B | TinyGSM: verifier size beats generator size [VERIFIED: https://arxiv.org/html/2312.09241] |
| Store (human-authored) | 20B tokens | 0.3–1T | multi-T + science shards (after review) | RETRO gains do not diminish from 150M to 7B [VERIFIED: https://arxiv.org/abs/2112.04426]; MassiveDS shows no saturation up to 1.4T [VERIFIED: https://arxiv.org/html/2407.12854] |
| Skill library (verified programs) | 10³–10⁴ functions | 10⁵ | 10⁶ | [HYPOTHESIS]: carries over to bigger cores unchanged |
| Test-time depth r and `<think>` | r ≤ 32 | ≤ 64 | ≤ 128 | Huginn [VERIFIED: https://arxiv.org/abs/2502.05171] |
| Samples × verifier N | ≤ 64 | 256 | 1,024+ | 0.5B + search > GPT-4o on MATH-500 [VERIFIED: https://arxiv.org/abs/2502.06703] |
| Tools | Python, SymPy, NetworkX | + Z3, Lean, geometry engine | + simulators, docking, MD (gated) | AlphaGeometry 25/30 [VERIFIED: https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/]; AlphaProof silver [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/] |
| Sparse memory (Pool M) | 0 | product-key memory ~300M, if it beats a matched-FLOPs control | ~6B | Memory layers beat dense models with >2× the compute [VERIFIED: https://arxiv.org/abs/2412.09764] |
| Verified self-play data | 5–50% | majority | dominant | R1-Zero: AIME 15.6 → 77.9 from RL [VERIFIED: https://arxiv.org/html/2501.12948] |

### 9.2 Laws to measure (X5, before S1 finishes)

- **Grid:** {5, 10, 20, 40}M trunks at 100 tokens/param. Seeds: 3 at ≤10M, 2 at 20M, 1 at 40M.
- **Inference knobs swept free on every checkpoint:** r ∈ {1, 4, 16}, N ∈ {1, 16}, store ∈ {1B, 5B, 20B}.
- **Fitted metric:** smooth retrieval-conditioned NLL on held-out post-cutoff text, plus answer NLL per row family (R = retrieval, T = tool, Q = reasoning, F = fluency). Noisy accuracies are not used for the fit.
- **Pre-registration:** the 81M prediction for each family is published *before* S1 completes. The 1B prediction is published before any 1B run.
- **Going forward** requires all of the following:
  - each axis is monotone with fit residual < 2 points;
  - the 81M point lands inside its predicted interval;
  - the store gap (on vs off) does not shrink with size;
  - r does not saturate by r=4.
- **If any fails,** the axis is declared non-scaling *for that family*, in public, and its parameters are reassigned.
- **Limit of the claim:** a 5–40M fit can inform the 0.1B → 1B step at best. **No claim is made about 10B from a sub-100M fit.**

### 9.3 How we know the design scales *better than an LLM*

- **Control arm:** ORRERY-0, a plain GDN LM of equal params with retrieval + tools + best-of-N.
- **Test:** compare slopes against log(training + inference FLOPs) on the X5 grid.
- **Admission:** every exotic module enters only if it beats the control by ≥2 points on the frozen dev suite at equal FLOPs (§10.3).
- **Pivot:** if ALEMBIC's slope is not better than the control's, the 1B tier is built as the control, and we say so.

### 9.4 Compute wall

- 1B at Chinchilla scale is about 1.2e20 FLOP, about 9 TPU-weeks at 12% MFU: tight but feasible on free quota.
- 10B is about 1.2e22, roughly 7 years of free quota [ESTIMATE, compute digest].
- 10B therefore needs TRC or grants, earned by publishing the X5 law and the M1 scoreboard openly.
- Meanwhile store, library, tools, test-time N/r and self-play keep compounding with no retraining.

### 9.5 North star and biosafety

**The compiler mapping** (this is why COMPILER is the spine):

| Discovery concept | ALEMBIC component |
|---|---|
| Hypothesis | Program |
| Simulator / assay | Executor |
| Measurement | Verifier |
| Lab notebook | Store |
| Validated protocols | Skill library |
| Scientific taste | Judge |

Only the tool registry changes as the system moves into science.

**Seed of a scientific world model** (admission candidate A5, from ORRERY): a small head that predicts a tool or experiment's outcome before execution, with prediction error ("surprise") triggering re-planning. Later, this is how the system decides which expensive experiment is worth running [HYPOTHESIS].

**Biosafety: hard constraints, never tuned.**
1. **M1 and every mid-boss round ship no bio/chem shards and no bio/chem tools.**
2. Adding any of them requires an **external** dual-use review, not self-review.
3. When added:
   - the store and tool registry exclude pathogen-enhancement, toxin-synthesis and agent-weaponization content, using a blocklist plus a classifier audit;
   - the planner carries a **non-learned** refusal blocklist for hazardous chemistry and biology. Refusal is a rule, not an energy trade-off;
   - every wet-lab-actionable output requires review by a qualified human expert;
   - bio data and models are released only after dual-use review.
4. Program traces are the primary audit surface. Any admitted latent reasoning module must ship with a thought-probe decoder before it touches a bio task.
5. Any >100M structure predictor is a disclosed north-star-track tool, never part of the milestone.

---

## 10. 12-week roadmap, first experiment, risks, rejected alternatives

### 10.1 Roadmap

| Week | TPU v5e-8 (20 h) | T4×2 (30 h) | 3060 / laptop CPU | Milestone |
|---|---|---|---|---|
| 1 | **X0** kernel MFU benchmark; device asserts | X0 parity on sm_75; vLLM teacher throughput | X0 kernel + parity + param CI; quotas, `df -h`, `free -g`, `nproc`; tokenizer fertility; **TRC application** | **G0: toolchain gate** |
| 2–3 | X1 20M runs; X5 grid starts | Teacher: judge-ranking prefill; chat regeneration | X1 5M/10M; X2 recurrence grid; challenger harness (GDN vs Mamba-2); S0 store build (CPU) | — |
| 4–5 | X3 retrieval 50M (4 arms); X5 grid | Teacher: OMI-2 → IR; index build on 1B shard | X1 judge/chat evals; X4 compile-vs-CoT 30M (TPU share); decontamination | **G1 (wk 5): wall slopes read** |
| 6 | X4 finish; X5 fit → **publish 81M predictions** | Teacher: Korean; logit subset | Full store + BM25 core build | **G2: admission review, architecture freeze** |
| 7–8 | **S1 30B** (22.6 h + slack) → freeze keys → index (1.1 h) | Eval of S1 checkpoints; RL rollout infra | Weekly dev panel | **Week 8: pre-registered escalation report to the user** |
| 9–10 | **S2 15B** (16.8 h); **S3** (1.2 h) | Rollouts; judge data top-up | Dev panel; ARC DSL search harness | — |
| 11 | **S4** (1.5 h); **S5** GRPO updates (1.9 h) | GRPO rollouts | Rollouts; tool pool | — |
| 12 | **S6** (0.5 h); 81M vs prediction check | **R0 scoreboard**: full Tier A + B on one checkpoint | ARC-AGI-1 monthly job | **R0 published, losses included** |

- After week 12: self-play S7, admissions A1–A5, store growth.
- **R1** at about week 20; **M1** at month 12. Each round publishes everything.

### 10.2 The first experiments, specified to start coding tomorrow

**X0, "the kernel gate" (week 1). This is literally tomorrow's work.**

Repository layout, all JAX:
```
alembic/
  config.py          # dataclass: d=640, H=5, dh=128, ffn=1536, V=32768, e=256, n_pre=3, n_core=4, n_coda=4, n_judge=3, chunk=128
  kernels/gdn.py     # gdn_recurrent_ref(q,k,v,alpha,beta,S0) and gdn_chunkwise(q,k,v,alpha,beta,S0,C=128)
  model/trunk.py     # GDNLayer, Trunk(prelude, core×r with injector, coda), forward(tokens, r) via lax.while_loop(r_max=32, mask)
  model/judge.py     # reverse-scan GDN tower + fusion MLP (4 heads)
  tests/test_gdn_parity.py
  tests/test_param_count.py   # asserts == 81_108_694 and < 100e6 (re-derived from calc/alembic_budget.py)
  bench/mfu.py
```

Specification:
- **Shapes:** q, k, v ∈ [B, T, H=5, 128]; α, β ∈ [B, T, H]; state S ∈ [B, H, 128, 128], fp32.
- **Reference:** a sequential `lax.scan` over T implementing S_t = α_t S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ and o_t = S_t q_t.
- **Chunkwise version:** C=128. The within-chunk delta updates are written as the WY/UT dense-matmul form, with the cumulative decay products in log space and fp32, and the state is carried across chunks by `lax.scan`. The exact block form is to be matched against the NVlabs reference implementation [ESTIMATE].
- **Parity test:** at B=2, T=1024, random inputs, max |o_chunk − o_ref| < 1e-3 (fp32) and < 2e-2 (bf16 on TPU / fp16 on T4), and final-state relative error < 1e-3.
- **Gradient test:** chunkwise vs reference grads through `jax.grad`, relative error < 1e-2.
- **MFU benchmark:** full 81M trunk forward + backward at r=2, sequence length 2048, per-device batch swept over {8, 16, 32}; meshes 8×1 vs 4×2 on v5e-8; data parallelism on T4×2; also the 3060.
- **Gates:**
  - TPU MFU ≥ 12% (the planning value) → proceed. 8–12% → re-plan S1 down to 20B. < 8% → T4-first plan.
  - JAX-CUDA passes parity on sm_75 → T4s train. Otherwise T4s are generation/eval only (§7.4).
  - The parameter-count test is green.

**X1, "the wall probe" (weeks 2–5). The pre-registered kill test for bosses 1–3.**

- **Models:** 5M, 10M and 20M trunks with the same shape family (the d and layer counts come from the budget script's width/depth rules), each with a proportional judge tower.
- **Trunk training:** a 2B-token S1-mini (fluency + 10% retrieval-reading) plus a 0.5B S4-mini SFT (plan → section → copy-realize traces), with and without top-16 logit distillation.
- **Commonsense arm:** judge trained on 300k teacher-ranked items + 5M self-made pairs (+ train splits on Open). Evaluated *generatively* on HellaSwag-val, WinoGrande-dev and KoBEST-COPA-dev, store off, on both Open and Strict tracks.
- **Chat arm:** pooled pairwise against stored GPT-3.5 answers on AlpacaEval's 805 prompts (CI ≈ ±3.5) plus 160 MT-Bench turns. The panel must pass ρ ≥ 0.8 calibration first. Separately, 80 writing, roleplay and humanities prompts compare the plan+copy-realize pipeline against plain sampling at 20M, and Korean prompts are run with volunteer raters (a small n is acceptable for slope reading).

| Outcome | Condition | Action |
|---|---|---|
| **GO** | At 20M: HellaSwag-val ≥ 45, WinoGrande-dev ≥ 56 (Open), slope ≥ 3 points per doubling; chat win rate rising ≥ 5 points per doubling of params *or* r; pipeline ≥ plain + 10 points; logit distillation ≥ +3 points | Release Pools G/J at S1/S3 |
| **PIVOT (once)** | Slope 1–3 points per doubling | Rebalance judge data (more ranking distillation, fewer self-made pairs) and re-run |
| **KILL** | Flat (< 1 point per doubling) | The row is declared parameter-bound at 0.1B in the week-8 report, with the curve; effort shifts to the free axes; the 1B decision goes to the user |

**X2, recurrence (weeks 2–4, 3060).**
- 20M trunks trained at r ∈ {1, 4, 16}, plus a `<think>` arm at r=4. Evaluated on Reasoning Gym held-out generators (harder levels) and on train/test consistency of per-query r.
- **GO:** r=16 beats r=1 by ≥ 8 points; the per-query-r model is within 2 points of fixed-r training at every r.
- **KILL:** < 3 points beyond r=4. Then r_max = 4, and the core's tied capacity goes into width and the coda.

**X3, retrieval (weeks 4–6).**
- 50M model. Arms: closed-book; in-stream K=8; in-stream adaptive K + hops; LIBRARIAN-style static keys. Also frozen vs rebuilt keys, and the θ ablation.
- Metrics: NQ-open dev, recall@32, a synthetic key→value recall probe at 4k tokens, the leakage audit.
- **GO:** NQ ≥ 30 EM at 50M; recall@32 ≥ 70%; RA beats closed-book by ≥ 15 EM; positive slope across store {1B, 5B, 20B}.
- **KILL:** recall@32 < 60% → 512-bit codes and BM25-heavy fusion. NQ < 20 → the reader is the bottleneck, so the fan-out merge (A1) is promoted.

**X4, does compiling beat memorizing (weeks 4–6).**
- 30M IR model vs a 30M chain-of-thought twin with the same data (OMI-2 execution-filtered + generators; no TinyGSM), 8B tokens each. Evaluated on a decontaminated GSM8K-like dev split.
- **GO:** IR maj@16 ≥ 40% and ≥ 2× the CoT twin; r=8 beats r=1 by ≥ 5 points.
- **KILL:** IR < 20%. The data supply is then the bottleneck: redirect T4 time to verified PoT.

**X5, scaling grid (weeks 2–6).** §9.2.

### 10.3 Admission queue (after G2; each needs ≥ +2 points on the frozen dev suite at equal FLOPs vs the current system)

| ID | Candidate | Cost |
|---|---|---|
| A1 | Fan-out READ with sigmoid-gated state merge (LIBRARIAN) | ~3.2k params |
| A2 | Z3 in the sandbox | 0 params |
| A3 | Human-written phrase index with hard nearest-neighbour lookup (CORTEX) | Pool R, 163,840 |
| A4 | Wikidata triples + alias-dictionary entity linking | 0 params; licence verified first [ESTIMATE that it is CC0] |
| A5 | Tool-outcome prediction head | ≤ 0.5M |
| A6 | Test-time fine-tuning for ARC | 0 params counted; disclosed wall-clock |

Rejected candidates are logged with their measured deltas.

### 10.4 Top risks and kill criteria

| # | Risk | Detection | Response |
|---|---|---|---|
| 1 | Chat and commonsense walls are parameter-bound | X1 slopes (week 5) | Week-8 escalation to the user; Pools G/J; 1B track decision is the user's |
| 2 | Autoformalization ceiling (language → IR fails) | X4 | Grammar masks, repair loops, library; if IR < 20%, rebuild data |
| 3 | Contamination fakes a win; retrieval systems exploit leakage more than baselines [VERIFIED: https://arxiv.org/abs/2112.04426] | F9 audits | Publish with/without flagged items; private post-cutoff headline |
| 4 | Judge learns dataset artifacts or gets reward-hacked | Strict track; transfer to KoBEST and a fresh commonsense set; held-out property tests | Train the judge on executed outcomes wherever possible; eval judges never used as a reward |
| 5 | TPU MFU < 8% or v5e unavailable | X0; weekly quota log | T4-first plan (~17 T4-weeks); TRC |
| 6 | JAX-CUDA fails on sm_75 | X0 | T4s become generation/eval only, eager PyTorch inference port with parity tests |
| 7 | Teacher throughput is below 0.1B/week | Week-1 benchmark | Shrink judge items to 0.75M; lean on CPU procedural generators |
| 8 | CPU-bound tools and eval | Eval logs | Vectorized evaluators, cache, laptop overnight, async queue |
| 9 | Laptop SSD < 150 GB | Day 1 | 10B store tier |
| 10 | Local judge invalid | ρ < 0.8 | No chat number published |
| 11 | Licences (BY-SA spans, Korean unknown licences, teacher terms) | Manifest | Citations on copy; per-sample provenance; no-teacher track |
| 12 | Solo-developer overload | Weekly | Scope fence (§5.9), one codebase, admission rule |
| 13 | Benchmark gravity: tuning toward rows | — | Pinned modes, frozen generator manifest, held-out BBH families, Tier C |

### 10.5 Rejected alternatives and why

| Alternative | Why rejected |
|---|---|
| Any softmax-attention backbone, hybrids (StripedHyena, LFM2, GDN+SWA), HRM/TRM-Att, LLaDA/Mercury, BLT, Huginn as published | Violates F6 as published. The *mechanisms* (recurrent depth, recursion) are ported |
| Standard dense LLM pretraining at 0.1B | A capacity wall: MT-Bench ~2, TriviaQA 4.1, GSM8K 1.4 at 135M [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M] |
| MoE | The cap is on *total* params, so MoE buys nothing here |
| Product-key memory at M1 | Memory values count toward the 100M; kept as Pool M for 1B+ [VERIFIED: https://arxiv.org/abs/2412.09764] |
| CORTEX full stack (hippocampus, Titans test-time SGD, axial latent workspace, span tagger, 200M-phrase index) | Too many interacting modules for one person; stale phrase keys; test-time SGD breaks batching. Pieces are admission candidates |
| ORRERY-L latent diffusion + energy descent | Unproven for language; ~3× cost per refinement step; energy hacking. ORRERY-0 discipline kept as the control arm |
| Separate 12M verifier (CORTEX) | Too small for commonsense and chat judgment; replaced by a tower that reads the full trunk |
| Phrase pointer softmax over positions or ≤64 items | Gray zone under F6; replaced by the suffix automaton (A3 uses hard nearest-neighbour lookup only) |
| Up to 10% synthetic text in the gated store (OUROBOROS) | A partial teacher-cache channel; replaced by F3, human-only |
| Benchmark train splits banned outright (ORRERY) | GPT-3.5's own exposure is unknown, and FiD-class references train on them. Kept as the reported Strict track |
| Triton / flash-linear-attention kernels on T4 | Triton 3.3+ excludes sm_75; one pure-JAX codebase instead |
| 175M+ retrievers (Contriever-class) | Alone they break the cap [VERIFIED: https://arxiv.org/html/2407.12854]; frozen trunk keys cost 163,840 params |
| Distilling from GPT-3.5/4 (TinyGSM, MetaMathQA, UltraChat) | ToS, plus contamination of the head-to-head |
| Multi-accounting or tunnel-served APIs to stretch quota | Against ToS; never |

---

## 11. References

Only URLs that appear in the research digests and proposals given to this synthesis.

**Targets, benchmarks and evaluation**
- https://arxiv.org/abs/2303.08774 · https://arxiv.org/abs/2404.14219 · https://arxiv.org/abs/2312.11805 · https://evalplus.github.io/leaderboard.html · https://arxiv.org/abs/2407.02485
- https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard · https://arxiv.org/abs/2306.05685 · https://huggingface.co/spaces/lmsys/mt-bench · https://github.com/tatsu-lab/alpaca_eval · https://github.com/lmarena/arena-hard-auto/blob/main/misc/past_leaderboards.md
- https://arxiv.org/abs/2402.11548 · https://arxiv.org/abs/2309.02706 · https://arcprize.org/blog/oai-o3-pub-breakthrough · https://arxiv.org/abs/2307.04721 · https://github.com/EleutherAI/lm-evaluation-harness · https://github.com/openai/simple-evals · https://arxiv.org/abs/1907.10641

**Small models and tricks**
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M · https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct · https://arxiv.org/html/2402.14905 · https://arxiv.org/html/2509.24945v3 · https://qwenlm.github.io/blog/qwen2.5-llm/ · https://arxiv.org/html/2505.09388
- https://developers.googleblog.com/en/introducing-gemma-3-270m/ · https://arxiv.org/abs/2306.11644 · https://arxiv.org/html/2309.05463 · https://arxiv.org/html/2312.09241 · https://arxiv.org/abs/2305.07759 · https://github.com/KellerJordan/modded-nanogpt · https://arxiv.org/html/2404.05405

**Recursion and architectures**
- https://arxiv.org/abs/2510.04871 · https://arcprize.org/blog/hrm-analysis · https://arcprize.org/blog/arc-prize-2025-results-analysis · https://arxiv.org/abs/2502.05171
- https://arxiv.org/abs/2412.06464v3 · https://github.com/NVlabs/GatedDeltaNet · https://arxiv.org/abs/2503.14456 · https://arxiv.org/abs/2405.21060v1 · https://arxiv.org/abs/2501.00663 · https://arxiv.org/abs/2412.09764 · https://arxiv.org/abs/2008.02217 · https://arxiv.org/abs/2509.14252

**Retrieval and externalization**
- https://arxiv.org/abs/2112.04426 · https://arxiv.org/abs/2208.03299 · https://arxiv.org/html/2407.12854 · https://arxiv.org/abs/2007.01282 · https://arxiv.org/abs/2307.06962 · https://arxiv.org/abs/2310.11511
- https://huggingface.co/blog/embedding-quantization · https://arxiv.org/abs/2407.03618 · https://huggingface.co/minishlab/potion-base-8M

**Tools, test-time compute, self-improvement**
- https://arxiv.org/abs/2211.10435 · https://arxiv.org/abs/2302.04761 · https://arxiv.org/abs/2408.03314 · https://arxiv.org/abs/2502.06703 · https://arxiv.org/html/2501.12948 · https://arxiv.org/html/2203.14465
- https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/ · https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/

**Data, licensing and contamination**
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu · https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus · https://huggingface.co/datasets/HuggingFaceTB/cosmopedia · https://arxiv.org/abs/2506.20920v1 · https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
- https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/main/README.md · https://www.huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/blob/main/README.md · https://arxiv.org/abs/2406.08464v1 · https://arxiv.org/pdf/2505.24760 · https://arxiv.org/pdf/2404.07353
- https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/ · https://download.wikipedia.com/legal.html · https://arxiv.org/pdf/2507.04329 · https://arxiv.org/pdf/2510.09426 · https://huggingface.co/datasets/OpenLab-NLP/corpus-ko-23m/blob/main/README.md
- https://openai.com/en-GB/policies/terms-of-use/ · https://arxiv.org/pdf/2311.04850v1 · https://arxiv.org/pdf/2411.03923v1 · https://arxiv.org/pdf/2411.15124

**Compute, toolchain and operations**
- https://www.kaggle.com/discussions/product-announcements/607202 · https://github.com/ARahim3/kaggle-tpu-lab · https://github.com/Kaggle/kaggle-cli/issues/1197 · https://gpuperhour.com/blog/free-cloud-gpus-and-credits · https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html
- https://docs.cloud.google.com/tpu/docs/v5e · https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf · https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md · https://docs.jax.dev/en/latest/installation.html
- https://sites.research.google/trc/faq/ · https://huggingface.co/docs/hub/en/storage-limits

---

*One line:* ALEMBIC-0 puts procedures and a judge in 81.1M attention-free weights, facts in a 20B-token human-written library, depth in time, and truth in executors. It budgets 6.4 of 12 free TPU-weeks and expects to win 3–6 rows at M1. It names the four bosses it will probably lose at 0.1B, and it has the measured slopes on those bosses in front of the user by week 8, before any large run.
