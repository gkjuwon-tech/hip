# OUROBOROS-0 v2: A Pragmatic, Scaling-First Hybrid (proposal `scaler_v2`)

Chief-architect proposal, revised after red-team review. Philosophy: **use proven linear-time recurrent parts, add recurrent depth for test-time compute, and put knowledge in a datastore and tools. Every axis must have a measured scaling law before it is scaled. Build the smallest system that can test the bet, and add modules only when an ablation earns them.** Date: 2026-10-19.
Labels: [VERIFIED: url] = from the research digests' primary sources; [ESTIMATE] = my arithmetic or judgment; [HYPOTHESIS] = untested design bet.

---

## 1. Thesis

A dense LLM spends its parameters on three things:
- **(a) memorized facts,**
- **(b) depth per token,**
- **(c) procedures**: the reusable skills of reading, decomposing, computing, checking and writing.

Facts are the most expensive. Knowledge capacity is about 2 bits per parameter [VERIFIED: https://arxiv.org/html/2404.05405]. A 135M model trained on 2T tokens still scores 4.1% on TriviaQA [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct]. Depth is the second cost, because a fixed-depth network is a fixed-size circuit.

The bet has three parts:
- **Replace (a) with a datastore.** It costs 0 parameters and is worth about 10–25× parameters on knowledge tasks [VERIFIED: https://arxiv.org/abs/2112.04426].
- **Replace (b) with weight-tied recurrent depth.** A 7M recursive network beats frontier LLMs on ARC-AGI-1 [VERIFIED: https://arxiv.org/abs/2510.04871].
- **Replace the "being right" part of (c) with exact verifiers and tools.** PAL beats PaLM-540B CoT by 15 points [VERIFIED: https://arxiv.org/abs/2211.10435]. A 0.5B model plus search beats GPT-4o on MATH-500 [VERIFIED: https://arxiv.org/abs/2502.06703].

That leaves the weights two jobs: **a compact library of procedures**, and **enough fluency to write well**. v1 underweighted the second job. v2 treats fluency as a first-class job with its own experiment (E0) and its own parameter pool.

**What v2 changes in spirit.** v1 accepted losing the rows that decide the fight. v2 does not accept a loss, and it does not hide one either. The operating rule is **"win-or-extend"**:
- The <100M cap, the fair-play rules and the one-checkpoint rule never change.
- No victory is declared until every gated row is won.
- Each milestone grows only the axes that are free under the cap: store, test-time compute, tools and self-generated data.
- Each milestone publishes the full scoreboard, losses included.

The user said "losing is not acceptable." Under this rule a loss is a scheduled campaign, never a reported win. The "mid-boss" framing still holds: we do not tune per benchmark to get there.

**Honesty up front.** With today's evidence, open-ended chat, HellaSwag, WinoGrande, MMLU/ARC-C and code are the likely losses at M1 (§5). v2 puts an early go/kill experiment on each of these walls, so we learn within weeks rather than months whether the plan's slope can reach them.

---

## 2. Fair-play rules (pre-registered)

1. **Parameters** are every trainable weight that runs at inference: embeddings, backbone, recurrent core, retrieval projections, state-merge gates, and the value/verifier head.
   - Hard cap: **< 100M**. v2 uses **75.56M**.
   - Training-only weights (an optional JEPA predictor, auxiliary probes) and teacher models are **not counted**, but they are disclosed with their sizes.
2. **Not parameters, but disclosed as data:** tokenizer, BM25 statistics, IVF centroids and binary/PQ codebooks (learned, so their sizes are reported), and the datastore text and its codes. Each datastore snapshot is dated and hashed.
3. **Datastore provenance rules** (new):
   - **(a)** Every store vector is produced by the in-budget encoder.
   - **(b)** Synthetic (teacher-generated) text is **≤10% of store tokens**. It must be generated before any eval prompt was seen, decontaminated like everything else, and disclosed per source.
   - **(c)** A **human-text-only store** track is reported next to the headline, so readers can see how much of any win comes from stored teacher text. Without this, the store would work as a lookup-table copy of a >100M teacher at inference.
4. **Allowed at inference:**
   - the frozen offline datastore;
   - a sandbox containing **only** the Python stdlib, numpy, sympy and z3, plus a unit-test runner for tests the model writes itself.
   - The sandbox image is built from an allow-list. It has no ML libraries, no weight files and no network, and its hash is published. Everything runs on CPU with 0 parameters.
5. **Banned at inference:**
   - any neural network over 100M parameters, and any neural network outside the counted budget;
   - any LLM API and live web search;
   - routing, prompts or heads keyed on dataset identity;
   - running the hidden benchmark tests during generation.
6. **Training data:**
   - Open-licensed teacher models may generate data. Each teacher and its size are logged, and a "no-teacher" ablation track is reported.
   - **No GPT-3.5/4 outputs anywhere.** Using them breaks the OpenAI ToS [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/] and contaminates the head-to-head. This excludes TinyGSM, MetaMathQA, UltraChat and TinyStories from training. They remain usable as *published evidence* only.
   - **Official benchmark train splits** (WinoGrande, HellaSwag, ARC, GSM8K, MBPP train sets): pre-registered as **allowed and disclosed** on the headline "Open" track, because GPT-3.5's exposure to them is unknown. A "Strict" track with no benchmark train splits is also reported. Test and validation splits are never used for training.
7. **Architecture rule, stated up front:** no softmax (or Hopfield-equivalent) attention over the token sequence, at any layer, and no pointer softmax over token positions.
   - A GDN block keeps the projection and MLP macro-structure of a Transformer block, and reviewers may call it a linear Transformer.
   - **The non-Transformer claim rests on system properties:** constant-size recurrent state, weight-tied recurrent depth, externalized memory, and tool-grounded verification. It does not rest on the block's anatomy. We say this up front.
   - **No gray zone remains at M1:** product-key memory is cut, and copying is a 0-parameter deterministic controller operation (§3.2).
8. **Decontamination:**
   - 13-gram, MinHash and embedding-cosine (>0.9) scrubs of the training data **and the datastore** against every eval set and its mirrors, plus a URL block-list of benchmark-hosting domains, including Korean exam-hosting sites for KMMLU and HAE-RAE in FineWeb-2-ko.
   - Audit logging per eval item. For generative QA, the % of top-k passages containing the gold string. For multiple choice, the maximum n-gram overlap of the question and of each option text with retrieved passages (the gold letter alone means nothing).
   - A private post-cutoff test set (fresh math, code, Korean and chat prompts) is the headline sanity check.
9. **Scoring:**
   - Generative answers everywhere, shot counts matched to the target row, one checkpoint and one config for all rows, and 3 seeds for anything sampled.
   - FLOPs and wall-clock per query are published next to every score, along with active FLOPs/token at each r.
10. **Judging chat:**
    - Training-time judges and eval judges come from **different model families**.
    - Eval uses a panel of at least 2 local open-weight judges, run in both orders, against stored GPT-3.5 answers. The panel's rank correlation with the stored GPT-4-judge scores of ≥5 reference models is published.
    - A blind volunteer human eval (100 prompts, English and Korean) is reported.
    - An LMArena listing is pursued as the real arbiter. A 75M model runs on CPU, so hosting it is not compute-bound [ESTIMATE].

---

## 3. Architecture

### 3.1 Component diagram (M1 build; cut modules listed below)

```
                       ┌──────────────── OFFLINE (0 params) ──────────────────┐
                       │ DATASTORE v0: 30B tok (≤10% synthetic)               │
                       │  ├─ dense: 256-bit binary codes over ALL 234M chunks │
                       │  │   (IVF, in-RAM ~7.5 GB) + int8 rescoring (mmap)    │
                       │  ├─ BM25: compressed on-disk inverted index over a   │
                       │  │   5B-token core (en/ko Wikipedia, top FineWeb-Edu, │
                       │  │   Python docs) + Korean char-n-gram BM25           │
                       │  └─ suffix automaton over each retrieved set (copy)   │
                       │ TOOLS: allow-listed sandbox (stdlib/numpy/sympy/z3)   │
                       │        + test-runner. Hash published.                 │
                       └────────────▲──────────────────────▲─────────────────┘
                                    │ SEARCH(q)/hop        │ PY/SOLVE/CHECK
 tokens ─► [Factorized Emb 8.55M] ─►┌──────────────┐       │
                                    │ PRELUDE      │ 4× GDN, d=640, 5 heads×128
                                    │ 20.05M       │ (L1–2 frozen after Stage 1
                                    └──────┬───────┘  = KEY ENCODER)
     retrieved chunks ─► prelude+core(@query r) ─► per-(layer,iter) states
                                           │           │ STATE MERGE S0 = Σ g_i·S_i
                                           │           │ + top-2 chunks APPENDED as text
                                           ▼           ▼
                         ┌───────────────────────────────────────────┐
                         │ RECURRENT CORE (weight-tied) 25.07M       │◄─ input injection
                         │ 5× GDN, iterated r times per QUERY (1..32)│   adapter 0.82M
                         │ + learned <think> tokens for per-position │
                         │   extra compute (time-axis micro-steps)   │
                         └───────────────────┬───────────────────────┘
                                             ▼
                         ┌───────────────────────────────────────────┐
                         │ CODA 4× GDN 20.05M → tied LM head          │
                         │ VALUE HEAD on trunk 0.83M: p(correct),     │
                         │   preference, difficulty→(r, N, K budget)  │
                         └───────────────────┬───────────────────────┘
                                             ▼
                         ┌───────────────────────────────────────────┐
                         │ CONTROLLER (code, 0 params): compute-      │
                         │ optimal budget, adaptive K & search hops,  │
                         │ <copy> suffix-match extension, sample N,   │
                         │ exact-verify, value-rank, vote             │
                         └───────────────────────────────────────────┘
```

**Cut from M1. Each can come back only through an ablation that earns it (§10):**
- product-key memory (now a 1B+ scaling knob);
- Titans-style test-time memory MLP;
- Lean 4;
- the separate 9M critic branch (replaced by a value head on the shared trunk);
- the per-token halting head (replaced by per-query r plus think tokens);
- the phrase-copy pointer head (replaced by suffix-match copy);
- RWKV-7 as an E1 contender;
- the JEPA loss. It becomes an optional ablation only; it is training-only either way.

**Backbone:** Gated DeltaNet (GDN). It beats Mamba-2 and DeltaNet on language modeling, commonsense and in-context retrieval [VERIFIED: https://arxiv.org/abs/2412.06464v3]. Mamba-2 is the single pre-registered challenger in E1 [VERIFIED: https://arxiv.org/abs/2405.21060v1].

**Width:** d=640 with **5 heads × head_dim 128** (v1 had 9×64). This follows the red-team MFU point: a 128-wide head fills a 128×128 matrix tile, while a 64-wide head half-fills it [ESTIMATE: v5e MXU tile size not verified; measured in the E1b MFU gate]. Chunk size 128.

**Depth:** 4 prelude + 5r core + 4 coda layers. That is 33 effective layers at r=5 and 48 at r=8, which keeps MobileLLM's deep-and-thin advantage for this size class [VERIFIED: https://arxiv.org/html/2402.14905].

### 3.2 The three mechanisms that v1 got wrong, now defined precisely

**(1) Recurrent depth: per query, not per token.**

The red team is right that per-token adaptive r is ill-defined in v1. In the Huginn-style layout, iteration i of the core is a full pass over the sequence and keeps its own per-layer state across time. If token t−1 halted at r=2, the iteration-5 state for t−1 does not exist.

I considered the red team's option (b), extra micro-steps in one shared time recurrence, in its literal form: each micro-step feeds on the previous micro-step's output. That version is **serial across the whole expanded sequence**. Micro-step i's input is only known after micro-step i−1 finishes, so chunkwise-parallel training is impossible, and training throughput would collapse [ESTIMATE].

v2 therefore splits test-time compute into two well-defined axes, each trained the way it is used:
- **Depth axis, r per query.** The controller fixes r for a whole request (or a whole answer segment) using the value head's difficulty output. Training samples r per sequence from {1, 2, 4, 8, 16, 32} (log-uniform, mean ≈ 4 in Stage 2). Train and test therefore run the same rule.
  - TPU: one XLA `while_loop` with a static r_max and masking, not 5 separate graphs.
  - GPU: CUDA graphs with r_max masking.
- **Time axis, learned `<think>` tokens.** This is the parallelizable form of option (b). The model may emit `<think>` tokens. Each one has a *fixed* learned input embedding, so the expanded sequence stays fully parallel under chunkwise training, while the recurrent state integrates extra update steps at hard positions. This gives per-position adaptivity, and it is well-defined because a `<think>` token is just a token.
  - Training: `<think>` tokens are inserted into procedure traces before hard steps, initially by heuristic (before tool calls and before answer tokens). RLVR then rewards using fewer of them.
- E1 measures train/test consistency for both axes (§10).

**(2) Retrieval fusion: state priming plus append, with an honest fallback.** [HYPOTHESIS]
- Each retrieved chunk runs once through prelude and core **at the query's r**, with all chunks batched in parallel.
- Its per-(layer, iteration) delta states are merged into the query's initial states with learned per-head gates: S₀ = Σ g_i·S_i.
  - The gates come from a 640→5 projection per layer over the chunk's final hidden state, for 9 layers (4 prelude + 5 core).
- **The top-2 chunks are also appended as plain text** in front of the query. Answer spans therefore exist as real tokens, and the copy operation has text to match.
- Capacity: 9 layers × 5 heads × 128² ≈ 0.74M floats per iteration of state for K=8×128 = 1,024 retrieved tokens. That is about 720 floats per retrieved token, against v1's ~180 [ESTIMATE]. It is still lossy, which is why the top-2 append exists.
- **K is adaptive (8 to 64)** under controller control. The model may issue up to 3 `<search>` hops, and candidates are reranked by the value head.
- **Fallback if E2 kills state merge:** pure text-append, the proven FiD/RETRO-style pattern. It is compliant but costs about K× more serial scan.

**(3) Copying: a 0-parameter, compliant suffix-match operation.**

v1's phrase-copy head pointed at token positions. That is a softmax over positions, and after state priming the chunk tokens no longer exist to point at. v2 replaces it with a controller operation:
1. The model emits `<copy>` followed by 1–3 normal tokens.
2. The controller looks up that prefix in a **suffix automaton** built over the retrieved text and the prompt.
3. The controller extends the match greedily. It scores the proposed continuation with the model's **own vocabulary distribution** in one teacher-forced chunkwise pass, which is cheap for a linear RNN. Extension stops at a sentence boundary or when the model's token probability falls below a threshold τ, tuned on dev.

No pointer, no softmax over positions, and no parameters. The CoG-style phrase index [VERIFIED mechanism: https://arxiv.org/abs/2307.06962] stays out of M1.

### 3.3 Inference dataflow

1. **Encode** the prompt through the embedding and prelude. The value head estimates difficulty, which sets the budget (r, N, K).
2. **Plan.** The model emits tool tokens: `<search>q</search>`, `<py>`, `<sympy>`, `<z3>`, `<check>`, `<copy>`, `<think>`, `<answer>`.
3. **Retrieve.**
   - Dense candidates: IVF lists over binary codes from frozen prelude layers 1–2 plus a 640→256 projection, rescored with int8.
   - BM25 candidates over the 5B-token core.
   - The two lists are fused by reciprocal rank fusion, then the value head reranks the top K.
   - Optional further hops.
4. **Prime state.** Merge the K chunks into the initial states and append the top 2 as text.
5. **Think.** The core runs r iterations per token, and the model may insert `<think>` tokens.
6. **Act.** Tool calls run in the allow-listed sandbox, and their outputs are spliced back as tokens.
7. **Write.** The coda produces token logits, and `<copy>` triggers suffix-match extension.
8. **Verify and select.**
   - The controller samples N candidates where the value head is uncertain, following Snell's compute-optimal allocation [VERIFIED: https://arxiv.org/abs/2408.03314].
   - Exact checks come first: execution, SymPy equality, self-written tests.
   - The value head (same trunk, prompted with `<judge>`) breaks ties.
   - The answer is a majority vote or best-of-N.

---

## 4. Exact parameter budget

GDN layer at d=640 (5 heads × 128) [ESTIMATE, arithmetic]:

| Component | Formula | Params |
|---|---|---|
| q, k, v, o projections | 4d² | 1,638,400 |
| output gate | d² | 409,600 |
| α/β gates | 2·d·H | 6,400 |
| short conv (k=4) on q, k, v | 3·d·4 | 7,680 |
| SwiGLU MLP, hidden 1536 | 3·640·1536 | 2,949,120 |
| norms | — | 1,920 |
| **Total per layer** | | **5,013,120** |

| # | Module | Shape / formula | Params |
|---|---|---|---|
| 1 | Factorized embedding (tied output); 256-d bottleneck per red-team 2.6 | 32,768×256 + 256×640 | 8,552,448 |
| 2 | Prelude: 4 GDN layers (L1–2 frozen after Stage 1) | 4 × 5,013,120 | 20,052,480 |
| 3 | Recurrent core: 5 GDN layers, weight-tied, iterated r times | 5 × 5,013,120 | 25,065,600 |
| 4 | Coda: 4 GDN layers (generator capacity, up from 3) | 4 × 5,013,120 | 20,052,480 |
| 5 | Input-injection adapter | Linear(2d→d) + bias | 819,840 |
| 6 | Retrieval query/key projection | 640×256 | 163,840 |
| 7 | State-merge gates | 9 layers × 640×5 | 28,800 |
| 8 | Value/verifier head on the shared trunk (p(correct), preference, difficulty, budget) | 640→1280→4 | 825,604 |
| 9 | Final norm | — | 1,280 |
| | **TOTAL (M1 base)** | | **75,562,372 ≈ 75.56M** |
| | Headroom under the cap | | **24.44M** |

**The headroom is a pre-registered decision pool, not a slush fund.** It is spent only on these triggers:
- **Pool G (generator), +10.03M:** 2 more coda layers. Released if E0 shows the chat win rate rising with params (§10).
- **Pool M (memory), +10.06M:** product-key memory, re-admitted only if E2-extended shows a factual-QA slope at fixed FLOPs. Its top-k softmax over parameter slots would then be declared as a gray zone, with a ReLU variant reported alongside.
- **Worst case with both pools:** 95.65M, still under 100M.

Removed from v1 (11.4M in all):
- PKM 10.06M → Pool M;
- test-time memory 2.32M;
- critic branch 9.34M → value head 0.83M on the trunk;
- halting head and phrase-copy head 0.15M → 0 params;
- JEPA predictor 1.33M → training-only, uncounted.

**The verifier now has trunk capacity.** The value head reads the full 74.7M trunk, which answers the TinyGSM point that the verifier should not be smaller than the generator [VERIFIED: https://arxiv.org/html/2312.09241].

**Active parameters per token (forward):**

| r | Active params |
|---|---|
| 1 | ≈73.7M |
| 4 | ≈149M |
| 8 | ≈249M |
| 16 | ≈450M |

[ESTIMATE] These are published next to every score.

---

## 5. Beating the scoreboard, row by row

Targets are the max reported GPT-3.5 scores. Estimates are for milestone M1 (month 12). "P(win)" is my probability of beating target + CI on the Open track. **These are judgment calls, not measurements** [ESTIMATE throughout]. Changes from v1 are marked Δ.

| Row (target) | Mechanism | Est. M1 | P(win) | Why |
|---|---|---|---|---|
| NQ-open (38.6) | adaptive K + hops + append top-2 + suffix copy | 40–50 | 0.65 | FiD (220M reader) reached 48.2 [VERIFIED: https://arxiv.org/abs/2007.01282]; the risk is our smaller reader |
| TriviaQA (85.8) | same | 70–82 Δ | 0.20 Δ | Llama-2-7B with a 1.4T store reaches 77.0 [VERIFIED: https://arxiv.org/html/2407.12854]. Adaptive K and hops help, but this is a hidden wall |
| ARC-AGI-1 (≈5, need >7.1) | recurrent core + test-time training on augmented demos | 8–20 Δ | 0.60 Δ | Red-team 2.5 accepted. TRM is a grid specialist trained for about 4 H100 × 3 days, and TRM-MLP drops to 29.6 [VERIFIED: https://arxiv.org/html/2510.04871v1]. A general text model under one checkpoint is weaker |
| GSM8K (78.1) | program-of-thought + Python + trunk verifier + 64 votes | 60–78 | 0.30 Δ | The TinyGSM 63.1 evidence used banned GPT-3.5 data [VERIFIED: https://arxiv.org/pdf/2312.09241]. We rebuild from OpenMathInstruct-2 (CC-BY-4.0) plus procedural data, with a trunk-sized verifier |
| MATH (34.1) | SymPy + search | 25–40 | 0.35 | Qwen2.5-0.5B-Instruct scores 34.4 [VERIFIED: https://qwenlm.github.io/blog/qwen2.5-llm/]; the tool takes over the algebra |
| BBH (68.3) | executable subtasks go to Python; the rest to the core | 55–70 | 0.35 | About 30% of BBH is exactly executable |
| DROP F1 (64.1) | append + suffix-copy spans + calculator | 55–70 | 0.40 | No retrieval needed; arithmetic is offloaded |
| KMMLU (42.47) | Korean store + reader | 38–48 | 0.40 | The target is low (chance is 25) |
| HAE-RAE (~49–55) | Korean dictionary/wiki store | 45–55 | 0.35 | Vocabulary and loanword subsets are lookups; history is hard |
| KoBEST (79.3) | reader + negation curriculum + commonsense verifier | 62–75 | 0.20 | COPA and HellaSwag inherit the commonsense wall |
| MMLU (71.4) | exam-explanation store + option elimination + verifier | 50–60 | 0.08 | Retrieval adds only +3–4 points on reasoning-heavy knowledge [VERIFIED: https://arxiv.org/html/2407.12854]; Atlas-770M reached 56.3 [VERIFIED: https://arxiv.org/abs/2208.03299] |
| ARC-Challenge (87.4) | textbook store + verifier + ARC train split (Open track) | 55–70 | 0.07 | Qwen2.5-0.5B gets 35.6 |
| HumanEval+ / HE (70.7 / 76.8) | generate, self-test, execute, repair (8 rounds) | 40–58 | 0.08 | phi-1-small (350M) reached 45 [VERIFIED: https://arxiv.org/abs/2306.11644] |
| MBPP+ / MBPP (69.7 / 82.5) | same + API-doc retrieval | 50–65 | 0.08 | same |
| HellaSwag (85.5) | per-option verifier + retrieved analog narratives | 45–62 | 0.02 | Nothing ≤0.5B exceeds 52.1 |
| WinoGrande (81.6) | contrastive verifier + WinoGrande train split (Open) + synthetic pairs | 55–70 | 0.03 | Synthetic data moved phi-1.5 from 60.4 to 73.4, at 1.3B [VERIFIED: https://arxiv.org/html/2309.05463] |
| MT-Bench pairwise vs stored GPT-3.5 (need ≳58% with 160 turns) | plan → realize → self-critique → best-of-8 with the trunk verifier | 10–30% | 0.03 | Every ~130M model scores 2.0–2.5 on MT-Bench vs 8.39 [VERIFIED: https://arxiv.org/html/2402.14905] |
| AlpacaEval 2 LC (22.35) | same | 5–15 | 0.03 | same |
| Korean chat / creative writing | same + style exemplars from the human-text store | weak | <0.02 | Smallest Arena model to beat GPT-3.5 is about 2.6B [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard] |

**Plain verdict.** P(sweeping every Tier A+B row at M1) is roughly 0 [ESTIMATE]. That holds for v1 and v2, and for every published system under 1B. What v2 changes is **when we find out and what we do about it**: each wall now has an experiment that reads its slope by week 11, and the win-or-extend rule (§1) governs what happens after M1.

**Wall campaigns, each tied to a go/kill:**
- **Chat (E0, weeks 2–5)** [HYPOTHESIS].
  - The generator's job is cut into short, locally fluent pieces: an outline, then 2–4 sentence sections realized one at a time, then a self-critique-and-revise pass, then best-of-8 ranked by the trunk verifier.
  - Exemplar retrieval comes from the **human-text store** (StackExchange-style answers and how-to pages; licenses per manifest), so the human-only track stays meaningful.
  - Preference training (DPO/GRPO) uses a training judge from a different family than the eval judges.
  - **E0 measures whether judged quality has any slope at this scale before we spend on it.**
- **Commonsense (E3b, weeks 7–11)** [HYPOTHESIS].
  - Each option is a hypothesis, and the trunk verifier scores it contrastively.
  - Data: about 2M situation/consequence pairs with hard negatives, about 0.2B teacher tokens generated on the TPU in weeks 1–6, plus official train splits on the Open track.
  - Tested at the **deployable** size: the verifier is the trunk itself.
- **Code.** Execution-guided repair, self-tests and pass-rate-weighted voting. The binding limit is base pass@k, so code-with-tests data gets 15% of Stage 2 and 0.5B teacher tokens.
- **Knowledge (MMLU/ARC-C).**
  - An exam-style explanation store: textbook chunks and worked solutions, decontaminated.
  - Option elimination by the verifier.
  - The E2-extended datastore slope decides whether "more store" can carry this row inside the cap.

---

## 6. Training recipe

**Training cost per token** [ESTIMATE]:
- Stage 1, r̄ = 2: N_active ≈ 98.8M, so 6N ≈ 5.9e8. Adding 20% for stop-grad neighbor encoding gives **≈ 7.1e8 FLOPs/token**.
- Stage 2, r̄ ≈ 6, truncated backprop through the last 4 iterations (counted at full forward+backward, to be conservative): ≈ **1.3e9 FLOPs/token**.

| Stage | Tokens | Data | Objectives |
|---|---|---|---|
| 0. Infra (CPU / 3060 / T4), weeks 0–4 | — | **Tokenizer:** 32k BPE over en/ko/code. Seeded with the 2,350 common Korean syllables plus byte fallback, not all 11,172 syllables; the freed vocabulary goes to Korean morphemes (red-team 2.6). **Datastore v0:** 30B tokens, ≤10% synthetic. **Decontamination pipeline** and allow-listed sandbox image | — |
| T. Teacher generation (TPU, weeks 1–6, while the JAX port is written) | ≈4.4B teacher tokens if the third-party ~0.74B/week holds [VERIFIED (third-party): https://github.com/ARahim3/kaggle-tpu-lab] | Budget: 0.4B math program-of-thought (from 2–3M OpenMathInstruct-2 problems [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2], answer-matched only); 0.2B commonsense contrast pairs; 1.5B simple-register en/ko instruction/chat (for E0 and SFT); 0.5B code with tests (execution-filtered); 0.5B Korean procedures; ~1.3B buffer. Teacher: an open-weight model whose license permits training on outputs, verified per model card | — |
| 1. Fluency + retrieval pretraining | 80B | FineWeb-Edu-dedup 45%, FineWeb-2-ko + kowiki 15%, Cosmopedia 10%, Python-Edu/Stack-permissive 15%, FineMath-4+/OpenWebMath 10%, KILT-style QA 5% [VERIFIED sources: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus, https://arxiv.org/abs/2506.20920v1]. 50% of batches run with retrieval on, over held-out store text | Next-token loss; r sampled per sequence, log-uniform, mean 2. At the end of the stage, freeze L1–2 and the embedding and build the dense index |
| 2. Procedure mid-training | 40B (≈10B unique × 4 epochs) | Reasoning Gym generators [VERIFIED: https://arxiv.org/pdf/2505.24760]; re-ARC on the train split only [VERIFIED: https://arxiv.org/pdf/2404.07353]; SymPy/Z3 generators; program→exact-output traces; tool-call traces with `<think>`/`<copy>`; teacher PoT math; code with tests; Korean procedures. Only **verified** samples are kept | Next-token loss on calls and results (tool outputs masked); deep supervision on every iteration; value-head loss (correct/incorrect labels from executors); r mean 6 |
| 3. SFT | 3B | SmolTalk (Apache-2.0) [VERIFIED: https://www.huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/blob/main/README.md]; Tulu-3 minus NC subsets [VERIFIED: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/main/README.md]; teacher-regenerated en/ko instructions | SFT. The value head's preference output is trained from a **training judge of a different family** than the eval panel |
| 4. RLVR | ~0.5B rollout tokens | Math, code, logic and retrieval-QA with exact-match rewards; rollouts on T4 and 3060 | GRPO [VERIFIED method: https://arxiv.org/html/2501.12948]. Rewards for correctness, fewer `<think>` tokens and lower r, and tool use where the answer is checkable |
| 5. Self-play flywheel (continuous) | grows from 10% to 50% of the mix | The model plus a proposer prompt generates frontier problems (50–70% solve rate); an exact checker gates every accepted solution | Expert iteration, STaR-style [VERIFIED: https://arxiv.org/html/2203.14465] |

**Token totals:**
- Overtraining ratio ≈ 120B / 75.6M ≈ **1,600 tokens/param**. SmolLM2-135M used about 15,000 [VERIFIED: https://arxiv.org/abs/2502.02737]. We deliberately trade raw tokens for data density.
- Stage FLOPs: Stage 1 is 80B × 7.1e8 ≈ 5.7e19; Stage 2 is 40B × 1.3e9 ≈ 5.2e19. **Total ≈ 1.1e20 FLOPs** [ESTIMATE].

---

## 7. Compute plan

Quotas come from third-party sources and are verified in-account on day 1: **TPU v5e-8 ≈ 20 h/week, 9 h sessions; T4×2 ≈ 30 h/week** [VERIFIED: https://www.kaggle.com/discussions/product-announcements/607202, https://gpuperhour.com/blog/free-cloud-gpus-and-credits].

**MFU is now planned conservatively.** The red team's 8–15% estimate for chunkwise GDN on v5e is adopted as the **planning value (12%)**. A measured ≥15% is a hard gate before Stage 1 (E1b).

| Platform | Peak | Planning MFU | FLOPs/week | Stage-1 tokens/week |
|---|---|---|---|---|
| TPU v5e-8 | 8×197 = 1,576 TFLOPS bf16 [VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e] | 12% (the JAX tutorial measured 16–27% on a tiny GPT [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html]) | 1576e12 × 0.12 × 72,000 ≈ 1.36e19 | ≈ 19B |
| T4×2 | 2×65 TFLOPS fp16 [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf] | 20% | ≈ 2.8e18 (if all 30 h) | ≈ 4B |
| 3060 laptop (6 GB, sm_86, bf16) | ~25 TFLOPS effective [ESTIMATE] | 35% | unmetered | E1 grid, E0, evals, RL rollouts |

**TPU budget** [ESTIMATE]:

| Item | TPU-weeks |
|---|---|
| Teacher generation, weeks 1–6 | 6 |
| Stages 1+2 (1.1e20 ÷ 1.36e19) | ≈ 8 |
| 75M-scale ablations and the E2/E3 points | ≈ 3 |
| RL | ≈ 1 |
| **Total** | **≈ 18 of 52** |

That leaves about 2× slack for MFU shortfalls, availability gaps and reruns.

**T4 budget, re-booked to fit 30 h/week** (red-team 1.2):

| Job | h/week | Notes |
|---|---|---|
| Index builds and refreshes | 4 | Encoder pass over 30B tokens is about 1e18 FLOPs [ESTIMATE] |
| RL rollouts, GSM8K/code repair sampling | 10 | Stage 4 onward; earlier, E2/E3 training |
| Teacher generation top-up (vLLM fp16) | 10 | After week 6, when the TPU moves to training |
| Full Tier A eval | 6 | Amortized: a full run of ≈15–30 T4-h happens **monthly**. Weekly evals use a stratified 10% dev slice **on the 3060** |
| **Total** | **30** | |

Two more rules keep the T4 quota free:
- Procedural generators and BM25 builds run in **Kaggle CPU sessions** or on the laptop CPU. Whether CPU sessions draw on GPU quota is [ESTIMATE, verify].
- The 75M model fits a 6 GB 3060 for evals and RL rollouts (1.2 GB weights and optimizer state at 16 B/param).

**If the TPU is unavailable** (red-team 1.3; v5e-8 is sometimes oversubscribed or disabled [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab]):
- **Stage 1 shrinks to 20B tokens** (≈260 tokens/param, still over 10× Chinchilla) and **Stage 2 to 30B**: 1.4e19 + 3.9e19 ≈ 5.3e19 FLOPs. That is about 19 weeks of all T4 time.
- During those weeks every other T4 job moves to the 3060 and the CPU, and teacher generation drops to whatever the 3060 can do (a 1.5B fp16 teacher fits in 6 GB [ESTIMATE]).
- It still fits in 12 months, but with almost no slack.
- Apply to **TPU Research Cloud in week 1** [VERIFIED: https://sites.research.google/trc/faq/]. It is free quota, but VM and storage costs are "minimal" rather than zero.

**Storage, corrected** (red-team 1.4) [ESTIMATE]:
- Dense: 234M chunks × 32 B of binary codes ≈ 7.5 GB in RAM. int8 rescoring vectors (256-d ≈ 60 GB) stay memory-mapped on disk.
- BM25 is **not** built over the full 30B tokens. That would be about 150 GB uncompressed, which matches the red-team figure. Instead it is a compressed on-disk inverted index (Lucene/PISA-class, about 2–3 B/posting) over a **5B-token core**: 39M chunks × ~80 terms × ~3 B ≈ 9–10 GB.
- The full store and all indexes live on the **laptop SSD**. Shards are published as public HF datasets with cards, because the private quota is only 100 GB [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits].
- Kaggle eval sessions mount only the binary codes, the BM25 core and the text shards needed. Peak RAM is about 18 GB of the ~30 GB available.

**One codebase** (red-team 7.1):
- **E1 runs on the 3060** using existing flash-linear-attention / mamba kernels. sm_86 runs current Triton; on sm_75 Triton ≥3.3 does not [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md]. This is a disposable ablation harness on library code, not our codebase.
- **Only the E1 winner is ported, to one JAX codebase** with pure-XLA chunkwise kernels (`lax.scan` over 128-token chunks, matmul inside each chunk, fp32 state). It is used for TPU training.
- Inference and evals run on GPU in the same JAX code if JAX CUDA works on sm_75/sm_86 [ESTIMATE, verify day 1]. Otherwise a small pure-PyTorch *inference-only* port is kept, with a numerical-parity test against JAX on every checkpoint.
- **If the E1b MFU gate fails on TPU** (below 10% after tuning): the project goes T4-only, and the PyTorch path becomes the single codebase (pinned Triton 3.2 or pure-PyTorch chunkwise). There are never two training codebases at once.
- **Build order:** core and retrieval → tools → verifier head → think/copy → anything an ablation earns.

**T4 numerics (sm_75, no bf16):**
- fp16 matmuls with dynamic loss scaling.
- Delta-rule states, log-space decay gates and scan accumulators kept in **fp32**.
- Gradient clipping per recurrence iteration.
- Activation checkpointing per core iteration, so memory stays flat in r.

**Persistence and sessions:**
- Checkpoint every 45 minutes to /kaggle/working, then push to the HF Hub.
- One compiled `while_loop` (static r_max, masked) avoids the ~10% compile overhead of 5 r-buckets [ESTIMATE].

---

## 8. Inference cost per query

Forward FLOPs per token ≈ 2·N_active: **1.5e8 at r=1, 3.0e8 at r=4, 5.0e8 at r=8** [ESTIMATE]. Priming runs prelude and core only, ≈ 2.4e8 per retrieved token at r=4.

| Query type | Budget | FLOPs | Latency [ESTIMATE] |
|---|---|---|---|
| Factoid QA | 1–3 searches (~100 ms CPU each); priming K=8×128 at r=4 (2.5e11); top-2 append; 20 output tokens | ~3e11 | ~0.4 s on T4 / 3060 |
| GSM8K | 64 samples × 300 tokens × r=8 (9.6e12) + verifier pass at r=4 (7.7e12) + 64 Python runs | ~1.7e13 | ~4–8 s, batched (the recurrent state is constant-size, so 64-way batching is cheap) |
| HumanEval | 16 samples × 400 tokens × up to 8 repair rounds × r=8 | ~2.6e13 | ~10–30 s + execution |
| Chat turn | outline + 8 drafts × 500 tokens × r=4 + verifier + one revise pass | ~3e12 | ~5–8 s. Serial depth (4 + 5r + 4 layer calls per token) dominates: about 10 ms/token eager at r=8, ~3–5 ms with CUDA graphs and r_max masking |
| ARC-AGI task | test-time training (~200 inner steps) + 32 samples | ~1.5e15 | ~1–3 min. All 400 tasks take ~7–20 h, run on the 3060 |

**Honest wall-clock note.** With CUDA graphs, padding to a fixed r_max mask erases the savings from a lower r inside a batch. Batches are therefore grouped by the controller's chosen r, and measured wall-clock is reported, not theoretical FLOPs.

Evaluation cadence:
- Full Tier A suite: ≈15–30 T4-h, or ≈40–60 h on the 3060 [ESTIMATE], monthly.
- 10% dev slice: weekly on the 3060.

---

## 9. Scaling story

| Axis | 0.1B (M1) | 1B | 10B | Measured by |
|---|---|---|---|---|
| Core params | 75.6M (+24M pools) | ~700M dense core | ~4B dense core | loss and pass@k vs N on {5M, 20M, 50M, 75M}, a 15× span |
| Sparse memory params | 0 (Pool M if earned) | ~300M PKM | ~6B PKM; memory layers scale to 128B memory params [VERIFIED: https://arxiv.org/abs/2412.09764] | factual QA vs PKM size at fixed FLOPs |
| Datastore | 30B tokens (≤10% synthetic) | 300B | 2T+ (no saturation seen up to 1.4T [VERIFIED: https://arxiv.org/html/2407.12854]) | QA accuracy vs log(store) **at a fixed 50M core**: 3B / 10B / 30B (100B stretch) |
| Test-time depth r | ≤32 | ≤64 | ≤128 | accuracy vs log(r), Huginn-style [VERIFIED mechanism: https://arxiv.org/abs/2502.05171] |
| `<think>` tokens | learned, RL-shaped | same | same | accuracy vs think-token budget |
| Samples × verifier | 64 | 256 | 1,024+ | accuracy vs log(N) |
| Tools | Python / SymPy / Z3 | + Lean, geometry DDAR engine | + docking, MD, protein-structure tools (gated) | coverage |
| Self-play data | 10–50% of the mix | majority | dominant | verified tokens/week |

**What the scaling claim rests on, honestly** (red-team 5.2):
- A parameter law fit on 5M–75M (15×) can predict the 0.1B → 1B step at best. **I do not claim it predicts 10B.**
- Inside the free envelope, the scaling story rests on the axes that grow **without retraining the core**: store size, r, think tokens, N × verifier, tools and self-play. Each gets its own slope at fixed core size in E1/E2.
- The composite fit Acc(N, D_store, r, k) is used only to choose the next step. A step is taken only if the fit predicts the held-out next point within its CI.
- Bigger cores raise base pass@k, which *multiplies* the value of search, because test-time compute needs a non-trivial base success rate [VERIFIED: https://arxiv.org/abs/2408.03314].

**Compute wall** [ESTIMATE]:
- 1B at 20B tokens is 1.2e20 FLOPs, about 9 TPU-weeks at 12% MFU. Possible on free quota, but tight.
- 10B at Chinchilla scale is about 1.2e22 FLOPs. That is out of reach without TRC or grants.
- Meanwhile every no-retraining axis keeps compounding, so capability keeps rising inside the free envelope even without TRC.

**Toward the north star** [HYPOTHESIS]:
- Science is a verifier-rich environment: proofs, simulations, assays. The design's main structural bet is the loop **propose → act with tools → verify → keep the verified data**.
- **Biosafety is a hard stop, not a knob:**
  - no pathogen-enhancement, toxin-synthesis or dual-use protocol content in the store or the training data;
  - the tool registry is allow-listed;
  - the controller enforces a refusal policy;
  - every wet-lab-adjacent output requires qualified human review;
  - bio models and data go through dual-use review before release.

---

## 10. First experiments (go/kill)

**E0: Fluency/chat ceiling (weeks 2–5, 3060). New; it addresses red-team 2.1.**
- **Setup:**
  - 20M and 50M GDN models × r ∈ {1, 4}, trained on about 1–2B tokens of teacher-regenerated, simple-register en/ko instruction data, a license-clean form of the TinyStories idea.
  - Pairwise scoring by the 2-judge eval panel against stored GPT-3.5 answers.
  - Slopes are measured on AlpacaEval's 805 prompts (±3.5-point CI [ESTIMATE]), because MT-Bench's 160 turns give only about ±8 points.
- **E0b:** the plan → realize → critique → best-of-8 pipeline vs plain sampling at 50M, and the embedding bottleneck at 192-d vs 256-d.
- **GO:** the win rate rises ≥5 points per doubling of params *or* of r, and the pipeline adds ≥3 points over plain sampling.
- **KILL (flat curve):**
  - Release Pool G into the generator anyway, and test whether width at fixed params beats depth.
  - **Publicly downgrade the chat rows to "expected loss at M1 — needs a new idea"**, with the measured curve attached.
  - Redirect chat effort to the store side (human exemplars) and to RL against the panel-independent training judge.
- **If GO:** release Pool G (+10M coda) at Stage 1.

**E1: Backbone × recurrence grid (weeks 1–6, 3060), and E1b: the JAX port MFU gate (weeks 4–8, TPU).**
- **E1:**
  - GDN vs Mamba-2 at 5M/20M/50M, 2–5B tokens each.
  - GDN at {20M, 50M} × r ∈ {1, 4, 16}, plus a `<think>`-token arm at r=4.
  - Metrics: validation loss, Reasoning Gym accuracy, needle recall, and the **train/test consistency of per-query r**: accuracy when trained on sampled r and tested at each fixed r, against training at that fixed r.
  - **GO:** r=16 beats r=1 by ≥8 points on procedural tasks; 50M@r=16 ≥ the 75M-equivalent extrapolation at r=1 (recurrence worth ≥1.5× params); and the per-query-r model is within 2 points of fixed-r training at every r.
  - **KILL:** a gain under 3 points beyond r=4. Then cap r at 4 and move the core's tied capacity into width and the coda.
- **E1b:** the winner ported to JAX, at the 75M configuration on v5e-8.
  - **GO:** measured MFU ≥15%.
  - **Below 15% but ≥10%:** tune (chunk 128→256, 4-way × 2 mesh) and proceed with the planning numbers.
  - **Below 10%:** T4-only plan, with the PyTorch path as the single codebase (§7).

**E2: Retrieval fusion and the datastore slope (weeks 5–9, T4/TPU).**
- **Setup:**
  - 50M model. Arms: no retrieval, text-append (K=8), state-merge (K=8), and state-merge + top-2 append.
  - Evaluated on NQ, TriviaQA and MMLU-dev with full leakage audits.
  - **Datastore slope:** store sizes 3B / 10B / 30B at a fixed core, with human-only vs ≤10%-synthetic stores.
- **GO:** state-merge + append recovers ≥80% of text-append's gain at ≤40% of its FLOPs; NQ ≥ 30 EM at 50M; and a positive log-store slope.
- **KILL:**
  - Under 50% of the append gain: switch to pure text-append.
  - If even text-append adds less than +10 NQ points over closed-book at 50M, the reader is too small. Move Pool M into the prelude.
- **E2-extended (Pool M):** 10M PKM vs none at matched FLOPs. Re-admit PKM only if factual QA improves by ≥3 points.

**E3: Procedures and commonsense (weeks 8–11).**
- **(a) GSM8K:** 75M stack with program-of-thought, Python, the trunk verifier and 64 votes. Data: answer-matched OpenMathInstruct-2 PoT conversions plus procedural generators. No TinyGSM.
  - **GO:** ≥55% on a decontaminated dev split by week 11.
  - **KILL:** <30%. The data supply is then the bottleneck: redirect T4 generation to verified PoT and pause the other stages.
- **(b) Commonsense:** the **deployable verifier** (the trunk's value head on a 50M proxy, scaled to 75M at Stage 2), trained on about 2M teacher contrast pairs (TPU-generated, weeks 1–6) plus the WinoGrande train split (Open track; the Strict track is reported too). Measured on the WinoGrande dev set.
  - **GO:** ≥60.
  - **KILL:** ≤54. Then publicly downgrade the commonsense rows to "expected loss at M1" and re-plan. Honesty beats hope.

**Timeline:**

| Weeks | Work |
|---|---|
| 0–1 | Day-1 checks; TRC application |
| 1–6 | E1, E0; TPU teacher generation |
| 4–8 | JAX port, E1b |
| 5–9 | E2 |
| 8–11 | E3 |
| 12–28 | Stage 1 |
| 28–38 | Stage 2 |
| 38–42 | SFT + RL |
| 42–52 | Self-play, full scoreboard, M1 report with every loss shown |

---

## 11. Top risks

1. **The fluency/chat wall** (highest). No evidence exists that any model under 1B reaches GPT-3.5 chat quality. E0 now reads the slope by week 5. Mitigations: Pool G, plan-then-realize decoding, best-of-N with the trunk verifier, human exemplars. It may still lose, and under win-or-extend it would then be a published open campaign, not a hidden one.
2. **The commonsense wall** (HellaSwag/WinoGrande). Retrieval offers no lookup key for these rows. The verifier route is now tested at deployable size (E3b).
3. **Retrieval fusion is lossy** (E2). Top-2 append and adaptive K give partial cover. The full fallback costs about 8× more serial scan but stays compliant.
4. **TPU MFU and availability.** Planned at 12% with a ≥15% gate. A T4-only fallback is budgeted (§7). A TRC application goes in during week 1.
5. **Synthetic-data throughput.** TPU teacher generation is front-loaded in weeks 1–6, and procedural CPU generators carry most of Stage 2.
6. **Judge Goodharting.** Train and eval judges come from different families; a 2-judge eval panel with published correlation; blind volunteer human eval; LMArena as the arbiter.
7. **Hidden-teacher channels.** A ≤10% synthetic store cap with a human-only track, and an allow-listed sandbox with a published hash.
8. **Datastore contamination creating a fake win.** RETRO-style models exploit leakage more strongly than baselines [VERIFIED: https://arxiv.org/abs/2112.04426]. Mitigations: the audit, including option-text overlap for multiple choice, Korean exam-site block-lists, and the private post-cutoff set.
9. **fp16 recurrence instability on T4.** Mitigated by fp32 state, log-space gates and per-iteration clipping.
10. **Solo-developer overload.** One training codebase, two backbone contenders, eight modules cut from M1, and a fixed build order.
11. **Licensing.** NC and BY-SA manifests; BY-SA text stays in the store; never closed-API outputs; teacher licenses checked per model card.
12. **Scaling past 1B depends on TRC or grants.** The no-retraining axes keep the project productive without them.
13. **Biosafety at the north-star stage.** A hard gate, as described in §9.

---

## Changelog v1 → v2

**Fatal issues**
- **2.1, chat had no experiment:** added E0, the fluency/chat ceiling test (weeks 2–5, 3060), with AlpacaEval-sized CIs, GO/KILL rules and Pool G (+10M generator) tied to its outcome. Added the "win-or-extend" rule: losses are never reported as wins, and the cap never moves.
- **7.1, two codebases and too many modules:** E1 runs on the 3060 with library kernels, and only the winner is ported to one JAX training codebase. Cut from M1: PKM, Titans memory, Lean, RWKV-7, the critic branch, the halting head, the copy head and JEPA. Fixed build order.

**Major issues**
- **1.1 MFU:** head_dim 128 (d=640, 5 heads) and chunk 128; planning MFU 12%; a ≥15% gate (E1b) with explicit fallbacks.
- **1.2 T4 overbooked:** T4 re-booked to 30 h/week; weekly evals and RL on the 3060; full evals monthly; teacher generation front-loaded onto the TPU in weeks 1–6; CPU sessions for procedural generators and BM25.
- **1.3 no-TPU fallback:** recomputed at 20B + 30B tokens (≈19 T4-weeks), with every other job moved off the T4s; TRC application in week 1.
- **1.4 BM25 size:** corrected to about 150 GB uncompressed for the full store. BM25 now covers a 5B-token core only (compressed on disk, ~10 GB); dense binary codes cover everything; the store is hosted on the laptop SSD with public HF shards.
- **2.2 GSM8K evidence relied on banned data:** TinyGSM is evidence only; data is rebuilt from OpenMathInstruct-2 PoT, answer-matched; the verifier is a value head on the full trunk.
- **2.3 TriviaQA capped by K=8 merging:** adaptive K from 8 to 64, up to 3 search hops, verifier reranking, top-2 append and suffix-match spans.
- **2.4 commonsense campaign over budget:** about 2M pairs generated on the TPU, tested with the deployable verifier; train-split rule pre-registered (Open gated, Strict reported).
- **3.1 judge Goodharting:** separate judge families for training and eval; 2-judge eval panel with published correlation; blind human eval; LMArena.
- **3.2 teacher text in the store:** ≤10% synthetic cap, in-budget encoder only, human-text-only track.
- **3.3 sandbox could hide a network:** allow-listed image with no ML libraries, no weights, no network, hash published.
- **4.2 phrase-copy head:** replaced by 0-parameter suffix-automaton copy that is scored by the model's own vocabulary distribution.
- **5.1 per-token r ill-defined:** r is now per query, trained by per-sequence sampling, with a consistency check in E1. Per-position compute comes from learned `<think>` tokens. The red team's literal micro-step proposal would be serial across the whole expanded sequence and untrainable in parallel, so v2 adopts its parallelizable form (fixed-embedding think tokens).
- **5.2 narrow scaling span:** a 5M point is added (5M–75M, 15×); datastore and r slopes are measured at a fixed core; no claim of extrapolating to 10B.

**Minor issues**
- **1.5:** one masked `while_loop` replaces the r-buckets.
- **2.5:** ARC-AGI estimate lowered to 8–20%, P(win) 0.60.
- **2.6:** 2,350 Korean syllables plus byte fallback; 256-d embedding.
- **3.4:** multiple-choice audit uses option-text overlap; Korean exam-site block-lists.
- **3.5:** training-only weights are uncounted but disclosed; active FLOPs per r published.
- **4.1:** the rule and the system-level novelty claim are stated up front.
- **5.3:** PKM moved to an evidence-gated pool.
- **6.1:** CUDA graphs, batches grouped by r, wall-clock reported.
- **6.2:** evals run on the 3060.

**Parameters:** 86.96M → **75.56M** base, with 24.44M in two pre-registered pools (worst case 95.65M).

Saved to: /home/user/hip/asi-blueprint/proposals/scaler_v2.md
