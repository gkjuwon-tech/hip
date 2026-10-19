# Proposal v2: COMPILER-0, a neuro-symbolic compiler under 100M parameters

Author role: chief architect, "Neuro-Symbolic Compiler" track. Date: 2026-10-19. Revision of `compiler_v1.md` after the red-team critique.
Labels: [VERIFIED: url] = from the research digest or the red-team inputs (checked against the source there), or fetched in this session. [ESTIMATE] = my arithmetic or judgement. [HYPOTHESIS] = an untested design bet.

---

## 1. Thesis

A frontier LLM spends its parameters on three jobs:

1. **Storing facts.** That runs at about 2 bits per parameter, and retrieval is the only fix shown to work at small sizes [VERIFIED: https://arxiv.org/html/2404.05405].
2. **Emulating computation.** Carrying digits, tracking variables and searching options, all done imprecisely inside the weights.
3. **Producing fluent surface text.**

A Python interpreter does job 2 exactly, at zero parameters. A human-written datastore does job 1 with no counted parameters and never forgets. Human-written prose already contains fluent text for job 3, ready to copy and recombine. What has to live in the weights is two things:

- **A compiler.** It turns a messy natural-language task into a typed program over tools, reads the results, and decides when to stop.
- **A judge.** It ranks candidates when no exact verifier exists.

The bet is that compiler plus judge is much smaller than the thing it replaces. The evidence:

- A 125M generator that writes Python reaches 63.1% on GSM8K [VERIFIED: https://arxiv.org/html/2312.09241].
- PAL with Python beat PaLM-540B chain-of-thought by 15 points [VERIFIED: https://arxiv.org/abs/2211.10435].
- A 7M recursive net scores 44.6% on ARC-AGI-1 [VERIFIED: https://arxiv.org/html/2510.04871v1].
- Retrieval stood in for about 25x the parameters in RETRO [VERIFIED: https://arxiv.org/abs/2112.04426v1].
- A verifier added more than a bigger generator did: "the verifier seems to play a bigger role than generation model size" [VERIFIED: https://arxiv.org/html/2312.09241].

In this design, language is the I/O layer and the program trace is where the thinking happens. The north star lives in that trace too. Science is a verifier-rich environment (simulators, proofs, assays), and a compiler is the kind of agent that can drive one.

**What v2 changes in the thesis.** v1 treated the commonsense and chat rows as a side quest. v2 treats them as the main risk of the whole project, for two reasons:

- They are the rows most likely to kill the "never lose" requirement.
- Discrimination (pick the plausible option) is a different and cheaper skill than generation (write the plausible text). v2 therefore spends about 17% of the budget on a **bidirectional judge tower**. That money comes from deleting the LoRA mode adapters and the Titans memory.

The honest corollary has not moved. The compiler thesis is strong wherever a program, a lookup or a verifier exists. It is weakest where the skill *is* fluency or broad intuitive world-modeling. **Re-deriving every row from its gate (target + 95% CI), as the critique demanded, puts the probability of winning all of Tier A + B at v1 at roughly zero [ESTIMATE].** The critique's product-of-confidences figure (about 1e-13) overstates the precision, because the rows are correlated, but its conclusion is right. v2 does not hide this. Instead, it tests the walls that would kill the plan at **week 2**, and it pre-registers the escalation path with the user (§10).

---

## 2. Fair-play rules (binding)

1. **Parameters.** Every learned number executed at inference counts:
   - the token embedding, every backbone layer, the judge tower and all heads;
   - the retrieval projection and the per-token retrieval weights;
   - the **unigram tokenizer scores** (32,768 scalars);
   - **any learned costs inside tools** (for example, the CRF costs of a MeCab-ko-class morphological analyzer). v2 avoids that case by using a purely rule-based Korean pre-splitter.

   The total must be under 100,000,000. The count is produced by a script run against the real instantiated module, not a formula (§4).
2. **Data, not parameters (disclosed with sizes):**
   - datastore text, BM25 statistics, IVF centroids and PQ codebooks (learned, so their sizes are reported);
   - the phrase index;
   - the program-library source code (text).
3. **Allowed at inference:** a frozen, dated, hashed offline datastore; a Python 3 sandbox (nsjail or seccomp) with the standard library, NumPy, SymPy and NetworkX. Z3 is added once it passes the §10 ablation gate. Everything runs on CPU, offline. Lean 4 is moved to the north-star track, not the milestone.
4. **Banned at inference:** any neural network over 100M parameters, all LLM APIs, the live web, and any eval item or its solution in the datastore, phrase index or library.
5. **Inference stores must be human-written or self-generated** (critique F3). The datastore, exemplar bank, phrase index and skill library may contain only:
   - (a) human-authored text and code with known provenance, or
   - (b) text and programs produced by COMPILER-0 itself (self-play), which already passed an executable check.

   **Teacher-model text (Cosmopedia, OpenMathInstruct-2 solutions, Magpie-style regenerations, synthetic SmolTalk/Tulu turns) is allowed in training only, never in an inference store.** Retrieving teacher text at inference would be a lookup-table approximation of a >100M network, which is exactly what rule 4 bans.
6. **Training time.** Open-licensed teacher models may generate training data, and every teacher and its size is disclosed. A "no-teacher" ablation track is reported.
   - **No GPT-3.5/4 outputs anywhere** (OpenAI ToS prohibits using output "to develop models that compete with OpenAI" [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/]). This excludes TinyGSM, MetaMathQA and original UltraChat.
   - Every Tulu-3 and SmolTalk subset card is audited for OpenAI-generated responses. Any subset that contains them, or whose card is unclear, is dropped [HYPOTHESIS that some do; audit pending].
   - NC-licensed subsets (for example Tulu's No Robots, CC-BY-NC-4.0 [VERIFIED: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/main/README.md]) are logged in a manifest so that a commercial fork can drop them.
7. **Benchmark train splits** (critique F1). Published train splits (NQ, TriviaQA, HellaSwag, WinoGrande, DROP, GSM8K, MBPP-train, ARC train, KILT) may be used, under these conditions:
   - They go into **one shared training mix** that produces **one checkpoint** with **no dataset routing**.
   - Their use is disclosed.
   - A "no-benchmark-train-split" score is reported as a Tier C row.

   Rationale: FiD's NQ score, which the scoreboard uses as a reference, was trained on NQ train [VERIFIED: https://arxiv.org/abs/2007.01282]. GPT-3.5's own training mix is undisclosed.
8. **No routing by dataset identity.** The planner sees only the prompt. Hidden unit tests are never run. Every eval answer logs its retrieval and tool trace. "Abstain" is disabled in eval mode, because abstaining scores as wrong.
9. **Decontamination.** Applied to training data, the datastore, the phrase index and the library:
   - 8- to 13-gram filtering plus MinHash, and an embedding scrub at cosine > 0.9 with our own retriever.
   - **Whole-domain blocklists:** wikihow.com and ActivityNet captions (the HellaSwag sources), public Korean exam sites (the KMMLU/HAE-RAE sources), and benchmark mirrors (HF dumps, GitHub copies of HumanEval/MBPP).
   - Retrieval-audit logging: the share of eval items whose top-k contains the gold answer string.
   - The overlap report is published.
   - Every store-dependent row is reported **with and without the store**.
10. **Generator manifest freeze** (critique F5). The procedural-generator list is frozen and hashed *before* anyone looks at eval items. **Five BBH task families are drawn by a seeded random draw from the 27 task names and committed before generator design.** No generator may target them, and they are reported separately.
11. **Judging chat.** Two independent open-weight judges from different model families, both answer orders, plus a 100-pair blind human check. **The eval judges are never used as an RL reward or as a training filter.**
12. **Scoring.** Generative answers everywhere. Multiple-choice is answered as an emitted option letter: the system's own judge may pick it, because the judge is part of the system. Target = the maximum reported GPT-3.5 score per row. **Win = score ≥ target + 95% CI half-width. Metrics must match** (EM against EM, accuracy against accuracy).

---

## 3. Architecture

### 3.1 Component diagram

```
        INPUT (en/ko text, code, grid)
                 │
        ┌────────▼─────────┐  rule-based ko pre-split (josa/eomi suffix list, 0 learned params)
        │ Unigram 32k vocab│  ~7k entries reserved for Korean syllables/morphemes; byte fallback
        └────────┬─────────┘  (32,768 scores COUNTED as params)
                 │
 ┌───────────────▼─────────────────────────────────────────────────────────────┐
 │ CAUSAL TRUNK: Gated-DeltaNet, d=640, expand_v=1, ZERO softmax attention       │
 │  Prelude (3) ─► Recurrent Core (4 tied layers × r loops, r∈[1,32], halting)  │
 │             ─► Coda (3)                                                       │
 │  MODE TOKENS (0 params): <PLAN> <READ> <VERIFY> <REALIZE> <JUDGE>             │
 └──┬──────────────┬──────────────────┬───────────────────┬──────────────────────┘
    │ IR tokens     │ query vec 256-d  │ phrase query 256-d │ per-token states h_t
    ▼               ▼                  ▼                    ▼
 ┌────────────┐ ┌───────────────────┐ ┌──────────────────┐ ┌─────────────────────────────┐
 │ Program IR │ │ RETRIEVER          │ │ PHRASE INDEX     │ │ JUDGE TOWER (bidirectional) │
 │ grammar-   │ │ on-disk BM25 +     │ │ external MIPS,   │ │ 3 BACKWARD GDN layers over   │
 │ constrained│ │ binary dense       │ │ top-k ≤ 64 human │ │ the trunk's states + fusion  │
 │ decoding   │ │ (doc side = E·P    │ │ phrases; copied  │ │ MLP. One tower, two jobs:    │
 │            │ │ mean-pooled, CPU)  │ │ verbatim         │ │ VERIFY traces + rank options │
 └─────┬──────┘ │ RRF fusion         │ └──────────────────┘ └──────────────▲──────────────┘
       │        └─────────┬──────────┘                                     │
       ▼                  ▼                                                │
 ┌───────────────────────────────────────────────────────────┐            │
 │ EXECUTORS (0 params, CPU): Python sandbox · SymPy ·        │── results ─┘
 │ NetworkX · test runner · ARC grid DSL · LIBRARY (text)     │   (spliced into the stream)
 │ [Z3, KG, geometry engine: added only after an ablation]    │
 └───────────────────────────────────────────────────────────┘
       ▲
 ┌─────┴──────────────────────────────────────────────────────────────┐
 │ DATASTORE (data): ≤50B tokens v1, HUMAN-WRITTEN or SELF-GENERATED   │
 │ only; 128-tok chunks; on-disk BM25; 256-bit binary vectors; phrase  │
 │ index; exemplar bank of human answers; frozen, hashed snapshot      │
 └────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why each v1 component was cut or changed

| v1 component | v2 decision | Reason |
|---|---|---|
| 4 LoRA mode adapters (6.55M) | **Mode tokens (0 params)** | They were capacity spent on switching behaviour. The judge needs that capacity more |
| Titans fast-weight memory (1.64M) | **Deleted** | Redundant: a GDN state already *is* a delta-rule fast-weight matrix. Fp16 test-time gradient updates were a risk |
| Separate 6M verifier tower | **Merged into the judge tower** | One bidirectional scorer covers both trace verification and option plausibility |
| Pointer head over ingested spans | **External phrase index, MIPS, top-k ≤ 64** | v1's head was softmax over the token sequence, i.e. attention (critique §4) |
| 6.3M static index embedder | **Doc side = the trunk's tied embedding E × projection P, mean-pooled with a learned per-token weight (32,768 params)** | Saves 6.3M, and the doc side is recomputable on CPU |
| Jamo split + 24k vocab | **32k unigram, about 7k Korean entries, rule-based pre-split** | Jamo splitting lengthens Korean sequences on a fixed-state recurrence (critique B9) |
| Lean 4, KG, Z3, MCTS, library learning in v0.1 | **Staged** (§6). Each is admitted only after an ablation shows ≥ +2 points | Solo-developer scope (critique §7) |

### 3.3 The Program IR

The core does not emit free-form chain-of-thought. It emits a **typed intermediate representation**: a strict subset of Python plus tool primitives.

```python
facts = SEARCH("ko+en", "boiling point of water at 2000 m", k=32)  # BM25 + dense, RRF
ev    = READ(facts, ask="temperature in C")                        # span + citation
x     = SOLVE(sympy_eq)                                            # SymPy; SAT(z3) once admitted
ok    = CHECK(pred, x)                                             # executable assertion
best  = JUDGE([c1, c2, c3, c4], ctx=prompt, evidence=ev)            # judge tower ranking
ANSWER(x, cite=ev.cite) | RENDER(plan, style="concise", lang="ko")  # I/O layer
```

Decoding is grammar-constrained. Illegal tokens are masked, so every sample parses (0 params).

### 3.4 The judge tower

This is the v2 bet against the F-wall [HYPOTHESIS].

HellaSwag, WinoGrande, KoBEST-COPA/HellaSwag, ARC-C and MMLU are all **discrimination** tasks: given k options, pick one. A causal LM has to *generate* or *likelihood-score* each ending left to right. A bidirectional reader sees the whole (context, option) pair at once.

- The best published WinoGrande systems in 2019 (fine-tuned discriminative models, RoBERTa-class) reached 59.4–79.1% depending on training-data size [VERIFIED: https://arxiv.org/abs/1907.10641]. This was a time when 1B-class causal LMs were near chance.
- That ceiling is still **below the 83.7 gate**. Even the strongest version of the discriminative route does not clear WinoGrande on prior evidence.

**Mechanism:**

1. The causal trunk runs forward over `[context ; <JUDGE> ; option_i]` (r = 2 loops).
2. Three GDN layers scan the trunk's hidden states **backward** (right to left; still a recurrence, no softmax).
3. The fusion MLP takes `[h_fwd_last ; h_bwd_first]` (1280-d) → 1024 → 1 and produces a scalar energy per option.
4. Options are ranked, and the argmax letter is emitted.
5. In VERIFY use, the same tower scores `(prompt, program, execution trace, answer)`.

**Training signal, roughly in order of expected value [HYPOTHESIS]:**

1. **Teacher-ranking distillation.** An open-weight Apache-licensed teacher on T4 scores candidate endings for about 20–50M synthetic multiple-choice items built from *human* text (FineWeb-Edu, Wikipedia; wikihow.com and ActivityNet blocked). We distill the teacher's *ranking distribution*, not its text. Strong-to-weak logit distillation is the Qwen3 small-model recipe [VERIFIED: https://arxiv.org/html/2505.09388]. This is training-time only and allowed under rule 6, and the teacher is disclosed.
2. **Benchmark train splits** under rule 7: HellaSwag train (about 40k), WinoGrande-XL train, the ARC train split, KoBEST train sets.
3. **Self-made contrastive triples** (hundreds of millions, free): shuffled steps, swapped entities, negated outcomes, wrong-referent pronoun substitutions.
4. **Verification labels** from S2/S5 rollouts: (trace, was it correct?).

### 3.5 Inference dataflow

1. **Tokenize.** Korean goes through the rule-based pre-split.
2. **Prelude encode** under `<PLAN>`. The trunk emits a 256-d query vector. First retrieval: on-disk BM25 plus binary dense, reciprocal-rank fusion, top-32. It runs for every query.
3. **Read.** Retrieved chunks are scanned under `<READ>` by plain recurrent ingestion (no cross-attention). Chunks are short (128 tokens), and the program can re-retrieve, so exact recall does not rely on a long carried state.
4. **Plan.** The recurrent core loops r times and samples N IR programs. The halting head and controller head set r and N: easy prompts get N=1, r≈2; hard prompts get N up to 64, r up to 32. This follows the compute-optimal allocation idea [VERIFIED: https://arxiv.org/abs/2408.03314].
5. **Execute.** Programs are generated whole, then executed asynchronously on CPU (critique C6), with a 2 s timeout per call and at most 10 tool calls per program. Multi-turn tool loops at inference use a host-side loop with prefix caching of the recurrent state. A GDN state is constant-size, so resuming after a tool result costs only the new tokens [ESTIMATE].
6. **Verify.** Exact checks (CHECK asserts, self-written tests, SymPy substitution), then majority vote over executed answers, then the judge tower's VERIFY energy. These are combined by a calibrated logistic model fit on dev rollouts [ESTIMATE].
7. **Search.** If the best score is below threshold, the controller repairs using the error message, re-retrieves with a reformulated query, or deepens r, within the caps in §8.
8. **Realize** under `<REALIZE>`. The trunk writes glue tokens. At each step it may emit `<COPY>`, which issues a 256-d phrase query to the **external phrase index**. Of the top-64 human-written phrases, the trunk's own vocabulary head plus 64 extra logits (q·phrase_emb) decide which phrase to copy verbatim. This is softmax over ≤64 *external items*, under the gray-zone clause in the archs digest, reported separately. It is not attention over the token sequence.
9. **Emit** the answer with citations (which also satisfies the BY-SA attribution obligation for Wikipedia/StackExchange spans [VERIFIED: https://download.wikipedia.com/legal.html]) and a trace hash.

---

## 4. Exact parameter budget

Layer definition, fixed in config with `expand_v=1`, d=640, 10 heads × 64:

- q, k, v, o and output-gate projections: 5·d² = 2,048,000.
- β and α (decay) projections: 2·d·10 = 12,800.
- Depthwise short convs, k=4, on q/k/v: 4·3d = 7,680.
- Two RMSNorms: 1,280.
- SwiGLU FFN, hidden 1707: 3·640·1707 = 3,277,440.
- **Per layer: 5,347,200.**

Computed by script, not by the 13d² approximation.

| # | Module | Shape / formula | Params |
|---|---|---|---|
| 1 | Token embedding, tied with the LM head | 32,768 × 640 | 20,971,520 |
| 2 | Prelude: 3 GDN layers | 3 × 5,347,200 | 16,041,600 |
| 3 | Recurrent core: 4 weight-tied GDN layers (looped, counted once) | 4 × 5,347,200 | 21,388,800 |
| 4 | Coda: 3 GDN layers | 3 × 5,347,200 | 16,041,600 |
| 5 | Loop-input injector (concat(state, prelude out) → d) | 2·640² + 640 | 819,840 |
| 6 | **Judge tower: 3 backward GDN layers** | 3 × 5,347,200 | 16,041,600 |
| 7 | Judge fusion MLP (1280 → 1024 → 1) | 1280·1024 + 1024 + 1024 + 1 | 1,312,769 |
| 8 | Retrieval query/doc projection P (640 → 256) | 640 × 256 | 163,840 |
| 9 | Per-token retrieval weights (doc-side pooling) | 32,768 | 32,768 |
| 10 | Phrase-query projection (640 → 256) | 640 × 256 | 163,840 |
| 11 | Unigram tokenizer scores (counted per rule 1) | 32,768 | 32,768 |
| 12 | Halting + controller head (r, N) | 640·64 + 64 + 64·4 + 4 | 41,284 |
| 13 | Numeric-slot head (marks IR variables) | 640·128 + 128 | 82,048 |
| 14 | Final norm | 640 | 640 |
| | **Total counted** | | **93,134,917** |
| | Headroom under 100M | | 6,865,083 |

Checks:

- **expand_v trap** (critique §7.4). If the implementation forces `expand_v=2`, each layer grows to 6,578,560 and the total becomes **109,142,597, over budget**. The fix: keep `expand_v=1`, or drop the FFN hidden to 1300, which brings the total under 99M with expand_v=2 [computed by script]. **The CI test is `sum(p.size for p in params) < 100e6` on the instantiated JAX module, and it fails the build otherwise.**
- **Effective depth.** Per token it is 3 + 4r + 3 layer-passes: 38 at r=8, from 10 unique causal layers. This transplants the Huginn/TRM idea onto an attention-free block [VERIFIED idea: https://arxiv.org/abs/2502.05171].
- **Headroom (6.87M) has two allowed uses**, decided by experiments, not taste:
  - (a) a dedicated 4M doc-side retrieval table, if E2 shows the E·P pooling recall is too low;
  - (b) one extra unique REALIZE/coda layer (5.35M), if E1 shows realize quality is the bottleneck.

---

## 5. Benchmark-by-benchmark plan vs the GPT-3.5 scoreboard

Confidence = P(v1 at week 24 ≥ **gate**), with gate = target + 95% CI half-width from the baseline scoreboard [ESTIMATE, all rows]. Estimates are deliberately conservative. Where v1 used the wrong anchor, the corrected anchor is cited.

| Row | Target | **Gate** | Mechanism | Anchor evidence | Est. | Conf. |
|---|---|---|---|---|---|---|
| ARC-AGI-1 pass@2 | 10.6 (text-davinci-003, 800 public tasks [VERIFIED: https://arxiv.org/abs/2307.04721]) | ≈13.6 on 400 eval tasks | Grid-DSL program search + per-task test-time training of a small head + judge ranking | TRM 7M 44.6, attention-free TRM-MLP 29.6 [VERIFIED: https://arxiv.org/html/2510.04871v1]; CompressARC 76K about 20% [VERIFIED: https://arcprize.org/blog/arc-prize-2025-results-analysis] | 15–30 | 0.7 |
| NQ-open EM | 38.6 | 40.2 | Retrieve + READ + exact span; NQ train allowed (rule 7) | FiD 440M total: 48.2 [VERIFIED: https://arxiv.org/abs/2007.01282] | 36–47 | 0.5 |
| KMMLU | 42.47 | 43.0 | Korean human-written store (ko-Wikipedia, FineWeb-2-ko; exam sites blocked) + cross-lingual retrieval + judge option scoring | Polyglot-Ko-1.3B 27.97 [VERIFIED: https://ar5iv.labs.arxiv.org/html/2402.11548]; GPT-3.5 is weak here | 36–48 | 0.3 |
| HAE-RAE (per subset) | ~30–63 per subset | +~2.5 | LW/RW/SN = Korean dictionary lookups; HI/GK = retrieval; RC = READ | GPT-3.5 HI 30.32, GK 35.80 [VERIFIED: https://arxiv.org/abs/2309.02706] | per subset mixed; the RC subset is the risk | 0.25 (all subsets) |
| DROP F1 | 64.1 | 65.1 | READ spans → Python arithmetic; DROP train allowed | no ≤100M anchor | 55–70 | 0.35 |
| BBH 0-shot | 68.3 | 69.4 | About half the tasks are executable (Dyck, boolean, arithmetic, sorting, dates, shuffled objects, web of lies, navigate); the language-heavy half goes through the judge | critique B5 arithmetic: ~95 × 13 + ~50 × 14 ≈ 71 | 60–74 | 0.3 |
| GSM8K 0-shot | 78.1 | 80.3 | IR + Python + maj@32 + judge VERIFY | TinyGSM 125M gen alone 63.1 (GPT-3.5-generated data, which we cannot use); + 1.3B verifier 76.1; interpolated ~69 with a ~125M verifier [ESTIMATE from https://arxiv.org/html/2312.09241] | 65–78 | 0.15 |
| MATH 4-shot | 34.1 | 35.4 | SymPy + brute-force small-case enumeration + (if admitted) a deductive geometry engine | best ≤150M: MATH500 4.6 [VERIFIED: https://arxiv.org/html/2509.24945v3] | 15–32 | 0.1 |
| TriviaQA **EM** | 82.9 EM (0-shot, 1106) [VERIFIED: https://arxiv.org/abs/2407.02485] | ~83.6 | Retrieve + READ; TriviaQA train allowed; metric-matched (critique B7) | FiD-base EM 65.0 [VERIFIED: https://arxiv.org/abs/2007.01282] | 60–75 | 0.03 |
| KoBEST avg | 79.3 (En-prompt avg) | — | READ for BoolQ, judge for COPA/HellaSwag/SentiNeg; KoBEST train allowed | GPT-3.5 Ko-prompt subsets 82.34/57.88/40.00/91.69 [VERIFIED: https://arxiv.org/abs/2309.02706] | 62–78 | 0.15 |
| HumanEval+ / HumanEval | 70.7 / 76.8 | 77.7 / 83.8 | Generate + self-tests + execute-repair + human API-doc retrieval | phi-1-small 350M: 45; phi-1 1.3B: 50.6 [VERIFIED: https://arxiv.org/abs/2306.11644] | 35–60 | 0.02 |
| MBPP+ / MBPP | 69.7 / 82.5 | 74.3 / 87.1 | Same | same | 40–62 | 0.02 |
| HellaSwag 10-shot | 85.5 | 86.2 | Judge tower (§3.4); store **without** wikihow/ActivityNet | SmolLM2-135M 42.1 [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M]; phi-1.5 1.3B 47.6 [VERIFIED: https://arxiv.org/html/2309.05463] | 50–72 | 0.02 |
| WinoGrande 5-shot | 81.6 | 83.7 | Judge tower + pronoun-binding programs | 2019 SOTA ceiling 79.1 [VERIFIED: https://arxiv.org/abs/1907.10641]; phi-1.5 73.4 | 58–72 | 0.01 |
| MMLU 5-shot | 71.4 | 72.1 | Exam-explanation store (human textbooks, decontaminated) + **per-option evidence retrieval** (query = question + option_i) + judge contrastive scoring + STEM → SymPy | **Atlas-770M 5-shot 38.9** (56.3 was the full/transfer setting; v1 misread this) [VERIFIED: https://arxiv.org/abs/2208.03299 via the tiny_sota digest] | 38–52 | 0.01 |
| ARC-Challenge | 87.4 | 89.3 | Elementary-science textbook shard + per-option evidence + judge | Qwen2.5-0.5B 35.6 [VERIFIED: https://qwenlm.github.io/blog/qwen2.5-llm/] | 42–60 | 0.01 |
| MT-Bench | 8.39 | pairwise > 50% + CI | Plan → copy-realize; math/code/reasoning/extraction turns go to the compiler | ~125–135M models: 1.98–2.52 [VERIFIED: https://arxiv.org/html/2402.14905] | 3–5 | <0.01 |
| AlpacaEval 2 LC | 22.35 | + CI | Same, pairwise vs stored GPT-3.5 outputs, two judges | — | 3–12 | <0.02 |
| Arena-Hard v0.1 | 24.8 | + CI | Same | — | 1–6 | <0.01 |

### Where the confidence comes from, grouped

- **Likely wins:** ARC-AGI-1, and NQ-open (coin flip).
- **Contested:** KMMLU, HAE-RAE, DROP, BBH.
- **Long shots:** GSM8K, MATH, KoBEST.
- **Expected losses at 0.1B:** code, TriviaQA-EM, HellaSwag, WinoGrande, MMLU, ARC-C, all chat rows.

**The overall win condition is not reachable by v1, by my own estimates.** I am not hiding this to protect the proposal. The plan exists to (a) win what is winnable, (b) measure the gap on everything else with a named lever, and (c) find out by week 8, not week 24, whether the F rows need 1B parameters.

### Chat, creative writing, Korean fluency (pre-registered escalation row)

GPT-3.5's chat quality comes from RLHF over massive preference data. The attack:

- **Plan-then-realize:** PLAN emits an outline program (intents, sections, constraints such as "3 bullets, formal, Korean").
- Each section is grounded in retrieval from the **human-written exemplar bank** (StackExchange, Wikipedia, human-authored instruction datasets under the manifest).
- REALIZE copies phrases from the external phrase index. CoG-style copy LMs improve by enlarging the index without retraining [VERIFIED: https://arxiv.org/abs/2307.06962].
- Format constraints are **executed as checks** (word count, bullet count, language-ID, JSON validity). This is the one chat-adjacent area where real gains are likely: IFEval-style following.

Even a perfect ~9 on the math, coding, reasoning and extraction turns, plus ~3 on writing, roleplay, humanities and STEM explanation, averages about 6 [ESTIMATE, critique B8]. **This row is pre-registered with the user as the escalation row. It is the most likely reason the "never lose" requirement cannot be met at 0.1B by any design known to me.**

---

## 6. Training recipe

### 6.1 Staged minimal core (critique §7)

- **v0.1 (weeks 1–8):** GDN trunk + looped core + mode tokens + Python/SymPy sandbox + on-disk BM25 over Wikipedia en+ko + judge tower.
- **Each later admission** must show ≥ +2 points on the frozen dev suite in a single ablation. The admission queue: binary dense retrieval → phrase index → Z3 → ARC DSL → GRPO → skill library → KG → geometry engine. Lean stays on the north-star track.

### 6.2 Stages

All counts are token-passes [ESTIMATE].

| Stage | Weeks | Tokens | Data | Objective |
|---|---|---|---|---|
| S0 Tokenizer + datastore v0 + throughput bench | 1–2 | — | 32k unigram (fertility measured for ko/en/code first); human-written store: en+ko Wikipedia, FineWeb-Edu-dedup subset, FineWeb-2-ko, OpenWebMath/arXiv, permissive Python-Edu (human code from Stack v2 [VERIFIED: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus]) → ≤50B tokens, decontaminated | On-disk BM25 + CPU binary dense build |
| S1 Base language + reading | 2–6 | 30B | 15% fluency (human edu en/ko + Cosmopedia v1 *training-only* [VERIFIED: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia]); 35% retrieval-conditioned text (chunk + retrieved neighbours prepended in-stream); 30% code; 20% math web | Next-token + LLM-JEPA embedding loss [VERIFIED: https://arxiv.org/abs/2509.14252]; r ~ Poisson(4) |
| S2 Procedure + tool traces | 5–10 | 20B | Reasoning Gym 100+ generators [VERIFIED: https://arxiv.org/pdf/2505.24760] (manifest frozen, 5 BBH families held out); re-ARC train split [VERIFIED: https://arxiv.org/pdf/2404.07353]; own SymPy/graph/date generators; KILT provenance traces [VERIFIED: https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/]; OpenMathInstruct-2 (CC-BY-4.0) rewritten to IR, **kept only if execution reproduces the answer** [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2]; benchmark train splits (rule 7) | Next-token on IR, tool outputs masked from the loss |
| S3 Judge | 6–10 (in parallel) | 5B | Teacher-ranking distillation on 20–50M synthetic MC items from human text; benchmark train splits; contrastive triples; S2 rollout correctness labels | Listwise ranking loss + BCE on verification |
| S4 Chat/realize SFT | 10–12 | 2B | Human-written instruction data + audited SmolTalk/Tulu-3 subsets (training only), rewritten to outline → copy-realize traces | Next-token + copy loss + format-check rewards |
| S5 RLVR + self-play | 12–24 | **2–5B** rollout tokens (cut from 20B, critique C6) | Problem proposer under `<PLAN>`, aiming at the 50–70% solve frontier; verifiable-only acceptance | GRPO with exact rewards [VERIFIED approach: https://arxiv.org/html/2501.12948]; STaR-style expert iteration |

### 6.3 Recurrence training (fixes critique C2's contradiction)

Training uses TRM-style **deep supervision on detached outer steps**:

- The forward pass runs r loops.
- Loss is applied at up to 3 supervision points.
- Each point backpropagates through only its **last 2 inner loops**, plus the prelude via the injector. Earlier loops run without gradient.

There is no full BPTT anywhere.

### 6.4 Teachers

- Open-weight 1.5B–7B models run on T4 via vLLM, producing about 0.1–0.6B synthetic tokens per week [ESTIMATE from the compute digest].
- **Teacher outputs never enter an inference store** (rule 5).
- The no-teacher track drops OpenMathInstruct-2, Cosmopedia, teacher-ranking distillation and regenerated chat, and keeps generators, human text, train splits and self-play.

### 6.5 Library learning

[HYPOTHESIS, admitted late] After each S5 round, program fragments that verify correctly and are reused often are refactored into named library functions, stored as text.

- The library contains only human open-source code plus self-play output. Teacher-written code is banned.
- The library is frozen, hashed and decontaminated before every eval.
- It is retrieved, never inlined wholesale.

---

## 7. Compute plan

### 7.1 FLOPs per training token, recomputed

Mean r=4 gives 22 forward layer-passes:

- 14 of them carry gradient (coda 3 + last 2 loops × 4 = 8 + prelude 3) at 6N.
- The other 8 run forward-only at 2N.
- The LM head is 21M at 6N.

Total: 6·14·5.35M + 2·8·5.35M + 6·21M ≈ **6.6e8 FLOPs/token** [ESTIMATE]. Judge training (S3) costs about 2·22·5.35M (frozen trunk forward) + 6·5·5.35M ≈ 4e8 FLOPs/token.

### 7.2 Throughput, at a realistic MFU (critique C1)

| Platform | Peak | MFU assumed | FLOPs/week | Tokens/week |
|---|---|---|---|---|
| TPU v5e-8 (8 × 197 TFLOPS bf16 [VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e]; on Kaggle [VERIFIED: https://www.kaggle.com/discussions/product-announcements/607202]), 20 h | 1,576 TFLOPS | **10–18%** (JAX tutorial: 16–27% for a tiny GPT, 13% at 8-way DP [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html]) | 1.1–2.0e19 | **17–31B** |
| T4×2 (65 TFLOPS fp16 each [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf]), 30 h | 130 TFLOPS | 15–25% | 2.1–3.5e18 | 3.2–5.3B |
| RTX 3060 laptop, ~25 TFLOPS, ~40 h usable | ~25 | 30% | 1.1e18 | ~1.6B |

S1–S4 total about 57B tokens, which is **2–3.5 TPU-weeks**. If the TPU is unavailable (users report oversubscription [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab]), the T4-only plan takes 11–18 weeks. That is slower but still feasible: 100M × 20 tokens = 2B tokens is Chinchilla-optimal, and the plan is far past it.

### 7.3 Engineering

- **One JAX codebase.**
  - JAX supports SM 5.2+ on CUDA 12 and SM 7.5+ on CUDA 13, so the T4 (sm_75) and the 3060 (sm_86) are both covered [VERIFIED: https://docs.jax.dev/en/latest/installation.html]. **No Triton and no Pallas-GPU anywhere**, which removes the Triton 3.3+ sm_75 problem [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md].
  - The chunked delta-rule is pure `jnp`: matmuls inside `lax.scan` over chunks of 64, static shapes, sequences bucketed to 2048.
  - The core uses `lax.fori_loop` with r sampled per batch.
  - Head dimension 64 pads on 128-wide MXUs. **Week-1 benchmark:** head dimension 64 vs 128 (5 heads × 128) at d=640, 4-way DP × 2-way model sharding on v5e-8, plus the 3060. Every number in this section is then replaced with measured values.
- **T4 numerics:** fp16 matmuls; state, decay products and scan accumulators in fp32; dynamic loss scaling (no bf16 on T4 [VERIFIED: https://discuss.vllm.ai/t/no-throughput-improvement-for-quantized-qwen-2-5-7b-instruct/1901]). Data parallelism over PCIe.
- **Persistence:** checkpoint every 45 minutes to `/kaggle/working`, then push to the HF Hub (100 GB private [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits]). Sessions are chained.

### 7.4 Datastore and storage (critique C3–C5)

- **BM25 on disk.** An in-memory bm25s index over 50B tokens would need about 150–250 GB of RAM [ESTIMATE, critique C3]. v2 uses a Lucene/tantivy-class **on-disk inverted index** on the laptop SSD. On Kaggle, BM25 runs only over Wikipedia en+ko plus the textbook shards: about 31M English chunks plus Korean, roughly 10–20 GB on disk [ESTIMATE].
- **Dense side:**
  - 256-bit binary codes: 390M chunks × 32 B = **12.5 GB**, mmapped.
  - Rescoring needs **no stored int8 vectors**. The doc embedding is a weighted mean of E·P rows, so it is recomputed on CPU from the text of the top-1,000 candidates (about 1,000 × 128 lookups) [ESTIMATE, the critique C4 fix].
- **Storage cap.** v1's datastore is ≤50B tokens, about 60–70 GB zstd [ESTIMATE].
  - The full store lives on the laptop SSD. **The user must confirm about 150 GB free.** If not, the cap is 20B tokens.
  - Kaggle sessions mount a 10–15B-token high-yield subset as private datasets (quota 100–200 GB, unconfirmed).
  - The index stores document IDs that point to public HF datasets. Text is fetched **offline, before eval**, into a frozen, hashed snapshot.
  - The store grows by quality (textbooks, worked human solutions), not raw size.
- **Self-play CPU** (critique C6). 2–5B rollout tokens at about 250 tokens per program is 8–20M executions. At 20–50 executions/s on about 4 vCPU that is 2–12 days of CPU [ESTIMATE]. Mitigations:
  - A vectorized NumPy evaluator for pure-arithmetic programs.
  - Laptop CPU overnight.
  - A cache keyed by program hash.
  - Asynchronous execution decoupled from TPU decoding, so there are no host round trips inside the JAX loop.

---

## 8. Inference cost and latency per query

Per generated token: 2 × (10 + 4(r−1)) × 5.35M + 2 × 21M (head), which is ≈ **0.15 GFLOP at r=1 and 0.45 GFLOP at r=8** (corrected from v1's 0.33). The judge adds about 2 × 3 × 5.35M ≈ 0.03 GFLOP per scored token [ESTIMATE].

| Query type | Work | r | Tool calls | GPU FLOPs | Wall-clock, 3060 laptop [ESTIMATE] |
|---|---|---|---|---|---|
| Factoid (NQ) | 1 × 60 tokens + read 32 × 128 | 2 | 1 search | ~1.5e12 | 0.3–1 s (on-disk BM25 ~100–300 ms) |
| MC (MMLU/HellaSwag) | per-option retrieval (4 searches) + judge over 4 × ~200 tokens | 2 | 4 searches | ~5e11 | 0.5–1.5 s |
| GSM8K | 32 × 250 | 4 | 32–64 exec | ~2.5e12 | 2–6 s (CPU-bound) |
| MATH hard | 64 × 600 + repairs | 8–16 | ~200 | ~4e13 | 20–90 s |
| HumanEval | 16 × 400 + 3 repair rounds | 8 | ~100 test runs | ~1.2e13 | 15–60 s |
| ARC-AGI task | DSL search 1e4–1e5 candidates + test-time training | 16–32 | 1e5 grid evals | ~1e15 | 2–10 min |
| Chat turn | plan + 400 realize tokens + copy queries | 2 | 2–4 searches + ~20 phrase MIPS | ~3e11 | 1–3 s |

**Eval scheduling** (critique §6):

- Full Tier A is CPU-bound. MATH alone is about 1M tool calls, 5–14 h [ESTIMATE].
- ARC-AGI is 13–67 h for 400 tasks and runs as a **separate monthly job** on the laptop.
- Stratified dev subsets run weekly; full sets run monthly.
- FLOPs and wall-clock per query are reported next to every score.

---

## 9. Scaling story

Five axes scale independently [HYPOTHESIS, to be measured]:

1. **Core and judge parameters (0.1B → 1B → 10B).**
   - More parameters give better NL-to-program translation and a higher base pass@k, which multiplies the value of search [VERIFIED trend: https://arxiv.org/abs/2408.03314].
   - They also give a stronger judge and REALIZE. **This is the axis that has to close the F-wall and chat.**
   - Anchors: phi-1.5 at 1.3B reaches WinoGrande 73.4 [VERIFIED: https://arxiv.org/html/2309.05463], and Gemma-2-2B-it's Arena score of 1163.8 beats gpt-3.5-0125's 1132.3 [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard]. Together they suggest the F rows become winnable at about 1–3B *with* our judge and stores [ESTIMATE].
   - Cost: 1B at Chinchilla is about 1.2e20 FLOPs, roughly 6–10 TPU-weeks at the corrected MFU, so it is feasible on free quota. **10B needs about 1.2e22 FLOPs, which requires TPU Research Cloud** [VERIFIED: https://sites.research.google/trc/faq/]. Apply in month 3 with the public v1 results (the application is free; VM/GCS side costs are "generally minimal" but not zero).
2. **Datastore (≤50B → 2T tokens).** Scales the knowledge rows. MassiveDS saw no saturation up to 1.4T tokens [VERIFIED: https://arxiv.org/html/2407.12854]. The store can be swapped without retraining. Growing it is bounded by disk (C5), so beyond about 1 TB it waits for grants.
3. **Skill library (text).** It grows with every self-play round and transfers to a bigger core unchanged.
4. **Tools.** Each new verifier or simulator opens a new domain at 0 parameters.
5. **Test-time compute.** N, r and search depth grow with hardware, not parameters.

### Measuring predictability (critique S1)

- Training grid: **{5M, 10M, 20M, 50M, 100M}** (20× span). 3 seeds at ≤20M, 2 seeds at 50M, 1 seed plus the main run at 100M. That is about 12 training runs, roughly 1.5 TPU-weeks in total [ESTIMATE].
- r ∈ {1, 4, 16}, N ∈ {1, 16} and store size ∈ {5B, 50B} are **inference-time knobs**, swept on every checkpoint at no training cost.
- Fit: score ≈ f(params, r, N, store) per row family (R, T, Q, F).
- **The 1B decision** rests on the F-family slope. If the 5M→100M slope, extrapolated with its seed-variance band, reaches the HellaSwag and WinoGrande gates by ≤3B, we take the 1B branch. If it does not, the judge mechanism itself is wrong, and we say so.

### North star

A scientific-discovery agent is a compiler over experiments:

| Discovery concept | COMPILER-0 component |
|---|---|
| Hypothesis | Program |
| Simulator / assay | Executor |
| Result | Verifier |
| Lab notebook | Datastore |
| Validated protocols | Library |
| Expert taste in ranking hypotheses | Judge |

Only the tool registry changes; the architecture stays the same. A >100M structure predictor would be a disclosed north-star-track tool, never part of the 0.1B milestone.

**Biosafety (hard stop):**

- The tool registry and datastore exclude pathogen-enhancement, toxin-synthesis and agent-weaponization content.
- PLAN carries a refusal policy, enforced by a non-learned blocklist plus a review gate.
- Any wet-lab-actionable output needs human expert review.
- Bio data and models are released only after dual-use review.
- This is never traded off against capability.

---

## 10. First three experiments (go/kill), reordered so the kill-risk is tested first

**E0, week 1 (infrastructure, not an experiment).** Measure:

- GDN tokens/s and MFU on v5e-8, T4×2 and the 3060;
- Korean/English/code tokenizer fertility;
- on-disk BM25 latency;
- `df -h` / `free -g` / quota on Kaggle.

Then replace every [ESTIMATE] in §7.

**E1: "F-wall and chat probe" (weeks 2–4, moved up from weeks 6–12; critique B8, §8.4).**

- 5M/10M/20M trunks plus judge towers, trained on the S3 recipe (train splits + teacher-ranking distillation + contrastive triples). The trunk is a quick 2B-token S1-mini.
- Evaluated generatively on HellaSwag and WinoGrande validation, **with the store off** (wikihow/ActivityNet blocked regardless).
- Chat mechanism probe: a 20M copy-realize system vs a 20M plain-generation twin, on 80 writing/roleplay/humanities prompts, judged pairwise by the two open judges.
- **GO** if, at 20M: HellaSwag ≥ 45 and WinoGrande ≥ 56; the 5→20M slope extrapolates to ≥ 65 on both at 100M; and copy-realize beats the plain twin by ≥ 15 points of pairwise win rate.
- **PIVOT** if the slope is flat (< 2 points per doubling). Re-balance the judge data (more distillation, fewer triples) and re-run once.
- **ESCALATE (week 8, at the 100M judge checkpoint)** if HellaSwag < 65 or WinoGrande < 65, or if the chat pairwise win rate vs stored GPT-3.5 MT-Bench answers is < 20%. **I then tell the user plainly:** the "all rows at 0.1B" goal is not reachable by this design. I present the measured slope and propose the 1B branch (free quota) and the TRC application. The decision is the user's.

**E2: "Does compiling beat memorizing at 30M?" (weeks 2–6, in parallel).**

- A 30M GDN compiler (S1-mini + S2-mini, 8B tokens) on GSM8K-style IR vs a 30M pure-CoT twin with the same data and budget.
- No TinyGSM data, since it is GPT-3.5 output. Problems come from OpenMathInstruct-2 (execution-filtered) plus our own generators.
- **GO** if, on a decontaminated GSM8K-like dev split: IR maj@16 ≥ 40% and ≥ 2× the CoT twin, AND r=8 beats r=1 by ≥ 5 points.
- **KILL/PIVOT** if IR maj@16 < 20%, or recurrence adds < 2 points. The pivot drops the looped core and spends those parameters on width (or on a second unique coda layer).

**E3: "Retrieval-reading at 100M" (weeks 5–9).**

- The 100M trunk with the ≤50B human-written store, on NQ-open dev and KMMLU dev, scored with and without the store.
- **GO** if NQ EM ≥ 38 and KMMLU ≥ 40.
- **KILL the E·P pooled doc embedder** if gold-passage recall@32 on NQ dev is < 70%. Spend 4M of headroom on a dedicated doc table, or move to BM25-only plus trunk reranking of the top-100.

---

## 11. Top risks

1. **The F-wall and chat are real.** P(losing at least one of HellaSwag, WinoGrande or chat at v1) > 95% [ESTIMATE]. Mitigation: E1 at week 2, the escalation at week 8, and the 1B/TRC branch. **This risk alone decides whether "never lose" is achievable at 0.1B, and it is pre-registered with the user.**
2. **The judge may only learn the train splits.** Distillation plus train splits can teach dataset artifacts. Mitigation: report the no-train-split Tier C score, use KoBEST and a fresh private commonsense set as transfer checks, and run adversarial filtering on our own synthetic items.
3. **Contamination makes the win fake.** Retrieval systems copy verbatim [VERIFIED: https://arxiv.org/html/2112.04426], and paraphrase leaks evade n-gram filters [VERIFIED: https://arxiv.org/pdf/2311.04850v1]. Mitigation: the three-layer scrub, whole-domain blocklists, with/without-store reporting, and the private post-cutoff set as the headline.
4. **Autoformalization ceiling.** If NL → program fails, the tools are useless. AlphaProof failed both combinatorics problems [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/]. Mitigation: grammar-constrained decoding, repair loops, and (late) the library.
5. **The MMLU and ARC-C knowledge-reasoning gap.** Retrieval gives only about +3–4 points on reasoning-heavy knowledge tasks [VERIFIED: https://arxiv.org/html/2407.12854]. Per-option evidence plus the judge is an unproven bridge. These rows are expected losses at 0.1B.
6. **Recurrent recall weakness.** Mitigation: short chunks, re-retrieval and the delta-rule state. Titans was deleted, so there is no extra memory to lean on.
7. **Throughput.** If the measured MFU is < 10%, S1–S4 stretch beyond 4 TPU-weeks. TPU availability is not guaranteed. Mitigation: E0 measures first, and the T4 fallback is 11–18 weeks.
8. **CPU-bound self-play and eval.** Mitigation: vectorized evaluators, caching, the laptop CPU, and a reduced S5.
9. **Verifier gaming in RLVR.** Mitigation: held-out property tests, never running benchmark hidden tests, and eval judges never used as a reward.
10. **Licensing and ToS drift.** Teacher licenses, BY-SA content, and Korean datasets with unknown licenses (35.3% of Korean HF datasets [VERIFIED: https://arxiv.org/pdf/2507.04329]). Mitigation: a per-sample provenance manifest, the subset audits, and the no-teacher track.
11. **Protocol variance.** GPT-3.5 targets vary by up to 20 points between sources. We gate on the max and match metrics, but a protocol mismatch can still flip a row.
12. **Solo scope.** Even staged, this is large. Mitigation: one JAX codebase, the v0.1 minimal core, and the +2-point admission rule.
13. **Biosafety** on the north-star path: a hard stop, governed as in §9.

---

*Summary of the bet:* procedures in weights, facts in a human-written store, truth from executors, plausibility from a bidirectional judge, fluency borrowed from humans. COMPILER-0 v2 has a credible path to beating GPT-3.5 on about 2 to 6 of about 20 rows at 93.1M parameters on zero budget. It has **no demonstrated path** on the commonsense and open-chat rows at that size. v2 tests that wall in week 2 and puts a measured answer in front of the user by week 8.

---

## Changelog (v1 → v2)

**Fatal issues fixed**

- **B1, MMLU misread:** Atlas-770M's 5-shot score is 38.9, not 56.3. MMLU is now estimated at 38–52 with confidence 0.01. Per-option evidence retrieval and judge scoring were added.
- **F3, teacher text at inference:** new rule 5. Every inference store is human-written or self-generated only. Teacher text is training-only. Teacher code is banned from the library.
- **B8, chat:** the estimate is lowered to MT-Bench 3–5. Chat is pre-registered as the escalation row, and its probe moves to week 2.
- **§7, solo scope:** one JAX codebase (sm_75 support verified at https://docs.jax.dev/en/latest/installation.html), no Triton or Pallas-GPU, a v0.1 minimal core, and a +2-point admission rule for every extra component.
- **Overall-win honesty:** P(all Tier A+B) ≈ 0 at v1 is stated outright. The critique's 1e-13 product is too precise given the row correlation, but its conclusion is right.

**Major issues fixed**

- **Confidences:** every one is re-derived from the gate (target + CI). GSM8K 0.45 → 0.15; MATH 0.5 → 0.1; code 0.15 → 0.02; ARC-C 0.1 → 0.01; BBH 0.55 → 0.3; ARC-AGI gate raised to about 13.6 (text-davinci-003 10.6%); TriviaQA metric-matched to 82.9 EM.
- **F1:** a benchmark-train-split rule (one mix, one checkpoint, disclosed, plus a no-split Tier C score).
- **F2:** LoRA adapters (6.55M) and Titans (1.64M) deleted, and the separate verifier merged. These fund a 17.4M bidirectional GDN judge tower trained by teacher-ranking distillation, train splits and contrastive triples. Pushback recorded: even this route has a verified prior ceiling (WinoGrande SOTA 79.1 in 2019, https://arxiv.org/abs/1907.10641) below the 83.7 gate, so WinoGrande stays an expected loss.
- **F5:** whole-domain blocklists (wikihow.com, ActivityNet, Korean exam sites); with/without-store reporting; generator manifest frozen and 5 BBH families held out by seeded draw; two judges plus 100 blind human pairs; judges never used as a reward.
- **§4, pointer head:** it was attention over the sequence. It is replaced by an external phrase index (MIPS, top-k ≤ 64, softmax only over external items, reported separately).
- **C1:** TPU MFU lowered to 10–18%, giving 17–31B tokens/week. A week-1 benchmark replaces the estimates.
- **C3:** on-disk BM25. Kaggle BM25 covers Wikipedia plus textbooks only.
- **C5:** datastore capped at ≤50B tokens. The full store lives on the laptop SSD (user to confirm about 150 GB free); Kaggle mounts a 10–15B subset; frozen hashed snapshot.
- **C6:** S5 cut to 2–5B rollout tokens, with whole-program generation and asynchronous CPU execution.
- **B4:** MATH anchored to the 4.6 MATH500 score of the best ≤150M model.
- **S1:** scaling grid widened to 5M–100M with seeds. Inference knobs are swept for free.
- **expand_v trap:** the parameter count is scripted per layer (5,347,200), with expand_v=1 pinned. An expand_v=2 build would be 109.1M, so FFN hidden 1300 is the fallback. A CI assertion runs against the real module.

**Minor issues fixed**

- **C2:** coherent TRM-style detached deep supervision. Training FLOPs corrected to about 6.6e8/token; inference at r=8 corrected to about 0.45 GFLOP/token.
- **C4:** no stored int8 vectors. The static doc side is recomputed on CPU for rescoring.
- **B5:** BBH is about half executable, not "most."
- **B9:** jamo splitting dropped. The 32k unigram vocab has about 7k Korean entries, with fertility measured first.
- **F4:** unigram scores counted as parameters. The Korean pre-splitter is pure rules, not a CRF.
- **F6:** citations are mandatory for copied BY-SA spans.
- **§6:** abstain disabled in eval. ARC-AGI runs as a separate monthly job.
- **S3:** the library is frozen, decontaminated and retrieved, not inlined.
- **Lean, KG, Z3, geometry engine:** staged or moved to the north-star track.
- **Teacher-free math:** TinyGSM is excluded from E2's data.

**Parameter total:** 91,488,857 → **93,134,917** (script-verified; 6,865,083 headroom reserved for two experiment-gated uses).
