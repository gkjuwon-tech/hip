# Reasoning-Only GPT-3.5 Scoreboard (post-pivot, "aphasic genius")

Analyst topic: reason_baseline. Date: 2026-10-19. Labels: [VERIFIED: url] means the number was read from that source in this session or in `research/baseline.md`. [ESTIMATE] means my own calculation or judgement. [HYPOTHESIS] means an untested claim. [UNVERIFIED] means the number appears only in a secondary source.

## 0. TL;DR

1. **Every reasoning benchmark below can be scored from formal output alone**: a number, a choice letter, a span copied from the input, code, a plan, or a proof. None needs natural-language generation, so the renderer never touches scoring. The core still has to **read** natural language. Comprehension is the real input bottleneck.
2. **Target = the maximum reported GPT-3.5 score** across snapshots (0301/0613/1106/0125), text-davinci-003, and prompting methods, *including tool-augmented GPT-3.5* such as Logic-LM, which uses a solver, and Tree-of-Thoughts (ToT) search. The mid-boss is GPT-3.5 at its best.
3. **Definition of "아득히" (by a wide margin)**: on each gated row, score ≥ **G = T + max(10, (100−T)/3)**. That means beating the target by 10 points or closing a third of the remaining error, whichever is larger. The 95% CI lower bound must also exceed T, and the robust twin of the row (§4) must clear its own G.
4. **Rows where GPT-3.5 is weakest, and where a <100M search/verify core is best placed** [ESTIMATE]: Game of 24 (GPT-3.5 at most 19%), Blocksworld (6.1%), ARC-AGI-1 (10.6%), LiveCodeBench (22.6%), AR-LSAT (26.4%), GPQA (30.6%), KMMLU-Math (30.3%). **Hardest rows**: HumanEval+/MBPP (70–83), GSM8K (78.1), PrOntoQA (85.0) and other high-T rows, because the margin rule then demands 88–95.
5. **Contamination cuts both ways.** GPT-3.5 likely saw GSM8K/HumanEval [ESTIMATE] and loses 16.9% relative on GSM-Plus [VERIFIED: https://arxiv.org/html/2402.19255v2]. Gate on fresh or procedural twins.

## 1. Raw evidence: GPT-3.5 reasoning scores

Abbreviations: 3.5 = gpt-3.5-turbo (snapshot given when known); d003 = text-davinci-003.

| Benchmark | Score(s) | Model / protocol | Source |
|---|---|---|---|
| GSM8K | 57.1 (5-shot CoT); **78.1** (1106, 0-shot CoT); 74.9 (CoT-Hub); 75.8 (Faithful-CoT repo); 73.62 (GSM-Plus paper) | 3.5 | [VERIFIED: https://arxiv.org/abs/2303.08774], [VERIFIED: https://arxiv.org/abs/2404.14219], [VERIFIED: https://github.com/FranxYao/chain-of-thought-hub], [VERIFIED: https://github.com/veronica320/Faithful-COT], [VERIFIED: https://arxiv.org/html/2402.19255v2] |
| GSM-Plus (10,552 perturbed items) | **61.19** (PDR 16.88%) | 3.5 | [VERIFIED: https://arxiv.org/html/2402.19255v2] (Table 4, via search snippet) |
| GSM-Symbolic | no GPT-3.5 row found; abstract: adding one irrelevant clause drops models by "up to 65%" | — | [VERIFIED: https://arxiv.org/abs/2410.05229] |
| MATH | **34.1** (4-shot Minerva prompt, Nov-2023 API); 43.1 on OpenAI's GPT-4o-mini page (openai.com returned 403) | 3.5 | [VERIFIED: https://arxiv.org/abs/2312.11805]; 43.1 [UNVERIFIED] |
| SVAMP / ASDiv / MultiArith / AQuA | **83.0 / 81.7 / 95.3 / 53.5** (greedy, Faithful-CoT prompt, Apr-2023) | 3.5 | [VERIFIED: https://github.com/veronica320/Faithful-COT] |
| BBH | 66.6 (3-shot CoT); 68.32 (1106, 0-shot); **70.1** (3.5, CoT-Hub); **70.7** (d003, CoT-Hub) | 3.5 / d003 | [VERIFIED: https://arxiv.org/abs/2312.11805], [VERIFIED: https://arxiv.org/abs/2404.14219], [VERIFIED: https://github.com/FranxYao/chain-of-thought-hub] |
| BBH Logical Deduction | 40.00 standard / 42.33 CoT / **65.67** Logic-LM (3.5); d003 62.00 | Logic-LM | [VERIFIED: https://arxiv.org/html/2305.12295v2] |
| ARC-AGI-1 | no official 3.5 score; d003 85/800 = **10.6%** (public train+eval, ASCII); GPT-4o 5% | | [VERIFIED: https://arxiv.org/abs/2307.04721], [VERIFIED: https://arcprize.org/blog/oai-o3-pub-breakthrough] |
| ARC-AGI-2 | none reported; ≈0 [ESTIMATE]. Reference: TRM-7M 7.8 | | [VERIFIED: https://arxiv.org/abs/2510.04871] |
| HumanEval / HumanEval+ | 48.1 (GPT-4 report); **76.8 / 70.7** (EvalPlus, greedy) | 3.5 | [VERIFIED: https://arxiv.org/abs/2303.08774], [VERIFIED: https://evalplus.github.io/leaderboard.html] |
| MBPP / MBPP+ | **82.5 / 69.7** (EvalPlus); 77.8 (1106, 3-shot) | 3.5 | [VERIFIED: https://evalplus.github.io/leaderboard.html], [VERIFIED: https://arxiv.org/abs/2404.14219] |
| Natural2Code (held out) | 62.3 | 3.5 | [VERIFIED: https://arxiv.org/abs/2312.11805] |
| LiveCodeBench, code generation (problems since Sep-2023) | 0125: **22.6** (easy 56.8 / medium 10.8 / hard 0.1); 0301: 20.8 | pass@1, 10 samples, T=0.2 | [VERIFIED: https://arxiv.org/html/2403.07974] App. D.2 |
| LCB self-repair / test-output prediction / execution | 23.9 / 35.4 (0125); execution 33.9, 34.8 with CoT (0301) | | same |
| CRUXEval-I / -O | 49.2 / 50.0; with CoT 49.1 / **63.3** (pass@1, 26 Oct 2023) | 3.5 | [VERIFIED: https://arxiv.org/pdf/2401.03065] Tables 2–3 |
| DROP (F1) | **64.1** (3-shot); 70.2 on the GPT-4o-mini page [UNVERIFIED] | 3.5 | [VERIFIED: https://arxiv.org/abs/2303.08774] |
| GPQA (Main 448 / Diamond 198 / Extended 546) | 0-shot **29.8 / 30.6 / 30.0**; few-shot CoT 28.0 / 29.6 / 28.2; Phi-3 report: 29.0 | gpt-3.5-turbo-16k | [VERIFIED: https://arxiv.org/html/2311.12022] Table 6; [VERIFIED: https://arxiv.org/abs/2404.14219] |
| MedQA | **63.4** (1106, 2-shot) | 3.5 | [VERIFIED: https://arxiv.org/abs/2404.14219] |
| StrategyQA | 51.5 (Faithful-CoT prompt); 70.3 with CoT (AutoReason paper, search snippet only) | 3.5 | [VERIFIED: https://github.com/veronica320/Faithful-COT]; 70.3 [UNVERIFIED: https://arxiv.org/pdf/2412.06975] |
| LogiQA 2.0 test / zh / OOD | **52.37 / 53.18 / 38.44** | 3.5 API | [VERIFIED: https://ar5iv.labs.arxiv.org/html/2304.03439] |
| ReClor dev / AR-LSAT test | **57.38 / 20.42** | 3.5 API | same |
| FOLIO | 45.09 standard / 57.35 CoT / **62.74** Logic-LM (3.5); d003 54.60 / 57.84 / 61.27 | | [VERIFIED: https://arxiv.org/html/2305.12295v2] |
| ProofWriter (depth-5 subset) | 35.50 / 49.17 / 58.33 (3.5); d003 Logic-LM **71.45** | | same |
| PrOntoQA (hardest subset) | 47.40 / 67.80 / 61.00 (3.5); d003 CoT 83.00, Logic-LM **85.00** | | same |
| AR-LSAT | 3.5 Logic-LM **26.41**; d003 25.54 | | same |
| MuSR (Murder Mysteries 250 / Object Placement 256 / Team Allocation 250) | best: **70.0** (1-shot CoT+) / **58.2** / **78.8** (few-shot CoT+); 0-shot CoT 56.0 / 48.4 / 46.4 | 3.5 | [VERIFIED: https://ar5iv.labs.arxiv.org/html/2310.16049] Table 7 |
| miniF2F (Isabelle, 244 valid / 244 test) | LEGO-Prover with ChatGPT, 100 attempts: **57.0 / 50.0**; Subgoal-Learning (ChatGPT) 48.0 / 45.5; Thor (700M, no LLM) 28.3 / 29.9 | mix of 3.5 snapshots; informal proofs come from GPT-4 or humans | [VERIFIED: https://arxiv.org/pdf/2310.00656] Table 1 |
| Game of 24 (100 hard games) | IO 6% / CoT 3% / **ToT 19%** | 3.5 | [VERIFIED: https://arxiv.org/pdf/2305.10601] App. B.2, Table 5 |
| Blocksworld (PlanBench, natural language, 600 instances) | **37/600 = 6.1%** one-shot, 1.3% zero-shot; Mystery BW 0/600; Logistics 1/200 | "GPT-3.5 (ChatGPT)" | [VERIFIED: https://arxiv.org/html/2305.15771v2] Table 1 |
| KMMLU Math (5-shot Direct) | **30.3** (4-choice, chance 25); KMMLU STEM 44.64 | 3.5 | [VERIFIED: https://arxiv.org/html/2402.11548] Tables 3, 20 |
| AIME / AMC | no primary GPT-3.5 number found. ≈0–5% on AIME [ESTIMATE] | | — |
| CLRS, Sudoku, Korean CSAT math | no GPT-3.5 number found (Sudoku-Extreme: DeepSeek-R1 0.0, TRM 87.4) | | [VERIFIED: https://arxiv.org/abs/2510.04871] |

**Protocol warning** [ESTIMATE]: protocol alone moves GPT-3.5 up to 21 points (GSM8K 57→78, HumanEval 48→77). We beat the maximum and report under that row's protocol.

## 2. Benchmark properties: answer format, NL-free scoring, knowledge need

Key: NLG-free = scorable with no natural-language generation. K = needs world knowledge (retrieval). The core must always read natural-language input itself.

| Benchmark | Answer format | NLG-free? | K? | Notes |
|---|---|---|---|---|
| GSM8K, GSM-Plus, SVAMP, ASDiv | number | yes | no | Program-of-thought plus executor. The bottleneck is reading the problem |
| MATH, AIME/AMC | LaTeX expression / integer | yes (SymPy equivalence check) | light (theorems) | A theorem/lemma datastore helps |
| BBH (27 tasks) | choice / short token / number | yes; a few tasks (word sorting, Dyck) emit token lists, which count as structured output, not prose | ~5 tasks light (sports, date, movie) | 23/27 tasks are closed-world [ESTIMATE] |
| ARC-AGI-1/2 | grid | yes | no | Pure abstraction |
| HumanEval(+), MBPP(+), LiveCodeBench | code | yes (unit tests) | API docs | Code is a formal output, so the core writes it |
| CRUXEval-I/O | Python literal | yes | no | Execution reasoning |
| DROP | number / date / **span** | yes, if spans are *copied* by pointer (the 0-parameter suffix-automaton copy) | no | F1 scoring. A copy is not generation |
| GPQA, MedQA | choice | yes | **yes** | Retrieval required. GPQA is "Google-proof" |
| StrategyQA | yes/no | yes | **yes** | Implicit multi-hop facts |
| LogiQA 2.0, ReClor, AR-LSAT | choice | yes | no | Dense natural-language comprehension |
| FOLIO, ProofWriter, PrOntoQA | True/False/Unknown | yes | no | Autoformalize to FOL, then solve (the core does the autoformalization) |
| MuSR | choice | yes | commonsense | ~1,000-word narratives. The hardest reading task here |
| miniF2F | formal proof (Isabelle/Lean) | yes (proof checker) | lemma library | Fully machine-verified |
| Game of 24 | arithmetic expression | yes | no | Exhaustive search solves it, so treat as a sanity row |
| Blocksworld / Mystery BW | action plan | yes (VAL validator) | no | A classical planner solves it if the core formalizes correctly |
| CLRS-Text, Sudoku | trace / grid | yes | no | Procedurally generated, so contamination-proof |
| KMMLU-Math, CSAT math | choice / integer | yes | no | Requires reading Korean |

Conclusion [ESTIMATE]: every row is NLG-free; ~4 need retrieval. The renderer blindfold costs nothing on scoring.

## 3. The scoreboard

G = T + max(10, (100−T)/3), rounded. N = item count. CI = binomial 95% half-width at p = G [ESTIMATE]. One checkpoint and one config for every row. No benchmark-identity routing.

### Tier A — gated, closed-world (no retrieval)

| # | Row (gated version) | T (max GPT-3.5) | **G (아득히)** | N | ±CI at G |
|---|---|---|---|---|---|
| A1 | GSM8K, 0-shot | 78.1 | **88.1** | 1,319 | 1.7 |
| A2 | GSM-Plus (robust twin of A1) | 61.2 | **74.1** | 10,552 | 0.8 |
| A3 | MATH (MATH500 for dev, full 5k for gate) | 43.1 (conservative; verified value 34.1) | **62.1** | 5,000 | 1.3 |
| A4 | SVAMP | 83.0 | **93.0** | 1,000 [ESTIMATE] | 1.6 |
| A5 | BBH, all 27 tasks, 3-shot | 70.7 (d003) | **80.7** | 6,511 | 1.0 |
| A6 | ARC-AGI-1 public eval, pass@2 | 10.6 | **40.4** | 400 | 4.8 |
| A7 | HumanEval+ | 70.7 | **80.7** | 164 | 6.0 |
| A8 | MBPP+ | 69.7 | **79.8** | 378 | 4.0 |
| A9 | LiveCodeBench, problems after our data freeze | 22.6 | **48.4** | ≥300 [ESTIMATE] | 5.7 |
| A10 | CRUXEval-O | 63.3 | **75.6** | 800 | 3.0 |
| A11 | DROP F1 | 70.2 (conservative; verified value 64.1) | **80.2** | 9,536 | 0.8 |
| A12 | FOLIO / ProofWriter / PrOntoQA (Logic-LM splits) | 62.7 / 71.5 / 85.0 | **75.2 / 81.5 / 95.0** | — | — |
| A13 | LogiQA 2.0 / ReClor | 52.4 / 57.4 | **68.2 / 71.6** | — | — |
| A14 | MuSR MM / OP / TA | 70.0 / 58.2 / 78.8 | **80.0 / 72.1 / 88.8** | 250/256/250 | 5.0/5.5/3.9 |
| A15 | Game of 24, 100 hard games plus a fresh set | 19 | **46** (realistically ~100) | 100 | 9.8 |
| A16 | Blocksworld, natural language, 600 instances | 6.1 | **37.4** | 600 | 3.9 |
| A17 | KMMLU-Math | 30.3 | **53.5** | — | — |

### Tier B — gated, retrieval allowed (frozen offline datastore, decontaminated)

| # | Row | T | **G** | N |
|---|---|---|---|---|
| B1 | GPQA-Diamond, 0-shot | 30.6 | **53.7** | 198 (±7) |
| B2 | MedQA (USMLE, 4-option) | 63.4 | **75.6** | 1,273 [ESTIMATE] |
| B3 | StrategyQA | 70.3 [UNVERIFIED] | **80.3** | — |

### Tier C — north-star, reported but not gated

| Row | GPT-3.5 reference | Why it matters |
|---|---|---|
| ARC-AGI-2, via Kaggle ARC Prize private set | ≈0 [ESTIMATE]; TRM 7.8 | Hard abstraction; the private set rules out contamination |
| miniF2F-test, Isabelle or Lean, 100 attempts, **no human or GPT-4 informal proofs** | 50.0 (GPT-3.5 *with* GPT-4/human proof sketches) | Machine-checked proof. It is the strongest RSI substrate (§5) |
| Mystery Blocksworld; CLRS-Text; Sudoku-Extreme | 0/600; none; none | Meaning-stripped planning, algorithmic length generalization |
| AR-LSAT | 26.4 | Constraint satisfaction from natural language |
| AIME 2025/2026 (post-freeze) | ≈0–5 [ESTIMATE] | Fresh competition math |
| LCB-hard, Natural2Code-style private set | 0.1 / 62.3 | Code headroom |
| Korean CSAT math (exam released after freeze) | none | Fresh Korean reasoning |

### Overall win condition ("아득히", concrete)

1. **Every Tier A and Tier B row ≥ G.** On rows with N < 500, the 95% CI lower bound must also be > T.
2. **Aggregate**: the geometric mean of (our score / T) over Tier A+B is ≥ 1.30 [ESTIMATE threshold].
3. **Robustness gate**: our relative drop from GSM8K to GSM-Plus is ≤ GPT-3.5's 16.88%. Our LiveCodeBench and private-set scores are ≥ G. **A win that exists only on the public sets does not count.**
4. **Compute disclosure**: report each row at two budgets, 1 sample and the full search budget, with average T4-seconds per item. "아득히" must hold at a budget of ≤ 60 T4-seconds per item on average [ESTIMATE cap; keeps one full benchmark pass under ~22 GPU-hours for GSM8K]. Unbounded search does not count.
5. **Parameter gate**: CI asserts < 100M executed trainable weights. The renderer is excluded. A renderer-alone control is published per row: the renderer gets the blindfolded core output with answers masked, or the raw question for the "renderer cheat ceiling".

## 4. Anti-contamination protocol

1. **Commit-reveal.** Before training starts, publish SHA-256 hashes of every private eval file: fresh math (200), code (100), Korean math (100), and logic (200), written or generated after the data freeze. Reveal them at evaluation time.
2. **Procedural twins for every public row** [HYPOTHESIS that they track the originals]:
   - GSM8K → GSM-Plus [VERIFIED exists: https://arxiv.org/html/2402.19255v2], plus our own GSM-Symbolic-style templates [VERIFIED method: https://arxiv.org/abs/2410.05229] with fresh seeds and NoOp distractor clauses.
   - MATH → numeric re-parametrization of MATH500, with the answer recomputed by SymPy. Unsolvable re-parametrizations are dropped.
   - BBH → regenerate the algorithmic tasks (Dyck, boolean expressions, multistep arithmetic, word sorting, logical deduction, object tracking) from their generators with new seeds.
   - ARC → re-arc-style generators for *training only*. Gate on the public eval set; the north star is the Kaggle private set, where submissions are free.
   - Code → LiveCodeBench contest-date window strictly after the freeze, plus EvalPlus extra tests.
   - Logic → PrOntoQA/ProofWriter generators with fresh fictional ontologies. MuSR's neurosymbolic generator gets new seeds [VERIFIED generator described: https://ar5iv.labs.arxiv.org/html/2310.16049].
   - Planning → the PlanBench instance generator with new seeds and new Mystery obfuscations.
   - Game of 24 → all fresh 4-tuples, hard-ranked by solver search depth.
3. **13-gram filter** on training data, the datastore, and the skill/program library against every eval set and its mirrors (HF dumps, GitHub copies). Publish the overlap report. Plant canary strings in the private sets.
4. **Data provenance ban**: no GPT-3.5/GPT-4-generated training data. This **excludes TinyGSM** (12.3M GPT-3.5-written problems) and any data distilled from OpenAI models. Allowed: human-written public data with a compatible license, our own core's verified self-play, and open-weight-model data only if its license permits and it is disclosed [ESTIMATE that this fits the fair-play spirit; the user should confirm].
5. **Self-generated data is verified only by executors, checkers, or solvers**, never by eval labels. The eval sets are physically absent from the training container (hash check at boot).
6. **GPT-3.5 contamination asymmetry**: GPT-3.5 likely saw GSM8K and HumanEval [ESTIMATE]. The fresh-twin rows therefore measure the real gap. Report both the public and the twin numbers side by side.

## 5. Scaling ladder and the RSI hook

The user's actual goal is recursive self-improvement (RSI). Every row above has a **mechanical verifier**: unit tests, a SymPy check, VAL, an Isabelle/Lean checker, a grid match, a FOL solver. That makes each row a source of **verified self-generated data**. Generate problems, attempt them, keep only the verified solutions, add them to the skill library and the training set, and repeat. The verifier, not a teacher, sets the ceiling, so the student can pass the teacher; the ceiling becomes verifier coverage [HYPOTHESIS]. AlphaZero-style self-play is the closed-game existence proof; math, proof and code have free checkers.

Measure RSI per generation g. Each item below is [HYPOTHESIS] for the metric and [ESTIMATE] for the threshold:

- **Δ_fresh(g)**: score gain on the commit-revealed private sets between generations at fixed params and fixed budget. **RSI is "on" if Δ_fresh > 0 for ≥ 3 consecutive generations with no new human data.**
- **Solve-rate frontier**: the hardest procedurally generated difficulty level solved at ≥ 50%, for Game-of-24 depth, Blocksworld block count, CLRS length, and MATH level. It should rise monotonically.
- **Library compression**: average program length on held-out solutions should fall as the library grows (LEGO-Prover: 50.4 with skill library vs 47.1 without [VERIFIED: https://arxiv.org/pdf/2310.00656]).
- **Scaling axes to plot**: params (0.1B → 1B), test-time samples (1 → 1,000), library size, verified-data volume, datastore size. Each is a log-x curve on the Tier A twins. "Predictably smarter" means a fitted power law with R² ≥ 0.9 on ≥ 4 points.

## 6. Walls and routes around them [ESTIMATE / HYPOTHESIS]

| Wall | Why | Route |
|---|---|---|
| High-T rows (PrOntoQA 85 → 95, SVAMP 83 → 93, GSM8K 78 → 88) | The last 10 points are reading errors, not reasoning errors | Formalize, execute, then a bidirectional judge reranks n candidates. Test-time *self-consistency over formalizations* |
| MuSR / LogiQA / ReClor | Long dense text, <100M encoder | No prose generation frees ~30–50% of params for the reader [HYPOTHESIS] |
| GPQA 53.7 | Graduate knowledge; "Google-proof" | Textbook-scale datastore plus multi-hop retrieval. Probably the last row to fall |
| HumanEval+ 80.7 | 164 items, so the CI is ±6 | Generate → self-test → repair loop. Gate on LiveCodeBench to avoid overfitting a small set |
| Planning / 24 / Sudoku | Trivial for a solver *once formalized* | Allowed only if the core emits the formal problem itself; report formalization accuracy separately |
| Bio use of the GPQA/MedQA engine | Dual use | Hard stop: no pathogen/toxin tasks in self-play generators; biosafety-filtered datastore |

## 7. Gaps

- No primary GPT-3.5 numbers found for AIME/AMC, CLRS, Sudoku, GSM-Symbolic, Korean CSAT math, or BBH per-task (beyond Logical Deduction; Faithful-CoT repo also has Date 73.5, Sports 52.3). The GPT-4o-mini page values (MATH 43.1, DROP 70.2) were blocked (403) and are used only as conservative targets.
- StrategyQA 70.3 and GSM-Plus Table 4 came from search snippets; N for SVAMP, MedQA, KMMLU-Math and Logic-LM splits not re-verified.
