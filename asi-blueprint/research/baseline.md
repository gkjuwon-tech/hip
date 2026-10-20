# Baseline: The GPT-3.5 Scoreboard (the "mid-boss")

Analyst topic: baseline. Date: 2026-10-19. Labels: [VERIFIED: url] = number read from the primary source in this session; [ESTIMATE] = my calculation or judgement; [HYPOTHESIS] = untested claim.

## 0. TL;DR

1. "GPT-3.5" is not one model. It is at least 4 API snapshots (0301/0314, 0613, 1106, 0125), and the reported numbers vary a lot between sources: HumanEval is 48.1 in the GPT-4 report but 76.8 on EvalPlus, and GSM8K is 57.1 vs 78.1. **Rule: the target is the MAX reported score across snapshots and sources.** The mid-boss is the strongest GPT-3.5 on record, not the weakest.
2. OpenAI simple-evals never published a gpt-3.5 row (checked every README revision, 2024-04 to 2024-07) [VERIFIED: https://github.com/openai/simple-evals]. The sources used are the GPT-4, Gemini and Phi-3 reports, EvalPlus, LMArena, AlpacaEval, KMMLU and HAE-RAE.
3. Zero budget means we cannot call the GPT-3.5 API to re-measure it. Every target below is a published number. Chat benchmarks are compared head-to-head against **stored public GPT-3.5 outputs**, judged by a local judge.
4. Hardest walls for a <100M system: HellaSwag (85.5), WinoGrande (81.6) and open-ended chat (MT-Bench 8.39, Arena). A retriever or a tool cannot answer these. The best published sub-150M model scores 42.1 on HellaSwag and 51.3 on WinoGrande [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct].
5. Easiest wins: NQ (38.6 EM) and ARC-AGI (GPT-3.5 ≈ 0–5%). Retrieval QA with a 220M reader already reached 48.2 EM on NQ in 2020 [VERIFIED: https://arxiv.org/abs/2007.01282], and a 7M recursive model scores 45% on ARC-AGI-1 [VERIFIED: https://arxiv.org/abs/2510.04871].

## 1. Reported GPT-3.5 scores (raw evidence)

| Benchmark | Score | Snapshot | Protocol | Source |
|---|---|---|---|---|
| MMLU | 70.0 | 2023 (unspecified) | 5-shot | [VERIFIED: https://arxiv.org/abs/2303.08774] Table 2 |
| MMLU | 71.4 | 1106 | 5-shot, MS internal pipeline, temp 0 | [VERIFIED: https://arxiv.org/abs/2404.14219] |
| HellaSwag | 85.5 | 2023 | 10-shot | [VERIFIED: https://arxiv.org/abs/2303.08774] |
| HellaSwag | 78.8 | 1106 | 5-shot | [VERIFIED: https://arxiv.org/abs/2404.14219] |
| ARC-Challenge | 85.2 | 2023 | 25-shot | [VERIFIED: https://arxiv.org/abs/2303.08774] |
| ARC-Challenge | 87.4 | 1106 | 10-shot | [VERIFIED: https://arxiv.org/abs/2404.14219] |
| WinoGrande | 81.6 | 2023 | 5-shot | [VERIFIED: https://arxiv.org/abs/2303.08774] |
| WinoGrande | 68.8 | 1106 | 5-shot | [VERIFIED: https://arxiv.org/abs/2404.14219] |
| GSM8K | 57.1 | 2023 | 5-shot CoT | [VERIFIED: https://arxiv.org/abs/2303.08774] |
| GSM8K | 78.1 | 1106 | 0-shot CoT | [VERIFIED: https://arxiv.org/abs/2404.14219] |
| MATH | 34.1 | Nov-2023 API | 4-shot (Minerva prompt) | [VERIFIED: https://arxiv.org/abs/2312.11805] Table 2 |
| BBH | 66.6 | Nov-2023 API | 3-shot CoT | [VERIFIED: https://arxiv.org/abs/2312.11805] |
| BBH | 68.32 | 1106 | 0-shot | [VERIFIED: https://arxiv.org/abs/2404.14219] |
| DROP (F1) | 64.1 | 2023 | 3-shot | [VERIFIED: https://arxiv.org/abs/2303.08774] |
| HumanEval | 48.1 | 2023 | 0-shot | [VERIFIED: https://arxiv.org/abs/2303.08774] |
| HumanEval / HumanEval+ | 76.8 / 70.7 | Nov-2023 | greedy pass@1 | [VERIFIED: https://evalplus.github.io/leaderboard.html] (results.json) |
| Natural2Code (held-out, no web leakage) | 62.3 | Nov-2023 | 0-shot | [VERIFIED: https://arxiv.org/abs/2312.11805] |
| MBPP / MBPP+ | 82.5 / 69.7 | Nov-2023 | greedy pass@1 | [VERIFIED: https://evalplus.github.io/leaderboard.html] |
| MBPP | 77.8 (1106, 3-shot, Phi-3); 52.2 (Mixtral) | | | [VERIFIED: https://arxiv.org/abs/2404.14219], [VERIFIED: https://arxiv.org/abs/2401.04088] |
| TriviaQA | 85.8 | 1106 | 5-shot | [VERIFIED: https://arxiv.org/abs/2404.14219] |
| TriviaQA (EM / Acc) | 82.9 / 91.7 | 1106 | 0-shot, no RAG | [VERIFIED: https://arxiv.org/abs/2407.02485] Table 2 |
| TriviaQA, DPR split (EM / Acc) | 70.1 / 81.3 | 0613 | 0-shot | [VERIFIED: https://arxiv.org/abs/2407.02485] Table 9 |
| NaturalQuestions (EM) | 38.6 | 1106 | 0-shot, no RAG | [VERIFIED: https://arxiv.org/abs/2407.02485] |
| NQ, DPR split (EM) | 35.2 | 0613 | 0-shot | [VERIFIED: https://arxiv.org/abs/2407.02485] |
| GPQA | 29.0 | 1106 | 2-shot CoT | [VERIFIED: https://arxiv.org/abs/2404.14219] |
| MedQA | 63.4 | 1106 | 2-shot | [VERIFIED: https://arxiv.org/abs/2404.14219] |
| MT-Bench | 7.94 / 8.39 / 8.32 | 0314 / 0613 / 1106 | GPT-4 judge, single-answer grading | [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard] leaderboard_table_20250804.csv; 7.94 also in [VERIFIED: https://arxiv.org/abs/2306.05685] |
| AlpacaEval 2.0 LC / raw win | 22.35 / 14.10 (0613); LC 19.30 (1106), 18.09 (0301) | | GPT-4-Turbo judge | [VERIFIED: https://github.com/tatsu-lab/alpaca_eval] leaderboard CSV |
| AlpacaEval 1.0 win vs text-davinci-003 | 89.37 (0301), 86.26 (1106) | | GPT-4 judge | same |
| Arena-Hard-Auto v0.1 | 24.8 (0613), 23.3 (0125); style-controlled 34.9 / 34.6 | | judge gpt-4-1106, baseline gpt-4-0314 = 50 | [VERIFIED: https://github.com/lmarena/arena-hard-auto/blob/main/misc/past_leaderboards.md] |
| Chatbot Arena (2023-12) | 1112 (0613), 1106 (0314) | | Elo/BT | [VERIFIED: https://lmsys.org/blog/2023-12-07-leaderboard/] |
| Chatbot Arena (2025-08-29) | 1132.3 (0125, rank 180/242), 1100.9 (1106); style control 1225.9 / 1203.6 | | BT | [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard] elo_results_20250829.pkl, read with an allow-list unpickler |
| Arena, Korean category | 1049.6 (0125, rank 133/151, 1,317 battles) | | BT | same |
| KMMLU | 42.47 | (unspecified) | 5-shot Direct | [VERIFIED: https://arxiv.org/abs/2402.11548] Table 3 |
| KMMLU-Hard | 21.10 Direct / 20.70 CoT | | 5-shot | same, Table 4 (biased low, see §5) |
| HAE-RAE Bench | 51.2 (Ko prompt) / 55.4 (En prompt) | 0613 | 0-shot, generative | [VERIFIED: https://arxiv.org/abs/2309.02706] Table 11 |
| HAE-RAE subsets (Ko prompt) | LW 62.13, SN 55.56, RW 63.46, HI 30.32, GK 35.80, RC 60.18 | 0613 | 0-shot | same, Table 14 |
| KoBEST (BoolQ, COPA, HellaSwag, SentiNeg) | 68.0 Ko / 79.3 En; Ko subsets 82.34 / 57.88 / 40.00 / 91.69 | 0613 | 0-shot, generative | same, Tables 11, 13 |
| ARC-AGI-1 | no official GPT-3.5 score. Context: GPT-3 0% (2020), GPT-4o 5% (2024) | | | [VERIFIED: https://arcprize.org/blog/oai-o3-pub-breakthrough] |
| ARC (800 public train+eval, ASCII) | text-davinci-003 85/800 = 10.6%; gpt-4-0613 77/800 | | out-of-the-box | [VERIFIED: https://arxiv.org/abs/2307.04721] |

Also: GPT-3.5 + RAG reaches 46.7 EM on NQ (RankRAG). GPT-4 report Fig. 5 has no GPT-3.5 Korean-MMLU number.

## 2. Reference points: what small systems score today

| System (params) | Data points vs GPT-3.5 target | Source |
|---|---|---|
| SmolLM2-135M (base / instruct) | HellaSwag 42.1, ARC avg 43.9, MMLU-cloze 31.5, TriviaQA 4.1, WinoGrande 51.3, GSM8K 1.4, BBH 28.2 | [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct] |
| Qwen2.5-0.5B base / instruct (≈0.5B) | MMLU 47.5, HellaSwag 52.1, ARC-C 35.6, BBH 20.3; instruct GSM8K 49.6, MATH 34.4, HumanEval 35.4, MBPP 49.6 | [VERIFIED: https://qwenlm.github.io/blog/qwen2.5-llm/] |
| phi-1-small (350M) / phi-1 (1.3B) | HumanEval 45% / 50.6% | [VERIFIED: https://arxiv.org/abs/2306.11644] |
| FiD-base (220M reader + DPR retriever) | NQ 48.2 EM, TriviaQA 65.0 EM (fine-tuned per dataset) | [VERIFIED: https://arxiv.org/abs/2007.01282] |
| TRM (7M) | ARC-AGI-1 45%, ARC-AGI-2 8% | [VERIFIED: https://arxiv.org/abs/2510.04871] |
| Gemma-2-2B-it (2.6B [ESTIMATE]) | Arena 1163.8, which beats gpt-3.5-0125 (1132.3); Llama-3.2-1B 1067.4 | [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard] |

What the reference points say [ESTIMATE]: on reasoning and code, 0.35–0.5B models reach about 45–100% of target. On commonsense and chat, nothing under 1B comes close; the smallest Arena match is about 2.6B (26× our budget). Retrieval beat closed-book NQ with 440M total (reader + retriever).

## 3. Classification: which lever solves each benchmark

R = retrieval-solvable, T = tool-solvable (executor/CAS/verifier), Q = reasoning-heavy, F = fluency-heavy. Weights are my judgement [ESTIMATE].

| Benchmark | Mix | Implication for a <100M system |
|---|---|---|
| TriviaQA, NQ | R90 F10 | Wikipedia retrieval + short-answer extractor; risk is recall@k |
| MMLU, KMMLU | R60 Q30 F10 | Exam knowledge; STEM subsets need T; KMMLU needs a Korean datastore |
| HAE-RAE | R60 F40 | Loan/rare words, nomenclature = Korean dictionary lookup; RC is F/Q |
| MedQA, GPQA | R50 Q50 | North-star relevant; GPQA is "Google-proof", so R helps less |
| GSM8K, MATH | T60 Q40 | Program-of-thought + SymPy; bottleneck is formalizing the problem |
| HumanEval+, MBPP+ | T50 Q30 F20 | Generate, run self-written tests, repair |
| DROP | Q50 T30 F20 | Span extraction + calculator; no retrieval needed |
| BBH | Q70 T30 | Dyck, boolean, arithmetic, sorting tasks are exactly executable |
| ARC-AGI-1/2 | Q100 | Program search / test-time training; favors tiny recursive models |
| ARC-Challenge | R50 Q50 | Grade-school science; textbook retrieval helps |
| HellaSwag, KoBEST-HellaSwag | F90 | Pure continuation plausibility; hardest wall |
| WinoGrande, KoBEST-COPA | F60 Q40 | Adversarial commonsense; no retrieval key |
| KoBEST BoolQ/SentiNeg | F60 R40 | Reading + negation |
| MT-Bench, AlpacaEval, Arena(-Hard) | F70 Q15 R15 | Breadth, style, multi-turn; use LC / style-controlled variants |

Score mass [ESTIMATE]: about 35% R, 20% T, 20% Q, 25% F. Externalized knowledge plus verification covers about 55%. The F quarter decides the fight, and it is the part a small parameter budget hurts most.

## 4. Proposed "all domains" scoreboard (targets to beat)

**Win condition per row:** our score ≥ target + the 95% CI half-width. Use a paired bootstrap when GPT-3.5 per-item outputs exist, otherwise the binomial CI at N items [ESTIMATE; item counts from dataset cards, not re-verified]. **Overall win = every row in Tier A and Tier B won. Tier C rows are reported, not gating.** One checkpoint, one system config, for all rows.

### Tier A — gated academic (harness: EleutherAI lm-evaluation-harness, task names verified in repo [VERIFIED: https://github.com/EleutherAI/lm-evaluation-harness])

| Domain | Benchmark (harness task) | Target to beat (max reported) | N | ±95% CI |
|---|---|---|---|---|
| Knowledge | MMLU 5-shot (`mmlu_generative` / `mmlu`) | 71.4 | 14,042 | 0.7 |
| Korean knowledge | KMMLU 5-shot Direct (`kmmlu_direct`) | 42.47 | ~35k | 0.5 |
| Korean culture | HAE-RAE 0-shot (`haerae`: gk, hi, lw, rw, sn) | per subset, Table 14 Ko; mean of 5 = 49.45 [ESTIMATE]; headline 55.4 (En prompt) | ~1.5k [ESTIMATE] | ~2.5 |
| Korean NLU | KoBEST 0-shot (`kobest_boolq/copa/hellaswag/sentineg`) | 79.3 (En-prompt avg); per-subset Ko 82.34/57.88/40.00/91.69 | — | — |
| Math | GSM8K 0-shot CoT (`gsm8k_cot_zeroshot`); 5-shot target 57.1 | 78.1 | 1,319 | 2.2 |
| Math | MATH 4-shot (`minerva_math_*`; also `minerva_math500`) | 34.1 | 5,000 | 1.3 |
| Code | HumanEval+ (`humaneval_plus`) and HumanEval | 70.7 / 76.8 | 164 | 7.0 |
| Code | MBPP+ (`mbpp_plus`) and MBPP | 69.7 / 82.5 | 378 | 4.6 |
| Reasoning | BBH 0-shot (`bbh_cot_zeroshot`) / 3-shot CoT (`bbh_cot_fewshot`) | 68.3 / 66.6 | 6,511 | 1.1 |
| Reading + arithmetic | DROP 3-shot F1 (`drop`) | 64.1 | 9,536 | 1.0 |
| Commonsense | HellaSwag 10-shot (`hellaswag`) | 85.5 | 10,042 | 0.7 |
| Commonsense | WinoGrande 5-shot (`winogrande`) | 81.6 | 1,267 | 2.1 |
| Science | ARC-Challenge 10-shot / 25-shot (`arc_challenge`) | 87.4 / 85.2 | 1,172 | 1.9 |
| Open-domain QA | TriviaQA 5-shot (`triviaqa`, rc.nocontext) | 85.8 | ~11k [ESTIMATE] | 0.6 |
| Open-domain QA | NQ-open 0-shot (`nq_open`) | 38.6 | 3,610 | 1.6 |
| Abstraction | ARC-AGI-1 public eval, pass@2 | 5.0 (GPT-4o upper bound for GPT-3.5 [ESTIMATE]) | 400 | 2.1 |

### Tier B — gated chat / instruction following (zero-cost protocol)

| Benchmark | Target | Zero-budget protocol |
|---|---|---|
| MT-Bench | 8.39 | Official GPT-4 judging costs API money. Instead: (1) pairwise, our answers vs the stored `gpt-3.5-turbo.jsonl` answers [VERIFIED exists: https://huggingface.co/spaces/lmsys/mt-bench data/mt_bench/model_answer/], judged by a local open-weight judge on Kaggle, both orders, win rate > 50% + CI. (2) Single-score with the same judge, calibrated on the stored answers of ≥5 reference models [HYPOTHESIS: rank correlation with the GPT-4 judge good enough; must be measured]. |
| AlpacaEval 2.0 LC | 22.35 | Pairwise vs the stored `results/gpt-3.5-turbo-0613/model_outputs.json` [VERIFIED exists: https://github.com/tatsu-lab/alpaca_eval], same local judge, length-controlled. |
| Arena-Hard-Auto v0.1 | 24.8 (34.9 style-controlled) | Same approach; GPT-3.5 answers are distributed with the benchmark data [ESTIMATE]. |
| Chatbot Arena | 1132 overall / 1226 style-controlled; Korean 1050 | Stretch only: needs LMArena to host the model. Not gating until it is listed. |

### Tier C — north-star and anti-overfit (report, do not gate)

GPQA 29.0, MedQA 63.4, Natural2Code-style fresh code (GPT-3.5 62.3, held out), ARC-AGI-2 (GPT-3.5 ≈ 0 [ESTIMATE]; TRM 8%), plus a **private held-out set written after the training-data cutoff** (fresh math, code and Korean questions).

## 5. Eval protocol and fair-play rules (scoring side)

1. **Generative scoring everywhere.** GPT-3.5 numbers come from generated answers, and a non-autoregressive architecture may not define log-likelihoods. Multiple-choice tasks are therefore scored by generated-answer exact match (`*_generative` / `generate_until`); log-likelihood scores are secondary. This also blocks protocol arbitrage: GPT-3.5's generative KoBEST-HellaSwag 40.0 and COPA 57.88 are suspiciously low.
2. **Shot count matches the target row**, or report both settings.
3. **Parameters** = every trainable weight run at inference (retriever, encoders, embeddings, reranker, decoder, verifier). The total must be < 100M. The datastore (text plus index vectors) is excluded but disclosed: size, sources, build script.
4. **Allowed:** a frozen offline public-text datastore, Python/SymPy/Z3, deterministic solvers. **Banned:** any neural net > 100M, LLM APIs, live web search, and test items or their solutions in the datastore.
5. **Decontamination:** 13-gram filter on training data and datastore against every eval set, including mirrors (HF dumps, GitHub copies of HumanEval/MBPP, KMMLU/HAE-RAE source exam sites). Publish the overlap report.
6. **No dataset-identity routing, prompts or heads.** Hidden code tests are never run during generation. Run 3 seeds for anything sampled.
7. **Known bias:** KMMLU-Hard was built from items GPT-3.5 and others failed [VERIFIED: https://arxiv.org/abs/2402.11548]. Report it, but gate on full KMMLU. GPT-3.5 may be contaminated on public sets; the private Tier C set is the honest check.

## 6. Walls, ranked by expected difficulty [ESTIMATE]

| # | Row | Best known ≤0.5B | Route around [HYPOTHESIS] |
|---|---|---|---|
| 1 | Chat (MT-Bench/AlpacaEval/Arena-Hard) | far below; smallest Arena match ≈2.6B | Retrieved exemplar responses, plan-then-realize decoding, test-time critique loops |
| 2 | HellaSwag 85.5 | 42.1 (135M), 52.1 (0.5B) | Retrieval of similar narratives (decontaminate WikiHow/ActivityNet), energy-based reranking |
| 3 | WinoGrande 81.6 | 51.3 (≈ chance) | Commonsense knowledge-graph store, multi-hypothesis verification |
| 4 | ARC-C 87.4 / MMLU 71.4 | 35.6 / 47.5 | Dense + sparse retrieval with a small reader |
| 5 | HumanEval+ 70.7 | 45 (phi-1-small, HumanEval) | Execution-guided search, self-tests, API-doc retrieval |
| 6 | GSM8K 78.1 / MATH 34.1 | 49.6 / 34.4 | Program-of-thought + executor + verifier search |
| 7 | Korean rows | data scarcity | Korean dictionary, ko-Wikipedia and exam datastore |
| 8 | NQ / TriviaQA | FiD 48.2 NQ (440M) | Retrieval-first |
| 9 | ARC-AGI-1 | TRM 45% at 7M | Recursive solver + test-time training |

## 7. Gaps (not verified)

- GPT-3.5 numbers on the GPT-4o-mini launch page: openai.com returned 403, so they are unused. No primary source found for IFEval or MMLU-Pro.
- N for TriviaQA, KMMLU and HAE-RAE, and whether Arena-Hard ships GPT-3.5 answers: [ESTIMATE].
- Whether a local judge on 2×T4 tracks GPT-4-judge rankings: [HYPOTHESIS]; calibrate on the stored outputs first.
