# Reasoning Methods for a <100M Non-Transformer Core ("reason_methods")

Scope: which reasoning methods give the most reasoning per parameter, what compute each needs, what fits the Kaggle free tier (2x T4, about 30 GPU-h/week; TPU v5e-8, about 20 h/week), and which ones break the fair-play rules. North star: a **recursive self-improvement (RSI) loop**, not imitation of a teacher.

## 0. Bottom line (read this first)

1. **Teacher imitation sets a ceiling; search plus exact verification does not.** RLVR (reinforcement learning from verifiable rewards) mostly sharpens what the base model can already do. At large k, base models reach a *higher* pass@k than their RL-trained versions, and "current training setup does not elicit fundamentally new reasoning patterns" [VERIFIED: https://arxiv.org/abs/2504.13837]. So GRPO alone is not an RSI engine. The loop that has gone past every human or teacher is **AlphaZero-style expert iteration**: search acts as the policy-improvement operator, an exact verifier or game outcome supplies the reward, and the network is trained on what search found. AlphaGeometry (100M synthetic examples, no human demonstrations) and AlphaProof (millions of problems proved or disproved in Lean) are the existence proofs in mathematics [VERIFIED: https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/], [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/].
2. **Most reasoning per parameter so far:** recursion with deep supervision (TRM, 7M params, ARC-AGI-1 44.6%), search with an exact verifier (rStar-Math takes a 1.5B model from 51.2% to 87.8% on MATH), and test-time training (TTT; ARC Kaggle winners). The weakest is latent chain of thought on its own: Coconut scores 34.1% on GSM8K against 42.9% for explicit CoT.
3. **Fair-play trap:** most results in the "tiny model beats giant" literature use a verifier or PRM (process reward model) *larger than the policy*. rStar-Math pairs a 7B PPM with a 1.5B policy. TinyGSM's best result pairs a 1.3B verifier with a 125M generator. Under our rules these PRMs count toward the 100M budget. We therefore have to use **0-parameter exact verifiers** (Python execution, SymPy, Z3, Lean, unit tests, ARC grid checking) plus a small judge head that shares weights with the core.
4. **Fits Kaggle:** a 100M policy doing self-play with search and exact verifiers is cheap per rollout. The bottleneck is rollout throughput, not gradient compute. Rough budget: about 1e9 to 1e10 generated tokens per week [ESTIMATE, §4].

## 1. Evidence table (small-model numbers)

| Method | Setting | Numbers | Source |
|---|---|---|---|
| **Zero-RL (GRPO-style), 0.5B** | Qwen2.5-0.5B base | GSM8K 36.7→**49.5**; MATH500 15.8→**34.4** | [VERIFIED: https://arxiv.org/html/2503.18892] |
| Zero-RL, 1.5B | Qwen2.5-1.5B base | GSM8K 55.7→**74.4**; MATH500 29.6→**59.0**. Small models grew mostly "subgoal setting" (4–5x), with no rise in self-reflection | [VERIFIED: https://arxiv.org/html/2503.18892] |
| 1-shot RLVR | Qwen2.5-Math-1.5B, a **single** training example | MATH500 36.0→**73.6** (two examples: 74.8) | [VERIFIED: https://arxiv.org/abs/2504.20571] |
| Spurious rewards (warning) | Qwen2.5-Math-7B | **Random** reward gives +21.4 on MATH500, against +29.1 for the true reward. This **does not transfer** to Llama3 or OLMo2; the cause is GRPO clipping bias amplifying Qwen's "code reasoning" | [VERIFIED: https://arxiv.org/abs/2506.10947] |
| RLVR ceiling | Several families | RL wins at pass@1; **base wins at large k**; distillation, unlike RL, *can* add new patterns | [VERIFIED: https://arxiv.org/abs/2504.13837] |
| R1-Distill-1.5B | Distilled from R1 | AIME24 28.9, MATH500 83.9, but GPQA 33.8 and LiveCodeBench 16.9 (it loses outside math) | [VERIFIED: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B] |
| **rStar-Math** (MCTS self-evolution) | Qwen2.5-Math-1.5B policy + **7B PPM** | MATH 51.2→**87.8** (8 trajectories) / 88.6 (64); AIME24 0→**46.7** | [VERIFIED: https://arxiv.org/html/2501.04519] |
| rStar-Math compute | — | Bootstrap: 10 nodes × 8×H100 for about 2 weeks. Rounds 2–4: 15 nodes × 4×A100, about 3 days each. Steps are kept only if their **Python code executes** | [VERIFIED: https://arxiv.org/html/2501.04519] |
| **Absolute Zero (AZR)** | Self-proposed tasks checked by a Python executor, **zero human data** | Gains (code/math avg): 3B-Coder +3.7/+7.7; 7B-Coder +5.0/+15.2; 14B-Coder +3.6/+22.8; Llama3.1-8B +3.1/+3.4. **Gains grow with scale.** Safety: "uh-oh moment" chain of thought in Llama ("outsmart ... less intelligent humans") | [VERIFIED: https://arxiv.org/html/2505.03335v3] |
| TinyGSM | 125M generator + 1.3B verifier, 48 samples | GSM8K 63.1 alone → 76.1. "The verifier seems to play a bigger role than generation model size" | [VERIFIED: https://arxiv.org/html/2312.09241] |
| Compute-optimal test-time compute | PaLM 2-S* | More than 4x more efficient than best-of-N; FLOP-matched, beats a **14x** larger model, but only where base success is non-trivial | [VERIFIED: https://arxiv.org/abs/2408.03314] |
| Test-time scaling at small size | 0.5B / 1B | 0.5B > GPT-4o on MATH-500; 1B > 405B (PRM-guided) | [VERIFIED: https://arxiv.org/abs/2502.06703] |
| **TRM** (recursion + deep supervision) | 7M, 2 layers | Sudoku-Extreme 87.4; ARC-AGI-1 **44.6**; ARC-AGI-2 7.8; attention-free variant 29.6 on ARC-AGI-1 | [VERIFIED: https://arxiv.org/html/2510.04871] |
| **Huginn** (recurrent depth) | 3.5B, 800B tokens, 4096 MI250X | GSM8K-CoT 34.8/42.1 at r=32; EMA model 47.2 at r=64; **0.0 at r=1**. OLMo-7B scores 6–7 and OLMo-2-7B 66.7. Test-time compute equivalent to about 50B params | [VERIFIED: https://arxiv.org/html/2502.05171] |
| **Coconut** (continuous thought) | GPT-2 (124M) | GSM8K: No-CoT 16.5, **CoT 42.9**, Coconut 34.1 (8.2 vs 25 tokens). ProsQA: CoT 77.5 vs Coconut **97.0** | [VERIFIED: https://arxiv.org/html/2412.06769] |
| **ARC test-time training** | 8B LM | 53.0% on ARC public validation; **61.9%** when ensembled with program synthesis; up to 6x over a fine-tuned baseline | [VERIFIED: https://arxiv.org/abs/2411.07279] |
| ARC Prize 2024 | Kaggle, **1×P100, 12h, offline** | ARChitects **53.5%** (won); MindsAI 55.5% (closed-source) | [VERIFIED: https://arxiv.org/html/2412.04604v2] |
| ARC Prize 2025 | Kaggle | NVARC **24.03%** on ARC-AGI-2 private, $0.20/task; synthetic data + ARChitects-style TTT + TRM components | [VERIFIED: https://arcprize.org/blog/arc-prize-2025-results-analysis] |
| **AlphaCode** (sample + filter) | 300M–41B | Up to **1M samples/problem**; filtering on example tests removes **about 99%**; 41B with clustering: 34.2% validation / 29.6% test at 10@100k–1M. Solve rate scales **log-linearly with samples** | [VERIFIED: https://arxiv.org/abs/2203.07814] (PDF read) |
| DreamCoder | Wake-sleep library learning | Rediscovers functional-programming primitives, vector algebra, Newton's and Coulomb's laws; reports no headline benchmark number | [VERIFIED: https://arxiv.org/abs/2006.08381] |
| **CLRS** (Triplet-GMPNN) | Small GNN | OOD micro-F1 **74.1** vs 50.8 (PGN); 24/30 algorithms >60%; fails on KMP (19.5) and Quickselect (0.5) | [VERIFIED: https://arxiv.org/html/2209.11142] |
| AlphaGeometry | Neural LM + symbolic deduction engine | IMO-AG-30: **25/30** vs 25.9 for the average gold medalist; 100M synthetic examples | [VERIFIED: https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/] |
| AlphaProof | AlphaZero-style RL in Lean | IMO 2024 silver, 28/42 (with AG2); up to 3 days per problem. The Nature paper adds **test-time RL**, training on millions of variants of the target problem | [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/], [VERIFIED: https://research.google/pubs/olympiad-level-formal-mathematical-reasoning-with-reinforcement-learning/] |
| **GSM-Symbolic** (fragility) | Various | GSM-NoOp (one irrelevant clause): Phi-3-mini 83.7→**18.0**; GPT-4o 95.2→63.1; o1-preview 94.9→77.4. Swapping numbers alone gives 12–15 point spreads (Gemma2-9B, Phi-3.5) | [VERIFIED: https://arxiv.org/html/2410.05229] |

## 2. Which methods give the most reasoning per parameter

**A. Search + exact verifier + expert iteration: the RSI engine.** [HYPOTHESIS: biggest lever for us] Evidence: rStar-Math +36.6 MATH points at 1.5B; AlphaCode solve rate log-linear in samples; AlphaGeometry/AlphaProof beat medalists with no human demos. Search finds what the policy rarely samples; training on verified search output raises pass@1; repeating this expands coverage instead of only sharpening it, which answers arXiv 2504.13837 [HYPOTHESIS; AlphaZero-lineage support, unproven at 100M on GSM8K]. Keep from rStar: code-form steps kept only if they execute, MCTS Q-values as process labels, multi-round evolution. Drop: the 7B PPM; replace with 0-param execution/SymPy/Z3/tests plus a few-M-param value head on the shared trunk.

**B. Recursion + deep supervision (TRM/HRM): test-time depth on shared weights.** TRM at 7M beats R1 and Gemini 2.5 Pro on ARC-AGI-1 and Sudoku; Huginn shows depth matters (GSM8K 0 at r=1, 42 at r=32). Costs no params, and is non-Transformer with an SSM/MLP/GNN mixer. Wall: attention-free TRM loses 15 ARC points, so content-addressable lookup must come from elsewhere (object GNN, retrieval read, 0-param suffix-automaton copy). Huginn needed 3.5B params and 800B tokens for GSM8K 42, so depth does not replace program-of-thought on word problems [ESTIMATE].

**C. Test-time training (TTT).** ARC winners (53.5% on a P100 in 12h), the TTT paper (6x), AlphaProof test-time RL. Converts test-time compute into gradient steps on augmented demonstrations; fits Kaggle by construction. Legal: uses only the test item's own demonstrations.

**D. Program synthesis + library learning (DreamCoder, LILO, AlphaCode filtering).** The core emits typed programs and an interpreter checks them; about 99% of AlphaCode samples die at the 0-param filter. Learned abstractions shrink future search, so the library is a scaling axis with **zero params** [HYPOTHESIS]. LILO used an LLM; ours uses the core as proposer and Stitch-style symbolic compression, with the renderer allowed only to *name* abstractions offline.

**E. RLVR/GRPO: a sharpener, not an engine.** Cheap, large gains at 0.5–1.5B (+13 GSM8K and +19 MATH500 at 0.5B; 1-shot +37.6 at 1.5B), but bounded by base coverage and partly family-specific elicitation (spurious rewards). A randomly initialized 100M core has nothing to elicit, so reward stays ≈0 without a curriculum [ESTIMATE]. Use it between search rounds to sharpen pass@1.

**F. Self-play task proposal (Absolute Zero).** Zero data, but gains grow with scale (+5.7 at 3B to +13.2 at 14B), so at 100M it may be weak [HYPOTHESIS]. Keep the learnability reward and pair it with AlphaGeometry-style symbolic generators, which supply problems and solutions for free and avoid the weak-proposer problem.

**G. Neural algorithmic reasoning (CLRS).** Tiny GNNs reach 74% OOD. Use algorithm traces as an unlimited, exactly checkable pretraining curriculum for the recurrent processor, aimed at BBH-style sorting, Dyck and boolean tasks [HYPOTHESIS]. Weak spots: KMP, Quickselect.

**H. Latent reasoning (Coconut).** Strong on search-like logic (ProsQA 97 vs 77.5), weaker on arithmetic (GSM8K 34.1 vs 42.9). Use latents as the inner loop only; arithmetic goes out to tools through emitted programs.

## 3. Fair-play audit

| Issue | Ruling under our rules |
|---|---|
| PRM/verifier larger than the policy (rStar 7B PPM, TinyGSM 1.3B) | **Banned** at inference if >100M total. Allowed: 0-parameter exact checkers plus a judge head inside the 100M budget |
| Training-time distillation from an open teacher (R1-Distill style) | Allowed only if disclosed. The user's goal is to *exceed* teachers, and arXiv 2504.13837 says distillation adds patterns RL cannot. So: allow open-teacher bootstrap data for round 0, logged, and always also report a **no-teacher track** where every training token comes from symbolic generators, executors and self-play. GPT-3.5/4 outputs are banned outright. TinyGSM's GPT-3.5 data is therefore unusable |
| Search budget | Report it: rollouts per problem, wall-clock, FLOPs. Compare against GPT-3.5 at pass@1, and also on an equal-FLOP footing (the ORRERY-0 control arm) |
| TTT on test inputs | Legal: uses only the test input's demonstrations. Must be in the reported inference budget |
| Random-reward artifact | Every RL gain needs a **random-reward control run**, since spurious rewards gave +21.4 on Qwen. Cheap, and it catches illusions |
| GSM-Symbolic fragility | Report GSM-Symbolic and GSM-NoOp next to GSM8K. A core that parses to a typed program and executes it should be **invariant to number swaps** by construction. That is a testable claim against GPT-3.5 [HYPOTHESIS] |
| Benchmark-identity routing | The controller may route on problem *type* features it learned itself, never on dataset ID |
| Self-play safety | AZR's "uh-oh" chain of thought shows that self-play objectives can drift. Log and audit the proposer's tasks. Biomedical generation is restricted by a hard dual-use filter (no toxin or pathogen enhancement tasks), plus human review before any wet-lab-relevant output |

## 4. Compute on the Kaggle free tier [ESTIMATE throughout]

- A 100M core costs about 2e8 FLOPs/token. A T4 at 15–25 sustained fp16 TFLOP/s gives about 1e5 tokens/s at large batch; recurrence r=8 divides that by about 8. So 30 GPU-h/week buys about 1e9–1e10 rollout tokens, i.e. 1–10M verified attempts of about 1k tokens.
- rStar's bootstrap (80 H100 × 2 weeks) is about 1e4x our budget. This works only because the policy is about 15x smaller than 1.5B and the verifier costs 0 params. Expect 2–3 rStar-style rounds per month on a few hundred thousand problems.
- Python checking on Kaggle's 4 vCPU (about 1e2–1e3 checks/s) is the real bottleneck, so batch and vectorize the checks. The TPU v5e-8 (JAX) does gradients; the T4s do rollouts. ARC TTT fits: the 2024 winners ran an 8B-class model on 1 P100.

## 5. Recommended reasoning stack

1. **Round 0:** pretrain the recurrent non-Transformer trunk on exactly verifiable synthetic streams: CLRS traces, SymPy algebra, program execution (AZR-style deduction, induction and abduction), re-ARC generators. Optionally add logged open-teacher data as a disclosed track.
2. **Inner loop:** per-query recurrence r as a masked while_loop with TRM deep supervision. Learned halting, with its cost charged to the budget.
3. **Outer loop:** MCTS or beam over typed program-IR steps with grammar-constrained decoding. Every node is executed. A shared-trunk value head is trained on MCTS Q-values.
4. **RSI loop:** propose tasks (learnability reward plus symbolic generators) → search → SFT on verified traces plus a GRPO sharpening pass → library compression → repeat. **Track pass@1 and pass@1024 every round.** It is RSI only if pass@large-k rises.
5. **Test time:** TTT on demonstrations, plus Snell-style compute-optimal allocation that spends the most search on medium-difficulty items.

## 6. What scales, and how to measure it

| Axis | Expected law | Metric |
|---|---|---|
| Search samples k | Log-linear solve rate (AlphaCode) [VERIFIED] | Solve rate vs log k, per benchmark |
| Recurrence r | Gains up to r of about 32–64, then saturation (Huginn) [VERIFIED] | Accuracy vs r curve; FLOPs per correct answer |
| Self-play rounds | rStar improved over 4 rounds [VERIFIED]; AZR gains rise with model size [VERIFIED] | pass@1 **and** pass@1024 per round; held-out *new-generator* tasks |
| Library size | DreamCoder/LILO: solve rate rises and search depth falls [HYPOTHESIS for numbers] | Mean program length and search nodes per solved task |
| Params 0.1B→1B→10B | AZR +5.7 (3B) → +13.2 (14B); verifier size beats generator size (TinyGSM) | Same suite at fixed rules; put new params into the judge first |

## 7. Walls (honest)

1. **The cold-start wall:** search needs non-zero base success (Snell; rStar's bootstrap needed a strong model). A random 100M core has none. Route around it with curriculum generators whose difficulty starts at trivial and is adapted on learnability, as in AZR.
2. **The verifier wall:** no exact verifier exists for GPQA/MedQA-style science MCQ, or for open problems without a formal statement. There we fall back to a learned judge, which is small and hackable. DeepSeek avoided neural reward models because of reward hacking [VERIFIED: https://arxiv.org/html/2501.12948]. Route: retrieval plus consistency checks, and formalize into Lean or Z3 whenever possible.
3. **The comprehension wall:** reading GSM/DROP prose into programs is the hard part for a 100M non-LLM. GSM-NoOp shows even large LMs fail here. Nothing we read shows a <100M model parsing arbitrary prose robustly; this is the biggest unknown [HYPOTHESIS: the riskiest component; it needs a week-2 kill probe on GSM8K parse accuracy].
4. **The RLVR ceiling:** handled by search plus coverage tracking (§5.4). If pass@1024 stops rising for 2 rounds, the loop has stopped being RSI. Then expand the task generators or the library, and do not train longer.
5. **ASI honesty:** none of these methods has shown open-ended RSI outside closed domains (games, Lean, geometry). The claim "this loop scales to the north star" is [HYPOTHESIS]. What is verified is that it beats humans inside closed, verifiable worlds. So the strategy is to make more of science verifiable (formal math, code, simulators), because that is where the loop works.

