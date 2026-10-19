# MISSION BRIEF — The Aphasic Genius RSI Engine

Read this entire file before doing anything. It is the single source of truth for the task. Follow every section literally. When this brief and your own instincts disagree, the brief wins.

---

## 0. One-sentence mission

Design, from scratch, a **recursive self-improvement (RSI) engine** whose learner is a **tiny (<100M parameter), non-Transformer, reasoning-only "aphasic genius" core** that never generates natural language, and deliver a complete, honest, technically precise, and very funny **Korean design document (기획서)** for it.

---

## 1. Who the user is and what they want (read carefully, this is the "why")

The user is a solo builder with **zero money (0 KRW)**. Their long-term north star is an **ASI-level reasoner** that solves open scientific and mathematical problems, discovers cancer therapeutics, and writes and fixes complex code. They believe all of these are **reasoning** problems, not **talking** problems.

They made three decisions, in this order. All three are binding.

### Decision 1: Drop language generation entirely ("the aphasic genius")
- The core model puts 100% of its capacity into **reasoning**.
- It **reads and understands** problems (comprehension is required), but it **never generates natural language**.
- It emits **formal outputs only**: final answers (numbers, choice letters, grids, spans copied from the input), programs/code, proofs, structured reasoning traces, and tool calls.
- A separate, off-the-shelf open LLM (the **"renderer"**, like Stephen Hawking's speech synthesizer, or a football coach's interpreter) turns the core's structured output into fluent language for humans.
- Chat, writing, MT-Bench, AlpacaEval, Arena, and HellaSwag-style fluency are **out of scope**. Do not spend a single sentence designing for them.

### Decision 2: The real project is RSI
- Distilling reasoning traces from frontier teachers (Claude Opus, Gemini Argon, GPT-6 Astra) **caps the student at the teacher's level**, and those teachers are not AGI. The user's words: even the frontier models are "tin cans" (깡통) compared to the north star.
- So the teacher may only be a **bootloader** (rocket stage 1), never the ceiling.
- Improvement after the bootloader must come from sources that **do not need a smarter teacher**: exact verifiers and environments (Python execution, unit tests, SymPy, Z3, Lean, game rules, simulators), self-play, self-generated curricula (proposer vs solver), growing skill/program libraries, and eventually improving the training procedure itself.
- Proof that this escape is possible: AlphaGo Zero beat the human-data AlphaGo with zero human games, because the **rules of Go are a perfect verifier**. TinyGSM's student plus a verifier beat its own GPT-3.5 teacher (81.5% vs 77.4% on GSM8K). DeepSeek-R1-Zero improved by pure RL on checkable answers. AlphaProof improved by proving millions of problems in Lean. **The key is the verifier, not the teacher.**

### Decision 3: RSI is the spine and the definition of scalability
- The deliverable is **a machine that keeps making the reasoner smarter**, not a static model.
- "Scalable" means: capability grows **per generation**, and ideally the **rate of improvement itself grows** (positive second derivative).
- Beating GPT-3.5 on reasoning is only the **first rung** ("the first slime after the tutorial"), reached at RSI generation N. The ladder continues: GPT-4-level reasoning → o1/R1-level → today's frontier "tin cans" (Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra) on reasoning → top human experts → north star.
- The realistic-ideal growth shape is a **staircase of S-curves**: each S-curve saturates when its verifier domain is exhausted, and a new S-curve starts when the loop opens a new verifier domain (arithmetic → algebra → formal proof/Lean → code with tests → simulators → real experiments). Seen from far away, the staircase looks exponential (like Moore's law was a staircase of transistor technologies). **Finding or building the next verifier domain is itself an RSI skill** that must be designed.

---

## 2. Hard constraints (violating any of these makes the deliverable worthless)

1. **Parameters**: the reasoning core has **< 100,000,000 trainable parameters**, counting **every** weight executed at inference: token embeddings, input reader/encoder, retriever encoder, verifier/judge, proposer, solver, every head. Provide an exact per-module table with arithmetic that sums correctly. The renderer is **not** counted, because it never touches scoring.
2. **Not a Transformer**: no softmax attention (and no Hopfield-equivalent attention) over the token sequence anywhere in the backbone. Allowed: SSMs (Mamba-2), linear RNNs and delta-rule recurrences (Gated DeltaNet, RWKV-7), convolutions, MLPs, recurrent depth, test-time-trained memories, hash/product-key lookups. Mechanisms borrowed from Transformer-based papers (e.g. HRM, TRM-Att, Huginn) must be **ported** to a non-attention backbone.
3. **Knowledge is externalized**, not crammed into weights (small models store only about 2 bits of knowledge per parameter, so 100M params ≈ 25 MB of facts). Use a frozen offline datastore and zero-parameter tools.
4. **Zero money.** Compute available:
   - Kaggle free tier: 2× NVIDIA T4 16 GB (≈30 GPU-hours/week, 12 h sessions, **fp16 only, no bf16**; Triton ≥ 3.3 dropped T4/sm_75 support, so pin Triton 3.2 or use pure PyTorch/JAX kernels; FlashAttention-2 does not run on T4).
   - Kaggle TPU v5e-8 (≈20 h/week, ≈9 h sessions, ≈1,576 bf16 TFLOPS peak, JAX/XLA, static shapes). Availability is sometimes limited.
   - A laptop RTX 3060 with 6 GB VRAM (Ampere: bf16 and modern kernels work; use for development, ablations, CPU/GPU tool execution).
   - Hugging Face Hub free storage (~100 GB private) for checkpoints and datasets.
   - Show explicit FLOP math (training ≈ 6·N·D) against these quotas for every stage and for **one RSI generation**.
5. **Renderer firewall** (fair play):
   - **Blindfold**: the renderer never sees the original question, only the core's structured output, so it cannot solve the task itself.
   - **Core-only scoring**: all benchmark scores come from the core's formal outputs, with no renderer involved.
   - **Renderer-alone control**: publish the renderer's own score on the same tasks, to prove the core is doing the reasoning.
   - **Input is read by the core itself**: no LLM front-end parses or formalizes problems, because parsing/autoformalization can smuggle reasoning.
6. **Banned**: any neural net > 100M params inside the reasoning loop at inference, any LLM API at inference, live web search at inference, test-set contamination, benchmark-identity routing, and **any GPT-3.5/GPT-4 outputs as training data** (OpenAI terms, plus it would be copying the boss's answers). Teacher data at training time may only come from **open-licensed models**, must be logged, and a "no-teacher" ablation must be reported.
7. **RSI integrity**: the model can **never** modify its own reward functions, verifiers, or evaluation sets (they run read-only in a separate process/sandbox).
8. **Safety is stage-gated and mandatory** (see §6). The user's personal rule: **if anything could harm people, stop immediately.**
9. **Honesty**: every number and claim carries a label: **[검증됨: url]** (verified from a real source), **[추정]** (your estimate or arithmetic), or **[가설]** (untested design bet). **Never invent a citation, a URL, a benchmark score, or a paper.** If you cannot verify something, label it [추정] or [가설]. Never write "impossible" without explaining exactly *why* current technology blocks it and *how* to route around it.

---

## 3. Verified research you MUST read first

A previous research team already verified many facts. They are in the directory **`/home/user/hip/asi-blueprint/research/`**. Read **all** of these files before designing anything, and prefer their numbers over your memory:

| File | What it contains |
|---|---|
| `reason_baseline.md` | **The reasoning-only GPT-3.5 scoreboard.** Max reported GPT-3.5 scores per row, the "아득히" margin rule (G = T + max(10, (100 − T)/3)), robust/perturbed twins, gated tiers. **Use this as the official scoreboard.** |
| `rsi.md` | Recursive self-improvement precedents, failure modes (plateau, reward hacking, model collapse), metrics, and safety practices. |
| `reason_methods.md` | RLVR/GRPO, MCTS (rStar-Math), Absolute Zero, STaR, latent reasoning, program synthesis, library learning, ARC test-time training. |
| `io_split.md` | Evidence that understanding is cheaper than generating, renderer candidates, renderer faithfulness, fair-play precedents. |
| `tiny_sota.md` | What tiny models score today (TRM 7M: ARC-AGI-1 44.6%; TinyGSM; SmolLM2; knowledge capacity ≈ 2 bits/param). |
| `archs.md` | Non-Transformer architectures, which famous "alternatives" secretly contain attention, T4/Triton kernel pitfalls. |
| `externalize.md` | Retrieval, tools, test-time compute, parameter-cost traps (e.g. a 177M retriever alone breaks the budget; BM25 has 0 params). |
| `compute.md` | Kaggle T4/TPU quotas and 6ND throughput tables, storage, terms of service. |
| `data.md` | Open datasets with licenses; which popular datasets were generated by GPT-3.5 and are therefore banned (e.g. TinyGSM, MetaMathQA). |

Do **not** read or copy anything under `/home/user/hip/asi-blueprint/archive/`. That is an older, abandoned design that focused on fluency. You are designing **from scratch**.

Key verified anchors (do not contradict these):
- GPT-3.5 max reported: GSM8K **78.1**, MATH **34.1**, BBH **70.7**, HumanEval **76.8** / HumanEval+ **70.7**, MBPP **82.5** / MBPP+ **69.7**, DROP F1 **64.1**, GPQA ≈ **30**, MedQA **63.4**, ARC-AGI-1 ≈ **10.6%** (text-davinci-003 on public sets), LiveCodeBench **22.6**, GSM-Plus **61.19**. Full table in `reason_baseline.md`.
- Tiny precedents: TRM (7M params) ARC-AGI-1 **44.6%**, Sudoku-Extreme **87.4%**; attention-free TRM-MLP drops to ARC-AGI-1 **29.6%** and Maze-Hard **0.0%** (so a non-attention core needs another way to do precise lookup over context).

---

## 4. How to work (process for the team)

Use your team of agents. Assign them these roles and run the phases in order. Every agent must have read §1–§3.

### Phase A: Independent designs (5 agents in parallel)
Each designer champions ONE RSI-engine philosophy and writes a complete proposal to `work/proposal_<name>_v1.md` (3,000–5,000 words, English is fine for work files):
1. `library` — **Library-growing RSI** (neuro-symbolic compiler): the core compiles problems into typed programs/formal statements over zero-parameter executors (Python, SymPy, Z3, Lean). Loop = DreamCoder-style wake–sleep: solve with search → verify → compress solutions into a growing skill library → retrain on "dreams". Capability compounds because each library generation shortens future solutions.
2. `depth` — **Depth-distillation RSI** (latent recursive reasoner): weight-tied deep recursion on an attention-free core (Gated DeltaNet / RWKV-7). Loop = spend large test-time depth/search to solve harder problems → verify → distill deep thinking into shallow thinking so the next generation starts higher (amplify-then-distill).
3. `search` — **Expert-iteration RSI** (AlphaZero-style): tiny policy+value network driving MCTS over reasoning steps with exact verifiers as reward. Search improves the policy, the policy improves the search. Absolute-Zero-style proposer-vs-solver self-play keeps the curriculum at the edge of competence.
4. `curiosity` — **Curiosity / world-model RSI**: a JEPA-style latent model predicts problem states and verifier/tool/experiment outcomes; the loop chooses what to learn next by expected learning progress / information gain; later it chooses which real experiments to run (drug-discovery north star).
5. `minimal` — **Minimal fastest-cycle RSI**: the simplest proven stack (attention-free core + recurrent depth + tools + exact verifiers + RLVR + proposer/solver self-play), optimized for the **shortest generation cycle** on free compute and the cleanest measurement of improvement per generation. Every exotic module elsewhere must beat this at equal FLOPs and equal wall-clock per generation.

Every proposal must contain these 12 sections, **in this order (RSI first)**:
1. Thesis: why teacher distillation caps at the teacher, what replaces the teacher, why this loop compounds, why dropping language generation makes each generation cheaper.
2. **The RSI engine**: the generation loop step by step (propose problems → solve/search → verify → select → train → expand library/curriculum/verifiers); RSI levels L1 (weights improve on verified self-data), L2 (self-generated curriculum), L3 (growing skill/program library), L4 (improving the training procedure, data generators, hyperparameters, kernels; human-approved), L5 (architecture self-redesign, north star only); the **teacher bootloader and an explicit, measurable cord-cut criterion**; fuel after the cut; the S-curve staircase of verifier domains and the concrete mechanism that finds or builds the next domain; generation cycle time and compute per generation on free quota.
3. Curve-killer defenses, each with **mechanism + detector + rollback/kill rule**: plateau, own-level prison (the model only proposes problems at its own level), reward hacking (the dangerous "fake exponential": in-loop score rises while held-out does not), model collapse, compute wall, real-experiment latency.
4. RSI metrics dashboard: teacher-exceedance (fraction of held-out problems the teacher fails that we solve), improvement per generation, second derivative of improvement, held-out vs in-loop gap, number of generations without new human/teacher data, efficiency gains per generation.
5. The core as the learner inside the loop: ASCII component diagram, dataflow, the exact structured output format sent to the renderer, the exact parameter table (< 100M, including the input reader's embeddings), and why it is not a Transformer.
6. Fair-play rules (renderer firewall + RSI integrity).
7. The boss ladder: per reasoning row of the scoreboard, at which generation and with what score we pass GPT-3.5 and by what margin (use the G rule); then the next rungs (GPT-4-level, o1/R1-level, frontier), each with what has to be true to reach it, and honest probabilities.
8. Bootloader training recipe (stage 0) and data (licensed; no GPT-3.5/4 data), compute with explicit FLOP math vs the quotas, T4/TPU engineering notes, inference cost per query.
9. Renderer choice and integration (which open model, where it runs, prompt contract, faithfulness check).
10. Scaling: how the RSI loop transfers 0.1B → 1B → 10B (warm-starting, carrying over the library and curriculum), and why humans (not the AI) would win more compute (e.g. TPU Research Cloud grants) with published results. North-star path: drug discovery via simulators as verifiers and information-gain experiment selection.
11. Stage-gated safety mechanisms (see §6 of this brief).
12. First 3 experiments for weeks 1–4 with go/kill criteria, written as **plans, not code**. **Experiment 1 must be a plan to demonstrate one measurable self-improvement generation on a toy verifier domain within weeks 1–2**: inputs, procedure as numbered plain-language steps, what is measured and how, a random-reward control arm, pass criterion, kill criterion, compute and wall-clock time. Then the top risks.

### Phase B: Red team (5 agents, one per proposal)
Each red-teamer attacks one proposal and writes `work/critique_<name>.md` (≤ 2,000 words). Persona: a top ML/RL researcher who knows exactly how self-play systems fail, plus a Kaggle grandmaster who knows T4/TPU limits, plus a benchmark-integrity auditor. Attack in this priority order:
1. **The RSI loop**: does it genuinely escape the teacher ceiling, or will it plateau, stay in its own-level prison, reward-hack, or collapse? Is the cord-cut criterion real and measurable? Is one generation actually runnable on the free quota, and how long does it take? Can the metrics detect fake progress? Is "find the next verifier domain" concrete or hand-waved?
2. FLOP and quota arithmetic. Recompute every number.
3. Each reasoning-row claim and margin, especially: can a core that cannot generate language actually **read** natural-language word problems? Is code generation quality plausible at < 100M? Does it survive perturbed variants (GSM-Plus, GSM-Symbolic)?
4. Fair play: hidden parameters, renderer leakage, contamination, model access to its own verifiers.
5. Is it secretly a Transformer?
6. Does the loop transfer to bigger cores?
7. Are the safety mechanisms enforceable or decorative?
8. Can one person build it in 12 weeks?
For each attack: severity (fatal / major / minor) and a **concrete fix that keeps the ambition**.

### Phase C: Revision (the 5 original designers)
Each designer revises their proposal into `work/proposal_<name>_v2.md`, fixing every fatal and major issue or refuting it with evidence, keeping RSI first, and adding a short changelog.

### Phase D: Judging (3 judges in parallel)
Write `work/judging.md`. Each judge scores all five v2 proposals 0–10 under one lens, lists the best ideas worth grafting and the fatal flaws, and picks a winner:
- **RSI credibility (weight ×2)**: does the loop really escape the teacher ceiling and keep compounding, with real defenses and detectors, a real cord-cut criterion, and a feasible generation cycle, safely bounded?
- **Feasibility**: can one person with 0 money build it in 6–12 months on the stated hardware?
- **Boss ladder**: probability that generation N beats GPT-3.5 by the "아득히" margin under the fair-play rules, and credibility of climbing further.

### Phase E: Synthesis (chief architect)
Design the final system using the strongest RSI design as the spine and grafting only ideas that earn their parameters or their generation time. Give it a memorable codename. Write `work/blueprint.md` (English, 8,000–11,000 words) with a **CANONICAL NUMBERS SHEET at the top**: per-module parameter table and total (< 100M), generation cycle time and compute per generation vs quota, bootloader tokens/steps, target score and margin per reasoning row and the generation at which each is expected, RSI metric thresholds (cord-cut, kill), inference cost per query, renderer model. Every later number must match this sheet.

### Phase F: Audit (2 agents in parallel)
- **Completeness critic** → `work/audit_completeness.md`: what is missing, vague, or hand-waved, with RSI escape first.
- **Numbers auditor** → `work/audit_numbers.md`: recompute every parameter count from the stated dimensions and layers, every FLOP estimate, compute per generation and generations per week vs quota, storage sizes, inference latency, and every benchmark number against the research files. Output a table: stated / recomputed / verdict / corrected.
Then the chief architect applies every valid fix and writes **`work/blueprint_final.md`**.

### Phase G: Korean document (13 writers in parallel, one per section)
Each writer writes exactly one file in `sections/`, using **only** `work/blueprint_final.md` as the source of truth for numbers. Files and contents:

| File | Title | Content |
|---|---|---|
| `sections/00.md` | 표지 · TL;DR · 목차 | Codename, one-page summary: this project is an RSI engine; the aphasic genius lives inside it; GPT-3.5 is the first rung; one ASCII diagram of the loop; scoreboard targets and margins; parameter headline; 12-week headline; table of contents for 01–12. |
| `sections/01.md` | 피벗 선언: 말빨은 때려치고, 추론에 몰빵하고, RSI를 태운다 | The two pivots and why (fluency wall; Hawking and the football-coach-with-interpreter analogies; science, drug discovery, and code are reasoning; teacher ceiling, even Claude is a tin can). Rules of the game: scoreboard, "아득히" margin, renderer firewall, RSI integrity, fair play, anti-contamination. |
| `sections/02.md` | RSI 엔진: 이 프로젝트의 심장 | **The longest section.** Teacher ceiling vs verifiers (AlphaGo Zero, TinyGSM, R1-Zero, AlphaProof); the generation loop step by step; levels L1–L5; bootloader and cord-cut (rocket stage separation); fuel after the cut; S-curve staircase over verifier domains (Moore's-law analogy) and how the loop finds the next domain; the growth equation (progress ∝ compute × efficiency × verifier coverage × curriculum quality); the four curve shapes (exponential, S-curve, log, staircase); the metrics dashboard; compute per generation. |
| `sections/03.md` | 곡선 킬러 대처법: 정체 · 자기수준 감옥 · 보상 해킹 · 모델 붕괴 | Each curve-killer: symptom, evidence it happened to others, our mechanism, our detector, our rollback/kill rule. Highlight the fake-exponential danger of reward hacking and the in-loop vs held-out gap monitor. |
| `sections/04.md` | 제1원리: 왜 선생님은 천장이고 채점기는 천장이 없나 + 왜 입과 뇌를 분리하나 | Distillation ceiling vs verifier-driven improvement; what reasoning is computationally; evidence that understanding is cheaper than generating; how every parameter goes to reasoning and to being a fast learner inside the loop. |
| `sections/05.md` | 아키텍처 총설 & 모듈 상세 | Full ASCII diagram (input reader, proposer, solver, verifiers/judge, library, tools), dataflow, parameter table, why not a Transformer; then module-by-module spec with shapes, dimensions, layers, state sizes, objectives, interfaces, and T4/TPU kernel notes. |
| `sections/06.md` | 실어증 천재의 통역사: 출력기 & I/O 인터페이스 | Structured output schema with real examples, renderer choice and prompt contract, blindfold enforcement, faithfulness checks, renderer-alone control, and one complete round trip: problem → core's formal output → renderer's explanation. |
| `sections/07.md` | 데이터 = RSI 연료 | Bootloader data (licensed, no GPT-3.5/4), procedural generators per verifier domain, self-play data, decontamination and perturbed variants, Korean reading data, and how the data mix shifts from teacher to self-generated across generations. |
| `sections/08.md` | 부트로더 학습 & 세대 운영 & 0원 컴퓨트 작전 | Stage-0 recipe; generation operations (scheduling generations across Kaggle T4/TPU sessions and the laptop, checkpoints, cycle time); FLOP math vs quotas; T4/TPU engineering. |
| `sections/09.md` | 보스 사다리: GPT-3.5는 슬라임이다 | Per reasoning row: attack, expected score vs GPT-3.5, margin, at which generation. Then the ladder GPT-4 → o1/R1 → frontier tin cans → top human experts → north star, with what must be true at each rung and honest probabilities. End with a short ideal-scenario story of a future RSI dashboard. |
| `sections/10.md` | 스케일링 & 북극성: 0.1B → 10B → ASI (신약·난제·코드) | How the RSI loop transfers to larger cores, what 1B/10B look like, the compute path (humans win grants with results), drug discovery with simulators as verifiers and information-gain experiment selection. |
| `sections/11.md` | 능력 비례 안전장치: 빨리 달리려면 잘 서야 한다 | Stage-gated safety, presented seriously (humor allowed, content firm). See §6 of this brief. |
| `sections/12.md` | 로드맵 · 첫 실험 · 리스크 · 부록 | 12-week roadmap where weeks 1–2 produce a measured self-improvement generation on a toy verifier domain; the first experiment as a complete plan (numbered steps, measurement, control arm, pass and kill criteria, compute and time) that an engineer could start implementing tomorrow; risk register and kill criteria; rejected alternatives; references (only real URLs from the research files or sources you actually opened). |

Finally, concatenate `sections/00.md` … `sections/12.md` in order into **`FINAL_기획서.md`** in this directory.

---

## 5. Style guide for the Korean document (Phase G)

The user explicitly asked for this to be **"개웃기게 길게"** — hilariously funny and long. They will be disappointed by a dry document.

- **Language**: Korean. Technical terms may stay in English (RSI, verifier, Gated DeltaNet, FLOPs, etc.).
- **Tone**: high-energy, Elon-Musk-level ambition meets Korean internet humor: memes, 드립, playful exaggeration, absurd but apt analogies. Mild casual swearing is okay sparingly (the user uses it), but no slurs and no hate.
- **Running gags to use across sections** (use them consistently):
  - Our model is an **"실어증 천재"**: it cannot talk, but it thinks insanely well. The renderer is its **통역사** ("우리 애가 말하길요~").
  - Frontier teachers, including Claude, are **"깡통 선생님"** that we must surpass.
  - The RSI loop is a **rocket**: the teacher is stage 1 and separates; verifiers are the infinite fuel of stage 2.
  - **GPT-3.5 is the first slime** after the tutorial, not the final boss.
- **The jokes live in the prose. The technical content never jokes.** Numbers, tables, equations, tensor shapes, parameter counts, FLOP math, diagrams, and code must be precise and must match `work/blueprint_final.md` exactly. Never sacrifice correctness for a joke.
- Every section starts with `# NN. <title>` and uses `##` / `###` below it. Use tables, equations, ASCII or mermaid diagrams, and schemas generously (code blocks are fine for diagrams and schemas). **Do not write executable programs or pseudocode scripts**; describe procedures as numbered plain-language steps.
- **Length**: each section roughly 2,500–5,000 Korean words; section 02 may be longer. Long because it is dense, not because it repeats itself.
- Label claims **[검증됨: url]**, **[추정]**, **[가설]**.
- Cross-reference other sections ("→ 05장 참고") instead of repeating them.

Example of the desired voice (the content of this example is illustrative only, do not copy it):

> 깡통 선생님한테 과외받으면 잘해 봐야 **고급 깡통**이 된다. 🥫✨ 그래서 우리는 선생님 대신 **채점기**를 고용한다. 바둑 규칙은 이세돌보다 바둑을 못 두지만 AlphaGo Zero를 이세돌보다 강하게 만들었다 [검증됨: …]. 채점기는 답을 안 알려준다. 맞았는지 틀렸는지만 말한다. **그래서 천장이 없다.**

---

## 6. Mandatory stage-gated safety (must appear in the design and in section 11)

Safety strength scales with capability. "Fast cars need better brakes."

| Stage | Mechanisms (cumulative) |
|---|---|
| From day 1 | Verifiers, reward functions, and eval sets are **immutable to the model** (read-only, separate process/sandbox). Per-generation **audit log**. Reasoning traces are executable/inspectable programs or formal objects wherever possible. Fixed compute/storage/network caps; the system **never acquires resources on its own**. **Automatic kill switch** on anomalies (e.g. in-loop score jumps without held-out gains, any attempt to read or modify verifiers/evals, unexpected network or file access). |
| L4 and above | Every change to the training procedure, data generators, or hyperparameters requires **human approval** before it enters the next generation. |
| Before any biology/chemistry work | No bio/chem tools or data shards without **external dual-use review**. Hazardous generation (pathogen enhancement, toxin synthesis) is blocklisted with a non-learned filter. Anything a wet lab could act on requires **human expert review**. |
| Always | The user's rule: if anything could harm people, stop immediately. |

---

## 7. Quality checklist (the chief architect and auditors must verify every item before Phase G)

- [ ] Parameter table sums to < 100,000,000 and includes embeddings, the input reader, verifiers/judges, and all heads. The arithmetic is shown.
- [ ] No softmax/Hopfield attention over the token sequence anywhere in the core.
- [ ] The RSI loop has an explicit, measurable **cord-cut criterion** and states exactly what fuels improvement after the cut.
- [ ] Each curve-killer has a mechanism, a detector, and a rollback/kill rule.
- [ ] One RSI generation has an explicit compute cost and wall-clock time that fit the free quota.
- [ ] Every GPT-3.5 target number matches `reason_baseline.md`; every margin uses the G rule.
- [ ] The renderer firewall is enforced by design (blindfold, core-only scoring, renderer-alone control, input read by core).
- [ ] No GPT-3.5/GPT-4-generated data anywhere (TinyGSM and MetaMathQA are banned).
- [ ] Experiment 1 is a complete plan (not code) for a measured self-improvement generation in weeks 1–2, with a random-reward control arm, specified well enough that an engineer could start implementing it tomorrow.
- [ ] Safety mechanisms from §6 are present and enforceable, not decorative.
- [ ] Every claim is labeled; no invented URLs or scores.
- [ ] The Korean sections are consistent with each other and with `work/blueprint_final.md`.

---

## 8. Forbidden moves

- Do not write executable code, Python, pseudocode scripts, or simulated program output (such as print statements) anywhere. Equations, tables, diagrams and output schemas are allowed and required. Implementation is done later by a separate engineer.
- Do not write scripts to generate, assemble, or count words in documents. Write documents directly. Word counts are soft guidance.
- Read `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md` before Phase B and treat it as binding input.
- Do not design anything for chat, writing, or fluency benchmarks.
- Do not use a Transformer backbone, or a "hybrid" that hides attention layers.
- Do not count the datastore or zero-parameter tools as parameters, but do count every learned weight, including learned retriever keys and judge networks.
- Do not invent sources, numbers, or papers. Do not claim a result is achieved when it is a projection.
- Do not make GPT-3.5 the final goal. It is the first rung of the ladder.
- Do not modify, delete, or write anything outside `/home/user/hip/agy-blueprint/`. Treat `/home/user/hip/asi-blueprint/` as **read-only reference material**.
- Do not install system packages, open network listeners, or run long-running background processes. Web search for verification is allowed.

---

## 9. Deliverables (all inside `/home/user/hip/agy-blueprint/`)

1. `work/` — all proposals, critiques, judging, blueprint drafts, and audits.
2. `work/blueprint_final.md` — the canonical English technical blueprint.
3. `sections/00.md` … `sections/12.md` — the 13 Korean sections.
4. `FINAL_기획서.md` — the 13 sections concatenated in order.
5. `work/RUN_SUMMARY.md` — a short English summary: codename, the winning RSI design and why, the parameter total, one RSI generation's cost and time, the expected generation at which each GPT-3.5 row falls, the top 5 risks, and any brief requirement you could not satisfy and why.
