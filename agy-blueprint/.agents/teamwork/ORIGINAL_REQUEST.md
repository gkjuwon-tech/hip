# Original User Request

## 2026-10-19T09:02:49Z

# Teamwork Project Prompt

> Requested team: Full team

> **BINDING SPEC:** Every agent on the team must read `/home/user/hip/agy-blueprint/BRIEF.md` in full before doing any work. This prompt is only a summary; wherever they differ, `BRIEF.md` wins.

Design, from scratch, a **recursive self-improvement (RSI) engine** whose learner is a **tiny (<100M parameter), non-Transformer, reasoning-only "aphasic genius" core** that never generates natural language, and deliver a complete, honest, technically precise, and very funny **Korean design document (기획서)** for it.

Working directory: `/home/user/hip/agy-blueprint`
Integrity mode: benchmark
Reference material: Read-only research files in `/home/user/hip/asi-blueprint/research/` (`reason_baseline.md`, `rsi.md`, `reason_methods.md`, `io_split.md`, `tiny_sota.md`, `archs.md`, `externalize.md`, `compute.md`, `data.md`) and specification `/home/user/hip/agy-blueprint/BRIEF.md`.

---

## 1. Core Decisions & Philosophy (§1)

### Decision 1: Drop Language Generation Entirely ("The Aphasic Genius")
- Core model puts 100% capacity into reasoning. Comprehends inputs, never generates natural language.
- Emits formal outputs only: final answers, programs/code, proofs, structured reasoning traces, tool calls.
- Separate open LLM ("renderer" / interpreter) translates formal outputs into fluent text for humans.
- Chat, writing, MT-Bench, AlpacaEval, Arena, and HellaSwag-style fluency are strictly out of scope.

### Decision 2: The Real Project is RSI (Escaping the Teacher Ceiling)
- Distilling reasoning traces from frontier teachers (Claude Opus, Gemini Argon, GPT-6 Astra) caps the student at the teacher's level; frontier models are "tin cans" (깡통) compared to the north star.
- The teacher is merely a bootloader (rocket stage 1), never the ceiling.
- Post-bootloader improvement must come from sources that do not need a smarter teacher: exact verifiers and environments (Python execution, unit tests, SymPy, Z3, Lean, game rules, simulators), self-play, self-generated curricula (proposer vs solver), growing skill/program libraries, and training procedure self-improvement.
- Historical proof: AlphaGo Zero beat human-data AlphaGo with zero human games because Go rules are a perfect verifier. TinyGSM student + verifier beat its GPT-3.5 teacher (81.5% vs 77.4% on GSM8K). DeepSeek-R1-Zero improved via pure RL on checkable answers. AlphaProof improved by proving millions of Lean theorems. **The key is the verifier, not the teacher.**

### Decision 3: RSI is the Spine and Definition of Scalability
- Deliverable is a machine that keeps making the reasoner smarter, not a static model.
- Capability grows per generation, and the rate of improvement itself grows (positive second derivative).
- Beating GPT-3.5 on reasoning is only the first rung ("the first slime after the tutorial"). The ladder continues: GPT-4-level → o1/R1-level → frontier tin cans → top human experts → north star.
- Growth shape: staircase of S-curves. Each S-curve saturates when its verifier domain is exhausted; a new S-curve begins when the loop opens a new verifier domain (arithmetic → algebra → formal proof/Lean → code with tests → simulators → real experiments). Seen from afar, the staircase appears exponential (analogous to Moore's law). **Finding or building the next verifier domain is itself an RSI skill that must be designed.**

---

## 2. 7-Phase Team Execution Process (§4)

### Phase A: 5 Parallel Independent Proposals (`work/proposal_<name>_v1.md`)
Each designer champions ONE RSI philosophy (3,000–5,000 words):
1. `library` — **Library-growing RSI (neuro-symbolic compiler)**: compiles problems into typed programs/formal statements over zero-parameter executors (Python, SymPy, Z3, Lean). Loop = DreamCoder wake-sleep: solve with search → verify → compress solutions into growing skill library → retrain on "dreams". Capability compounds because each library generation shortens future solutions.
2. `depth` — **Depth-distillation RSI (latent recursive reasoner)**: weight-tied deep recursion on attention-free core (Gated DeltaNet / RWKV-7). Loop = spend large test-time depth/search to solve harder problems → verify → distill deep thinking into shallow thinking so next generation starts higher (amplify-then-distill).
3. `search` — **Expert-iteration RSI (AlphaZero-style)**: tiny policy+value network driving MCTS over reasoning steps with exact verifiers as reward. Search improves policy, policy improves search. Absolute-Zero proposer-vs-solver self-play keeps curriculum at edge of competence.
4. `curiosity` — **Curiosity / world-model RSI**: JEPA-style latent model predicts problem states and verifier/tool/experiment outcomes; loop selects learning targets by expected learning progress / information gain; later selects real experiments to run (drug-discovery north star).
5. `minimal` — **Minimal fastest-cycle RSI**: simplest proven stack (attention-free core + recurrent depth + tools + exact verifiers + RLVR + proposer/solver self-play), optimized for shortest generation cycle on free compute and cleanest measurement of improvement per generation. Baseline that every exotic module must beat at equal FLOPs and wall-clock.

#### 12 Mandatory Proposal Sections (In Exact Order, RSI First):
1. **Thesis**: why teacher distillation caps at teacher, what replaces teacher, why loop compounds, why dropping language generation makes each generation cheaper.
2. **The RSI Engine**: generation loop step-by-step (propose → solve/search → verify → select → train → expand library/curriculum/verifiers); RSI levels L1–L5; teacher bootloader & explicit measurable cord-cut criterion; post-cut fuel; S-curve staircase over verifier domains & mechanism to find/build next domain; generation cycle time & compute per generation on free quota.
3. **Curve-Killer Defenses** (mechanism + detector + rollback/kill rule): plateau, own-level prison, reward hacking (fake exponential: in-loop vs held-out gap), model collapse, compute wall, real-experiment latency.
4. **RSI Metrics Dashboard**: teacher-exceedance, improvement per generation, second derivative, held-out vs in-loop gap, generations without new human/teacher data, efficiency gains per generation.
5. **The Core as Learner**: ASCII component diagram, dataflow, structured output format to renderer, exact parameter table (< 100M total, including embeddings), why it is not a Transformer.
6. **Fair-Play Rules**: renderer firewall (blindfold, core-only scoring, renderer-alone control, direct core reading) + RSI integrity (immutable verifiers/evals).
7. **The Boss Ladder**: per reasoning row of scoreboard (`reason_baseline.md`), generation and score beating GPT-3.5 by G margin ($G = T + \max(10, (100-T)/3)$); subsequent rungs (GPT-4, o1/R1, frontier), prerequisites, honest probabilities.
8. **Bootloader Recipe (Stage 0) & Data**: licensed data (no GPT-3.5/4 data), explicit 6ND FLOP math vs quotas, T4/TPU engineering notes, inference cost per query.
9. **Renderer Choice & Integration**: open model choice, execution location, prompt contract, faithfulness check.
10. **Scaling**: RSI loop transfer 0.1B → 1B → 10B (warm-starting, carrying library/curriculum), compute acquisition pathway (grants won by published results), drug-discovery north star via simulators and information-gain experiment selection.
11. **Stage-Gated Safety Mechanisms**: Day 1, L4+, and bio/chem safeguards (§6).
12. **First 3 Experiments (Weeks 1–4) & Top Risks**: Experiment 1 must demonstrate one measurable self-improvement generation on a toy verifier domain in weeks 1–2 with runnable pseudocode; risk register and kill criteria.

### Phase B: 5 Red Team Critiques (`work/critique_<name>.md`, ≤ 2,000 words)
Attacking RSI loop, FLOP/quota math, scoreboard claims/margins, fair play, hidden attention, transferability, safety, and 12-week feasibility with severity ratings and concrete fixes.

### Phase C: 5 Revisions (`work/proposal_<name>_v2.md`)
Resolving all fatal/major critique issues with evidence and short changelogs.

### Phase D: Judging (`work/judging.md`)
Score proposals 0–10 across RSI credibility (weight ×2), Feasibility, and Boss ladder; identify graftable ideas; pick winning spine.

### Phase E: Chief Architect Synthesis (`work/blueprint.md`, 8,000–11,000 words)
Synthesis with CANONICAL NUMBERS SHEET at the top.

### Phase F: Parallel Audits & Final Blueprint
`work/audit_completeness.md`, `work/audit_numbers.md` (recomputing every parameter, FLOP, quota, benchmark number), resulting in `work/blueprint_final.md`.

### Phase G: 13 Korean Sections (`sections/00.md` … `sections/12.md`) & `FINAL_기획서.md`
Using only `work/blueprint_final.md` as ground truth.

| File | Title | Content |
|---|---|---|
| `sections/00.md` | 표지 · TL;DR · 목차 | Codename, one-page summary: RSI engine mission, aphasic genius core, GPT-3.5 as first rung; loop ASCII diagram; scoreboard targets and margins; parameter headline (<100M); 12-week headline; TOC for 01–12. |
| `sections/01.md` | 피벗 선언: 말빨은 때려치고, 추론에 몰빵하고, RSI를 태운다 | The two pivots and why (fluency wall; Hawking & football-coach-with-interpreter analogies; science, drug discovery, and code are reasoning; teacher ceiling, Claude is a tin can). Rules: scoreboard, "아득히" margin, renderer firewall, RSI integrity, fair play, anti-contamination. |
| `sections/02.md` | RSI 엔진: 이 프로젝트의 심장 | Teacher ceiling vs verifiers (AlphaGo Zero, TinyGSM, R1-Zero, AlphaProof); generation loop step by step; levels L1–L5; bootloader & cord-cut (rocket stage separation); fuel after cut; S-curve staircase over verifier domains (Moore's law analogy) & finding next domain; growth equation; 4 curve shapes; metrics dashboard; compute per generation. |
| `sections/03.md` | 곡선 킬러 대처법: 정체 · 자기수준 감옥 · 보상 해킹 · 모델 붕괴 | Each curve-killer: symptom, precedent evidence, mechanism, detector, rollback/kill rule. Highlight fake-exponential danger of reward hacking & in-loop vs held-out gap monitor. |
| `sections/04.md` | 제1원리: 왜 선생님은 천장이고 채점기는 천장이 없나 + 왜 입과 뇌를 분리하나 | Distillation ceiling vs verifier improvement; what reasoning is computationally; evidence understanding is cheaper than generating; why every parameter goes to reasoning and fast in-loop learning. |
| `sections/05.md` | 아키텍처 총설 & 모듈 상세 | Full ASCII diagram (reader, proposer, solver, verifiers/judge, library, tools), dataflow, parameter table, why not Transformer; module-by-module spec with shapes, dimensions, layers, state sizes, objectives, interfaces, T4/TPU kernel notes. |
| `sections/06.md` | 실어증 천재의 통역사: 출력기 & I/O 인터페이스 | Structured output schema with real examples, renderer choice & prompt contract, blindfold enforcement, faithfulness checks, renderer-alone control, complete round trip (problem → formal output → renderer explanation). |
| `sections/07.md` | 데이터 = RSI 연료 | Bootloader data (licensed, no GPT-3.5/4), procedural generators per verifier domain, self-play data, decontamination & perturbed variants, Korean reading data, shift from teacher to self-generated data. |
| `sections/08.md` | 부트로더 학습 & 세대 운영 & 0원 컴퓨트 작전 | Stage-0 recipe; generation operations (scheduling across Kaggle T4/TPU and laptop, checkpoints, cycle time); FLOP math vs quotas; T4/TPU engineering. |
| `sections/09.md` | 보스 사다리: GPT-3.5는 슬라임이다 | Per reasoning row: attack, expected score vs GPT-3.5, margin, generation reached. Ladder: GPT-4 → o1/R1 → frontier tin cans → top human experts → north star; prerequisites & honest probabilities; future RSI dashboard story. |
| `sections/10.md` | 스케일링 & 북극성: 0.1B → 10B → ASI (신약·난제·코드) | Transfer of RSI loop to larger cores, 1B/10B shapes, compute acquisition via grant wins, drug discovery with simulators as verifiers and info-gain selection. |
| `sections/11.md` | 능력 비례 안전장치: 빨리 달리려면 잘 서야 한다 | Stage-gated safety: Day 1 immutability/kill switches, L4+ human approval, bio/chem dual-use review, human harm stop rule. |
| `sections/12.md` | 로드맵 · 첫 실험 · 리스크 · 부록 | 12-week roadmap (weeks 1–2: measured self-improvement generation on toy verifier domain); Experiment 1 runnable pseudocode; risk register and kill criteria; rejected alternatives; references with verified URLs only. |

Concatenate `sections/00.md` … `sections/12.md` in order into **`FINAL_기획서.md`**, and write **`work/RUN_SUMMARY.md`**.

---

## 3. Style Guide & Voice (§5)

- **Language & Tone**: Korean. High-energy, Elon-Musk-level ambition meets Korean internet humor (memes, 드립, playful exaggeration). Mild casual swearing acceptable sparingly; no slurs or hate.
- **Running Gags**:
  - Model = **"실어증 천재"** (cannot speak, reasons brilliantly).
  - Renderer = **"통역사"** ("우리 애가 말하길요~").
  - Frontier teachers = **"깡통 선생님"**.
  - RSI loop = **"로켓"** (teacher is stage 1 booster; verifiers are infinite stage 2 fuel).
  - **"GPT-3.5는 튜토리얼 직후 만나는 첫 번째 슬라임"**.
- **The jokes live in the prose. The technical content never jokes.** Numbers, tables, equations, tensor shapes, parameter counts, FLOP math, diagrams, and code must be mathematically and physically precise, matching `work/blueprint_final.md` exactly.
- **Length**: Dense and extensive (2,500–5,000 Korean words per section; section 02 even longer).
- **Claim Labels**: Every claim labeled `[검증됨: url]`, `[추정]`, or `[가설]`.
- **Desired Voice Example**:
  > 깡통 선생님한테 과외받으면 잘해 봐야 **고급 깡통**이 된다. 🥫✨ 그래서 우리는 선생님 대신 **채점기**를 고용한다. 바둑 규칙은 이세돌보다 바둑을 못 두지만 AlphaGo Zero를 이세돌보다 강하게 만들었다 [검증됨: …]. 채점기는 답을 안 알려준다. 맞았는지 틀렸는지만 말한다. **그래서 천장이 없다.**

---

## 4. Forbidden Moves (§8)

- Do not design anything for chat, writing, or fluency benchmarks.
- Do not use a Transformer backbone, or a "hybrid" that hides attention layers.
- Do not count the datastore or zero-parameter tools as parameters, but do count every learned weight, including learned retriever keys and judge networks.
- Do not invent sources, numbers, or papers. Do not claim a result is achieved when it is a projection.
- Do not make GPT-3.5 the final goal. It is the first rung of the ladder.
- Do not modify, delete, or write anything outside `/home/user/hip/agy-blueprint/`. Treat `/home/user/hip/asi-blueprint/` as read-only reference material.
- Do not install system packages, open network listeners, or run long-running background processes. Web search for verification is allowed.

---

## 5. Acceptance Criteria & Quality Checklist (§7)

- [ ] Parameter table sums to < 100,000,000 and includes embeddings, input reader, verifiers/judges, and all heads with verified arithmetic shown.
- [ ] No softmax/Hopfield attention over token sequence anywhere in the core.
- [ ] Explicit measurable cord-cut criterion defined, detailing post-cut fuel sources.
- [ ] Each curve-killer has mechanism, detector, and rollback/kill rule.
- [ ] One RSI generation compute cost and wall-clock time fit within free quota (Kaggle T4/TPU, laptop RTX 3060).
- [ ] Every GPT-3.5 target matches `reason_baseline.md`; every margin conforms to the G rule ($G = T + \max(10, (100 - T)/3)$).
- [ ] Renderer firewall enforced by design (blindfold, core-only scoring, renderer-alone control, direct core reading).
- [ ] No GPT-3.5/GPT-4 generated data anywhere (TinyGSM, MetaMathQA banned).
- [ ] Experiment 1 demonstrates measured self-improvement in weeks 1–2 on a toy verifier domain with runnable pseudocode.
- [ ] Safety mechanisms from §6 are present and structurally enforceable.
- [ ] Every claim labeled ([검증됨: url], [추정], [가설]); zero hallucinated sources.
- [ ] All 13 Korean sections generated, mutually consistent and matching `work/blueprint_final.md`, concatenated into `FINAL_기획서.md`, plus `work/RUN_SUMMARY.md`.

## Follow-up — 2026-10-19T09:38:06Z

URGENT: BINDING UPDATE FROM USER FOR PROJECT ORCHESTRATOR AND ALL WORKERS.

An external audit of all five Phase A proposals has been placed at `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md`. It is BINDING input for Phase B onward (BRIEF.md hard constraints still apply).

Three immediate process changes:
1. Word counts are soft guidance only. The library and curiosity designers must stop counting or trimming words and submit their current proposals now. Do not write scripts that generate or assemble documents (`generate_proposal.py`, `assemble.py`); write documents directly.
2. NO CODE anywhere in proposals, the blueprint, or the Korean sections. Experiments must be specified as plans only (inputs, procedure, measurement, control arm, pass and kill criteria, compute and time); implementation will be done by a separate engineer.
3. Start Phase B immediately. Every red-teamer must read `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md` and attack its sections 1 and 2 in addition to its own findings, and every Phase C revision must fix them while keeping the ideas in section 3.

Relay this immediately to the Project Orchestrator (9d100903-3938-4939-8cb4-21dca26a45b4) and ensure every worker enforces these rules.

## Clarification — 2026-10-19T09:39:30Z

CLARIFICATION OF PREVIOUS DIRECTIVE — READ LITERALLY:

(A) "NO CODE" means ONLY: no executable programs, no Python/pseudocode scripts, no functions or loops, no simulated output like print statements. Everything else is ALLOWED AND REQUIRED: equations and formulas, parameter tables with arithmetic, FLOP calculations, ASCII diagrams in code blocks, mermaid diagrams, the JSON or schema of the structured output format, configuration values, and pseudo-algorithm descriptions written as numbered plain-language steps. Do not delete any table, equation, diagram, or schema.

(B) Replaces checklist item "runnable pseudocode" in ORIGINAL_REQUEST.md and BRIEF.md: Experiment 1 must instead be a complete experiment PLAN (inputs, procedure as numbered plain-language steps, measurement, random-reward control arm, pass criterion, kill criterion, compute and wall-clock time).

(C) Do NOT start Phase B on an unfinished file. Wait until the curiosity designer submits a COMPLETE proposal containing all 12 sections, then run Phase B on all five proposals.

(D) "Submit now" means stop trimming words, not stop writing: every proposal must still contain all 12 required sections in order.

Ensure all workers receive this clarification immediately.
