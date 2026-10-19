# ANVIL-0: a depth-distillation RSI engine for an aphasic reasoner

*Proposal recursive_v1. Philosophy: **amplify with depth, verify exactly, distill depth into instinct, repeat.** Depth is the hammer, the verifier is the anvil, and every generation's shallow instinct is the previous generation's deep thought.*

Labels: [VERIFIED: url] means read in this session or in the on-disk research files. [ESTIMATE] means my arithmetic or judgement. [HYPOTHESIS] means an untested design bet. Parameter and FLOP figures come from a script run in this session against the ALEMBIC-0 GDN layer formula (5,012,618 params per layer at d=640).

---

## 1. Thesis: the RSI bet

**1.1 Why teacher distillation caps at the teacher.** Imitation's fixed point is the teacher. A verifier helps the student pass it a little: TinyGSM's 1.3B generator plus 1.3B verifier scored 81.5 on GSM8K against its GPT-3.5 teacher's 77.4 [VERIFIED: https://arxiv.org/html/2312.09241]. Then the data runs out, because the teacher wrote every problem. RL on a fixed prior does not escape either: base models beat RLVR models at large pass@k, and current RL "does not elicit fundamentally new reasoning patterns" [VERIFIED: https://arxiv.org/abs/2504.13837]. Systems that went far beyond every teacher shared one recipe: **an improvement operator, plus an exact judge, plus distillation of the operator's output back into the policy.** AlphaGo Zero used MCTS plus game rules and beat AlphaGo Lee 100-0 after 3 days on 4 TPUs, with no human data [VERIFIED: https://deepmind.google/blog/alphago-zero-starting-scratch/].

**1.2 What replaces the teacher: the model's own deep self.** ANVIL's improvement operator is **depth plus search plus test-time training**, run on the same weights. A weight-tied core gets smarter with depth r. Huginn scores 0.0 on GSM8K at r=1 and 42.1 at r=32, with test-time compute "equivalent to about 50B params" [VERIFIED: https://arxiv.org/html/2502.05171]. A 7M TRM recursing with deep supervision reaches 44.6% on ARC-AGI-1 [VERIFIED: https://arxiv.org/html/2510.04871]. So for every checkpoint M_g there is a stronger amplified version A(M_g): run at r=32, N=16 candidates, with tree search over the program IR and TTT where demonstrations exist. **A(M_g) is the teacher for M_{g+1}.** Exact verifiers decide which of its outputs are true, and distillation trains M_{g+1} to produce at depth r₀=4 what M_g produced only at depth 32 with search. M_{g+1}'s own deep version then stands on a higher floor.

**1.3 Why this compounds and does not merely converge** [HYPOTHESIS, with mechanisms]:
1. **The fuel is the amplification gap**, Δ_amp(g) = solve(A(M_g)) − solve(M_g at r₀) on a fresh problem stream. Search reaches solutions outside the policy's support, which answers the pass@k-narrowing critique of RLVR.
2. **The depth ratchet.** Once the shallow policy has absorbed what used to need depth 32, depth 32 then operates on top of that. Depth gains stack on a better-trained iterate: in HRM, training with refinement adds 15+ points even to single-pass inference [VERIFIED: https://arcprize.org/blog/hrm-analysis].
3. **The cost ratchet gives the positive second derivative.** A problem class that moves from (r=32, N=16) to (r=4, N=1) gets about 90x cheaper per attempt (§2.6), and the freed compute goes to the frontier. Under fixed quota, efficiency is the multiplier we control, and depth distillation raises it by construction.
4. **The library and curriculum ratchets.** Macros (L3) and a regret-rewarded proposer (L2) persist across generations and across core sizes.

**1.4 Why dropping language generation makes each generation cheaper** [ESTIMATE]:
- **Shorter rollouts.** Outputs are 40–200 tokens of typed IR, not 300–1,000 tokens of chain-of-thought prose, so rollouts are 3–10x cheaper.
- **A small output head.** A 4,096-token formal vocabulary needs a 2.6M-param head; Gemma 3 270M spends 170M on embeddings [VERIFIED: https://developers.googleblog.com/en/introducing-gemma-3-270m/].
- **A reward bit on every rollout.** Every output is machine-checkable.
- **Latent thinking.** Reasoning lives in depth, so thinking longer costs FLOPs on a fixed-size state, not KV-cache growth.

**Talking is the expensive, unverifiable part. Remove it and the loop runs entirely in verifiable territory.**

---

## 2. The RSI engine

### 2.1 One generation, step by step

```
            ┌──────────────────────── GENERATION g (≈1 week on free quota) ───────────────────────┐
 archive ──►│ 1 PROPOSE   proposer (same trunk) + 0-param generators emit problems in typed IR,    │
 library    │             rendered to NL by templates; difficulty targeted at the learnability edge│
 curriculum │ 2 SCREEN    M_g at r0=4, N=1  → solved-easy (discarded/replay) | unsolved → frontier │
            │ 3 AMPLIFY   frontier: r ∈ {8,16,32} ladder × N≤16 × IR beam/MCTS (+TTT on ARC)      │
            │ 4 VERIFY    serialized IR → sandboxed exact verifiers (separate process, read-only)  │
            │ 5 SELECT    keep verified; dedup by execution signature; regret score; caps per skill│
            │ 6 DISTILL   M_{g+1} ← M_g + loss(shallow r0 output = verified deep output)          │
            │             + latent target (deep final state) + deep supervision + GRPO sharpen    │
            │ 7 COMPRESS  Stitch-style compression of verified programs → new IR macros (L3)      │
            │ 8 GATE      frozen held-out evaluator: Δ_fresh, H-gap, diversity, controls → promote │
            │             or roll back; human gate for any L4 diff                                │
            └──────────────────────────────────────────────────────────────────────────────────────┘
```

Details that matter:

- **Step 1, PROPOSE.** Three sources:
  - (a) **0-param symbolic generators**: Reasoning Gym (100+ verifiable task families with difficulty control) [VERIFIED: https://arxiv.org/pdf/2505.24760], re-ARC training-split generators [VERIFIED: https://arxiv.org/pdf/2404.07353], and our own SymPy/Z3/PlanBench-style generators.
  - (b) **Backward construction**: execute a composition of k library macros, then render the problem from the program. This is the AlphaGeometry recipe [VERIFIED: https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/]. Raising k guarantees problems **above** the current level.
  - (c) **A learned proposer**: the trunk in `<PROPOSE>` mode, rewarded by regret (A(M_g) solves the problem, M_g at r₀ does not). Plain minimax "often produces unsolvable environments" [VERIFIED: https://arxiv.org/pdf/2012.02096].
- **Step 2, SCREEN.** A problem solved at r₀, N=1 is not new fuel; it keeps only a small replay weight.
- **Step 3, AMPLIFY.** Escalation ladder: (r=8, N=4), then (16, 8), then (32, 16 with IR-step beam). The first verified success gives the problem's cheapest solving budget, which becomes its **difficulty label**. ARC-like tasks also get TTT with rank-8 core adapters trained on the task's own demonstrations, the method behind the 2024 Kaggle winners' 53.5% on 1×P100 [VERIFIED: https://arxiv.org/html/2412.04604v2].
- **Steps 4–5, VERIFY and SELECT.** Only exact 0-param checkers (§6). Deduplicate by **execution signature** (outputs on 32 probe inputs), cap any skill at ≤5% of new data, and record the solving depth.
- **Step 6, DISTILL** (the core of ANVIL). For a verified (x, deep output y*, deep final state h*₃₂):
  - L = CE(y* | x, r₀=4) + 0.5·CE(y* | x, r=2) + λ·‖P(h_{r₀}) − sg(h*₃₂)‖² + deep supervision.
  - The latent term asks the shallow iterate to land where the deep one converged. P is the shared final norm (0 params). h* is recomputed on TPU, and gradients flow through the last 2 iterations only, as in TRM [VERIFIED mechanism: https://arxiv.org/html/2510.04871].
  - A short GRPO pass then sharpens pass@1. Its gains count only above a random-reward arm, since random rewards gave +21.4 on MATH-500 for Qwen [VERIFIED: https://arxiv.org/abs/2506.10947].
- **Step 7, COMPRESS.** Repeated sub-programs become macros, after DreamCoder [VERIFIED: https://arxiv.org/abs/2006.08381] and LILO/Stitch [VERIFIED: https://arxiv.org/html/2310.19791v2]. They fill 1,024 macro slots reserved in the output head (counted in §5.3).
- **Step 8, GATE.** See §3–4.

### 2.2 RSI levels

| Level | What improves | ANVIL mechanism | Status |
|---|---|---|---|
| L1 | Weights from self-generated verified data | Amplify → verify → depth-distill → GRPO sharpen | From Gen 1 |
| L2 | The curriculum | Regret-rewarded proposer, backward construction with rising k, POET-style stepping-stone archive [VERIFIED: https://arxiv.org/pdf/1901.01753] | From Gen 2 |
| L3 | Skill and program library | Stitch compression into macro slots; latent skill keys (macro-token embeddings in the head) | From Gen 3 |
| L4 | The procedure: r schedule, λ, data mix, generators, kernels, new verifier domains | AlphaEvolve-style search over configs and generators [VERIFIED: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/], scored by 10M-param proxy generations. **Every L4 diff is human-approved.** | From Gen ~8, human-gated |
| L5 | Architecture | North star only, behind the strongest gates (§11) | Not planned |

Skip learned optimizers: VeLO cost about 4,000 TPU-months [VERIFIED: https://arxiv.org/abs/2211.09760] and "is not faster than competing optimizers" [VERIFIED: https://arxiv.org/html/2310.18191v2].

### 2.3 The bootloader and the explicit cord cut

**Stage-1 rocket (Gen 0).** The cold-start wall is real. Search needs non-zero base success [VERIFIED: https://arxiv.org/abs/2408.03314], and the generation-verification gap is smallest in small models [VERIFIED: https://arxiv.org/abs/2412.02674]. Gen 0 trains on three sources (§8):
- 0-param generator streams, which make up the bulk;
- open licensed solutions that we verify and rewrite, such as OpenMathInstruct-2 (CC-BY-4.0) [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2], kept only where re-execution matches the gold answer;
- permissive code.

**Frontier closed teachers (Claude Opus, Gemini Argon, GPT-6 Astra) are not used.** Their APIs cost money, and OpenAI's terms forbid using output "to develop models that compete" [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/]. The ceiling to break is therefore the open bootstrap data plus a runnable reference teacher, **Qwen3-4B in thinking mode** [VERIFIED: https://huggingface.co/Qwen/Qwen3-4B]. Teacher-exceedance is measured against that teacher and against published GPT-3.5 numbers.

**Cord-cut criterion** (all three must hold):
1. **Two consecutive generations** with Δ_fresh > 0 (CI lower bound > 0), using **zero** new teacher or human tokens in those generations.
2. **Ablation parity.** Gen g retrained without the teacher shard scores within CI of, or above, the version with it on the fresh sets.
3. **TE ≥ 10%.** We solve at least 10% of held-out problems that the reference teacher fails at k=8.

After the cut, the teacher shard stays as decaying replay (≤10%). **Hard deadline:** no cut by Gen 6 is a kill-level finding. It would mean the loop only sharpens the bootloader, and that would be published.

**Fuel after the cut:** Δ_amp × verifier coverage. Concretely: rising-difficulty generators, backward-constructed macro compositions, proposer tasks, new domains (§2.4) and **hindsight relabeling**. A program that executes but answers the wrong question is still a verified pair for the problem it *did* solve, which is AZR-style induction from the executor alone [VERIFIED method: https://arxiv.org/html/2505.03335v3].

### 2.4 The S-curve staircase and how the loop finds the next step

Each verifier domain is one S-curve, because Δ_amp shrinks as the shallow policy absorbs it. The envelope stays steep only if new domains open before old ones saturate [ESTIMATE schedule].

| Step | Domain | Exact verifier (0 params) | Opens at | Feeds benchmarks |
|---|---|---|---|---|
| D1 | Arithmetic, Countdown/Game-of-24, expression search | Python eval | Gen 0 | Game of 24, BBH arithmetic |
| D2 | Word problems → programs | Python + SymPy equivalence | Gen 1 | GSM8K, GSM-Plus, SVAMP, DROP |
| D3 | Propositional/FOL logic, constraint puzzles | Z3, FOL prover | Gen 2 | FOLIO, ProofWriter, PrOntoQA, AR-LSAT, BBH logic |
| D4 | Algorithmic traces (sort, Dyck, graph) | Reference implementations | Gen 2 | BBH algorithmic, CRUXEval |
| D5 | ARC-style grids | Exact grid match; TTT | Gen 3 | ARC-AGI-1/2 |
| D6 | Planning | VAL validator | Gen 4 | Blocksworld, Mystery Blocksworld |
| D7 | Code with tests | Sandbox + self-generated tests with mutation adequacy | Gen 5 | HumanEval+, MBPP+, LiveCodeBench |
| D8 | Algebra/number theory, MATH-level | SymPy + numeric cross-check | Gen 6 | MATH, KMMLU-Math, AIME |
| D9 | Formal proof (Lean 4 / Isabelle) | Proof kernel | Gen 8+ | miniF2F (LEGO-Prover with ChatGPT: 57.0/50.0 [VERIFIED: https://arxiv.org/pdf/2310.00656]) |
| D10 | Physics/chemistry simulators (approximate verifiers) | Simulator + held-out real data | North star, stage S3 | Science reasoning |
| D11 | Real experiments | Human-run lab, information-gain selected | North star, stage S4 | Discovery |

**Domain prospecting is itself an RSI skill.** It is human-led at first, then model-proposed and human-approved at L4. A domain is admitted when (a) its exact checker survives a red team of 1,000 adversarial outputs with zero false accepts; (b) A(M_g) has pass@256 > 0 on a seed set (non-zero cold start); (c) pass@1 at r₀ is below 30% (headroom); and (d) its data lifts another domain's fresh score (transfer). **Saturation detector:** Δ_amp < 1 point for 2 generations marks a domain mature. Its fuel share decays to replay and the next domain opens. The model may propose a generator-plus-checker spec, but a human approves the checker; the model never writes the verifier that judges it.

### 2.5 Why this is AlphaZero for latent reasoning

Policy net → shallow core plus judge; MCTS → depth plus IR search plus TTT; game rules → exact verifiers. The new idea: **the improvement operator is mostly the network's own iterated computation**, so it grows with the network and can be distilled at the latent level as well as the output level [HYPOTHESIS].

### 2.6 Generation cycle time and compute per generation

One generation is one week, matching the Kaggle quota reset (toy domains: hours on the laptop). Forward FLOPs per token (2·N_active, script-computed): **2.42e8 at r=4, 1.41e9 at r=32**. A typical attempt is 420 positions (300 input + 120 IR) [ESTIMATE throughout].

| Item | Arithmetic | Result |
|---|---|---|
| Screen attempt (r=4, N=1) | 420 × 2.42e8 | 1.0e11 FLOP |
| Max amplification (r=32, N=16) | 16 × 420 × 1.41e9 | 9.5e12 FLOP (≈ 93× a screen) |
| TPU v5e-8 / week | 20 h × 3,600 × 1.576e15 × 0.30 MFU [peak VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e] | 3.4e19 |
| T4×2 / week | 30 h × 3,600 × 1.3e14 × 0.25 MFU [peak VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf] | 3.5e18 |
| Weekly total | | ≈ 3.8e19 |
| Amplification (55%: TPU rollouts in fixed r-buckets, T4 code/eval) | 2.1e19 / ~1e13 per escalated problem (failures run the full ladder) | **≈ 4e6 problems screened, ≈ 2e6 escalated** |
| Distillation (30%) | per token 6 × 1.21e8 (train at r₀) + 1.41e9 (recompute deep state) ≈ 2.1e9 | ≈ 5e9 tokens/week; the replay buffer (≤1e9 tokens) gets ~5 passes |
| Controls + eval (15%) | random-reward arm, equal-FLOP rejection-sampling arm (ORRERY-0 discipline), evals | — |

**The real bottleneck is CPU verification**, not FLOPs. About 4 vCPU manage roughly 1e2–1e3 Python checks/s [ESTIMATE], and 2e6 escalations × ~8 checks take 4.5–45 h. Mitigations: vectorized checkers for arithmetic and grids (~1e5/s), pre-forked sandbox pools, and the laptop as a 24/7 verification farm. Check the TPU VM's CPU count on day 1.

---

## 3. Curve-killer defenses

All detectors run in the frozen evaluator process.

| Killer | Mechanism | Detector | Kill / rollback rule |
|---|---|---|---|
| **Verifier exhaustion (S-curve)** | Domain staircase and prospecting (§2.4); backward construction with rising k | Δ_amp < 1 pt for 2 gens; > 60% of fuel from mature domains | Open the next domain; empty queue plus Δ_fresh ≤ 0 for 3 gens: stop, publish, escalate to humans |
| **Own-level prison (log curve)** | Depth beyond shallow support; regret proposer; above-level compositions; hindsight relabel; POET archive | **Frontier index** (max k solved at ≥50%) flat 2 gens; **pass@256 of A(M_g)** flat (boundary test of https://arxiv.org/abs/2504.13837); "depth-only discoveries" → 0 | Don't train longer: raise k, re-seed the archive, open a domain. 3 flat gens: roll back, human-approved L2 change |
| **Reward hacking (fake exponential)** | Separate read-only verifier process fed only serialized IR (defeats the `__eq__` and stack tricks METR saw in o3 [VERIFIED: https://metr.org/evaluations/openai-o3-report/]); **two independent checkers** per domain; hidden tests; canary bait | **H-gap** (in-loop minus held-out); checker disagreement; canaries; out-of-grammar output | H > 5 pts with flat held-out, or any canary: **kill switch**, quarantine, roll back, log an incident (precedent: DGM "objective hacking" [VERIFIED: https://sakana.ai/dgm/]) |
| **Model collapse** | **Accumulate, never replace** [VERIFIED: https://arxiv.org/abs/2404.01413v2]; seed data ≥10% forever; signature dedup; ≤5% per skill; latent targets only from verified runs | Distinct-signature ratio; tail coverage; IR-op entropy; **seed-twin** score | Diversity drop > 15% or seed-twin drop > 2 pts: roll back, up-weight tails |
| **Compute wall** | Cost ratchet; compute-optimal allocation [VERIFIED: https://arxiv.org/abs/2408.03314]; vectorized verifiers; humans apply to TRC [VERIFIED: https://sites.research.google/trc/faq/] | **FLOPs per new frontier solve**; compute share on already-solvable items | Rising 3 gens: pause L1, one human-gated L4 efficiency gen |
| **Real-experiment latency** (north star) | Calibrated simulators as approximate verifiers; batched, information-gain-selected, human-run experiments | Simulator-vs-real calibration error; realized information gain | Miscalibrated simulator demoted to prior; nothing bio/chem without §11 |
| **Spurious RL signal** | Random-reward arm every generation | Gain(true) − gain(random) | GRPO step discarded if not significantly above the random arm |

---

## 4. RSI metrics dashboard

Published every generation by the frozen evaluator, on commit-revealed fresh sets regenerated with new seeds each generation.

| Metric | Definition | Healthy sign |
|---|---|---|
| **Δ_fresh(g)** | Score_g − Score_{g−1} on fresh private sets at r₀, N=1, and at a fixed amplified budget | > 0, CI lower bound > 0 |
| **a(g) = Δ_fresh(g) − Δ_fresh(g−1)** | Second derivative of improvement | ≥ 0 over 3-gen windows across the staircase envelope. Decay inside a domain is expected; the envelope must not decay |
| **Δ per 1e18 FLOP** | Compute-normalized gain | Rising means efficiency is compounding |
| **TE(g)** | Of held-out problems the reference teacher fails (k=8), the fraction we solve | Monotone up; ≥10% at cord cut |
| **G0** | Consecutive generations with zero new human or teacher tokens | Grows by 1 each gen after the cut |
| **H-gap** | In-loop proposer-task score − held-out score | Flat or shrinking |
| **Δ_amp(g)** | A(M_g) − M_g@r₀ on the frontier stream | Positive; its value is the fuel gauge |
| **Distillation efficiency η** | (M_{g+1}@r₀ − M_g@r₀) / Δ_amp(g) | ≥ 0.3 [ESTIMATE threshold] |
| **pass@256 boundary** | A(M_g) pass@256 vs A(M_{g−1}) | Rising means the support is genuinely expanding |
| **Diversity / collapse** | Distinct-signature ratio, tail coverage, IR-op entropy | No drop > 15% |
| **Library** | Macro reuse rate; description length per solved task | Reuse up, length down. LEGO-Prover: 50.4 with library vs 47.1 without [VERIFIED: https://arxiv.org/pdf/2310.00656] |
| **Controls** | Random-reward arm; equal-FLOP rejection-sampling-SFT arm at fixed r | ANVIL beats both |

---

## 5. The reasoning core: the best learner inside the loop

### 5.1 Selection criterion

The core is chosen for **generations per week × gain per generation**, not peak score: it is cheap to retrain, carries its own improvement operator (iterable depth), has distillable latent states, and has a bidirectional reader and a ranking judge (never a reward).

### 5.2 Component diagram and dataflow

```
 NL problem (EN/KO), read by the core itself; 0-param normalizer only
   │
   ▼
 [Unigram tokenizer 16k] → [factorized embedding 16k×256→640]
   │
   ▼
 READER: 4 GDN layers, bidirectional (2 L→R, 2 R→L scans, fused)    ← Caduceus-style BiSSM idea
   │  h_read                                                                      [VERIFIED: https://arxiv.org/abs/2403.03234]
   ▼
 ┌─► RECURRENT CORE: 4 weight-tied GDN layers, iterated r times ◄── injector W[h_{i-1}; h_read]
 │      (masked while_loop; halting head; TTT rank-8 adapters on q,k,v,o)
 └──────┘   r ∈ {1..32}; latent "scratch" positions via <think> tokens
   │  h_r
   ▼
 EMITTER: 3 GDN layers → formal head (4,096 tokens: IR ops, digits, types, 1,024 macro slots)
   │        grammar mask (typed IR, 0 params) + <copy> via suffix automaton (0 params)
   ▼
 CoreTrace (typed IR program + values) ──► EXECUTORS / VERIFIERS (separate sandbox process)
   │                                           Python · SymPy · Z3 · VAL · Lean · grid-match
   ▼
 JUDGE: 2 GDN layers scanning R→L over trunk states + fusion MLP
        heads: energy/verify, process value, difficulty, halting-calibration
        (ranks candidates inside search; NEVER a training reward)
   │
   ▼
 answer (scored directly) ──► CoreTrace JSON ──► RENDERER (outside the 100M; blindfolded)
```

Dataflow: the reader runs once; the core iterates r times; the emitter writes grammar-masked IR; executor values re-enter as tool tokens; the judge ranks N candidates; a 0-param controller runs the escalation ladder.

### 5.3 Parameter budget (script-computed, every executed weight)

GDN layer at d=640, 5 heads × 128, SwiGLU 1536, expand_v=1: **5,012,618**.

| # | Module | Composition | Params | Running total |
|---|---|---|---|---|
| 1 | Tokenizer scores (counted) | 16,384 | 16,384 | 16,384 |
| 2 | Input embedding, factorized | 16,384×256 + 256×640 | 4,358,144 | 4,374,528 |
| 3 | Reader | 4 GDN (bidirectional) | 20,050,472 | 24,425,000 |
| 4 | Recurrent core (counted once) | 4 GDN, weight-tied | 20,050,472 | 44,475,472 |
| 5 | Loop injector | Linear(1280→640)+bias | 819,840 | 45,295,312 |
| 6 | Segment embeddings | 8×640 | 5,120 | 45,300,432 |
| 7 | Emitter | 3 GDN | 15,037,854 | 60,338,286 |
| 8 | Formal output head (untied) | 640×4,096 | 2,621,440 | 62,959,726 |
| 9 | Final norm | 640 | 640 | 62,960,366 |
| 10 | Judge tower | 2 GDN, R→L | 10,025,236 | 72,985,602 |
| 11 | Judge input norm | 1,280 | 1,280 | 72,986,882 |
| 12 | Judge fusion MLP (4 heads) | 1280→512→4 | 657,924 | 73,644,806 |
| 13 | Halting head | 640→64→1 | 41,089 | 73,685,895 |
| 14 | Retrieval query projection (Tier-B rows; BM25 is 0 params) | 640×256 | 163,840 | 73,849,735 |
| 15 | TTT adapters (rank 8, core q/k/v/o) | 4 layers × 4 × 10,240 | 163,840 | **74,013,575** |
| | **Headroom** | | | **25,986,425** |
| P1 | Pre-registered pool: +1 core layer (released if the r-slope saturates early) | | 5,012,618 | |
| P2 | Pre-registered pool: +1 judge layer (released if judge depth helps ranking) | | 5,012,618 | 84,038,811 worst case |

Embeddings are 5.9% of the budget, against 63% for Gemma 3 270M. Active params per token are 58.6M at r=1, 121M at r=4 and 706M at r=32. A CI assertion counts every leaf plus tokenizer scores, plus TTT adapters at their maximum rank, and fails the build at ≥ 100,000,000 (from ALEMBIC-0).

### 5.4 Structured output format (CoreTrace v0)

```json
{
  "trace_id": "g12-7f3a",
  "task_kind": "word_problem|logic|grid|code|plan|proof|choice",
  "answer": {"type": "int|rational|expr|choice|grid|code|plan|bool3", "value": "18"},
  "program": "let a = 16 - 3 - 4 ; let b = mul(a, 2) ; return b",
  "steps": [
    {"id": "s1", "op": "sub", "args": ["16","3","4"], "value": "9", "checked": true},
    {"id": "s2", "op": "mul", "args": ["s1","2"], "value": "18", "checked": true}
  ],
  "entities": [{"span_id": "c3", "text": "eggs", "copied": true}],
  "verifier": {"name": "py_exec", "impl_hash": "sha256:…", "verdict": "pass", "cross_check": "sympy:pass"},
  "confidence": {"judge_energy": -3.1, "agreement": "14/16"},
  "budget": {"r": 8, "N": 4, "flops": 1.6e12, "ttt_steps": 0},
  "library_macros_used": ["m0412"]
}
```

There is deliberately **no `question` field**. Copied entity spans carry IDs and are logged, which supports the blindfold check in §6.

### 5.5 Why non-Transformer, and why GDN/RWKV-7

1. **Rule compliance**: no softmax over the sequence anywhere.
2. **Constant-size state makes depth cheap**: 81,920 floats (327.7 KB fp32) per layer, so 16-way batching at r=32 costs megabytes, not a growing KV cache.
3. **The delta rule is in-context error correction**: Gated DeltaNet beats Mamba2 on in-context retrieval [VERIFIED: https://arxiv.org/abs/2412.06464v3], and RWKV-7 recognizes all regular languages [VERIFIED: https://arxiv.org/abs/2503.14456].
4. **Bidirectional scans give a cheap reader and judge**: ELECTRA-Small (14M) scores 79.9 on GLUE vs 78.8 for GPT-1 (117M) [VERIFIED: https://arxiv.org/abs/2003.10555].

**Honest gap:** attention-free TRM loses 15 points on ARC (29.6 vs 44.6) [VERIFIED: https://arxiv.org/html/2510.04871]. Content lookup must come from the delta-rule state, the suffix-automaton copy and object-level grid IR, measured in E3. GDN is the default and RWKV-7 the ablation arm, chosen by r-slope per FLOP on the T4.

---

## 6. Fair play and RSI integrity

**Renderer firewall** (refines the defaults in the same spirit):
1. **Core-only scoring** of `CoreTrace.answer`; CI fails a scoring run if the renderer is loaded.
2. **Blindfold by type**: the renderer accepts only `CoreTrace`, and a static check requires zero shared 8-grams with the question, except logged copied spans.
3. **Renderer-alone controls** per row: (a) raw question (its cheat ceiling); (b) empty or shuffled trace; (c) answer-masked trace. (b) and (c) must be ≤ chance + 2σ.
4. **Counterfactual swap**: the renderer must carry a corrupted answer ≥99% of the time [HYPOTHESIS threshold], or the run is disqualified.
5. **Input purity**: the core reads the problem itself behind a 0-param normalizer, with no dataset-ID feature; a shuffled-format probe must give identical answers.

**General bans:** any >100M net at inference; LLM APIs; GPT-3.5/4 outputs anywhere (TinyGSM, MetaMathQA and UltraChat are out); benchmark-identity routing; and eval items in training data, the datastore or the library (13-gram plus execution-signature dedup). Every row is reported at N=1 and at the full budget, with a ≤60 T4-s/item average.

**RSI integrity (day 1):**
- Verifiers, reward code, eval sets and the evaluator run in a **separate read-only, hash-pinned process**, re-hashed every generation. The learner's only channel to them is a serialized CoreTrace over a pipe, and any access to verifier paths triggers the kill switch.
- Eval sets are physically absent from the training container, checked by hash at boot.
- The proposer creates *problems*, never *checkers*.
- The judge ranks inside search but is never the reward, because self-rewarding loops saturate within a few rounds [VERIFIED: https://mlanthology.org/iclr/2025/wang2025iclr-cream].

---

## 7. The boss ladder

Weeks 1–4 are experiments, weeks 3–6 the bootloader, and Gen 1 starts in week 7, one generation per week. G = T + max(10, (100−T)/3) from reason_baseline. The generation column is an [ESTIMATE] of when each row first clears G; P is my probability of clearing G by Gen 30 at <100M under full rules.

| Row | GPT-3.5 T | Gate G | Expected first-pass Gen | Projected score at that Gen | P(by Gen 30) |
|---|---|---|---|---|---|
| Game of 24 (hard 100 + fresh) | 19 [VERIFIED: https://arxiv.org/pdf/2305.10601] | 46 | Gen 1 | ≥ 95 | 0.9 |
| Blocksworld NL | 6.1 [VERIFIED: https://arxiv.org/html/2305.15771v2] | 37.4 | Gen 6 | 60–80 (formalization-limited) | 0.6 |
| CRUXEval-O | 63.3 | 75.6 | Gen 8 | 78–85 | 0.45 |
| FOLIO / ProofWriter / PrOntoQA | 62.7 / 71.5 / 85.0 [VERIFIED: https://arxiv.org/html/2305.12295v2] | 75.2 / 81.5 / 95.0 | Gen 6 / 6 / 10 | 78 / 90 / 95 | 0.35 (all three) |
| ARC-AGI-1 eval, pass@2 | 10.6 | 40.4 | Gen 15 | 40–48 with TTT | 0.35 |
| KMMLU-Math | 30.3 | 53.5 | Gen 12 | 55–60 | 0.3 |
| BBH | 70.7 | 80.7 | Gen 18 | 80–84 (knowledge tasks drag) | 0.3 |
| SVAMP | 83.0 | 93.0 | Gen 12 | 92–95 | 0.3 |
| GSM8K / GSM-Plus | 78.1 / 61.2 [VERIFIED: https://arxiv.org/html/2402.19255v2] | 88.1 / 74.1 | Gen 20 | 88–91 / 80+ (program invariance) | 0.25 |
| DROP F1 | 70.2 | 80.2 | Gen 20 | 78–82 | 0.25 |
| MATH | 43.1 | 62.1 | Gen 24 | 55–65 | 0.2 |
| MBPP+ / HumanEval+ | 69.7 / 70.7 | 79.8 / 80.7 | Gen 25 | 75–82 | 0.15–0.2 |
| LiveCodeBench (post-freeze) | 22.6 [VERIFIED: https://arxiv.org/html/2403.07974] | 48.4 | Gen 28 | 35–50 | 0.15 |
| LogiQA 2.0 / ReClor / MuSR | 52.4 / 57.4 / 58–79 | 68–89 | Gen 28+ | reading-limited | 0.15 |
| GPQA-D / MedQA / StrategyQA (retrieval) | 30.6 / 63.4 / 70.3 | 53.7 / 75.6 / 80.3 | > Gen 30 | knowledge-limited | 0.05–0.1 |

**Honest aggregate.**
- P(every Tier A+B row ≥ G on one checkpoint by Gen 30, <100M) ≈ **0.04**.
- P(a decisive win on the closed-world formal rows: Game of 24, Blocksworld, logic trio, CRUXEval, ARC-AGI-1, BBH-algorithmic) ≈ **0.4**.
- P(the RSI loop itself is demonstrated, i.e. cord cut plus G0 ≥ 5 plus TE ≥ 10%) ≈ **0.5**. That is the result that matters most.

The win-or-extend rule applies: the boss is never quietly redefined.

**Next rungs** (gates set from scores published at attempt time; only verified reference points are listed):

| Rung | Verified reference points | What must be true | P (36 months) |
|---|---|---|---|
| GPT-4-class reasoning | GPT-4o: MATH-500 74.6, AIME24 9.3, GPQA-D 49.9, LiveCodeBench 32.9 [VERIFIED: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B]; ARC-AGI-1 GPT-4o 5% [VERIFIED: https://arcprize.org/blog/oai-o3-pub-breakthrough] | D8 and D9 (Lean) online; 1B core via TRC; library ≥ 10k macros; Δ per gen still positive at 1B | 0.10 |
| o1/R1-class | R1-Zero AIME24 77.9 pass@1 / 86.7 maj [VERIFIED: https://arxiv.org/html/2501.12948]; Grok-4-thinking ARC-AGI-1 66.7 [VERIFIED: https://arxiv.org/html/2510.04871] | Long-horizon search over lemma libraries; formalization of competition problems by the core itself; ≥1e21 FLOP/yr (grants) | 0.03 |
| Frontier "tin cans" (Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra) | No verified numbers in our material | 10B core, simulator domains, acceleration a(g) > 0 sustained across ≥3 domain transitions | < 0.01 |
| Top human experts → north star | AlphaProof IMO silver 28/42 [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/] | All of the above plus real-experiment loops under §11 | not estimated |

---

## 8. Bootloader recipe (stage 0), data and compute

**Data**, licensed only, with no GPT-3.5/4 outputs:

| Shard | Source | Share of Gen 0 tokens |
|---|---|---|
| Procedural verifiable streams | Reasoning Gym, re-ARC (train split), own SymPy/Z3/PlanBench/CLRS generators | 55% |
| NL→program pairs | Templates plus disclosed **offline** Qwen3-1.7B paraphrase, 13-gram-filtered; GSM-NoOp-style distractors [VERIFIED method: https://arxiv.org/html/2410.05229] | 20% |
| Open verified solutions | OpenMathInstruct-2 rewritten to IR, kept only if re-execution matches | 10% |
| Code | Stack v2 permissive / Python-Edu (ODC-BY), functions with executable tests | 10% |
| Reading | FineWeb-Edu and Korean FineWeb-2: reader span-denoising only | 5% |

**Objectives:** IR cross-entropy under the grammar mask, with r sampled from p(r) ∝ 1/r on 1..16, TRM deep supervision, a halting loss, and judge training on executed outcomes (energy) and Monte-Carlo step values.

**Compute:** Gen 0 trains on about 10B tokens [ESTIMATE]. At about 2.42e8 × 3 (fwd+bwd) × 1.3 (deep supervision) ≈ 9.4e8 FLOP/token, that is 9.4e18 FLOP, about 28% of one TPU week. Chinchilla-optimal for 100M is about 2B tokens, so we overtrain 5x on procedural data, which is cheap.

**T4 and TPU engineering.** There is no bf16 on T4, so run fp16 matmuls with fp32 accumulation and keep Γ, T and the carried state S in fp32. Triton ≥3.3 dropped sm_75 [VERIFIED: https://raw.githubusercontent.com/triton-lang/triton/v3.3.0/README.md], so pin a Triton 3.2-era torch and keep a pure-PyTorch chunkwise fallback. On TPU, use JAX with ALEMBIC-0's chunkwise WY form and r as a masked `while_loop` over static r-buckets {4, 8, 16, 32}. Prefer 4-way data parallelism if 8-way MFU is poor (13% vs 27% [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html]). The 3060 is the kernel-dev and toy-RSI box.

**Inference cost per query** [ESTIMATE]: easy items (r=4, N=1) take about 1e11 FLOP, ≈0.01 T4-s. Hard items (r=32, N=16) take 9.5e12 FLOP, ≈0.3 T4-s of compute; wall-clock of 2–10 s is dominated by sequential depth and verification. ARC with TTT takes 1e14–1e15 FLOP per task, disclosed separately. All of this sits well inside the 60 T4-s/item cap.

---

## 9. Renderer choice and integration

- **Default: Qwen3-1.7B in non-thinking mode** (Apache-2.0, 32K context [VERIFIED: https://huggingface.co/Qwen/Qwen3-1.7B]); **Qwen3-4B** for Korean output. Both fit in fp16/int4 on a T4 or the 3060. Avoid Llama (naming clause) for release.
- **Three tiers.** Tier 0: 0-param templates emit answer, code, plan and proof verbatim. Tier 1: constrained verbalization of the CoreTrace fact sheet, with numbers, entities and code copy-masked. Tier 2: free prose, used only when Tier 1 fails, followed by a round-trip audit. The core re-parses the prose to IR and diffs it, and AlignScore (offline audit only) [VERIFIED: https://arxiv.org/abs/2305.16739] checks it; target ≤1% unsupported claims.
- Qwen3-0.6B already scores 77.6 on MATH-500 in thinking mode [VERIFIED: https://arxiv.org/html/2505.09388]. **The renderer's strength is why the blindfold and swap tests are mandatory.** It is never inside the RSI loop.

---

## 10. Scaling: how the RSI loop transfers from 0.1B to 1B to 10B

**What carries over unchanged:** verifiers, the domain staircase, the archive, the IR-level macro library, generators, the verified replay buffer, the eval harness and the safety stack. **Most of what RSI builds lives outside the weights** [HYPOTHESIS].

**Warm-start by depth distillation across sizes.** The 1B core's Gen 0 is distilled from **A(M_100M)**, the small core's maximally amplified verified outputs, so it starts at the small core's *deep* level. A larger core is also a stronger improvement operator, so Δ_amp should grow with size, as in the AZR trend: +5.7 at 3B, +10.2 at 7B, +13.2 at 14B [VERIFIED: https://arxiv.org/html/2505.03335v3].

**The measurement that justifies the extrapolation.** Run the identical loop at 25M, 50M and 100M for 6 generations each, and fit Δ per generation and η against params. If Δ grows with size, the 1B and 10B story has evidence; if not, fix the loop, not the size. "Predictably smarter" means a power-law fit with R² ≥ 0.9 on ≥4 points for each axis: params, r, N, library size, verified-data volume and number of domains.

**Compute reality** [ESTIMATE, from compute.md]: 1B at Chinchilla scale is about 1.2e20 FLOP (≈3 TPU-weeks), so 1B generations become multi-week. 10B needs TRC or grants (≈7 years of free quota otherwise). Humans apply to TRC with published results; the AI never acquires resources.

**North-star path: drug discovery**, reachable only through stages S3–S4 (§11). Simulators (docking, molecular dynamics, property calculators) act as **approximate** verifiers, each calibrated against held-out public assay data and demoted to a prior wherever calibration fails. The core proposes candidates in typed IR, simulators score them, and information-gain selection picks the few whose measurement would most reduce uncertainty. **Humans** run any real experiment. Objectives are therapeutic only. The domain is off by default: no pathogen, toxin or enhancement tasks in any generator, hazard blocklists on generation, and external dual-use review before any bio/chem tool or shard is mounted.

---

## 11. Stage-gated RSI safety

Safety strength scales with capability. Every stage inherits everything above it.

| Stage | Capability | Brakes |
|---|---|---|
| **S0** (day 1) | Toy domains, L1 | Immutable hash-pinned verifiers and evals in a separate process; serialized-IR channel; per-generation audit manifest (data, config and verifier hashes, sampled traces, metrics); **automatic kill switch** (H-gap > 5 with flat held-out, any canary, any verifier-path access, held-out jump > 3σ pending review); fixed GPU-hour, disk and network caps; no learner network |
| **S1** | Benchmarks, L2–L3 | Audit sampling of proposer tasks (AZR's "uh-oh moment" [VERIFIED: https://arxiv.org/html/2505.03335v3]); diversity monitors; the g−1 rollback is always kept |
| **S2** | L4 | **Human approval of every L4 diff**, with its proxy-run evidence; reward, verifiers, evals, caps and this table are untouchable |
| **S3** | Simulators | External dual-use review before any bio/chem tool or shard; hazard blocklist |
| **S4** | Wet-lab-actionable output | Expert review of every actionable item; no autonomous ordering or execution |
| **S5** | L5 | Not authorized; needs separate governance |

Hacking has been shown to generalize into sabotage in production RL [VERIFIED: https://arxiv.org/abs/2511.18397]. So at every stage a detected hack counts as both a failure and an incident.

---

## 12. First three experiments (weeks 1–4) and top risks

**E1 (weeks 1–2): a measurable self-improvement generation on a toy verifier domain.**
- **Setup.** Countdown/Game-of-24 expression search plus multi-step integer word templates, checked by Python eval. A 6M GDN core (d=256) on the laptop or one T4. Gen 0 is trained on 0-param generator data only, at k ≤ 3 operations. Then 4 generations of amplify (r=16, N=8) → verify → distill to r₀=2, with no new external data.
- **Arms:** (A) ANVIL; (B) equal-FLOP rejection-sampling SFT at fixed r=2; (C) random reward; (D) no latent loss.
- **GO** if all hold: A's fresh-seed r₀ solve rate rises ≥5 pts in each of 3 consecutive generations; the frontier index rises ≥1 operation; A beats B by ≥3 pts at Gen 4 and beats C clearly; the H-gap stays < 3.
- **KILL** depth distillation (fall back to plain expert iteration) if A ≤ B at Gen 4. **KILL the RSI claim** if Δ_fresh ≤ 0 by Gen 3.

**E2 (weeks 2–3): the reading wall.**
- **Setup.** A 20–40M bidirectional reader plus constrained emitter trained on synthetic NL→IR pairs. Measure GSM8K execution accuracy at N=1 and N=16, and GSM-NoOp invariance.
- **KILL or pivot** to a bigger or deeper reader if accuracy is < 25% at N=1 after 30 GPU-h.

**E3 (weeks 3–4): is depth compressible on grids?**
- **Setup.** On re-ARC data, plot ARC-AGI-1 eval accuracy against r ∈ {1, 4, 16, 32}, run one distillation generation, then add TTT.
- **GO** if the r-slope is positive, η ≥ 0.3, and pass@2 ≥ 15% with TTT.
- **KILL** the latent-grid route (move grids to program-synthesis IR) if the r-slope ≤ 0 or η < 0.1.
- **Also:** red-team every verifier with 1,000 adversarial outputs; zero false accepts are required.

**Top risks** (honest):
1. **Comprehension wall.** No <100M core has been shown to read arbitrary prose into IR; T5-base reaches only 32.7% NL→FOL equivalence [VERIFIED: https://arxiv.org/html/2509.22338v2]. E2 is the tripwire.
2. **Depth may not distill.** If η is small, the cost ratchet fails (E1, E3).
3. **Low base success outside toy domains**, even with the gap maximized by exact verifiers. Mitigations: curriculum from k=1 and backward construction.
4. **Attention-free content lookup** (−15 ARC points in TRM-MLP).
5. **CPU verification throughput** caps fuel per week.
6. **TPU availability** ("oversubscribed" reports [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab]); the T4-only fallback is about 10x slower.
7. **Acceleration may never appear.** No LLM self-improvement paper we found shows a positive second derivative. ANVIL's claim of acceleration across domain transitions is [HYPOTHESIS] and will be reported as measured, failure included.

**The bet in one line:** if E1 shows depth-distilled generations beating equal-FLOP expert iteration with zero new data, the project owns a self-improvement engine whose fuel is verifiers, not teachers. Every later boss becomes a question of adding verifier domains and compute, which are things humans can win with published results.
