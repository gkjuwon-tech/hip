# FOUNDRY: a library-growing RSI engine for an aphasic reasoner (proposal compiler_v1)

Philosophy: **neuro-symbolic compiler + DreamCoder-style wake-sleep.** The core never talks. It reads a problem, compiles it into a typed program or formal statement, and runs that on zero-parameter executors (Python, SymPy, Z3, Lean). Every generation, the RSI loop turns verified solutions into a larger library of reusable abstractions. Labels: [VERIFIED: url] means read in this session or in `research/*.md`. [ESTIMATE] means my arithmetic. [HYPOTHESIS] means an untested design bet.

---

## 1. Thesis: the RSI bet

**Distillation has a ceiling.** A student trained to imitate a teacher's traces is optimized toward the teacher's output distribution, so its target is the teacher's error rate. RLVR on top of that mostly sharpens what is already there. At large k, base models beat their RL-trained versions on pass@k, and "current training setup does not elicit fundamentally new reasoning patterns" [VERIFIED: https://arxiv.org/abs/2504.13837]. The frontier "tin cans" are not AGI, so imitating them caps us below AGI.

**The verifier replaces the teacher.** Every documented escape from a teacher ceiling had an exact external verifier:
- AlphaGo Zero used the rules of Go and beat AlphaGo Lee 100-0 after 3 days on 4 TPUs [VERIFIED: https://deepmind.google/blog/alphago-zero-starting-scratch/].
- AlphaProof used the Lean kernel [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/].
- Absolute Zero used a Python executor and zero external data [VERIFIED: https://arxiv.org/html/2505.03335v3].
- TinyGSM's student plus verifier scored 81.5 on GSM8K, against 77.4 for its GPT-3.5 teacher [VERIFIED: https://arxiv.org/html/2312.09241].

A verifier only certifies answers, so the ceiling moves from "teacher quality" to **verifier coverage × search reach**.

**Why this particular loop should compound, not just climb** [HYPOTHESIS, the central bet]. Program search cost is roughly b^L: b is the branching factor and L the program length in primitives. A library abstraction that replaces a recurring k-primitive fragment cuts L by about k−1 for every problem that uses it. The search cost of those problems falls by a factor of about b^(k−1). Linear gains in compression therefore give exponential gains in reachable difficulty. Problems that were out of reach become solvable. Their solutions contain new recurring fragments, which become the next abstractions.
- DreamCoder ran this wake/abstraction/dream cycle and rediscovered "the basics of modern functional programming, vector algebra and classical physics" [VERIFIED: https://arxiv.org/abs/2006.08381v1].
- LILO, with Stitch compression, solved harder tasks than DreamCoder [VERIFIED: https://arxiv.org/html/2310.19791v2].
- LEGO-Prover scored 50.4 on miniF2F with its skill library vs 47.1 without [VERIFIED: https://arxiv.org/pdf/2310.00656].
- The library also lives **outside the weights**. It costs 0 parameters, transfers unchanged to a 1B or 10B core, and can be inspected line by line. That makes it the safest form of self-improvement I know of.

**Why dropping language generation makes each generation cheaper** [ESTIMATE]:
1. **Short outputs.** An IR program for a GSM8K problem is about 50–150 tokens vs 300–1,000 for a chain of thought: about 5x fewer rollout tokens.
2. **A ~4k formal output vocabulary** instead of 32k–256k frees budget; Gemma 3 270M spends 170M on embeddings [VERIFIED: https://developers.googleblog.com/en/introducing-gemma-3-270m/].
3. **Everything is checkable.** Prose needs a hackable learned judge; programs run or they don't, so 100% of generated data is exact-filtered.
4. **Retraining touches reading and compiling, never fluency:** a generation is a few TPU-hours of SFT (§8).

The deliverable is **an engine, not a checkpoint**; the core is chosen as the best learner inside the loop.

---

## 2. The RSI engine

### 2.1 One generation, step by step

```
            ┌──────────── IMMUTABLE ZONE (separate process/VM user, read-only, hash-pinned) ─────────────┐
            │ VERIFIERS: py-sandbox · SymPy · Z3 · VAL · test-runner · ARC-DSL · Lean4  │ EVAL SETS (sealed)│
            │ REWARD CODE · held-out evaluator · audit-log writer (append-only, hash-chained)              │
            └──────▲─────────────────────────────▲───────────────────────────────▲────────────────────────┘
                   │ JSON in / JSON verdict out   │                               │ scores only
 ┌─────────────────┴──────┐   ┌──────────────────┴─────┐   ┌─────────────────────┴──────┐
 │1 PROPOSE               │──►│2 SOLVE (wake)          │──►│3 VERIFY                     │
 │ symbolic generators    │   │ grammar-masked search  │   │ exact checks, hidden tests │
 │ + core proposer (IR)   │   │ over IR + library calls│   │ counterfactual inputs      │
 │ + archive mutations    │   │ k up to 4096, TTT(ARC) │   └──────────┬─────────────────┘
 │ learnability target    │   └────────────────────────┘              ▼
 └────────▲───────────────┘                                ┌─────────────────────────────┐
          │                                                │4 SELECT                     │
 ┌────────┴───────────────┐   ┌────────────────────────┐   │ semantic dedupe, MDL-shortest│
 │7 GATE + EXPAND         │◄──│6 TRAIN (dream)         │◄──│ per task, diversity caps     │
 │ held-out promotion     │   │ SFT replays(refactored)│   └──────────┬──────────────────┘
 │ rollback, domain scout │   │ + fantasies + seeds    │              ▼
 │ human gate for L4      │   │ value head, GRPO sharpen│  ┌─────────────────────────────┐
 └────────────────────────┘   └────────────────────────┘◄──│5 COMPRESS (abstraction sleep)│
                                                           │ Stitch-style; refactor corpus│
                                                           └─────────────────────────────┘
```

1. **PROPOSE.** The curriculum for generation g is a mixture from four sources.
   - (a) **0-parameter symbolic generators**: Reasoning Gym's 100+ task families [VERIFIED: https://arxiv.org/pdf/2505.24760], re-ARC's 400 generators, training split only [VERIFIED: https://arxiv.org/pdf/2404.07353], SymPy algebra, fresh PrOntoQA-style ontologies, CLRS-style traces, Game-of-24, PlanBench.
   - (b) **Core-proposed tasks.** The emitter, in `<PROPOSE>` mode, writes an IR program plus an input spec. The executor computes the answer, so a proposed task is correct by construction.
   - (c) **Archive mutations** (POET-style stepping stones [VERIFIED: https://arxiv.org/pdf/1901.01753]): perturbed versions of tasks solved last generation.
   - (d) **Human train splits** (GSM8K-train, MATH-train, MBPP-train, mathlib statements), never test splits.
   - Proposals are reweighted by **learnability** L = p̂(1−p̂). Here p̂ is the current solve rate over 8 samples. Tasks with p̂ ∈ {0, 1} get zero weight, which is the PAIRED/AZR idea of rewarding tasks at the edge of solvability [VERIFIED: https://arxiv.org/pdf/2012.02096, https://arxiv.org/abs/2505.03335].
   - **Surface text for proposed tasks** comes from template grammars with paraphrase rules (0 parameters). An offline open renderer can paraphrase at *training time only*, logged and disclosed. It never sees a benchmark item, and a no-paraphraser ablation arm is always run.
2. **SOLVE (wake).** The core samples whole IR programs under a grammar mask. The mask is worth about one model size at 0 parameters: T5-Base+PICARD scores 66.6 vs 65.3 for unconstrained T5-Large [VERIFIED: https://github.com/ElementAI/picard].
   - Library entries are callable primitives, chosen by a 0-softmax top-k lookup (§5).
   - Search runs best-of-k with execution-guided repair, with beam/MCTS over IR statements for hard tasks. Every node is executed; rStar-Math keeps only steps whose code runs [VERIFIED: https://arxiv.org/html/2501.04519].
   - The budget is Snell-style: little search on easy tasks, most search on medium ones [VERIFIED: https://arxiv.org/abs/2408.03314].
3. **VERIFY.** Programs are serialized to text and checked by the immutable verifier process.
   - Checks: execution, SymPy equivalence, Z3 entailment, VAL plan validity, hidden unit tests, ARC demo pairs, Lean kernel.
   - **Counterfactual check:** for proposed and generated tasks, the program is re-run on 3 fresh inputs drawn from the generator. This catches programs that hard-code one answer.
4. **SELECT.**
   - Keep only verified solutions.
   - Deduplicate by **semantic fingerprint**: the hash of the outputs on 16 shared random probe inputs.
   - Keep the **MDL-shortest** verified program per task, plus up to 2 structurally distinct alternatives for diversity.
   - No single skill cluster may exceed 5% of the generation's data [ESTIMATE cap].
5. **COMPRESS (abstraction sleep).**
   - Run Stitch-style top-down compression over the corpus of selected programs, including all prior generations [VERIFIED method lineage: https://arxiv.org/html/2310.19791v2].
   - A candidate abstraction is admitted only if it lowers corpus description length by ≥ 0.5% [ESTIMATE threshold], is used in ≥ 20 tasks from ≥ 2 generators, passes property tests generated from its usage sites, and contains no literal over 16 tokens (the anti-lookup-table rule, §6).
   - The corpus is then **refactored** to use the new primitives. That shortens the training targets, so the policy learns to think in the new vocabulary.
   - Each entry stores its signature, type, tests, provenance (generation id and source tasks) and a usage counter. Entries unused for 5 generations are archived, not deleted.
6. **TRAIN (dream).** Expert iteration in three parts.
   - SFT on refactored replays **plus fantasies**: programs sampled from the library grammar, executed on random inputs and given template surfaces (DreamCoder's dream-sleep). The fantasies teach the reader-compiler what the new primitives mean.
   - **Seed data is never discarded;** accumulating rather than replacing data avoids model collapse [VERIFIED: https://arxiv.org/abs/2404.01413v2].
   - The value/judge head trains on search outcomes: success, Q-values and cheapest-budget class.
   - An optional GRPO sharpening pass is run, always with a **random-reward control arm** [VERIFIED method: https://proceedings.mlr.press/v306/shao26d.html].
7. **GATE + EXPAND.**
   - The sealed evaluator scores candidate generation g on the frozen held-out suite (§4). Promotion requires a held-out gain that the in-loop gain does not disproportionately exceed. Otherwise, roll back to g−1 and keep the data, which is still verified.
   - Then the **domain scout** runs (§2.4). Any L4 change goes into a human approval queue.

### 2.2 RSI levels

| Level | What improves | In FOUNDRY | When |
|---|---|---|---|
| L1 | Weights, on self-generated verified data | Steps 2–6 (expert iteration plus GRPO sharpening) | Gen 1 onward |
| L2 | Curriculum | Learnability-weighted proposer, archive mutations, the core writing new tasks | Gen 1 onward |
| L3 | Skill/program library | Abstraction sleep plus refactoring; also formalization templates (NL→IR idioms) | Gen 1 onward. **The headline lever** |
| L4 | Procedure: data generators, search hyperparameters, compression thresholds, kernels | AlphaEvolve-style search with exact evaluators over a *whitelisted config surface*: curriculum mixture weights, k/r schedule, Stitch arity, sampler temperature, generator difficulty knobs, scan kernels. Each diff is human-approved before it enters the next generation | Gen ~10 onward, behind the human gate |
| L5 | Architecture | North star only. Proposals are written to a human-read design log, never auto-applied | Not in this plan |

AlphaEvolve shows that L4 works on real stacks: a 23% kernel speedup that cut Gemini training time by 1% [VERIFIED: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/]. Learned optimizers are skipped. VeLO took about 4,000 TPU-months and was not faster in independent tests [VERIFIED: https://arxiv.org/abs/2211.09760, https://arxiv.org/html/2310.18191v2].

### 2.3 The teacher bootloader and the cord cut

- **Stage-1 rocket (gens 0–2).**
  - Human train splits and human-authored text: FineWeb-Edu for reader pretraining, plus the GSM8K/MATH/MBPP train sets.
  - 0-parameter generators.
  - **One** optional teacher channel: an Apache-2.0 open model (Qwen3-4B [VERIFIED license: https://huggingface.co/Qwen/Qwen3-4B]) translates *train-split* problems into our IR. Each translation is kept only if it executes to the gold answer.
  - Every teacher token is tagged `src=teacher`. No GPT-3.5/4 data anywhere, so TinyGSM and MetaMathQA are out [VERIFIED: https://arxiv.org/pdf/2312.09241, https://arxiv.org/pdf/2309.12284].
- **A no-teacher twin arm** runs from gen 0 with identical compute and zero `src=teacher` tokens.
- **Cord-cut criterion.** Teacher intake stops at the first generation g at which *either* condition holds:
  - (a) for 2 consecutive generations, the no-teacher arm is within the 95% CI of the teacher arm on held-out, **and** TE > 0 (§4);
  - (b) g = 3, a hard deadline whatever the result.
- After the cut, existing teacher-tagged data is down-weighted by 0.5 per generation and leaves the training mix by g+4. The library never contains teacher-written code; that rule holds from day 1.
- The G0 counter (generations with no new human or teacher data) starts at the cut.
- **Fuel after the cut:**
  1. The amplification gap: search plus verifier vs the raw policy.
  2. Learnability-edge proposals.
  3. Library compression, which makes formerly unreachable tasks reachable.
  4. New verifier domains.
  5. Formalization skills: every NL→IR idiom the library learns enlarges the set of problems that can be checked at all.

### 2.4 The S-curve staircase, and finding the next step

Each verifier domain is expected to give one S-curve. The envelope is the bet.

| Step | Domain | Verifier (0 params) | Opens when | Expected saturation signal |
|---|---|---|---|---|
| D1 | Arithmetic word problems, Game of 24, Countdown | Python executor | Gen 0 | Solve frontier (operand count / depth) stops rising |
| D2 | Algebra, calculus, number theory (MATH-style) | SymPy equivalence | D1 frontier ≥ 80% at mid difficulty | MATH-level-5 twin flat |
| D3 | Logic, CSP, planning (FOLIO/ProofWriter/AR-LSAT/Blocksworld) | Z3, Prover9-style, VAL | Gen ~4 | Mystery-Blocksworld flat |
| D4 | Code with tests | Hardened sandbox + hidden tests | Gen ~6 | LiveCodeBench-after-freeze flat |
| D5 | ARC grids | DSL + demo pairs; per-task TTT | Gen ~6 | Generator-fresh ARC flat |
| D6 | Formal proof | Lean 4 kernel + mathlib | Gen ~10 | miniF2F-style flat |
| D7 | Simulators (physics, materials; bio/chem only after external review) | Simulator + held-out real measurements | North-star track | Sim-vs-real gap |
| D8 | Real experiments | Humans in the loop | North star | Latency-bound |

**Domain scout (an RSI skill, human-gated)** [HYPOTHESIS]. Each generation it scores candidates from a human-maintained registry of licensed solvers:

score = (library coverage of the domain's seed problems) × (verifier throughput per CPU-second) × (transfer: held-out gain on *existing* domains when 5% of the mix is from the candidate) ÷ (safety class).

Coverage = fraction of seed problems the library expresses in ≤ 1.5x a human reference's IR length. The core may *draft verifier adapters* (wrapping a solver's I/O in our JSON schema); a human red-teams and hash-pins each one. **The model never activates a verifier itself.** The deeper version of this skill is autoformalization, which turns informal problems into checkable statements and so grows verifier coverage directly; it is checked by round-trip re-reading and execution on known answers.

### 2.5 Cycle time and compute per generation [ESTIMATE]

The weekly free quota is TPU v5e-8 about 20 h, T4x2 about 30 h, plus Kaggle/laptop CPUs [VERIFIED: research/compute.md]. Target: **2 generations per week** at the 100M scale, 1 per week once D4–D6 are in the mix.

| Step | Budget per generation | Hardware |
|---|---|---|
| Rollouts | ~1.2e18 FLOP (≈ 3M problems × up to 64 samples, with early stop; per-problem cost below) | T4x2 15 h (fp16) and/or TPU 3 h |
| Verify | ~1–2e7 executions | CPU: Kaggle 4 vCPU × 2 sessions + laptop + TPU-VM host (core count to be checked with `nproc`) |
| Compress | ~2–6 CPU-hours (Stitch-class) | Laptop |
| Train | 1–2B tokens SFT ≈ 1.5e18 FLOP | TPU v5e-8, ~1.5 h at 25% MFU |
| Gate eval | ~5e16 FLOP + CPU | T4 |

- **Per-problem rollout cost:** reader 200 tok × 2 × 34M ≈ 1.4e10; core r=4 ≈ 2.4e10; emitter 64 × 120 tok × 2 × 22.7M ≈ 3.5e11; total ≈ 3.9e11 FLOP, about 15 ms on T4x2 at 20% MFU.
- **Verification binds, not FLOPs** (1e2–1e3 Python checks/s on 4 vCPU [VERIFIED estimate: research/reason_methods.md]). Mitigations: output-hash cache, a vectorized NumPy evaluator for straight-line IR, stop at first success on easy tasks, spend surplus GPU on medium tasks only.

---

## 3. Curve-killer defenses

| Killer | Mechanism | Detector | Kill / rollback rule |
|---|---|---|---|
| **Verifier exhaustion** (S-curve top) | Domain scout opens the next domain; library primitives carry over | Per-domain Δ < 0.5 pt for 2 generations, and the solve-frontier difficulty flat | Open the next domain from the registry. If none is approved, freeze the domain's mix share at 10% (retention) and move the budget elsewhere |
| **Own-level prison** (log curve; RLVR narrowing) | Search beyond the policy's support (k up to 4096, MCTS); learnability-edge proposer; archive stepping stones; refactoring makes long solutions short | **pass@256 on held-out** tracked next to pass@1 [VERIFIED method: https://arxiv.org/abs/2504.13837v4]. Prison = pass@1 rises while pass@256 is flat or falling | 2 generations of flat pass@256 → stop GRPO sharpening, raise the search budget and the archive-mutation share. 4 generations → declare a plateau on the dashboard and require a new domain |
| **Reward hacking** (fake exponential) | Verifier in a separate OS user and process, communicating only by JSON. The learner cannot import, patch or read the verifier, which defeats the `__eq__` override and stack-reading tricks METR saw in o3 [VERIFIED: https://metr.org/evaluations/openai-o3-report/]. Hidden tests; counterfactual re-execution; IR has no `import`, no I/O, no reflection | H = in-loop score − held-out score (§4). Canary tasks whose only "solutions" exploit a known-planted verifier quirk. Sandbox syscall audit | H rises > 5 pts with held-out flat for 1 generation → automatic rollback. Any canary solved, or any syscall toward verifier paths → **kill switch**: halt the loop, preserve state, human review |
| **Model collapse** | Accumulate, never replace. Seeds stay in every mix at ≥ 20%. MDL-shortest plus 2 diverse alternatives. Per-cluster caps | Distinct-fingerprint ratio; per-generator tail coverage (solve rate on the hardest 10% decile); entropy of IR-token unigrams | Any monitor falls > 15% relative vs gen g−1 → roll back the generation's data mix and raise the seed share |
| **Library bloat / overfit abstractions** (FOUNDRY-specific) | MDL admission threshold; ≥ 2 generators of origin; property tests; archive after 5 idle generations | Held-out description length per solved task; library-on vs library-off ablation each generation | Library-on loses to library-off on held-out → revert that generation's abstractions |
| **Compute wall** | Raise efficiency, not compute: shorter programs, cached execution, compute-optimal allocation, L4 search over kernels and schedules (human-gated). Humans apply for TRC with published results [VERIFIED: https://sites.research.google/trc/faq/] | Solved-held-out-items per GPU-hour, per generation | Efficiency flat for 3 generations → L4 review; never self-acquire resources |
| **Real-experiment latency** (D7–D8) | Simulator pre-screen; batch selection by expected information gain; held-out real assays from public data | Sim-vs-real rank correlation on held-out assays | Correlation below a pre-registered floor → simulator demoted to ranking only, never used as a reward |

---

## 4. RSI metrics dashboard

All metrics are computed by the sealed evaluator on frozen sets and published per generation.

| Metric | Definition | Healthy | Alarm |
|---|---|---|---|
| **TE** (teacher-exceedance) | Among held-out items that the bootloader teacher (Qwen3-4B, k=8, same IR harness) fails, the fraction FOUNDRY solves | Rising after the cut | Falls 2 generations in a row |
| **Δ_g** | Held-out composite score S_g − S_{g−1}, at fixed params and fixed per-query budget | > 0 | ≤ 0 for 3 generations |
| **a_g** (second derivative) | Δ_g − Δ_{g−1}, also per GPU-hour. Reported over a 3-generation moving window, since single generations are noisy | ≥ 0 within each S-curve's rise. Envelope a ≥ 0 across domains | Negative for 4 generations with no new domain queued |
| **H** (hacking gap) | In-loop proposer-task solve rate minus the frozen held-out solve rate | Stable ±3 | +5 pts with flat held-out |
| **G0** | Consecutive generations with zero new human or teacher tokens | Increasing | Reset requires a published justification |
| **pass@1 / pass@256** | On held-out twins | Both rising | Only pass@1 rising |
| **Random-reward delta** | GRPO gain minus the random-reward arm | > 2 pts | ≤ 0 means the gain is spurious |
| **Efficiency** | Mean IR length per solved held-out task; search nodes per solve; solves per GPU-hour | Falling / falling / rising | — |
| **Library** | Size, reuse rate (share of new solutions calling learned primitives), depth of abstraction stack | Reuse > 30% by gen 10 [ESTIMATE] | Reuse falling while size grows (bloat) |
| **Diversity** | Distinct-fingerprint ratio; tail-decile coverage | Stable | −15% |
| **Transfer** | Δ_g of the same loop at 25M / 50M / 100M | Δ grows with size, as AZR's gains did with scale [VERIFIED: https://arxiv.org/html/2505.03335v3] | Δ flat in size: the loop is the bottleneck |

**Definition of "RSI is on":** Δ_fresh > 0 on the commit-revealed private sets for ≥ 3 consecutive generations with G0 ≥ 3 and H stable [ESTIMATE threshold, from research/reason_baseline.md §5].

---

## 5. The reasoning core: the learner inside the loop

### 5.1 Component diagram

```
 NL problem (en/ko, as given; 0-param normalization only)
   │  unigram tokenizer 16k (ko josa/eomi pre-split rules, 0 params)
   ▼
 EMB 16384×256 → 640 (factorized)  + mode/segment emb
   ▼
 READER: 6 Gated-DeltaNet layers, alternating L→R / R→L  (bidirectional, no softmax)
   │  h_read (per-token)        ──────────────────────────────┐
   ▼                                                           │
 CORE: 3 GDN layers, weight-tied × r (masked while_loop),      │
       injector [h_{i-1}; h_read] each loop, TRM deep supervision
   ▼                                                           │
 EMITTER: 4 causal GDN layers over IR tokens (state-initialized│
   from core output; reads core states as prefix)              │
   │  grammar mask (LL(1) over typed IR, 0 params)             │
   │  <lib> → 640→256 query → top-k Hamming lookup over        │
   │          library-entry codes (codes computed by READER)   │
   │  <copy> → suffix automaton over input (0 params)          │
   ▼                                                           ▼
 IR program ─► VERIFIER PROCESS (JSON)          JUDGE: 2 GDN layers R→L over
              result tokens re-enter stream      [h_read; candidate states] + fusion MLP
                                                 → verify-energy, process value,
                                                   difficulty, learnability
```

### 5.2 Dataflow for one query

Read once → core iterates r times (r from the difficulty head, capped) → emitter samples k grammar-masked programs; `<lib>` does a hard top-k Hamming lookup over entry codes, so the library grows without changing the output vocabulary → verifier executes, results splice back, ≤ 3 repair rounds → select by exact checks, then majority over executed answers, then judge energy → emit `CoreTrace`.

### 5.3 Structured output format (the only thing the renderer ever sees)

```json
{"trace_id":"g17-0x9f3a…","core_hash":"sha256:…","library_version":"L17",
 "answer":{"type":"int","value":42},
 "program":"a = 6*7\nANSWER(a)",
 "steps":[{"id":"s1","op":"mul","in":[6,7],"out":42,"lib":null,
           "copied_spans":[{"src":"input","start":31,"end":44}]}],
 "verification":[{"kind":"exec","status":"pass"},{"kind":"counterfactual","status":"pass","n":3}],
 "confidence":0.93,"budget":{"r":4,"k":16,"flops":3.1e11,"cpu_s":0.4},
 "facts":["quantity_A = 6","quantity_B = 7"],
 "lang":"ko"}
```

There is no `question` field. Copied input spans appear only through `copied_spans` and `facts`, and they are logged. Lean proofs and code are carried verbatim in `program`.

### 5.4 Parameter budget (every executed weight; CI asserts the total)

d = 640, 5 heads × 128, SwiGLU 1536, expand_v = 1. One GDN layer has 5,012,618 parameters (formula reused from ALEMBIC-0, archive `blueprint_final.md` §0.1).

| # | Module | Composition | Params |
|---|---|---|---|
| 1 | Input tokenizer scores (counted) | 16,384 | 16,384 |
| 2 | Factorized input embedding | 16384×256 + 256×640 | 4,358,144 |
| 3 | Reader | 6 GDN (bidirectional, alternating) | 30,075,708 |
| 4 | Recurrent core | 3 GDN, weight-tied, counted once | 15,037,854 |
| 5 | Loop injector | Linear(1280→640) + bias | 819,840 |
| 6 | Emitter | 4 causal GDN | 20,050,472 |
| 7 | IR vocabulary (tied in/out) | 4096×640 | 2,621,440 |
| 8 | Library query projection | 640×256 | 163,840 |
| 9 | Judge tower | 2 GDN, R→L | 10,025,236 |
| 10 | Judge fusion | 1280→1024→4 | 1,315,844 |
| 11 | Learnability/difficulty head | 640→64→3 | 41,219 |
| 12 | Mode/segment embeddings | 16×640 | 10,240 |
| 13 | Final norms | 3×640 | 1,920 |
| | **Total** | | **84,538,141** |
| | Headroom under 100M | | 15,461,859 |

- The embedding share is 5.2%, against 63% for Gemma 3 270M.
- Library entries, their 256-bit codes, the datastore, BM25 statistics and the suffix automaton are **data, not parameters**. They are disclosed with sizes and hashes.
- Headroom pool: +2 judge layers (10.0M), released only if the X-judge ablation shows a positive judge-depth slope. TinyGSM found that verifier size matters more than generator size [VERIFIED: https://arxiv.org/html/2312.09241].
- The CI test computes the count from the instantiated module: `sum(leaf.size) + tokenizer_scores < 100e6`. The test derives its expected value from the config and never hard-codes it (the CI rule is ALEMBIC-0's).

### 5.5 Why non-Transformer, and why this shape

- **No softmax attention over positions anywhere.** Mixers are gated delta-rule recurrences with constant state [VERIFIED mechanism: https://arxiv.org/abs/2412.06464v3]. The library lookup and retrieval are hard, non-differentiable top-k over external items.
- **Bidirectional reader:** ELECTRA-Small (14M) beats GPT-1 (117M) on GLUE [VERIFIED: https://arxiv.org/abs/2003.10555]; T5-base 220M ≈ LLaMA-8B on NL→FOL [VERIFIED: https://arxiv.org/html/2509.22338v2].
- **Weight-tied core:** TRM at 7M scores 44.6 on ARC-AGI-1 [VERIFIED: https://arxiv.org/html/2510.04871]; Huginn scores 0 at r=1 vs 42 at r=32 [VERIFIED: https://arxiv.org/html/2502.05171].
- **Why it is a good learner in the loop:** constant-size state forks in O(1), so 64–4,096-way sampling is cheap; IR outputs are short; retraining takes minutes to hours; the reader doubles as the library-code encoder, so the library index refreshes every generation at near-zero cost.
- **The honest caveat:** a GDN block keeps the projection-plus-MLP layout of a Transformer block. The rule we enforce is "no softmax/Hopfield attention over the token sequence".

---

## 6. Fair-play rules

**Renderer firewall:**
1. **Core-only scoring.** The harness scores `CoreTrace.answer` / `program`; CI fails if the renderer is loaded in the scoring process.
2. **Blindfold by type.** The renderer gets a `CoreTrace` with no question field; a static check requires 0 shared 8-grams with the question outside logged `copied_spans`.
3. **Renderer-alone controls per row:** raw question (published cheat ceiling), empty/shuffled trace and answer-masked trace; the last two must stay ≤ chance + 2σ.
4. **Counterfactual swap.** A corrupted answer must survive rendering ≥ 99% of the time [HYPOTHESIS threshold, from research/io_split.md].
5. **Input read by the core.** Only disclosed 0-parameter normalization in front of it; no dataset-ID feature; a shuffled-format probe checks invariance.

**General:** no neural net >100M at inference, no LLM APIs, no GPT-3.5/4 data; 13-gram + MinHash + embedding-similarity decontamination of training data, datastore **and library** [VERIFIED protocol basis: research/data.md]; commit-reveal private sets; search budget disclosed per row (≤ 60 T4-seconds average); an ORRERY-0 equal-FLOP control arm (plain recurrent policy + sampling + same verifiers, no library, no recurrence) on every claim.

**RSI integrity (FOUNDRY-specific):**
- Verifiers, reward code, eval sets and the evaluator are hash-pinned and owned by a different OS user. The learner process has no write permission and no network access. Hashes are checked at every generation boundary and logged.
- **Library hygiene.** No entry may hold a literal over 16 tokens, a number table, or an 8-gram match to an eval item. The evaluator re-scans entries against sealed eval hashes and reports only pass/fail. This blocks the obvious route by which a growing library memorizes answers.
- The proposer cannot write to the eval generators' seeds. Held-out twins use generator seeds sealed before gen 0.

---

## 7. The boss ladder

Gates G = T + max(10, (100−T)/3), with T the max reported GPT-3.5 score [VERIFIED targets: research/reason_baseline.md]. Generation numbers assume about 2 generations per week. Every row is an [ESTIMATE].

| Row | T | **G** | Expected gen to clear G | P(clear G at <100M) | Why |
|---|---|---|---|---|---|
| Game of 24 | 19 | 46 | 2–4 | 0.95 | Search plus executor; GPT-3.5's ToT reaches only 19 [VERIFIED: https://arxiv.org/pdf/2305.10601] |
| Blocksworld (NL) | 6.1 | 37.4 | 6–10 | 0.75 | VAL plus planner once formalized; reading is the risk |
| CRUXEval-O | 63.3 | 75.6 | 8–14 | 0.45 | Predicting execution *without* running it (executor disabled for this row) |
| ARC-AGI-1 (pass@2) | 10.6 | 40.4 | 10–20 | 0.55 | TRM 44.6 at 7M; DSL library plus TTT |
| FOLIO / ProofWriter / PrOntoQA | 62.7 / 71.5 / 85.0 | 75.2 / 81.5 / 95.0 | 8–16 | 0.5 / 0.6 / 0.35 | Z3 plus self-autoformalization; LINC shows a +38 margin with a solver [VERIFIED: https://arxiv.org/abs/2310.15164] |
| GSM8K / GSM-Plus | 78.1 / 61.2 | 88.1 / 74.1 | 16–30 | 0.30 | The reader wall; number-swap invariance by construction helps on GSM-Plus |
| MATH | 43.1 | 62.1 | 20–40 | 0.20 | SymPy helps; long multi-concept problems resist |
| BBH | 70.7 | 80.7 | 12–24 | 0.35 | 23/27 tasks closed-world |
| DROP F1 | 70.2 | 80.2 | 16–30 | 0.30 | Copy plus executor; dense reading |
| MuSR / LogiQA / ReClor | 58–79 | 68–89 | — | 0.15 | Long narratives; weakest fit for a compiler |
| HumanEval+ / MBPP+ / LCB | 70.7 / 69.7 / 22.6 | 80.7 / 79.8 / 48.4 | 25–50 | 0.15 / 0.15 / 0.25 | Code needs an API prior; LCB's low T makes it the best code row |
| GPQA-Diamond / MedQA | 30.6 / 63.4 | 53.7 / 75.6 | — | 0.05 / 0.10 | Knowledge-bound; no exact verifier |

**Joint probability** that every Tier A+B row clears G on one checkpoint at <100M: about 0.03 [ESTIMATE]. Probability that ≥ 50% of rows clear G: about 0.35. **Win-or-extend** applies: the cap and the gates never move, losses are published with their curves, and a 1B decision goes to the user (rule from ALEMBIC-0 F13).

**Next rungs** (requirements, not dates):
- **GPT-4-level:** a 1B core (~9 TPU-h per SFT generation, §10); D1–D5 libraries saturated with reuse > 40%; robust reading (GSM-NoOp drop ≤ 5 pts, vs GPT-4o's 95.2→63.1 [VERIFIED: https://arxiv.org/html/2410.05229]). P ≈ 0.15.
- **o1/R1-level:** long-horizon Lean search (D6) with test-time RL on problem variants, the AlphaProof recipe [VERIFIED: https://research.google/pubs/olympiad-level-formal-mathematical-reasoning-with-reinforcement-learning/]; competitive on AIME-after-freeze (R1-Zero: 15.6→77.9 on AIME24 [VERIFIED: https://arxiv.org/html/2501.12948]); TRC compute and a 1–10B core. P ≈ 0.05.
- **Frontier tin cans:** autoformalization covering combinatorics (AlphaProof failed both IMO 2024 problems) and science MCQ. Verifier coverage, not parameters, decides this rung. P < 0.02.
- **Top human experts, then the north star:** formal statements plus new scientific verifiers. P not estimable.

---

## 8. Bootloader recipe (stage 0), data, compute

**Data** (licensed; no GPT-3.5/4 data):
- **Reader pretraining:** ELECTRA-style replaced-token detection on 3B tokens of FineWeb-Edu (ODC-BY) and FineWeb-2-ko (ODC-By); kowiki stays in the datastore only (BY-SA) [VERIFIED: research/data.md]. ELECTRA-Small reached GLUE 74.1 after 6 GPU-hours [VERIFIED: https://arxiv.org/abs/2003.10555].
- **Compiler SFT:** Reasoning Gym, re-ARC (train split), SymPy/Z3 generators, CLRS-style traces, human train splits, mathlib statements (D6), and OpenMathInstruct-2 *problems* (CC-BY-4.0) [VERIFIED: https://huggingface.co/datasets/nvidia/OpenMathInstruct-2], re-derived into IR, kept only if they execute to the gold answer, tagged `src=teacher`.
- **Optional teacher channel (gens 0–2):** Qwen3-4B NL→IR on train splits, execution-filtered.

**FLOP math** (6ND; v5e-8 at 1,576 TFLOP/s peak, 25% MFU ≈ 3.9e14 FLOP/s; T4x2 at 20% ≈ 2.6e13) [ESTIMATE]:
- Reader RTD: 6 × 3.5e7 × 3e9 = 6.3e17. About 27 TPU-minutes, or about 7 T4x2-hours.
- Compiler SFT: the effective N with r̄ = 3 is 85M + 2 × 15M ≈ 1.15e8. 6 × 1.15e8 × 4e9 tokens = 2.8e18, about 2 TPU-hours.
- Controller/judge labels: about 1e18.
- **Stage 0 totals about 5e18 FLOP: one TPU week is about 28e18.** Compute is not the constraint. Data generation and verification CPU are.
- The year's envelope is about 2e21 FLOP [VERIFIED estimate: research/compute.md]. That is about 1,300 generations at 1.5e18 each, so the loop is CPU-bound and wall-clock-bound, not FLOP-bound.

**T4 engineering:**
- fp16 matmuls with fp32 accumulation; recurrent state and decay kept in fp32 (no bf16 on T4).
- Pin a Triton-3.2-era stack, or use a pure-PyTorch chunked GDN fallback (Triton ≥ 3.3 dropped sm_75) [VERIFIED: research/archs.md].
- On TPU: JAX chunked scan, static shape buckets (IR length 64/128/256), r as a masked `while_loop`.
- Checkpoint to the HF Hub every 45 minutes, with the generation state and the library as a versioned dataset.

**Inference cost per query** [ESTIMATE]:
- Default budget (r=4, k=16): about 1.5e11 FLOP, about 10 ms of GPU plus about 0.3 s of CPU.
- Search rows (k=1,024): about 2.5e13 FLOP, about 1 s of GPU plus CPU-bound verification of about 5–20 s.
- ARC with TTT: about 1e15 per task, disclosed separately.
- All modes are under the 60 T4-second average.

---

## 9. Renderer

- **Default:** Qwen3-1.7B, non-thinking mode, Apache-2.0, about 3.4 GB fp16 on the 3060.
- **Korean:** Qwen3-4B (int4 on the 3060, fp16 on the T4) or Gemma 3 4B. Gemma's terms pass down to users.
- **Avoid Llama** for release because of its naming clause [VERIFIED: research/io_split.md].
- **Tiers:** (0) templates emit answers, code and proofs verbatim, covering about 100% of benchmark outputs; (1) the renderer verbalizes `facts` + `steps` with numbers and entities copy-masked from the trace; (2) free prose, round-trip audited by the core's own reader plus offline AlignScore [VERIFIED: https://arxiv.org/abs/2305.16739].
- Renderer FLOPs are reported, never counted, never scored.

---

## 10. Scaling: 0.1B → 1B → 10B, and the north star

**What transfers unchanged** is the RSI state, not the weights: the library (text plus tests), the task archive with learnability statistics, the verifier registry, all accumulated verified traces, and the dashboard history. A new core inherits generation g's world on day 1.

**Warm start** [HYPOTHESIS]: function-preserving width growth (Net2Net-style) and duplicated tied core layers, then one generation trained on the 100M core's *verified* traces (no teacher ceiling, since only verified outputs are used), then the bigger core runs the loop natively.

**Compute:** a 1B SFT generation (2B tokens) is 1.2e19 FLOP, about 9 TPU-hours, so one generation per week fits. 10B Chinchilla pretraining is 1.2e22, about 7 years of free quota [VERIFIED estimate: research/compute.md]: **10B requires TRC**, applied for by humans with published results.

**Transfer test:** the identical loop runs at 25M/50M/100M from week 3. If Δ_g grows with size, as AZR's +5.7 (3B) → +13.2 (14B) did [VERIFIED: https://arxiv.org/html/2505.03335v3], the 1B–10B extrapolation has evidence; if flat, the loop is the bottleneck, not parameters. "Predictably smarter" = power-law fit, R² ≥ 0.9 on ≥ 4 points of held-out score vs generation-compute, per domain.

**North-star path: drug discovery** (stated at policy level only):
- **Verifiers become approximate.** Physics-based simulators and models trained on public assay data score candidates. They are **rankers, never sole rewards**: docking-style scores are easy to exploit, so this is the highest reward-hacking risk in the project [HYPOTHESIS].
- **Ground truth** is held-out real measurements from public datasets, sealed like eval sets.
- **Experiment selection** maximizes expected information gain about the simulator's error, not predicted potency. The loop improves the *verifier* as much as the policy.
- **Biosafety comes first:** no bio/chem tools, shards or generators without external dual-use review; hazardous-target blocklists in proposer, datastore and library; no pathogen, toxin or enhancement tasks, ever; human expert review of every wet-lab-actionable output; no release without review.

---

## 11. Stage-gated RSI safety

| Stage | Capability | Brakes (cumulative) |
|---|---|---|
| S0 (gens 0–10) | L1–L3 on D1–D5 | Immutable hash-pinned verifiers and evals in a separate OS user. Append-only hash-chained audit log per generation (data manifest, config, verifier hashes, 1% sampled traces, library diff). Learner has no network. Fixed disk and GPU-hour caps. Kill switch on H, canary or syscall events |
| S1 (L4 on) | Procedure search over the whitelisted config surface | **Human approval per L4 diff** before the next generation. Diffs are human-readable, and each L4 change is a separate, reversible generation |
| S2 (D6 Lean, 1B core) | Formal math at scale | Proposer-output audit sampling (5%) for goal drift, after AZR's "uh-oh moment" [VERIFIED: https://arxiv.org/html/2505.03335v3]. Hacking-generalization probes [VERIFIED concern: https://arxiv.org/abs/2511.18397] |
| S3 (D7 simulators) | Science | External dual-use review before any bio/chem domain opens. Blocklists. Simulator-as-ranker rule |
| S4 (D8 experiments) | Real world | Human expert review of every actionable output. No autonomous lab access. Release review |

At every stage the model cannot modify reward functions, verifiers, eval sets or caps; there is no self-acquisition of compute, storage, network or accounts (one Kaggle account, ToS-compliant); and the kill switch halts the loop and preserves state for review.

**Triggers:** H +5 with flat held-out; a held-out jump > 3σ (held for review, not auto-celebrated); any write or import toward verifier paths; any canary hit; any blocklist hit.

---

## 12. First experiments (weeks 1–4), with go/kill criteria

**E1 (weeks 1–2): one measurable self-improvement generation on a toy verifier domain.**
- Domain: Countdown/Game of 24, plus a list-function DSL (map/filter/fold tasks), both with exact executors.
- Core: an S-config of about 6M parameters (d=256) trained on the laptop and T4.
- Arms at equal compute: (A) full FOUNDRY generation; (B) expert iteration without compression; (C) the gen-0 model trained longer on gen-0 data (the "just more training" control); (D) random-reward GRPO.
- Held-out: generator-fresh seeds sealed on day 1, at difficulty above the training range.
- **GO** if A beats C by ≥ 5 pts on held-out (CI excludes 0), A beats B by ≥ 2 pts, mean held-out program length falls ≥ 10%, pass@256 rises, and H is stable.
- **KILL/PIVOT** if A ≤ C within CI after 3 generations. That would mean the loop adds nothing over training, and the library thesis is wrong at this scale.

**E2 (weeks 2–3): reader kill probe.**
- A 20–40M BiGDN reader plus emitter on 0-parameter synthetic NL→IR pairs, plus GSM8K-train.
- **GO:** GSM8K-test execution accuracy ≥ 25% at N=1 after 30 GPU-hours, and GSM-Plus relative drop ≤ 16.9% [VERIFIED thresholds: research/io_split.md, research/reason_baseline.md].
- **KILL:** < 15%. Then the reading wall is real, so the bet moves to puzzle and formal rows first, and the reader moves to the top of the headroom priority.

**E3 (weeks 3–4): compounding test on D1+D2 with a 50M core, 4 generations.**
- Measure Δ_g and a_g, with library on/off, the ORRERY-0 equal-FLOP control, and Δ_g at 25M vs 50M. Also run the renderer swap test on 500 items.
- **GO** if the library arm beats ORRERY-0 by ≥ 3 pts on held-out at gen 4, Δ_g decays no faster than the control, reuse > 15%, and Δ at 50M ≥ Δ at 25M.
- **KILL:** the library arm ≤ control. Then FOUNDRY falls back to plain search plus expert iteration, and that is published.

**Top risks:**
1. **Reading wall.** A <100M core parsing arbitrary prose is unproven; Phi-3-mini drops 83.7→18.0 on GSM-NoOp [VERIFIED: https://arxiv.org/html/2410.05229].
2. **Small models can't self-verify** [VERIFIED: https://arxiv.org/abs/2412.02674]. Mitigation: exact verifiers only; the judge ranks, never rewards.
3. **Abstractions overfit to generators** and miss human-written problems. Detector: library on/off on human-written held-out.
4. **CPU verification throughput** and template-surface shift vs benchmark prose.
5. **TPU availability** and solo engineering load (grammar mask, sandbox, Stitch ≈ 3 weeks; E1 starts on a minimal DSL).
6. **Acceleration is unproven.** I found no LLM self-improvement paper with a_g > 0 [VERIFIED observation: research/rsi.md §8]. The staircase envelope is a labeled bet.
