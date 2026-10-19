# External Audit of Phase A Proposals (binding input for Phase B onward)

An independent auditor read all five `work/proposal_*_v1.md` files in full, recomputed the numbers, ran the Experiment 1 code, and checked citations against the research files and the web. This file is **binding input** for the Phase B red team, the Phase C revisions, and every later phase. Where it conflicts with an earlier instruction, this file wins, except that BRIEF.md hard constraints (§2) always apply.

---

## 0. Three process changes, effective immediately

1. **Word counts are soft guidance, not a target.** Do not spend time counting or trimming words. Do not write scripts to generate or assemble documents (`generate_proposal.py`, `assemble.py`). Write the documents directly. Any designer still trimming must submit the current version now.
2. **No code in any proposal, blueprint, or section.** This replaces the BRIEF.md instruction to include pseudo-code for Experiment 1. Experiments must be specified as a **plan only**: inputs, procedure, what is measured and how, the baseline/control arm, the pass criterion, the kill criterion, and the expected compute and wall-clock time. Implementation will be done by a separate engineer after the design is final. Reason: every Phase A Experiment 1 contained code that hard-coded its own success (see §2).
3. **The Phase B red team must attack the issues in §1 and §2 below in addition to its own findings**, and the Phase C revisions must fix them.

---

## 1. Systematic problems found in all five proposals

### 1.1 Recurrent depth is missing from all FLOP and time estimates (5/5)
Every proposal uses a weight-tied recurrent core iterated r = 8–16 times, but computes FLOPs as if the core runs once. The correct per-token cost is
`FLOPs/token ≈ 2 × (N_non_core + r × N_core)` for inference and `6 × (N_non_core + r × N_core)` for training.
This understates rollout and training cost by roughly **2.6×–9×**. Recompute every compute estimate, every generation cycle time, and every "generations per week" figure with the real r.

### 1.2 Inference latency is physically impossible (5/5)
Claimed per-query latencies (0.0007 s, 0.28 ms, 0.9 ms, 6 ms) treat sequential token-by-token decoding as compute-bound at full GPU throughput. Sequential decoding is **memory-bandwidth and kernel-launch bound**. Example: reading ~160 MB of weights per token at ~300 GB/s on a T4 is ~0.5 ms per token per recursion, so 150 tokens × r recursions is tens of milliseconds to seconds, and MCTS with 64 simulations is seconds. Recompute latency honestly.

### 1.3 "Finding the next verifier domain" is a fixed list (5/5)
BRIEF.md §1 Decision 3 requires that finding or building the next verifier domain is itself an RSI skill. All five proposals instead walk a human-written list (arithmetic → algebra → Lean → code → simulators). Design an actual mechanism, or state honestly which part is human-curated and why. Note: the model must **never write or modify its own verifiers** (BRIEF.md §2.7), so any domain-building mechanism must keep verifier code human-reviewed or drawn from existing trusted tools.

### 1.4 The deepest hole: there is no verifier for READING (5/5)
Executors check that a program **runs** and returns a value. They do not check that the program **correctly formalizes the natural-language problem**. Inside the RSI loop, on self-generated natural-language problems, a core can learn to misread problems and still get rewarded. The aphasic genius's real weak point is not speaking, it is reading. Required fixes (choose and justify):
- **Reverse generation:** procedural generators produce the ground-truth program first and render it into natural-language problem text, so every in-loop problem carries an exact formalization to verify against.
- **Paraphrase consistency:** multiple paraphrases of the same problem must yield the same answer.
- **Counterfactual perturbation:** changing numbers or entities in the problem must change the answer in exactly the predicted way (GSM-Symbolic / GSM-Plus style).
- Gate on held-out perturbed twins (GSM-Plus, GSM-Symbolic) as in `reason_baseline.md`.

### 1.5 Missing required items (5/5 unless noted)
- The **"no-teacher" ablation** required by BRIEF.md §2.6 is missing everywhere. Add it, plus teacher-data logging.
- **Frontier "tin cans" are outdated** (4/5): proposals name Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro, Claude 3.7 Sonnet, Gemini 2.0/2.5 Pro. Use the brief's names: **Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra**.
- **PrOntoQA (T 85.0 → G 95.0)** was dropped by depth and curiosity; **ReClor** and **StrategyQA** by some. Restore every gated row from `reason_baseline.md`.
- **MuSR** must be gated per subtask, not on the average.
- The **cord-cut teacher** must be something that can actually run on free compute (≤ 14B on 2×T4, ~27B-class on TPU per `compute.md`), or be evaluated from a fixed, pre-computed teacher-failure set. A 70B teacher cannot run here.

### 1.6 Unjustified probabilities
"100% probability", "P > 98%", "95%", "70% for GPT-4-level" are unsupported and violate the honesty rule. Every probability needs a stated basis or must be replaced with a qualitative risk level plus the experiment that would measure it.

### 1.7 Projections shown as results
Expected scores, "95% CI lower bound" columns, and "passed at generation N" cells must be labeled **[추정]** or **[가설]**. They are projections, not measurements. Several sit exactly 1–3 points above the G threshold, which looks reverse-engineered; give the reasoning behind each number.

### 1.8 ARC-AGI targets exceed the evidence
Targets of 46–47.5% on ARC-AGI-1 exceed TRM-Att (44.6%, uses attention), while the attention-free TRM-MLP reaches 29.6% and Maze-Hard 0.0% (`tiny_sota.md`, `archs.md`). A non-attention core needs an explicit replacement for precise content lookup over its context. Specify it or lower the target with justification.

### 1.9 Parameter tables omit modules the text relies on
Recount with every module the design actually uses: the backward direction of any "bidirectional" reader (≈ +4–8M), the input embedding for the formal-output vocabulary (unless tied, state it), copy/pointer heads (needed for DROP spans), the Gated DeltaNet α-gate, output gate and short convolutions, proposer heads, any retriever encoder for Tier B rows (GPQA, MedQA, StrategyQA). The totals must stay < 100M after recounting.

### 1.10 Bootloader data totals do not add up
Several proposals list data sources that sum to more tokens than the stated D. Make the mix sum correctly.

---

## 2. Citation laundering (real papers cited for claims they do not make)

No invented URLs were found, which is good. But real sources were attached to claims they do not support. Fix every instance; if no source supports a claim, label it [추정] or [가설].

| Citation | Claimed for | What it actually says |
|---|---|---|
| arXiv 2504.13837 | "Distillation caps the student at the teacher" (used by 4/5 as the core argument) | The opposite on distillation: it finds that **distillation can add new reasoning patterns**, while **RLVR narrows the reasoning boundary** (base models beat RL-tuned models at large pass@k). This is a warning about the GRPO step in every loop. Re-examine how RLVR is used, and pair it with search/exploration that expands coverage. |
| ESMFold | "0-parameter physical verifier" | A large neural network built on the ESM-2 protein language model. Not zero-parameter, not exact. |
| Gated DeltaNet update | `S_{t-1} ⊙ (1 − β k kᵀ)` and variants | Correct form: `S_t = α_t S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`. |
| arXiv 2307.06962 (CoG, "Copy Is All You Need") | Renderer vocabulary masking | A phrase-copy language model paper. |
| arXiv 2303.08774 (GPT-4 report) | Distillation theory; span-copy pointer | Neither topic. |
| arXiv 2306.11644 (phi-1) | Distillation ceiling | Textbook-quality data paper; no ceiling result. |
| arXiv 2407.03618 (bm25s) | "~800 verifier checks/s" | A BM25 library; no verifier throughput. |
| arXiv 2511.18397 | Sandbox SHA-256 boot hashes; OPCW/SMILES hazard filter | Reward-hacking / inoculation-prompting paper. |
| AlignScore (2305.16739) | 8-gram blindfold check, swap test, suffix-automaton masking | None of these. The swap-test idea comes from `io_split.md`. |
| StripedHyena page | "No sliding window or hybrid attention" | StripedHyena **is** an attention hybrid. |
| TRM (2510.04871) | "Executing exactly 41M recurrent parameters" | TRM is a 7M model. |
| METR o3 report | nsjail sandboxes and canary files | Covers neither. |
| LLM-JEPA (2509.14252) | Latent transition models predicting verifier outcomes | A fine-tuning objective for Transformer LLMs. |
| DreamCoder (2006.08381) | "Stitch-style" compression | Stitch is a different paper. |
| Unverified scores tagged [검증됨] | MATH 43.1, DROP 70.2, StrategyQA 70.3 | `reason_baseline.md` marks these UNVERIFIED; verified values are MATH 34.1, DROP 64.1, StrategyQA 51.5. Gate on the max per the brief if you wish, but label them correctly. BBH 70.7 is text-davinci-003 (CoT-Hub), not arXiv 2312.11805 (66.6). |
| `[검증됨: archs.md]` / `[검증됨: compute.md]` | Various | Local research files are not primary sources. Cite the URL they cite, and keep their original label (many are ESTIMATE). |
| "TRC up to 110+ TPU devices" | Grant size | Unsupported; the official TRC page gives no per-project figure, and VM/storage costs apply. |
| Model names | "Qwen3-1.7B-Instruct", "Gemma 3 4B" linked to the 1B page | Use exact model names and matching links. |
| Dataset names | "FineWeb-2 Korean Educational Subset", "Cosmopedia v2 Formal STEM" | Not shown in `data.md`. Use only datasets that exist. |

---

## 3. Ideas worth keeping and grafting (do not lose these in revision)

These are the strongest ideas from Phase A. The chief architect should consider each one:
1. **Random-reward control arm as a cord-cut gate** (minimal): verifier-trained improvement must beat a random-reward control by ≥ 5 points before the teacher is cut. This directly detects fake self-improvement.
2. **Depth regret** (depth): `R = max(0, Score_deep − Score_shallow)` selects problems solvable only with more thinking; a parameter-free answer to the own-level prison.
3. **Depth-distillation ratchet** (depth): yesterday's deep-search solutions become today's shallow-pass skills, freeing deep search for harder problems.
4. **Zero-parameter skill library that carries across scales** (library): capability stored outside the weights transfers to a bigger core on day 1.
5. **Counterfactual swap test for the renderer** (4/5): corrupt the core's answer; the renderer must repeat the corrupted answer, otherwise it is solving the task.
6. **Saturation coefficient Ψ = pass@1/pass@64 × (1 − ΔDL/DL)** (library): a measurable trigger for moving to the next S-curve.
7. **EDLI = Δaccuracy per GPU-hour** (curiosity): a compute-aware domain-switch trigger.
8. **Search internalization metric E_g = rollouts needed at g−1 / rollouts needed at g to reach 80%** (search).
9. **Compression ratio CR** of solutions per generation (depth, library).
10. **Constant-size recurrent state per MCTS node, no KV cache** (search): a concrete reason a non-Transformer core suits tree search (speedup still unmeasured).
11. **Value-head pruning in latent space before sandbox execution** (curiosity): cuts the most expensive step.
12. Curiosity designs must address the **"noisy TV" problem**: ensemble variance can be maximized by noise, so learning-progress or verifier-grounded surprise is required.
