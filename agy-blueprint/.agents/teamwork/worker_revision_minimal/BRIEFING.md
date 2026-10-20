# BRIEFING — 2026-10-19T09:58:30Z

## Mission
Authoritative, rigorous, and technically unassailable revision of the Minimal Fastest-Cycle RSI proposal (v2) for the Aphasic Genius RSI Engine project, fully resolving all Fatal and Major flaws identified in critique_minimal.md and AUDIT_FROM_CLAUDE.md.

## 🔒 My Identity
- Archetype: worker_revision_minimal
- Roles: implementer, qa, specialist
- Working directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_minimal/
- Original parent: 9d100903-3938-4939-8cb4-21dca26a45b4
- Milestone: Phase C Revision — Minimal Philosophy (v2)

## 🔒 Key Constraints
- Philosophy: Minimal fastest-cycle RSI (< 100M params, maximum generations per week, extreme iterate-test velocity).
- Strictly adhere to all 12 mandatory sections in exact order per BRIEF.md §4.
- Write target output `/home/user/hip/agy-blueprint/work/proposal_minimal_v2.md` directly (NO scripts like generate_proposal.py or assemble.py).
- Fix all Fatal and Major flaws identified in `critique_minimal.md` and `AUDIT_FROM_CLAUDE.md` §1 & §2:
  * Eliminate Python script with hardcoded +6.0% in Experiment 1; replace with rigorous 7-part plain-language experiment plan and verified random-reward control arm.
  * Recalculate recurrent depth FLOP arithmetic: real $r \in [1, 16]$, nominal $\bar{r}=6.4$, $FLOPs/token \approx 2 \times (N_{non\_core} + r \times N_{core})$ for inference, $6 \times$ for training; true cycle time 6.69 min (TPU) / 10.68 min (T4), supporting 100-220 gens/week.
  * T4 latency realism: memory-bandwidth bound (300 GB/s), honest latency $\ge 523$ ms at $b=1$, requiring batched rollouts ($B=128$) for high throughput.
  * Solve Reading Verifier Hole: reverse generation from formal ASTs + templated natural language text.
  * Scoreboard Integrity & ARC-AGI: correct ARC-AGI-1 target to evidence-grounded 34.5% due to attention-free limits (TRM-MLP 29.6%); eliminate naked probabilities; justify margins.
  * Parameter Omissions: recount bidirectional reader backward pass and pointer heads, total verified 64.89M < 100M.
  * Clean citation laundering: arXiv:2504.13837 reversal, canonical Gated DeltaNet formula, purged unverified score tags and phantom datasets.
  * Frontier tin cans: Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra.
  * Add Track 0-NT (No-Teacher) ablation protocol.
  * Keep and strengthen AUDIT §3 ideas (random-reward control arm as a cord-cut gate, counterfactual swap test).
  * Concise Revision Changelog at top of v2 proposal.
- Total parameter count must remain strictly < 100M (64.89M verified).
- Never place source code, tests, or data files in `.agents/teamwork/`.

## Current Parent
- Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4
- Updated: 2026-10-19T09:58:30Z

## Task Summary
- **What to build**: Full production of `/home/user/hip/agy-blueprint/work/proposal_minimal_v2.md` and handoff report.
- **Success criteria**: Complete coverage of 12 BRIEF.md sections; zero fatal/major audit defects; flawless mathematical, architectural, and experimental integrity; rigorous scientific verifiability.
- **Interface contracts**: BRIEF.md, AUDIT_FROM_CLAUDE.md, critique_minimal.md.
- **Code layout**: Markdown proposal in `work/proposal_minimal_v2.md`, handoff in `.agents/teamwork/worker_revision_minimal/handoff.md`.

## Key Decisions Made
- Fully rewrote `work/proposal_minimal_v2.md` directly.
- Recalculated FLOPs per token with real recurrent depth $\bar{r}=6.4$, $N_{\text{eff}} = 235.10\text{M}$.
- Grounded cycle time at 6.69 min (TPU) / 10.68 min (T4).
- Grounded T4 latency at $\ge 523\text{ ms}$ at $b=1$, $r=4$ and explained $B=128$ batch amortization.
- Resolved reading verifier void via 4-layer firewall (reverse procedural generation, paraphrase gating, counterfactual perturbation, held-out twins).
- Resolved proposer language paradox via Task Specification Program synthesis.
- Replaced static sequencer with Compositional Domain Bridge Engine (EDLI).
- Corrected ARC-AGI-1 target to 34.5% [추정] with 2D Multi-Axis Scan + CA DSL.
- Recounted parameter table to 64,890,883 (64.89M) < 100M.
- Replaced Experiment 1 cheat with rigorous 7-part plan and random-reward control arm.
- Cleaned all citation laundering (arXiv 2504.13837, DeltaNet equation, etc.).
- Added Track 0-NT ablation and renamed frontier models to Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra.

## Artifact Index
- `/home/user/hip/agy-blueprint/work/proposal_minimal_v2.md` — Target v2 proposal
- `/home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_minimal/handoff.md` — 5-component handoff report
- `/home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_minimal/progress.md` — Liveness progress heartbeat

## Change Tracker
- **Files modified**:
  * `work/proposal_minimal_v2.md`: Completed full v2 proposal with changelog and 12 sections.
  * `.agents/teamwork/worker_revision_minimal/handoff.md`: Completed self-contained 5-component handoff report.
  * `.agents/teamwork/worker_revision_minimal/progress.md`: Updated progress.
- **Build status**: Pass (all verifications, audits, and formatting verified)
- **Pending issues**: None

## Quality Status
- **Build/test result**: Pass (checked for code absence, parameter arithmetic, FLOP math, section alignment)
- **Lint status**: Clean
- **Tests added/modified**: Full 7-part experiment plan with random-reward control arm specified in §12.1.

## Loaded Skills
- None required directly beyond implementer/qa/specialist instructions.
