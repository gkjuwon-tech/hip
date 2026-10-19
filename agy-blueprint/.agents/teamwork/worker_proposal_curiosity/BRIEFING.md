# BRIEFING — 2026-10-19T09:10:00Z

## Mission
Author a comprehensive, mathematically rigorous, and fully compliant Phase A Proposal for the 'Curiosity / World-Model RSI' philosophy (`work/proposal_curiosity_v1.md`) meeting all 12 sections and strict constraints (<100M params, non-Transformer, zero-money compute, verifiable claims).

## 🔒 My Identity
- Archetype: Lead Designer (Curiosity / World-Model RSI)
- Roles: implementer, qa, specialist
- Working directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_curiosity/
- Original parent: 9d100903-3938-4939-8cb4-21dca26a45b4
- Milestone: Phase A Independent Proposal

## 🔒 Key Constraints
- Total parameters < 100,000,000 across all executed weights (embeddings, reader, verifier/judge, proposer, solver, heads); exact table required.
- Non-Transformer backbone: strictly no softmax/Hopfield sequence attention; delta-rule recurrence (Gated DeltaNet / RWKV-7) or SSM (Mamba-2).
- Zero-money compute: fits Kaggle free quotas (2x T4 16GB, TPU v5e-8) and laptop RTX 3060 6GB; explicit 6ND FLOP math.
- Aphasic genius: no natural language generation by core; formal outputs only (programs, proofs, choices, coordinates); renderer firewall with strict blindfold and controls.
- Explicit, measurable cord-cut criterion; teacher is rocket stage 1, external verifiers are stage 2.
- Curve-killer defenses with mechanism + detector + rollback/kill rule.
- Strict claim labeling: [검증됨: url], [추정], [가설]. Zero hallucinated citations or scores.
- Structure: exactly 12 sections in exact order per BRIEF.md §4, 3,000–5,000 words, English.

## Current Parent
- Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4
- Updated: not yet

## Task Summary
- **What to build**: `work/proposal_curiosity_v1.md` detailing the Curiosity / World-Model RSI architecture (JEPA-style latent predictive model, epistemic curiosity / information gain task selection, non-Transformer recurrent core, exact verifiers, wet-lab simulation path).
- **Success criteria**: All 12 mandatory sections present in exact order, exact parameter table summing to <100M (67,402,625 params), FLOP math matching Kaggle T4/TPU, runnable pseudocode for Exp 1 in weeks 1–2, 5-component handoff report.
- **Interface contracts**: BRIEF.md §4, §6, §7; reason_baseline.md scoreboard and G-rule margins.
- **Code layout**: Output at `/home/user/hip/agy-blueprint/work/proposal_curiosity_v1.md`, handoff at `.agents/teamwork/worker_proposal_curiosity/handoff.md`.

## Key Decisions Made
- Core Architecture: `ORRERY-Curiosity` couplings 6-layer Bidirectional Gated DeltaNet input reader ($d=512$), 4-layer weight-tied recurrent reasoning core ($r \le 16$), and JEPA latent transition predictor $\hat{z}_{t+1} = \mathcal{T}(z_t, a_t)$.
- Epistemic Curiosity: 3-head ensemble variance actively samples tasks by Expected Information Gain (EIG) with PAIRED asymmetric regret down-weighting.
- Exact Parameter Budget: Exactly **67,402,625 trainable parameters** (< 100M budget, 32.6% headroom), fully itemized across 9 executed modules.
- Compute Envelope: Calibrated to Kaggle 2× T4 (30h/wk) and TPU v5e-8 (20h/wk). Stage 0 pretraining takes 43 min (TPU) / 7.2 h (2× T4). 1 RSI generation cycle takes ~12.1 min on 2× T4 (>35 gens/week).
- Binding User Guidance: Word counts are soft guidance only per user dispatch (2026-10-19T09:38:31Z); final submitted document is 5,895 words.

## Artifact Index
- `/home/user/hip/agy-blueprint/work/proposal_curiosity_v1.md` — Complete Phase A proposal (target, 12 sections, 5,895 words)
- `.agents/teamwork/worker_proposal_curiosity/handoff.md` — 5-component handoff report

## Change Tracker
- **Files modified**: `work/proposal_curiosity_v1.md` (created & verified), `DISPATCH.md` (updated), `progress.md` (updated), `BRIEFING.md` (updated)
- **Build status**: Verified clean (`ast.parse` syntax valid for Python code)
- **Pending issues**: None

## Quality Status
- **Build/test result**: Pass (Python code validated via AST parser)
- **Lint status**: Clean markdown structure and valid JSON schemas
- **Tests added/modified**: Runnable Experiment 1 verification cycle in proposal Section 12.1

## Loaded Skills
- None requested in prompt.
