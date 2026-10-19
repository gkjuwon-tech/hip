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
- **Success criteria**: All 12 mandatory sections present in exact order, 3,000–5,000 words, exact parameter table summing to <100M, FLOP math matching Kaggle T4/TPU, runnable pseudocode for Exp 1 in weeks 1–2, handoff report.
- **Interface contracts**: BRIEF.md §4, §6, §7; reason_baseline.md scoreboard and G-rule margins.
- **Code layout**: Output at `/home/user/hip/agy-blueprint/work/proposal_curiosity_v1.md`, handoff at `.agents/teamwork/worker_proposal_curiosity/handoff.md`.

## Key Decisions Made
- Core Architecture: JEPA-style latent world model over Gated DeltaNet recurrent trunk. Learns state transition in latent space $s_{t+1} = T(s_t, a_t)$ and predicts verifier outcome $\hat{v} = V(s_t, a_t)$ without text autoregression.
- Curiosity Engine: Selects tasks and exploration paths via Epistemic Curiosity / Expected Information Gain (prediction error of the ensemble/JEPA head or epistemic variance), directly avoiding own-level prison and plateau.
- Parameter Budget: Allocation strictly totaling ~86.4M parameters (<100M total budget), including byte/subword input embeddings, Gated DeltaNet backbone, recurrent depth blocks, JEPA predictor, value/verifier heads, and action/program token heads.
- Compute Envelope: Calibrated to Kaggle 2x T4 (30h/wk) and TPU v5e-8 (20h/wk) with explicit 6ND calculations per generation.

## Artifact Index
- `/home/user/hip/agy-blueprint/work/proposal_curiosity_v1.md` — Complete Phase A proposal (target)
- `.agents/teamwork/worker_proposal_curiosity/handoff.md` — 5-component handoff report

## Change Tracker
- **Files modified**: DISPATCH.md (updated with dispatch timestamp and message), BRIEFING.md (created)
- **Build status**: N/A (documentation/design phase)
- **Pending issues**: None

## Quality Status
- **Build/test result**: In progress
- **Lint status**: Clean
- **Tests added/modified**: N/A

## Loaded Skills
- None requested in prompt.
