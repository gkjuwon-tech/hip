# BRIEFING — 2026-10-19T09:55:00Z

## Mission
Revise the 'depth' philosophy proposal into work/proposal_depth_v2.md, resolving all fatal and major flaws from critique_depth.md and AUDIT_FROM_CLAUDE.md §1 & §2, strengthening §3 ideas, updating parameters, FLOPs, latencies, reading verification, scoreboard, and experiment plan with NO CODE.

## 🔒 My Identity
- Archetype: Lead Revision Designer (Depth Philosophy)
- Roles: implementer, qa, specialist
- Working directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_depth/
- Original parent: 9d100903-3938-4939-8cb4-21dca26a45b4
- Milestone: Phase C Revision (proposal_depth_v2.md)

## 🔒 Key Constraints
- Fix all Fatal and Major flaws identified in critique_depth.md and AUDIT_FROM_CLAUDE.md §1 & §2
- Solve Distillation Collapse Paradox: replace shallow/deep cosine state matching with token-level formal AST policy distillation
- Recompute FLOPs with real r recursions ($r \in [8, 16]$: FLOPs/token $\approx 2 \times (N_{non\_core} + r \times N_{core})$ for inference, $6 \times$ for training). Use curriculum depth and TPU v5e-8 pretraining to respect Kaggle session/quota limits
- Fix T4 latency: memory-bandwidth bound sequential decoding ($300\text{ GB/s}$), batched rollouts ($B=32$), honest numbers
- Solve Reading Verifier Hole: reverse procedural generation + GSM-Plus/Symbolic perturbation twins
- NO CODE: Replace Experiment 1 Python code with a rigorous 7-part plain-language experiment plan
- Parameter Omissions: recount backward reader, DeltaNet $\alpha$/output gates, proposer heads, verified total < 100M (re-dimensioned to 75.93M)
- Clean citation laundering (phi-1, TRM, bm25s, ESMFold, etc.)
- Frontier tin cans: Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra
- Restore PrOntoQA ($G=95.0$), StrategyQA ($G=80.3$), decoupled MuSR subtasks
- Keep and strengthen AUDIT §3 ideas (depth regret $R = \max(0, \text{Score}_{\text{deep}} - \text{Score}_{\text{shallow}})$, depth-distillation ratchet, counterfactual swap test)
- Add a concise Revision Changelog at the top of proposal_depth_v2.md
- Write proposal_depth_v2.md directly (NO scripts like generate_proposal.py)

## Current Parent
- Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4
- Updated: 2026-10-19T09:55:00Z

## Task Summary
- **What to build**: Comprehensive, flaw-free, mathematically rigorous proposal_depth_v2.md adhering to 12 mandatory sections
- **Success criteria**: All audit & critique flaws resolved, clean honest calculations, zero code, verified sources
- **Interface contracts**: BRIEF.md §4, AUDIT_FROM_CLAUDE.md, critique_depth.md

## Change Tracker
- **Files modified**: work/proposal_depth_v2.md (created and validated)
- **Build status**: PASS — all 12 sections present, all math and parameter tables verified
- **Pending issues**: None

## Quality Status
- **Build/test result**: Pass (mathematical verification of parameter sums, FLOP schedules, and latencies)
- **Lint status**: Clean (zero code contraband, pure plain-language experiment plan)
- **Tests added/modified**: 7-part plain-language Experiment 1 plan in §12.2

## Loaded Skills
- None required

## Key Decisions Made
- Re-dimension core to $d=576, d_{ffn}=1440$ with 2-layer bidirectional + 2-layer causal reader, 8-layer weight-tied recurrent core, 2-layer causal emitter. Total 75,928,490 params.
- Adopt dual-objective policy-level distillation ratchet (formal AST token distillation) instead of cosine latent state matching.
- Move Stage 0 pretraining to TPU v5e-8 ($1.02\text{ h}$) with curriculum recurrence depth ($r=1 \to 2 \to 4$), total $1.45 \times 10^{18}$ FLOPs.
- Ground T4 sequential decoding latency in $300\text{ GB/s}$ memory bandwidth ($3.53\text{ ms/token}$ at $r=8$ batch 1, $176.5\text{ ms}$ for 50 tokens; batched $B=32$ for high-throughput rollouts).
- Plug Reading Verifier Hole using 4-tier symmetric reading verification: reverse procedural generation, 3-way paraphrase invariance, counterfactual perturbation twins (GSM-Plus/Symbolic), and held-out reading gates.
- Strictly adhere to NO CODE constraint in §12: complete 7-part plain-language experiment plan.

## Artifact Index
- work/proposal_depth_v2.md — Target revised proposal
- .agents/teamwork/worker_revision_depth/handoff.md — Handoff report
