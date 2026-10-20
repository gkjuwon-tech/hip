# BRIEFING — 2026-10-19T09:57:15Z

## Mission
Author the comprehensive, rigorous v2 proposal for the Search philosophy (Expert-iteration RSI / AlphaZero-style) resolving all critique and audit findings.

## 🔒 My Identity
- Archetype: worker_revision_search
- Roles: implementer, qa, specialist
- Working directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_search
- Original parent: 9d100903-3938-4939-8cb4-21dca26a45b4
- Milestone: Phase C Revision — Search Philosophy (v2)

## 🔒 Key Constraints
- Philosophy: Expert-iteration RSI (AlphaZero-style)
- Strictly follow all 12 mandatory sections in exact order per BRIEF.md §4
- Direct writing of proposal_search_v2.md (NO Python generation/assembly scripts)
- Address all Fatal and Major flaws from critique_search.md and AUDIT_FROM_CLAUDE.md §1 & §2
- Recurrent Depth FLOP recalculation with real r in [1, 8] (FLOPs/token ≈ 2 × (N_non_core + r × N_core) for inference, 6 × for training)
- T4 latency realism (memory-bandwidth bound sequential decoding @ 300 GB/s, Snell adaptive search, downscaled task batching to fit 2.0h/gen, 10-12 gens/week)
- Tri-Verification Reading Harness (reverse procedural generation, 3-way paraphrase consistency, counterfactual perturbation)
- Own-Level Prison & Cold-Start solutions: Depth Regret, 50% procedural curriculum injection, backward goal decomposition
- NO CODE: Replace Experiment 1 Python code with a rigorous 7-part plain-language experiment plan with random-reward control arm
- Parameter omissions recount (< 100M total verified, including DROP pointer head, formal IR embeddings, short convs, bidirectional layers)
- Clean citation laundering (distillation theory, Stitch, AlignScore, Gated DeltaNet update)
- Frontier tin cans: Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra
- Restore PrOntoQA (G=95.0) and unbundle MuSR subtasks
- Keep and strengthen AUDIT §3 ideas (search internalization metric E_g, constant-size recurrent state per MCTS node / no KV cache, counterfactual swap test)
- Concise Revision Changelog at top of proposal_search_v2.md

## Current Parent
- Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4
- Updated: 2026-10-19T09:51:45Z

## Task Summary
- **What to build**: Flaw-free Revision v2 Proposal for the Search philosophy (`proposal_search_v2.md`) and handoff report (`handoff.md`).
- **Success criteria**: Complete resolution of fatal/major critiques, rigorous FLOP and latency budgets, 12 mandatory sections per BRIEF.md §4, zero python code in exp 1.
- **Interface contracts**: `/home/user/hip/agy-blueprint/BRIEF.md`, `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md`, `/home/user/hip/agy-blueprint/work/critique_search.md`
- **Code layout**: Written directly to `/home/user/hip/agy-blueprint/work/proposal_search_v2.md`

## Key Decisions Made
- Fully recalculated all FLOPs with $r=4$ (median) and $r=8$ (max), finding $N_{\text{eff}} = 214.6\text{M}$ ($4.292 \times 10^8$ infer, $1.288 \times 10^9$ train FLOPs/token).
- Grounded T4 sequential decoding latency in memory bandwidth ($230\text{ GB/s}$ sustained), yielding 0.202s per 100-token greedy pass and 0.9–1.25s per task with Snell adaptive MCTS (18 mean simulations).
- Downscaled batching to 5,000 tasks/gen to ensure a 1.0–2.0h generation cycle, fitting 10–12 generations/week within Kaggle free quotas.
- Established the Tri-Verification Reading Harness (reverse procedural generation, 3-way paraphrase consistency, counterfactual perturbation).
- Solved Own-Level Prison and Cold-Start with parameter-free Depth Regret, 50% procedural curriculum injection, and backward goal decomposition.
- Recounted all parameter omissions: DROP pointer head (0.41M), formal IR embeddings (1.31M), 2 bi-layers (22.16M), total inference weights = 81,639,425 (< 100M).
- Purged all code from Experiment 1, replacing with a complete 7-part plain-language experimental plan featuring a random-reward control arm.
- Cleaned all cited papers, updated frontier models to Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra, and restored all scoreboard rows.

## Artifact Index
- `/home/user/hip/agy-blueprint/work/proposal_search_v2.md` — Target v2 proposal (written and verified)
- `/home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_search/handoff.md` — Handoff report

## Change Tracker
- **Files modified**: `work/proposal_search_v2.md` (created), `BRIEFING.md` (updated), `progress.md` (updated)
- **Build status**: PASS (Document complete, self-consistent, and fully verified)
- **Pending issues**: Writing handoff.md and sending completion message

## Quality Status
- **Build/test result**: PASS (All 12 sections present, all fatal/major critique flaws resolved)
- **Lint status**: Zero code blocks in Experiment 1; format fully compliant with project guidelines
- **Tests added/modified**: 7-part plain-language experiment plan with random-reward control arm specified

## Loaded Skills
- None
