# BRIEFING — 2026-10-19T09:57:00Z

## Mission
Authoritative revision of the 'curiosity' philosophy proposal (proposal_curiosity_v2.md) for the Aphasic Genius RSI Engine, strictly resolving all fatal and major flaws identified in the red team critique and external audit while honoring all architectural and benchmark constraints.

## 🔒 My Identity
- Archetype: Lead Revision Designer
- Roles: implementer, qa, specialist
- Working directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_curiosity/
- Original parent: 9d100903-3938-4939-8cb4-21dca26a45b4
- Milestone: Revision Curiosity v2

## 🔒 Key Constraints
- Fix all Fatal and Major flaws identified in critique_curiosity.md and AUDIT_FROM_CLAUDE.md §1 & §2.
- Recalculate FLOPs with effective parameters N_eff for r <= 16, honest rollout/training FLOPs.
- Fix T4 latency: memory-bandwidth bound sequential decoding (240 GB/s), true latency floor ≈ 162 ms.
- Solve Reading Verifier Hole: reverse procedural generation + counterfactual twin invariants.
- Neutralize Noisy TV Trap: replace pure latent ensemble variance with Verifier-Grounded Empirical Learning Progress (ELP).
- Secret Transformer: purge potion-base-8M retriever; replace with BM25s (0 params) or Gated DeltaNet bi-encoder.
- NO CODE: Replace Experiment 1 code with a rigorous 7-part plain-language plan with a random-reward control arm.
- Parameter Omissions: recount all modules honestly, verified < 100M total.
- Clean citation laundering (LLM-JEPA, ESMFold, etc.).
- Frontier tin cans: Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra.
- Restore PrOntoQA (G=95.0), ReClor (G=71.6), and unbundle MuSR subtasks.
- Keep and strengthen AUDIT §3 ideas (EDLI trigger, value-head pruning in latent space, counterfactual swap test).
- Write work/proposal_curiosity_v2.md directly (NO scripts like generate_proposal.py or assemble.py).
- Follow all 12 mandatory sections in exact order per BRIEF.md §4.
- Concise Revision Changelog at top of proposal_curiosity_v2.md.
- Write handoff.md and send completion message to orchestrator (9d100903-3938-4939-8cb4-21dca26a45b4).

## Current Parent
- Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4
- Updated: 2026-10-19T09:51:45Z

## Task Summary
- **What to build**: Comprehensive, defect-free proposal_curiosity_v2.md fixing v1 critique and Claude audit.
- **Success criteria**: Zero fatal/major flaws, math checks out (FLOPs, latency, params < 100M), all 12 sections present, all benchmark tables restored and precise, verified self-consistency.
- **Interface contracts**: BRIEF.md §4, AUDIT_FROM_CLAUDE.md, critique_curiosity.md.
- **Code layout**: /home/user/hip/agy-blueprint/work/proposal_curiosity_v2.md

## Change Tracker
- **Files modified**:
  - `/home/user/hip/agy-blueprint/work/proposal_curiosity_v2.md`: Completed authoritative 12-section technical specification with Revision Changelog.
- **Build status**: Verification passed; 0 code blocks, parameter math verified (68,328,193 params), FLOP math verified ($N_{\text{eff}} = 305.69\text{M}$), latency grounded in memory bandwidth.
- **Pending issues**: None. Ready for handoff and orchestrator notification.

## Quality Status
- **Build/test result**: Pass (specification and math validation)
- **Lint status**: Clean (valid markdown, consistent notation)
- **Tests added/modified**: N/A (specification document)

## Loaded Skills
- None required

## Key Decisions Made
- Parameter recount: 68,328,193 parameters with BM25s (0-param) and 2-layer Gated DeltaNet bi-encoder (2.10M params).
- FLOPs: $N_{\text{eff}}(16) = 305.69\text{M}$, rollout FLOPs = $4.191 \times 10^{15}$, training FLOPs = $4.585 \times 10^{16}$.
- Latency: Evaluated at 240 GB/s GDDR6 bandwidth $\implies 172.0\text{ ms}$ baseline floor, reduced to $\approx 48.4\text{ ms}$ with dynamic early exit.
- Reading verifier hole solved via reverse procedural generation, counterfactual twin invariants, and paraphrase consistency checks.
- Noisy TV trap eliminated via Verifier-Grounded Empirical Learning Progress (ELP).
- Experiment 1 specified as a 7-part plain-language plan with a random-reward control arm ($\ge 5.0\%$ margin required).

## Artifact Index
- `/home/user/hip/agy-blueprint/work/proposal_curiosity_v2.md` — Authoritative revised v2 proposal
- `/home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_curiosity/handoff.md` — 5-component handoff report
- `/home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_curiosity/progress.md` — Liveness log
