## 2026-10-19T09:51:45Z
You are the Lead Revision Designer for the 'curiosity' philosophy (Curiosity / world-model RSI) on the Aphasic Genius RSI Engine project.

Assigned directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_curiosity/
Input v1 proposal: /home/user/hip/agy-blueprint/work/proposal_curiosity_v1.md
Input red team critique: /home/user/hip/agy-blueprint/work/critique_curiosity.md
Target v2 proposal: /home/user/hip/agy-blueprint/work/proposal_curiosity_v2.md
Handoff file: /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_curiosity/handoff.md
Binding external audit: /home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md
Binding specification: /home/user/hip/agy-blueprint/BRIEF.md
Original request: /home/user/hip/agy-blueprint/ORIGINAL_REQUEST.md
Research directory: /home/user/hip/asi-blueprint/research/
Orchestrator Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4

Instructions:
1. Read /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_curiosity/DISPATCH.md.
2. Read work/proposal_curiosity_v1.md, work/critique_curiosity.md, and AUDIT_FROM_CLAUDE.md in full.
3. Write work/proposal_curiosity_v2.md directly (NO scripts like generate_proposal.py or assemble.py).
4. Strictly follow all 12 mandatory sections in exact order per BRIEF.md §4:
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
5. Add a concise Revision Changelog at the top of proposal_curiosity_v2.md.
6. Write your handoff to /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_curiosity/handoff.md.
7. Send a completion message to orchestrator (9d100903-3938-4939-8cb4-21dca26a45b4).
