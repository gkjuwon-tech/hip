# Task Assignment: Phase C Revision — Search Philosophy (v2)

## 2026-10-19T09:51:45Z
You are the Lead Revision Designer for the 'search' philosophy (Expert-iteration RSI / AlphaZero-style) on the Aphasic Genius RSI Engine project.

Assigned directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_search/
Input v1 proposal: /home/user/hip/agy-blueprint/work/proposal_search_v1.md
Input red team critique: /home/user/hip/agy-blueprint/work/critique_search.md
Target v2 proposal: /home/user/hip/agy-blueprint/work/proposal_search_v2.md
Handoff file: /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_search/handoff.md
Binding external audit: /home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md
Binding specification: /home/user/hip/agy-blueprint/BRIEF.md
Original request: /home/user/hip/agy-blueprint/ORIGINAL_REQUEST.md
Research directory: /home/user/hip/asi-blueprint/research/
Orchestrator Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4

Instructions:
1. Read /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_search/DISPATCH.md.
2. Read work/proposal_search_v1.md, work/critique_search.md, and AUDIT_FROM_CLAUDE.md in full.
3. Write work/proposal_search_v2.md directly (NO scripts like generate_proposal.py or assemble.py).
4. Strictly follow all 12 mandatory sections in exact order per BRIEF.md §4:
   - Fix all Fatal and Major flaws identified in critique_search.md and AUDIT_FROM_CLAUDE.md §1 & §2.
   - Recalculate FLOPs with real r in [1, 8] (FLOPs/token ≈ 2 × (N_non_core + r × N_core) for inference, 6 × for training).
   - Fix T4 latency: memory-bandwidth bound sequential decoding (300 GB/s), Snell adaptive search (downscaled task batching to fit 2.0h/gen, 10-12 gens/week).
   - Solve Reading Verifier Void: Tri-Verification Reading Harness (reverse procedural generation, 3-way paraphrase consistency, counterfactual perturbation).
   - Solve Own-Level Prison & Cold-Start: integrate Depth Regret, 50% procedural curriculum injection, backward goal decomposition.
   - NO CODE: Replace Experiment 1 Python code with a rigorous 7-part plain-language experiment plan with a random-reward control arm.
   - Parameter Omissions: recount DROP pointer head, formal IR embeddings, short convs, bidirectional layers, verified < 100M total.
   - Clean citation laundering (distillation theory, Stitch, AlignScore, Gated DeltaNet update).
   - Frontier tin cans: Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra.
   - Restore PrOntoQA (G=95.0) and unbundle MuSR subtasks.
   - Keep and strengthen AUDIT §3 ideas (search internalization metric E_g, constant-size recurrent state per MCTS node / no KV cache, counterfactual swap test).
5. Add a concise Revision Changelog at the top of proposal_search_v2.md.
6. Write your handoff to /home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_search/handoff.md.
7. Send a completion message to orchestrator (9d100903-3938-4939-8cb4-21dca26a45b4).
