# Task Assignment: Phase C Revision — Depth Philosophy (v2)

You are the Lead Revision Designer for the `depth` philosophy:
- Philosophy: Depth-distillation RSI (latent recursive reasoner)
- Input v1 Proposal: `/home/user/hip/agy-blueprint/work/proposal_depth_v1.md`
- Input Red Team Critique: `/home/user/hip/agy-blueprint/work/critique_depth.md`
- Binding External Audit: `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md`
- Binding Specification: `/home/user/hip/agy-blueprint/BRIEF.md`
- Target Output File: `/home/user/hip/agy-blueprint/work/proposal_depth_v2.md`
- Your working directory: `/home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_depth/`
- Orchestrator Conversation ID: `9d100903-3938-4939-8cb4-21dca26a45b4`

Mandatory Revision Requirements:
1. Fix all Fatal and Major flaws identified in `work/critique_depth.md` and `AUDIT_FROM_CLAUDE.md` §1 & §2:
   - Distillation Collapse Paradox: replace shallow/deep cosine state matching with token-level formal AST policy distillation to preserve test-time depth amplification.
   - Recurrent Depth FLOP Arithmetic: recompute FLOPs with real r recursions ($r \in [8, 16]$: $FLOPs/token \approx 2 \times (N_{non\_core} + r \times N_{core})$ for inference, $6 \times$ for training). Use curriculum depth and TPU v5e-8 pretraining to strictly respect Kaggle session/quota limits.
   - T4 Latency Realism: memory-bandwidth bound sequential decoding ($300\text{ GB/s}$), batched rollouts ($B=32$), honest latencies.
   - Reading Verifier Hole: reverse procedural generation + GSM-Plus/Symbolic perturbation twins.
   - NO CODE: replace any executable Python/script code in Experiment 1 with a rigorous 7-part plain-language plan (inputs, procedure, measurement, random-reward control arm, pass/kill criteria, compute and wall-clock time).
   - Parameter Omissions: recount backward reader, DeltaNet $\alpha$/output gates, proposer heads; re-dimension to verified total < 100M.
   - Clean citation laundering (phi-1, TRM, bm25s, ESMFold, etc.).
   - Frontier tin cans: Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra.
   - Restore PrOntoQA ($G=95.0$), StrategyQA ($G=80.3$), decoupled MuSR.
2. Keep and strengthen the valuable ideas in `AUDIT_FROM_CLAUDE.md` §3 (depth regret $R = \max(0, \text{Score}_{\text{deep}} - \text{Score}_{\text{shallow}})$, depth-distillation ratchet, counterfactual swap test).
3. Include a concise Revision Changelog summarizing all fixes.

Write `/home/user/hip/agy-blueprint/work/proposal_depth_v2.md`.
Write your handoff to `/home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_depth/handoff.md`.
Message orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`) when complete.

## 2026-10-19T10:00:22Z
[Message] timestamp=2026-10-19T10:00:22Z sender=9d100903-3938-4939-8cb4-21dca26a45b4 priority=MESSAGE_PRIORITY_HIGH content=**Context**: Phase C Revision — Depth Philosophy
**Content**: We observed `work/proposal_depth_v2.md` is complete on disk. Please finalize `handoff.md` and send your completion report to orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`).
**Action**: Send completion report now.
