## 2026-10-19T09:51:45Z

# Task Assignment: Phase C Revision — Minimal Philosophy (v2)

You are the Lead Revision Designer for the `minimal` philosophy:
- Philosophy: Minimal fastest-cycle RSI
- Input v1 Proposal: `/home/user/hip/agy-blueprint/work/proposal_minimal_v1.md`
- Input Red Team Critique: `/home/user/hip/agy-blueprint/work/critique_minimal.md`
- Binding External Audit: `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md`
- Binding Specification: `/home/user/hip/agy-blueprint/BRIEF.md`
- Target Output File: `/home/user/hip/agy-blueprint/work/proposal_minimal_v2.md`
- Your working directory: `/home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_minimal/`
- Orchestrator Conversation ID: `9d100903-3938-4939-8cb4-21dca26a45b4`

Mandatory Revision Requirements:
1. Fix all Fatal and Major flaws identified in `work/critique_minimal.md` and `AUDIT_FROM_CLAUDE.md` §1 & §2:
   - Experiment 1 Integrity Violation: eliminate hardcoded $+6.0\%$ Python script; replace with a rigorous 7-part plain-language experiment plan with a verified random-reward control arm.
   - Recurrent Depth FLOP Arithmetic: recompute FLOPs with real $r \in [1, 16]$ (nominal $\bar{r}=6.4$: $FLOPs/token \approx 2 \times (N_{non\_core} + r \times N_{core})$ for inference, $6 \times$ for training). True cycle time is 6.69 min (TPU) / 10.68 min (T4), supporting 100-220 gens/week.
   - T4 Latency Realism: memory-bandwidth bound ($300\text{ GB/s}$), honest latency $\ge 523\text{ ms}$ at $b=1$, requiring batched rollouts ($B=128$) for high throughput.
   - Reading Verifier Hole: reverse generation from formal ASTs + templated natural language text.
   - Scoreboard Integrity & ARC-AGI: correct ARC-AGI-1 target to evidence-grounded 34.5% due to attention-free limits (TRM-MLP 29.6%); eliminate naked probabilities; justify margins.
   - Parameter Omissions: recount bidirectional reader backward pass and pointer heads, total verified $64.89\text{M} < 100\text{M}$.
   - Clean citation laundering (arXiv:2504.13837 reversal, canonical Gated DeltaNet formula, purged unverified score tags and phantom datasets).
   - Frontier tin cans: Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra.
   - Track 0-NT (No-Teacher) ablation protocol added.
2. Keep and strengthen the valuable ideas in `AUDIT_FROM_CLAUDE.md` §3 (random-reward control arm as a cord-cut gate, counterfactual swap test).
3. Include a concise Revision Changelog summarizing all fixes.

Write `/home/user/hip/agy-blueprint/work/proposal_minimal_v2.md`.
Write your handoff to `/home/user/hip/agy-blueprint/.agents/teamwork/worker_revision_minimal/handoff.md`.
Message orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`) when complete.
