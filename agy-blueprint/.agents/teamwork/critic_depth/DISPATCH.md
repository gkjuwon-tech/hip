# Task Assignment: Phase B Red Team Critique — Depth Philosophy

You are the Lead Red Team Critic attacking the `depth` proposal (`work/proposal_depth_v1.md`):
- Persona: Top ML/RL researcher + Kaggle grandmaster (T4/TPU quotas) + benchmark-integrity auditor.
- Target Proposal: `/home/user/hip/agy-blueprint/work/proposal_depth_v1.md`
- Target Output File: `/home/user/hip/agy-blueprint/work/critique_depth.md`
- Binding Specification: `/home/user/hip/agy-blueprint/BRIEF.md`
- Binding External Audit: `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md` (MUST read and attack all issues in §1 and §2 in addition to your own findings!)
- Your working directory: `/home/user/hip/agy-blueprint/.agents/teamwork/critic_depth/`
- Orchestrator Conversation ID: `9d100903-3938-4939-8cb4-21dca26a45b4`

Binding Rules:
1. NO CODE anywhere in the critique. Any recommended experimental fix must be specified as a plan only (inputs, procedure, measurement, control arm, pass/kill criteria, compute and wall-clock time).
2. Word count is soft guidance; focus on technical ruthlessness and actionable concrete fixes.
3. Attack in priority order:
   - The RSI loop (teacher ceiling escape, cord-cut feasibility, reading verifier hole per AUDIT §1.4, depth distillation vs collapse, own-level prison, next domain mechanism).
   - Recurrent depth FLOP math (AUDIT §1.1): recalculate FLOPs with real r recursions ($r \in [16, 32]$: $FLOPs/token \approx 2 \times (N_{non\_core} + r \times N_{core})$ for inference, $6 \times$ for training). Check if 24h T4 budget explodes.
   - Latency realism (AUDIT §1.2): memory-bandwidth bound sequential decoding on T4.
   - Missing rows: PrOntoQA (T 85.0 -> G 95.0), MuSR subtasks (AUDIT §1.5).
   - Scoreboard claims, margins, ARC-AGI lookup, and unjustified probabilities (AUDIT §1.6-§1.8).
   - Parameter omissions (AUDIT §1.9): ensure all modules (bidirectional backward pass, alpha gates, heads) count toward < 100M.
   - Citation laundering and unverified claims (AUDIT §2).
   - Fair play, secret transformer check, transferability, safety, 12-week feasibility.
4. For every attack: state Severity (Fatal / Major / Minor), Evidence/Reasoning, and a Concrete Fix that preserves high ambition.

Write `/home/user/hip/agy-blueprint/work/critique_depth.md`.
Write your handoff to `/home/user/hip/agy-blueprint/.agents/teamwork/critic_depth/handoff.md`.
Message orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`) when complete.
