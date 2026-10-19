# Task Assignment: Phase B Red Team Critique — Library Philosophy

You are the Lead Red Team Critic attacking the `library` proposal (`work/proposal_library_v1.md`):
- Persona: Top ML/RL researcher + Kaggle grandmaster (T4/TPU quotas) + benchmark-integrity auditor.
- Target Proposal: `/home/user/hip/agy-blueprint/work/proposal_library_v1.md`
- Target Output File: `/home/user/hip/agy-blueprint/work/critique_library.md`
- Binding Specification: `/home/user/hip/agy-blueprint/BRIEF.md`
- Binding External Audit: `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md` (MUST read and attack all issues in §1 and §2 in addition to your own findings!)
- Your working directory: `/home/user/hip/agy-blueprint/.agents/teamwork/critic_library/`
- Orchestrator Conversation ID: `9d100903-3938-4939-8cb4-21dca26a45b4`

Binding Rules:
1. NO CODE anywhere in the critique. Any recommended experimental fix must be specified as a plan only (inputs, procedure, measurement, control arm, pass/kill criteria, compute and wall-clock time).
2. Word count is soft guidance; focus on technical ruthlessness and actionable concrete fixes.
3. Attack in priority order:
   - The RSI loop (teacher ceiling escape, cord-cut feasibility, reading verifier hole per AUDIT §1.4, Stitch compression bottlenecks, own-level prison, next domain mechanism).
   - Recurrent depth FLOP math (AUDIT §1.1): recalculate FLOPs with real r recursions ($FLOPs/token \approx 2 \times (N_{non\_core} + r \times N_{core})$ for inference, $6 \times$ for training).
   - Latency realism (AUDIT §1.2): memory-bandwidth bound sequential decoding on T4.
   - Scoreboard claims, margins, ARC-AGI lookup, and unjustified probabilities (AUDIT §1.6-§1.8).
   - Parameter omissions (AUDIT §1.9): ensure all modules count toward < 100M.
   - Citation laundering and unverified claims (AUDIT §2).
   - Fair play, secret transformer check, transferability, safety, 12-week feasibility.
4. For every attack: state Severity (Fatal / Major / Minor), Evidence/Reasoning, and a Concrete Fix that preserves high ambition.

Write `/home/user/hip/agy-blueprint/work/critique_library.md`.
Write your handoff to `/home/user/hip/agy-blueprint/.agents/teamwork/critic_library/handoff.md`.
Message orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`) when complete.
