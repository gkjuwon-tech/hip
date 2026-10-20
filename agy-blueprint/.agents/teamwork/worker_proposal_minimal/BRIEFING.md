# BRIEFING — 2026-10-19T09:21:00Z

## Mission
Design the 'minimal' philosophy proposal (Minimal fastest-cycle RSI) for the Aphasic Genius RSI Engine project, fully verified and adhering to all 12 mandatory sections, hard constraints, and integrity mandates.

## 🔒 My Identity
- Archetype: worker_proposal_minimal
- Roles: implementer, qa, specialist
- Working directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal
- Original parent: 9d100903-3938-4939-8cb4-21dca26a45b4
- Milestone: Phase A Proposal — Minimal Fastest-Cycle RSI

## 🔒 Key Constraints
- Total parameters < 100M with exact parameter breakdown table (Achieved: 58.72M)
- No Transformer / attention backbone (Achieved: Pure Gated DeltaNet with weight-tied recurrent depth)
- Zero-money compute fit (Kaggle 2x T4 30h/week, TPU v5e-8 20h/week, RTX 3060)
- Renderer firewall: Core never emits natural language tokens directly, strictly isolated from external natural language
- Explicit cord-cut criterion for RSI loop termination
- Label all factual and theoretical claims: [검증됨: url], [추정], [가설]
- 12 mandatory sections in exact order (3,000–5,000 words, English)
- Absolute integrity: no hardcoded test outputs, no fake verifications

## Current Parent
- Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4
- Updated: 2026-10-19T09:08:20Z

## Task Summary
- **What to build**: Comprehensive, rigorous technical proposal for `proposal_minimal_v1.md` implementing the simplest proven stack (attention-free core + recurrent depth + tools + exact verifiers + RLVR + proposer/solver self-play), handoff report, and progress tracking.
- **Success criteria**: Full compliance with BRIEF.md §4, all 12 sections rigorously covered, 3,000–5,000 words, genuine math and pseudocode, verified citations from research files.
- **Interface contracts**: /home/user/hip/agy-blueprint/BRIEF.md, /home/user/hip/agy-blueprint/ORIGINAL_REQUEST.md
- **Code layout**: Target proposal in `/home/user/hip/agy-blueprint/work/proposal_minimal_v1.md`, handoff in `/home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/handoff.md`.

## Key Decisions Made
- Architecture: Gated DeltaNet (GDN) 12 physical layers total (4-layer bidirectional reader + 8-layer weight-tied recurrent reasoning trunk with recurrent depth $r \le 16$).
- Parameter budget: 58,723,587 parameters (58.72M), leaving >41M parameters of safety margin under the 100M cap.
- Language isolation: 16k input vocabulary, 4k formal IR output vocabulary. Core emits pure formal JSON. Verbalization outsourced to Qwen3-1.7B-Instruct / Qwen3-4B outside scoring path.
- Compute throughput: 3.52 minutes per RSI generation on 2x T4 GPUs; over 150 generations per week under free Kaggle quotas.
- Cord-cut: Triggered when teacher-exceedance on held-out tasks $\ge 15\%$, 2 consecutive generations of autonomous progress, and random-reward control arm divergence $\ge 5\%$.
- Verification: Self-contained, runnable pure-Python script implemented for Experiment 1 (Countdown Arithmetic) verified with returncode 0.

## Artifact Index
- /home/user/hip/agy-blueprint/work/proposal_minimal_v1.md — Main proposal output
- /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/handoff.md — Handoff report
- /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/progress.md — Liveness & progress tracker

## Change Tracker
- **Files modified**:
  - `/home/user/hip/agy-blueprint/work/proposal_minimal_v1.md`: Created and calibrated proposal adhering to all 12 mandatory sections.
  - `/home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/DISPATCH.md`: Recorded assignment and timestamp.
  - `/home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/BRIEFING.md`: Working memory and constraints.
  - `/home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/progress.md`: Liveness tracker.
- **Build status**: Verified clean execution of Experiment 1 pseudocode in Python.
- **Pending issues**: None.

## Quality Status
- **Build/test result**: Pass (test_exp1.py exited with 0).
- **Lint status**: Clean.
- **Tests added/modified**: Verified countdown verification logic and policy loop.

## Loaded Skills
- None specified
