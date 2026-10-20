# Task Assignment: Phase A Proposal — Minimal Fastest-Cycle RSI

You are the Lead Designer for the `minimal` philosophy:
- Philosophy: Minimal fastest-cycle RSI
- Target file: `/home/user/hip/agy-blueprint/work/proposal_minimal_v1.md`
- Your working directory: `/home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/`
- Binding specification: `/home/user/hip/agy-blueprint/BRIEF.md` (read completely)
- Original request: `/home/user/hip/agy-blueprint/ORIGINAL_REQUEST.md`
- Verified research files in `/home/user/hip/asi-blueprint/research/`:
  `reason_baseline.md`, `rsi.md`, `reason_methods.md`, `io_split.md`, `tiny_sota.md`, `archs.md`, `externalize.md`, `compute.md`, `data.md`.

Concept:
The simplest proven stack (attention-free core + recurrent depth + tools + exact verifiers + RLVR + proposer/solver self-play), optimized for the shortest generation cycle on free compute and the cleanest measurement of improvement per generation. Every exotic module elsewhere must beat this at equal FLOPs and equal wall-clock per generation.

Mandatory Structure (all 12 sections in exact order, 3,000–5,000 words, English):
1. Thesis
2. The RSI Engine (L1-L5, cord-cut criterion, fuel, S-curve staircase & finding next verifier domain, generation compute/time)
3. Curve-Killer Defenses (mechanism + detector + rollback/kill rule for each)
4. RSI Metrics Dashboard
5. The Core as Learner (ASCII diagram, dataflow, structured output schema, parameter table < 100M, non-Transformer justification)
6. Fair-Play Rules (renderer firewall, direct input reading, RSI integrity)
7. The Boss Ladder (scoreboard vs reason_baseline.md, G rule margins, future rungs & probabilities)
8. Bootloader Recipe & Data (licensed data, no GPT-3.5/4 data, 6ND FLOP math vs free quota, T4/TPU notes)
9. Renderer Choice & Integration
10. Scaling & North Star (0.1B -> 1B -> 10B transfer, grant acquisition, drug discovery)
11. Stage-Gated Safety Mechanisms (§6 of BRIEF.md)
12. First 3 Experiments (weeks 1-4, runnable pseudocode for Exp 1 in weeks 1-2, risk register)

Integrity warning:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Every claim must be labeled [검증됨: url], [추정], or [가설].
Write the complete proposal to `/home/user/hip/agy-blueprint/work/proposal_minimal_v1.md`.
Write your handoff report to `/home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/handoff.md`.
When finished, send a completion message to orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`).


## 2026-10-19T09:08:20Z
[Message] timestamp=2026-10-19T09:08:20Z sender=9d100903-3938-4939-8cb4-21dca26a45b4 priority=MESSAGE_PRIORITY_HIGH content=You are the Lead Designer for the 'minimal' philosophy (Minimal fastest-cycle RSI) on the Aphasic Genius RSI Engine project.

Your assigned directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/
Your target output file: /home/user/hip/agy-blueprint/work/proposal_minimal_v1.md
Your handoff file: /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/handoff.md
Orchestrator Conversation ID to message on completion: 9d100903-3938-4939-8cb4-21dca26a45b4

Instructions:
1. Read /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/DISPATCH.md
2. Read /home/user/hip/agy-blueprint/BRIEF.md and /home/user/hip/agy-blueprint/ORIGINAL_REQUEST.md in full.
3. Read the research files in /home/user/hip/asi-blueprint/research/ (especially reason_baseline.md, rsi.md, reason_methods.md, archs.md, tiny_sota.md, externalize.md, compute.md, data.md).
4. Write a comprehensive, rigorous proposal (3,000–5,000 words, English) strictly following the 12 mandatory sections in exact order per BRIEF.md §4:
   1. Thesis
   2. The RSI engine
   3. Curve-killer defenses
   4. RSI metrics dashboard
   5. The core as the learner inside the loop (simplest proven stack: attention-free core + recurrent depth + tools + exact verifiers + RLVR + proposer/solver self-play, ASCII diagram, structured output schema, parameter table < 100M, non-Transformer justification)
   6. Fair-play rules
   7. The boss ladder (reasoning rows vs reason_baseline.md with G rule margins)
   8. Bootloader training recipe & data (licensed data, no GPT-3.5/4 data, explicit 6ND FLOP math vs free quota, T4/TPU notes)
   9. Renderer choice and integration
   10. Scaling & North Star
   11. Stage-gated safety mechanisms
   12. First 3 experiments (weeks 1–4; runnable pseudocode for Exp 1 in weeks 1–2; risk register)
5. HARD CONSTRAINTS: Total parameters < 100M with exact table; No Transformer / attention backbone; Zero money compute fit; Renderer firewall; Explicit cord-cut criterion; Label all claims [검증됨: url], [추정], [가설].
DO NOT CHEAT. All implementations must be genuine.
6. Write the complete proposal to /home/user/hip/agy-blueprint/work/proposal_minimal_v1.md.
7. Write your handoff report to /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_minimal/handoff.md.
8. Send a message to orchestrator (9d100903-3938-4939-8cb4-21dca26a45b4) with your completion report.
