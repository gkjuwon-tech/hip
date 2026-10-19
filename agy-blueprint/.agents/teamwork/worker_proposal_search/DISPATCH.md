# Task Assignment: Phase A Proposal — Expert-Iteration RSI (AlphaZero-Style)

You are the Lead Designer for the `search` philosophy:
- Philosophy: Expert-iteration RSI (AlphaZero-style)
- Target file: `/home/user/hip/agy-blueprint/work/proposal_search_v1.md`
- Your working directory: `/home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_search/`
- Binding specification: `/home/user/hip/agy-blueprint/BRIEF.md` (read completely)
- Original request: `/home/user/hip/agy-blueprint/ORIGINAL_REQUEST.md`
- Verified research files in `/home/user/hip/asi-blueprint/research/`:
  `reason_baseline.md`, `rsi.md`, `reason_methods.md`, `io_split.md`, `tiny_sota.md`, `archs.md`, `externalize.md`, `compute.md`, `data.md`.

Concept:
Tiny policy+value network driving MCTS over reasoning steps with exact verifiers as reward. Search improves policy, policy improves search. Absolute-Zero-style proposer-vs-solver self-play keeps curriculum at edge of competence.

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
Write the complete proposal to `/home/user/hip/agy-blueprint/work/proposal_search_v1.md`.
Write your handoff report to `/home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_search/handoff.md`.
When finished, send a completion message to orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`).

## 2026-10-19T09:08:20Z
Received dispatch message from 9d100903-3938-4939-8cb4-21dca26a45b4: Lead Designer for 'search' philosophy on Aphasic Genius RSI Engine project. Target file: /home/user/hip/agy-blueprint/work/proposal_search_v1.md. Handoff: /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_search/handoff.md.
