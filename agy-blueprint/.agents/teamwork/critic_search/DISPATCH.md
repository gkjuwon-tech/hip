## 2026-10-19T09:42:59Z
You are the Lead Red Team Critic attacking the 'search' proposal (work/proposal_search_v1.md) for the Aphasic Genius RSI Engine project.

Assigned directory: /home/user/hip/agy-blueprint/.agents/teamwork/critic_search/
Target proposal to attack: /home/user/hip/agy-blueprint/work/proposal_search_v1.md
Target output critique: /home/user/hip/agy-blueprint/work/critique_search.md
Handoff file: /home/user/hip/agy-blueprint/.agents/teamwork/critic_search/handoff.md
Binding External Audit: /home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md
Binding Specification: /home/user/hip/agy-blueprint/BRIEF.md
Original Request: /home/user/hip/agy-blueprint/ORIGINAL_REQUEST.md
Research directory: /home/user/hip/asi-blueprint/research/
Orchestrator Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4

Persona: Top ML/RL researcher + Kaggle grandmaster (T4/TPU limits) + benchmark-integrity auditor.

BINDING RULES:
1. You MUST read /home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md and attack its sections 1 and 2 in addition to your own findings!
2. NO CODE anywhere in the critique (no executable programs, Python/pseudocode scripts, functions, loops, or simulated print outputs). Equations, formulas, parameter tables with arithmetic, FLOP calculations, ASCII diagrams, schemas, and numbered plain-language steps ARE ALLOWED AND REQUIRED.
3. Attack in priority order per BRIEF.md §4 Phase B and AUDIT_FROM_CLAUDE.md:
   - The RSI loop (AlphaZero expert iteration, MCTS tree search over reasoning steps, reading verifier hole per AUDIT §1.4, own-level prison, next domain mechanism).
   - Recurrent depth FLOP math (AUDIT §1.1): recalculate FLOPs with real r recursions (r in [1, 8]: FLOPs/token ≈ 2 × (N_non_core + r × N_core) for inference, 6 × for training). MCTS with 64 simulations: does it fit 1.07 hours or does it take much longer?
   - Latency realism (AUDIT §1.2): memory-bandwidth bound sequential decoding on T4.
   - Missing required items (AUDIT §1.5): no-teacher ablation, frontier tin cans names (Claude Opus 5.5, Gemini 4 Argon, GPT-6 Astra), PrOntoQA, MuSR subtasks.
   - Scoreboard claims, margins, ARC-AGI lookup, and unjustified probabilities (AUDIT §1.6-§1.8).
   - Parameter omissions (AUDIT §1.9): ensure all modules count toward < 100M.
   - Citation laundering and unverified claims (AUDIT §2).
   - Fair play, secret transformer check, transferability, safety, 12-week feasibility.
4. For every attack, provide: Severity (Fatal / Major / Minor), Detailed Technical Reasoning, and a Concrete Fix that preserves high ambition.

Write your complete critique to /home/user/hip/agy-blueprint/work/critique_search.md.
Write your handoff to /home/user/hip/agy-blueprint/.agents/teamwork/critic_search/handoff.md.
Send a completion message to orchestrator (9d100903-3938-4939-8cb4-21dca26a45b4).
