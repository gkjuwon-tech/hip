# Run Summary

- **Codename:** PROJECT HEPHAESTUS.
- **How it was produced:** Phases A–E (5 proposals, red team, revisions, judging, synthesis) were run by an Antigravity Gemini 3.8 Flash team, with an external Claude audit injected before Phase B. The Antigravity account hit its Gemini quota at the start of Phase F. Claude then did Phase F (`audit_numbers.md`, `audit_completeness.md`), wrote `blueprint_final.md`, and wrote the 13 Korean sections directly.
- **Winning RSI design:** expert-iteration MCTS (search spine). Grafts:
  - depth regret and depth-conditioned recursion (depth),
  - the zero-parameter Stitch skill library and the Ψ saturation trigger (library),
  - the random-reward control gate (minimal),
  - verifier-grounded learning progress and value-head pruning (curiosity).
  - Reason it won: highest RSI credibility and a feasible generation cycle.
- **Parameters:** 81,648,961 (< 100M; corrected from 81,694,209 by fixing the grid-adapter count).
- **One RSI generation:** ≈1.1–1.3 h on 2× T4, giving ≈10–12 generations/week. Stage 0 bootloader: 6.44e18 FLOPs, 3.8–5.7 h on TPU v5e-8.
- **Expected generation per GPT-3.5 row:** see sections/09.md. Targets are derived from G and are not forecasts. Confidence:
  - high: Game of 24, Blocksworld, FOLIO, ProofWriter, PrOntoQA, SVAMP
  - low: reading-heavy (LogiQA, ReClor, MuSR), code generation, and knowledge-heavy Tier B rows
- **Top risks:**
  1. Comprehension wall for a < 100M reader.
  2. Sandbox verification throughput on 4 vCPUs.
  3. T4 Triton support.
  4. Reward hacking through verifier edge cases.
  5. Code/knowledge rows missing the margin (handled by win-or-extend).
- **Brief requirements not fully met:**
  - Autonomous discovery of new verifier domains is only partial: verifiers are human-installed, and the model searches for reductions between domains.
  - Projected scores are targets, not forecasts.
  - Korean sections are shorter than the 2,500–5,000-word guide, by choice, to finish the run directly.
