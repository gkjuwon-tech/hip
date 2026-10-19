# BRIEFING — 2026-10-19T09:10:00Z

## Mission
Design a rigorous, comprehensive, technically airtight Phase A proposal for the Library-Growing RSI (neuro-symbolic compiler) philosophy per BRIEF.md §4.

## 🔒 My Identity
- Archetype: implementer, qa, specialist
- Roles: implementer, qa, specialist
- Working directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_library/
- Original parent: 9d100903-3938-4939-8cb4-21dca26a45b4
- Milestone: Phase A: Independent Designs

## 🔒 Key Constraints
- Philosophy: Library-growing RSI (neuro-symbolic compiler)
- Target file: /home/user/hip/agy-blueprint/work/proposal_library_v1.md
- Total trainable parameters < 100,000,000 (every weight counted: reader, embeddings, solver, proposer, verifier/judge heads)
- Non-Transformer backbone: zero softmax/Hopfield sequence attention (Gated DeltaNet / RWKV-7 / BiSSM / Mamba-2)
- Zero money: Kaggle 2x T4 (30h/wk, fp16 only, Triton 3.2), TPU v5e-8 (20h/wk, JAX/bf16), laptop RTX 3060 (6GB)
- Strict renderer firewall: core reads input, outputs typed IR, blindfolded renderer, scoring on formal outputs only
- Explicit, measurable cord-cut criterion & fuel after the cut
- S-curve staircase of verifier domains & concrete domain expansion mechanism
- Exact 6ND FLOP math vs quotas for bootloader and 1 RSI generation
- No GPT-3.5/GPT-4 generated training data (TinyGSM, MetaMathQA banned)
- Every claim labeled [검증됨: url], [추정], [가설]
- 12 mandatory sections in exact order per BRIEF.md §4 (3,000–5,000 words, English)

## Current Parent
- Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4
- Updated: not yet

## Task Summary
- **What to build**: Phase A Proposal for Library-Growing RSI (`work/proposal_library_v1.md`)
- **Success criteria**: 12 sections strictly ordered, 3,000–5,000 words, parameter arithmetic summing to < 100M, complete curve-killer defenses, verifiable FLOP math, runnable pseudocode for Exp 1.
- **Interface contracts**: BRIEF.md §4, §6; reason_baseline.md; DISPATCH.md
- **Code layout**: Output at `/home/user/hip/agy-blueprint/work/proposal_library_v1.md`, handoff at `.agents/teamwork/worker_proposal_library/handoff.md`

## Key Decisions Made
- Architecture: Gated DeltaNet / RWKV-7 recurrent core + bidirectional SSM reader + typed DSL emitter + Stitch-style library compression + 0-parameter exact executors (Python, SymPy, Z3, Lean).
- Wake-Sleep cycle adapted from DreamCoder/LILO into a high-throughput, non-attention, parameter-capped RSI engine.

## Artifact Index
- /home/user/hip/agy-blueprint/work/proposal_library_v1.md — Primary proposal document
- /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_library/handoff.md — 5-component handoff report
- /home/user/hip/agy-blueprint/.agents/teamwork/worker_proposal_library/progress.md — Liveness heartbeat

## Change Tracker
- **Files modified**: None yet
- **Build status**: N/A
- **Pending issues**: Drafting proposal_library_v1.md

## Quality Status
- **Build/test result**: N/A
- **Lint status**: 0 violations
- **Tests added/modified**: N/A

## Loaded Skills
- None required from external Antigravity skill paths.
