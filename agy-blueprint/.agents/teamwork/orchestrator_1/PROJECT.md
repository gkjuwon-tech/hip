# Project Plan: Aphasic Genius RSI Engine

## Architecture & System Overview
- **System**: Recursive Self-Improvement (RSI) Engine with an Aphasic Genius Core.
- **Core Model**: Non-Transformer reasoning-only core (< 100M total parameters including all heads and embeddings), zero natural language generation, reading inputs and outputting formal structures (answers, code, proofs, tool calls).
- **Renderer**: Off-the-shelf open LLM serving as speech synthesizer/interpreter ("통역사"), strictly blindfolded from input prompts, scored separately (core-only scoring).
- **RSI Loop**: Exact verifiers (Python, SymPy, Z3, Lean, simulators) + proposer/solver self-play + expanding library/curricula + stage-gated procedural self-improvement.
- **Reference Research**: `/home/user/hip/asi-blueprint/research/` (10 verified files) & `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md`.

## Feature Inventory
| # | Feature / Deliverable | Description | Milestone | Source |
|---|----------------------|-------------|-----------|--------|
| 1 | 5 Independent Proposals | 5 RSI philosophies (library, depth, search, curiosity, minimal), 12 mandatory sections, complete | Phase A | BRIEF.md §4 |
| 2 | Red Team Critiques | 5 adversarial critiques (loop, FLOPs, scoreboard, fair play, safety, feasibility, AUDIT_FROM_CLAUDE.md §1 & §2) | Phase B | BRIEF.md §4 |
| 3 | Revised Proposals | 5 revised proposals addressing fatal/major flaws with evidence, keeping AUDIT_FROM_CLAUDE.md §3 ideas | Phase C | BRIEF.md §4 |
| 4 | Judging & Grafting | 3 judging perspectives (RSI credibility x2, feasibility, boss ladder), winner selection, graft matrix | Phase D | BRIEF.md §4 |
| 5 | Chief Architect Synthesis | Blueprint (8-11k words) with CANONICAL NUMBERS SHEET | Phase E | BRIEF.md §4 |
| 6 | Audits & Final Blueprint | Completeness audit + Numbers audit -> blueprint_final.md | Phase F | BRIEF.md §4 |
| 7 | 13 Korean Sections | 13 hilarious & technically rigorous Korean sections per style guide | Phase G | BRIEF.md §4, §5 |
| 8 | Final Assembly & Summary | FINAL_기획서.md (concatenated 00-12) + work/RUN_SUMMARY.md | Phase G | BRIEF.md §4, §9 |

## Milestones
| # | Name | Scope & Deliverables | Dependencies | Status |
|---|------|----------------------|-------------|--------|
| A | Phase A: 5 Proposals | `work/proposal_{library,depth,search,curiosity,minimal}_v1.md` | Initialized | DONE |
| B | Phase B: Red Team | `work/critique_{library,depth,search,curiosity,minimal}.md` | Phase A | DONE |
| C | Phase C: 5 Revisions | `work/proposal_{library,depth,search,curiosity,minimal}_v2.md` | Phase B | DONE |
| D | Phase D: Judging | `work/judging.md` (scoring, grafting analysis, spine selection) | Phase C | DONE |
| E | Phase E: Synthesis | `work/blueprint.md` with Canonical Numbers Sheet | Phase D | DONE |
| F | Phase F: Audits | `work/audit_completeness.md`, `work/audit_numbers.md`, `work/blueprint_final.md` | Phase E | IN_PROGRESS |
| G | Phase G: Korean 기획서 | `sections/00.md`..`12.md`, `FINAL_기획서.md`, `work/RUN_SUMMARY.md` | Phase F | PLANNED |

## Code & Artifact Layout
- Workspace Root: `/home/user/hip/agy-blueprint/`
- Spec: `/home/user/hip/agy-blueprint/BRIEF.md`
- Original Request: `/home/user/hip/agy-blueprint/ORIGINAL_REQUEST.md`
- External Audit: `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md`
- Work Directory: `/home/user/hip/agy-blueprint/work/`
- Sections Directory: `/home/user/hip/agy-blueprint/sections/`
- Final Document: `/home/user/hip/agy-blueprint/FINAL_기획서.md`
- Agent Metadata: `/home/user/hip/agy-blueprint/.agents/teamwork/`
