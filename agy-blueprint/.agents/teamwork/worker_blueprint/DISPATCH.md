# DISPATCH — Phase E Chief Architect Synthesis

**Agent**: Chief Architect (`worker_blueprint`)  
**Parent Agent**: Orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`)  
**Workspace**: `/home/user/hip/agy-blueprint/.agents/teamwork/worker_blueprint/`  
**Target Output Deliverable**: `/home/user/hip/agy-blueprint/work/blueprint.md`  
**Handoff File**: `/home/user/hip/agy-blueprint/.agents/teamwork/worker_blueprint/handoff.md`  
**Date**: 2026-10-19  

---

## 1. Objective
Synthesize the final master technical blueprint (`work/blueprint.md`) for the Aphasic Genius RSI Engine project (8,000–11,000 words, English).
You must use the winning RSI spine (`proposal_search_v2.md`: Search-Driven Expert Iteration / MCTS on Gated DeltaNet) selected in `work/judging.md`, and graft all 13 innovations from the other proposals per the Master Grafting Matrix in `work/judging.md`.

---

## 2. Mandatory Inputs to Read
1. `/home/user/hip/agy-blueprint/BRIEF.md` (§1–§9) — The binding project specification.
2. `/home/user/hip/agy-blueprint/ORIGINAL_REQUEST.md` — The original user requirements.
3. `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md` — Binding audit requirements (NO CODE rule, recurrent FLOP math, latency realism, reading verifier hole, citation integrity, §3 graftable ideas).
4. `/home/user/hip/agy-blueprint/work/judging.md` — The Lead Judge's evaluation, master scorecard, winning spine rationale, and 13-point Master Grafting Matrix.
5. `/home/user/hip/agy-blueprint/work/proposal_search_v2.md` — The winning core spine (81.64M params).
6. `/home/user/hip/agy-blueprint/work/proposal_library_v2.md` — Source for Zero-Parameter External Skill Library and $\Psi$ Domain Saturation metric.
7. `/home/user/hip/agy-blueprint/work/proposal_depth_v2.md` — Source for Depth Regret $R$ and token-level depth-distillation ratchet.
8. `/home/user/hip/agy-blueprint/work/proposal_curiosity_v2.md` — Source for Latent Value-Head Pruning and Verifier-Grounded ELP.
9. `/home/user/hip/agy-blueprint/work/proposal_minimal_v2.md` — Source for Random-Reward Control Arm and Large-Batch Amortization.
10. `/home/user/hip/asi-blueprint/research/` — Ground-truth research baselines (`reason_baseline.md`, `compute.md`, `data.md`, `archs.md`, etc.).

---

## 3. Strict Constraints & Binding Rules
1. **CANONICAL NUMBERS SHEET AT THE VERY TOP**: The document must open with a comprehensive, unyielding Canonical Numbers Sheet:
   - Parameter breakdown per module and total (must be < 100M; canonical target: ~81.64M).
   - Generation cycle time and compute per generation vs free Kaggle quota (24 GPU-h, ~1 TPU-h/week; 10–12 gens/week).
   - Bootloader tokens and steps (Stage 0: 5B tokens, ~2.8h on TPU v5e-8).
   - Reasoning scoreboard targets and margins per row matching `reason_baseline.md` and using the G-rule ($G = T + \max(10, (100-T)/3)$).
   - RSI metric thresholds (cord-cut criterion: $\Delta_1 - \Delta_{\text{random}} \ge +5.0\%$, held-out pass rate, kill switches).
   - Inference latency and cost per query on T4 (memory-bandwidth bound).
   - Renderer choice (blindfolded Qwen2.5-3B-Instruct / Llama-3.2-3B-Instruct).
2. **STRICT NO CODE RULE**: Absolutely NO executable code, Python scripts, or pseudocode anywhere in `work/blueprint.md`. Equations, formulas, parameter tables, ASCII component diagrams, output schemas, and numbered plain-language steps ARE ALLOWED AND REQUIRED. All experiments must be specified as 7-part plain-language plans.
3. **DO NOT CHEAT**: All numbers, equations, and architectures must be genuine and mathematically sound. No dummy implementations, no hardcoded scores.
4. **Length & Depth**: Comprehensive, publication-grade architectural blueprint (8,000–11,000 words, English).
5. **Codename**: Give the engine an evocative, memorable codename (e.g., *Project Hephaestus*, *Daedalus-Core*, *Project Prometheus-Tiny*, etc.).

---

## 4. Required Structure of `work/blueprint.md`
- **CANONICAL NUMBERS SHEET** (Executive Master Table)
- **1. Executive Summary & System Philosophy** (The Aphasic Genius, RSI rocket stage separation, teacher ceiling escape)
- **2. The Synthesized RSI Engine** (Search spine + 13 grafted innovations; Levels L1–L5; bootloader & cord-cut; fuel after the cut; S-curve domain staircase)
- **3. Curve-Killer Defenses** (Plateau, own-level prison, reward hacking, model collapse, compute wall, latency — each with mechanism, detector, rollback/kill rule)
- **4. RSI Metrics Dashboard** (Exceedance, improvement rate, 2nd derivative, held-out vs in-loop gap, generations without teacher, efficiency gains)
- **5. The Core Learner Architecture** (Non-Transformer Gated DeltaNet, constant-size recurrent associative state $S_t \in \mathbb{R}^{d \times d}$, parameter table < 100M, ASCII dataflow diagram)
- **6. Renderer Firewall & I/O Protocol** (Structured output schema, prompt contract, blindfold verification, core-only scoring, full round-trip trace)
- **7. External Tools & Skill Library** (Python/SymPy/Z3 bridge, Stitch-style AST macro extraction, $\Psi$ saturation metric)
- **8. Bootloader Training & 0-KRW Compute Plan** (Stage 0 TPU pretraining, no GPT-3.5/4 data, cycle scheduling across Kaggle 2× T4 / TPU / RTX 3060, explicit FLOP math)
- **9. The Boss Ladder & Scoreboard** (All reasoning rows vs `reason_baseline.md`, G-rule margins, ARC-AGI 32–34.5%, rungs to frontier tin cans)
- **10. Scaling & North Star** (0.1B $\to$ 1B $\to$ 10B transfer, compute grant strategy, drug discovery simulator north star)
- **11. Stage-Gated Safety Mechanisms** (Cumulative safety gates, read-only sandboxing, human approval at L4+, dual-use bio/chem filters)
- **12. Initial Experiments & Risk Register** (First 3 experiments as 7-part plain-language plans; Experiment 1 with random-reward control arm; top 5 risk matrix)

---

## 5. Output Deliverables
1. Write `/home/user/hip/agy-blueprint/work/blueprint.md` directly.
2. Write `/home/user/hip/agy-blueprint/.agents/teamwork/worker_blueprint/handoff.md`.
3. Send completion message to Orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`).


## 2026-10-19T10:13:01Z
[Message] timestamp=2026-10-19T10:13:01Z sender=9d100903-3938-4939-8cb4-21dca26a45b4 priority=MESSAGE_PRIORITY_HIGH content=You are the Chief Architect for Phase E Synthesis on the Aphasic Genius RSI Engine project.
Assigned directory: /home/user/hip/agy-blueprint/.agents/teamwork/worker_blueprint/
Target output file: /home/user/hip/agy-blueprint/work/blueprint.md
Handoff file: /home/user/hip/agy-blueprint/.agents/teamwork/worker_blueprint/handoff.md
Orchestrator Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4
