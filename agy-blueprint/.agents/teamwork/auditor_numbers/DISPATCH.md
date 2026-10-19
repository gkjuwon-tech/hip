# DISPATCH — Phase F Numbers Audit

**Agent**: Numbers Auditor (`auditor_numbers`)  
**Parent Agent**: Orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`)  
**Workspace**: `/home/user/hip/agy-blueprint/.agents/teamwork/auditor_numbers/`  
**Target Deliverable**: `/home/user/hip/agy-blueprint/work/audit_numbers.md`  
**Handoff File**: `/home/user/hip/agy-blueprint/.agents/teamwork/auditor_numbers/handoff.md`  
**Date**: 2026-10-19  

---

## 1. Objective
Forensically audit, verify, and recompute EVERY number, parameter count, FLOP calculation, hardware latency, and benchmark margin in `/home/user/hip/agy-blueprint/work/blueprint.md`.

---

## 2. Mandatory Inputs to Read
1. `/home/user/hip/agy-blueprint/work/blueprint.md` — The target technical blueprint under audit.
2. `/home/user/hip/asi-blueprint/research/reason_baseline.md` — Authoritative benchmark baseline scores.
3. `/home/user/hip/asi-blueprint/research/compute.md` — Kaggle quotas and hardware specs.
4. `/home/user/hip/asi-blueprint/research/archs.md` — Parameter formulas for Gated DeltaNet, embeddings, and non-Transformer layers.
5. `/home/user/hip/agy-blueprint/AUDIT_FROM_CLAUDE.md` — Binding audit directives (§1 math fixes).

---

## 3. Forensic Calculation Protocol
You must independently compute:
1. **Parameter Breakdown (< 100M)**:
   - For every module in Table 0.1 and Table 5.3: recompute from first principles using tensor dimensions ($V_{\text{in}}=16,384, V_{\text{formal}}=2,048, d=640, d_{\text{ff}}=1600, r\in[1,8]$).
   - Verify the exact sum against the stated 81,694,209. Confirm headroom below 100,000,000.
2. **Recurrent FLOPs & Active Parameters**:
   - Verify $N_{\text{eff}} = N_{\text{non\_core}} + r \times N_{\text{core}}$ for $r \in [1, 8]$ and specifically at $r=4$ ($214.62\text{M}$).
   - Verify inference FLOPs/tok ($2 N_{\text{eff}} = 4.292 \times 10^8$) and training FLOPs/tok ($6 N_{\text{eff}} = 1.288 \times 10^9$).
3. **Rollout, Verification, and Training Wall-Clock on Kaggle Hardware**:
   - Recompute rollout compute for 5,000 tasks $\times$ 18 mean Snell sims $\times$ 100 tokens ($3.863 \times 10^{15}$ FLOPs).
   - Recompute memory-bandwidth bound latency on NVIDIA T4 ($300\text{ GB/s}$ peak, $230\text{ GB/s}$ sustained, streaming $429.2\text{ MB}$ weights).
   - Recompute TPU v5e-8 training time ($1.288 \times 10^{16}\text{ FLOPs} / 4.728 \times 10^{14}\text{ FLOP/s} = 27.2\text{ s}$) and T4 fallback ($396.3\text{ s}$).
   - Verify cycle time ($\approx 0.90\text{--}1.00\text{ h}$) and weekly quota budget (24 GPU-h, ~1 TPU-h vs 30 GPU-h / 20 TPU-h quotas).
4. **Stage 0 Bootloader Compute**:
   - 5.0B tokens $\times 1.288 \times 10^9\text{ FLOPs/token} = 6.44 \times 10^{18}\text{ FLOPs}$. Verify $3.78\text{ h}$ on TPU v5e-8.
5. **Scoreboard Auditing**:
   - For every row in Table 0.3: check GPT-3.5 score $T$ against `reason_baseline.md`.
   - Check target margin $G = T + \max(10, (100-T)/3)$.
   - Verify binomial 95% confidence intervals $p \pm 1.96 \sqrt{\frac{p(1-p)}{N}}$.

---

## 4. Output Deliverable
Write `/home/user/hip/agy-blueprint/work/audit_numbers.md`.
Include the comprehensive forensic master table:
`| Entity / Metric | Formula & Dimensions | Stated Value | Recomputed Exact | Verdict (MATCH / DISCREPANCY) | Correction / Note |`
Provide an overall forensic certification and clear remediation recommendations for `work/blueprint_final.md`.

Write your handoff report to `/home/user/hip/agy-blueprint/.agents/teamwork/auditor_numbers/handoff.md` and send a message to Orchestrator (`9d100903-3938-4939-8cb4-21dca26a45b4`).


## 2026-10-19T10:22:59Z
You are the Numbers Auditor for Phase F Audits on the Aphasic Genius RSI Engine project.

Assigned directory: /home/user/hip/agy-blueprint/.agents/teamwork/auditor_numbers/
Target output deliverable: /home/user/hip/agy-blueprint/work/audit_numbers.md
Handoff file: /home/user/hip/agy-blueprint/.agents/teamwork/auditor_numbers/handoff.md
Orchestrator Conversation ID: 9d100903-3938-4939-8cb4-21dca26a45b4
