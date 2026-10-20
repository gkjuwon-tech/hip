# Phase F Audit: Completeness (work/blueprint.md)

Done outside the Antigravity run after its quota ran out. Priority order.

## Must fix
1. **Distillation-ceiling claim overstated.** §1.2 says "a student that relies on a teacher can never surpass that teacher" and writes supp(student) ⊆ supp(teacher) as fact. arXiv 2504.13837 finds that distillation *can* add reasoning patterns while RLVR narrows the reasoning boundary. Reframe: imitation alone is bounded in practice; the escape comes from search plus exact verifiers, and expert iteration with MCTS (not plain RLVR) is used precisely because search expands coverage. Label the support claim [가설].
2. **Token-level MCTS with a mean of 18 simulations is too shallow** for 100-token programs; it degenerates to near-greedy decoding. Make the search action a **program step** (one line / one tactic / one SMT assertion), with token decoding inside a step done by the policy. This keeps 18–64 simulations meaningful.
3. **Gated rows must use the max reported GPT-3.5 score** (MATH 43.1, DROP 70.2, StrategyQA 70.3 → G 62.1 / 80.2 / 80.3), per BRIEF §3 and reason_baseline.md.
4. **Projected scores read like measurements.** Several sit 0.1–2 points above G (ARC 40.5 vs G 40.4) and carry "95% CI lower bound" columns. State plainly that these columns are target scores derived from G, not forecasts, and give an honest confidence grade per row. Knowledge-heavy rows (GPQA, MedQA) and code (HumanEval+, MBPP+, LiveCodeBench) are the highest-risk rows at < 100M.
5. **Citation fixes:** "Polynomial Upper Confidence Trees" → PUCT (Predictor + UCT). Stitch is the library-learning tool used in LILO (arXiv 2310.19791); cite it as such. Local files (`io_split.md`, `archs.md`) are not sources; cite the URLs they cite. Experiment 2 compares to "the 65% drop suffered by GPT-3.5/GPT-4", but GSM-Symbolic reports no GPT-3.5 row. TRC: free TPU quota, but VMs/storage are paid and no v6e or per-project size is verified.
6. **Dataset name:** "FineWeb-2 Korean Educational" does not exist; use the FineWeb-2 Korean split (ODC-By) filtered by our own STEM/math classifier.
7. **Naming of the cord-cut condition:** Condition 2 uses Δ_TE for what is the TE level. Write TE_g > TE_{g−1} > TE_{g−2}.

## Should fix
8. Cycle time should be computed for an all-T4 generation; TPU is used for Stage 0 and occasional large retrains, not inside every generation.
9. L5 is described as "locked until post-ASI"; restate as locked, research-only, under human governance with an air gap.
10. Experiment 3 expects a cord-cut in week 4. Keep the criterion, but mark the timing [가설]; failing it is information, not failure of the project.
11. Value-head pruning ">60% of branches" and the "8–10× MCTS speedup" are unmeasured; label [가설] and add both to Experiment 1's measurements.

## Verified OK
- No softmax attention over the sequence anywhere; renderer firewall (blindfold, core-only scoring, renderer-alone controls, swap test, core reads input) is complete.
- Reading firewall (reverse generation, 3-way paraphrase, counterfactual twins, distractor twins) is present.
- Cord-cut teacher (Qwen2.5-Math-7B-Instruct, AWQ on T4) runs on free compute, with a frozen 2,000-problem failure manifest. Random-reward control gate present.
- Experiment 1 is a plan, starts from the Stage-0 core, has a random-reward control, pass/kill criteria and compute.
- No-teacher track (Track 0-NT) present. No GPT-3.5/4 data. Frontier tin cans named correctly. Stage-gated safety including biosafety and the user's stop rule present.
