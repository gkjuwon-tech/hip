# io_split: the "brain vs mouth" split (aphasic-genius pivot)

Scope: is understanding cheaper in parameters than generating? How big must the reader/formalizer be? Which open renderer can verbalize the core's formal output on T4/3060? How do we prove the renderer does no reasoning? Labels: [VERIFIED: url], [ESTIMATE], [HYPOTHESIS].

## 0. Bottom line

1. **Understanding is cheap.** Per parameter and per FLOP, discriminative readers beat generative LMs. ELECTRA-Small (14M) scores 79.9 on GLUE dev, against 78.8 for GPT-1 (117M), which used 29x the training compute [VERIFIED: https://arxiv.org/abs/2003.10555]. DeBERTa-v3-xsmall has a 22M backbone and scores MNLI 88.1/88.3 [VERIFIED: https://huggingface.co/microsoft/deberta-v3-xsmall]. On SuperGLUE, few-shot GPT-3 175B (71.8) only roughly ties a fine-tuned 335M BERT-Large (69.0) [VERIFIED: https://arxiv.org/abs/2005.14165]. **Reading is not where the 100M budget goes.**
2. **The aphasic core still generates, but it generates formal language, not English.** The pivot removes the cost of fluency: open vocabulary, style, and recalling world knowledge. It does not remove the cost of formalization. A 220M T5-base reaches only 32.7% equivalence on NL→FOL [VERIFIED: https://arxiv.org/html/2509.22338v2]. On Spider it reaches 59.4 EM, or 66.6 with grammar-constrained decoding [VERIFIED: https://github.com/ElementAI/picard]. **The real wall is reading → formal IR, not rendering.**
3. **The mouth is a free commodity.** Apache-2.0 renderers from 0.6B to 4B exist: Qwen3-0.6B/1.7B/4B and SmolLM3-3B. The risk is not that the renderer is too weak. The risk is that it is **too strong**: it can add claims, fix wrong answers, or re-solve the task. The protocol in §4 is built to catch that.
4. **Precedent exists, and it carries a warning.** At IMO 2024, AlphaProof/AlphaGeometry needed experts to translate the problems into Lean/DSL *and to translate the proofs back*. Compute took 2–3 days [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/], [VERIFIED: https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad]. The hardest step was done by humans, outside the scored loop. Our rule (4), "the core reads its own input", is stricter. That buys credibility, but it is also the **single biggest risk** to the milestone.

## 1. Evidence: understanding is cheaper than generating

### 1.1 Small encoders vs generative LMs

| Model | Params | Type | Score | Source |
|---|---|---|---|---|
| ELECTRA-Small | 14M | replaced-token detection | GLUE dev **79.9**, test 78.0 | [VERIFIED: https://arxiv.org/abs/2003.10555] |
| ELECTRA-Small, 6.25% trained | 14M | same | GLUE dev **74.1** after **6h on 1 V100** | same |
| BERT-Small (same compute) | 14M | MLM | GLUE dev 75.1 | same |
| GPT-1 | 117M | causal LM | GLUE dev 78.8; 4.0e19 FLOPs | same |
| BERT-Base / ELECTRA-Base | 110M | MLM / RTD | GLUE dev 82.2 / 85.1 | same |
| TinyBERT / MobileBERT | 14.5M / 25.3M | distilled | GLUE test 80.6 / 81.0 | same |
| MiniLM (distilled) | ~50% of teacher | self-attention distillation | keeps >99% of teacher accuracy on SQuAD2.0 and GLUE | [VERIFIED: https://arxiv.org/abs/2002.10957] |
| DeBERTa-v3-xsmall | 22M backbone + 48M embeddings | RTD + GDES | MNLI **88.1/88.3**, SQuAD2.0 84.8 F1 | [VERIFIED: https://huggingface.co/microsoft/deberta-v3-xsmall] |
| BERT-Large, fine-tuned | 335M | MLM | SuperGLUE **69.0**, BoolQ 77.4 | [VERIFIED: https://arxiv.org/abs/2005.14165] |
| GPT-3, few-shot | 175B | causal LM | SuperGLUE **71.8**, BoolQ 76.4, WiC 49.4 (chance) | same |
| GPT-3, few-shot reading | 175B | causal LM | **DROP 36.5 F1**, RACE-h 46.8; fine-tuned SOTA 89.1 / 90.0 | same |

What the table shows:
- **About 500x fewer parameters for the same NLU score.** The GPT-3 authors themselves conjecture that "a large bidirectional model would be stronger at fine-tuning" [VERIFIED: same]. Caveat: fine-tuned vs few-shot is not apples to apples. Our core is trained on its tasks, so the fine-tuned regime is the relevant one [ESTIMATE].
- **Discriminative objectives are about 30x more compute-efficient at small scale.** A 74.1 GLUE score for 6 GPU-hours fits inside one Kaggle session.
- **Embeddings eat small models.** DeBERTa-v3-xsmall spends 48M of its ~70M parameters on a 128K vocabulary. Gemma 3 270M spends 170M on embeddings [VERIFIED: https://developers.googleblog.com/en/introducing-gemma-3-270m/]. An aphasic core needs no large *output* vocabulary. A small input vocabulary or byte reader, plus a formal output vocabulary of a few thousand tokens, frees about 40–60% of the 100M budget for reasoning [ESTIMATE].

### 1.2 Bidirectional SSM readers (allowed under the no-Transformer rule)

- **Caduceus** (BiMamba + reverse-complement equivariance, ICML 2024). At **1.9M params** it beats attention models of 117M–500M on 8 of 18 Nucleotide-Transformer tasks. On variant-effect prediction more than 100k bp from the TSS, it beats Enformer (252M). The paper's claim is "exceeds the performance of 10x larger models" [VERIFIED: https://arxiv.org/abs/2403.03234]. The lever is bidirectionality plus a symmetry prior, not scale. It still loses splice-site tasks to NT-v2 (0.940 vs 0.983).
- Implication [HYPOTHESIS]: the reader should be a **bidirectional SSM/RNN encoder**, BiMamba-style or GatedDeltaNet run in both directions. It reuses ALEMBIC-0's "bidirectional judge tower", so one encoder serves as both reader and verifier. T4 caveat: Triton ≥3.3 dropped sm_75. Pin Triton 3.2 or use a pure-PyTorch chunked scan [VERIFIED in archs.md].

### 1.3 Understanding ≠ reasoning

GLUE-level reading saturates at 14–70M parameters. GSM8K, DROP and BBH need multi-step computation over what was read. GPT-3's DROP 36.5 is a reasoning failure, not a reading failure. So the split has three parts, not two:

**reader (cheap) → reasoner (where the budget goes: recurrence r, search, tools) → formal emitter (grammar-constrained, small vocabulary).**

## 2. Neuro-symbolic interface: how big must the formalizer be?

| System | Formalizer | Engine | Result | Source |
|---|---|---|---|---|
| PAL | Codex | Python | GSM8K **+15 abs** over PaLM-540B CoT | [VERIFIED: https://arxiv.org/abs/2211.10435] |
| SatLM | code-davinci (GPT-3.5-class) | Z3 | GSM-Sys **69.4** vs 46.5 CoT / 43.4 PAL; full GSM8K 84.8 with self-consistency | [VERIFIED: https://arxiv.org/abs/2305.09656] |
| Logic-LM | GPT-3.5/4 | Prover9, Z3, CSP, Pyke | **+39.2%** over standard prompting, +18.4% over CoT (5 datasets) | [VERIFIED: https://arxiv.org/abs/2305.12295] |
| LINC | StarCoder+ **15.5B** | Prover9 | ProofWriter: **+38 abs** over GPT-3.5 CoT, +10 over GPT-4 CoT | [VERIFIED: https://arxiv.org/abs/2310.15164] |
| NL→FOL fine-tune | **T5-base 220M** / T5-3B / Flan-T5-XXL | equivalence check | Equiv **32.7 / 36.2 / 38.8**; with predicates given, 50.2 / 51.5 / 54.8. LLaMA3.1-8B: 32.0. **T5-base ≈ LLaMA-8B** | [VERIFIED: https://arxiv.org/html/2509.22338v2] |
| Text-to-SQL (Spider dev) | T5-Base / Large / 3B | SQL | EM 59.4 / 65.3 / 71.5; +PICARD **66.6 / 69.1 / 75.5** | [VERIFIED: https://github.com/ElementAI/picard] |
| TinyGSM | 125M program generator | Python (+1.3B verifier) | GSM8K **63.1** alone, 76.1 with the verifier | [VERIFIED in tiny_sota.md, arXiv 2312.09241] |
| phi-1-small | 350M | code | HumanEval 45% | [VERIFIED: https://arxiv.org/abs/2306.11644] |
| AlphaGeometry | LM proposes constructions | DD+AR symbolic engine | IMO geometry **25/30** (gold medalist average 25.9); 100M synthetic examples | [VERIFIED: https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/] |

What the table shows:
- **Encoder-decoder formalizers punch about 30x above their weight.** T5-base 220M roughly matches LLaMA-8B on NL→FOL, and the paper credits explicit input encoding. So: **bidirectional encoder + small constrained decoder**, not decoder-only [HYPOTHESIS, consistent with the data].
- **Grammar-constrained decoding is worth about one model size, at 0 parameters.** T5-Base+PICARD (66.6) beats unconstrained T5-Large (65.3). Keep ALEMBIC-0's typed IR + grammar mask.
- **Formalization quality is the ceiling.** Every LLM+solver win above uses a formalizer of ≥15B, with one exception: TinyGSM at 125M. But TinyGSM's 12.3M problems were **GPT-3.5-generated**, which our rules ban. The replacement must come from **0-parameter generators**. SymPy/Z3/Lean synthesize the problems, which are then rendered to NL by templates, and by the open renderer *offline, at training time only*. The core learns the inverse map, NL→program [HYPOTHESIS: the AlphaGeometry recipe applied to NL]. Renderer-written training data is allowed because the renderer is a <5B open model, not GPT-3.5/4. It must be disclosed and must never see benchmark items.
- **Projection [ESTIMATE].** A 60–80M bidirectional reader + constrained emitter, with execution feedback and self-consistency (N=16–64), plausibly reaches GSM8K 60–75%. That beats 57.1 (the GPT-3.5 figure in the GPT-4 report) but does **not** clear the 78.1 max-snapshot figure by a wide margin. Clearing it widely needs search plus a verifier, i.e. test-time compute, not more parameters.
- **Wall.** AlphaProof failed both IMO-2024 combinatorics problems [VERIFIED in externalize.md]. Informal, combinatorial and common-sense problems resist formal IR. Two routes around it: (a) a "neural step" escape hatch in the IR, a TRM-style recurrent latent step that the core executes itself; (b) honest per-benchmark reporting of formalization coverage.

## 3. Renderer side

### 3.1 Candidates (outside the 100M, never in scoring)

| Model | Params | License | Context | Fit [ESTIMATE] | Notes | Source |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | Apache-2.0 | 32K | 3060 fp16 ≈1.3GB | MATH-500 77.6 (thinking): **strong reasoner, so blindfold risk is high** | [VERIFIED: https://arxiv.org/html/2505.09388] |
| Qwen3-1.7B | 1.7B (1.4B non-embedding) | Apache-2.0 | 32K | 3060 fp16 ≈3.4GB | thinking is on by default; set `enable_thinking=False` | [VERIFIED: https://huggingface.co/Qwen/Qwen3-1.7B] |
| Qwen3-4B | 4.0B (3.6B non-embedding) | Apache-2.0 | 32K (131K YaRN) | 3060 4-bit ≈2.5–3GB; T4 fp16 ≈8GB | best fluency in budget | [VERIFIED: https://huggingface.co/Qwen/Qwen3-4B] |
| SmolLM3-3B | 3B | Apache-2.0 | 64K (128K YaRN) | 3060 4-bit | 6 languages, no Korean | [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM3-3B] |
| Gemma 3 1B / 4B | 1B / 4B | Gemma Terms + Prohibited Use Policy (gated) | 32K / 128K | fits | 140+ languages incl. Korean; restrictions pass down to users | [VERIFIED: https://huggingface.co/google/gemma-3-1b-it], [VERIFIED: https://ai.google.dev/gemma/terms] |
| Llama 3.2 1B / 3B | 1B / 3B | Llama 3.2 Community License | – | fits | 700M-MAU clause; derivatives must be named "Llama…" | [VERIFIED: https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt] |

Recommendation [HYPOTHESIS]: **default to Qwen3-1.7B in non-thinking mode**, and use **Qwen3-4B or Gemma 3 4B for Korean**. Avoid Llama for open release because of the naming clause. T4 has no bf16, so run fp16 or int4. Renderer FLOPs are reported, but are not counted as core compute.

### 3.2 Faithfulness: the renderer must not add claims

- Neural data-to-text hallucinates. ToTTo (121K examples) was built because noisy datasets made hallucination unmeasurable, and n-gram metrics "are often insufficient to detect hallucinations" [VERIFIED: https://research.google/blog/totto-a-controlled-table-to-text-generation-dataset/]. PARENT-style metrics score the output against the source data, not just against a reference text [VERIFIED: https://aclweb.org/anthology/2022.findings-acl.223.pdf].
- Checking is cheap. AlignScore (355M RoBERTa, 4.7M training examples) "matches or even outperforms" ChatGPT/GPT-4-based consistency metrics across 22 datasets [VERIFIED: https://arxiv.org/abs/2305.16739]. It is used offline as an audit tool only, never in scoring.
- **The mouth has three tiers [HYPOTHESIS]:**
  - **Tier 0, templates (0 params).** Answers, numbers, choices, code and Lean proofs are emitted verbatim from the IR. This covers ~100% of benchmark outputs.
  - **Tier 1, constrained verbalization.** The renderer gets a JSON fact sheet of the claims in the core's trace. Every number, entity and code span is copy-masked from that sheet, using ALEMBIC's 0-parameter suffix-automaton copy as a logits mask.
  - **Tier 2, free prose.** Used only when tier 1 fails, and always followed by the round-trip audit (§4.5).

## 4. Fair-play: proving the renderer is not the brain

**Precedents.** The hypothesis-only NLI baseline beats the majority class while seeing only half the input. Leak channels must therefore be measured, not assumed closed [VERIFIED: https://aclanthology.org/S18-2023/]. The architectural precedent is AlphaGeometry's split, where the LM proposes and a symbolic engine deduces. AlphaProof's human translation of test problems is the **anti-precedent** [VERIFIED above]. Gemini Deep Think (IMO 2025, 35/42) dropped the formal step and worked end-to-end in NL within 4.5h [VERIFIED: deepmind URL in §0]. That shows the industry moving to "one model reads and reasons". We keep the formal core, but must match the end-to-end standard on input.

Protocol (refines the default rules, same spirit):
1. **No renderer calls in the scoring path.** The harness scores `core.answer` directly. CI fails the run if the renderer is loaded during scoring. This sits next to the CI parameter-count assertion (≤100M, every executed weight, including embeddings).
2. **Blindfold, enforced by type.** The renderer input is a `CoreTrace` (IR nodes + values) with no `question` field. A static check requires zero shared 8-grams between the renderer prompt and the question, except spans the core explicitly copied into the IR, which are logged.
3. **Published renderer-alone controls.** Run the renderer on (a) the raw question, (b) an empty or shuffled trace, and (c) the trace with the answer masked. (a) is published as an upper bound; Qwen3-0.6B's MATH-500 77.6 means it will be high. **Pass condition: (b) and (c) ≤ chance + 2σ.**
4. **Counterfactual swap.** Corrupt the core's answer (42→41, B→C) and render. The renderer must carry the *wrong* answer ≥99% of the time [HYPOTHESIS threshold]. If it "fixes" the answer, it is reasoning, and the run is disqualified. This is the cleanest information-flow test.
5. **Round-trip audit.** The core's own reader re-parses the rendered text into IR and diffs it against the original, and AlignScore checks it against the fact sheet. Target: ≤1% unsupported claims on a 1K sample [ESTIMATE].
6. **Input purity.** No LLM or >100M parser in front of the core. Only 0-parameter tokenizer, regex and number normalization, disclosed. The core gets no dataset-name feature. A shuffled-format probe checks this: the same problem in another benchmark's prompt format must give the same answer.
7. **Provenance.** No GPT-3.5/4 outputs anywhere, so TinyGSM data is out. Renderer paraphrases are logged and deduplicated against the test sets (13-gram).

## 5. What scales (brain vs mouth)

| Axis | Scales? | Measure |
|---|---|---|
| Reader params | Saturates early on NLU: 79.9 at 14M → 85.1 at 110M [VERIFIED ELECTRA] | Formalization equivalence vs reader size |
| Grammar-constrained emitter | ≈ one model size, free [VERIFIED PICARD] | Execution accuracy with and without the mask |
| Test-time compute (r, N, search) | Main lever [VERIFIED in externalize.md: Snell] | Accuracy vs log FLOPs at fixed params |
| Verifier share | Verifier > generator (TinyGSM) [VERIFIED in tiny_sota.md] | Ablation inside the 100M |
| Verified self-generated data | AlphaGeometry pattern | Accuracy vs synthetic set size |
| Renderer size | **Must not affect scores** | Swap test ≥99%; control (c) at chance |

**Scaling bet [HYPOTHESIS]:** with the mouth outsourced, every added core parameter goes to reading→IR and reasoning, not to vocabulary and style. A 10B core here is a 10B formalizer+prover, not a 10B chatbot. **Falsifier:** formalization accuracy on the NL→FOL/NL→program suite fails to improve log-linearly from 0.1B to 1B.

## 6. Week-2 kill probes

1. Train a 20–40M BiSSM reader + constrained emitter on 0-parameter synthetic NL→SymPy pairs. Kill or pivot if GSM8K execution accuracy at N=1 is below 25% after 30 GPU-h [ESTIMATE threshold; TinyGSM reached 63.1 at 125M with 12M examples].
2. Run the renderer swap test on 500 items with Qwen3-1.7B. If faithfulness is below 95%, drop to tiers 0 and 1 only.
3. Run renderer-alone control (c). Any recovery above chance means the blindfold has a bug.

## Sources
https://arxiv.org/abs/2003.10555 · https://arxiv.org/abs/2002.10957 · https://huggingface.co/microsoft/deberta-v3-xsmall · https://arxiv.org/abs/2005.14165 · https://arxiv.org/abs/2403.03234 · https://arxiv.org/abs/2211.10435 · https://arxiv.org/abs/2305.09656 · https://arxiv.org/abs/2305.12295 · https://arxiv.org/abs/2310.15164 · https://arxiv.org/html/2509.22338v2 · https://github.com/ElementAI/picard · https://arxiv.org/abs/2306.11644 · https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/ · https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/ · https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad · https://huggingface.co/Qwen/Qwen3-1.7B · https://huggingface.co/Qwen/Qwen3-4B · https://arxiv.org/html/2505.09388 · https://huggingface.co/HuggingFaceTB/SmolLM3-3B · https://huggingface.co/google/gemma-3-1b-it · https://ai.google.dev/gemma/terms · https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt · https://developers.googleblog.com/en/introducing-gemma-3-270m/ · https://research.google/blog/totto-a-controlled-table-to-text-generation-dataset/ · https://aclweb.org/anthology/2022.findings-acl.223.pdf · https://arxiv.org/abs/2305.16739 · https://aclanthology.org/S18-2023/ · on-disk research/{tiny_sota,externalize,archs,baseline}.md
