# io_split: the "brain vs mouth" split (aphasic-genius pivot)

Scope: is it cheaper in parameters to understand than to generate? How big must the reader/formalizer be? Which open renderer can verbalize the core's formal output on T4/3060? How do we prove the renderer does no reasoning? Labels: [VERIFIED: url], [ESTIMATE], [HYPOTHESIS].

## 0. Bottom line

1. **Understanding is cheap. Discriminative readers dominate generative LMs per parameter and per FLOP.** ELECTRA-Small (14M) scores 79.9 on GLUE dev, against 78.8 for GPT-1 (117M, 29x the training compute). It trained in 4 days on one V100 [VERIFIED: https://arxiv.org/abs/2003.10555]. DeBERTa-v3-xsmall has a 22M backbone and gets MNLI 88.1/88.3 [VERIFIED: https://huggingface.co/microsoft/deberta-v3-xsmall]. On SuperGLUE, few-shot GPT-3 175B (71.8) only roughly ties a fine-tuned 335M BERT-Large (69.0) [VERIFIED: https://arxiv.org/abs/2005.14165]. **Reading is not where the 100M budget goes.**
2. **The aphasic core still has to generate. It generates *formal* language, not English.** Code, programs, logic forms and proofs are sequences too. The pivot removes the cost of fluency (open vocabulary, style, world-knowledge recall). It does not remove the cost of formalization. Today's evidence puts NL→program/logic as the bottleneck. A 220M T5-base reaches only 29.5% exact / 32.7% equivalence on NL→FOL [VERIFIED: https://arxiv.org/html/2509.22338v2]. On Spider it reaches 59.4% exact-match, rising to 66.6% with grammar-constrained decoding (PICARD) [VERIFIED: https://github.com/ElementAI/picard]. **The real wall is reading → formal IR, not rendering.**
3. **The mouth is a solved, free commodity.** Apache-2.0 renderers exist at 0.6B–3B: Qwen3-0.6B/1.7B/4B and SmolLM3-3B [VERIFIED: https://huggingface.co/Qwen/Qwen3-1.7B], [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM3-3B]. They fit a 6GB 3060 at fp16 (≤1.7B) or 4-bit (3–4B) [ESTIMATE]. The risk is not that the renderer is too weak. The risk is that it is **too strong**: it can add claims, fix wrong answers, or "re-solve" the task. The fair-play protocol in §4 is built to catch exactly that.
4. **There is precedent: formal engine reasons, a separate component handles language.** For AlphaProof, IMO 2024 problems were "manually translated into formal mathematical language", and a separate fine-tuned Gemini formalizer built the training library from about 1M informal problems [VERIFIED: https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/]. That is the honest caveat: **DeepMind kept the hardest part (autoformalization of the test problems) out of the scored loop by doing it by hand.** Our rule (4), "the core reads the input itself", is stricter than AlphaProof's protocol. That is good for credibility, but it is the single biggest risk to the milestone.

## 1. Evidence: understanding is cheaper than generating

### 1.1 Small encoders vs generative LMs (same benchmarks)

| Model | Params | Type | Score | Source |
|---|---|---|---|---|
| ELECTRA-Small | 14M | discriminative (replaced-token detection) | GLUE dev **79.9**; GLUE test 78.0 (MNLI 79.7, QNLI 87.7) | [VERIFIED: https://arxiv.org/abs/2003.10555] |
| ELECTRA-Small, 6.25% trained | 14M | same | GLUE dev **74.1** after **6h on 1 V100** (8.9e16 FLOPs) | same |
| BERT-Small (same compute) | 14M | MLM | GLUE dev 75.1 | same |
| GPT-1 | 117M | causal LM | GLUE dev 78.8; test 75.9; 4.0e19 FLOPs (25d on 8 P6000) | same |
| BERT-Base / ELECTRA-Base | 110M | MLM / RTD | GLUE dev 82.2 / 85.1 | same |
| TinyBERT / MobileBERT | 14.5M / 25.3M | distilled encoders | GLUE test 80.6 / 81.0 | same (Table 8) |
| DeBERTa-v3-xsmall | 22M backbone + 48M embeddings (128K vocab) | RTD + GDES | MNLI **88.1/88.3**, SQuAD2.0 84.8 F1 | [VERIFIED: https://huggingface.co/microsoft/deberta-v3-xsmall] |
| BERT-Large (fine-tuned) | 335M | MLM | SuperGLUE **69.0**, BoolQ 77.4, RTE 71.7, WiC 69.6 | [VERIFIED: https://arxiv.org/abs/2005.14165] |
| GPT-3 few-shot | 175B | causal LM | SuperGLUE **71.8**, BoolQ 76.4, RTE 69.0, **WiC 49.4** (chance level) | same |
| GPT-3 few-shot reading | 175B | causal LM | **DROP 36.5 F1**, RACE-h 46.8, against fine-tuned SOTA 89.1 / 90.0 | same (Table 3.7) |

Read-outs:
- **About 500x fewer parameters for the same NLU score.** A 335M fine-tuned encoder roughly ties 175B few-shot on SuperGLUE. GPT-3's authors themselves conjecture that "a large bidirectional model would be stronger at fine-tuning" [VERIFIED: https://arxiv.org/abs/2005.14165]. Caveat: fine-tuned vs few-shot is not apples to apples. Our core *is* trained on its tasks, so the fine-tuned regime is the relevant one [ESTIMATE].
- **Discriminative objectives are about 30x more compute-efficient at small scale.** ELECTRA-Small beats GPT-1 with 1/29 the FLOPs [VERIFIED above]. At 74.1 GLUE for 6 GPU-hours, it fits inside one Kaggle session.
- **Embeddings eat small models.** DeBERTa-v3-xsmall spends 48M of about 70M parameters on a 128K vocabulary. Gemma 3 270M spends 170M on embeddings [VERIFIED: https://developers.googleblog.com/en/introducing-gemma-3-270m/]. The aphasic core does not need a large *output* vocabulary at all, because it emits formal tokens. A small input vocabulary or byte/char reader plus a tiny formal output vocabulary could free about 40–60% of the 100M budget [ESTIMATE].

### 1.2 Bidirectional SSM readers (non-Transformer, fits our backbone rule)

- **Caduceus** (BiMamba + reverse-complement equivariance, ICML 2024). At **1.9M params** it beats attention models of 117M–500M on 8 of 18 Nucleotide-Transformer tasks. Example: H3K4me2 MCC 0.487 vs 0.319 for NT-v2 500M. On long-range variant-effect prediction, Caduceus-PS exceeds NT-v2 500M, and beyond 100k bp from the TSS it beats Enformer (252M). The abstract claims it "exceeds the performance of 10x larger models" [VERIFIED: https://arxiv.org/abs/2403.03234]. The lever is bidirectionality plus a symmetry prior, not scale. It still loses splice-site tasks to NT-v2 (0.940 vs 0.983).
- Implication [HYPOTHESIS]: the core's reader can be a **bidirectional SSM/RNN encoder** (BiMamba-style, or GatedDeltaNet run in both directions). It is allowed under the no-Transformer rule, and the bidirectional pass is exactly what decoder-only LMs lack. This reuses the pre-pivot "bidirectional judge tower" idea: one encoder serves both as reader and as verifier. T4 caveat: Mamba kernels need Triton, and Triton ≥3.3 dropped sm_75 (see archs.md). Pin Triton 3.2 or use a pure-PyTorch chunked scan [VERIFIED in archs.md].

### 1.3 But: understanding ≠ reasoning

GLUE/SuperGLUE-level reading saturates at 14–70M. The tasks that matter (GSM8K, DROP, BBH) need *multi-step* computation over what was read. GPT-3's DROP 36.5 is a reasoning failure, not a reading failure. The split therefore has three parts, not two:

**reader (cheap) → reasoner (where the budget goes: recurrence, search, tools) → formal emitter (grammar-constrained, small vocab).**

## 2. Neuro-symbolic interface: how big must the formalizer be?

| System | Reader/formalizer | Engine | Result | Source |
|---|---|---|---|---|
| PAL | Codex (≈175B-class) | Python | GSM8K **+15 abs** over PaLM-540B CoT | [VERIFIED: https://arxiv.org/abs/2211.10435] |
| SatLM | GPT-3.5-class (code-davinci) | Z3 SAT/SMT | GSM-Sys **69.4** greedy vs 46.5 CoT, 43.4 PAL; 80.9 with self-consistency; full GSM8K 84.8 SC (+2.4 over ProgLM) | [VERIFIED: https://arxiv.org/abs/2305.09656] |
| Logic-LM | GPT-3.5/4 | Prover9, Z3, CSP, Pyke | **+39.2%** over standard prompting, **+18.4%** over CoT, averaged over 5 logic datasets | [VERIFIED: https://arxiv.org/abs/2305.12295] |
| LINC | StarCoder+ **15.5B** as semantic parser to FOL | Prover9 | ProofWriter: beats GPT-3.5 CoT by **38 abs** and GPT-4 CoT by **10 abs**; failure modes complementary to CoT | [VERIFIED: https://arxiv.org/abs/2310.15164] |
| NL→FOL fine-tune | **T5-base 220M** / T5-3B / Flan-T5-XXL 11B vs LLaMA3.1-8B, Mistral-24B | equivalence checker | Exact 29.5 / 33.0 / 35.8; Equiv **32.7 / 36.2 / 38.8**, rising to 50.2 / 51.5 / 54.8 when predicates are given. LLaMA-8B: 32.0 Equiv. **T5-base ≈ LLaMA-8B.** Predicate extraction is the main bottleneck. | [VERIFIED: https://arxiv.org/html/2509.22338v2] |
| Text-to-SQL (Spider dev) | T5-Base / Large / 3B | SQL executor | EM 59.4 / 65.3 / 71.5. With PICARD constrained decoding: **66.6 / 69.1 / 75.5** (+7.2 at base); exec 68.4 / 72.9 / 79.3 | [VERIFIED: https://github.com/ElementAI/picard] |
| TinyGSM | 125M Python-program generator | Python + 1.3B verifier | GSM8K **63.1** alone; 76.1 with the verifier | [VERIFIED in tiny_sota.md, arXiv 2312.09241] |
| phi-1-small | 350M | code | HumanEval 45% | [VERIFIED: https://arxiv.org/abs/2306.11644] |
| AlphaGeometry | LM proposes auxiliary constructions | symbolic deduction engine (DD+AR) | IMO geometry **25/30** (gold avg 25.9); 100M synthetic examples | [VERIFIED: https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/] |

Read-outs:
- **Encoder-decoder formalizers punch about 30x above their weight.** T5-base 220M ≈ LLaMA-8B on NL→FOL [VERIFIED above]. The paper attributes this to the encoder explicitly encoding the input, which matches §1. Our reader+emitter should be **encoder (bidirectional) + small constrained decoder**, not decoder-only [HYPOTHESIS, consistent with the data].
- **Grammar-constrained decoding is worth roughly one model size.** On Spider, T5-Base+PICARD (66.6) beats unconstrained T5-Large (65.3) [VERIFIED above]. It costs 0 params, so it is pure gain under the 100M rule. This keeps "typed program IR + grammar-constrained decoding" from ALEMBIC-0.
- **The data says formalization quality is the ceiling.** Every LLM+solver win above uses a ≥15B formalizer, except TinyGSM (125M). TinyGSM's 12.3M training problems were **GPT-3.5-generated**, which our rules ban. A replacement must come from **0-param generators**: SymPy/Z3/Lean-side problem synthesis, rendered to NL by templates and by the *renderer offline at training time only*. The core then learns the inverse map, NL→program [HYPOTHESIS; this is the AlphaGeometry recipe applied to NL]. Check: is renderer-written training data allowed? The renderer is a <5B open model, not GPT-3.5/4, so yes. It must be disclosed, and the renderer must never see benchmark items.
- **Projection for the core [ESTIMATE].** Scaling from T5-base 220M (≈50% equivalence with given predicates) and TinyGSM 125M (63%), a 60–80M bidirectional reader + constrained emitter with execution feedback and self-consistency (N=16–64) plausibly reaches GSM8K 60–75%. That beats 57.1 (GPT-4-report GPT-3.5) but does **not** clear the max-snapshot 78.1 "by a wide margin". Clearing the 78.1 GSM8K bar by a wide margin needs search plus a verifier, i.e. test-time compute. Parameters alone will not do it.
- **Wall.** AlphaProof failed both IMO-2024 combinatorics problems [VERIFIED in externalize.md], and its test problems were hand-formalized. Informal, combinatorial or "common-sense" problems resist formal IR. Route around: (a) the IR includes a "neural step" escape hatch, a recurrent latent reasoning step (TRM-style) that the core executes itself, so not everything must be formal; (b) report formalization coverage per benchmark honestly.

## 3. Renderer side

### 3.1 Candidates (renderer is outside the 100M, never in scoring)

| Model | Params | License | Context | Fit [ESTIMATE] | Notes | Source |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | Apache-2.0 | 32K | 3060 fp16 ≈1.3GB | MATH-500 77.6 (thinking): **strong reasoner, so blindfold risk is high** | [VERIFIED in tiny_sota.md: https://arxiv.org/html/2505.09388] |
| Qwen3-1.7B | 1.7B (1.4B non-emb) | Apache-2.0 | 32K | 3060 fp16 ≈3.4GB; T4 fp16 fine | thinking mode on by default; force `enable_thinking=False` | [VERIFIED: https://huggingface.co/Qwen/Qwen3-1.7B] |
| Qwen3-4B | 4B | Apache-2.0 | – | 3060 4-bit ≈2.5–3GB | best fluency in budget | [ESTIMATE; license per Qwen3 family] |
| SmolLM3-3B | 3B | **Apache-2.0** | 64K (128K YaRN) | 3060 4-bit; T4 fp16 ≈6GB | 6 native languages (no Korean) | [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM3-3B] |
| Gemma 3 1B / 4B | 1B / 4B | Gemma Terms + Prohibited Use Policy (gated) | 32K / 128K | fits | **140+ languages, incl. Korean**; restrictions pass down to users | [VERIFIED: https://huggingface.co/google/gemma-3-1b-it], [VERIFIED: https://ai.google.dev/gemma/terms] |
| Llama 3.2 1B / 3B | 1B / 3B | Llama 3.2 Community License | – | fits | 700M-MAU clause; "Built with Llama"; derivative models must be named "Llama…" | [VERIFIED: https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt] |

Recommendation [HYPOTHESIS]: **default Qwen3-1.7B (Apache, non-thinking mode). Use Gemma 3 4B for Korean output.** Avoid Llama for open release because of the naming clause. Note: bf16 is unavailable on T4, so run the renderer in fp16 or int4. Renderer FLOPs are not counted in core test-time compute, but report them.

### 3.2 Faithfulness: the renderer must not add claims

- Neural data-to-text hallucinates. ToTTo (121K train) was built because noisy datasets made hallucination unmeasurable, and n-gram metrics "are often insufficient to detect hallucinations" [VERIFIED: https://research.google/blog/totto-a-controlled-table-to-text-generation-dataset/]. Faithfulness-aware metrics such as PARENT score against the source data, not just the reference text [VERIFIED: https://aclweb.org/anthology/2022.findings-acl.223.pdf].
- Automatic checking is cheap. AlignScore (355M RoBERTa, trained on 4.7M examples from 7 tasks) "matches or even outperforms" ChatGPT/GPT-4-based factual-consistency metrics on 22 datasets [VERIFIED: https://arxiv.org/abs/2305.16739]. It can serve as an offline **audit tool**. It is not part of the core and not in scoring, so no 100M violation.
- Design, a three-tier mouth [HYPOTHESIS]:
  1. **Tier 0, templates (0 params):** answers, numbers, choice letters, code blocks and Lean proofs are emitted verbatim from the core's IR. This covers about 100% of benchmark outputs.
  2. **Tier 1, constrained verbalization:** the renderer gets a JSON "fact sheet" (the claims list from the core's trace). Decoding is constrained so every number, entity and code span is copied from the fact sheet. This is a copy-only lexicon, reusing ALEMBIC's 0-param suffix-automaton copy as a logits mask.
  3. **Tier 2, free prose:** runs only after tier 1 fails, and is always followed by a round-trip check (§4).

## 4. Fair-play: proving the renderer is not the brain

Precedents. The **hypothesis-only baseline** for NLI showed that a model seeing half the input beats the majority class on many datasets: dataset artifacts leak answers [VERIFIED: https://aclanthology.org/S18-2023/]. The lesson: **leak channels must be measured, not assumed closed.** AlphaGeometry's split (LM proposes, symbolic engine deduces) and AlphaProof's split (Lean verifies, formalizer separate) are the architectural precedents [VERIFIED above]. AlphaProof's hand-formalization of test problems is the anti-precedent: we must not let a non-core component formalize test inputs.

Protocol (refines the default rules, same spirit):

1. **Scoring path contains zero renderer calls.** The benchmark harness consumes `core.answer` (number, choice, grid, code, proof) directly. A CI assertion fails the run if the renderer process is even loaded during scoring (process-level import check), alongside the existing CI parameter-count assertion (≤100M, every executed weight counted, including the reader's embeddings).
2. **Blindfold, enforced by type.** The renderer input type is `CoreTrace` (IR nodes + values). It has no `question: str` field. The question text never crosses the boundary. Static check: the renderer prompt must have **zero n-gram overlap of 8 or more tokens with the question** (automated grep). Exceptions are spans the core explicitly copied into IR, and those are logged.
3. **Renderer-alone control (published).** Run the renderer on (a) the raw question, which gives its own benchmark score (an upper bound on what it could contribute), (b) an empty or shuffled CoreTrace, and (c) the CoreTrace with the answer field masked. **Requirement: answer recoverable from (c) ≤ chance + 2σ.** Qwen3-0.6B gets MATH-500 77.6, so (a) will be high. That is exactly why (b) and (c) must be near chance.
4. **Counterfactual swap test (faithfulness).** Corrupt the core's answer (e.g., 42→41, choice B→C) and render. The renderer must carry the *wrong* answer ≥99% of the time [HYPOTHESIS threshold]. If it "fixes" errors, it is reasoning, and the result is disqualified. This is the cleanest information-flow audit: it measures whether the output depends on the core.
5. **Round-trip / entailment audit.** Re-parse the rendered text with the core's own reader into IR and diff it against the original. Also score with AlignScore against the fact sheet. Report the fraction of unsupported claims on a 1K sample. Target ≤1% [ESTIMATE].
6. **Input-side purity (rule 4).** No LLM or >100M parser in front of the core. Tokenizer, regex or number normalization (0 params) is allowed and disclosed. Benchmark-identity routing is banned: the core gets no dataset-name feature. Test with a shuffled-format probe, where the same problem in a different benchmark's prompt format must give the same answer.
7. **Training-data provenance.** No GPT-3.5/4 outputs (this disqualifies reusing TinyGSM data). Renderer-generated NL paraphrases of solver-synthesized problems are allowed, logged, and deduplicated against test sets with 13-gram overlap checks.

## 5. What scales (brain vs mouth)

| Axis | Scales? | How to measure |
|---|---|---|
| Reader params (14M→70M) | Saturates early on NLU (GLUE 79.9@14M → 85.1@110M) [VERIFIED ELECTRA] | Formalization equivalence accuracy vs reader size |
| Emitter + constrained grammar | +7 EM per size step from constraints [VERIFIED PICARD] | Exec accuracy with and without grammar mask |
| Test-time compute (recurrence r, samples N, search) | Main lever [VERIFIED in externalize.md: Snell] | Accuracy vs log(FLOPs) at fixed params |
| Verifier share of params | Verifier > generator (TinyGSM +7.2 vs +5.1) [VERIFIED in tiny_sota.md] | Ablation inside the 100M budget |
| Self-generated verified data | AlphaGeometry pattern | Accuracy vs synthetic set size |
| Renderer size | **Not counted, must not affect scores** | Swap test stays ≥99%; renderer-alone (c) stays at chance |

**Key scaling claim [HYPOTHESIS]:** because the mouth is outsourced, each added parameter of the core goes to reading→IR and reasoning, not to vocabulary and style. A 10B core in this design should look like a 10B formalizer+prover, not a 10B chatbot. That is the bet. It is falsified if the core's formalization accuracy does not improve log-linearly from 0.1B to 1B on the NL→FOL / NL→program suite.

## 6. Week-2 kill probes (io_split specific)

1. Train a 20–40M bidirectional SSM reader + constrained emitter on 0-param synthetic NL→SymPy pairs. Kill or pivot if GSM8K exec-accuracy at N=1 is below 25% after 30 GPU-h [ESTIMATE threshold, compare TinyGSM 63.1 at 125M with 12M examples].
2. Renderer swap test on 500 items with Qwen3-1.7B. If faithfulness is below 95%, drop to tier-0/1 only.
3. Renderer-alone (c) leak test. Any above-chance recovery means a bug in the blindfold.

## Sources actually used
https://arxiv.org/abs/2003.10555 · https://huggingface.co/microsoft/deberta-v3-xsmall · https://arxiv.org/abs/2005.14165 · https://arxiv.org/abs/2403.03234 · https://arxiv.org/abs/2211.10435 · https://arxiv.org/abs/2305.09656 · https://arxiv.org/abs/2305.12295 · https://arxiv.org/abs/2310.15164 · https://arxiv.org/html/2509.22338v2 · https://github.com/ElementAI/picard · https://arxiv.org/abs/2306.11644 · https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/ · https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/ · https://huggingface.co/Qwen/Qwen3-1.7B · https://arxiv.org/html/2505.09388 · https://huggingface.co/HuggingFaceTB/SmolLM3-3B · https://huggingface.co/google/gemma-3-1b-it · https://ai.google.dev/gemma/terms · https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt · https://developers.googleblog.com/en/introducing-gemma-3-270m/ · https://research.google/blog/totto-a-controlled-table-to-text-generation-dataset/ · https://aclweb.org/anthology/2022.findings-acl.223.pdf · https://arxiv.org/abs/2305.16739 · https://aclanthology.org/S18-2023/ · plus on-disk research/{tiny_sota,externalize,archs,baseline}.md
