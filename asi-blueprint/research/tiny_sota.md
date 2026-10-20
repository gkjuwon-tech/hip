# tiny_sota: How well the best tiny models score, and what got them there

Scope: models of 150M or fewer, plus notable ones up to about 1.5B. Date: 2026-10-19.
Labels: **[V:Sx]** means VERIFIED at source Sx, with the URL in the Sources table at the end. **[ESTIMATE]** = my arithmetic or judgment. **[HYPOTHESIS]** = untested design claim.

## 0. Summary

1. **No model of 150M or fewer comes within 25 points of GPT-3.5 on MMLU, HellaSwag, WinoGrande, or MT-Bench.** Best at this size: MMLU about 31.5 (cloze), HellaSwag about 42, WinoGrande 51–53 (guessing gives 50), MT-Bench about 2.5. GPT-3.5: 70.0, 85.5, 81.6, 7.94. [V:S4,S6,S1,S2]
2. **Small models fail first on closed-book world knowledge, and that failure is a hard capacity limit.** The limit is about 2 bits of knowledge per parameter [V:S29]. SmolLM2-135M gets **4.1% on TriviaQA** even after 2T training tokens [V:S4].
3. **Multi-step math is limited mostly by training data, and it can be fixed.** At about 125–140M parameters, GSM8K goes from 1.4% (generic web data) to 16.3% (curated data) [V:S4,S7]. It reaches **63.1%** with 12.3M synthetic problems whose answers are Python programs, and **76.1%** with a 1.3B verifier added [V:S14].
4. **Recursion and test-time compute are the strongest evidence that tiny models can reason.** A 7M recursive network beats DeepSeek R1, o3-mini-high and Gemini 2.5 Pro on Sudoku-Extreme (87.4% vs 0.0%) and ARC-AGI-1 (44.6% vs 15.8–37.0%), though not Grok-4-thinking (66.7%) [V:S17]. The gain comes mostly from training-time refinement loops, not "hierarchy" [V:S16].
5. **Retrieval moves the knowledge ceiling at every model size.** Atlas-770M scores 56.3 on MMLU vs 42.4 closed-book [V:S27]. RETRO's gains "do not diminish" from 150M to 7B [V:S26]. The gains are much smaller on tasks that need reasoning [V:S28].

## 1. The target: GPT-3.5 reference numbers

| Benchmark | GPT-3.5 | Source |
|---|---|---|
| MMLU (5-shot) | 70.0 | [V:S1] |
| HellaSwag (10-shot) | 85.5 | [V:S1] |
| AI2 ARC (25-shot) | 85.2 | [V:S1] |
| WinoGrande (5-shot) | 81.6 | [V:S1] |
| HumanEval (0-shot) | 48.1 | [V:S1] |
| DROP F1 (3-shot) | 64.1 | [V:S1] |
| GSM8K (5-shot) | 57.1 in the GPT-4 report; **77.4** as measured by TinyGSM | [V:S1,S14] |
| MT-Bench | 7.94 | [V:S2] |
| KMMLU (Korean, 5-shot) | 42.47 (random guessing gives 25) | [V:S3] |

The two GSM8K figures differ by 20 points from protocol alone, so fair-play rules must pin the snapshot, prompts and shot count. [ESTIMATE]

## 2. Generalist models of 170M or fewer

| Model (params, train tokens) | HellaSwag | PIQA | ARC-e / ARC-c | WinoGrande | Other | Source |
|---|---|---|---|---|---|---|
| GPT-2 small (117M) | – | – | – | – | LAMBADA 45.99; Natural Questions at or below a trivial 1.0% baseline (1.5B: 4.1%) | [V:S10] |
| Pythia-70M | – | 59.5 | – | – | LAMBADA 18.5 | [V:S9] |
| Pythia-160M (300B) | 30.2 | 61.4 | 43.2 / 24.1 | 51.9 | 6-task avg 40.6 | [V:S8] |
| **Mamba-130M** (300B, attention-free state-space model) | 35.3 | 64.5 | 48.0 / 24.3 | 51.9 | LAMBADA 44.3; avg **44.7** (+4.1 over Pythia-160M) | [V:S8] |
| RWKV-169M (recurrent) | 31.9 | 63.9 | 42.5 / 25.3 | 51.5 | 8-task avg 43.6 | [V:S6] |
| **MobileLLM-125M** (1T, 30 layers, d=576) | 38.9 | 65.3 | 43.9 / 27.1 | 53.1 | 8-task avg 46.3; MT-Bench 2.33; AlpacaEval win rate 24.07% | [V:S6] |
| MobileLLM-LS-125M (shares weights between layers) | 39.5 | 65.7 | 45.8 / 28.7 | 52.1 | avg 47.0; MT-Bench **2.52** | [V:S6] |
| **SmolLM2-135M** (2T) | **42.1** | **68.4** | ARC avg 43.9 | 51.3 | MMLU-cloze 31.5; MMLU-Pro 10.85; TriviaQA 4.1; GSM8K 1.4 | [V:S4] |
| SmolLM2-135M-Instruct | 40.9 | 66.3 | ARC avg 37.3 | – | IFEval 29.9; MT-Bench 1.98; BBH 28.2; GSM8K 1.4 | [V:S4] |
| **MobileLLM-R1-140M-base** (about 4.2T) | – | – | – | – | **GSM8K 16.3** (vs 1.8 for SmolLM2-135M); MATH500 4.6; **HumanEval 15.9** (vs 0.0) | [V:S7] |

Averages cover different task sets; compare only within one source.

## 3. Notable models from 0.27B to 1.5B

| Model | Key scores | Source |
|---|---|---|
| **Gemma 3 270M**: 170M of its parameters are embeddings, 100M are the transformer blocks; 256k vocabulary; 6T tokens | Base model: HellaSwag 40.9, PIQA 67.7, ARC-c 29.0, WinoGrande 52.0, TriviaQA 15.4. Instruction-tuned: **IFEval 51.2**, BBH 26.7, GSM8K 8.4, MATH500 6.8 | [V:S20,S21,S7] |
| Qwen2.5-0.5B: 0.49B total, 0.36B non-embedding; Qwen2.5 series trained on up to 18T tokens | Base: MMLU 47.5, GSM8K 41.6, HumanEval 30.5, HellaSwag 52.1, WinoGrande 56.3. Instruct: IFEval 27.9, MATH 34.4 | [V:S18,S22] |
| **Qwen3-0.6B**: 36T tokens; distilled from Qwen3-32B and Qwen3-235B by matching their output probabilities | Base: **MMLU 52.81**, GSM8K 59.59, MATH 32.44, MMMLU 50.16. Thinking mode: IFEval 59.2, **MATH-500 77.6**, Arena-Hard 8.5, INCLUDE 35.9 | [V:S19] |
| MobileLLM-R1-950M (about 11.7% of Qwen3's token count) | MATH500 74.0, GSM8K 67.5, AIME'24 15.5, LiveCodeBench 19.9 | [V:S7] |
| phi-1 (1.3B; about 7B tokens, 1B of them synthetic textbook data written by GPT-3.5) | HumanEval **50.6**, MBPP 55.5. phi-1-small (**350M**): HumanEval **45** | [V:S12] |
| phi-1.5 (1.3B; 150B training tokens, about 20B of them synthetic) | WinoGrande 73.4, ARC-c 44.4, HellaSwag 47.6, MMLU 37.6, GSM8K 40.2 (solved by writing code), HumanEval 34.1 | [V:S13] |
| DeepSeek-R1-Distill-Qwen-1.5B | MATH-500 **83.9**, AIME'24 28.9, GPQA-Diamond 33.8, LiveCodeBench 16.9. For comparison, GPT-4o-0513: 74.6 / 9.3 / 49.9 / 32.9 | [V:S24] |

phi-1.5 gives a clean data ablation at fixed 1.3B. Web-only → synthetic "textbook" data [V:S13]:
- WinoGrande: 60.4 → 73.4
- GSM8K: under 3 → 40.2
- HumanEval: 17.2 → 34.1
- MMLU: 30.9 → 37.6

## 4. Narrow specialists (domain-limited wins)

| System | Result | Caveat | Source |
|---|---|---|---|
| TinyGSM, 125M generator | GSM8K **63.1** alone; **76.1** with a 1.3B verifier picking from 48 samples; the 1.3B+1.3B pair reaches **81.5** | 12.3M GPT-3.5-generated problems with Python solutions; math-only. Per the paper, "the verifier seems to play a bigger role than generation model size." | [V:S14] |
| phi-1-small (350M) | HumanEval 45 | Python only | [V:S12] |
| **TRM** (7M parameters, 2-layer network applied recursively) | Sudoku-Extreme **87.4** (attention-free MLP version, 5M); Maze-Hard 85.3; ARC-AGI-1 **44.6**; ARC-AGI-2 **7.8**. Same tests: DeepSeek R1 0.0 / 0.0 / 15.8 / 1.3; Gemini 2.5 Pro 37.0 / 4.9 on ARC; Grok-4-thinking 66.7 / 16.0 | About 1,000 training examples with 1,000× augmentation. The attention-free version drops on ARC-AGI-1 (29.6 vs 44.6). Won first place in the ARC Prize 2025 paper awards. | [V:S17,S25] |
| HRM (27M) | Sudoku-Extreme 55.0; Maze-Hard 74.5; ARC-AGI-1 **40.3 claimed → 32 verified** on ARC Prize's semi-private set; ARC-AGI-2 2 verified | Its modules are encoder-only **Transformer** blocks. Ablations show that "outer-loop refinement" is the main driver, and that a same-size Transformer comes within about 5 points. | [V:S15,S16] |
| CompressARC (76K params) | About 20% on ARC-AGI-1 | Trains per puzzle, about 20 minutes on one RTX 4070, no pretraining | [V:S25] |
| NVARC (Kaggle) | 24.03% on ARC-AGI-2 private set | Synthetic data plus test-time training, run under Kaggle's compute limits | [V:S25] |
| TinyStories (under 10M) | Fluent, grammatical multi-paragraph stories | Vocabulary of a 3–4-year-old; GPT-Neo/GPT-2 at 125M "rarely" stay coherent | [V:S11] |

## 5. Small models with retrieval (knowledge kept outside the weights)

| System | Result | Source |
|---|---|---|
| RETRO, 172M | Retrieval adds 30% parameters over the 132M baseline. Helps "at all model sizes" (150M–7B); "improvements do not diminish"; 172M uses up to 10 passages, 7B up to 40. On Natural Questions, the 7.5B scores 45.5 vs 30.4 closed-book. | [V:S26] |
| RETRO caveat | "exploit[s] leakage more strongly than baseline models" by copying overlapping chunks | [V:S26] |
| Atlas-770M (BERT-base retriever) | MMLU (full/transfer) **56.3 vs 42.4** closed-book (+13.9); 5-shot 38.9 vs 29.2. Atlas-11B: 42.4% on Natural Questions from 64 examples; MMLU 0-shot 47.1, above GPT-3 175B's 5-shot 43.9. | [V:S27] |
| MassiveDS (1.4T-token datastore) | Llama-2-7B gains: TriviaQA 64.1→77.0, NQ 26.6→34.6, MMLU 45.8→49.3, MedQA 36.6→39.4. Retriever has 177M params. Small gains on reasoning tasks; Pythia near random on MMLU even at 12B. | [V:S28] |

## 6. Test-time compute and verifiers

| Result | Source |
|---|---|
| Qwen2.5-0.5B-Instruct with compute-optimal test-time scaling: MATH-500 **76.4** vs GPT-4o 74.6. Llama-3.2-1B at N=512: 72.2 vs Llama-3.1-405B's 71.4. **But the process reward models that score each step are 1.5B–72B**, so this would break a 100M all-in budget. | [V:S23] |
| Compute-optimal test-time compute: more than 4× as efficient as best-of-N; beats a 14× larger model at matched FLOPs, only if the base model has "somewhat non-trivial success rates." | [V:S30] |
| HRM: 1→2 refinement loops +13 points; 8 loops about 2×; training with refinement adds 15+ even to single-pass inference. | [V:S16] |
| TinyGSM, 125M generator: verifier 125M→1.3B adds +7.2; generator 125M→1.3B adds +5.1. | [V:S14] |

## 7. Catalog of tricks and measured gains

| Trick | Measured effect | Source |
|---|---|---|
| Deep and thin (about 30 layers instead of 12 at 125M) plus shared embeddings and grouped-query attention | +2.7 points over the previous best 125M model | [V:S6] |
| Sharing weights between neighbouring layers (same params, more compute) | +0.7 more (125M) | [V:S6] |
| Training far past compute-optimal, in stages | SmolLM2-135M: 2T tokens (1.7B: about 11T); Gemma 3 270M: 6T; Qwen3-0.6B: 36T | [V:S4,S5,S20,S19] |
| Curated data plus distillation in mid-training (MobileLLM-R1) | 140M: GSM8K 1.8→16.3, HumanEval 0→15.9 | [V:S7] |
| Synthetic textbook data (phi) | 1.3B: WinoGrande +13, GSM8K +37 | [V:S13] |
| Distillation from a large teacher (Qwen3 strong-to-weak) | Needs about 1/10 of the GPU-hours of the full pipeline | [V:S19] |
| Solutions written as executable code (TinyGSM, phi-1.5 "via coding") | 125M reaches 63.1 on GSM8K | [V:S14] |
| Attention-free backbone (Mamba) | +4.1 on the 6-task average vs Pythia-160M at equal tokens | [V:S8] |
| Optimizer and engineering (modded-nanogpt: Muon optimizer, FP8, hashed n-gram embeddings) | 124M reaches GPT-2's 3.28 validation loss in under 330M tokens vs 10B for the llm.c baseline (about 30× fewer). Record: about 40 seconds on 8×H100. | [V:S31] |
| Labelling training data by domain | Junk-heavy data shrinks knowledge capacity up to 20×; a domain tag recovers most of it (down to 2×). Int4 quantization costs more than 2× capacity; int8 costs nothing. | [V:S29] |

## 8. Where small parameter counts fail, ordered from hardest to fix to easiest

1. **Closed-book factual knowledge (a capacity limit).** Models store about 2 bits per parameter after 1,000 exposures to a fact, and about 1 bit after 100 [V:S29].
   - 100M params ≈ 0.2 Gbit, about 25 MB of facts at most [ESTIMATE].
   - 7B's 14 Gbit exceeds English Wikipedia plus textbooks [V:S29]; 100M holds about 1/70 of that [ESTIMATE].
   - Observed: TriviaQA 4.1 (135M) and 15.4 (270M); GPT-2 small on Natural Questions is at the trivial 1% baseline [V:S4,S20,S10].
   - **Only demonstrated fix: retrieval** [V:S27,S28].
2. **Open-ended chat quality.** MT-Bench is 1.98–2.52 at about 125–135M vs 7.94 for GPT-3.5 [V:S4,S6,S2]. It needs knowledge, fluency and reasoning at once; no tiny-model route is demonstrated [ESTIMATE].
3. **Commonsense disambiguation.** WinoGrande is 51–53 at every size up to 270M [V:S4,S6,S20]. It rises only at 0.5B (56.3) and with synthetic data at 1.3B (73.4) [V:S18,S13]; HellaSwag stays ≤42 up to 270M.
4. **Korean and other multilingual knowledge.** Polyglot-Ko-1.3B scores 27.97 on KMMLU vs 42.47 for GPT-3.5 [V:S3]. Qwen3-0.6B scores 43.1 on multilingual MMLU (MMMLU) [V:S19]. Big vocabularies cost: 63% of Gemma 270M is embeddings [V:S21].
5. **Multi-step math and code.** This is limited by data and tools, and can be fixed. GSM8K goes 1.4 → 16.3 → 63.1 at about 125–140M [V:S4,S7,S14]. A 350M model reaches 45 on HumanEval [V:S12].
6. **Following instruction formats.** This is the cheapest to fix: IFEval goes 29.9 → 51.2 → 59.2 (135M → 270M → 0.6B) [V:S4,S20,S19].
7. **Closed-world puzzles.** Tiny recursive networks already win here [V:S17]. But wins are narrow, augmentation-heavy, and shrink under verification (HRM 40.3 → 32) [V:S16].

## 9. Implications for a non-Transformer model under 100M

- **[HYPOTHESIS]** Split into a small reasoning-and-reading core plus an external datastore. Caveats:
  - Published retrievers (BERT-base in Atlas, 177M in MassiveDS [V:S27,S28]) alone use the whole 100M budget; the retriever must be tiny or non-neural (BM25, n-gram), and fair-play rules must say if that is allowed.
  - Retrieval helps little with reasoning [V:S28]: it fixes item 1, not items 2–3.
- **[HYPOTHESIS]** Recursion on a shared core (TRM, HRM, MobileLLM-LS) is the best-evidenced way to scale reasoning without adding parameters [V:S17,S16,S6]. The verifier counts toward the 100M budget. TinyGSM suggests putting more of the parameters into the verifier than into the generator [V:S14].
- **[HYPOTHESIS] Content-addressable memory is still needed without attention.** Attention-free cores lose on spatial and long-range tasks: TRM-MLP falls from 44.6 to 29.6 on ARC [V:S17]. HRM and TRM-Att both use attention blocks [V:S15,S17]. A non-Transformer design therefore needs some other way to look up stored content, such as state-space mixing (Mamba +4.1 [V:S8]) combined with explicit reads from retrieval or memory.
- **[ESTIMATE] Compute does not allow a SmolLM-style recipe.**
  - SmolLM2-135M's run: 6 × 1.35e8 × 2e12 ≈ 1.6e21 FLOPs, about 370 days nonstop on two T4s at about 25 effective TFLOP/s each (40% of 65 FP16 peak).
  - At an assumed (unverified) Kaggle quota of about 30 GPU-hours/week: about 4e18 FLOPs, or about 7B tokens/week at 100M params.
  - So combine modded-nanogpt-level token efficiency [V:S31] with dense synthetic or distilled data [V:S12,S14].
- **[ESTIMATE]** Honest gap: no published system of 150M or fewer shows a route to GPT-3.5 on HellaSwag, WinoGrande or MT-Bench. Those, plus MMLU without retrieval, are the boss fights; math, code and instruction following look reachable.

## Sources

| Key | URL |
|---|---|
| S1 | https://arxiv.org/html/2303.08774 (GPT-4 Technical Report) |
| S2 | https://lmsys.org/blog/2023-06-22-leaderboard/ ; https://arxiv.org/pdf/2306.05685 |
| S3 | https://ar5iv.labs.arxiv.org/html/2402.11548 (KMMLU) |
| S4 | https://huggingface.co/HuggingFaceTB/SmolLM2-135M |
| S5 | https://arxiv.org/abs/2502.02737 (SmolLM2 paper) |
| S6 | https://arxiv.org/html/2402.14905 (MobileLLM) |
| S7 | https://arxiv.org/html/2509.24945v3 (MobileLLM-R1) |
| S8 | https://arxiv.org/html/2312.00752 (Mamba) |
| S9 | https://arxiv.org/pdf/2304.01373 (Pythia) |
| S10 | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf (GPT-2) |
| S11 | https://arxiv.org/abs/2305.07759 (TinyStories) |
| S12 | https://arxiv.org/abs/2306.11644 (phi-1) |
| S13 | https://arxiv.org/html/2309.05463 (phi-1.5) |
| S14 | https://arxiv.org/html/2312.09241 (TinyGSM) |
| S15 | https://arxiv.org/html/2506.21734 (HRM) |
| S16 | https://arcprize.org/blog/hrm-analysis |
| S17 | https://arxiv.org/html/2510.04871 (TRM) |
| S18 | https://qwenlm.github.io/blog/qwen2.5-llm/ |
| S19 | https://arxiv.org/html/2505.09388 (Qwen3) |
| S20 | https://huggingface.co/google/gemma-3-270m |
| S21 | https://developers.googleblog.com/en/introducing-gemma-3-270m/ |
| S22 | https://huggingface.co/Qwen/Qwen2.5-0.5B |
| S23 | https://arxiv.org/html/2502.06703 (Can 1B surpass 405B) |
| S24 | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| S25 | https://arcprize.org/blog/arc-prize-2025-results-analysis |
| S26 | https://arxiv.org/abs/2112.04426 (RETRO) |
| S27 | https://arxiv.org/abs/2208.03299 (Atlas) |
| S28 | https://arxiv.org/html/2407.12854 (MassiveDS) |
| S29 | https://arxiv.org/html/2404.05405 (Physics of LMs 3.3, knowledge capacity) |
| S30 | https://arxiv.org/abs/2408.03314 (Scaling test-time compute) |
| S31 | https://github.com/KellerJordan/modded-nanogpt |
