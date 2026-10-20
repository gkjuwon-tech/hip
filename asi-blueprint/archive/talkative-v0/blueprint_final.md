# ALEMBIC-0: Final Blueprint

**A sub-100M, attention-free, neuro-symbolic compiler that distills tasks into programs, facts into a library, and judgement into a bidirectional judge.**

Final canonical version, superseding `blueprint_v1.md`. It applies the completeness critique and the numbers audit of v1. Date: 2026-10-19.

Labels:
- **[VERIFIED: url]**: a number or fact taken from a primary source cited in the research digests, the proposals, or a search made for this version.
- **[ESTIMATE]**: my arithmetic or judgement.
- **[HYPOTHESIS]**: an untested design bet.

Every derived number (parameters, FLOPs, hours, bytes, per-query cost) comes from one script, `/home/user/hip/asi-blueprint/calc/alembic_budget.py`. The v1 script is kept as `calc/alembic_budget_v1.py` for provenance.
- **If this document and the script disagree, the script wins.**
- `python3 calc/alembic_budget.py --check` is the CI gate. It asserts:
  - the 100M cap, the pool exclusions and the per-query cap for every controller mode;
  - that the TPU and T4 plans fit their 12-week windows.

It currently prints `OK`.

**Why the name.** An alembic is the vessel alchemists used to distill. The system distills four things:
- a messy prompt into a typed program;
- facts into an external library;
- open teachers into *rankings* and aligned logits rather than retrievable text;
- its own verified self-play into its next training set.

The "-0" marks the first round of a campaign, not a finished system.

**Lineage:**
- **Spine: COMPILER-0.** Typed program IR, executors, bidirectional judge tower, skill library, and a north-star mapping in which only the tool registry changes. It had the highest summed score across the three lenses (feasibility 7.0, win 2.4, north star 8.2).
- **Engineering skeleton: OUROBOROS-0 (scaler).** d=640 with 5 heads × 128, per-query recurrence r, `<think>` tokens, suffix-automaton copy, pre-registered parameter pools, the scope fence, and the "win-or-extend" rule.
- **Admission law: ORRERY (worldmodel).** ORRERY-0 control-arm discipline, frozen-lower-layer retriever keys, state forking for multiple choice (now after the stem), and the pooled chat gate.
- **Scaling methodology: CORTEX + LIBRARIAN.** A scaling grid published before training and fit on smooth retrieval-conditioned NLL, a concept skeleton with long-tail fact dropout, the pointer-not-mixer rule, the count-vs-fit parameter rule, and the human-only gated store.

---

## 한국어 요약 (1쪽)

**무엇을 만드나.**
- ALEMBIC-0는 트랜스포머가 아닌 시스템이다. 어디에도 softmax attention이 없다.
- 학습 가능한 파라미터는 **81,110,658개**다. 토크나이저 점수까지 포함하며, 상한 1억 개에서 1,889만 개가 남는다.
- 일은 이렇게 나눈다.
  - **지식**은 사람이 쓴 200억 토큰짜리 외부 저장소에 둔다.
  - **정확한 계산**은 Python/SymPy 실행기가 맡는다.
  - **추론 깊이**는 같은 4개 층을 r번 반복해서 시간 축으로 얻는다.
  - **"맞았는가"**는 실행 검증기와 판정기(judge tower)가 판단한다.
- 가중치에 남는 것은 네 가지다. 언어를 프로그램으로 바꾸는 번역기, 근거를 읽고 복사하는 독해기, 반복되는 사고 단계, 그리고 판정기.

**솔직한 전망.**
- 12개월 차(M1)에 GPT-3.5를 넘을 가능성이 있는 행은 3–6개다. ARC-AGI-1, NQ, DROP, KMMLU가 유력하고, BBH와 MATH도 가능성이 있다.
- **모든 행에서 동시에 이길 확률은 0.1% 미만이다 [추정].**
- 0.1B에서 질 가능성이 큰 "보스"는 네 가지다.
  - 채팅과 글쓰기
  - HellaSwag와 WinoGrande
  - MMLU와 ARC-Challenge
  - 코드
- 각 보스마다 파라미터를 늘리지 않고 돌릴 수 있는 손잡이와, 5–8주 안에 결과가 나오는 반증 실험을 붙였다.

**계산 예산** (스크립트 출력).
- Kaggle TPU v5e-8: 12주 240시간 중 **203시간**을 쓴다(여유 15%).
- T4×2: 11주 330시간 중 **279시간**을 쓴다(여유 15%).
- CPU: 약 **1,000 CPU-시간**.
- TPU를 못 쓰면 T4만으로 약 **38 T4×2-주**가 걸린다. 일정이 몇 달 밀리며, 숨기지 않고 공개한다.

**결정 시점.**
- 5주차에 벽 실험(X1)의 기울기를 측정한다.
- **8주차에 사용자에게 보고한다.** 0.1B로는 도달할 수 없는 행(PARAMETER-BOUND)을 곡선과 함께 보여 준다.
- 1B 트랙으로 갈지는 사용자가 결정한다. 지는 행을 "이긴 것"으로 바꾸거나 목표를 몰래 낮추는 일은 없다.

**공정 규칙.**
- 1억 파라미터 상한에는 모든 학습 가중치가 들어간다. 토크나이저 점수와 보정기(calibrator)도 포함한다.
- 추론 중에 다른 LLM이나 신경망을 호출하지 않는다.
- 테스트셋 오염을 금지한다.
- 체크포인트 하나, 전역 설정 하나만 쓴다. 벤치마크별 설정은 금지한다.
- 진 행도 그대로 공개한다.

**확장성.**
- 커지는 축은 파라미터만이 아니다. 저장소 크기, 검증된 스킬 라이브러리, 테스트 시 연산(r, N), 도구, 검증된 자기 생성 데이터가 모두 축이다.
- "10B면 100배 똑똑해지나?"에는 **예측할 수 없다**고 답한다. 5–40M 실험은 0.1B → 1B 한 단계만 근거 있게 예측한다.
- 10B를 무료 할당량으로 학습하면 약 24년이 걸린다. 그래서 TRC나 연구 지원이 필요하다.

**안전.**
- M1에는 생물·화학 데이터와 도구가 없다.
- 외부 이중용도(dual-use) 검토자가 없으면 바이오 트랙은 시작하지 않는다.

**사용자에게 요청하는 것.**
1. **1주차:**
   - 노트북 SSD 여유 공간이 150GB 이상인지 확인한다.
   - `nproc`과 RAM, Kaggle 할당량을 확인한다.
   - TPU Research Cloud에 신청한다.
2. **8주차:** 보고서를 검토하고 1B 트랙 여부를 결정한다.
3. **10주차까지:** 한국어 채팅을 평가할 자원봉사 평가자를 5–10명 모집한다(동의서 포함, §9.6).

---

## What changed from v1 (audit and critique items applied)

| Area | v1 | Final |
|---|---|---|
| Param total | 81,108,694 | **81,110,658**. Adds A_log, dt_bias and the per-head output norm (+138 per GDN layer, 14 layers) and the 32-param selection calibrator. expand_v=1 is pinned |
| Judge total | 16,353,284 (missed its input norm) | **16,354,978** |
| Pool exclusions | Not checked | All pools + 384-d embedding = 100,588,576 and all pools + static keys = 104,712,352. Both are **over the cap**, are declared mutually exclusive, and are asserted by CI |
| S1 / S2 mean r | "≈2" / "≈4" | **2.333** (uniform {1,2,4}) / **4.733** (p(r) ∝ 1/r on 1..16) |
| Training FLOP model | 6N only | 6N + deep-supervision coda pass + checkpoint recompute + delta-rule mixer (+11.7% per layer) |
| TPU plan | 127.8 h, 47% slack | **203.1 h = 10.16 TPU-weeks, 15% slack** (S1 30B kept; X3 and X4 resized) |
| T4-only fallback | "17 T4-weeks" | **37.6 T4×2-weeks** in total, lean plan included (§0.3) |
| Teacher work | 2.5 weeks | **144.1 h raw (4.8 T4×2-weeks), 216 h with retries**. Prefill at 20% MFU (15% for 7B). Adds a 7B adversarial filter and the S6 judge labelling |
| Missing budget rows | — | Eval panel, reference ladder, Korean proxy, controller labels, process-value labels, S6 samples and labels, a CPU-hour sheet, ARC laptop hours |
| Per-query cap | N=64 at r=16 implied (2.9e13, over) | Controller envelope: N=64 needs r ≤ 8; r=16 needs N ≤ 32; tokens per sample are truncated to fit. CI asserts every mode |
| Per-row pinned modes | Pinned (r, N, K) per row | **One global input-only policy**; pinning is banned (F10) |
| MC state forking | Contradicted per-option retrieval | **Fork after [shots + stem]**; per-option evidence follows the fork; question → evidence → question sandwich |
| X1 GO rule | slope ≥ 3 pts/doubling (a pre-registered loss) | **GO only if the extrapolated upper 95% bound at the M1 point clears the gate**; a Bradley–Terry reference ladder for chat |
| Cross-tokenizer distillation | Assumed to work | Boundary-aligned top-k distillation [HYPOTHESIS], measured in X1, with sequence-level fallback |
| Gates | HumanEval 83.8, MBPP 87.1, TriviaQA 86.4 | **83.3, 86.3, 86.5** |
| 10B wall | "≈7 years" | **24.3 years** at 12% MFU with r-overhead (16.9 years for plain 6ND) |
| Storage | 5.0 GB RAM for codes | **6.25 GB** (+1.25 GB of int64 ids); the store does not fit HF private storage, so it ships as a recipe plus hashes |
| New | — | WY/UT chunk equations; IR CFG; READ/JUDGE re-entrancy; hyperparameters; token-level data mix; gap ledger; NLL→accuracy link; sandbox hardening; release policy; Korean summary; day-1 checklist; harness wrapper |

---

## 0. CANONICAL NUMBERS SHEET

Every later section uses exactly these numbers. All come from `calc/alembic_budget.py`.

### 0.1 Parameter budget

Shared shape:
- d = 640, 5 heads × head_dim 128, **expand_v = 1 (pinned)**.
- SwiGLU hidden 1536, vocabulary 32,768, embedding bottleneck 256, GDN chunk 128.
- The NVlabs/FLA reference defaults to expand_v=2. Taking it literally would add 3d² ≈ 1.23M per layer, about +17M over the 14 layers [ESTIMATE].

One Gated DeltaNet (GDN) layer has **5,012,618** params:
- q, k, v, o projections, 4d² = 1,638,400
- output gate, d² = 409,600
- α/β gate projections, 2·d·H = 6,400
- short conv (k=4) on q, k, v, 3·d·4 = 7,680
- SwiGLU, 3·640·1536 = 2,949,120
- two pre-RMSNorms = 1,280
- A_log + dt_bias (per head), 2H = 10
- per-head output RMSNorm weight = 128

| # | Module | Composition | Trainable params |
|---|---|---|---|
| 1 | Unigram tokenizer scores (counted, F2; not a JAX leaf, CI adds it) | 32,768 scalars | 32,768 |
| 2 | Factorized token embedding, tied to the LM head | 32,768×256 + 256×640 | 8,552,448 |
| 3 | Prelude | 3 GDN layers (L1–L2 + proj = retriever key encoder) | 15,037,854 |
| 4 | Recurrent core (weight-tied, iterated r times per query) | 4 GDN layers, counted once | 20,050,472 |
| 5 | Loop-input injector | Linear(2d→d) + bias | 819,840 |
| 6 | Coda | 4 GDN layers | 20,050,472 |
| 7 | Final norm | 640 | 640 |
| 8 | Judge tower (right-to-left GDN scan over trunk states) | 3 GDN layers | 15,037,854 |
| 9 | Judge input norm | 2·640 | 1,280 |
| 10 | Judge fusion MLP, 4 heads (energy/verify, chat reward, process value, difficulty) | 1280→1024→4 | 1,315,844 |
| 11 | Retrieval query/key projection | 640×256 | 163,840 |
| 12 | Controller head (r-level 6, N-level 4, route 8) | 640→64→18 | 42,194 |
| 13 | Segment embeddings (prompt / chunk / tool / IR / answer / judge / think / pad) | 8×640 | 5,120 |
| 14 | Selection calibrator (logistic; 3 features + bias per route × 8 routes) | 32 | 32 |
| | **TOTAL, M1 base** | JAX leaves 81,077,890 + tokenizer 32,768 | **81,110,658** |
| | Headroom under 100,000,000 | | 18,889,342 |

The judge (rows 8–10) totals **16,354,978**. Its forward FLOP per scored token is **3.27e7**. The embedding share of the budget is **10.54%**.

**Pre-registered pools.** These are released only by a named experiment outcome:

| Pool | Content | Params | Released by |
|---|---|---|---|
| G (generator) | +2 coda layers | 10,025,236 | X1 chat: the extrapolated upper bound clears the gate (§10.2) |
| J (judge) | +1 judge layer | 5,012,618 | X1 commonsense: positive judge-depth slope with an upper bound clearing the gate |
| R (phrase query) | 640→256 projection for admission candidate A3 | 163,840 | A3 admitted |
| | **Worst case with all pools** | | **96,312,352** (3,687,648 left) |

**Compatibility table** (CI asserts these):

| Combination | Total | Status |
|---|---|---|
| All pools + 40k vocabulary (Korean contingency, +1,858,624) | 98,170,976 | Allowed (1,829,024 left) |
| All pools + 384-d embedding ablation (+4,276,224) | 100,588,576 | **Over the cap: mutually exclusive** |
| All pools + static key table arm (+8.4M) | 104,712,352 | **Over the cap: mutually exclusive with Pool G** |

**Active parameters, effective depth (3 + 4r + 4) and forward FLOP per generated token.** The FLOP figures include the chunkwise delta-rule mixer, which is 1.174M FLOP per layer per token, or 11.7% of the layer's matmuls [ESTIMATE].

| r | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| N_active | 64.5M | 85.4M | 127.1M | 210.6M | 377.6M | 711.5M |
| Effective depth (layers) | 11 | 15 | 23 | 39 | 71 | 135 |
| Forward FLOP/token (incl. mixer) | 1.42e8 | 1.88e8 | 2.81e8 | 4.67e8 | 8.38e8 | 1.58e9 |
| Matmul-only (2·N_active) | 1.29e8 | 1.71e8 | 2.54e8 | 4.21e8 | 7.55e8 | 1.42e9 |

Memory:
- **Recurrent state:** 81,920 floats = 327.7 KB per layer in fp32. Counting 3 + 4r + 4 unrolled layer states, that is 12.78 MB per sequence at r=8.
- **Training state:** fp32 master weights + Adam moments + gradients come to 1.30 GB.

**Scaling-family configs** for X1 and X5. They share the tokenizer, use H·dh = d and dh = 128, and are counted by the same function as M1:

| Config | d | H | ffn | e | pre/core/coda/judge | Total params | Non-embedding |
|---|---|---|---|---|---|---|---|
| S (~5M) | 256 | 2 | 640 | 64 | 1/1/1/1 | 5.79M | 3.65M |
| M (~10M) | 384 | 3 | 768 | 96 | 1/1/1/1 | 10.46M | 7.24M |
| L (~20M) | 512 | 4 | 1024 | 128 | 1/2/1/1 | 19.99M | 15.70M |
| XL (~40M) | 640 | 5 | 1536 | 128 | 2/2/2/1 | 41.42M | 37.11M |
| X3 generator (50M-class) | 640 | 5 | 1536 | 256 | 3/3/2/0 | 51.0M | — |
| X4 generator (35M-class) | 640 | 5 | 1536 | 128 | 2/2/2/0 | 36.4M | — |

At small sizes the embedding table is a large share of the total. Fits are therefore reported against both total and non-embedding params.

### 0.2 Datastore (gated track: human-authored or self-generated only)

| Item | M1 value |
|---|---|
| Store tokens | **20B** on the laptop SSD. Kaggle mounts a 10B subset |
| Chunks | 156.25M × 128 tokens |
| Dense keys in RAM | 256-bit binary codes 5.0 GB + FAISS int64 ids 1.25 GB = **6.25 GB**, IVF. 512-bit fallback: 11.25 GB |
| Rescoring | int8 256-d **40 GB** mmap on the laptop; PQ-32 **5.0 GB** (2.5 GB for the Kaggle 10B subset) |
| BM25 | Compressed on-disk inverted index over a **~6B-token core**: en+ko Wikipedia, ko-Wiktionary, open textbooks, Python docs. ≈ 2.5–10 GB including Korean character bigrams |
| Text | uint16 memmap, **40 GB raw on disk** (a memmap cannot be compressed). zstd ≈ 27–31 GB, for transfer only [ESTIMATE] |
| Kaggle mount (10B subset) | Raw text 20 GB + codes and ids 3.1 GB + PQ 2.5 GB + BM25 core 2.5–10 GB ≈ **28–36 GB** |
| RAM on Kaggle for the 10B subset | Codes + ids + PQ = **5.62 GB** of the ~29–32 GB session RAM [VERIFIED third-party: https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/main/kaggle_20251121.md] |
| Laptop footprint (20B) | 91.2 GB core + BM25 ≈ **94–101 GB**, plus build staging. **Day-1 check: free SSD ≥ 150 GB, else the store tier drops to 10B** |
| Key-encoder build | **2.07e7 FLOP/token** (embedding gather + L1–L2 + projection; no LM head). 20B tokens = **0.61 TPU-h** per build, or 13.1 h on the 3060. 2 builds are budgeted. I/O-bound: Kaggle mounts only 10B, so a 20B build runs as two 10B halves |
| Persistence | The store (~100 GB) **does not fit** HF private storage (100 GB [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits]) next to checkpoints and logits. It ships as a **deterministic build recipe + per-shard hashes**. Codes and ids (6.25 GB) and the 10B Kaggle subset are private datasets; licences permitting, a public dataset copy follows R0 |

### 0.3 Training compute (TPU v5e-8)

**Planning values [ESTIMATE].**
- Training MFU is 12% on TPU v5e-8. That gives 6.81e17 FLOP per TPU-hour and **1.36e19 FLOP per 20-hour TPU-week**.
- Batched decode on the TPU runs at 4% MFU, or 2.27e17 FLOP/h.
- T4×2 at 20% gives **2.81e18 FLOP per 30-hour week**.

**Per-token training FLOP** = 6·N_active(r̄) + a deep-supervision coda+head pass at ⌈r/2⌉ + recompute of each checkpointed core iteration + 3× mixer FLOPs.
- This is conservative: it charges full backprop through every iteration, although gradients reach only the last 2.
- S1 costs **8.23e8 FLOP/token**, so the TPU scans **16.5B S1 tokens per TPU-week**.

| Stage | Tokens | r | FLOP | TPU-h |
|---|---|---|---|---|
| S1 Language + reading | 30B scanned | uniform {1,2,4}, r̄=2.333 (N_active 92.3M) | 2.47e19 | 36.3 |
| S2 Procedures + IR/tool traces | 15B | p(r) ∝ 1/r on 1..16, r̄=4.733 (N_active 142.4M) | 1.95e19 | 28.6 |
| S3 Judge (trunk forward at r=2 + judge training) | 3B | 2 | 8.6e17 | 1.3 |
| S4 Instruction/chat SFT | 2B | uniform {1,2,4} | 1.65e18 | 2.4 |
| S5 RLVR (GRPO) policy updates | 1B | 4 | 1.17e18 | 1.7 |
| S6 DPO: 50k pairs × 1,200 tokens × 3 epochs = 0.18B, policy + reference forward | 0.18B | uniform {1,2,4} | 1.85e17 | 0.3 |
| **Main subtotal** | 51.2B trained | | **4.8e19** | **70.5** |

TPU batched-decode jobs (4% MFU):

| Job | FLOP | TPU-h |
|---|---|---|
| S5 single-shot rollouts, 1.6B tokens at r=4 (whole programs, executed on CPU afterwards) | 4.5e17 | 2.0 |
| Controller + difficulty labels: 20k problems × r ∈ {1,2,4,8,16} × 64 samples × 300 tokens | 7.36e17 | 3.2 |
| Process-value Monte-Carlo labels: 100k traces × 4 prefixes × 8 completions × 200 tokens, r=4 | 1.8e17 | 0.8 |
| S6 samples: 50k prompts × 4 × 400 tokens, r=4 | 2.25e16 | 0.1 |
| **Decode subtotal** | | **6.1** |

Experiments:

| Experiment | FLOP | TPU-h |
|---|---|---|
| X1 wall probes: S/M/L × (±aligned KD) × 2.5B tokens; judges on 0.59B tokens × (Open, Strict) | 2.08e18 | 3.1 |
| X2 recurrence: L config at r ∈ {1, 4, 16} + `<think>` arm at r=4, 2B each | 3.62e18 | 5.3 |
| X3 retrieval: 50M-class, 8 arms × 1.5B scanned | 7.37e18 | 10.8 |
| X4 IR vs CoT: 35M-class, 2 × 4B, p(r) ∝ 1/r | 5.45e18 | 8.0 |
| X5 grid S/M/L/XL at 100 tokens/param, seeds 3/3/2/1 | 3.13e18 | 4.6 |
| X5 ORRERY-0 control grid (r=1, no judge), 1 seed | 1.38e18 | 2.0 |
| **Experiments subtotal** | **2.3e19** | **33.8** |

**TPU total = (main + experiments) × 1.8 + decode × 1.8 + index (1.22 h, 2 builds) + X0 (3.0 h) = 203.1 TPU-h = 10.16 TPU-weeks** of the ~12 in the 12-week window. **Slack is 15%.**

The ×1.8 reserve for failures and restarts is held globally, not per stage. Weeks 7–12 carry about 40 h of spare (§10.1). A full S1 restart fits in that spare but pushes S6 into week 13.

**T4-only fallback, if the TPU is unavailable.** The lean plan cuts S1 to 12B and S2 to 8B, halves X3 and X4, runs X5 with one seed and drops the control grid.
- Training + experiments × 1.8 = 6.95e19 FLOP = **24.8 T4×2-weeks**.
- Decode jobs at 5% T4 MFU: **3.6 weeks**.
- The teacher/eval sheet: **9.3 weeks**.
- Total: **37.6 T4×2-weeks**, about 8.7 months of free T4 quota.
- The full (non-lean) plan on T4 alone would need 45.6 T4×2-weeks for training and experiments.
- R0 then slips from week 12 to about week 40 [ESTIMATE]. This is stated plainly and not hidden. The mitigation is the TRC application in week 1 (§7.2).

**Tokens per parameter.** 51.2B trained tokens over 81.1M params is about 631 tokens/param, about 32× Chinchilla [ESTIMATE]. Pretraining FLOPs are not the binding constraint, since the window could scan about 198B S1-type tokens. The binding constraints are:
- data density;
- teacher throughput;
- CPU-bound tools;
- evaluation.

### 0.4 T4×2 sheet (weeks 2–12, 330 h) and CPU sheet

Planning values:
- Teacher prefill MFU is 20% for a 1.5B fp16 teacher and 15% for a 7B teacher (AWQ or TP=2; dequantization overhead) [ESTIMATE].
- Teacher decode for the 1.5B is 0.3B tokens/week [ESTIMATE, compute digest range 0.3–0.65B].

| T4×2 job | Hours |
|---|---|
| Teacher decode: OMI-2 → IR rewrite, 0.3B (1.5B teacher) | 30.0 |
| Teacher decode: plan/section chat traces en/ko, 0.2B | 20.0 |
| Teacher decode: Korean procedures/instructions, 0.1B | 10.0 |
| Teacher prefill: judge ranking, 1.5M items × 300 tokens (context prefix shared across options), 1.5B | 14.4 |
| Teacher prefill: adversarial filter, 0.3M items × 500 tokens (8 candidate corruptions), 7B | 32.5 |
| Teacher prefill: aligned top-16 logits, 0.3B + 0.05B X1 mini-subset, 1.5B | 11.2 |
| Family-A judge: S6 labels, 50k pairs × 2 orders × 1,200 tokens, 7B | 26.0 |
| **Teacher subtotal: 144.1 h = 4.80 T4×2-weeks; × 1.5 retries** | **216.2** |
| Eval panel: X1 chat arm (6 variants, ≈ 965 judged pairs each × 2 judges × 2 orders × 1,500 tokens) | 8.4 |
| Eval panel: R0 pooled 1,465 pairs × 2 × 2 + ρ ≥ 0.8 calibration | 2.7 |
| Reference-ladder answers (3 models × 965 × 500 tokens) + Korean proxy answers | 3.0 |
| S5 multi-turn tool rollouts (code repair, search hops), 0.4B tokens at r=4 | 4.8 |
| X0 parity + vLLM throughput benchmark | 4.0 |
| S1/S2 checkpoint dev evals | 10.0 |
| R0 full Tier A+B, neural part, with seeds | 30.0 |
| **T4 total** | **279.1 of 330 h (15% slack)** |

Using the 7B teacher for **all** 1.5M ranking items would cost 97 h instead of 14.4 h. That is why only the hardest 0.3M items get the 7B (§6.4).

**CPU sheet [ESTIMATE].** The mean execution cost is 0.15 s per sandboxed program, including timeouts.

| CPU job | CPU-hours |
|---|---|
| S5 RLVR executions (2.0B rollout tokens / 250 = 8M programs) | 333 |
| Controller-label executions (6.4M) | 267 |
| Process-value executions (3.2M) | 133 |
| Decontamination (13-gram, MinHash, Korean 20-gram) over ~70B tokens | 120 |
| Store build: tokenize 20B + BM25 over the 6B core + IVF training | 80 |
| Cross-tokenizer logit alignment, 0.35B tokens | 20 |
| Monthly full-eval tool calls (MATH's ~1M calls dominate) | 60 |
| **Total** | **≈ 1,013 CPU-h** |
| ARC-AGI-1 monthly job (400 tasks × 2–10 min) | **13–67 laptop-hours per month** |

**Where the CPU comes from.** The laptop core count is unknown until day 1 (`nproc`), and Kaggle GPU sessions have about 4 vCPU [ESTIMATE]. TPU VM hosts may expose many more vCPUs [ESTIMATE: check with `nproc` in the first TPU session]. If they do, the S5 and controller-label executions run there.

At 12 usable cores the S5 executions take about 28 wall-hours. If the day-1 core count is below 8, S5 rollouts drop to 1.0B tokens and the controller-label set drops to 10k problems.

### 0.5 Targets (maximum reported GPT-3.5) and gates (target + 95% CI half-width)

| Row | Target | Gate | M1 estimate | P(win at M1) |
|---|---|---|---|---|
| NQ-open EM | 38.6 | 40.2 | 38–48 | 0.55 |
| TriviaQA (harness `triviaqa`, 5-shot EM) | 85.8 | **86.5** | 65–78 | 0.05 |
| MMLU 5-shot | 71.4 | 72.1 | 42–55 | 0.02 |
| KMMLU 5-shot direct | 42.47 | 43.0 | 38–48 | 0.35 |
| HAE-RAE (5 harness subsets; conservative superset target, §2) | 55.4 | 57.9 | 46–58 | 0.20 |
| KoBEST avg | 79.3 | 80.8 | 62–76 | 0.10 |
| GSM8K 0-shot | 78.1 | 80.3 | 60–76 | 0.15 |
| MATH 4-shot | 34.1 | 35.4 | 22–36 | 0.25 |
| BBH 0-shot | 68.3 | 69.4 | 55–70 | 0.30 |
| DROP F1 | 64.1 | 65.1 | 56–70 | 0.40 |
| HumanEval+ / HumanEval | 70.7 / 76.8 | 77.7 / **83.3** | 35–58 | 0.03 |
| MBPP+ / MBPP | 69.7 / 82.5 | 74.3 / **86.3** | 45–63 | 0.03 |
| ARC-Challenge | 87.4 | 89.3 | 50–68 | 0.03 |
| HellaSwag 10-shot | 85.5 | 86.2 | 48–68 | 0.03 |
| WinoGrande 5-shot | 81.6 | 83.7 | 56–70 | 0.02 |
| ARC-AGI-1 pass@2 (400 eval; protocol-mismatched conservative target, §2) | 10.6 | 13.6 | 10–25 | 0.55 |
| Chat pooled pairwise vs stored GPT-3.5 (N ≈ 1,465) | 50% | 52.6% | 10–28% | 0.02 |
| Korean MT-Bench-style pairwise vs open proxy (Arena-Ko ≥ 1049.6) | 50% | 50% + CI | 8–25% | 0.02 |

**P(every gated row at M1) < 0.1% [ESTIMATE].** That is the honest headline. The plan exists to turn it into a measured campaign (§1 F13, §8, §10).

### 0.6 Inference cost per query

The cap is **2e13 neural FLOP per query**, including the mixer. ARC-AGI is excepted and disclosed.

| Query type | Work | FLOP | Wall-clock [ESTIMATE] |
|---|---|---|---|
| Factoid (NQ) | sandwich: question 40 + 8 chunks × 128 + question 40, r=2, 20 decode tokens | 2.12e11 | 0.3–1 s (dominated by retrieval: IVF Hamming + ~1,000 SSD rescoring reads) |
| MC (MMLU-style), 4 options | shared [5 shots + stem] 600 tokens, **fork**; per option 4 × 128 evidence + stem 100 + option 20, r=2, plus judge | 6.72e11 | 0.5–1.5 s |
| GSM8K | 32 IR programs × 250 tokens at r=4 + judge verify 32 × 400 + final 400 | 2.78e12 | 3–8 s (CPU-bound) |
| Chat turn | plan + 8 drafts × 400 at r=4, judge, one 600-token revise, 1,500 prompt/evidence tokens at r=2 | 1.46e12 | 3–8 s |
| HumanEval | 16 candidates × 400 tokens × 4 rounds at r=8, plus judge | 1.28e13 | 15–60 s |
| MATH hard | 64 × 600 at r=8 (or 16 × 600 at r=16 = 8.05e12) | 1.79e13 | 20–90 s |
| ARC-AGI-1 task (input-type route "2-D integer grids with demo pairs") | DSL program search + exact verification on demo pairs | ~1e15 | 2–10 min (laptop monthly job) |

**Controller envelope.** The controller enforces N · T · FLOP(r) ≤ 2e13 and truncates T when needed. CI checks every mode:

| Mode | N=1, r=1, T=2000 | N=16, r=4 | N=64, r=4 | N=64, r=8 | N=16, r=16 | N=32, r=16 | N=8, r=32 |
|---|---|---|---|---|---|---|---|
| FLOP (T=600 unless noted) | 2.84e11 | 2.7e12 | 1.08e13 | 1.79e13 | 8.05e12 | 1.61e13 | 7.59e12 |

N=64 at r=16 is not a mode: at that setting the cap allows only 373 tokens per sample.

**Why 2e13 [ESTIMATE].** GPT-3.5's size is undisclosed. If it were a 20B dense model, one 1,000-token query would cost 2·20e9·1,000 = 4e13 FLOP, and our cap would be 0.50× of that. If it were 175B, the query would cost 3.5e14 and our cap would be 0.06×. Under any public size guess, ALEMBIC spends no more neural compute per query than GPT-3.5 itself.

**Tool limits:** ≤ 10 tool calls per program (nested neural calls included, §5.4), ≤ 200 executions per query, ≤ 60 s CPU per query (ARC excepted). FLOPs and wall-clock are reported per row.

---
## 1. Rules of the game

These are binding and pre-registered. They change only by a dated, public amendment made **before** any affected eval is run.

**F1. The cap.** Fewer than 100,000,000 trainable parameters executed at inference.
- **Count method.** The count is taken on the *instantiated* JAX module plus the non-JAX fitted values, by a CI assertion: `sum(p.size for p in leaves(params)) + tokenizer_scores < 100e6`. It currently returns 81,077,890 + 32,768 = **81,110,658**.
- **The count includes:**
  - embeddings, every trunk layer, the tied core counted once, the judge, every head and projection;
  - the selection calibrator and the tokenizer's learned scores;
  - the initial weights of any test-time memory, if one is ever admitted.
- **Source of truth.** `tests/test_param_count.py` **derives** the expected count from `config.py` through the same function the budget script uses (`model_params`). It does not hard-code a constant.

**F2. Count vs fit** (LIBRARIAN's rule, made crisp):
- **Counted toward the cap:** every value fitted by gradient or EM that is executed on the query path at inference. That means network weights, tokenizer scores, the selection calibrator and learned projections.
- **Disclosed in bytes, not counted.** Three kinds:
  - **(i)** Statistics produced only by counting or indexing a frozen store: BM25 statistics, inverted indexes, suffix automata, the alias dictionary.
  - **(ii)** Compression codebooks of the store: IVF centroids, PQ codebooks, binarization thresholds. These are capped at **64 MB in total**. They quantize stored vectors and can only approximate what the store already holds [ESTIMATE judgement].
  - **(iii)** Scalar hyperparameters such as the copy threshold τ.
- **Not counted but disclosed** with name, size, family and licence: training-only weights (any EMA target, auxiliary probes) and all teacher and judge models.

**F3. Gated store = human-authored or self-generated.** Every inference store holds only two kinds of content. The stores are the datastore, the BM25 core, the exemplar bank, the skill library and any later phrase index.
- **(a)** Pre-existing human-authored public text, code or human-curated structured data. Web crawls are limited to Common Crawl snapshots from **2022 or earlier**. FineWeb-Edu's per-crawl subsets make this mechanical [VERIFIED: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu]. The assumption that pre-2023 crawls hold little LLM text stays [ESTIMATE].
- **(b)** Artifacts produced by ALEMBIC itself that passed an executable check.

Further rules on the stores:
- **Teacher text never enters any inference store.** That covers Cosmopedia (Mixtral-generated [VERIFIED: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia]), OpenMathInstruct-2 solutions, Magpie-style data and regenerated chat. Retrieving it would be a lookup-table copy of a >100M network.
- Teacher-written code is banned from the skill library.
- Every vector in every index is produced by the in-budget encoder.
- A labelled "synthetic-store" track may be reported. It never gates.

**F4. Allowed at inference.** Everything runs on CPU with 0 parameters.
- The frozen, dated, hashed offline store.
- An **allow-listed, hardened sandbox image** (§9.6): Python stdlib, NumPy, SymPy, NetworkX, plus Z3 once admitted. No ML libraries, no weight files, no network. The image hash is published.
- A deterministic test runner for tests the model wrote itself, including tests extracted from I/O examples in the prompt.
- Hodel's ARC DSL as library text [VERIFIED: https://arxiv.org/pdf/2404.07353].

**F5. Banned at inference:**
- any neural network over 100M, or **any learned model outside the counted budget, of any size**. That includes language-ID models (fastText-style), safety classifiers and rerankers. Language detection is a **rule-based Unicode-script ratio** (Hangul U+AC00–U+D7A3 and Jamo vs Latin vs other);
- LLM APIs and live web search;
- eval items or their solutions in any store;
- running hidden benchmark tests during generation;
- routing, prompts, heads or budgets keyed on dataset identity;
- "abstain" in eval mode.

*Input-type* routing is allowed. Examples: "the input is a 2-D integer grid with demonstration pairs" or "the input contains a Python function signature". The detector's code is published, and its decisions are logged per row.

**F6. Architecture rule, pointer-not-mixer:**
- No softmax (or Hopfield-equivalent [VERIFIED: https://arxiv.org/abs/2008.02217]) normalization over token positions or over retrieved items, anywhere in the inference forward pass.
- A softmax may appear **only as an output distribution over discrete choices**: vocabulary, r-level, route, or an argmax over a few candidate answers.
- A chosen item re-enters the model only as discrete tokens.
- Linear-attention / delta-rule recurrences with a constant-size state are explicitly permitted.
- Softmax inside *training losses* (cross-entropy, InfoNCE over candidates, listwise ranking) is a loss, not a token mixer, and is permitted.

**F7. Training data:**
- Open-licensed teachers only, each logged.
- **No GPT-3.5/4 output anywhere.** OpenAI's terms forbid using output "to develop models that compete" [VERIFIED: https://openai.com/en-GB/policies/terms-of-use/], and it would contaminate the head-to-head. This excludes TinyGSM, MetaMathQA, the original UltraChat and TinyStories from weights training. They are cited as evidence only.
- Every SmolTalk and Tulu-3 subset card is audited. Subsets with OpenAI-generated or unclear responses are dropped.
- NC subsets go in a manifest so a commercial fork can drop them. One example is No Robots in Tulu-3 [VERIFIED: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture/blob/main/README.md].
- A **no-teacher track** is reported.

**F8. Benchmark train splits and eval-task material:**
- **Open track (gates).** Official *train* splits (NQ, TriviaQA, GSM8K, MATH, MBPP-train, DROP, HellaSwag, WinoGrande, ARC, KoBEST, KILT) may enter **one shared training mix** for **one checkpoint**. Their use is disclosed.
- **Strict track.** A track with zero multiple-choice benchmark train splits is trained and reported for every row (ORRERY's discipline).
- Any row whose win holds on Open but not on Strict is flagged "win depends on train split" in every publication.
- Test and validation splits are never trained on.
- **ARC-AGI eval tasks.**
  - The demonstration pairs of an eval task are used at inference **only as the exact verifier** of candidate programs.
  - Training on eval-task demonstrations (test-time fine-tuning, A6) is reported only as a separate row, "ARC-AGI-1 (TTT)", which never gates.
  - TRM's 44.6 relies on augmentation and training that include the eval tasks' demonstration pairs [ESTIMATE: as we read the paper, https://arxiv.org/abs/2510.04871; re-check]. It is therefore compared against the TTT row, not the gated row.

**F9. Decontamination** of training data **and every store**, against every eval set and its mirrors:
1. 13-gram + MinHash near-duplicate filtering.
2. An embedding scrub at cosine > 0.9 with our own key encoder, because n-grams miss rephrasings [VERIFIED: https://arxiv.org/pdf/2311.04850v1]. **Ordering:** the key encoder exists only after S1. So the embedding scrub runs on the store and on S2–S6 data. S1 data is scrubbed with steps 1 and 3–6 before training and **audited post hoc** with the encoder. The flagged share is published, and if it exceeds 0.1% of S1 tokens, the affected rows carry a contamination flag.
3. **Korean character 20-grams**, because word 13-grams miss short Korean multiple-choice stems.
4. **Function signature + docstring matching** for code.
5. **Whole-source blocklists:** wikihow.com and ActivityNet captions (HellaSwag's sources); Korean licence-exam and quiz sites (KMMLU/HAE-RAE sources); benchmark-hosting sites; GitHub mirrors of HumanEval/MBPP.
6. An **exam-format filter**: drop documents with ≥ 3 numbered option blocks (①②③④ or (A)–(D)) followed by an answer key [ESTIMATE heuristic].
7. A **per-item retrieval audit**:
   - generative QA: the % of top-k chunks containing the gold string;
   - multiple choice: the maximum n-gram overlap of the stem and of each option with the retrieved chunks;
   - continuation tasks: "gold ending ⊂ top-k" must be ≈ 0% before any HellaSwag number is published;
   - any item whose stem overlaps a retrieved chunk by > 50% of characters is flagged, and scores are reported with and without flagged items.
8. A **private post-cutoff set** (fresh math, code, Korean and chat prompts written after the store snapshot) is the headline sanity number.
9. Every store-dependent row is reported **with and without the store**.
10. The procedural-generator manifest is frozen and hashed before generator design. **Five BBH task families are drawn by a seeded random draw and committed as held out**; no generator may target them.

**F10. Scoring and configuration:**
- **Generative everywhere.** Multiple choice is answered by an emitted letter, which the system's own judge may choose, because the judge is part of the system. Log-likelihood numbers are secondary.
- Shot counts and answer-extraction rules match the target row's protocol. These are the only per-row settings, and they are defined by the benchmark, not by us.
- **One checkpoint and one global inference policy serve every row.**
  - The controller head picks the budget (r, N, K, tool budget) **from the input alone**, under one published global cap (2e13; ARC's 1e15 applies only to the input-type grid route, F5).
  - **Nothing is pinned per row.**
  - Chosen budgets are logged per item and summarized per row.
  - A "pinned-mode" ablation may be reported, labelled as benchmark-specific hyperparameters. It never gates.
- 3 seeds for sampled rows.
- **Win = score ≥ target + 95% CI half-width**, with matched metrics (EM vs EM). Paired bootstrap is used where stored GPT-3.5 per-item outputs exist, otherwise a binomial CI.

**F11. Judging chat:**
- **Families.** The training-time judge comes from family A. The eval panel has ≥ 2 judges from families B and C. The families are disjoint.
  - Family A (training judge and teachers): **Qwen2.5** (1.5B and 7B Instruct; licence permission to train on outputs checked per model card on day 1 [ESTIMATE that they are Apache-2.0]).
  - Eval panel: family B **Llama-3.1-8B-Instruct** and family C **Gemma-2-9B-it**, both 4-bit on a T4 [ESTIMATE that they fit; Mistral-7B-Instruct is the substitute].
- Both answer orders, length-controlled variants only.
- **Calibration first.** The panel must reach Spearman ρ ≥ 0.8 against published GPT-4-judge MT-Bench scores of ≥ 5 models whose stored answers exist [VERIFIED stored answers: https://huggingface.co/spaces/lmsys/mt-bench]. **If the panel misses ρ ≥ 0.8, no chat number is published.**
- Plus a 100-pair blind hand check by the author, and consenting Korean volunteer raters for the Korean gate (§9.6).
- **Eval judges are never used as a reward or a training filter.**

**F12. Compute disclosure.**
- Per row: mean FLOPs, wall-clock, candidates, r, tool calls and retrieval provenance, following the FLOPs-matched framing of Snell et al. [VERIFIED: https://arxiv.org/abs/2408.03314].
- The per-query neural cap is 2e13 FLOP. ARC-AGI is disclosed separately at ~1e15 per task.

**F13. Win-or-extend** (from OUROBOROS):
- The cap, the rules and the one-checkpoint requirement never move.
- No victory is declared until **every** gated row (Tier A + B) clears its gate on one checkpoint.
- Each round grows only the axes that are free under the cap: the store, test-time compute within the cap, tools, verified self-play data.
- Every round publishes the full scoreboard, losses included.
- If the measured slopes say a row cannot be reached under 100M, that is published with the curve, and the decision about the 1B track goes to the user. **The boss is never quietly redefined.**

**F14. Biosafety hard stop** (§9.5). It applies in every round and is never traded for capability.

---

## 2. The GPT-3.5 scoreboard

"GPT-3.5" is at least four API snapshots (0301/0314, 0613, 1106, 0125). Published numbers vary by up to 20 points depending on protocol. HumanEval is 48.1 in the GPT-4 report but 76.8 on EvalPlus; GSM8K is 57.1 vs 78.1 [VERIFIED: https://arxiv.org/abs/2303.08774, https://evalplus.github.io/leaderboard.html, https://arxiv.org/abs/2404.14219].

**Rule: the target is the maximum reported score per row, across snapshots, sources and prompt languages.** We fight the strongest GPT-3.5 on record. We cannot re-measure it (0 KRW), so the targets are published numbers, and chat is compared head-to-head against stored public GPT-3.5 outputs.

### Tier A: gated academic rows (EleutherAI lm-evaluation-harness task names [VERIFIED: https://github.com/EleutherAI/lm-evaluation-harness])

The harness version is pinned by commit hash on day 1. That pins splits and prompts.

| Domain | Row (harness task) | Target | Source of target | N | Gate |
|---|---|---|---|---|---|
| Knowledge | MMLU 5-shot (`mmlu_generative`) | 71.4 (1106) | [VERIFIED: https://arxiv.org/abs/2404.14219] | 14,042 | 72.1 |
| Korean knowledge | KMMLU 5-shot direct (`kmmlu_direct`) | 42.47 | [VERIFIED: https://arxiv.org/abs/2402.11548] | ~35k | 43.0 |
| Korean culture | HAE-RAE 0-shot (`haerae`: gk, hi, lw, rw, sn) | 55.4: see the note below | [VERIFIED: https://arxiv.org/abs/2309.02706] | ~1.5k [ESTIMATE] | 57.9 |
| Korean NLU | KoBEST 0-shot (`kobest_*`) | 79.3 (En-prompt average) | [VERIFIED: https://arxiv.org/abs/2309.02706] | — | 80.8 [ESTIMATE CI] |
| Math | GSM8K 0-shot CoT | 78.1 (1106) | [VERIFIED: https://arxiv.org/abs/2404.14219] | 1,319 | 80.3 |
| Math | MATH 4-shot (`minerva_math_*`) | 34.1 | [VERIFIED: https://arxiv.org/abs/2312.11805] | 5,000 | 35.4 |
| Code | HumanEval+ / HumanEval | 70.7 / 76.8 | [VERIFIED: https://evalplus.github.io/leaderboard.html] | 164 | 77.7 / 83.3 |
| Code | MBPP+ / MBPP | 69.7 / 82.5 | same | 378 | 74.3 / 86.3 |
| Reasoning | BBH 0-shot | 68.3 | [VERIFIED: https://arxiv.org/abs/2404.14219] | 6,511 | 69.4 |
| Reading + arithmetic | DROP 3-shot F1 | 64.1 | [VERIFIED: https://arxiv.org/abs/2303.08774] | 9,536 | 65.1 |
| Commonsense | HellaSwag 10-shot | 85.5 | same | 10,042 | 86.2 |
| Commonsense | WinoGrande 5-shot | 81.6 | same | 1,267 | 83.7 |
| Science | ARC-Challenge 10-shot | 87.4 (1106) | [VERIFIED: https://arxiv.org/abs/2404.14219] | 1,172 | 89.3 |
| Open-domain QA | TriviaQA 5-shot (`triviaqa`, rc.nocontext, harness default split) | 85.8 (1106) | same | ~11k [ESTIMATE] | 86.5 |
| Open-domain QA | NQ-open 0-shot | 38.6 (1106) | [VERIFIED: https://arxiv.org/abs/2407.02485] | 3,610 | 40.2 |
| Abstraction | ARC-AGI-1 public eval, pass@2 | 10.6: see the note below | [VERIFIED: https://arxiv.org/abs/2307.04721] | 400 | 13.6 |

**Protocol notes. These are mismatches, labelled rather than hidden.**

- **TriviaQA.** We do not know which split Phi-3 used for 85.8. We score the harness default (`rc.nocontext`) and label the match [ESTIMATE].
  - Other published GPT-3.5 numbers are lower: 82.9 EM 0-shot, and 70.1 EM on the DPR split [VERIFIED: https://arxiv.org/abs/2407.02485].
  - By the max rule, the 85.8 target stands.
- **HAE-RAE.**
  - The 55.4 headline is the **En-prompt mean over 6 subsets including RC**. The harness `haerae` task has 5 subsets.
  - The Ko-prompt per-subset values for the 5 harness subsets average 49.45 [ESTIMATE, recomputed from the paper's Ko-prompt Table 14 values: LW 62.13, SN 55.56, RW 63.46, HI 30.32, GK 35.80].
  - **Day-1 task:** extract the En-prompt per-subset values from arXiv 2309.02706 and gate on their 5-subset mean, with the prompt language matched.
  - Until then the gate stays at the conservative 55.4 + CI = 57.9, labelled "mismatched superset". It can only become easier, never harder, once matched.
- **ARC-AGI-1.**
  - The target is the strictest GPT-3.5-family number on record: text-davinci-003, 85/800 on train + eval tasks, out of the box.
  - Our row is pass@2 on the 400 eval tasks, so the protocols differ.
  - 10.6 is kept because it is higher than the baseline digest's 5.0 GPT-4o upper bound [VERIFIED: https://arcprize.org/blog/oai-o3-pub-breakthrough]. It is labelled as protocol-mismatched.

### Tier B: gated chat (zero-budget protocol)

| Row | GPT-3.5 reference | ALEMBIC protocol |
|---|---|---|
| **Pooled chat gate** | MT-Bench 8.39 (0613) [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard]; AlpacaEval 2 LC 22.35 (0613) [VERIFIED: https://github.com/tatsu-lab/alpaca_eval]; Arena-Hard v0.1 24.8 (0613) [VERIFIED: https://github.com/lmarena/arena-hard-auto/blob/main/misc/past_leaderboards.md] | Pairwise vs stored GPT-3.5 answers on MT-Bench (160 turns) + AlpacaEval (805) + Arena-Hard (500). N ≈ 1,465, CI ≈ ±2.6 (AlpacaEval alone ±3.45). **Gate: pooled length-controlled win rate ≥ 52.6%**, both orders, F11 panel. Per-benchmark numbers are also reported |
| **Korean chat gate** | Arena Korean 1049.6 (0125) [VERIFIED: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard] | Stored GPT-3.5 Korean chat outputs have not been found. Korean MT-Bench-style prompts are compared pairwise against an **open proxy model with an Arena-Korean rating ≥ 1049.6**, read off the leaderboard on day 1. Blind, consenting volunteer Korean raters, both orders. Gate: win rate > 50% + CI. The Arena-Korean and overall ratings are Bradley–Terry scores fitted on different vote pools and are not on a common scale, so we make **no** claim that Korean is a relatively weak GPT-3.5 skill |
| English proxy (secondary) | Gemma-2-2B-it Arena 1163.8 > gpt-3.5-0125's 1132.3 [VERIFIED: same leaderboard] | Pairwise on fresh prompts. Transitivity is a [HYPOTHESIS], so this is reported but not gated |
| Chatbot Arena listing | 1132 overall / 1226 style-controlled | Stretch goal: an LMArena listing as the real arbiter. A model this small runs on CPU [ESTIMATE] |

### Tier C: reported, not gating (anti-overfit and north star)

- GPQA 29.0 and MedQA 63.4 [VERIFIED: https://arxiv.org/abs/2404.14219].
- Natural2Code-style fresh code: GPT-3.5 scores 62.3 [VERIFIED: https://arxiv.org/abs/2312.11805].
- ARC-AGI-2: GPT-3.5 ≈ 0 [ESTIMATE]; TRM 7.8 [VERIFIED: https://arxiv.org/abs/2510.04871]. Also the "ARC-AGI-1 (TTT)" row (F8).
- The private post-cutoff set.
- The no-store, no-teacher, Strict (no-split) and pinned-mode tracks.
- KMMLU-Hard, reported only, because it was built from items GPT-3.5 failed [VERIFIED: https://arxiv.org/abs/2402.11548].
- **North-star progress metrics (§9.3)**, which are not benchmark rows.

---

## 3. First-principles decomposition: where a dense LLM spends parameters, and where ALEMBIC moves each job

| Job | How a dense LLM pays for it | Evidence that it is expensive | ALEMBIC's move | What still lives in weights |
|---|---|---|---|---|
| **1. Long-tail facts** | About 2 bits/param of knowledge [VERIFIED: https://arxiv.org/html/2404.05405], so 100M ≈ 25 MB of facts [ESTIMATE] | SmolLM2-135M gets 4.1 on TriviaQA after 2T tokens [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct] | **Human-authored 20B-token store** + BM25 + binary dense keys + per-option retrieval + suffix-automaton copy. RETRO matched GPT-3 on the Pile with ~25× fewer params [VERIFIED: https://arxiv.org/abs/2112.04426]. FiD with 440M total reached 48.2 NQ EM [VERIFIED: https://arxiv.org/abs/2007.01282] | A **concept skeleton**: frequent entities, taxonomy, "what kind of thing is this", so the trunk can write good queries (LIBRARIAN). Long-tail fact dropout keeps rare facts out of the weights (§6.3) |
| **2. Exact computation** | Emulated digit carrying and variable tracking | GSM8K 1.4 at 135M [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M] | **Typed program IR** → Python/SymPy/NetworkX sandbox. PAL beat PaLM-540B CoT by 15 points [VERIFIED: https://arxiv.org/abs/2211.10435] | The *translation* from language to program (autoformalization). This is the irreducible core skill |
| **3. Serial reasoning depth** | One layer = one step, so depth costs params | TRM (7M) reaches 44.6% on ARC-AGI-1 by recursion, with the eval-demo caveat in F8 [VERIFIED: https://arxiv.org/abs/2510.04871]. Huginn (3.5B) scales test-time recurrence to a compute load equal to that of a ~50B-parameter model. That is a **FLOP equivalence, not a capability equivalence** [VERIFIED: https://arxiv.org/abs/2502.05171] | **Weight-tied 4-layer core iterated r ∈ {1…32} per query** + learned `<think>` tokens. Effective depth goes from 11 to 135 layers with no new params | The 4 tied layers: a reusable "step" procedure |
| **4. "Being right"** | Implicit in the likelihood | TinyGSM: "the verifier seems to play a bigger role than generation model size" [VERIFIED: https://arxiv.org/html/2312.09241] | **Exact verifiers first** (execution, SymPy, self-tests, ARC demo pairs); then the **judge tower** where no exact verifier exists; sampling N under a compute-optimal controller [VERIFIED: https://arxiv.org/abs/2408.03314] | The 16.35M judge, which reads the whole trunk (§5.6) |
| **5. Plausibility / commonsense** | An implicit world model inside the generator | WinoGrande is 51–53 at every size ≤ 270M: SmolLM2-135M 51.3 [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct]; Gemma-3-270M 52.0 [VERIFIED: https://huggingface.co/google/gemma-3-270m] | **Discrimination, not generation**: per-option state forks scored by a bidirectional judge trained on gold-labelled minimal pairs, teacher rankings and adversarially filtered corruptions | Judge + trunk. **Still parameter-bound**; the hardest wall (§8.3) |
| **6. Surface fluency** | Memorized phrasing in almost every output token | MT-Bench is 1.98 for SmolLM2-135M-Instruct [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct] and 2.33–2.52 for MobileLLM at 125–350M [VERIFIED: https://arxiv.org/html/2402.14905] | Shrink the generator's job: **plan → section-by-section realize → copy exact spans via suffix automaton → critique → best-of-N under judge reward**. Boundary-aligned top-k logit distillation from an open teacher [HYPOTHESIS; the recipe's evidence is Qwen3's same-tokenizer strong-to-weak distillation, https://arxiv.org/html/2505.09388]. The phrase index is an admission candidate (A3) | Trunk generator capacity, plus Pool G if earned. **Also parameter-bound; the other hardest wall** |
| **7. Instruction format** | Learned from RLHF data | IFEval rises 29.9 → 51.2 → 59.2 from 135M to 270M to 0.6B [VERIFIED: https://huggingface.co/HuggingFaceTB/SmolLM2-135M, https://huggingface.co/google/gemma-3-270m, https://arxiv.org/html/2505.09388] | **Format constraints compiled into rule-based executable checks** (word count, bullet count, Unicode-script language check, JSON validity), enforced at decode time with repair | Small |

**What remains in the 81.1M weights:** a compiler (language to typed program and back), a reader (scan evidence, copy spans), a reusable recurrent step, and a judge. Everything else is a disk, CPU or FLOP problem, and those are the resources a 0-KRW builder can grow.

The judges agreed on one honest corollary: **jobs 5 and 6 cannot be fully externalized**, because no exact verifier exists for "is this ending plausible" or "is this answer good". They are where the fight is decided. §8.3 states what is known, what is bet, and how each bet is killed early.

---
## 4. Architecture overview

### 4.1 Diagram

```
                ┌────────────────────── OFFLINE, 0 params (data, disclosed, hashed) ─────────────────────┐
                │ STORE 20B tok human-authored (pre-2023 crawl cut): en/ko Wikipedia, ko-Wiktionary,       │
                │   FineWeb-Edu(≤2022), FineWeb-2-ko(≤2022, if dated), OpenWebMath/arXiv (no q-bio),       │
                │   Python-Edu/Stack-permissive, StackExchange-style human Q&A, open textbooks             │
                │   dense: 156M chunks × 256-bit codes + ids (IVF, 6.25 GB RAM) → int8 / PQ-32 rescore     │
                │   BM25: compressed on-disk index over ~6B-token core (Wikipedia, Wiktionary, textbooks)   │
                │ SKILL LIBRARY (text): human OSS code + self-play verified IR fragments, frozen per eval   │
                │ SANDBOX (hardened, hashed): stdlib · numpy · sympy · networkx · test runner · ARC DSL    │
                └──────▲────────────────────────────────────────────▲─────────────────────▲───────────────┘
                       │ SEARCH(q) → RRF top-K (K=4..32)             │ PY/SOLVE/CHECK       │ suffix automaton
                       │ (S1: BM25-only bootstrap)                   │ async CPU pool       │ over retrieved
  prompt ─► TOKENIZER (32k unigram, ko pre-split rules) ─► EMB (32768×256 → 640, tied) + SEG emb   │ text + prompt
                       │                                              │                      │
            ┌──────────┴──────────── TRUNK: causal Gated DeltaNet, d=640, 5 heads×128 ──────┴────────────┐
            │ PRELUDE 3 layers  (L1–L2 + 640→256 proj = retriever key/query encoder, frozen after S1)   │
            │ CORE 4 weight-tied layers × r   (r chosen per query by controller; injector each loop)    │
            │ CODA 4 layers → tied LM head                                                              │
            │ mode tokens (0 params): <PLAN> <READ> <IR> <REALIZE> <JUDGE> <think> <copy> <search> ...  │
            │ input order is a SANDWICH: question → evidence → question (read-twice idea, §4.2)         │
            │ MC: state FORKED after [shots+stem]; option-specific evidence follows each fork            │
            └───────┬──────────────────────┬─────────────────────────────┬───────────────────────────────┘
                    │ IR program tokens     │ answer / prose tokens        │ per-token states h_1..h_T
                    ▼ (CFG-constrained)     ▼                              ▼
          ┌──────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────────────────────┐
          │ EXECUTORS (CPU)  │  │ CONTROLLER (code, 0 par.)│  │ JUDGE TOWER: 3 GDN layers scanning h     │
          │ run whole program│  │ global budget policy     │  │ RIGHT-TO-LEFT + fusion MLP → 4 heads:    │
          │ READ/JUDGE are   │  │ (r,N,K) from input only, │  │ energy/verify · chat reward ·            │
          │ host callbacks   │  │ cap-enforced; verify,    │  │ process value · difficulty               │
          │ result → tokens  │  │ judge-rank, vote, repair │  │                                          │
          └──────────────────┘  └──────────────────────────┘  └──────────────────────────────────────────┘
```

### 4.2 Dataflow for one query

1. **Tokenize.** Korean goes through a rule-based josa/eomi pre-splitter (0 learned params) before the unigram model.
2. **Probe** (r=1 over the prompt). The controller head reads the pooled prelude state and the judge's difficulty output, and proposes a budget (r, N, K).
   - The controller code clips that budget to the global envelope, N · T · FLOP(r) ≤ 2e13 (§0.6).
   - The budget is fixed for the episode, or for one answer segment.
   - No dataset identity is visible to the controller (F10).
3. **Retrieve.**
   - The query vector is the mean-pooled L2 state of the question, passed through the 640→256 projection and then sign().
   - IVF Hamming search runs over the 156M binary codes. The top 1,000 are rescored with int8 (laptop) or PQ-32 (Kaggle).
   - In parallel, BM25 runs over the ~6B-token core, with Korean character bigrams.
   - Reciprocal-rank fusion gives the top K.
   - **Multiple choice:** one query per option (stem + option_i).
   - `<search>` may be re-issued for up to 3 hops with rewritten queries.
4. **Read (sandwich order).** The stream is [question] → [top-K chunks] → [question again] → generation.
   - Evidence enters the recurrent state through ordinary delta-rule writes. There is no read-fusion module.
   - **Why the sandwich:** a fixed-state recurrence must decide what to keep while reading. JRT showed that recall in fixed-memory recurrent models depends on the order in which information arrives, and that reading twice closes much of the gap [VERIFIED: https://arxiv.org/abs/2407.05483].
   - Putting the question first lets the state know what to keep. Repeating it afterwards lets the model re-query its state. This is a cheaper variant of JRT's full re-read [HYPOTHESIS]; X3 compares evidence-first, sandwich and JRT-style read-twice.
   - Gated DeltaNet beats Mamba-2 and DeltaNet on in-context retrieval [VERIFIED: https://arxiv.org/abs/2412.06464v3].
5. **Compile.** Under `<IR>`, the trunk samples N whole programs in the typed IR (§5.4). CFG-constrained decoding masks illegal tokens, so every sample parses.
6. **Execute.**
   - Programs run asynchronously in a pre-forked hardened sandbox pool, with a 2 s timeout per call. Results are cached by program hash.
   - `READ` and `JUDGE` are host callbacks into the network (§5.4 re-entrancy).
   - Results are spliced back as tokens. Resuming a GDN after a tool result costs only the new tokens, because the state is constant-size [ESTIMATE].
7. **Verify and select.**
   - Exact checks first: CHECK asserts, self-written tests, SymPy substitution, ARC demo pairs.
   - Then majority vote over executed answers.
   - Then judge energy.
   - The three are combined by the 32-param logistic selection calibrator, which is counted in the cap and fit on dev rollouts.
   - Below threshold, the controller repairs using the error message, re-retrieves, or raises r, within the cap.
8. **Express.**
   - **Multiple choice:**
     - The state after [shots + stem] is computed once and **forked** per option, an O(1) copy of the constant-size state.
     - Each fork then reads [option-specific evidence → stem → option].
     - The judge scores each fork, and the trunk writes the winning letter.
     - The §0.6 cost reflects this: 600 shared tokens plus 4 × 632 per option.
   - **Prose:**
     - `<PLAN>` emits an outline of 3–8 lines: intent, constraints, evidence chunk ids.
     - Each section is realized under `<REALIZE>`. `<copy>` triggers suffix-automaton extension of an exact span.
     - Rule-based format checks run.
     - N drafts are ranked by chat reward, and one critique → revise pass with the same weights is allowed.
9. **Emit** with citations and a trace hash.
   - A copied span cites its source chunk id. That also satisfies BY-SA attribution for copied Wikipedia/StackExchange spans [VERIFIED: https://download.wikipedia.com/legal.html].
   - A sentence that was not copied cites the retrieved chunks whose BM25 overlap with it passes a fixed threshold. If none passes, it is marked `[no source]`.

### 4.3 Why this is not a Transformer, stated honestly

- **Mixer.** There is no softmax attention over token positions, anywhere. Every sequence mixer is a gated delta-rule linear recurrence with a constant-size state of 5 heads × 128 × 128 per layer. There is no KV cache, and memory per token is O(1).
- **Judge.** The judge is a right-to-left recurrence over the trunk's states, not attention.
- **Copying** is a 0-parameter suffix-automaton operation: no pointer and no softmax over positions.
- **Multiple choice** uses state forking, not cross-attention.
- **Retrieval** is a hard, non-differentiable top-K whose results enter only as tokens.
- **The honest caveat:** a GDN block keeps the projection-plus-MLP macro-structure of a Transformer block, and a reviewer may call it a *linear* Transformer. ALEMBIC's claim to novelty does not rest on block anatomy. It rests on system properties:
  - constant-state recurrence;
  - weight-tied test-time depth;
  - externalized human-written memory;
  - program-mediated reasoning with exact verification;
  - a discriminative judge.

  Rule F6 is the operational definition. It is published so a referee can disagree on the record.

---

## 5. Module-by-module specification

### 5.1 Tokenizer (32,768 counted scores)

- Unigram LM with byte fallback, trained on a 2B-token sample: 45% English, 35% Korean, 20% code.
- Korean handling:
  - The 2,350 most common Hangul syllables are seeded.
  - About 7k entries go to Korean syllables and morphemes.
  - A rule-based pre-splitter separates common josa/eomi suffixes from a fixed list, so no learned morphological-analyzer costs enter the count.
  - Jamo splitting is rejected, because it lengthens Korean sequences on a fixed-state recurrence.
- 32,768 fits uint16, so the store is a flat token memmap.
- **Day-1 gate:** measure fertility on kowiki. If Korean needs more than 1.6× as many tokens per character as English, move to 40k vocabulary: +1,858,624 params from headroom. That leaves 1,829,024 even with all pools [ESTIMATE threshold].

### 5.2 Factorized tied embedding (8,552,448)

- A 32,768 × 256 table plus a 256 → 640 up-projection, transposed for the LM head.
- Embeddings are 10.54% of the budget. Gemma-3-270M, for comparison, spends 63% on embeddings [VERIFIED: https://developers.googleblog.com/en/introducing-gemma-3-270m/].
- X1 ablates a 384-d bottleneck (+4,276,224). It is **mutually exclusive with running all pools** (§0.1).
- **Segment embeddings** (8 × 640) are **added once at the input**, to the token embedding before the prelude. They are not re-added per loop: the injector re-concatenates h_prelude, which already carries them.

### 5.3 Trunk: prelude (3), core (4 tied × r), coda (4)

**GDN layer, per head, with S ∈ ℝ^{dv×dk} = ℝ^{128×128}:**
- S_t = α_t · S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ, then o_t = S_t q_t.
- k and q are L2-normalized.
- α_t = exp(−softplus(·) · exp(A_log)) with dt_bias, which is the log-space decay of the reference [ESTIMATE: match the NVlabs code line by line, https://github.com/NVlabs/GatedDeltaNet].
- β_t = sigmoid(·).
- The output passes through a SiLU output gate and the per-head RMSNorm.
- A short depthwise conv (k=4) runs on q, k, v.
- SwiGLU 640 → 1536 → 640.
- **expand_v = 1.**
- Mechanism per GDN [VERIFIED: https://arxiv.org/abs/2412.06464v3].

**Chunkwise (WY/UT) form, written out.** I derived this for this version; the derivation is checked by the unit tests in §10.2.

Inside one chunk of C = 128 positions, per head, the inputs are:
- Q, K ∈ ℝ^{C×dk}, with rows q_r, k_r;
- V ∈ ℝ^{C×dv};
- the incoming state S₀;
- β ∈ ℝ^C;
- the in-chunk cumulative log-decay g_r = Σ_{i≤r} log α_i, with γ_r = exp(g_r).

Define the lower-triangular decay matrix Γ_{ri} = exp(g_r − g_i) for r ≥ i and 0 otherwise. Since r ≥ i, every entry is ≤ 1, so it cannot overflow. Then:

1. **UT transform:** A = strictly_lower( diag(β) (Γ ⊙ K Kᵀ) ), T = (I + A)⁻¹. This is a triangular solve by forward substitution, in fp32.
2. **WY factors:** W̃ = T · diag(β ⊙ γ) · K, and U = T · diag(β) · V.
3. **Corrected values:** V′ = U − W̃ S₀ᵀ.
4. **Outputs:** O = diag(γ) Q S₀ᵀ + ( Γ ⊙ lower_incl(Q Kᵀ) ) V′. Here the diagonal is included, and Γ_rr = 1.
5. **Carry:** S_C = γ_C S₀ + V′ᵀ · diag(exp(g_C − g)) · K.

*Derivation sketch.*
- The ungated product ∏(I − β_i k_i k_iᵀ) equals I − Σ w_i k_iᵀ, with w_r = β_r(k_r − Σ_{i<r} w_i k_iᵀk_r).
- Scalar decay commutes with that product. So the decayed value accumulator is H̃_r = Σ_i (γ_r/γ_i) u_i k_iᵀ, with u_r = β_r(v_r − Σ_{i<r} (γ_r/γ_i) u_i k_iᵀk_r).
- Rescaling w̃_i = γ_i w_i makes both use the same gated T.
- For C = 1 the formulas reduce to S₁ = α S₀ + β(v − α S₀ k)kᵀ and o = S₁ q, which matches the recurrence. This is unit test `test_chunk_c1_identity`.

Precision:
- QKᵀ, KKᵀ and the chunk matmuls run in fp16 (T4) or bf16 (TPU), **with fp32 accumulation** (`preferred_element_type=float32`).
- Γ, T and the carried state S stay fp32.
- The audit's warning applies: if intra-chunk matmuls had to run with fp32 *inputs* on a T4 (8.1 TFLOPS), wall-clock would rise by about 1.8× [ESTIMATE]. X0 measures this.

**Recurrence definition.**
- Unrolled, iteration i of the tied core is its own depth-layer with its own time-recurrent state S_{i,t}.
- **r is fixed per query, or per answer segment,** so every state is always defined.
- The injector re-concatenates the prelude output at every iteration: x_i = W[h_{i−1}; h_prelude], with h_0 = h_prelude.
  - W is initialized to [½I, ½I] plus N(0, 1e-4).
  - So x_1 = h_prelude exactly, and each later iteration starts as the average of the previous iterate and the prelude [HYPOTHESIS init].
- **Per-position extra compute** comes from learned `<think>` tokens. Each has a fixed input embedding, so the expanded sequence stays chunkwise-parallel in training.
- **`<think>` placement:**
  - S1 has none.
  - In S2, the trace generators insert 1–4 `<think>` tokens before each program step, with the count proportional to the generator's step-difficulty metadata.
  - In S5, the policy places its own, and a reward term penalizes the count.
  - At inference the model emits them itself.
- Changing r between segments (re-prefill) is an ablation in X2.

**Training the recurrence:**
- r is sampled per sequence. S1 and S4 draw **uniformly from {1, 2, 4}** (r̄ = 2.333). S2 draws from **p(r) ∝ 1/r on the integers 1..16** (r̄ = 4.733).
  - v1's "mean ≈ 2 / ≈ 4" was wrong. The continuous log-uniform version of the S2 sampler would give 5.41; we pin the integer distribution.
- TRM-style detached deep supervision: the coda and LM loss apply after iteration ⌈r/2⌉ (weight 0.5) and after iteration r (weight 1.0).
- Gradients flow through only the last 2 iterations, plus the injector path to the prelude. There is no full BPTT [VERIFIED mechanism: https://arxiv.org/abs/2510.04871].
- The budget script still charges full 6N backprop on every iteration, plus the second coda pass and the recompute, which is conservative.
- Accuracy vs r for r ∈ {1, …, 32} is reported on every row.

**State size:** 5 × 128 × 128 = 81,920 floats per layer, which is 327.7 KB in fp32. At r = 8 the 39 unrolled layer states total 12.78 MB per sequence, which is what makes 64-way best-of-N batching cheap [ESTIMATE].

**Mode tokens** (0 params, reserved vocabulary): `<PLAN> <READ> <IR> <REALIZE> <JUDGE> <think> <copy> <search> </search> <py> <answer> <cite>`.

### 5.4 Program IR, grammar, executors, re-entrancy (0 params)

**IR = an SSA-style, straight-line, indentation-free subset of Python plus tool primitives.** Indentation-sensitive Python is hard to grammar-mask, so the IR has no blocks. The CFG (EBNF):

```
program   := { stmt NL } final NL
stmt      := NAME "=" expr                      # each NAME assigned exactly once (SSA)
final     := "ANSWER" "(" expr [ "," "cite" "=" NAME ] ")"
           | "RENDER" "(" NAME "," "style" "=" STR "," "lang" "=" STR ")"
expr      := term { BINOP term } | "-" expr | "not" expr
term      := NUM | STR | NAME | call | list | listcomp | "(" expr ")" | NAME "[" expr "]"
call      := PRIM "(" [ arg { "," arg } ] ")"
arg       := expr | NAME "=" expr
list      := "[" [ expr { "," expr } ] "]"
listcomp  := "[" expr "for" NAME "in" expr [ "if" expr ] "]"
BINOP     := + - * / // % ** == != < <= > >= and or
PRIM      := SEARCH | READ | SOLVE | GRAPH | CHECK | JUDGE | SAT*        # SAT after admission A2
           | abs min max sum len round range sorted int float Fraction gcd lcm comb factorial
           | sympy.<allow-listed fn> | math.<allow-listed fn>              # fixed published list
```

The IR has no `def`, `while`, `import`, attribute access outside the allow-list, or I/O.

**Decoding mask.** The mask is a hand-written incremental LL(1) recognizer over token strings, so each sample parses. It is precomputed per tokenizer: a token-to-terminal-prefix trie [ESTIMATE engineering, about 1 week].

**Code tasks are not IR.** HumanEval and MBPP are decoded as ordinary Python. They are not grammar-masked. Each candidate is validated with `ast.parse` and repaired on error.

Example program:

```python
ev   = SEARCH(lang="ko+en", q="...", k=8)        # BM25 + dense RRF, returns chunk ids
sp   = READ(ev, ask="...")                       # span + citation (via <copy> suffix match)
x    = SOLVE("...")                              # SymPy; g = GRAPH(edges) for NetworkX
ok   = CHECK(x > 0)                              # executable assertion
best = JUDGE([c1, c2, c3, c4], evidence=sp)      # judge tower energy ranking
ANSWER(x, cite=sp)
```

**Re-entrancy (who holds what).**
- The controller holds the outer episode's trunk state and **freezes** it while a program runs.
- `SEARCH`, `SOLVE`, `GRAPH` and `CHECK` run in the sandbox.
- **`READ(ev, ask)` and `JUDGE(...)` are host callbacks**, implemented by the controller.
  - `READ` starts a **fresh** trunk pass over the sandwich [ask → chunks(ev) → ask] at the episode's r. It decodes ≤ 64 tokens with `<copy>` enabled. It does not fork the outer state.
  - `JUDGE` runs the judge tower over each candidate given the evidence.
- **Nesting depth is 1:** a nested call cannot emit programs.
- **Accounting:** each nested neural call counts as one of the ≤ 10 tool calls per program. Its FLOPs are added to the query's running total, which the controller checks against the 2e13 cap before each call. The call is refused if it would exceed the cap.
- Results re-enter the outer stream as tokens.

**Execution:**
- Whole programs are generated, then executed asynchronously in a pre-forked pool (laptop CPU + Kaggle/TPU-VM host CPUs).
- Results are cached by program hash.
- Pure-arithmetic programs go to a vectorized NumPy evaluator.
- Limits: ≤ 10 tool calls per program, ≤ 200 executions per query, ≤ 60 s CPU per query.

**ARC-AGI:**
- Programs are written in Hodel's DSL, and the demo pairs are an *exact* verifier.
- This is the cheapest abstraction row: enumerative plus trunk-proposed program search, with no separate grid network at M1.
- Training on eval-task demos is excluded from the gated row (F8).

### 5.5 Copy: suffix automaton (0 params; from OUROBOROS)

1. The model emits `<copy>` followed by 1–3 ordinary tokens.
2. The controller looks up that prefix in a suffix automaton built over [retrieved chunks; prompt; tool outputs].
3. It extends the match greedily, scoring each continuation token with the model's **own** vocabulary distribution in one teacher-forced chunkwise pass.
4. It stops at a sentence boundary or when p(token) < τ.

**τ policy:**
- There is one global τ, tuned once on the NQ-dev + DROP-dev span-EM sum and then frozen.
- τ is a disclosed scalar hyperparameter (F2).
- No per-row τ is allowed (F10).

Copied text re-enters the stream as tokens. This gives DROP and NQ exact spans and gives prose verbatim human phrasing, with no gray-zone softmax.

### 5.6 Judge tower (16,354,978 including input norm and fusion)

- **Input:** the trunk's per-token final states h_1..h_T, from a forward pass over [evidence; context; `<JUDGE>`; candidate] at the query's r (r = 2 in multiple-choice mode).
- **Body:** 3 GDN layers scan h_T → h_1 (right to left). Every judged position therefore sees the whole candidate and its context. This is a recurrence, not attention.
- **Fusion:** [h_fwd_last; h_bwd_first] (1280-d), RMSNorm, then 1280 → 1024 → 4 outputs:
  - **(a) energy / verify** (lower is better): used for MC options, programs and traces;
  - **(b) chat reward:** Bradley–Terry, trained in S6;
  - **(c) process value** on IR steps;
  - **(d) difficulty:** feeds the controller budget.
- **Label sources:**
  - **(a)** executed outcomes; gold labels of constructed minimal pairs; teacher ranking distributions; Open-track train-split labels.
  - **(c)** Monte-Carlo rollouts from step prefixes: the fraction of 8 completions whose execution is correct, budgeted in §0.3.
  - **(d)** the ordinal index of the cheapest (r, N) budget that solved the problem in the controller-label runs, with "unsolved" as the top class.
- **Why one tower, not CORTEX's separate 12M d=384 stack:** the tower reads features produced by the full trunk. Judging therefore draws on about 81M of representational capacity while adding only 16.35M (the win-lens graft).
- **Self-grading guard:** the judge is trained **only** on the label sources above, never on the generator's own preference.
- **Pool J (+1 layer)** is released only by the X1 rule (§10.2).

### 5.7 Retriever (163,840 counted; everything else is data)

- **Keys and queries:** the embedding + prelude L1–L2 + the 640→256 projection, mean-pooled, then sign() into 256-bit codes.
  - These tensors are **frozen after S1** (from ORRERY), so the 156M-chunk index never goes stale and nothing is double-counted.
  - **Cost of freezing:** freezing L1–L2 also freezes 2 of the 3 prelude layers for S2–S6. X3 measures this directly with an arm that leaves L1–L2 unfrozen and rebuilds the index.
  - **Fallback:** if freezing costs more than 2% of dev loss, or more than 2 points on any dev family, the index is rebuilt after each stage instead (0.61 TPU-h per build + I/O; 2 builds are budgeted).
- **S1 bootstrap.** The dense key encoder does not exist until the end of S1. So **S1's in-stream neighbours come from BM25 over the 6B-token core**, precomputed on CPU. X1's retrieval and copy arms are also BM25-only.
- **Tuning knobs:** K is adaptive in {4, 8, 16, 32}; up to 3 hops; reciprocal-rank fusion of dense and BM25.
- **Recall gate:** recall@32 on KILT/NQ dev must be within 5 points of exact fp32 search on a 1M-chunk sample. Otherwise move to 512-bit codes (11.25 GB of RAM including ids).
  - The 92.5% / 96.45% binary-quantization figures were measured on 1024-d vectors [VERIFIED: https://huggingface.co/blog/embedding-quantization]. They are **not assumed** to transfer.
- **Alternative arm in X3:** LIBRARIAN's static distilled key table (model2vec-style; potion-base-8M reaches about 92% of MiniLM at 7.56M params [VERIFIED: https://huggingface.co/minishlab/potion-base-8M]).
  - It would cost 8.4M from headroom and is **mutually exclusive with Pool G**.
  - It is kept only if it beats the frozen-trunk keys on recall@32 per rebuild-hour.

### 5.8 Controller head (42,194) and controller code (0 params)

- **Outputs:** an r-level (6 classes), an N-level (4) and a route (8 classes: factoid, MC, IR-math, code, chat, multi-hop, grid, other). Routes are input types, not datasets.
- **Training:** cross-entropy on the cheapest budget class that solved each problem in the **controller-label runs**: 20k verifiable training problems × r ∈ {1, 2, 4, 8, 16} × 64 samples, subsampled for N ∈ {1, 4, 16, 64}. These runs are budgeted at 3.2 TPU-h and 267 CPU-h.
- **Controller code** implements Snell-style compute-optimal allocation [VERIFIED: https://arxiv.org/abs/2408.03314] inside the §0.6 envelope:
  - easy prompts: r = 1–2, N = 1;
  - medium: N up to 16;
  - hard verifiable: N ≤ 64 at r ≤ 8, or N ≤ 32 at r = 16, or N ≤ 8 at r = 32.
- The controller sees only the input. One policy serves every row (F10).

### 5.9 Explicitly out of M1 (scope fence)

These are allowed back only through admission (§10.3):
- Titans/hippocampal test-time memory;
- CORTEX's latent workspace and axial grid scans;
- ORRERY's latent diffusion and energy descent;
- product-key memory (Pool M, 1B+ track);
- Lean 4;
- Wikidata triple store;
- phrase index;
- a JEPA loss;
- a separate grid reasoner;
- fan-out state merge.

---
