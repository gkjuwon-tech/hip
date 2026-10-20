# Compute Envelope Report (zero-budget, as of Oct 2026)

Labels: [VERIFIED: url] = number found at that URL during this research. [ESTIMATE] = my arithmetic or an inference from verified numbers. [HYPOTHESIS] = a plausible claim not yet tested. A "third-party" note means the source is not the platform's own docs.

---

## 0. TL;DR

1. **The Kaggle TPU v5e-8 is the main training engine, not the T4s.** Peak bf16 compute is 8 x 197 = 1,576 TFLOPS, against 2 x 65 = 130 TFLOPS fp16 for T4x2. That is about 12x on paper and about 8-10x at realistic utilization [ESTIMATE]. The catch is that it effectively requires JAX (or PyTorch/XLA) and static shapes.
2. **Tokens per week for a 100M-parameter model (6ND):** about **38-76B tokens/week on the TPU** (20 h) and about **3.5-9B tokens/week on T4x2** (30 h) [ESTIMATE]. Chinchilla-optimal for 100M parameters is about 2B tokens [ESTIMATE, 20 tokens per parameter], so pretraining FLOPs are **not** the bottleneck at 0.1B scale. Data quality, test-time compute, and the retrieval index are.
3. **Synthetic data is the scarce resource.** Teacher-model generation on T4 lands at roughly 10^8 tokens/week for a 7B teacher [ESTIMATE]. On TPU v5e-8, a third-party repo reports about 10k tok/s aggregate for a 27B model, which would be about 0.7B tokens/week [third-party, unverified].
4. **Storage:** HF Hub gives 100 GB private and "best-effort" public storage. Kaggle `/kaggle/working` is about 20 GB per session. RAM is about 29-32 GB per session, which caps an in-memory retrieval index at about 20 GB [ESTIMATE].
5. **Scaling to 1B-10B parameters will not fit this envelope** without TPU Research Cloud (TRC) or grants. A 10B model at Chinchilla scale needs about 1.2e22 FLOPs [ESTIMATE], which is about 25 years of Kaggle TPU quota. That is a hard wall to plan around.

---

## 1. Kaggle free tier

| Item | Value | Source |
|---|---|---|
| GPU options | 1x P100 16 GB **or** 2x T4 16 GB (32 GB total) | [VERIFIED: https://aimultiple.com/free-cloud-gpu] (third-party) |
| GPU weekly quota | ~30 h/week. It floats, is shown in account settings, and Kaggle's docs give no number. A 2021 source calls it a "30 h minimum" resetting Saturday 00:00 UTC | [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits], [VERIFIED: https://towardsdatascience.com/?p=111441] |
| GPU session max | 12 h (CPU/GPU) | [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits] (says it checked Kaggle docs on 21 Sep 2026) |
| TPU type | **TPU v5e-8.** v3-8 is being phased out ("New TPU v5e-8s! Phasing out v3-8s"). The official docs page still describes v3-8 | [VERIFIED: https://www.kaggle.com/discussions/product-announcements/607202], [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html] |
| TPU weekly quota | ~20 h/week. Consistent across third parties; I found no current official v5e-8 figure | [VERIFIED: https://aifree.dev/offer/kaggle-free-gpu-tpu], [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab] (third-party) |
| TPU session max | 9 h ("a little under nine hours" in practice) | [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits], [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab] |
| TPU availability | Users report v5e-8 as "oversubscribed, quota-limited, or temporarily disabled" at times. Phone verification is required, and some accounts also get a Persona ID check | [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab] (third-party summary of the Kaggle discussion) |
| TPU via CLI | `kaggle kernels push` with TPU settings may silently give a non-TPU image without torch_xla. Use the interactive editor or verify the device in-run | [VERIFIED: https://github.com/Kaggle/kaggle-cli/issues/1197] |
| Host RAM | ~29-32 GB (CPU/P100/T4x2) | [VERIFIED: https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/main/kaggle_20251121.md] (third-party KB, "as of 2024-2025") |
| CPU | 2 Xeon CPUs listed (core count not given; commonly 4 vCPU [ESTIMATE]) | same KB |
| Output disk `/kaggle/working` | ~20 GB persisted, plus a file-count limit | same KB |
| Private dataset quota | Conflicting: 100 GB vs 200 GB (book editions). Measured uncompressed | [VERIFIED: https://subscription.packtpub.com/book/data/9781835083208/3/ch03lvl1sec16/gathering-the-data] (book; check kaggle.com/docs/datasets) |
| Internet | Available in notebooks after phone verification. Off in many code competitions | [ESTIMATE] (commonly documented; Kaggle docs page did not render for me) |
| Colab Pro link bonus | Linked Colab Pro/Pro+ gives +15/+30 GPU h/week (paid, so excluded) | [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits] |

**Practical rules for Kaggle:**
- Run training as "Save & Run All" (commit) sessions so they don't depend on the browser tab staying open.
- Checkpoint every 30-60 minutes to `/kaggle/working`, then push to the HF Hub.
- Chain sessions: each new session loads the last checkpoint from the HF Hub or from a Kaggle dataset version.
- One account only (see ToS, section 6).

## 2. Hardware specs

| Device | Peak dense matmul | Memory / bandwidth | Notes |
|---|---|---|---|
| NVIDIA T4 | **65 TFLOPS FP16/mixed**, 8.1 TFLOPS FP32, 130 TOPS INT8, 260 TOPS INT4 | 16 GB GDDR6, 300 GB/s (datasheet text) | 70 W TDP. **No bf16**, so use fp16 with loss scaling [VERIFIED: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf]. vLLM forum confirms "T4 does not support bfloat16", dtype=half [VERIFIED: https://discuss.vllm.ai/t/no-throughput-improvement-for-quantized-qwen-2-5-7b-instruct/1901] |
| TPU v5e chip | **197 TFLOPS bf16**, 393 TOPS int8 | 16 GB HBM, ~800 GB/s | [VERIFIED: https://docs.cloud.google.com/tpu/docs/v5e]. v5e-8 = 8 chips = 1,576 TFLOPS, 128 GB HBM [ESTIMATE] |
| RTX 3060 Laptop | Not published for the laptop part. Desktop 3060 is 51.2 TFLOPS FP16 tensor (102.4 with sparsity) | **6 GB** GDDR6, 192-bit, **336 GB/s**, 3,840 CUDA cores, 120 tensor cores, up to 115 W TGP | [VERIFIED: https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3060-Laptop-GPU-Graphics-Card.497453.0.html], [VERIFIED: https://gigazine.net/gsc_news/en/20210113-nvidia-geforce-rtx-3060] |

**T4 caveats, all [ESTIMATE]:**
- The T4's 70 W power cap causes clock throttling under sustained tensor load, so real dense GEMM speed is typically well below 65 TFLOPS.
- Turing (sm_75) is not supported by FlashAttention-2 (Ampere and later). This is from memory: check the FlashAttention README.
- Triton supports sm_75, so custom Triton kernels for recurrent/SSM scans work, but with fp16 only.
- The 2 T4s connect over PCIe, not NVLink, so use DDP (data parallel) and avoid tensor parallelism.

**3060 Laptop caveat [ESTIMATE]:** GeForce Ampere runs FP16 with FP32 accumulate at half rate. Realistic training peak is therefore about 22-27 TFLOPS for bf16/fp16 with fp32 accumulate. Unlike the T4, the 3060 **has bf16**.

## 3. Training throughput: 6ND for a 100M-parameter model

FLOPs per token = 6N = 6e8. Attention FLOPs are not applicable, since the backbone is non-Transformer. The recurrent/scan overhead of an SSM-style backbone is folded into the MFU (model FLOPs utilization) range.

**Expected MFU, all [ESTIMATE]:**
- T4: 15-40%. Small hidden dimensions (512-1024) make GEMMs too small to saturate the chip.
- v5e at 100M scale: 20-40%. Reference point: the official JAX tutorial on Kaggle/Colab TPU reports only **16-27% FLOPs utilization** for a tiny GPT, and **13%** with 8-way data parallelism vs 27% with 4-way [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html]. Small models really are hard to keep busy, so tune batch size and mesh.

| Platform | Peak | MFU (low / mid / high) | tok/s (low / mid / high) | Hours/week | **Tokens/week** |
|---|---|---|---|---|---|
| Kaggle T4x2 | 130 TFLOPS | 15 / 25 / 40% | 32.5k / 54k / 87k | 30 | **3.5 / 5.9 / 9.4 B** |
| Kaggle TPU v5e-8 | 1,576 TFLOPS | 20 / 30 / 40% | 525k / 788k / 1.05M | 20 | **38 / 57 / 76 B** |
| RTX 3060 Laptop | ~25 TFLOPS | 25 / 35 / 45% | 10k / 15k / 19k | ~100 (thermal/duty limited) | **3.8 / 5.3 / 6.8 B** |

All rows are [ESTIMATE] from 6ND applied to verified peak numbers.

**Implications, all [ESTIMATE]:**
- **One TPU week is about 20-35 Chinchilla-optimal runs at 100M parameters** (about 2B tokens each). A year of quota (52 weeks x ~50B) is about 2.6T token-passes at 100M parameters, more than LLaMA-1 was trained on, although the unique-data limit bites first.
- Compute budget per year: TPU ≈ 1576e12 x 0.3 x 72,000 s x 52 ≈ **1.8e21 FLOPs/year**. T4x2 adds about 1.8e20. Total is about **2e21 FLOPs/year**, which is 0.6% of GPT-3 175B training compute (3.14e23, standard figure).
- **Scaling wall:** a Chinchilla-style budget of 6 x N x 20N gives 1.2e18 FLOPs at 100M parameters (minutes of TPU time), 1.2e20 at 1B (about 3 TPU-weeks), and **1.2e22 at 10B (about 7 years of free quota)**. Beyond about 1B parameters, the free envelope requires TRC or grants. This argues for a design whose scaling axes are **datastore size, test-time compute, tools, and self-generated data**, all of which grow without retraining the core.
- **Memory fit:** training a 100M model needs about 16 bytes/param (fp32 master + Adam moments + grads), which is 1.6 GB. It fits easily on a 6 GB 3060, a 16 GB T4, or per v5e chip. A 1B model is about 16 GB, so it needs sharding (FSDP over 8 v5e chips is fine; it does not fit on a T4 without 8-bit optimizers or offload).

**Hardware-to-workload fit [HYPOTHESIS]:**
- TPU v5e-8: pretraining or distillation of the core reasoner, and large-batch embedding of the retrieval corpus.
- T4x2: teacher-data generation (vLLM), evaluation harness, retriever index building, RL rollouts.
- 3060 Laptop: development, kernel debugging, small ablations (10-50M parameters), and running the final <100M system offline.

## 4. Synthetic data generation throughput (teacher models)

**Reference numbers, measured on an A100 80GB, vLLM 0.6.3, batch size 1** [VERIFIED: https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html]:
- Qwen2.5-1.5B: 183 tok/s (BF16), 214 tok/s (AWQ).
- Qwen2.5-7B: 84 tok/s (BF16), 148 tok/s (AWQ), 154 tok/s (GPTQ-Int4).

**Other references:**
- RTX 4090, vLLM, Qwen2.5-7B-AWQ: 85 tok/s single stream, 142 tok/s at batch 8 [VERIFIED: https://markaicode.com/benchmarks/vllm-qwen-benchmark/] (third-party).
- The vLLM forum warns that AWQ can be *slower* than fp16 at large batch because of dequantization overhead [VERIFIED: https://discuss.vllm.ai/t/no-throughput-improvement-for-quantized-qwen-2-5-7b-instruct/1901].

**T4 projections:** I found no measured T4 numbers. Single-stream decode is bandwidth-bound, at most bandwidth / weight bytes.

| Teacher on 1x T4 | Weights | Decode bound, batch 1 | Batched aggregate (vLLM, ~64-256 seqs) | Per week (2 T4 x 30 h) |
|---|---|---|---|---|
| Qwen2.5-1.5B fp16 | ~3.1 GB | ≤ ~97 tok/s, realistic ~50-70 | ~1,500-3,000 tok/s | **~0.3-0.65 B tokens** |
| Qwen2.5-7B AWQ int4 | ~5 GB | ≤ ~60 tok/s, realistic ~25-40 | ~300-800 tok/s | **~65-170 M tokens** |
| 14B AWQ across 2 T4 (tensor parallel over PCIe) | ~9 GB | ~15-20 tok/s | ~200-400 tok/s total | ~20-45 M tokens |

All T4 projections are [ESTIMATE].
- Batched aggregate assumes compute-bound decoding at 2N FLOPs/token and about 10-25% of 65 TFLOPS, after KV-cache limits on 16 GB.
- The 7B per-week figure equals tok/s x 2 GPUs x 108,000 s.

**TPU v5e-8 as a generator:** a third-party repo reports Qwen3.8-27B bf16 on vllm-tpu at about 130 tok/s single stream and about 540 tok/s at 8 streams. With many streams it reports about 10,300 tok/s, plus about 22 minutes of startup [VERIFIED: https://github.com/ARahim3/kaggle-tpu-lab] (third-party, not reproduced).
- At 10k tok/s x 72,000 s, that is **about 0.74B tokens/week of 27B-class text** [ESTIMATE].
- This competes with training for the same 20 h.

**Budgeting, all [ESTIMATE]:**
- Realistic high-quality distillation corpus: 0.1-0.7B tokens/week. That is about 5-35B tokens/year, filtered down by verification (code execution, math checkers) to maybe 30-50%.
- **Prefer verifiable generation** (code plus unit tests, math with checkable answers, proofs in Lean) so that compute buys *correct* tokens.
- **Licensing:** check each teacher's license for "use outputs to train other models" clauses.
  - Qwen2.5 uses Apache-2.0 for most sizes; some sizes use the Qwen license. Verify per model card.
  - Llama-3.x licenses require attribution and naming conventions.
  - **Never use GPT-3.5/4 outputs as training data.** It breaks OpenAI ToS and contaminates the head-to-head comparison.

## 5. Other "genuinely free" compute

| Option | Free allowance | Usefulness | Source |
|---|---|---|---|
| Google Colab free | ≤12 h sessions "depending on availability". Limits fluctuate and are not published. GPU varies (T4 commonly reported). **TPU is v5e-1 only (as of Oct 2025), which "can no longer support SPMD"** | Spare T4 for evals/generation. Free tier forbids SSH/remote desktop, web UIs that bypass the notebook, distributed workers, and multi-accounting | [VERIFIED: https://research.google.com/colaboratory/faq.html], [VERIFIED: https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html] |
| Google TPU Research Cloud (TRC) | Free Cloud TPU quota "on a temporary basis". "Anyone can express interest", with invites on a rolling basis. **You pay for the VMs and GCS** ("generally minimal"). You must publish openly and acknowledge TRC | **The only realistic path to 1B-10B.** Not literally 0 KRW: needs a GCP billing account and small VM/storage costs | [VERIFIED: https://sites.research.google/trc/faq/] |
| GCP free trial | $300 / 90 days, **but GPUs cannot be added during the trial** | Could pay the VM/GCS side costs of TRC | [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits] |
| Modal Starter | $30/month of free compute | About 10-30 GPU-hours/month of a mid-tier GPU [ESTIMATE]. Good for bursty evals | [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits] |
| Lightning AI free | 1 active Studio, restarts every 4 h, 50 GB persistent storage. Monthly credit amount unconfirmed | Persistent dev box and storage | [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits] |
| HF ZeroGPU | Free user: 5 min GPU/day; 60 s default function timeout | **Demo hosting only**, not training | [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits] |
| Paperspace Free | M4000 8 GB, 5 GB storage, 6 h max; availability untested | Negligible | same |
| SageMaker Studio Lab | Closed to new users | Not available | same |
| Grants | Modal academic (up to $10k, application); HF storage grants for high-impact OSS | Apply once results exist | [VERIFIED: https://gpuperhour.com/blog/free-cloud-gpus-and-credits], [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits] |

## 6. Storage and persistence

**Hugging Face Hub** [VERIFIED: https://huggingface.co/docs/hub/en/storage-limits]:
- Free account: **100 GB private**. Public storage is "best-effort"; beyond "the first few gigabytes" it must be "genuinely useful".
- Hard 500 GB per-file limit; files under 200 GB recommended.
- Fewer than 100k files per repo; at most 10k entries per folder.
- About 50-100 files per commit, with a 60 s HTTP timeout.
- **Use:** checkpoints (a 100M model is about 0.2 GB in bf16; with optimizer state about 1.2 GB), public datasets in Parquet, and retrieval index shards as public datasets with a dataset card.

**Kaggle Datasets:** private quota is 100-200 GB (unconfirmed) and mounts read-only into notebooks. This is the fastest way to get large static data (an index or corpus) into a session without download time.

**Retrieval datastore sizing, all [ESTIMATE]:**
- 100M chunks x (256-dim int8 = 256 B) = 25.6 GB.
- With PQ at 32 B/vector: 3.2 GB plus the raw text (~40-60 GB compressed for about 25B tokens).
- **Session RAM (~30 GB) is the binding constraint:** use PQ/IVF indexes (e.g. FAISS IVF-PQ) or memory-mapped indexes on local disk.
- **Kaggle local scratch disk:** the commonly reported figure is about 50-70 GB+ non-persistent [ESTIMATE, unverified]. Verify with `df -h` in-session.
- **Index embedding time:** embedding 25B tokens with a 20M-param encoder (2N = 4e7 FLOPs/token) is about 1e18 FLOPs. That is about 1 hour of T4x2 or minutes of TPU, so cheap.

## 7. Terms of service and fair-use constraints

1. **No multi-accounting to multiply quota.** Colab forbids it explicitly [VERIFIED: https://research.google.com/colaboratory/faq.html]. Kaggle's Terms (effective 4 Oct 2024) bind users to the Terms plus the Community Guidelines [VERIFIED: https://kaggle.com/terms]. I could not read the full clause list, so treat multi-accounting, crypto-mining, and quota circumvention as prohibited [ESTIMATE].
2. **Persistent servers on Kaggle/Colab** (exposing vLLM via a tunnel as an API) are a gray zone. Colab free tier forbids "bypassing the notebook UI to interact primarily through a web UI" and remote-control shells. Do not build infrastructure that depends on this.
3. **TRC:** results must be shared publicly, and research must follow Google's AI Principles [VERIFIED: https://sites.research.google/trc/faq/]. This is compatible with an open project, but the **drug-discovery and biology data/models must go through dual-use review before release**. This is a hard requirement in the blueprint, not a ToS item.
4. **Teacher-model licenses** govern synthetic data (section 4). No closed-API outputs.
5. **HF public storage** must be "genuinely useful": publish dataset cards and avoid dumping opaque intermediate shards.

## 8. Weekly operating plan (one account, 0 KRW) [ESTIMATE]

| Resource | Hours | Primary job | Output/week |
|---|---|---|---|
| TPU v5e-8 | 20 (3 sessions of ~6.5 h) | Core model pretraining and distillation (JAX) | ~40-60B training tokens at 100M parameters, or ~1-3 ablation sweeps at 10-50M |
| T4x2 | 30 (3 sessions of 10 h) | vLLM teacher generation, verifier filtering, evals, index build | ~0.1-0.6B synthetic tokens; full eval suite runs |
| RTX 3060 Laptop (6 GB) | ad hoc | Kernels, unit tests, small ablations, offline demo | Iteration speed |
| Colab free | opportunistic | Overflow evals | — |
| HF Hub | — | Checkpoint and dataset persistence | ≤100 GB private |

## 9. Key uncertainties to verify in-session (first day)

1. Run `nvidia-smi`, `df -h`, `free -g`, and `nproc` on Kaggle T4x2. Run `jax.devices()` on the TPU (expect 8 v5e devices, 16 GB each).
2. Check actual remaining GPU/TPU quota in Kaggle account settings. The API does not expose it [VERIFIED: https://pypi.org/project/kagglex/] (third-party tooling note).
3. Measure MFU of the chosen backbone at 50M/100M on both T4 (fp16) and v5e (bf16). Replace every [ESTIMATE] in section 3 with measured tok/s.
4. Benchmark vLLM on T4 with Qwen2.5-1.5B fp16 and 7B AWQ at batch 128 to replace the section 4 estimates.
5. Check TPU v5e-8 availability over 2-4 weeks. If it is often "oversubscribed", the T4-only plan cuts training tokens about 8x, which is still above Chinchilla for 100M parameters.
6. Apply to TRC early (the application is free). It is the only verified free path past about 1B parameters.

## Sources
- https://www.kaggle.com/discussions/product-announcements/607202
- https://github.com/ARahim3/kaggle-tpu-lab
- https://github.com/Kaggle/kaggle-cli/issues/1197
- https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html
- https://gpuperhour.com/blog/free-cloud-gpus-and-credits
- https://aimultiple.com/free-cloud-gpu
- https://towardsdatascience.com/?p=111441
- https://aifree.dev/offer/kaggle-free-gpu-tpu
- https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/main/kaggle_20251121.md
- https://subscription.packtpub.com/book/data/9781835083208/3/ch03lvl1sec16/gathering-the-data
- https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
- https://docs.cloud.google.com/tpu/docs/v5e
- https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3060-Laptop-GPU-Graphics-Card.497453.0.html
- https://gigazine.net/gsc_news/en/20210113-nvidia-geforce-rtx-3060
- https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html
- https://markaicode.com/benchmarks/vllm-qwen-benchmark/
- https://discuss.vllm.ai/t/no-throughput-improvement-for-quantized-qwen-2-5-7b-instruct/1901
- https://research.google.com/colaboratory/faq.html
- https://sites.research.google/trc/faq/
- https://huggingface.co/docs/hub/en/storage-limits
- https://kaggle.com/terms
- https://pypi.org/project/kagglex/
