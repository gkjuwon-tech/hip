# Phase F Audit: Numbers (work/blueprint.md)

Recomputed with explicit arithmetic. Antigravity hit its quota before Phase F, so this audit was done outside the Antigravity run.

| # | Item | Stated | Recomputed | Verdict | Corrected value |
|---|---|---|---|---|---|
| 1 | Gated DeltaNet block (d=640, d_ff=1600) | 5,539,200 | 6·640² + 3·640·1600 + 15·640 = 5,539,200 | ok | — |
| 2 | Reader (4 directional blocks) / core (8 blocks) | 22,156,800 / 44,313,600 | same | ok | — |
| 3 | 2D grid adapter | 52,224 | 3·3·8·8 + 8·640 + 2·640 = 6,976 | **wrong** | 6,976 |
| 4 | Total parameters | 81,694,209 | 81,648,961 | **wrong (by 45,248)** | 81,648,961 (< 100M, headroom 18,351,039) |
| 5 | N_non_core | 37,380,609 | 37,335,361 | minor | 37.34M |
| 6 | Recurrent state size per layer | "S ∈ R^{640×640} ≈ 320 KB fp16" | 640²·2 B = 800 KB for a full d×d state; 5 heads × 128² × 2 B = 160 KB for the multi-head state Gated DeltaNet actually uses | **wrong** | 160 KB/layer (multi-head, 5 × 128) |
| 7 | MCTS tree memory (256 nodes) | 80 MB | 256 × 8 layers × 160 KB = 320 MB (multi-head) | **wrong** | ≈ 0.32 GB, still fits a 16 GB T4 |
| 8 | Per-generation ExIt training tokens | 10.0M verified tokens | Rollouts only produce 9.0M tokens in total, so verified tokens must be fewer | **inconsistent** | ≈ 3.0M verified tokens × 3 epochs = 9.0M token-passes → 1.16e16 FLOPs; 357 s on 2×T4 at 25% MFU |
| 9 | Sandbox verification | 45,000 checks at 1,000/s = 45 s | Sandboxed subprocess checks typically cost tens of ms each; at ~80 checks/s on 4 vCPU this is ≈ 9 min [추정] | **optimistic** | 5–20 min per generation [추정] |
| 10 | Stage 0 time on TPU v5e-8 | 3.78 h at 30% MFU | 6.44e18 / (1576e12 × 0.20) = 5.7 h at 20% MFU (compute.md cites 16–27% MFU for small JAX models) | optimistic | 3.8–5.7 h; still one 9 h session |
| 11 | Generation cycle | 0.90 h (TPU) / 1.00 h (T4) | Mixing a T4 session and a TPU session inside one generation is not practical (separate sessions, XLA compile). All-T4: rollout 0.69 h + verify 0.1–0.3 h + train 0.1 h + I/O 0.2 h | optimistic | ≈ 1.1–1.3 h on T4 alone; 10–12 generations/week remains plausible including evals |
| 12 | Gated rows MATH / DROP / StrategyQA | G = 56.1 / 76.1 / 67.7 (from verified T) | BRIEF and reason_baseline.md gate on the max reported GPT-3.5 score: T = 43.1 / 70.2 / 70.3 → G = 62.1 / 80.2 / 80.3 | **rule mismatch** | Gate on 62.1 / 80.2 / 80.3 and note the verified values (34.1 / 64.1 / 51.5) |
| 13 | CRUXEval-O G | 75.6 | 63.3 + 12.23 = 75.5 | rounding | 75.5 |
| 14 | Latency, greedy 100 tokens at r=4 | 0.20 s | 429 MB/token at 230 GB/s = 1.87 ms → 0.19 s | ok | — |
| 15 | Rollout compute | 3.86e15 FLOPs | 9.0e6 × 4.29e8 | ok | — |
| 16 | Bootloader data total | 5.0B | 1.5 + 1.5 + 1.0 + 0.6 + 0.4 | ok | but "FineWeb-2 Korean Educational" is not an existing subset (see completeness audit) |
