"""ALEMBIC-0 FINAL: single source of truth for params, FLOPs, hours, bytes, per-query cost.
If blueprint_final.md and this script disagree, THIS SCRIPT WINS. CI re-runs it on every config change.
Run: python3 calc/alembic_budget.py           (prints every canonical number)
     python3 calc/alembic_budget.py --check   (asserts only; exit code != 0 on any violation)
All throughput/MFU inputs are [ESTIMATE] planning values; the day-1 X0 gate replaces them with measurements."""
import math, sys

# ----------------------------------------------------------------- 1. parameter model
def gdn_layer(d, H, dh, ffn):
    """One Gated DeltaNet layer. expand_v = 1 is PINNED (NVlabs/FLA default is 2: would add 3*d^2 per layer)."""
    assert H * dh == d, "H*dh must equal d (expand_v=1)"
    return (4*d*d            # q, k, v, o projections
            + d*d            # output gate
            + 2*d*H          # alpha / beta gate projections
            + 3*d*4          # short depthwise conv k=4 on q, k, v
            + 3*d*ffn        # SwiGLU
            + 2*d            # two pre-RMSNorms (mixer, MLP)
            + 2*H            # A_log and dt_bias (per head)
            + dh)            # per-head output RMSNorm weight (shared across heads)

def model_params(c):
    d, H, dh, ffn, V, e = c["d"], c["H"], c["dh"], c["ffn"], c["V"], c["e"]
    L = gdn_layer(d, H, dh, ffn)
    jh = c["judge_hidden"]
    m = {
        "tokenizer_unigram_scores (not a JAX leaf; CI adds it)": V,
        "embedding_factorized_tied": V*e + e*d,
        f"prelude_{c['n_pre']}_gdn": c["n_pre"]*L,
        f"core_{c['n_core']}_gdn_tied (counted once)": c["n_core"]*L,
        "injector_2d_to_d": 2*d*d + d,
        f"coda_{c['n_coda']}_gdn": c["n_coda"]*L,
        "final_norm": d,
        f"judge_tower_{c['n_judge']}_backward_gdn": c["n_judge"]*L,
        "judge_input_norm_2d": 2*d,
        f"judge_fusion_mlp_{2*d}_{jh}_4": 2*d*jh + jh + jh*4 + 4,
        "retrieval_proj_d_256": d*256,
        "controller_head_d_64_18": d*64 + 64 + 64*18 + 18,
        "segment_embeddings_8": 8*d,
        "selection_calibrator_8routes_x4": 8*4,   # logistic: 3 features + bias per route (fitted -> counted)
    }
    return L, m

M1 = dict(d=640, H=5, dh=128, ffn=1536, V=32768, e=256, n_pre=3, n_core=4, n_coda=4, n_judge=3, judge_hidden=1024)
# Scaling / wall-probe family (X1 uses S, M, L; X5 uses S, M, L, XL). Same tokenizer, H*dh = d, dh = 128.
FAMILY = {
    "S  (~5M)":  dict(d=256, H=2, dh=128, ffn=640,  V=32768, e=64,  n_pre=1, n_core=1, n_coda=1, n_judge=1, judge_hidden=256),
    "M  (~10M)": dict(d=384, H=3, dh=128, ffn=768,  V=32768, e=96,  n_pre=1, n_core=1, n_coda=1, n_judge=1, judge_hidden=384),
    "L  (~20M)": dict(d=512, H=4, dh=128, ffn=1024, V=32768, e=128, n_pre=1, n_core=2, n_coda=1, n_judge=1, judge_hidden=512),
    "XL (~40M)": dict(d=640, H=5, dh=128, ffn=1536, V=32768, e=128, n_pre=2, n_core=2, n_coda=2, n_judge=1, judge_hidden=768),
}

def trunk_parts(c):
    L, m = model_params(c)
    d, V, e = c["d"], c["V"], c["e"]
    head = V*e + e*d
    inj = 2*d*d + d
    judge = c["n_judge"]*L + 2*d + (2*d*c["judge_hidden"] + c["judge_hidden"] + c["judge_hidden"]*4 + 4)
    return L, head, inj, judge

def n_active(c, r):
    """Matmul params touched per generated token at recurrence r (prelude + r x core + coda + injector + head)."""
    L, head, inj, _ = trunk_parts(c)
    return (c["n_pre"] + c["n_coda"] + c["n_core"]*r)*L + head + r*inj

def eff_depth(c, r): return c["n_pre"] + c["n_core"]*r + c["n_coda"]

def mixer_fwd_flop_per_layer_token(c, C=128):
    """Chunkwise delta-rule mixer FLOPs not captured by 2*params (per layer, per token, forward). [ESTIMATE]"""
    dh = c["dh"]
    return c["H"] * (4*C*dh + 6*dh*dh + C*C/3 + 4*C*dh)

def fwd_flop(c, r):  # forward FLOP per token incl. mixer
    return 2*n_active(c, r) + eff_depth(c, r)*mixer_fwd_flop_per_layer_token(c)

def train_flop(c, rbar, deep_sup=True, recompute=True):
    """Per-token training FLOP: 6N_active(rbar) (conservative: full backprop through every iteration, although
    only the last 2 iterations receive gradients) + deep-supervision 2nd coda+head pass at ceil(r/2)
    + activation-checkpoint recompute of each core iteration + 3x mixer FLOPs."""
    L, head, inj, _ = trunk_parts(c)
    f = 6*n_active(c, rbar)
    if deep_sup:  f += (2/3 if rbar < 3 else 0.9) * 6*(c["n_coda"]*L + head)  # fraction of seqs with r>=2
    if recompute: f += 2*(c["n_core"]*L + inj)*rbar
    f += 3*eff_depth(c, rbar)*mixer_fwd_flop_per_layer_token(c)
    return f

L1, mods = model_params(M1)
TOTAL = sum(mods.values())
CAP = 100_000_000
JAX_LEAVES = TOTAL - M1["V"]
poolG, poolJ, poolR = 2*L1, L1, M1["d"]*256
POOLS = poolG + poolJ + poolR
vocab40k = (40_000 - 32_768)*(M1["e"] + 1)               # embedding rows + unigram scores
emb384 = (384 - 256)*(M1["V"] + M1["d"])                 # wider bottleneck ablation
static_keys = 8_400_000                                   # LIBRARIAN static key table arm (X3) [ESTIMATE size]
_, head1, inj1, JUDGE = trunk_parts(M1)

# ----------------------------------------------------------------- 2. hardware planning values [ESTIMATE]
TPU_PEAK, TPU_MFU, TPU_H_WK = 1576e12, 0.12, 20
TPU_DEC_MFU = 0.04                     # batched decode on TPU
T4X2_PEAK, T4_MFU, T4_H_WK = 130e12, 0.20, 30
T4_PREFILL_MFU_1B5, T4_PREFILL_MFU_7B = 0.20, 0.15    # 7B runs AWQ/TP=2: dequant overhead
T4_DEC_MFU = 0.05
G3060_EFF = 25e12*0.35                  # 3060 laptop fp16 w/ fp32 acc, 35% MFU
tpu_h = TPU_PEAK*TPU_MFU*3600; tpu_wk = tpu_h*TPU_H_WK
tpu_dec_h = TPU_PEAK*TPU_DEC_MFU*3600
t4_h = T4X2_PEAK*T4_MFU*3600; t4_wk = t4_h*T4_H_WK
TPU_WINDOW_H = 12*TPU_H_WK             # weeks 1-12
T4_WINDOW_H = 11*T4_H_WK               # weeks 2-12

R_S1 = sum([1, 2, 4])/3                                  # r ~ uniform{1,2,4}
_p = [1/r for r in range(1, 17)]; R_S2 = sum(r*p for r, p in zip(range(1, 17), _p))/sum(_p)   # p(r) ∝ 1/r on 1..16

# ----------------------------------------------------------------- 3. main training stages (TPU)
def S(c=M1):
    st = {}
    st["S1 lang+reading 30B scanned, r~U{1,2,4}"] = train_flop(c, R_S1)*30e9
    st["S2 procedures+IR 15B, p(r)∝1/r on 1..16"] = train_flop(c, R_S2)*15e9
    st["S3 judge 3B (trunk fwd r=2 + judge 6N)"] = (fwd_flop(c, 2) + 6*JUDGE)*3e9
    st["S4 SFT 2B, r~U{1,2,4}"] = train_flop(c, R_S1)*2e9
    st["S5 RLVR: 1.6B single-shot rollouts (TPU decode) -> charged below; 1B train r=4"] = train_flop(c, 4)*1e9
    st["S6 DPO 50k pairs x1200 tok x3 ep = 0.18B (policy 6N + ref 2N)"] = (train_flop(c, R_S1) + fwd_flop(c, R_S1))*0.18e9
    return st
ST = S()
MAIN = sum(ST.values())
DEC = {  # TPU batched decode jobs, charged at TPU_DEC_MFU
    "S5 rollouts 1.6B tok at r=4": fwd_flop(M1, 4)*1.6e9,
    "controller+difficulty labels: 20k probs x r{1,2,4,8,16} x 64 samp x 300 tok": sum(fwd_flop(M1, r) for r in [1, 2, 4, 8, 16])*20e3*64*300,
    "process-value MC labels: 100k traces x 4 prefixes x 8 x 200 tok, r=4": fwd_flop(M1, 4)*100e3*4*8*200,
    "S6 samples: 50k prompts x 4 x 400 tok, r=4": fwd_flop(M1, 4)*50e3*4*400,
}
DEC_H = sum(DEC.values())/tpu_dec_h
INDEX_FLOP_TOK = 2*(M1["e"]*M1["d"] + 2*L1 + M1["d"]*256)   # key encoder: emb gather (no LM head) + L1-L2 + proj
INDEX_H = 2*INDEX_FLOP_TOK*20e9/tpu_h                       # budget 2 builds (initial + one rebuild)
X0_H = 3.0                                                  # kernel/MFU benchmark sessions

# ----------------------------------------------------------------- 4. experiments (TPU)
fam = list(FAMILY.values())
def total_params(c): return sum(model_params(c)[1].values())
EX = {}
# X1: S/M/L trunks, 2 arms (+/- logit KD) x 2.5B tok (2B S1-mini + 0.5B SFT-mini); judge 0.59B tok x 2 tracks
x1 = 0
for c in fam[:3]:
    _, _, _, jg = trunk_parts(c)
    x1 += 2*train_flop(c, R_S1)*2.5e9 + 2*(fwd_flop(c, 2) + 6*jg)*0.59e9
EX["X1 wall probes S/M/L, +/-KD, Open+Strict judges"] = x1
EX["X2 recurrence L-config r in {1,4,16} + <think>@4, 2B each"] = sum(train_flop(fam[2], r)*2e9 for r in [1, 4, 16, 4])
c50 = dict(M1, n_core=3, n_coda=2, n_judge=0)              # ~50M generator for X3 (no judge)
EX["X3 retrieval 50M-class, 8 arms x 1.5B (scanned)"] = 8*train_flop(c50, R_S1)*1.5e9
c30 = dict(fam[3], n_judge=0)
EX["X4 IR vs CoT 35M-class, 2 x 4B, p(r)∝1/r"] = 2*train_flop(c30, R_S2)*4e9
seeds = [3, 3, 2, 1]
EX["X5 grid S/M/L/XL @100 tok/param, seeds 3/3/2/1"] = sum(s*train_flop(c, R_S1)*100*total_params(c) for c, s in zip(fam, seeds))
EX["X5 ORRERY-0 control grid (r=1, no judge), 1 seed"] = sum(train_flop(dict(c, n_judge=0), 1, deep_sup=False)*100*total_params(c) for c in fam)
EXP = sum(EX.values())
RETRY = 1.8
TPU_TOTAL_H = (MAIN + EXP)/tpu_h*RETRY + DEC_H*RETRY + INDEX_H + X0_H

# ----------------------------------------------------------------- 5. T4x2 sheet (weeks 2-12)
def pre(nparams, tok, mfu): return 2*nparams*tok/(T4X2_PEAK*mfu*3600)
DEC_RATE_1B5 = 0.3e9/T4_H_WK                               # 0.3B tok/week planning value for the 1.5B teacher
T4 = {
    "teacher decode: OMI-2 -> IR rewrite 0.3B (1.5B)": 0.3e9/DEC_RATE_1B5,
    "teacher decode: plan/section chat traces en/ko 0.2B": 0.2e9/DEC_RATE_1B5,
    "teacher decode: Korean procedures/instructions 0.1B": 0.1e9/DEC_RATE_1B5,
    "teacher prefill: judge ranking 1.5M items x 300 tok (prefix-shared), 1.5B": pre(1.5e9, 0.45e9, T4_PREFILL_MFU_1B5),
    "teacher prefill: adversarial filter 0.3M items x 500 tok (8 cands), 7B": pre(7.6e9, 0.15e9, T4_PREFILL_MFU_7B),
    "teacher prefill: aligned top-16 logits 0.3B (1.5B) + X1 mini 0.05B": pre(1.5e9, 0.35e9, T4_PREFILL_MFU_1B5),
    "family-A judge: S6 labels 50k pairs x 2 orders x 1200 tok, 7B": pre(7.6e9, 0.12e9, T4_PREFILL_MFU_7B),
}
TEACH_H = sum(T4.values())
TEACH_RETRY = 1.5
OTHER_T4 = {
    "eval panel: X1 chat arm 6 models x 965 pairs x 2 judges x 2 orders x 1500 tok": pre(8.5e9, 6*965*4*1500, T4_PREFILL_MFU_7B),
    "eval panel: R0 pooled 1465 pairs x 2 x 2 x 1500 + rho>=0.8 calibration": pre(8.5e9, 1465*4*1500 + 5*160*2*1500, T4_PREFILL_MFU_7B),
    "reference ladder answers (3 models x 965 x 500 tok) + Korean proxy answers": 3.0,
    "S5 multi-turn tool rollouts 0.4B tok at r=4 (T4 decode)": fwd_flop(M1, 4)*0.4e9/(T4X2_PEAK*T4_DEC_MFU*3600),
    "X0 parity + vLLM throughput benchmark": 4.0,
    "S1/S2 checkpoint dev evals": 10.0,
    "R0 full Tier A+B neural part (+seeds)": 30.0,
}
T4_TOTAL_H = TEACH_H*TEACH_RETRY + sum(OTHER_T4.values())

# ----------------------------------------------------------------- 6. CPU sheet [ESTIMATE]
EXEC_S = 0.15                                              # mean s per sandboxed execution incl. timeouts
CPU = {
    "S5 RLVR executions (2.0B/250 = 8M programs)": 8e6*EXEC_S/3600,
    "controller labels executions (6.4M)": 20e3*5*64*EXEC_S/3600,
    "process-value executions (3.2M)": 100e3*4*8*EXEC_S/3600,
    "decontamination (13-gram/MinHash/ko-20gram) over ~70B tok": 120.0,
    "store build: tokenize 20B + BM25 6B core + IVF train": 80.0,
    "cross-tokenizer logit alignment 0.35B tok": 20.0,
    "monthly full eval tool calls (MATH ~1M calls dominates)": 60.0,
}
ARC_LAPTOP_H = (400*2/60, 400*10/60)

# ----------------------------------------------------------------- 7. storage
CH = 20e9/128
STORE = {"chunks": CH, "binary256_GB": CH*32/1e9, "faiss_ids_GB": CH*8/1e9, "int8_256_GB": CH*256/1e9,
         "pq32_GB": CH*32/1e9, "text_uint16_raw_GB": 20e9*2/1e9}
LOGITS_GB = 0.3e9*16*4/1e9

# ----------------------------------------------------------------- 8. inference per query (neural FLOP incl. mixer)
f = lambda r: fwd_flop(M1, r); fj = 2*JUDGE
QCAP = 2e13
Q = {
    "factoid: q40 + 8x128 ev + q40 sandwich, r=2, 20 dec": (40 + 1024 + 40 + 20)*f(2),
    "MC 4 opts: shared 5-shot+stem 600, fork; per opt 4x128 ev + stem 100 + opt 20, r=2, +judge": (600 + 4*632)*f(2) + 4*632*fj,
    "GSM8K: 32 IR x 250 r=4 + judge 32x400 + final 400": 32*250*f(4) + 32*400*fj + 400*f(4),
    "chat: plan + 8x400 r=4 + judge + revise 600 r=4 + 1500 prompt/ev r=2": 8*400*f(4) + 8*400*fj + 600*f(4) + 1500*f(2),
    "HumanEval: 16 x 400 x 4 rounds r=8 + judge": 16*400*4*f(8) + 16*4*400*fj,
    "MATH hard: 64 x 600 r=8": 64*600*f(8),
    "MATH hard alt: 16 x 600 r=16": 16*600*f(16),
}
# controller envelope check: every (N, r, tokens) mode the global policy may pick must fit the cap
MODES = [(1, 1, 2000), (1, 2, 2000), (16, 4, 600), (64, 4, 600), (64, 8, 600), (16, 16, 600), (32, 16, 600), (8, 32, 600)]

def gpt35_query_flop(params, tokens=1000): return 2*params*tokens   # [ESTIMATE] size unknown

# ----------------------------------------------------------------- 9. scaling-wall arithmetic
RATIO_ACTIVE = n_active(M1, R_S1)/n_active(M1, 1)
def chinchilla_flop(trunk, tpp=20, ratio=None): return 6*trunk*(ratio or n_active(M1, R_S1)/(n_active(M1, 1)))*trunk*tpp

def checks():
    errs = []
    if TOTAL >= CAP: errs.append("M1 over cap")
    if TOTAL + POOLS >= CAP: errs.append("pools over cap")
    if TOTAL + POOLS + vocab40k >= CAP: errs.append("pools+40k over cap")
    # mutually exclusive combos (asserted as exclusions, not as allowed configs)
    excl = {"pools+emb384": TOTAL + POOLS + emb384, "pools+static_keys": TOTAL + POOLS + static_keys}
    for k, v in excl.items():
        if v < CAP: errs.append(f"{k} unexpectedly fits; update the exclusion table")
    for N, r, T in MODES:
        if N*T*f(r) > QCAP: errs.append(f"mode N={N} r={r} T={T} breaks cap: {N*T*f(r):.3g}")
    for k, v in Q.items():
        if v > QCAP: errs.append(f"query '{k}' breaks cap")
    if TPU_TOTAL_H > TPU_WINDOW_H: errs.append("TPU plan exceeds 12-week window")
    if T4_TOTAL_H > T4_WINDOW_H: errs.append("T4 plan exceeds 11-week window")
    return errs, excl

if __name__ == "__main__":
    errs, excl = checks()
    if "--check" in sys.argv:
        print("OK" if not errs else "\n".join(errs)); sys.exit(1 if errs else 0)
    print("== PARAMETERS ==")
    print("GDN layer", f"{L1:,d}")
    for k, v in mods.items(): print(f"  {k:52s} {v:>12,d}")
    print(f"TOTAL {TOTAL:,d}  (JAX leaves {JAX_LEAVES:,d} + tokenizer scores {M1['V']:,d})  headroom {CAP-TOTAL:,d}")
    print(f"judge total (3 GDN + input norm + fusion) {JUDGE:,d}   judge fwd FLOP/tok {fj:.3g}")
    print(f"PoolG {poolG:,d}  PoolJ {poolJ:,d}  PoolR {poolR:,d}  -> all pools {TOTAL+POOLS:,d}  left {CAP-TOTAL-POOLS:,d}")
    print(f"40k vocab +{vocab40k:,d} -> pools+40k {TOTAL+POOLS+vocab40k:,d}  left {CAP-TOTAL-POOLS-vocab40k:,d}")
    print(f"emb384 +{emb384:,d}; static keys +{static_keys:,d}; exclusions: " + ", ".join(f"{k}={v:,.0f}" for k, v in excl.items()))
    print(f"embedding share {(mods['embedding_factorized_tied'])/TOTAL*100:.2f}%")
    for r in [1, 2, 4, 8, 16, 32]:
        print(f"  r={r:2d} N_active {n_active(M1,r)/1e6:6.1f}M  depth {eff_depth(M1,r):3d}  fwd FLOP/tok {f(r):.3g} (matmul-only {2*n_active(M1,r):.3g})")
    print(f"mixer fwd FLOP/layer/tok {mixer_fwd_flop_per_layer_token(M1)/1e6:.3f}M = {mixer_fwd_flop_per_layer_token(M1)/(2*L1)*100:.1f}% of layer matmuls")
    print(f"state/layer {M1['H']*M1['dh']**2:,d} floats = {M1['H']*M1['dh']**2*4/1e3:.1f} KB; per seq at r=8 {eff_depth(M1,8)*M1['H']*M1['dh']**2*4/1e6:.2f} MB")
    print(f"train memory fp32 master+Adam+grads {TOTAL*16/1e9:.2f} GB")
    print("== FAMILY (X1/X5) ==")
    for k, c in FAMILY.items():
        Lc, mc = model_params(c); emb = mc["embedding_factorized_tied"]
        print(f"  {k}: total {sum(mc.values())/1e6:6.2f}M  non-embedding {(sum(mc.values())-emb-c['V'])/1e6:6.2f}M  layer {Lc:,d}")
    print(f"50M-class X3 generator {total_params(c50)/1e6:.1f}M ; 35M-class X4 {total_params(c30)/1e6:.1f}M")
    print("== COMPUTE ==")
    print(f"TPU FLOP/h {tpu_h:.3g}  /20h-week {tpu_wk:.3g} ; T4x2 /30h-week {t4_wk:.3g} ; TPU decode FLOP/h {tpu_dec_h:.3g}")
    print(f"r_bar S1 {R_S1:.3f} (N_act {n_active(M1,R_S1)/1e6:.1f}M) ; r_bar S2 {R_S2:.3f} (N_act {n_active(M1,R_S2)/1e6:.1f}M)")
    print(f"train FLOP/tok S1 {train_flop(M1,R_S1):.3g} ; S1 tokens per TPU-week {tpu_wk/train_flop(M1,R_S1):.3g}")
    for k, v in ST.items(): print(f"  {k:78s} {v:.3g}  {v/tpu_h:5.1f} TPU-h")
    print(f"MAIN {MAIN:.3g} FLOP  {MAIN/tpu_h:.1f} TPU-h")
    for k, v in DEC.items(): print(f"  decode {k:72s} {v:.3g}  {v/tpu_dec_h:5.1f} TPU-h@{TPU_DEC_MFU:.0%}")
    print(f"DECODE {DEC_H:.1f} TPU-h ; INDEX (2 builds) {INDEX_H:.2f} TPU-h (key enc {INDEX_FLOP_TOK:.3g} FLOP/tok) ; X0 {X0_H} h")
    for k, v in EX.items(): print(f"  {k:60s} {v:.3g}  {v/tpu_h:5.1f} TPU-h")
    print(f"EXPTS {EXP:.3g}  {EXP/tpu_h:.1f} TPU-h")
    print(f"TPU TOTAL = (main+expts)x{RETRY} + decode x{RETRY} + index + X0 = {TPU_TOTAL_H:.1f} TPU-h = {TPU_TOTAL_H/TPU_H_WK:.2f} TPU-weeks of {TPU_WINDOW_H/TPU_H_WK:.0f}; slack {(1-TPU_TOTAL_H/TPU_WINDOW_H)*100:.0f}%")
    # T4-only fallback: S1 12B, S2 8B, rest identical; all on T4x2
    fb = train_flop(M1, R_S1)*12e9 + train_flop(M1, R_S2)*8e9 + sum(v for k, v in ST.items() if not k.startswith(("S1", "S2")))
    fb_all = (fb + EXP)*RETRY + sum(DEC.values())*RETRY*(TPU_DEC_MFU/TPU_MFU) * (TPU_MFU*TPU_PEAK)/(T4_DEC_MFU*T4X2_PEAK) * (T4_MFU*T4X2_PEAK)/(TPU_MFU*TPU_PEAK)
    print(f"T4-ONLY fallback: training+expts x{RETRY} {(fb+EXP)*RETRY:.3g} FLOP = {(fb+EXP)*RETRY/t4_wk:.1f} T4x2-weeks ; + decode jobs {sum(DEC.values())*RETRY/(T4X2_PEAK*T4_DEC_MFU*3600)/T4_H_WK:.1f} T4x2-weeks")
    fbx = EX["X1 wall probes S/M/L, +/-KD, Open+Strict judges"] + EX["X2 recurrence L-config r in {1,4,16} + <think>@4, 2B each"] + EX["X3 retrieval 50M-class, 8 arms x 1.5B (scanned)"]/2 + EX["X4 IR vs CoT 35M-class, 2 x 4B, p(r)∝1/r"]/2 + sum(train_flop(c, R_S1)*100*total_params(c) for c in fam)
    fb_lean = (fb + fbx)*RETRY
    print(f"T4-ONLY LEAN plan (S1 12B, S2 8B, X3/X4 halved, X5 1 seed, no control grid): {fb_lean:.3g} FLOP = {fb_lean/t4_wk:.1f} T4x2-weeks + decode {sum(DEC.values())*RETRY/(T4X2_PEAK*T4_DEC_MFU*3600)/T4_H_WK:.1f} + teacher/eval {T4_TOTAL_H/T4_H_WK:.1f} = {fb_lean/t4_wk + sum(DEC.values())*RETRY/(T4X2_PEAK*T4_DEC_MFU*3600)/T4_H_WK + T4_TOTAL_H/T4_H_WK:.1f} T4x2-weeks total")
    print(f"full plan on T4 only would be {(MAIN+EXP)*RETRY/t4_wk:.1f} T4x2-weeks (training+expts)")
    print("== T4x2 SHEET (hours) ==")
    for k, v in T4.items(): print(f"  {k:80s} {v:6.1f} h")
    print(f"TEACHER {TEACH_H:.1f} h = {TEACH_H/T4_H_WK:.2f} T4x2-weeks ; x{TEACH_RETRY} = {TEACH_H*TEACH_RETRY:.1f} h")
    for k, v in OTHER_T4.items(): print(f"  {k:80s} {v:6.1f} h")
    print(f"T4 TOTAL {T4_TOTAL_H:.1f} h of {T4_WINDOW_H} h ; slack {(1-T4_TOTAL_H/T4_WINDOW_H)*100:.0f}%")
    print(f"7B ranking teacher for ALL 1.5M items instead: {pre(7.6e9,0.45e9,T4_PREFILL_MFU_7B):.0f} h")
    print("== CPU SHEET (CPU-hours) ==")
    for k, v in CPU.items(): print(f"  {k:62s} {v:6.0f}")
    print(f"CPU TOTAL {sum(CPU.values()):.0f} CPU-h ; ARC-AGI-1 monthly laptop {ARC_LAPTOP_H[0]:.0f}-{ARC_LAPTOP_H[1]:.0f} h")
    print("== STORAGE ==")
    print({k: round(v, 3) if v < 1e6 else f"{v:.4g}" for k, v in STORE.items()})
    print(f"RAM for 256-bit codes + ids {STORE['binary256_GB']+STORE['faiss_ids_GB']:.2f} GB ; 512-bit {2*STORE['binary256_GB']+STORE['faiss_ids_GB']:.2f} GB ; Kaggle 10B subset codes+ids+PQ {(STORE['binary256_GB']+STORE['faiss_ids_GB']+STORE['pq32_GB'])/2:.2f} GB")
    print(f"laptop footprint core: {STORE['binary256_GB']+STORE['faiss_ids_GB']+STORE['int8_256_GB']+STORE['pq32_GB']+STORE['text_uint16_raw_GB']:.1f} GB + BM25 2.5-10 GB")
    print(f"index rebuild on 3060: {INDEX_FLOP_TOK*20e9/G3060_EFF/3600:.1f} h ; logits {LOGITS_GB:.1f} GB")
    print("== INFERENCE PER QUERY ==")
    for k, v in Q.items(): print(f"  {k:90s} {v:.3g}")
    for N, r, T in MODES: print(f"  mode N={N:2d} r={r:2d} T={T}: {N*T*f(r):.3g} {'OK' if N*T*f(r)<=QCAP else 'OVER'}")
    print(f"max tokens/sample at N=64,r=16 under cap: {QCAP/(64*f(16)):.0f}")
    for p in [20e9, 175e9]: print(f"  GPT-3.5 1000-token query if {p/1e9:.0f}B dense: {gpt35_query_flop(p):.3g} FLOP ; cap ratio {QCAP/gpt35_query_flop(p):.2f}")
    print("== SCALING WALL ==")
    print(f"N_active(r_bar S1)/N_active(1) = {RATIO_ACTIVE:.2f}")
    for P in [1e9, 10e9]:
        fl = 6*P*RATIO_ACTIVE*20*P
        print(f"  {P/1e9:.0f}B trunk, 20 tok/param, r_bar 2.33: {fl:.3g} FLOP = {fl/tpu_wk:.1f} TPU-weeks = {fl/tpu_wk/52:.2f} years at 12% ; plain 6ND {6*P*20*P:.3g} = {6*P*20*P/tpu_wk:.1f} TPU-weeks = {6*P*20*P/tpu_wk/52:.2f} years")
    print("CHECKS:", "OK" if not errs else errs)
