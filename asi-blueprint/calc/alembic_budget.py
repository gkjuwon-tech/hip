# ALEMBIC-0 single source of truth: params, FLOPs, hours, bytes. If the blueprint disagrees, this wins.
d, H, dh, ffn, V, e = 640, 5, 128, 1536, 32768, 256
def gdn():
    return 4*d*d + d*d + 2*d*H + 3*d*4 + 3*d*ffn + 2*d
L = gdn()
mods = {
 "tokenizer_unigram_scores": V,
 "embedding_factorized_tied": V*e + e*d,
 "prelude_3_gdn": 3*L,
 "core_4_gdn_tied": 4*L,
 "injector_2d_to_d": 2*d*d + d,
 "coda_4_gdn": 4*L,
 "final_norm": d,
 "judge_tower_3_backward_gdn": 3*L,
 "judge_input_norm_2d": 2*d,
 "judge_fusion_mlp_1280_1024_4": 2*d*1024 + 1024 + 1024*4 + 4,
 "retrieval_proj_640_256": d*256,
 "controller_head_640_64_18": d*64 + 64 + 64*18 + 18,
 "segment_embeddings_8": 8*d,
}
tot = sum(mods.values())
print("GDN layer", L)
for k,v in mods.items(): print(f"{k:34s} {v:>12,d}")
print("TOTAL", f"{tot:,d}", "headroom", f"{100_000_000-tot:,d}")
poolG, poolJ, poolR = 2*L, L, d*256
print("PoolG", poolG, "PoolJ", poolJ, "PoolR", poolR, "worst", f"{tot+poolG+poolJ+poolR:,d}")
head = V*e + e*d
def nact(r): return (7+4*r)*L + head + r*(2*d*d+d)
for r in [1,2,4,8,16,32]: print("r",r,"N_active",f"{nact(r):,d}","eff_depth",7+4*r, "fwd FLOP/tok %.3g"%(2*nact(r)))
judge = 3*L + mods["judge_fusion_mlp_1280_1024_4"]
print("judge params", judge, "judge fwd FLOP/tok %.3g"%(2*judge))
tpu = 1576e12*0.12; tpu_h = tpu*3600; tpu_wk = tpu*72000
t4 = 130e12*0.20; t4_wk = t4*108000
print("TPU FLOP/h %.3g  per week %.3g ; T4x2 per week %.3g"%(tpu_h, tpu_wk, t4_wk))
print("S1 tokens/week at r=2: %.3g"%(tpu_wk/(6*nact(2))))
st = {
 "S1 lang+reading 30B r2": 6*nact(2)*30e9,
 "S2 procedures 15B r4": 6*nact(4)*15e9,
 "S3 judge 3B (trunk fwd r2 + judge 6N)": (2*nact(2)+6*judge)*3e9,
 "S4 SFT 2B r2": 6*nact(2)*2e9,
 "S5 RLVR 2B rollout(2N r4)+1B train(6N r4)": 2*nact(4)*2e9 + 6*nact(4)*1e9,
 "S6 pref 0.3B x2 r2": 6*nact(2)*0.6e9,
}
main=0
for k,v in st.items():
    main+=v; print(f"{k:45s} %.3g FLOP  %.1f TPU-h"%(v, v/tpu_h))
print("MAIN %.3g FLOP %.1f TPU-h %.2f TPU-weeks"%(main, main/tpu_h, main/tpu_wk))
# experiments
grid = sum(6*n*100*n for n in [5e6,10e6,20e6,40e6])  # 100 tok/param, ~1.3x for r & retrieval
grid*=1.5
x = {"X1 wall probes": 6*20e6*1.5*2e9*3 + 6*10e6*2e9*2 + 6*5e6*2e9*2,
     "X2 recurrence 20M x r{1,4,16} 2B": sum(6*(20e6*(0.4+0.6*r/2))*2e9 for r in [1,4,16]),
     "X3 retrieval 50M 4 arms 3B": 4*6*50e6*2*3e9,
     "X4 compile vs CoT 30M 8B x2": 2*6*30e6*2*8e9,
     "X5 scaling grid": grid}
ex=0
for k,v in x.items(): ex+=v; print(f"{k:40s} %.3g FLOP %.1f TPU-h"%(v, v/tpu_h))
print("EXPTS %.3g  %.1f TPU-h"%(ex, ex/tpu_h))
allf=(main+ex)*1.8
print("ALL x1.8 %.3g  %.1f TPU-h  %.2f TPU-weeks"%(allf, allf/tpu_h, allf/tpu_wk))
print("T4-only weeks for main+expts x1.8: %.1f"%(allf/t4_wk))
# index build
enc = 2*(head + 2*L + d*256)
print("key-encoder FLOP/tok %.3g ; 20B tok %.3g FLOP = %.2f TPU-h"%(enc, enc*20e9, enc*20e9/tpu_h))
# storage
chunks = 20e9/128
print("chunks %.3g ; binary256 %.2f GB ; int8-256 %.1f GB ; PQ32 %.2f GB ; text uint16 %.1f GB"%(chunks, chunks*32/1e9, chunks*256/1e9, chunks*32/1e9, 20e9*2/1e9))
# inference
f = lambda r: 2*nact(r); fj = 2*judge
q = {
 "factoid: 1088 tok prefill r2 + 20 dec": 1088*f(2)+20*f(2),
 "MC 4 options x 712 tok r2 + judge": 4*712*f(2)+4*712*fj,
 "GSM8K 32x250 r4 + judge 32x400": 32*250*f(4)+32*400*fj + 400*f(4),
 "HumanEval 16x400x4 rounds r8": 16*400*4*f(8) + 16*4*400*fj,
 "MATH hard 64x600 r8 (cap)": 64*600*f(8),
 "chat plan+8x400 r4 + judge + revise": 8*400*f(4)+8*400*fj+600*f(4)+1500*f(2),
}
for k,v in q.items(): print(f"{k:40s} %.3g FLOP"%v)
