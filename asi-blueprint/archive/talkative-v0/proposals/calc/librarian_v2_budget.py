"""LIBRARIAN v2: single source of truth for parameter and compute arithmetic.
Every number in librarian_v2.md that is derived (not cited) comes from here."""
V, E, d, H = 32768, 128, 640, 5          # vocab, factorized emb width, model width, heads (head dim 128)
F = 1728                                  # SwiGLU hidden (27*64)
P, R, C = 3, 8, 3                         # prelude, tied recurrent block, coda (unique layers)

mixer = 5*d*d + 2*d*H + 3*d*4 + 2*d       # q,k,v,o,out-gate + alpha/beta gates + short convs(k=4) + 2 norms
ffn = 3*d*F
layer = mixer + ffn
L = P + R + C
emb = V*E + E*d                           # factorized, tied with LM head
inject = 2*d*d + d
clerk = V*256
rerank = d*256 + 256*256
merge = d*H + H                           # per-chunk, per-head sigmoid merge gates (no softmax)
copy_phrase = 3*E*d + d + d*d             # phrase encoder over token embeddings (start,end,mean) -> d, then d->d
copy_query = d*d
copy_len = (16 + 8) * d
dispatch = d*(12+6) + (12+6)
lora = L * ((d+d)*32*3 + (F+d)*32)        # rank-32 on q,v,o,ffn-down of the unique layers
value = d*d + d + d + 1
seg = 4*d
norm = d
mods = dict(embedding=emb, prelude=P*layer, recurrent=R*layer, coda=C*layer, inject=inject, final_norm=norm,
            clerk=clerk, rerank=rerank, chunk_merge=merge, copy_phrase_enc=copy_phrase, copy_query=copy_query,
            copy_len_src=copy_len, dispatch=dispatch, critic_lora=lora, critic_value=value, segment=seg)
total = sum(mods.values())
scratch = (d*2*d + 2*d) + (2*d*d + d) + 2   # 640->1280->640 MLP init + lr/decay gates (v1+ only)
print("per-layer mixer", mixer, "ffn", ffn, "layer", layer)
for k, v in mods.items(): print(f"{k:16s}{v:>12,d}")
print(f"{'TOTAL v0':16s}{total:>12,d}")
print(f"{'+SCRATCH':16s}{scratch:>12,d}  -> {total+scratch:,d}  headroom {100_000_000-total-scratch:,d}")

# ---- compute ----
def n_eff(r):  # FLOP-params per token (matmul params touched), conservative: full 6N on all iterations
    return (P + C + R*r)*layer + inject*r + (V*E + E*d) + copy_query
for r in (1, 2, 4, 8, 16, 32):
    print(f"r={r:2d} N_eff={n_eff(r)/1e6:7.1f}M")
Ntr = n_eff(2)   # training mean r = 2 (heavy-tailed sampling)
stages = {"S1": 15e9, "S2": 25e9, "S3-S5": 12e9, "S7": 0.5e9}
fl = {k: 6*Ntr*v for k, v in stages.items()}
fl["S6"] = 4e9*2*n_eff(4) + 2e9*6*n_eff(4)   # rollouts at r=4 (2N) + train on 50% kept
tot = sum(fl.values())
for k, v in fl.items(): print(f"{k:6s}{v:.2e}")
print(f"TOTAL FLOPs {tot:.2e}")
for name, peak, mfu, hpw in (("v5e-8", 1576e12, 0.135, 20), ("T4x2", 130e12, 0.15, 30)):
    rate = peak*mfu; hrs = tot/rate/3600
    print(f"{name}: {rate:.2e} FLOP/s -> {hrs:.0f} h -> {hrs/hpw:.1f} quota-weeks")
# scaling grid: 4 trainings, store and r are inference knobs
grid = 0
for core_frac, toks in ((0.12, 3e9), (0.24, 3e9), (0.48, 3e9), (1.0, 3e9)):
    grid += 6*Ntr*core_frac*toks
print(f"E2 grid training FLOPs {grid:.2e} -> v5e-8 h {grid/(1576e12*0.135)/3600:.0f}")
# eval
items = 14042+35000+1500+2000+1319+5000+164+378+6511+9536+10042+1267+1172+11000+3610+400
print("TierA items", items, "GPU-h at 3s", items*3/3600)
# BM25 sizing
for toks in (6e9, 20e9):
    ch = toks/256; post = ch*150
    print(f"BM25 {toks/1e9:.0f}B tok: {ch/1e6:.0f}M chunks, {post:.1e} postings, raw 8B/post {post*8/1e9:.0f} GB, compressed ~1.5B/post {post*1.5/1e9:.0f} GB")
print("S2 neighbour-ids for 25e9/1024*... targets:", 25e9/4/1024, "targets; ids MB", 25e9/4/1024*8*4/1e6)
print("uint16 memmap 10B tokens GB", 10e9*2/1e9)
