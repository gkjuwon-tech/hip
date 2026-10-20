"""
김코어 v7.0 - 귀납 엔진: "규칙을 주지 마라, 예시만 보여줘라"

v6까지: 규칙(법칙 목록)을 받아서 다림질.
v7:     규칙은 숨겨져 있음. 색칠된 격자 예시 쌍 3~5개만 보고, 숨은 변환 프로그램을 스스로 찾아
        처음 보는 입력에 적용. (= ARC-AGI 형식, 진짜 ARC 데이터로 최종 시험)

  다림질 대응표
    산(울퉁불퉁)   = 현재 격자들과 목표 격자들의 차이 (모양 불일치 + 칸 불일치)
    다리미         = 격자 변환 부품 (회전/뒤집기/잘라내기/색바꾸기/타일/중력/채우기/반쪽 겹치기 ...)
    센서           = 매개변수(색, 색 대응표)는 예시에서 추론하고, 예시 전부에 맞을 때만 사용
    GNN 눈 대신    = 격자 눈: (현재, 목표) 관계를 보는 특징 + 부품 임베딩 -> 다음 수 점수
    범용 다리미    = 여러 문제에서 반복된 부품 연쇄를 새 부품으로 등록 (문제를 건너는 기술)
    자가 출제      = 숨은 프로그램을 무작위로 만들어 문제를 냄 -> 풀면서 학습 (선생님 데이터 0)

  정직성 장치
    - 대조군: 같은 탐색, 학습 0, 범용 다리미 0
    - 합성 시험: 학습보다 깊은 숨은 프로그램(3~5단계), 다른 시드
    - 과적합 거짓말 카운터: 예시는 전부 맞췄는데 시험 입력에서 틀린 경우를 따로 집계
    - 진짜 ARC: 시험 출력과 '정확히' 일치할 때만 정답 (ARC Prize 규칙대로 2회 시도)
"""
import os, sys, time, json, math, random, copy, heapq, glob, collections, itertools
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = dict(
    SEED=int(os.environ.get('V7_SEED', 1234)),
    GENS=int(os.environ.get('V7_GENS', 20)),
    TASKS_PER_GEN=120, TRAIN_BUDGET=80, EVAL_BUDGET=int(os.environ.get('V7_EVAL_BUDGET', 30)), ARC_BUDGET=int(os.environ.get('V7_ARC_BUDGET', 150)),
    MAX_DEPTH=4, WORKERS=int(os.environ.get('V7_WORKERS', 4)),
    MACRO_MIN_TASKS=6, MAX_MACROS=20, EVAL_EVERY=5,
    ARC_DIR=os.environ.get('ARC_DIR', os.path.join(os.path.dirname(HERE), 'arc', 'data')),
)
OUT = os.environ.get('V7_OUT', os.path.join(HERE, f"v7_out_s{CFG['SEED']}"))
os.makedirs(OUT, exist_ok=True)
_T0 = time.time()
_LOG = open(os.path.join(OUT, 'run.txt'), 'w', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); _LOG.write(s + '\n'); _LOG.flush()

# ==============================================================================
# 1. 격자 도구
# ==============================================================================
def G(x): return np.asarray(x, dtype=np.int8)
def key(g): return (g.shape, g.tobytes())
def keys(gs): return tuple(key(g) for g in gs)

def components(g, bg=0, diag=False):
    """같은 색으로 연결된 물체들: [(색, [(r,c)...])]"""
    h, w = g.shape; seen = np.zeros((h, w), bool); out = []
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1)] + ([(1, 1), (1, -1), (-1, 1), (-1, -1)] if diag else [])
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == bg: continue
            col = g[r, c]; stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in nb:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and g[yy, xx] == col:
                        seen[yy, xx] = True; stack.append((yy, xx))
            out.append((int(col), cells))
    return out

def bbox_crop(g, cells):
    ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
    return g[min(ys):max(ys) + 1, min(xs):max(xs) + 1]

def enclosed_mask(g):
    """배경(0) 칸 중 테두리와 연결되지 않은 칸 = 둘러싸인 칸"""
    h, w = g.shape; reach = np.zeros((h, w), bool); stack = []
    for r in range(h):
        for c in (0, w - 1):
            if g[r, c] == 0 and not reach[r, c]: reach[r, c] = True; stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if g[r, c] == 0 and not reach[r, c]: reach[r, c] = True; stack.append((r, c))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not reach[yy, xx] and g[yy, xx] == 0:
                reach[yy, xx] = True; stack.append((yy, xx))
    return (g == 0) & ~reach

def gravity(g, d):
    t = {'down': (g, False), 'up': (g[::-1], True), 'right': (g.T, False), 'left': (g.T[::-1], True)}
    a, flip = t[d]
    out = np.zeros_like(a)
    for c in range(a.shape[1]):
        col = a[:, c]; nz = col[col != 0]
        if len(nz): out[-len(nz):, c] = nz
    if d == 'down': return out
    if d == 'up': return out[::-1]
    if d == 'right': return out.T
    return out[::-1].T

def split_halves(g, axis):
    """가운데 구분선(한 색 줄)이 있거나 정확히 반으로 나뉘면 두 반쪽"""
    n = g.shape[axis]
    for k in range(1, n - 1):
        line = g[k, :] if axis == 0 else g[:, k]
        if (line == line[0]).all() and line[0] != 0 and k == n - 1 - k:
            a = g[:k] if axis == 0 else g[:, :k]; b = g[k + 1:] if axis == 0 else g[:, k + 1:]
            return a, b
    if n % 2 == 0:
        k = n // 2
        return (g[:k], g[k:]) if axis == 0 else (g[:, :k], g[:, k:])
    return None

# ==============================================================================
# 2. 다리미 (변환 부품): infer(현재들, 목표들) -> 매개변수 or None,  apply(격자, 매개변수) -> 격자 or None
# ==============================================================================
PRIMS = collections.OrderedDict()
def prim(name, infer=None):
    def deco(fn):
        PRIMS[name] = (infer or (lambda cur, tgt: ()), fn); return fn
    return deco

prim('rot90')(lambda g, p: np.rot90(g, -1))
prim('rot180')(lambda g, p: np.rot90(g, 2))
prim('rot270')(lambda g, p: np.rot90(g, 1))
prim('flip_lr')(lambda g, p: g[:, ::-1])
prim('flip_ud')(lambda g, p: g[::-1])
prim('transpose')(lambda g, p: g.T)
prim('anti_transpose')(lambda g, p: np.rot90(g, 2).T)

@prim('crop_nonzero')
def _crop(g, p):
    ys, xs = np.nonzero(g)
    if len(ys) == 0: return None
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def _obj_pick(g, mode):
    objs = components(g, diag=True)
    if not objs: return None
    if mode == 'largest': o = max(objs, key=lambda o: len(o[1]))
    elif mode == 'smallest': o = min(objs, key=lambda o: len(o[1]))
    else:
        cnt = collections.Counter(c for c, _ in objs)
        u = [o for o in objs if cnt[o[0]] == 1]
        if len(u) != 1: return None
        o = u[0]
    return bbox_crop(g, o[1])

prim('largest_obj')(lambda g, p: _obj_pick(g, 'largest'))
prim('smallest_obj')(lambda g, p: _obj_pick(g, 'smallest'))
prim('unique_color_obj')(lambda g, p: _obj_pick(g, 'unique'))
prim('upscale2')(lambda g, p: np.kron(g, np.ones((2, 2), np.int8)) if max(g.shape) <= 15 else None)
prim('upscale3')(lambda g, p: np.kron(g, np.ones((3, 3), np.int8)) if max(g.shape) <= 10 else None)

@prim('downscale2')
def _ds2(g, p):
    h, w = g.shape
    return g[::2, ::2] if h % 2 == 0 and w % 2 == 0 and h > 1 else None

@prim('downscale3')
def _ds3(g, p):
    h, w = g.shape
    return g[::3, ::3] if h % 3 == 0 and w % 3 == 0 and h > 2 else None

prim('tile_2x2')(lambda g, p: np.tile(g, (2, 2)) if max(g.shape) <= 15 else None)
prim('tile_1x2')(lambda g, p: np.tile(g, (1, 2)) if g.shape[1] <= 15 else None)
prim('tile_2x1')(lambda g, p: np.tile(g, (2, 1)) if g.shape[0] <= 15 else None)
prim('tile_3x3')(lambda g, p: np.tile(g, (3, 3)) if max(g.shape) <= 10 else None)
prim('mirror_lr')(lambda g, p: np.concatenate([g, g[:, ::-1]], 1) if g.shape[1] <= 15 else None)
prim('mirror_ud')(lambda g, p: np.concatenate([g, g[::-1]], 0) if g.shape[0] <= 15 else None)
prim('mirror_4')(lambda g, p: np.concatenate([np.concatenate([g, g[:, ::-1]], 1), np.concatenate([g[::-1], g[::-1, ::-1]], 1)], 0) if max(g.shape) <= 15 else None)
prim('sym_complete_lr')(lambda g, p: np.where(g == 0, g[:, ::-1], g))
prim('sym_complete_ud')(lambda g, p: np.where(g == 0, g[::-1], g))
prim('gravity_down')(lambda g, p: gravity(g, 'down'))
prim('gravity_up')(lambda g, p: gravity(g, 'up'))
prim('gravity_left')(lambda g, p: gravity(g, 'left'))
prim('gravity_right')(lambda g, p: gravity(g, 'right'))
prim('outer_border_only')(lambda g, p: (lambda m: np.where(m, g, 0))(np.pad(np.zeros((g.shape[0] - 2, g.shape[1] - 2), bool), 1, constant_values=True)) if min(g.shape) > 2 else None)

def _infer_recolor(cur, tgt):
    m = {}
    for a, b in zip(cur, tgt):
        if a.shape != b.shape: return None
        for x, y in zip(a.ravel().tolist(), b.ravel().tolist()):
            if m.setdefault(x, y) != y: return None
    if all(k == v for k, v in m.items()): return None
    return tuple(sorted(m.items()))

@prim('recolor', _infer_recolor)
def _recolor(g, p):
    m = dict(p); out = g.copy()
    for k, v in m.items(): out[g == k] = v
    return out

def _target_single_color(cur, tgt):
    cols = set()
    for b in tgt: cols |= set(np.unique(b).tolist())
    cols.discard(0)
    return (cols.pop(),) if len(cols) == 1 else None

def _infer_fill(cur, tgt):
    col = None
    for a, b in zip(cur, tgt):
        if a.shape != b.shape: return None
        m = enclosed_mask(a)
        if not m.any(): return None
        vals = set(b[m].tolist())
        if len(vals) != 1: return None
        v = vals.pop()
        if v == 0 or (col is not None and col != v): return None
        col = v
    return (col,)

prim('fill_enclosed', _infer_fill)(lambda g, p: (lambda m: np.where(m, p[0], g).astype(np.int8))(enclosed_mask(g)))

def _infer_keep(cur, tgt):
    cols = set.intersection(*[set(np.unique(b).tolist()) - {0} for b in tgt]) if tgt else set()
    return (cols.pop(),) if len(cols) == 1 else None
prim('keep_color', _infer_keep)(lambda g, p: np.where(g == p[0], g, 0).astype(np.int8))

def _infer_remove(cur, tgt):
    tc = set().union(*[set(np.unique(b).tolist()) for b in tgt])
    cc = set().union(*[set(np.unique(a).tolist()) for a in cur]) - tc - {0}
    return (cc.pop(),) if len(cc) == 1 else None
prim('remove_color', _infer_remove)(lambda g, p: np.where(g == p[0], 0, g).astype(np.int8))

def _halves_op(op, axis):
    def fn(g, p):
        s = split_halves(g, axis)
        if s is None or s[0].shape != s[1].shape: return None
        a, b = s[0] != 0, s[1] != 0
        m = {'and': a & b, 'or': a | b, 'xor': a ^ b, 'nor': ~(a | b)}[op]
        return np.where(m, p[0], 0).astype(np.int8)
    return fn
for op in ('and', 'or', 'xor', 'nor'):
    for ax, nm in ((1, 'lr'), (0, 'ud')):
        prim(f'halves_{op}_{nm}', _target_single_color)(_halves_op(op, ax))

PRIM_NAMES = list(PRIMS)

# ==============================================================================
# 3. 프로그램 실행 (매개변수는 예시에서 한 번 추론해 고정 -> 시험 입력에 그대로 적용)
# ==============================================================================
def step(cur, tgt, name, macros):
    """한 수 (부품 또는 범용 다리미) 적용 -> (새 현재들, [(부품, 매개변수)...]) or None"""
    seq = macros[name] if name in macros else (name,)
    prog = []
    for nm in seq:
        infer, fn = PRIMS[nm]
        try:
            p = infer(cur, tgt)
            if p is None: return None
            nxt = [fn(g, p) for g in cur]
        except Exception:
            return None
        if any(x is None or x.size == 0 or max(x.shape) > 30 for x in nxt): return None
        cur = [np.ascontiguousarray(x, dtype=np.int8) for x in nxt]; prog.append((nm, p))
    return cur, prog

def run_program(prog, g):
    for nm, p in prog:
        try:
            g = PRIMS[nm][1](g, p)
        except Exception:
            return None
        if g is None or g.size == 0: return None
        g = np.ascontiguousarray(g, dtype=np.int8)
    return g

# ==============================================================================
# 4. 격자 눈: (현재들, 목표들) 관계 특징
# ==============================================================================
def sym(g): return [float(np.array_equal(g, g[:, ::-1])), float(np.array_equal(g, g[::-1])),
                    float(g.shape[0] == g.shape[1] and np.array_equal(g, g.T))]

def is_sub(a, b):
    """b가 a 안의 부분 격자인가"""
    H, W = a.shape; h, w = b.shape
    if h > H or w > W or (h == H and w == W): return 0.0
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if np.array_equal(a[r:r + h, c:c + w], b): return 1.0
    return 0.0

NFEAT = 32
def features(cur, tgt, depth):
    fs = []
    for a, b in zip(cur, tgt):
        same = a.shape == b.shape
        diff = float((a != b).mean()) if same else 1.0
        ha = np.bincount(a.ravel(), minlength=10) / a.size; hb = np.bincount(b.ravel(), minlength=10) / b.size
        ca, cb = set(np.unique(a).tolist()), set(np.unique(b).tolist())
        f = [float(same), math.log(b.shape[0] / a.shape[0]), math.log(b.shape[1] / a.shape[1]),
             math.log(b.size / a.size), diff, float(np.abs(ha - hb).sum()) / 2,
             len(ca) / 10, len(cb) / 10, float(cb <= ca), float(ca <= cb),
             *sym(a), *sym(b), float(a.shape == b.shape[::-1] and a.shape[0] != a.shape[1]),
             is_sub(a, b) if a.size <= 300 else 0.0, float(b.shape[0] % a.shape[0] == 0 and b.shape[1] % a.shape[1] == 0 and b.size > a.size),
             float(a.shape[0] % b.shape[0] == 0 and a.shape[1] % b.shape[1] == 0 and a.size > b.size),
             min(len(components(a)), 20) / 20 if a.size <= 400 else 0.5, min(len(components(b)), 20) / 20 if b.size <= 400 else 0.5,
             float((a == 0).mean()), float((b == 0).mean()), float(enclosed_mask(a).any()) if a.size <= 400 else 0.0,
             float(same and ((a != b) & (a == 0)).sum() == (a != b).sum()), float(same and ((a != b) & (b == 0)).sum() == (a != b).sum())]
        fs.append(f)
    v = np.mean(np.array(fs, dtype=np.float32), 0)
    out = np.zeros(NFEAT, np.float32)
    x = np.concatenate([v, np.array([depth / 4.0, 1.0], np.float32)])[:NFEAT]
    out[:len(x)] = x
    return out

def distance(cur, tgt):
    d = 0.0
    for a, b in zip(cur, tgt):
        if a.shape != b.shape:
            d += 1.0 + abs(math.log(b.size / a.size)) * 0.3
        else:
            d += float((a != b).mean())
    return d / len(cur)

# ==============================================================================
# 5. 뇌: 자식 상태 특징 + 부품 임베딩 -> 점수 / 가치
# ==============================================================================
class InductionCore(nn.Module):
    def __init__(self, n_actions, D=64):
        super().__init__()
        self.emb = nn.Embedding(128, 24)
        self.f = nn.Sequential(nn.Linear(NFEAT, D), nn.GELU(), nn.Linear(D, D), nn.GELU())
        self.pol = nn.Sequential(nn.Linear(D + 24 + NFEAT, D), nn.GELU(), nn.Linear(D, 1))
        self.val = nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, 1))
        for h in (self.pol[-1], self.val[-1]):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    def score(self, parent_f, child_f, act_ids):
        hc = self.f(child_f)
        pf = parent_f.expand(len(act_ids), -1)
        logit = self.pol(torch.cat([hc, self.emb(act_ids), pf], -1)).squeeze(-1)
        val = torch.sigmoid(self.val(hc).squeeze(-1))
        return logit, val

# ==============================================================================
# 6. 탐색 (최우선 다림질)
# ==============================================================================
def solve(task, net, act_names, macros, budget, explore=False, rng=None, want=2, record=True):
    ins = [G(p['input']) for p in task['train']]; outs = [G(p['output']) for p in task['train']]
    tests = [G(p['input']) for p in task['test']]
    aid = {a: i for i, a in enumerate(act_names)}
    start = ins
    res = dict(solutions=[], expansions=0, steps=[], neg=[])
    heap = [(distance(start, outs), 0, start, (), (), 0.0)]
    seen = {keys(start)}; uid = 0; found_preds = set()
    while heap and res['expansions'] < budget:
        _, _, cur, prog, path, logp = heapq.heappop(heap)
        res['expansions'] += 1
        depth = len(path)
        if depth >= CFG['MAX_DEPTH']: continue
        children = []
        for a in act_names:
            r = step(cur, outs, a, macros)
            if r is None: continue
            nxt, p = r
            k = keys(nxt)
            if k in seen: continue
            seen.add(k)
            children.append((a, nxt, p))
        if not children: continue
        for a, nxt, p in children:
            if all(np.array_equal(x, y) for x, y in zip(nxt, outs)):
                full = prog + tuple(p)
                preds = [run_program(full, t) for t in tests]
                if any(x is None for x in preds): continue
                pk = tuple(key(x) for x in preds)
                if pk in found_preds: continue
                found_preds.add(pk)
                res['solutions'].append((full, preds, path + (a,)))
                if len(res['solutions']) == 1: res['first_at'] = res['expansions']
                if len(res['solutions']) >= want:
                    return res
        pf = torch.from_numpy(features(cur, outs, depth))[None]
        cf = torch.from_numpy(np.stack([features(nxt, outs, depth + 1) for _, nxt, _ in children]))
        ids = torch.tensor([aid[a] for a, _, _ in children])
        with torch.no_grad():
            lg, v = net.score(pf, cf, ids)
            if explore: lg = lg + 0.7 * torch.from_numpy(rng.gumbel(size=len(children))).float()
            lp = torch.log_softmax(lg, 0).numpy(); v = v.numpy()
            # 정책은 좋은 수를 '밀어줄' 수만 있고, 균등 분포보다 더 깎지는 못함 (처음 보는 문제에서 탐색이 막히지 않게)
            lp = np.maximum(lp, -math.log(len(children)))
        if record:
            res['neg'].append((pf[0].numpy(), cf.numpy(), ids.numpy()))
        for k, (a, nxt, p) in enumerate(children):
            d = distance(nxt, outs)
            pri = 0.5 * (logp + float(lp[k])) + 1.0 * math.log(max(float(v[k]), 0.05)) - 3.0 * d - 0.15 * (depth + 1)
            uid += 1
            heapq.heappush(heap, (-pri, uid, nxt, prog + tuple(p), path + (a,), logp + float(lp[k])))
    return res

# ==============================================================================
# 7. 자가 출제: 숨은 프로그램 무작위 생성 -> 문제
# ==============================================================================
def rand_input(rng):
    h, w = rng.randint(3, 10), rng.randint(3, 10)
    g = np.zeros((h, w), np.int8)
    style = rng.random()
    if style < 0.5:
        for _ in range(rng.randint(1, 4)):
            c = rng.randint(1, 9); hh, ww = rng.randint(1, max(1, h // 2)), rng.randint(1, max(1, w // 2))
            r0, c0 = rng.randint(0, h - hh), rng.randint(0, w - ww)
            g[r0:r0 + hh, c0:c0 + ww] = c
    elif style < 0.8:
        cols = rng.sample(range(1, 10), rng.randint(1, 3))
        mask = np.array([[rng.random() < 0.35 for _ in range(w)] for _ in range(h)])
        g[mask] = [rng.choice(cols) for _ in range(int(mask.sum()))]
    else:
        c = rng.randint(1, 9); g[0, :] = c; g[-1, :] = c; g[:, 0] = c; g[:, -1] = c
        if h > 4 and w > 4 and rng.random() < 0.5:
            g[h // 2, 1:-1] = rng.randint(1, 9)
    return g

PARAMLESS = [n for n in PRIM_NAMES if PRIMS[n][0].__name__ == '<lambda>']

def rand_program(rng, length):
    prog = []
    for _ in range(length):
        nm = rng.choice(PARAMLESS + ['recolor', 'keep_color', 'remove_color'])
        if nm == 'recolor':
            a, b = rng.sample(range(1, 10), 2); p = ((a, b),)
            p = tuple(sorted(((0, 0),) + p + tuple((c, c) for c in range(1, 10) if c != a)))
        elif nm in ('keep_color', 'remove_color'):
            p = (rng.randint(1, 9),)
        else:
            p = ()
        prog.append((nm, p))
    return prog

def make_task(rng, length):
    for _ in range(50):
        prog = rand_program(rng, length)
        pairs = []
        for _ in range(4):
            gi = rand_input(rng)
            if any(nm in ('keep_color', 'remove_color') for nm, _ in prog):
                cols = [p[0] for nm, p in prog if nm in ('keep_color', 'remove_color')]
                gi[rng.randrange(gi.shape[0]), rng.randrange(gi.shape[1])] = cols[0]
            go = run_program(prog, gi)
            if go is None or max(go.shape) > 30: break
            pairs.append((gi, go))
        if len(pairs) < 4: continue
        if all(np.array_equal(a, b) for a, b in pairs): continue
        if all(a.shape == b.shape and np.array_equal(a, b) for a, b in pairs[:3]): continue
        if len(set(key(b) for _, b in pairs)) < 3: continue
        return dict(train=[dict(input=a.tolist(), output=b.tolist()) for a, b in pairs[:3]],
                    test=[dict(input=pairs[3][0].tolist(), output=pairs[3][1].tolist())],
                    hidden=[n for n, _ in prog])
    return None

# ==============================================================================
# 8. 범용 다리미 (문제를 건너는 부품 연쇄)
# ==============================================================================
def mine_macros(solved_paths, macros, prim_names):
    cnt = collections.Counter()
    for path in solved_paths:
        flat = []
        for a in path: flat += list(macros.get(a, (a,)))
        seen = set()
        for k in (2, 3):
            for i in range(len(flat) - k + 1):
                seg = tuple(flat[i:i + k])
                if seg not in seen: seen.add(seg); cnt[seg] += 1
    new = {}
    for seg, c in cnt.most_common():
        if c < CFG['MACRO_MIN_TASKS'] or len(new) + len(macros) >= CFG['MAX_MACROS']: break
        name = '+'.join(seg)
        if name not in macros and name not in prim_names: new[name] = seg
    return new

# ==============================================================================
# 9. 병렬 실행
# ==============================================================================
_G = {}
def _job(args):
    task, budget, explore, seed, want = args
    return solve(task, _G['net'], _G['acts'], _G['macros'], budget, explore, np.random.default_rng(seed), want)

def run_parallel(net, acts, macros, tasks, budget, explore=False, seed=0, want=2):
    net.eval(); _G.update(net=net, acts=acts, macros=macros)
    args = [(t, budget, explore, seed * 7919 + i, want) for i, t in enumerate(tasks)]
    with mp.get_context('fork').Pool(CFG['WORKERS']) as pool:
        return pool.map(_job, args, chunksize=1)

def ranked(res):
    """오컴 + 다수결: 같은 시험 답을 내는 프로그램이 많을수록, 짧을수록 우선"""
    groups = collections.OrderedDict()
    for prog, preds, path in res['solutions']:
        k = tuple(key(x) for x in preds)
        g = groups.setdefault(k, [0, 99, prog, preds, path])
        g[0] += 1; g[1] = min(g[1], len(prog))
    order = sorted(groups.values(), key=lambda g: (-g[0], g[1]))
    return [(g[2], g[3], g[4]) for g in order]

def correct(task, res, attempts=2):
    gold = [G(p['output']) for p in task['test']]
    for prog, preds, _ in ranked(res)[:attempts]:
        if all(p.shape == g.shape and np.array_equal(p, g) for p, g in zip(preds, gold)):
            return True
    return False

def evaluate(net, acts, macros, tasks, budget, want=6):
    rs = run_parallel(net, acts, macros, tasks, budget, want=want)
    ok = sum(correct(t, r) for t, r in zip(tasks, rs))
    fit = sum(bool(r['solutions']) for r in rs)
    over = sum(bool(r['solutions']) and not correct(t, r) for t, r in zip(tasks, rs))
    fa = [r['first_at'] for r in rs if r['solutions']]
    return dict(acc=100.0 * ok / len(tasks), ok=ok, n=len(tasks), fit=fit, overfit=over, results=rs,
                first=float(np.mean(fa)) if fa else 0.0)

# ==============================================================================
# 10. 학습 데이터: 해 경로를 다시 걸으며 (부모, 형제들, 선택) 수집
# ==============================================================================
def collect(task, path, acts, macros):
    ins = [G(p['input']) for p in task['train']]; outs = [G(p['output']) for p in task['train']]
    cur = ins; data = []
    aid = {a: i for i, a in enumerate(acts)}
    for d, chosen in enumerate(path):
        sib, feats, ids, ci = [], [], [], None
        for a in acts:
            r = step(cur, outs, a, macros)
            if r is None: continue
            if a == chosen: ci = len(ids)
            feats.append(features(r[0], outs, d + 1)); ids.append(aid[a]); sib.append(r[0])
        if ci is None: return data
        data.append((features(cur, outs, d), np.stack(feats), np.array(ids), ci, len(path) - d))
        cur = sib[ci]
    return data

class Trainer:
    def __init__(self, net):
        self.net = net; self.opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        self.buf = collections.deque(maxlen=6000); self.neg = collections.deque(maxlen=6000)
    def train(self, iters=200, bs=32):
        if len(self.buf) < 16: return 0.0
        net = self.net; net.train(); tl = 0.0; buf = list(self.buf); neg = list(self.neg)
        for _ in range(iters):
            batch = random.sample(buf, min(bs, len(buf))); loss = 0.0
            for pf, cf, ids, ci, togo in batch:
                lg, v = net.score(torch.from_numpy(pf)[None], torch.from_numpy(cf), torch.from_numpy(ids).long())
                loss = loss + F.cross_entropy(lg[None], torch.tensor([ci]))
                # 가치: 선택된 수만 정답 라벨 (선택 안 된 형제도 다른 길로 풀릴 수 있으므로 0으로 가르치지 않음)
                loss = loss + 0.5 * F.binary_cross_entropy(v[ci:ci + 1].clamp(1e-4, 1 - 1e-4), torch.tensor([0.85 ** (togo - 1)]))
            if neg:
                for pf, cf, ids in random.sample(neg, min(8, len(neg))):
                    _, v = net.score(torch.from_numpy(pf)[None], torch.from_numpy(cf), torch.from_numpy(ids).long())
                    loss = loss + 0.2 * F.binary_cross_entropy(v.clamp(1e-4, 1 - 1e-4), torch.zeros(len(ids)))
            loss = loss / len(batch)
            self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); self.opt.step()
            tl += float(loss.detach())
        net.eval(); return tl / iters

# ==============================================================================
# 11. 진짜 ARC
# ==============================================================================
def load_arc(split):
    tasks = []
    for f in sorted(glob.glob(os.path.join(CFG['ARC_DIR'], split, '*.json'))):
        d = json.load(open(f)); d['id'] = os.path.basename(f)[:-5]; tasks.append(d)
    return tasks

# ==============================================================================
# 12. 메인
# ==============================================================================
def main():
    random.seed(CFG['SEED']); np.random.seed(CFG['SEED']); torch.manual_seed(CFG['SEED'])
    rng = random.Random(CFG['SEED'])
    acts = list(PRIM_NAMES); macros = {}
    net = InductionCore(128); control = copy.deepcopy(net); trainer = Trainer(net)
    hold_rng = random.Random(777)
    held = []
    while len(held) < 150:
        t = make_task(hold_rng, hold_rng.choice([3, 3, 4, 4, 5]))
        if t: held.append(t)
    arc_tr, arc_ev = load_arc('training'), load_arc('evaluation')
    if os.environ.get('V7_ARC_N'):
        k = int(os.environ['V7_ARC_N']); arc_tr, arc_ev = arc_tr[:k], arc_ev[:k]
    if os.environ.get('V7_HELD_N'):
        held = held[:int(os.environ['V7_HELD_N'])]
    log("=" * 100)
    log(" [김코어 v7.0] 귀납 엔진: 규칙은 숨김, 예시만 보고 숨은 변환 프로그램 발견")
    log(f"  변환 부품 {len(acts)}개 | 합성 시험 {len(held)}문제 (숨은 프로그램 3~5단계, 다른 시드) | 진짜 ARC 연습 {len(arc_tr)} + 평가 {len(arc_ev)}")
    log("=" * 100)
    t = time.time()
    c_held = evaluate(control, acts, {}, held, CFG['EVAL_BUDGET'])
    log(f"\n📊 [대조군: 같은 탐색, 학습 0, 범용 다리미 0] 합성 시험 {c_held['acc']:.1f}% (예시 적합 {c_held['fit']}, 과적합 거짓말 {c_held['overfit']}, 첫 해까지 평균 {c_held['first']:.1f}수 탐색) ({time.time() - t:.0f}s)")
    metrics = dict(config=CFG, control_held=c_held['acc'], gens=[])
    length = 1
    for gen in range(1, CFG['GENS'] + 1):
        tg = time.time()
        tasks = []
        while len(tasks) < CFG['TASKS_PER_GEN']:
            tt = make_task(rng, rng.randint(1, length))
            if tt: tasks.append(tt)
        rs = run_parallel(net, acts, macros, tasks, CFG['TRAIN_BUDGET'], explore=True, seed=gen, want=1)
        solved = []; okc = 0
        for tk, r in zip(tasks, rs):
            if r['solutions']:
                path = r['solutions'][0][2]
                if correct(tk, r, 1): okc += 1
                solved.append(path)
                trainer.buf.extend(collect(tk, path, acts, macros))
            else:
                trainer.neg.extend(r['neg'][:4])
        rate = len(solved) / len(tasks)
        loss = trainer.train()
        new = mine_macros(solved, macros, set(acts))
        for nm, seg in new.items():
            macros[nm] = seg; acts.append(nm)
        rec = dict(gen=gen, length=length, rate=rate, test_ok=okc / len(tasks), macros=len(macros), loss=loss, secs=time.time() - tg)
        line = (f"[제 {gen:02d} 세대 | 숨은 프로그램 최대 {length}단계] 자가 출제 {len(tasks)}문제 중 {len(solved)}개 해결 "
                f"(시험 입력까지 정답 {okc}) | 범용 다리미 {len(macros)}개{(' 신규: ' + ', '.join(new)) if new else ''} | {rec['secs']:.0f}s")
        if gen % CFG['EVAL_EVERY'] == 0 or gen == CFG['GENS']:
            e = evaluate(net, acts, macros, held, CFG['EVAL_BUDGET'])
            rec.update(held=e['acc'], overfit=e['overfit'])
            line += f"\n   🔒 합성 시험 (3~5단계, 예산 {CFG['EVAL_BUDGET']}): 대조군 {c_held['acc']:.1f}% → 김코어 {e['acc']:.1f}% (과적합 거짓말 {e['overfit']}, 첫 해까지 {c_held['first']:.1f} → {e['first']:.1f}수)"
        log(line); metrics['gens'].append(rec)
        if rate >= 0.6 and length < 5: length += 1
        with open(os.path.join(OUT, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)

    # ---------------- 최종
    log("\n" + "=" * 100)
    log(" 🏁 [최종 시험]")
    e_held = evaluate(net, acts, macros, held, CFG['EVAL_BUDGET'])
    e_held_nomac = evaluate(net, PRIM_NAMES, {}, held, CFG['EVAL_BUDGET'])
    log(f"   합성 시험: 대조군 {c_held['acc']:.1f}% | 김코어(범용 다리미 끔) {e_held_nomac['acc']:.1f}% | 김코어 {e_held['acc']:.1f}%  (과적합 거짓말 {e_held['overfit']})")
    arc = {}
    for name, ts in (('ARC 연습세트', arc_tr), ('ARC 평가세트', arc_ev)):
        t = time.time()
        cc = evaluate(control, PRIM_NAMES, {}, ts, CFG['ARC_BUDGET'])
        ee = evaluate(net, acts, macros, ts, CFG['ARC_BUDGET'])
        arc[name] = dict(control=cc['ok'], kim=ee['ok'], n=len(ts), overfit=ee['overfit'])
        solved_ids = [t_['id'] for t_, r in zip(ts, ee['results']) if correct(t_, r)]
        log(f"   {name} {len(ts)}문제: 대조군 {cc['ok']}개 ({cc['acc']:.1f}%) | 김코어 {ee['ok']}개 ({ee['acc']:.1f}%) | "
            f"예시만 맞고 시험 틀림(과적합) {ee['overfit']}개 | {time.time() - t:.0f}s")
        for t_, r in zip(ts, ee['results']):
            if correct(t_, r):
                log(f"      ✅ {t_['id']}: {' → '.join(n for n, _ in r['solutions'][0][0])}")
    log("\n" + "=" * 100)
    log(" 🏆 [김코어 v7 최종 성적표]")
    log(f"   합성 시험 (처음 보는 숨은 규칙 3~5단계): 대조군 {c_held['acc']:.1f}% → 김코어 {e_held['acc']:.1f}%")
    for k, v in arc.items():
        log(f"   {k}: 대조군 {v['control']}/{v['n']} → 김코어 {v['kim']}/{v['n']}  (과적합 {v['overfit']})")
    log(f"   범용 다리미: {', '.join(macros) if macros else '없음'}")
    log(f"   총 {time.time() - _T0:.0f}s")
    metrics['final'] = dict(held_control=c_held['acc'], held=e_held['acc'], held_nomacro=e_held_nomac['acc'], arc=arc, macros=list(macros))
    with open(os.path.join(OUT, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)

if __name__ == '__main__':
    main()
