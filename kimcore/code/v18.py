"""
김코어 v18 - v17 + 탐험 후보: 개념 후보를 '이미 푼 풀이'에서만 뽑지 않고, 숙제가 필요로 할 것 같은 재료(인식망)로 새 조각을 지어서 쓸모 관문에 같이 올림 (생성 후 검증)
김코어 v17 - v16 + 학교(개념 교류): 한 마리가 발명한 개념을 다른 마리들에게 제안, 받는 쪽은 자기 숙제를 실제로 뚫을 때만 수입 (문화 전파)
김코어 v16 - v15 + 숙제 시간: 못 푼 문제(숙제 노트)를 큰 시간 예산으로 다시 붙잡고, 오래 고민해서 푼 풀이를 개념 발명 재료로 (정답지 X, 진짜로 푼 것만)
김코어 v15 - v14 + 속도: 출제를 한 번에 병렬로(팔별 따로 X), 관문 비용 절감, 단계별 시간 기록
김코어 v14 - v13 + 병렬 쓸모 관문: 후보 개념 여러 개를 동시에 시험해서 '못 풀던 문제를 가장 많이 뚫는' 개념을 채택 (빠르고 더 똑똑한 선택)
김코어 v13 - v12 + '인식망을 얼마나 믿을지'(rec_scale)도 개체군 진화가 결정 (0=무시, 1=그대로, >1=강하게)
김코어 v12 (D3) - v11 + 쓸모 기준 개념 발명: 개념 후보를 넣었을 때 '이번 세대에 못 푼 문제'가 실제로 풀려야 등록
                  (못 뚫어도 압축 이득이 아주 크면 예외 허용)
김코어 v11 (D2) - v9 개체군 진화 + v10 탐색 엔진(아래→위 + 관찰 동등성) + 고전 개념 탐지기(관찰 전용)
(아래는 v9 설명)
김코어 v9.0 - 개선하는 기계를 개선하는 기계

v8.1 위에 지수 성장 고리 3개를 추가:
  고리 1. 개념 신용 장부: 최근 K세대 동안 새 해결에 한 번도 안 쓰인 개념은 은퇴 (쓸모 있는 층만 곱해지게)
  고리 3. 학습 진척도 출제기: (출제 깊이 x 개념 사용 비율) 팔 중, 해결률이 가장 빨리 변하는 팔에서 더 많이 출제
  고리 4. 김코어 개체군 (Population-Based Training): 설정값이 다른 김코어 4마리.
          4세대마다 '처음 보는 문제' 시합 -> 꼴찌가 1등을 복제하고 설정값을 돌연변이.
          = 배우는 방법(탐색 성향, 개념 문턱, 출제 성향, 학습량) 자체가 진화.
측정
  - 도달 거리: 시합 문제에서 푼 프로그램의 기본 재료 기준 크기(상위 25% 평균) -> 로그 눈금 기울기
  - 고정 탐침 (v8.1과 동일 300문제, 학습 금지), 과거의 나 vs 지금의 나
"""
import os, sys, time, json, math, random, copy, collections
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(os.environ.get('V18_SEED', 1234))
OUT = os.environ.get('V18_OUT', os.path.join(HERE, f'v18_out_s{SEED}'))
os.makedirs(OUT, exist_ok=True)
os.environ['V9BASE_OUT'] = os.path.join(OUT, '_base')
sys.path.insert(0, HERE)
import numpy as np, torch
os.environ.setdefault('V8_TIME_CAP', '1.0')
os.environ['V10_OUT'] = os.path.join(OUT, '_v10')
import v9base as B
import v11search as S_
B.search = S_.v10.search                         # v10 엔진 장착 (첫 해 = 가장 짧은 해)
from v9base import Library, Recognizer, base_size, leaves, show, tshow, subterms, task_features, prod_counts

CFG = dict(GENS=int(os.environ.get('V18_GENS', 16)), POP=4, TASKS_PER_MEMBER=int(os.environ.get('V18_TASKS', 40)),
           PBT_EVERY=4, SELECT_N=int(os.environ.get('V18_SELECT', 60)), RETIRE_AFTER=4, PROBE_EVERY=8,
           GATE_TASKS=int(os.environ.get('V18_GATE_TASKS', 12)), HOMEWORK=48, STUDY_N=int(os.environ.get('V18_STUDY_N', 0)), STUDY_CAP=float(os.environ.get('V18_STUDY_CAP', 4.0)), N_CAND=int(os.environ.get('V18_N_CAND', 6)), GATE_CAP=float(os.environ.get('V18_GATE_CAP', 0.3)),
           N_EXPLORE=int(os.environ.get('V18_N_EXPLORE', 10)))
_LOG = open(os.path.join(OUT, 'run.txt'), 'w', encoding='utf-8'); _T0 = time.time()
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); _LOG.write(s + '\n'); _LOG.flush()

ARMS = [(d, cb) for d in (3, 4, 5, 6, 7) for cb in (0.0, 0.5, 1.5)]

class Member:
    def __init__(self, k, rng):
        self.k = k; self.lib = Library(); self.net = Recognizer(); self.corpus = {}
        self.rec = collections.deque(maxlen=8000)
        self.hp = dict(EPS=float(rng.choice([0.05, 0.08, 0.12, 0.16])), MIN_GAIN_FRAC=float(rng.choice([0.005, 0.01, 0.02])),
                       easy_keep=float(rng.choice([0.1, 0.2, 0.35])), rec_iters=int(rng.choice([200, 300, 500])),
                       rec_scale=float([0.0, 0.5, 1.0, 1.5][k % 4]))
        self.p = {a: 0.5 for a in ARMS}; self.lp = {a: 0.3 for a in ARMS}
        self.credit = collections.defaultdict(list)   # 개념 -> 세대별 사용 횟수
        self.born = {}; self.retired = set(); self.score_hist = []; self.homework = []

    def arm_weights(self):
        return [self.lp[a] + 0.05 for a in ARMS]

    def apply_hp(self):
        B.CFG['EPS'] = self.hp['EPS']; B.CFG['MIN_GAIN_FRAC'] = self.hp['MIN_GAIN_FRAC']; B._G['rec_scale'] = self.hp['rec_scale']
        B._G['easy_keep'] = self.hp['easy_keep']; B._G['retired'] = set(self.retired)

    def mutate(self, rng):
        self.hp['rec_scale'] = float(np.clip(self.hp['rec_scale'] + rng.choice([-0.5, -0.25, 0.25, 0.5]), 0.0, 2.5))
        for kk in ('EPS', 'MIN_GAIN_FRAC', 'easy_keep'):
            self.hp[kk] = float(np.clip(self.hp[kk] * rng.choice([0.7, 0.85, 1.2, 1.4]), 0.002, 0.6))
        self.hp['rec_iters'] = int(np.clip(self.hp['rec_iters'] * rng.choice([0.7, 1.3]), 100, 900))

def concepts_used(p, lib):
    return set(st[1] for st, _ in subterms(p) if st[0] == 'p' and st[1] >= lib.n_base)

def member_gen(m, gen, seen_sigs, rng):
    """한 마리의 한 세대: 출제(학습 진척도 팔) -> 풀기 -> 신용 장부 -> 개념 발명 -> 은퇴 -> 꿈 -> 인식망"""
    m.apply_hp()
    w = np.array(m.arm_weights()); w = w / w.sum()
    arm_counts = collections.Counter(ARMS[i] for i in rng.choice(len(ARMS), size=CFG['TASKS_PER_MEMBER'], p=w))
    T = collections.OrderedDict(); t0 = time.time()
    tasks = gen_all(m, arm_counts, gen, seen_sigs); T['출제'] = time.time() - t0; t0 = time.time()
    B._G['retired'] = set(m.retired)
    rs = B.solve_all(m.lib, m.net, tasks, B.CFG['SEARCH_BUDGET']); T['풀기'] = time.time() - t0; t0 = time.time()
    solved = 0; sizes = []; used_now = collections.Counter(); by_arm = collections.defaultdict(list)
    for tk, r in zip(tasks, rs):
        ok = r['prog'] is not None and r['hidden_ok']
        by_arm[tk['arm']].append(ok)
        if not ok: continue
        solved += 1; tid = f"{m.k}-{gen}-{len(m.corpus)}"
        m.corpus[tid] = r['prog']; sizes.append(base_size(r['prog'], m.lib))
        for c in concepts_used(r['prog'], m.lib): used_now[c] += 1
        m.rec.append((task_features(tk), prod_counts(r['prog'], len(m.lib))))
    for a, oks in by_arm.items():                 # 학습 진척도 갱신
        new = float(np.mean(oks)); old = m.p[a]
        m.p[a] = 0.7 * old + 0.3 * new; m.lp[a] = 0.7 * m.lp[a] + 0.3 * abs(new - old) * 3
    for c in range(m.lib.n_base, len(m.lib)): m.credit[c].append(used_now.get(c, 0))
    failed = [tk for tk, r in zip(tasks, rs) if not (r['prog'] is not None and r['hidden_ok'])]
    # 숙제 노트: 여러 세대에 걸쳐 못 푼 문제 (최근 것 우선, 새로 풀린 건 지움)
    m.homework = (failed + [tk for tk in m.homework if tk.get('sig') not in set(t.get('sig') for t in tasks)])[:CFG['HOMEWORK']]
    # 숙제 시간: 숙제 노트의 오래된 문제부터 큰 시간 예산으로 (인식망 가중치 그대로) 다시 시도
    study = m.homework[-CFG['STUDY_N']:] if CFG['STUDY_N'] > 0 else []; n_study = 0
    if study:
        B._G.update(lib=m.lib, net=m.net, retired=set(m.retired))
        srs = B._pool(_job_study, study)
        done = set()
        for tk, r in zip(study, srs):
            if r['prog'] is not None and r['hidden_ok']:
                n_study += 1; done.add(id(tk))
                m.corpus[f"{m.k}-{gen}-s{len(m.corpus)}"] = r['prog']; sizes.append(base_size(r['prog'], m.lib))
                m.rec.append((task_features(tk), prod_counts(r['prog'], len(m.lib))))
        m.homework = [tk for tk in m.homework if id(tk) not in done]
    T['숙제시간'] = time.time() - t0; t0 = time.time()
    ilogs = invent_batch(m, m.homework[:CFG['GATE_TASKS']], random.Random(gen * 7 + m.k))
    if m.homework:                                     # 숙제 중 이번 개념으로 풀린 것은 지움
        B._G.update(gate_libs=[m.lib], gate_tasks=m.homework, retired=set(m.retired))
        ok_ = B._pool(_job_gate, [(0, ti) for ti in range(len(m.homework))])
        m.homework = [tk for tk, o in zip(m.homework, ok_) if not o]
    T['발명'] = time.time() - t0; t0 = time.time()
    for c in range(m.lib.n_base, len(m.lib)): m.born.setdefault(c, gen)
    retired_now = []
    for c in range(m.lib.n_base, len(m.lib)):    # 신용 장부: 최근 K세대 사용 0이면 은퇴
        if c in m.retired: continue
        h = m.credit[c]
        if gen - m.born[c] >= CFG['RETIRE_AFTER'] and len(h) >= CFG['RETIRE_AFTER'] and sum(h[-CFG['RETIRE_AFTER']:]) == 0:
            m.retired.add(c); retired_now.append(m.lib.names[c])
    dream = []
    Wd = np.zeros(len(m.lib) + 1); Wd[m.lib.n_base:len(m.lib)] = 1.0
    for c in m.retired: Wd[c] = -1e9
    if getattr(m, 'dream_key', None) == tuple(str(b) for b in m.lib.bodies) + tuple(sorted(m.retired)):
        dream = list(m.dream_cache)                     # 도서관이 그대로면 지난 꿈 재사용 (같은 분포)
    else:
        B._G.update(lib=m.lib, dream_W=Wd)
        for part in B._pool(_job_dream, [gen * 1000 + m.k * 10 + j for j in range(4)]): dream += part
        m.dream_key = tuple(str(b) for b in m.lib.bodies) + tuple(sorted(m.retired)); m.dream_cache = list(dream)
    T['꿈'] = time.time() - t0; t0 = time.time()
    B.train_recognizer(m.net, list(m.rec) + dream, len(m.lib), iters=m.hp['rec_iters'])
    T['인식망'] = time.time() - t0
    ilogs.append('  ⏱️ ' + ' '.join(f"{k} {v:.0f}s" for k, v in T.items()) + f" | 숙제 {len(m.homework)} (숙제 시간에 {n_study}/{len(study)} 해결)")
    return tasks, solved, sizes, ilogs, retired_now

from v9base import anti_unify, holes_to_body, infer, generalize, resolve, rewrite, frag_leaves, prim_leaves, compile_term, behavior, trivial_behavior, lib_signatures
import itertools as _it

def _job_gate(args):
    """(후보 번호, 문제 번호) -> 짧은 시간 안에 풀리나. 후보 -1 = 개념 없이 (기준선)"""
    ci, ti = args
    lib_ = B._G['gate_libs'][ci]; tk = B._G['gate_tasks'][ti]
    W = np.zeros(len(lib_) + 1)
    for i in B._G.get('retired') or (): W[i] = -1e9
    try:
        r = B.search(tk, lib_, W, 1000, time_cap=CFG['GATE_CAP'])
    except Exception:
        return False
    return r['prog'] is not None and r['hidden_ok']

def _candidates(lib, corpus, rng):
    """v8.1 발명기와 같은 후보 생성 (반통일화 + 압축 이득), 이득 순 정렬"""
    subs = []
    for tid, p in corpus.items():
        for st, d in subterms(p):
            if 3 <= leaves(st) <= 14 and st[0] in ('a', 'l'): subs.append((tid, st))
    if len(subs) < 4: return [], 0
    groups = collections.defaultdict(list)
    for tid, st in subs:
        groups[st[0] + (st[1][0] if st[0] == 'a' else 'l') + str(leaves(st) // 3)].append((tid, st))
    cands = set()
    for g in groups.values():
        g = g if len(g) <= 60 else rng.sample(g, 60)
        for (t1, a), (t2, b) in _it.combinations(g, 2):
            if t1 == t2: continue
            f, nh = anti_unify(a, b)
            if f is not None and nh <= 3 and prim_leaves(f) >= 2: cands.add((f, nh))
    before = sum(leaves(p) for p in corpus.values())
    scored = []
    for f, nh in cands:
        body = holes_to_body(f, nh)
        try:
            tt, s_ = infer(body, lib); tt = generalize(resolve(tt, s_))
        except (TypeError, RecursionError):
            continue
        used = 0; after = 0
        for tid, p in corpus.items():
            np_, n = rewrite(p, f, len(lib), nh); after += leaves(np_); used += (n > 0)
        if used < 2: continue
        gain = before - after - frag_leaves(f)
        if gain < max(2, B.CFG.get('MIN_GAIN_FRAC', 0.01) * before): continue
        scored.append((gain, f, nh, body, tt))
    scored.sort(key=lambda x: -x[0])
    return scored, before

def explore(m, fail_tasks, rng, sigs, bodies, n):
    """탐험 후보: 숙제 문제들에 대한 인식망 감(평균)으로 재료를 골라 작은 닫힌 프로그램을 지음. 의미가 새로운 것만"""
    if n <= 0 or not fail_tasks: return []
    lib = m.lib
    W = np.mean([B.weights_for(m.net, tk, len(lib)) for tk in fail_tasks], 0) * m.hp.get('rec_scale', 1.0)
    for c in m.retired: W[c] = -1e9
    nrng = np.random.default_rng(rng.randrange(2 ** 31)); out = []
    for _ in range(n * 15):
        if len(out) >= n: break
        tk = B.make_task(lib, W, nrng, maxdepth=int(nrng.integers(3, 6)), min_base=3, max_base=10)
        if tk is None: continue
        body = tk['src']
        if body in bodies: continue
        try:
            tt, s_ = infer(body, lib); tt = generalize(resolve(tt, s_)); val = compile_term(body, lib)(())
        except Exception:
            continue
        sg = behavior(val, tt)
        if sg is None or sg in sigs or trivial_behavior(sg, tt): continue
        sigs.add(sg); bodies.add(body)
        out.append((0, None, 0, body, tt, val))
    return out

def invent_batch(m, fail_tasks, rng):
    """병렬 쓸모 관문: 의미가 새로운 후보 N개를 한꺼번에 '못 푼 문제'에 꽂아 보고, 가장 많이 뚫는 것을 채택"""
    lib, corpus = m.lib, m.corpus; logs = []; n_rej = 0
    for _ in range(B.CFG['MAX_INVENT_PER_GEN']):
        if len(lib) >= B.CFG['MAX_PRODS'] or not fail_tasks: break
        scored, before = _candidates(lib, corpus, rng)
        sigs = set(lib_signatures(lib)); bodies = set(b for b in lib.bodies if b is not None)
        picks = []
        for gain, f, nh, body, tt in scored:
            if body in bodies: continue
            try: val = compile_term(body, lib)(())
            except Exception: continue
            sg = behavior(val, tt)
            if sg is not None and (sg in sigs or trivial_behavior(sg, tt)): n_rej += 1; continue
            if sg is not None: sigs.add(sg)
            picks.append((gain, f, nh, body, tt, val))
            if len(picks) >= CFG['N_CAND']: break
        picks += explore(m, fail_tasks, rng, sigs, bodies, CFG['N_EXPLORE'])
        if not picks: break
        libs = []
        for gain, f, nh, body, tt, val in picks:
            L = copy.copy(lib)
            for a in ('names', 'types', 'values', 'bodies', 'depth', 'base_leaves'): setattr(L, a, list(getattr(lib, a)))
            L.add('후보', tt, val, body, 1, 1); libs.append(L)
        libs.append(lib)                                # 마지막 = 기준선 (-1)
        B._G.update(gate_libs=libs, gate_tasks=fail_tasks, retired=set(m.retired))
        jobs = [(ci, ti) for ci in range(-1, len(picks)) for ti in range(len(fail_tasks))]
        res = dict(zip(jobs, B._pool(_job_gate, jobs)))
        base = [res[(-1, ti)] for ti in range(len(fail_tasks))]
        unl = [sum(1 for ti in range(len(fail_tasks)) if not base[ti] and res[(ci, ti)]) for ci in range(len(picks))]
        bi = max(range(len(picks)), key=lambda i: (unl[i], picks[i][0]))
        if unl[bi] == 0:
            logs.append(f"  🚪 [쓸모 관문] 후보 {len(picks)}개 전부 탈락 (못 푼 {len(fail_tasks)}문제 중 하나도 못 뚫음)"); break
        gain, f, nh, body, tt, val = picks[bi]
        depth = 1 + max([lib.depth[x[1]] for x, _ in subterms(body) if x[0] == 'p'] or [0])
        name = f"개념{len(lib) - lib.n_base + 1}"
        idx = lib.add(name, tt, val, body, depth, max(1, base_size(body, lib) - nh))
        if f is not None:
            for tid in list(corpus): corpus[tid] = rewrite(corpus[tid], f, idx, nh)[0]
        after = sum(leaves(p) for p in corpus.values())
        logs.append(f"  💡 [개념 발명{' (탐험)' if f is None else ''}] {name} : {tshow(tt)}  =  {show(body, lib)}   (압축 {before}→{after}, 층 {depth}) "
                    f"[후보 {len(picks)}개 중 최다 돌파: 못 풀던 {unl[bi]}/{len(fail_tasks)}]")
        fail_tasks = [tk for ti, tk in enumerate(fail_tasks) if not res[(bi, ti)]]   # 다음 개념은 남은 문제를 노림
    if n_rej: logs.append(f"  🗑️ [의미 중복/시시한 후보 {n_rej}개 거절]")
    return logs

def _job_study(task):
    return B.search(task, B._G['lib'], B._weights(task), 10 ** 9, time_cap=CFG['STUDY_CAP'])

def _job_dream(seed):
    lib = B._G['lib']; drng = np.random.default_rng(seed); out = []
    while len(out) < 60:
        tk = B.make_task(lib, B._G['dream_W'], drng, maxdepth=4)
        if tk is not None: out.append((task_features(tk), prod_counts(tk['src'], len(lib))))
    return out

def _job_gen_arm(args):
    seed, d, cb = args
    B._G['maxdepth'] = d; B._G['concept_bias'] = cb     # 작업자 프로세스 안에서만 바뀜
    t = B._job_gen(seed)
    if t is not None: t['arm'] = (d, cb)
    return t

def gen_all(m, arm_counts, gen, seen_sigs):
    """모든 팔의 문제를 풀 하나로 한 번에 출제 (팔마다 풀을 새로 띄우던 낭비 제거)"""
    B._G.update(lib=m.lib, net=m.net, seen_sigs=frozenset(seen_sigs), retired=set(m.retired))
    need = dict(arm_counts); out = []; sigs = set(); k = 0
    for _round in range(4):
        jobs = []
        for (d, cb), n in need.items():
            for i in range(n): jobs.append(((gen * 97 + m.k * 13 + d) * 1000003 + k + i, d, cb))
            k += n
        if not jobs: break
        for t in B._pool(_job_gen_arm, jobs):
            if t is not None and t.get('sig') not in sigs and need.get(t['arm'], 0) > 0:
                sigs.add(t.get('sig')); out.append(t); need[t['arm']] -= 1
        need = {a: n for a, n in need.items() if n > 0}
    return out

def inline(t, lib):
    """개념 참조를 본문으로 펼쳐서 기본 재료만으로 된 닫힌 항으로 (다른 개체 도서관에 옮기기 위해)"""
    k = t[0]
    if k == 'p': return inline(lib.bodies[t[1]], lib) if t[1] >= lib.n_base else t
    if k == 'l': return ('l', inline(t[1], lib))
    if k == 'a': return ('a', inline(t[1], lib), inline(t[2], lib))
    return t

def school(pop, n_before, gen):
    """학교: 이번 세대 새 개념을 다른 개체들에게 제안. 받는 쪽 숙제를 뚫는 것만 수입 (받는 쪽마다 최대 2개)"""
    offers = []
    for m in pop:
        for c in range(n_before[m.k], len(m.lib)):
            offers.append((m.k, m.lib.names[c], inline(m.lib.bodies[c], m.lib)))
    out = []
    if not offers: return out
    for r in pop:
        hw = r.homework[:CFG['GATE_TASKS']]
        if not hw: continue
        sigs = set(lib_signatures(r.lib)); bodies = set(inline(b, r.lib) for b in r.lib.bodies if b is not None)
        picks = []
        for src, nm, body in offers:
            if src == r.k or body in bodies: continue
            try:
                tt, s_ = infer(body, r.lib); tt = generalize(resolve(tt, s_)); val = compile_term(body, r.lib)(())
            except Exception:
                continue
            sg = behavior(val, tt)
            if sg is not None and (sg in sigs or trivial_behavior(sg, tt)): continue
            if sg is not None: sigs.add(sg)
            bodies.add(body); picks.append((src, nm, body, tt, val))
        if os.environ.get("SCHOOL_DEBUG"): print("picks", len(picks), "hw", len(hw))
        if not picks: continue
        for _round in range(2):
            if not picks or not hw: break
            libs = []
            for src, nm, body, tt, val in picks:
                L = copy.copy(r.lib)
                for a in ('names', 'types', 'values', 'bodies', 'depth', 'base_leaves'): setattr(L, a, list(getattr(r.lib, a)))
                L.add('후보', tt, val, body, 1, 1); libs.append(L)
            libs.append(r.lib)
            B._G.update(gate_libs=libs, gate_tasks=hw, retired=set(r.retired))
            jobs = [(ci, ti) for ci in range(-1, len(picks)) for ti in range(len(hw))]
            res = dict(zip(jobs, B._pool(_job_gate, jobs)))
            unl = [sum(1 for ti in range(len(hw)) if not res[(-1, ti)] and res[(ci, ti)]) for ci in range(len(picks))]
            bi = max(range(len(picks)), key=lambda i: unl[i])
            if os.environ.get("SCHOOL_DEBUG"): print("unl", unl)
            if unl[bi] == 0: break
            src, nm, body, tt, val = picks.pop(bi)
            name = f"개념{len(r.lib) - r.lib.n_base + 1}"
            r.lib.add(name, tt, val, body, 1, max(1, base_size(body, r.lib)))
            r.born[len(r.lib) - 1] = gen
            out.append(f"   🏫 [학교] 김코어#{r.k} ← 김코어#{src}의 {nm} 수입 → {name} = {show(body, r.lib)} [숙제 {unl[bi]}/{len(hw)} 뚫음]")
            hw = [tk for ti, tk in enumerate(hw) if not res[(bi, ti)]]
            r.homework = [tk for tk in r.homework if tk in hw or tk not in r.homework[:CFG['GATE_TASKS']]]
    return out

def selection_set(gen, seen_sigs):
    """시합 문제: 기본 재료로 만든 깊은 문제 중 0세대가 짧은 탐색으로 못 푸는 것 (매번 새로)"""
    B._G['easy_keep'] = 0.0; B._G['retired'] = set(); B.CFG['EPS'] = 0.08
    return [dict(type=t['type'], ex=t['ex'], base=t['base'])
            for t in B.gen_frontier(Library(), None, CFG['SELECT_N'], 6, 0.0, 5555 + gen, frozenset(seen_sigs))]

def compete(pop, tasks):
    out = []
    for m in pop:
        m.apply_hp()
        rs = B.solve_all(m.lib, m.net, tasks, B.CFG['SEARCH_BUDGET'])
        ok = [r for r in rs if r['prog'] is not None and r['hidden_ok']]
        sz = sorted([base_size(r['prog'], m.lib) for r in ok], reverse=True)
        reach = float(np.mean(sz[:max(1, len(sz) // 4)])) if sz else 0.0
        out.append((len(ok), reach))
    return out

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    pop = [Member(k, rng) for k in range(CFG['POP'])]
    probes = B.probe_set(9999)
    B._G['retired'] = set(); B.CFG['EPS'] = 0.08
    p0, _, ov0, _, rs0 = B.probe_eval(Library(), None, probes, ['쉬움'] * len(probes))
    levels = [B.level_of(r) for r in rs0]
    nfail = sum(1 for l in levels if l == '0세대 실패')
    seen_sigs = set(s for s in (B.task_sig(t, Library()) for t in probes) if s is not None)
    log("=" * 104)
    log(" [김코어 v18] 탐험 후보 + 학교(개념 교류) + 숙제 시간 + 고속화 + 병렬 쓸모 관문(최다 돌파 개념 채택) + 인식망 신뢰도 진화 + 쓸모 기준 개념 발명 + v10 엔진 + 개체군 4마리 진화 + 개념 신용 장부 + 학습 진척도 출제 + 고전 개념 탐지기")
    for m in pop: log(f"  김코어#{m.k} 설정: {m.hp}")
    log(f"  고정 탐침 {len(probes)}문제 (0세대 실패 {nfail}개) | 0세대 탐침 {p0:.1f}%")
    log("=" * 104)
    S0 = selection_set(0, seen_sigs)
    base_score = compete([Member(99, np.random.default_rng(0))], S0)[0]
    log(f"📊 [0세대 시합 기준] 처음 보는 깊은 문제 {len(S0)}개: 0세대 {base_score[0]}개 해결, 도달 거리 {base_score[1]:.1f}")
    metrics = dict(cfg=CFG, probe0=p0, nfail=nfail, base_select=base_score, gens=[], pbt=[])
    for gen in range(1, CFG['GENS'] + 1):
        tg = time.time(); lines = []
        n_before = {m.k: len(m.lib) for m in pop}
        for m in pop:
            tasks, solved, sizes, ilogs, retired_now = member_gen(m, gen, seen_sigs, rng)
            seen_sigs.update(t['sig'] for t in tasks if 'sig' in t)
            active = len(m.lib) - m.lib.n_base - len(m.retired)
            lines.append(f"   김코어#{m.k}: {solved}/{len(tasks)} | 푼 크기 평균 {np.mean(sizes) if sizes else 0:.1f} 최대 {max(sizes) if sizes else 0} | "
                         f"활성 개념 {active} (은퇴 {len(m.retired)}) | 최고 {max(m.lib.depth)}층" + (f" | 은퇴: {', '.join(retired_now)}" if retired_now else ''))
            lines += [f"     {l.strip()}" for l in ilogs if any(e in l for e in ('💡', '🚪', '⏱️'))]
        lines += school(pop, n_before, gen)
        rec = dict(gen=gen, secs=time.time() - tg)
        if gen % CFG['PBT_EVERY'] == 0:
            S = selection_set(gen, seen_sigs)
            res = compete(pop, S)
            order = sorted(range(len(pop)), key=lambda i: (-res[i][0], -res[i][1]))
            best, worst = pop[order[0]], pop[order[-1]]
            rec.update(select=[r[0] for r in res], reach=[r[1] for r in res], best=best.k)
            for m in pop:
                det = S_.detect_classics(m.lib)
                if det: lines.append(f"   🎓 [고전 개념 탐지] 김코어#{m.k}: " + ', '.join(f"{a}≡{b}" for a, b in det))
            lines.append(f"   🏁 시합 (처음 보는 깊은 문제 {len(S)}개): " + ' | '.join(f"#{pop[i].k} {res[i][0]}개/도달 {res[i][1]:.1f}" for i in range(len(pop))))
            if res[order[0]][0] > res[order[-1]][0]:
                old = dict(worst.hp)
                worst.lib = copy.deepcopy(best.lib); worst.net = copy.deepcopy(best.net); worst.corpus = dict(best.corpus)
                worst.rec = copy.deepcopy(best.rec); worst.credit = copy.deepcopy(best.credit); worst.born = dict(best.born)
                worst.retired = set(best.retired); worst.homework = list(best.homework); worst.p = dict(best.p); worst.lp = dict(best.lp); worst.hp = dict(best.hp)
                worst.mutate(rng)
                lines.append(f"   🧬 [진화] 김코어#{worst.k}(꼴찌) ← 김코어#{best.k}(1등) 복제 + 설정 돌연변이 {old} → {worst.hp}")
                metrics['pbt'].append(dict(gen=gen, best=best.k, worst=worst.k, hp=worst.hp))
        if gen % CFG['PROBE_EVERY'] == 0 or gen == CFG['GENS']:
            bm = pop[rec.get('best', 0)] if 'best' in rec else pop[0]
            bm.apply_hp()
            pg, byg, ovg, _, _ = B.probe_eval(bm.lib, bm.net, probes, levels)
            rec.update(probe=pg, probe_fail_solved=byg['0세대 실패'][0])
            lines.append(f"   🔒 고정 탐침 (1등 김코어): 0세대 {p0:.1f}% → {pg:.1f}% | 0세대 실패 {byg['0세대 실패'][0]}/{nfail} | 과적합 {ovg}")
        log(f"[제 {gen:02d} 세대] {rec['secs']:.0f}s"); [log(l) for l in lines]
        metrics['gens'].append(rec)
        with open(os.path.join(OUT, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)
    # ---- 최종
    S = selection_set(999, seen_sigs); res = compete(pop, S)
    bi = max(range(len(pop)), key=lambda i: res[i]); bm = pop[bi]
    s0 = compete([Member(98, np.random.default_rng(1))], S)[0]
    log("\n" + "=" * 104)
    log(f" 🥊 [과거의 나 vs 지금의 나] 처음 보는 깊은 문제 {len(S)}개: 0세대 {s0[0]}개 (도달 {s0[1]:.1f})  vs  최종 1등 김코어#{bm.k} {res[bi][0]}개 (도달 {res[bi][1]:.1f})")
    log(f" 🧠 [김코어#{bm.k}의 활성 개념]")
    for i in range(bm.lib.n_base, len(bm.lib)):
        if i in bm.retired: continue
        log(f"   {bm.lib.names[i]} ({bm.lib.depth[i]}층, 기본재료 {bm.lib.base_leaves[i]}) : {tshow(bm.lib.types[i])} = {show(bm.lib.bodies[i], bm.lib)}")
    log(f"   은퇴한 개념 {len(bm.retired)}개 | 최종 설정 {bm.hp}")
    for m in pop:
        det = S_.detect_classics(m.lib)
        log(f" 🎓 [고전 개념 탐지] 김코어#{m.k}: " + (', '.join(f"{a}≡{b}" for a, b in det) if det else '없음'))
    log(" 📈 [도달 거리 곡선] (로그 눈금 기울기가 커지면 가속)")
    pts = [(0, base_score[1])] + [(g['gen'], max(g['reach'])) for g in metrics['gens'] if 'reach' in g] + [(CFG['GENS'] + 1, res[bi][1])]
    for g, r in pts: log(f"   gen {g:02d} | 도달 {r:5.1f} | log {math.log(max(r, 1)):.2f} " + '█' * int(r))
    metrics['final'] = dict(past=s0, now=res[bi], best=bm.k, hp=bm.hp)
    with open(os.path.join(OUT, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)
    log(f" 총 {time.time() - _T0:.0f}s")

if __name__ == '__main__':
    main()
