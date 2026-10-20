"""
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
SEED = int(os.environ.get('V9_SEED', 1234))
OUT = os.environ.get('V9_OUT', os.path.join(HERE, f'v9_out_s{SEED}'))
os.makedirs(OUT, exist_ok=True)
os.environ['V9BASE_OUT'] = os.path.join(OUT, '_base')
sys.path.insert(0, HERE)
import numpy as np, torch
import v9base as B
from v9base import Library, Recognizer, base_size, leaves, show, tshow, subterms, task_features, prod_counts

CFG = dict(GENS=int(os.environ.get('V9_GENS', 24)), POP=4, TASKS_PER_MEMBER=int(os.environ.get('V9_TASKS', 40)),
           PBT_EVERY=4, SELECT_N=int(os.environ.get('V9_SELECT', 60)), RETIRE_AFTER=4, PROBE_EVERY=8)
_LOG = open(os.path.join(OUT, 'run.txt'), 'w', encoding='utf-8'); _T0 = time.time()
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); _LOG.write(s + '\n'); _LOG.flush()

ARMS = [(d, cb) for d in (3, 4, 5, 6, 7) for cb in (0.0, 0.5, 1.5)]

class Member:
    def __init__(self, k, rng):
        self.k = k; self.lib = Library(); self.net = Recognizer(); self.corpus = {}
        self.rec = collections.deque(maxlen=8000)
        self.hp = dict(EPS=float(rng.choice([0.05, 0.08, 0.12, 0.16])), MIN_GAIN_FRAC=float(rng.choice([0.005, 0.01, 0.02])),
                       easy_keep=float(rng.choice([0.1, 0.2, 0.35])), rec_iters=int(rng.choice([200, 300, 500])))
        self.p = {a: 0.5 for a in ARMS}; self.lp = {a: 0.3 for a in ARMS}
        self.credit = collections.defaultdict(list)   # 개념 -> 세대별 사용 횟수
        self.born = {}; self.retired = set(); self.score_hist = []

    def arm_weights(self):
        return [self.lp[a] + 0.05 for a in ARMS]

    def apply_hp(self):
        B.CFG['EPS'] = self.hp['EPS']; B.CFG['MIN_GAIN_FRAC'] = self.hp['MIN_GAIN_FRAC']
        B._G['easy_keep'] = self.hp['easy_keep']; B._G['retired'] = set(self.retired)

    def mutate(self, rng):
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
    tasks = []
    for (d, cb), n in arm_counts.items():
        ts = B.gen_frontier(m.lib, m.net, n, d, cb, gen * 97 + m.k * 13 + d, frozenset(seen_sigs))
        for t in ts: t['arm'] = (d, cb)
        tasks += ts
    B._G['retired'] = set(m.retired)
    rs = B.solve_all(m.lib, m.net, tasks, B.CFG['SEARCH_BUDGET'])
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
    ilogs = B.invent(m.lib, m.corpus, random.Random(gen * 7 + m.k), gen)
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
    drng = np.random.default_rng(gen * 1000 + m.k)
    while len(dream) < 400:
        tk = B.make_task(m.lib, Wd, drng, maxdepth=4)
        if tk is not None: dream.append((task_features(tk), prod_counts(tk['src'], len(m.lib))))
    B.train_recognizer(m.net, list(m.rec) + dream, len(m.lib), iters=m.hp['rec_iters'])
    return tasks, solved, sizes, ilogs, retired_now

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
    log(" [김코어 v9.0] 개선하는 기계를 개선하는 기계: 개체군 4마리 + 개념 신용 장부 + 학습 진척도 출제")
    for m in pop: log(f"  김코어#{m.k} 설정: {m.hp}")
    log(f"  고정 탐침 {len(probes)}문제 (0세대 실패 {nfail}개) | 0세대 탐침 {p0:.1f}%")
    log("=" * 104)
    S0 = selection_set(0, seen_sigs)
    base_score = compete([Member(99, np.random.default_rng(0))], S0)[0]
    log(f"📊 [0세대 시합 기준] 처음 보는 깊은 문제 {len(S0)}개: 0세대 {base_score[0]}개 해결, 도달 거리 {base_score[1]:.1f}")
    metrics = dict(cfg=CFG, probe0=p0, nfail=nfail, base_select=base_score, gens=[], pbt=[])
    for gen in range(1, CFG['GENS'] + 1):
        tg = time.time(); lines = []
        for m in pop:
            tasks, solved, sizes, ilogs, retired_now = member_gen(m, gen, seen_sigs, rng)
            seen_sigs.update(t['sig'] for t in tasks if 'sig' in t)
            active = len(m.lib) - m.lib.n_base - len(m.retired)
            lines.append(f"   김코어#{m.k}: {solved}/{len(tasks)} | 푼 크기 평균 {np.mean(sizes) if sizes else 0:.1f} 최대 {max(sizes) if sizes else 0} | "
                         f"활성 개념 {active} (은퇴 {len(m.retired)}) | 최고 {max(m.lib.depth)}층" + (f" | 은퇴: {', '.join(retired_now)}" if retired_now else ''))
            lines += [f"     {l.strip()}" for l in ilogs if '💡' in l]
        rec = dict(gen=gen, secs=time.time() - tg)
        if gen % CFG['PBT_EVERY'] == 0:
            S = selection_set(gen, seen_sigs)
            res = compete(pop, S)
            order = sorted(range(len(pop)), key=lambda i: (-res[i][0], -res[i][1]))
            best, worst = pop[order[0]], pop[order[-1]]
            rec.update(select=[r[0] for r in res], reach=[r[1] for r in res], best=best.k)
            lines.append(f"   🏁 시합 (처음 보는 깊은 문제 {len(S)}개): " + ' | '.join(f"#{pop[i].k} {res[i][0]}개/도달 {res[i][1]:.1f}" for i in range(len(pop))))
            if res[order[0]][0] > res[order[-1]][0]:
                old = dict(worst.hp)
                worst.lib = copy.deepcopy(best.lib); worst.net = copy.deepcopy(best.net); worst.corpus = dict(best.corpus)
                worst.rec = copy.deepcopy(best.rec); worst.credit = copy.deepcopy(best.credit); worst.born = dict(best.born)
                worst.retired = set(best.retired); worst.p = dict(best.p); worst.lp = dict(best.lp); worst.hp = dict(best.hp)
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
    log(" 📈 [도달 거리 곡선] (로그 눈금 기울기가 커지면 가속)")
    pts = [(0, base_score[1])] + [(g['gen'], max(g['reach'])) for g in metrics['gens'] if 'reach' in g] + [(CFG['GENS'] + 1, res[bi][1])]
    for g, r in pts: log(f"   gen {g:02d} | 도달 {r:5.1f} | log {math.log(max(r, 1)):.2f} " + '█' * int(r))
    metrics['final'] = dict(past=s0, now=res[bi], best=bm.k, hp=bm.hp)
    with open(os.path.join(OUT, 'metrics.json'), 'w') as f: json.dump(metrics, f, ensure_ascii=False, indent=1, default=str)
    log(f" 총 {time.time() - _T0:.0f}s")

if __name__ == '__main__':
    main()
