"""저장된 개체군(pop.pt)으로 고정 탐침을 두 탐색기로 다시 풀어 과적합 비교: v10 첫 해 vs v20 기본 재료 오컴"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('V9BASE_OUT', os.path.join(HERE, '_evo_tmp')); os.environ.setdefault('V10_OUT', os.path.join(HERE, '_evo_tmp10'))
import numpy as np, torch
import v9base as B
import v11search as S_
import v20search as S20
from v9base import Library, Recognizer, compile_term
ck = torch.load(sys.argv[1], weights_only=False)
probes = B.probe_set(9999)
sel_seed = int(os.environ.get('SEL_SEED', 0))
def rebuild(c):
    lib = Library()
    for n, t, b, d, bl in zip(c['names'], c['types'], c['bodies'], c['depth'], c['base_leaves']):
        lib.add(n, t, compile_term(b, lib)(()), b, d, bl)
    net = Recognizer(); net.load_state_dict(c['net']); net.eval()
    return lib, net
for c in ck:
    lib, net = rebuild(c)
    B._G['rec_scale'] = c['hp'].get('rec_scale', 1.0); B._G['retired'] = set(c['retired'])
    out = []
    for name, fn in (('v10 첫 해', S_.v10.search), ('v20 기본재료 오컴', S20.search)):
        B.search = fn
        rs = B.solve_all(lib, net, probes, B.CFG['SEARCH_BUDGET'])
        ok = sum(1 for r in rs if r['prog'] is not None and r['hidden_ok'])
        ov = sum(1 for r in rs if r['prog'] is not None and not r['hidden_ok'])
        out.append(f"{name}: 탐침 {ok}/{len(probes)} 과적합 {ov}")
    print(f"김코어#{c['k']} (개념 {len(c['names'])}개): " + ' | '.join(out), flush=True)
