import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('V9BASE_OUT', os.path.join(HERE, '_evo_tmp')); os.environ.setdefault('V10_OUT', os.path.join(HERE, '_evo_tmp10'))
import numpy as np, torch
import v9base as B, v11search as S_
from v9base import Library, Recognizer, compile_term, show
B.search = S_.v10.search
ck = torch.load(sys.argv[1], weights_only=False); c = ck[int(sys.argv[2])]
probes = B.probe_set(9999)
lib = Library()
for n, t, b, d, bl in zip(c['names'], c['types'], c['bodies'], c['depth'], c['base_leaves']):
    lib.add(n, t, compile_term(b, lib)(()), b, d, bl)
net = Recognizer(); net.load_state_dict(c['net']); net.eval()
B._G['retired'] = set(c['retired'])
def run(L, N, sc, tag):
    B._G['rec_scale'] = sc
    rs = B.solve_all(L, N, probes, B.CFG['SEARCH_BUDGET'])
    ok = sum(1 for r in rs if r['prog'] is not None and r['hidden_ok']); ov = [i for i, r in enumerate(rs) if r['prog'] is not None and not r['hidden_ok']]
    print(f"{tag}: 탐침 {ok}/300 과적합 {len(ov)}", flush=True); return rs, ov
run(Library(), None, 1.0, '기본 도서관 + 인식망 없음 (0세대)')
run(Library(), net if len(lib) == len(Library()) else None, 1.0, '(참고)')
rs, ov = run(lib, None, 1.0, '학습한 개념 + 인식망 없음')
rs2, ov2 = run(lib, net, c['hp'].get('rec_scale', 1.0), '학습한 개념 + 학습한 인식망')
for i in ov2[:6]:
    t = probes[i]
    print('  예:', show(t['src'], Library()), ' => 찾은 것:', show(rs2[i]['prog'], lib))
