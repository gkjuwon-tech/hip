#!/usr/bin/env python3
"""
LEAN-RSI: a self-improving theorem-prover seed on real Lean 4 + Mathlib.

The loop (every generation):
  1. PROPOSE   a curriculum governor picks tiers; families emit conjectures
               (plus a few known-false DECOYS that must never be proven).
  2. SEARCH    best-first tactic search through the Lean REPL, guided by a
               learned policy/value net on top of a hand-written prior.
  3. VERIFY    every claimed proof is re-elaborated from scratch as a full
               `theorem` in command mode (kernel check, heartbeat cap), must
               contain no sorry/admit, and `#print axioms` may show only
               propext / Classical.choice / Quot.sound.
  4. LEARN     expert iteration: the policy is trained on verified proof paths,
               the value net on solved-vs-failed states.
  5. GROW      non-trivial theorems become library lemmas (usable by later
               searches); recurring tactic sequences become parameterised macros.
  6. MEASURE   a fixed held-out set per tier, trained system vs. the frozen
               initial system (control), at a tight budget. Progress is only
               what the held-out curve says, never the in-loop counters.

Nothing here is trusted except Lean's kernel.
"""
import os, sys, json, time, math, random, re, heapq, zlib, threading, queue, subprocess, shutil, collections, traceback, copy
from fractions import Fraction

# ----------------------------------------------------------------------------- config
def _env(k, d, t=float):
    v = os.environ.get(k)
    return t(v) if v is not None else d

CFG = dict(
    MINUTES=_env('RSI_MINUTES', 170.0),        # total wall-clock for the loop
    WORKERS=_env('RSI_WORKERS', 4, int),       # parallel Lean REPL processes
    PROBS_PER_GEN=_env('RSI_PROBS', 48, int),
    TRAIN_BUDGET=_env('RSI_TRAIN_BUDGET', 64, int),   # tactic calls per training problem
    EVAL_BUDGET=_env('RSI_EVAL_BUDGET', 24, int),     # tactic calls per held-out problem (tight on purpose)
    PROBLEM_WALL_S=_env('RSI_PROBLEM_WALL', 60.0),
    TACTIC_TIMEOUT_S=_env('RSI_TACTIC_TIMEOUT', 10.0),
    CHECK_TIMEOUT_S=60.0,
    FINAL_HEARTBEATS=400000,
    EVAL_EVERY=_env('RSI_EVAL_EVERY', 4, int),
    HELDOUT_PER_TIER=_env('RSI_HELDOUT', 6, int),
    DECOY_RATE=0.06,
    RESTART_EVERY=300,                          # problems per REPL before a fresh process
    MAX_LIB_CANDS=3,
    MAX_MACROS=40,
    SEED=_env('RSI_SEED', 1234, int),
    MAX_GENS=_env('RSI_MAX_GENS', 10 ** 6, int),
)
ALLOWED_AXIOMS = {'propext', 'Classical.choice', 'Quot.sound'}
BANNED = re.compile(r'\b(sorry|admit|native_decide|decide\s*:=\s*true|implemented_by|extern|unsafe|axiom|ofReduceBool)\b')

LEAN_HOME = os.environ.get('LEAN_HOME', os.path.expanduser('~/lean'))
LEAN_VER = os.environ.get('LEAN_VER', 'v4.35.0')
OUT_DIR = os.environ.get('RSI_OUT', '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
os.makedirs(OUT_DIR, exist_ok=True)

_t0 = time.time()
_log_lock = threading.Lock()
_logf = open(os.path.join(OUT_DIR, 'run.txt'), 'a', encoding='utf-8')
def log(*a):
    s = f"[{(time.time() - _t0) / 60:6.1f}m] " + ' '.join(str(x) for x in a)
    with _log_lock:
        print(s, flush=True)
        _logf.write(s + '\n'); _logf.flush()

# ----------------------------------------------------------------------------- setup (Kaggle / fresh machine)
def sh(cmd, cwd=None, env=None, timeout=3600):
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout

def lean_env():
    e = os.environ.copy()
    e['PATH'] = os.path.join(LEAN_HOME, f'lean-{LEAN_VER[1:]}-linux', 'bin') + ':' + e['PATH']
    return e

def setup_lean():
    """Install Lean toolchain (no elan needed), build the REPL, fetch Mathlib + its prebuilt cache."""
    bindir = os.path.join(LEAN_HOME, f'lean-{LEAN_VER[1:]}-linux', 'bin')
    proj = os.path.join(LEAN_HOME, 'proj')
    repl = os.path.join(LEAN_HOME, 'repl', '.lake', 'build', 'bin', 'repl')
    os.makedirs(LEAN_HOME, exist_ok=True)
    if not os.path.exists(os.path.join(bindir, 'lean')):
        log('setup: downloading Lean', LEAN_VER)
        url = f'https://releases.lean-lang.org/lean4/{LEAN_VER}/lean-{LEAN_VER[1:]}-linux.tar.zst'
        tz = os.path.join(LEAN_HOME, 'lean.tar.zst')
        sh(f'curl -sSfL {url} -o {tz}')
        try:
            import zstandard
        except ImportError:
            sh(f'{sys.executable} -m pip install -q zstandard')
            import zstandard
        import tarfile
        with open(tz, 'rb') as f:
            tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(f), mode='r|').extractall(LEAN_HOME)
        os.remove(tz)
    env = lean_env()
    log('setup:', sh('lean --version', env=env).strip())
    if not os.path.exists(repl):
        log('setup: building REPL')
        if not os.path.isdir(os.path.join(LEAN_HOME, 'repl')):
            sh(f'git clone -q --depth 1 -b {LEAN_VER} https://github.com/leanprover-community/repl', cwd=LEAN_HOME)
        sh('lake build', cwd=os.path.join(LEAN_HOME, 'repl'), env=env)
    if not os.path.isdir(os.path.join(proj, '.lake', 'packages', 'mathlib', '.lake', 'build')):
        log('setup: fetching Mathlib + cache')
        os.makedirs(proj, exist_ok=True)
        with open(os.path.join(proj, 'lakefile.toml'), 'w') as f:
            f.write(f'name = "seed"\n[[require]]\nname = "mathlib"\ngit = "https://github.com/leanprover-community/mathlib4"\nrev = "{LEAN_VER}"\n')
        with open(os.path.join(proj, 'lean-toolchain'), 'w') as f:
            f.write(f'leanprover/lean4:{LEAN_VER}\n')
        sh('lake update', cwd=proj, env=env, timeout=7200)
        sh('lake exe cache get', cwd=proj, env=env, timeout=7200)
    log('setup: done')
    return proj, repl

# ----------------------------------------------------------------------------- Lean REPL worker
class LeanError(Exception):
    pass

class LeanREPL:
    """One `repl` process with Mathlib imported. JSON in, JSON out, blank-line separated."""
    def __init__(self, wid, proj, repl_bin, library):
        self.wid, self.proj, self.repl_bin, self.library = wid, proj, repl_bin, library
        self.proc = None
        self.restarts = 0
        self.start()

    def start(self):
        if self.proc is not None:
            try:
                self.proc.kill(); self.proc.wait(5)
            except Exception:
                pass
        self.proc = subprocess.Popen(['lake', 'env', self.repl_bin], cwd=self.proj, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=lean_env())
        self.q = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, self.q), daemon=True).start()
        r = self._send({'cmd': 'import Mathlib'}, timeout=900)
        self.base_env = r['env']
        self.lib_env, self.lib_count, self.n_problems = self.base_env, 0, 0
        self.sync_library()

    @staticmethod
    def _reader(proc, q):
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    def _send(self, obj, timeout):
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + '\n\n')
        self.proc.stdin.flush()
        buf, deadline = [], time.time() + timeout
        while True:
            rem = deadline - time.time()
            if rem <= 0:
                raise TimeoutError
            try:
                line = self.q.get(timeout=rem)
            except queue.Empty:
                raise TimeoutError
            if line is None:
                raise LeanError('repl died')
            if line.strip() == '':
                if buf:
                    try:
                        return json.loads(''.join(buf))
                    except json.JSONDecodeError:
                        pass
                continue
            buf.append(line)

    def send(self, obj, timeout):
        """Send; on timeout or crash, restart the process and re-raise as LeanError."""
        try:
            return self._send(obj, timeout)
        except (TimeoutError, LeanError, BrokenPipeError, OSError) as e:
            self.restarts += 1
            self.start()
            raise LeanError(f'restart after {type(e).__name__}')

    def sync_library(self):
        decls = self.library.decls_since(self.lib_count)
        if not decls:
            return
        r = self._send({'cmd': '\n\n'.join(d for _, d in decls), 'env': self.lib_env}, timeout=600)
        errs = [m for m in r.get('messages', []) if m.get('severity') == 'error']
        if errs:
            log(f'[w{self.wid}] WARNING library sync error: {errs[0]["data"][:200]}')
        self.lib_env = r['env']
        self.lib_count += len(decls)

    def maybe_recycle(self):
        self.n_problems += 1
        if self.n_problems >= CFG['RESTART_EVERY']:
            self.start()

    # --- proof-state API
    def start_goal(self, stmt):
        r = self.send({'cmd': f'example : {stmt} := by sorry', 'env': self.lib_env}, timeout=60)
        if any(m.get('severity') == 'error' for m in r.get('messages', [])) or not r.get('sorries'):
            return None, None
        s = r['sorries'][0]
        return s['proofState'], s['goal']

    def tactic(self, ps, tac):
        """-> ('done'|'open'|'fail', proofState, goals)"""
        r = self.send({'tactic': tac, 'proofState': ps}, timeout=CFG['TACTIC_TIMEOUT_S'])
        if 'message' in r or any(m.get('severity') == 'error' for m in r.get('messages', [])):
            return 'fail', None, None
        st = r.get('proofStatus', '')
        if st == 'Completed' and not r.get('goals') and not r.get('sorries'):
            return 'done', r['proofState'], []
        if st.startswith('Incomplete: open goals'):
            return 'open', r['proofState'], r.get('goals', [])
        return 'fail', None, None      # e.g. "Incomplete: contains sorry", errors

    def check_theorem(self, name, stmt, tactics, env=None):
        """Full re-elaboration in command mode + axiom audit. -> (ok, reason, axioms)"""
        text = theorem_text(name, stmt, tactics, heartbeats=CFG['FINAL_HEARTBEATS']) + f'\n\n#print axioms {name}'
        try:
            r = self.send({'cmd': text, 'env': self.lib_env if env is None else env}, timeout=CFG['CHECK_TIMEOUT_S'])
        except LeanError as e:
            return False, str(e), None
        if r.get('sorries'):
            return False, 'sorry', None
        axioms = None
        for m in r.get('messages', []):
            if m.get('severity') == 'error':
                return False, 'error: ' + m['data'][:160], None
            d = m.get('data', '')
            if 'declaration uses' in d:
                return False, 'sorry-warning', None
            if 'depends on axioms' in d:
                axioms = set(a.strip() for a in d.split('[', 1)[1].rsplit(']', 1)[0].split(','))
            elif 'does not depend on any axioms' in d:
                axioms = set()
        if axioms is None:
            return False, 'no axiom report', None
        if not axioms <= ALLOWED_AXIOMS:
            return False, f'axioms {sorted(axioms - ALLOWED_AXIOMS)}', axioms
        return True, 'ok', axioms


def theorem_text(name, stmt, tactics, heartbeats=None):
    pre = f'set_option maxHeartbeats {heartbeats} in\n' if heartbeats else ''
    body = '\n'.join('  ' + t for t in tactics)
    return f'{pre}theorem {name} : {stmt} := by\n{body}'

# ----------------------------------------------------------------------------- goal parsing
NUMTYPES = ('ℕ', 'ℤ', 'ℝ', 'ℚ')
PROP_MARKS = ('=', '≤', '<', '≥', '>', '∣', '∧', '∨', '¬', '→', '↔', '∃', '∀', '≠', '≡', 'Even', 'Odd', 'Prime', '∈', '⊆', 'True', 'False', 'Injective', 'Surjective', 'StrictMono', 'Continuous')

def parse_goal(g):
    ctx, tgt, in_t = [], [], False
    for ln in g.split('\n'):
        if not in_t and ln.startswith('case '):
            continue
        if ln.startswith('⊢'):
            in_t = True; tgt.append(ln[1:].strip()); continue
        if in_t:
            tgt.append(ln.strip()); continue
        if ln.startswith(' ') and ctx:
            ctx[-1] = (ctx[-1][0], ctx[-1][1] + ' ' + ln.strip()); continue
        if ' : ' in ln:
            names, typ = ln.split(' : ', 1)
            ctx.append((names.split(), typ.strip()))
    return ctx, ' '.join(tgt)

class Goal:
    __slots__ = ('raw', 'ctx', 'target', 'vars', 'hyps', 'names', 'typevars', 'inacc')
    def __init__(self, raw):
        self.raw = raw
        self.ctx, self.target = parse_goal(raw)
        self.vars, self.hyps, self.names, self.typevars, self.inacc = [], [], set(), set(), 0
        for names, typ in self.ctx:
            isprop = any(m in typ for m in PROP_MARKS) and typ not in NUMTYPES
            for n in names:
                self.names.add(n)
                if '✝' in n:
                    self.inacc += 1
                    continue
                if typ == 'Type' or typ.startswith('Type '):
                    self.typevars.add(n)
                elif isprop:
                    self.hyps.append((n, typ))
                elif typ in NUMTYPES or typ in self.typevars or typ.startswith('Set '):
                    self.vars.append((n, typ))

    def fresh(self, base, k=1):
        out, i = [], 1
        while len(out) < k:
            n = f'{base}{i}'
            if n not in self.names and n not in out:
                out.append(n)
            i += 1
        return out

def split_top(s, sep):
    """Split s on sep occurring at bracket depth 0."""
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c in '([{⟨':
            depth += 1
        elif c in ')]}⟩':
            depth -= 1
        if depth == 0 and s.startswith(sep, i):
            out.append(''.join(cur)); cur = []; i += len(sep); continue
        cur.append(c); i += 1
    out.append(''.join(cur))
    return [x.strip() for x in out]

def forall_binders(t):
    """'∀ (a b : ℝ) [inst : Group G], body' -> (['a','b','_'], body)"""
    if not t.startswith('∀'):
        return None, t
    s, names, i = t[1:].strip(), [], 0
    while i < len(s):
        c = s[i]
        if c in '([{⦃':
            close = {'(': ')', '[': ']', '{': '}', '⦃': '⦄'}[c]
            depth, j = 0, i
            while j < len(s):
                if s[j] == c: depth += 1
                elif s[j] == close:
                    depth -= 1
                    if depth == 0: break
                j += 1
            inner = s[i + 1:j]
            if c == '[':
                names.append('_')
            elif ' : ' in inner:
                names += inner.split(' : ')[0].split()
            i = j + 1
        elif c == ',':
            return names, s[i + 1:].strip()
        elif c == ' ':
            i += 1
        else:  # unbracketed "x y : T," or "x > 0,"
            j = s.find(',', i)
            if j < 0: return None, t
            seg = s[i:j]
            m = re.match(r'^([^\s:<>≤≥]+)\s*([<>≤≥≠])', seg)
            if ' : ' in seg:
                names += seg.split(' : ')[0].split()
            elif m:
                names += [m.group(1), '__h']   # binder predicate: introduces a hypothesis too
            else:
                names += seg.split()
            return names, s[j + 1:].strip()
    return None, t

NUM_RE = re.compile(r'(?<![\w.])\d+(?![\w.])')

# ----------------------------------------------------------------------------- CAS helper (factor hypotheses)
_sympy = None
def sympy():
    global _sympy
    if _sympy is None:
        import sympy as sp
        _sympy = sp
    return _sympy

def lean_to_sympy(e):
    e = e.replace('↑', '').replace('^', '**').replace('⁻¹', '**(-1)')
    if re.search(r'[^\w\s\+\-\*/\(\)\.]', e):
        return None
    try:
        return sympy().sympify(e)
    except Exception:
        return None

def sympy_to_lean(ex):
    s = str(ex).replace('**', '^')
    return s

def cas_factor_hyp(h, typ, var):
    if typ.count('=') != 1 or any(c in typ for c in '≤≥<>∣∧∨'):
        return None
    lhs, rhs = typ.split('=')
    a, b = lean_to_sympy(lhs), lean_to_sympy(rhs)
    if a is None or b is None:
        return None
    sp = sympy()
    x = sp.Symbol(var)
    poly = sp.expand(a - b)
    if poly.free_symbols != {x} or sp.degree(poly, x) < 2:
        return None
    f = sp.factor_list(poly)
    factors = [fa for fa, m in f[1] for _ in range(m)]
    if len(factors) < 2:
        return None
    prod = ' * '.join(f'({sympy_to_lean(fa)})' for fa in factors)
    return prod

# ----------------------------------------------------------------------------- candidate tactics
CLOSERS = ['norm_num', 'simp', 'omega', 'linarith', 'ring', 'decide', 'rfl', 'tauto', 'positivity',
           'nlinarith', 'group', 'abel', 'fun_prop', 'simp_all', 'field_simp', 'ring_nf', 'trivial', 'continuity']
SUM_TACS = ['rw [Finset.sum_range_succ]', 'simp [Finset.sum_range_succ]', 'simp only [Finset.sum_range_succ, Finset.sum_range_zero]']
SET_TACS = ['simp only [Set.mem_inter_iff, Set.mem_union, Set.mem_diff, Set.mem_compl_iff]', 'simp [Set.subset_def]']
MOD_SIMPS = ['simp [Nat.dvd_iff_mod_eq_zero, Nat.pow_mod, Nat.mul_mod, Nat.add_mod, *]',
             'simp [Int.dvd_iff_emod_eq_zero, pow_succ, Int.mul_emod, Int.sub_emod, Int.add_emod, *]']

def nl_hints(g):
    nv = [n for n, t in g.vars if t in ('ℝ', 'ℚ', 'ℤ')][:4]
    pos = [h for h, t in g.hyps if re.fullmatch(r'0 < \w+', t) or re.fullmatch(r'\w+ > 0', t)]
    nonneg = [h for h, t in g.hyps if re.fullmatch(r'0 ≤ \w+', t) or re.fullmatch(r'\w+ ≥ 0', t)]
    out = []
    if nv:
        diffs = [f'sq_nonneg ({a} - {b})' for i, a in enumerate(nv) for b in nv[i + 1:]]
        sums = [f'sq_nonneg ({a} + {b})' for i, a in enumerate(nv) for b in nv[i + 1:]]
        singles = [f'sq_nonneg {a}' for a in nv]
        prods = [f'mul_pos {a} {b}' for i, a in enumerate(pos) for b in pos[i + 1:]]
        prods += [f'mul_nonneg {a} {b}' for i, a in enumerate(nonneg) for b in nonneg[i + 1:]]
        H1 = (diffs + singles)[:8]
        out.append(H1)
        if sums or prods:
            out.append((diffs + prods + sums)[:9])
        if len(nv) >= 3:
            out.append([f'sq_nonneg ({" + ".join(nv[:3])})'] + diffs[:3] + prods[:3])
    return [f'nlinarith [{", ".join(h)}]' for h in out if h]

def gen_candidates(g, library, macros):
    """Ordered candidate tactics for the main goal. Order = hand-written prior (the 'novice mathematician')."""
    t, C = g.target, []
    add = C.append
    nums = sorted(set(int(x) for x in NUM_RE.findall(t) if int(x) < 1000))
    is_real = 'ℝ' in g.raw or 'ℚ' in g.raw
    # --- introductions first: they almost never hurt
    names, body = forall_binders(t)
    if names:
        hs = g.fresh('h', 4)
        nm = [hs.pop(0) if n == '__h' else n for n in names]
        nm = [n if (n == '_' or n not in g.names) else g.fresh(n)[0] for n in nm]
        arrows = len(split_top(body, '→')) - 1
        add('intro ' + ' '.join(nm + hs[:arrows]))
        add('intro ' + ' '.join(nm))
    elif len(split_top(t, '→')) > 1:
        k = len(split_top(t, '→')) - 1
        add('intro ' + ' '.join(g.fresh('h', k)))
        add('intro ' + g.fresh('h')[0])
    if t.startswith('¬'):
        add('intro ' + g.fresh('h')[0]); add('push_neg')
    # --- closers
    if '∑' not in t:
        add('norm_num'); add('linarith'); add('ring'); add('omega')
    add('simp')
    if is_real or any(ty in ('ℝ', 'ℤ') for _, ty in g.vars):
        C.extend(nl_hints(g))
    if g.typevars:
        add('group'); add('abel'); add('simp')
    if 'Continuous' in t or 'Differentiable' in t:
        add('fun_prop'); add('continuity')
    for c in CLOSERS:
        add(c)
    # --- structure of the target
    if len(split_top(t, '∧')) > 1 or len(split_top(t, '↔')) > 1:
        add('constructor')
    if len(split_top(t, '∨')) > 1:
        add('left'); add('right')
    if t.startswith('∃'):
        for k in (nums + [0, 1, 2])[:6]:
            add(f'use {k}')
    if any(s in t for s in ('∩', '∪', '\\', 'ᶜ', '⊆')) and ' = ' in t:
        add(f'ext {g.fresh("x")[0]}')
    if '⊆' in t:
        add(f'intro {g.fresh("x")[0]} {g.fresh("hx")[0]}')
    if any(s in t for s in ('∩', '∪', '\\', 'ᶜ', '∈')):
        C.extend(SET_TACS)
    if 'Injective' in t or 'StrictMono' in t or 'Monotone' in t:
        a, b = g.fresh('a')[0], g.fresh('b')[0]
        add(f'intro {a} {b} {g.fresh("h")[0]}')
    if 'Surjective' in t:
        add(f'intro {g.fresh("y")[0]}')
    if '∑' in t:
        C.extend(SUM_TACS)
    if '/' in t and is_real:
        add('field_simp')
        add('rw [ge_iff_le, div_add_div _ _ (by positivity) (by positivity), le_div_iff₀ (by positivity)]')
        add('rw [div_le_iff₀ (by positivity)]'); add('rw [le_div_iff₀ (by positivity)]')
    # --- variables
    for v, ty in g.vars:
        if ty == 'ℕ':
            add(f'induction {v}')
    if g.inacc:
        add('rename_i ' + ' '.join(g.fresh('k', 1) + g.fresh('ih', max(0, g.inacc - 1))))
    mods = sorted(set(int(m) for m in re.findall(r'(\d+) ∣', t) + re.findall(r'% (\d+)', t) if 2 <= int(m) <= 6))
    for v, ty in g.vars:
        if ty in ('ℕ', 'ℤ'):
            for m in mods[:2]:
                hm = g.fresh('hm')[0]
                cases = ' ∨ '.join(f'{v} % {m} = {r}' for r in range(m))
                add(f'rcases (by omega : {cases}) with ' + ' | '.join([hm] * m))
            if ty == 'ℕ':
                add(f'rcases Nat.even_or_odd {v} with ⟨k, rfl⟩ | ⟨k, rfl⟩')
    if mods:
        C.extend(MOD_SIMPS)
    for v, ty in g.vars:
        for m in re.findall(r'\|([^|]+)\|', t + ' ' + ' '.join(h for _, h in g.hyps)):
            if v in m:
                a, b = g.fresh('ha')[0], g.fresh('hb')[0]
                add(f'rcases abs_cases ({m.strip()}) with ⟨{a}, {b}⟩ | ⟨{a}, {b}⟩')
    # --- hypotheses
    for h, ty in g.hyps:
        if ty.startswith('∃') or ty.startswith('Even') or ty.startswith('Odd'):
            add(f'obtain ⟨{g.fresh("w")[0]}, {g.fresh("hw")[0]}⟩ := {h}')
        elif len(split_top(ty, '∧')) > 1:
            a, b = g.fresh('h', 2) if True else ('', '')
            add(f'obtain ⟨{a}, {b}⟩ := {h}')
        elif len(split_top(ty, '∨')) > 1:
            a = g.fresh('h')[0]
            add(f'rcases {h} with {a} | {a}')
        elif re.fullmatch(r'.+ \* .+ = 0', ty):
            a = g.fresh('h')[0]
            add(f'rcases mul_eq_zero.mp {h} with {a} | {a}')
        nv = [v for v, vt in g.vars if vt in ('ℝ', 'ℚ', 'ℤ')]
        if len(nv) == 1 and '=' in ty and '^' in ty:
            try:
                f = cas_factor_hyp(h, ty, nv[0])
            except Exception:
                f = None
            if f:
                add(f'have {g.fresh("hf")[0]} : {f} = 0 := by nlinarith [{h}]')
        if any(op in ty for op in ('∣', '%', '≡')):
            add(f'simp at {h}')
    add('simp at *'); add('ring_nf'); add('ring_nf at *'); add('norm_num at *'); add('push_cast'); add('exfalso')
    # --- library lemmas (retrieval by token overlap)
    for name, lem in library.retrieve(g, CFG['MAX_LIB_CANDS']):
        add(f'exact {name}' if not lem['binders'] else f'apply {name}')
        add(f'rw [{name}]'); add(f'simp [{name}]')
        args = instantiate_args(lem, g)
        for a in args[:2]:
            add(f'nlinarith [{name} {a}]'); add(f'linarith [{name} {a}]')
    # --- learned macros
    for m in macros.instantiate(g):
        add(m)
    # dedupe preserving order, drop banned
    seen, out = set(), []
    for c in C:
        if c not in seen and not BANNED.search(c):
            seen.add(c); out.append(c)
    return out

def instantiate_args(lem, g):
    """Fill a lemma's typed binders with goal variables of matching type (two orderings)."""
    bt = [ty for _, ty in lem['binders']]
    if not bt or any(ty not in NUMTYPES for ty in bt):
        return []
    pool = {ty: [v for v, vt in g.vars if vt == ty] for ty in set(bt)}
    outs = []
    for rot in range(2):
        used, args = collections.Counter(), []
        for ty in bt:
            p = pool.get(ty) or []
            if not p:
                return outs
            args.append(p[(used[ty] + rot) % len(p)]); used[ty] += 1
        outs.append(' '.join(args))
    return list(dict.fromkeys(outs))

# ----------------------------------------------------------------------------- library of proven lemmas + macros
TOK_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9.']*|\d+|[^\sA-Za-z_0-9]")
def tokens(s):
    return TOK_RE.findall(s)

class Library:
    def __init__(self):
        self.lemmas = collections.OrderedDict()   # name -> dict(stmt, tactics, binders, tier, toks, uses)
        self._decls = []
        self.lock = threading.Lock()

    def add(self, stmt, tactics, tier, family):
        with self.lock:
            name = f'L{len(self.lemmas):04d}'
            binders = parse_stmt_binders(stmt)
            self.lemmas[name] = dict(stmt=stmt, tactics=tactics, binders=binders, tier=tier, family=family,
                                     toks=set(tokens(stmt)) - STOP, uses=0)
            self._decls.append((name, theorem_text(name, stmt, tactics, heartbeats=CFG['FINAL_HEARTBEATS'])))
            return name

    def decls_since(self, k):
        with self.lock:
            return list(self._decls[k:])

    def retrieve(self, g, k):
        if not self.lemmas:
            return []
        gt = (set(tokens(g.target)) | set(t for _, h in g.hyps for t in tokens(h))) - STOP
        scored = []
        with self.lock:
            items = list(self.lemmas.items())
        for name, lem in items:
            inter = len(gt & lem['toks'])
            if inter == 0:
                continue
            s = inter / (len(lem['toks']) ** 0.5 + 1) + 0.05 * math.log1p(lem['uses'])
            scored.append((s, name, lem))
        scored.sort(key=lambda x: -x[0])
        return [(n, l) for _, n, l in scored[:k]]

    def note_uses(self, tactics):
        used = set(re.findall(r'\bL\d{4}\b', ' '.join(tactics)))
        with self.lock:
            for u in used:
                if u in self.lemmas:
                    self.lemmas[u]['uses'] += 1
        return used

def parse_stmt_binders(stmt):
    m = re.match(r'^∀\s+([^,]+?)\s*:\s*(ℕ|ℤ|ℝ|ℚ)\s*,', stmt)
    if not m:
        return []
    return [(n, m.group(2)) for n in m.group(1).split()]

STOP = set('∀ , : ( ) ℝ ℕ ℤ ℚ * + - = ^ → 0 1 2 3 ⊢ Type x y a b c n'.split())
VAR_NAMES = set('a b c d x y z n m p q r s t u v w'.split())
class Macros:
    """Mine recurring tactic n-grams from verified proofs; abstract variable names into slots $0,$1..."""
    def __init__(self):
        self.items = {}   # template -> dict(count, uses, nslots)
        self.proofs = []

    @staticmethod
    def abstract(seq):
        mapping, out = {}, []
        for t in seq:
            def rep(m):
                w = m.group(0)
                if w in VAR_NAMES:
                    if w not in mapping:
                        mapping[w] = f'${len(mapping)}'
                    return mapping[w]
                return w
            out.append(re.sub(r"(?<![\w$.])[a-z](?![\w'])", rep, t))
        return out, len(mapping)

    def mine(self, proofs):
        cnt = collections.Counter(); slots = {}
        for tacs in proofs:
            seen = set()
            for n in (2, 3):
                for i in range(len(tacs) - n + 1):
                    tpl, ns = self.abstract(tacs[i:i + n])
                    key = '; '.join(tpl)
                    if key not in seen:
                        seen.add(key); cnt[key] += 1; slots[key] = ns
        best = [(c, k) for k, c in cnt.items() if c >= 3]
        best.sort(key=lambda x: (-x[0], len(x[1])))
        new = {}
        for c, k in best[:CFG['MAX_MACROS']]:
            old = self.items.get(k, {})
            new[k] = dict(count=c, uses=old.get('uses', 0), nslots=slots[k])
        added = len(set(new) - set(self.items))
        self.items = new
        return added

    def instantiate(self, g):
        vs = [v for v, _ in g.vars]
        out = []
        if not hasattr(self, 'inst'):
            self.inst = {}
        for k, m in self.items.items():
            if m['nslots'] > len(vs):
                continue
            s = k
            for i in range(m['nslots'] - 1, -1, -1):
                s = s.replace(f'${i}', vs[i])
            self.inst[f'({s})'] = k
            out.append(f'({s})')
        return out

    def note_uses(self, tactics):
        n = 0
        for t in tactics:
            k = getattr(self, 'inst', {}).get(t)
            if k is not None:
                n += 1
                if k in self.items:
                    self.items[k]['uses'] += 1
        return n

# ----------------------------------------------------------------------------- policy / value net
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(1)
HDIM = 1 << 16

def h32(s):
    return zlib.crc32(s.encode('utf-8')) % HDIM

def norm_tok(t):
    if t.isdigit():
        return '<n>' if len(t) > 1 else t
    return t

def goal_feats(g):
    toks = [norm_tok(t) for t in tokens(g.target)]
    f = ['T:' + t for t in toks] + ['B:' + a + '_' + b for a, b in zip(toks, toks[1:])]
    f += ['HEAD:' + (toks[0] if toks else '')]
    for _, ty in g.vars:
        f.append('VT:' + ty)
    for _, h in g.hyps:
        ht = [norm_tok(t) for t in tokens(h)]
        f += ['H:' + t for t in ht]
    f.append(f'NV:{min(len(g.vars), 5)}'); f.append(f'NH:{min(len(g.hyps), 5)}'); f.append(f'INACC:{min(g.inacc, 3)}')
    return [h32(x) for x in f] or [0]

def act_norm(a):
    return re.sub(r"(?<![\w.])(h\w*|hm\w*|hf\w*|ih\w*|[a-z])(?![\w'])", 'V', a)

def act_feats(a):
    toks = tokens(a)
    f = ['A:' + t for t in toks] + ['A0:' + (toks[0] if toks else ''), 'FULL:' + act_norm(a)]
    return [h32(x) for x in f]

class PolicyNet(nn.Module):
    def __init__(self, D=64):
        super().__init__()
        self.ge = nn.EmbeddingBag(HDIM, D, mode='mean')
        self.ae = nn.EmbeddingBag(HDIM, D, mode='mean')
        self.gm = nn.Sequential(nn.Linear(D, D), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * D, 64), nn.ReLU(), nn.Linear(64, 1))
        self.vhead = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 1))
        for m in (self.head[-1], self.vhead[-1]):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        nn.init.normal_(self.ge.weight, std=0.1); nn.init.normal_(self.ae.weight, std=0.1)

    @staticmethod
    def bag(lists):
        flat, offs = [], []
        for l in lists:
            offs.append(len(flat)); flat.extend(l)
        return torch.tensor(flat, dtype=torch.long), torch.tensor(offs, dtype=torch.long)

    def goal_vec(self, glists):
        return self.gm(self.ge(*self.bag(glists)))

    def scores(self, gvec, alists):
        a = self.ae(*self.bag(alists))
        gv = gvec.expand(a.shape[0], -1)
        return self.head(torch.cat([gv, a, gv * a], -1)).squeeze(-1)

    def value(self, gvec):
        return torch.sigmoid(self.vhead(gvec).squeeze(-1))

PRIOR_SLOPE = 0.25   # logit drop per rank in the hand-written order

class Policy:
    """prior(rank) + learned residual. `net=None` is the control arm (prior only, no learning)."""
    def __init__(self, net=None):
        self.net = net

    def rank(self, g, cands):
        prior = -PRIOR_SLOPE * np.arange(len(cands), dtype=np.float32)
        if self.net is None:
            return prior, 0.5
        with torch.no_grad():
            gv = self.net.goal_vec([goal_feats(g)])
            s = self.net.scores(gv, [act_feats(c) for c in cands]).numpy()
            v = float(self.net.value(gv)[0])
        return prior + s, v

# ----------------------------------------------------------------------------- search
class SearchResult:
    __slots__ = ('ok', 'tactics', 'calls', 'steps', 'neg_states', 'ill', 'restarts', 'secs')
    def __init__(self):
        self.ok, self.tactics, self.calls, self.steps, self.neg_states, self.ill, self.restarts, self.secs = False, [], 0, [], [], False, 0, 0.0

def canon(goals):
    return '\n||\n'.join(re.sub(r'\s+', ' ', x) for x in goals)

def best_first(worker, stmt, policy, library, macros, budget, explore=0.0, rng=None):
    """Lazy best-first over (node, candidate-rank) edges. One popped edge = one Lean tactic call."""
    res, t_start = SearchResult(), time.time()
    try:
        ps0, goal0 = worker.start_goal(stmt)
    except LeanError:
        res.restarts += 1; return res
    if ps0 is None:
        res.ill = True; return res
    heap, uid, seen, nodes = [], 0, {canon([goal0])}, []

    def make_node(ps, goals, path, logp, parent_edges):
        g = Goal(goals[0])
        cands = gen_candidates(g, library, macros)
        logits, v = policy.rank(g, cands)
        if explore > 0:
            logits = logits + explore * rng.gumbel(size=len(logits)).astype(np.float32)
        lp = logits - (np.log(np.exp(logits - logits.max()).sum()) + logits.max())
        order = np.argsort(-lp)
        node = dict(ps=ps, goals=goals, g=g, cands=cands, lp=lp, order=order, path=path, logp=logp,
                    v=v, steps=parent_edges, next=0)
        nodes.append(node)
        return node

    def push(node):
        nonlocal uid
        if node['next'] >= len(node['order']):
            return
        c = node['order'][node['next']]
        pri = node['logp'] + node['lp'][c] + 1.0 * math.log(max(node['v'], 1e-3)) - 0.15 * len(node['goals'])
        heapq.heappush(heap, (-pri, uid, node)); uid += 1

    push(make_node(ps0, [goal0], [], 0.0, []))
    while heap and res.calls < budget and time.time() - t_start < CFG['PROBLEM_WALL_S']:
        _, _, node = heapq.heappop(heap)
        ci = node['order'][node['next']]; node['next'] += 1
        push(node)
        tac = node['cands'][ci]
        res.calls += 1
        try:
            st, ps, goals = worker.tactic(node['ps'], tac)
        except LeanError:
            # process was restarted: every proof state is gone; stop this problem honestly
            res.restarts += 1; break
        if st == 'fail':
            continue
        step = (node['g'], node['cands'], ci)
        if st == 'done':
            res.ok, res.tactics = True, node['path'] + [tac]
            res.steps = node['steps'] + [step]
            break
        key = canon(goals)
        if key in seen:
            continue
        seen.add(key)
        child = make_node(ps, goals, node['path'] + [tac], node['logp'] + float(node['lp'][ci]), node['steps'] + [step])
        push(child)
    if not res.ok:
        res.neg_states = [n['g'] for n in nodes[:12]]
    res.secs = time.time() - t_start
    return res

# ----------------------------------------------------------------------------- conjecture families (the world)
def lean_int(k):
    return f'({k})' if k < 0 else str(k)

def poly_lean(expr, vars_):
    """sympy expanded polynomial -> Lean syntax with explicit '*' and '^'."""
    sp = sympy()
    P = sp.Poly(expr, *vars_)
    terms = []
    for monom, coef in sorted(P.terms(), key=lambda t: (-sum(t[0]), [-e for e in t[0]])):
        c = int(coef)
        mono = ' * '.join((str(v) if e == 1 else f'{v} ^ {e}') for v, e in zip(vars_, monom) if e)
        mag = abs(c)
        body = mono if (mag == 1 and mono) else (f'{mag} * {mono}' if mono else str(mag))
        terms.append((c < 0, body))
    if not terms:
        return '0'
    s = ('-' if terms[0][0] else '') + terms[0][1]
    for neg, b in terms[1:]:
        s += (' - ' if neg else ' + ') + b
    return s

class Families:
    """Each family: fn(rng, d, decoy) -> stmt string (true unless decoy; decoys are certified false)."""
    def __init__(self):
        self.by_tier = collections.defaultdict(list)
        for name in dir(self):
            if name.startswith('f_'):
                fn = getattr(self, name)
                self.by_tier[fn.tier].append((name[2:], fn))
        self.tiers = sorted(self.by_tier)

    def sample(self, tier, rng, d, decoy=False):
        for _ in range(50):
            name, fn = self.by_tier[tier][rng.randrange(len(self.by_tier[tier]))]
            try:
                s = fn(rng, d, decoy)
            except Exception:
                s = None
            if s:
                return name, s
        return None, None

    def _t(tier):
        def deco(f):
            f.tier = tier; return f
        return deco

    # ---- T0 arithmetic facts
    @_t(0)
    def f_arith(self, r, d, decoy):
        hi = int(10 + 990 * d)
        a, b, c = r.randint(2, hi), r.randint(2, hi), r.randint(0, hi)
        v = a * b + c + (r.choice([-1, 1]) if decoy else 0)
        return f'({a} : ℕ) * {b} + {c} = {v}' if v >= 0 else None
    @_t(0)
    def f_gcd(self, r, d, decoy):
        g = r.randint(2, int(6 + 30 * d)); a, b = g * r.randint(2, 40), g * r.randint(2, 40)
        gg = math.gcd(a, b) + (1 if decoy else 0)
        return f'Nat.gcd {a} {b} = {gg}'
    @_t(0)
    def f_prime(self, r, d, decoy):
        hi = int(50 + 2000 * d)
        def isp(n): return n > 1 and all(n % k for k in range(2, int(n ** 0.5) + 1))
        for _ in range(200):
            n = r.randint(11, hi)
            if isp(n) != decoy:
                return f'Nat.Prime {n}'
    @_t(0)
    def f_powmod(self, r, d, decoy):
        a, e, m = r.randint(2, 9), r.randint(2, int(3 + 8 * d)), r.randint(3, 13)
        v = pow(a, e, m) if not decoy else (pow(a, e, m) + 1) % m
        return f'({a} : ℕ) ^ {e} % {m} = {v}'

    # ---- T1 linear equations / inequalities
    @_t(1)
    def f_lin_real(self, r, d, decoy):
        a, x = r.choice([k for k in range(-int(3 + 9 * d), int(4 + 9 * d)) if k]), r.randint(-9, 9)
        b = r.randint(-20, 20); c = a * x + b
        xx = x + (1 if decoy else 0)
        return f'∀ x : ℝ, {a} * x + {b} = {c} → x = {xx}'.replace('+ -', '- ')
    @_t(1)
    def f_lin_nat(self, r, d, decoy):
        a, x, b = r.randint(2, int(3 + 20 * d)), r.randint(0, 30), r.randint(0, 50)
        return f'∀ n : ℕ, {a} * n + {b} = {a * x + b} → n = {x + (1 if decoy else 0)}'
    @_t(1)
    def f_sys(self, r, d, decoy):
        x, y = r.randint(-9, 9), r.randint(-9, 9)
        p, q = r.randint(1, int(2 + 4 * d)), r.randint(1, int(2 + 4 * d))
        return f'∀ x y : ℝ, x + {p} * y = {x + p * y} → x - {q} * y = {x - q * y} → x = {x + (1 if decoy else 0)}'
    @_t(1)
    def f_lin_ineq(self, r, d, decoy):
        a, b, c = r.randint(1, int(2 + 8 * d)), r.randint(-20, 20), r.randint(-20, 20)
        bound = Fraction(c - b, a)
        if decoy:
            bound -= 1
        bs = f'{bound.numerator} / {bound.denominator}' if bound.denominator != 1 else str(bound.numerator)
        return f'∀ x : ℝ, {a} * x + {b} < {c} → x < {bs}'.replace('+ -', '- ')

    # ---- T2 polynomial identities and the old quadratic prison
    @_t(2)
    def f_ring_id(self, r, d, decoy):
        sp = sympy()
        nv = 2 if d < 0.5 else 3
        vs = sp.symbols('a b c')[:nv]
        nf = r.randint(2, 2 + int(2 * d))
        def factor():
            co = [r.randint(-3, 3) for _ in vs]
            if not any(co): co[0] = 1
            return sum(c * v for c, v in zip(co, vs)) + r.randint(-3, 3)
        fs = [factor() for _ in range(nf)]
        lhs = ' * '.join(f'({poly_lean(sp.expand(f), vs)})' for f in fs)
        rhs = sp.expand(sp.Mul(*fs))
        if decoy:
            rhs = rhs + r.choice([-1, 1]) * r.choice(vs) * r.choice(vs)
        typ = r.choice(['ℝ', 'ℤ'])
        return f'∀ {" ".join(map(str, vs))} : {typ}, {lhs} = {poly_lean(rhs, vs)}'
    @_t(2)
    def f_quad(self, r, d, decoy):
        R = int(3 + 9 * d)
        p, q = r.randint(-R, R), r.randint(-R, R)
        sp = sympy(); x = sp.Symbol('x')
        e = poly_lean(sp.expand((x - p) * (x - q)), [x])
        if decoy:
            if p == q or q + 1 == p:
                return None
            return f'∀ x : ℝ, {e} = 0 → x = {lean_int(p)} ∨ x = {lean_int(q + 1)}'
        return f'∀ x : ℝ, {e} = 0 → x = {lean_int(p)} ∨ x = {lean_int(q)}'
    @_t(2)
    def f_square_expand(self, r, d, decoy):
        k = r.randint(1, int(2 + 8 * d)); m = r.randint(1, int(2 + 6 * d))
        c = 2 * k * m + (1 if decoy else 0)
        return f'∀ a b : ℝ, ({k} * a + {m} * b) ^ 2 = {k * k} * a ^ 2 + {c} * a * b + {m * m} * b ^ 2'

    # ---- T3 logic and sets
    @_t(3)
    def f_taut(self, r, d, decoy):
        import itertools as it
        atoms = ['p', 'q', 'r']
        def rnd(depth):
            if depth == 0 or r.random() < 0.3:
                a = r.choice(atoms); return (a, a) if r.random() < 0.8 else (f'¬{a}', ('not', a))
            op = r.choice(['∧', '∨', '→'])
            A, B = rnd(depth - 1), rnd(depth - 1)
            return (f'({A[0]} {op} {B[0]})', (op, A[1], B[1]))
        def ev(t, env):
            if isinstance(t, str): return env[t]
            if t[0] == 'not': return not ev(t[1], env)
            a, b = ev(t[1], env), ev(t[2], env)
            return {'∧': a and b, '∨': a or b, '→': (not a) or b}[t[0]]
        schemas = [('(({P} → {Q}) → (¬{Q} → ¬{P}))', True), ('({P} ∧ {Q} → {Q} ∧ {P})', True),
                   ('(({P} → {Q}) → ({Q} → {R}) → ({P} → {R}))', True), ('(¬({P} ∨ {Q}) ↔ (¬{P} ∧ ¬{Q}))', True),
                   ('(({P} ∧ ({Q} ∨ {R})) ↔ (({P} ∧ {Q}) ∨ ({P} ∧ {R})))', True), ('(({P} → {Q}) → ({Q} → {P}))', False),
                   ('(({P} ∨ {Q}) → {P})', False)]
        depth = 0 if d < 0.4 else 1 if d < 0.8 else 2
        for _ in range(100):
            si = r.randrange(len(schemas)); sc = schemas[si][0]
            subs = {k: rnd(depth) for k in 'PQR'}
            s = sc.format(**{k: v[0] for k, v in subs.items()})
            # truth table on the rendered formula via a tiny parser is overkill: evaluate the schema with sub-trees
            def evs(env):
                vals = {k: ev(v[1], env) for k, v in subs.items()}
                P, Q, R = vals['P'], vals['Q'], vals['R']
                imp = lambda a, b: (not a) or b
                return {0: imp(imp(P, Q), imp(not Q, not P)), 1: imp(P and Q, Q and P),
                        2: imp(imp(P, Q), imp(imp(Q, R), imp(P, R))), 3: (not (P or Q)) == ((not P) and (not Q)),
                        4: (P and (Q or R)) == ((P and Q) or (P and R)), 5: imp(imp(P, Q), imp(Q, P)),
                        6: imp(P or Q, P)}[si]
            taut = all(evs(dict(zip(atoms, bits))) for bits in it.product([False, True], repeat=3))
            if taut != decoy:
                return f'∀ p q r : Prop, {s}'
    @_t(3)
    def f_sets(self, r, d, decoy):
        import itertools as it
        S = ['A', 'B', 'C']
        def rnd(depth):
            if depth == 0 or r.random() < 0.25:
                return r.choice(S)
            op = r.choice(['∩', '∪', '\\'])
            return (op, rnd(depth - 1), rnd(depth - 1))
        def show(t):
            return t if isinstance(t, str) else f'({show(t[1])} {t[0]} {show(t[2])})'
        def ev(t, env):
            if isinstance(t, str): return env[t]
            a, b = ev(t[1], env), ev(t[2], env)
            return {'∩': a and b, '∪': a or b, '\\': a and not b}[t[0]]
        def rewrite(t):
            if isinstance(t, str): return t
            op, a, b = t
            rules = []
            if op in '∩∪': rules.append((op, b, a))
            if op == '∩' and not isinstance(b, str) and b[0] == '∪': rules.append(('∪', ('∩', a, b[1]), ('∩', a, b[2])))
            if op == '∪' and not isinstance(b, str) and b[0] == '∩': rules.append(('∩', ('∪', a, b[1]), ('∪', a, b[2])))
            if op == '\\' and not isinstance(b, str) and b[0] == '∪': rules.append(('∩', ('\\', a, b[1]), ('\\', a, b[2])))
            if op == '\\' and not isinstance(b, str) and b[0] == '∩': rules.append(('∪', ('\\', a, b[1]), ('\\', a, b[2])))
            rules.append((op, rewrite(a), rewrite(b)))
            return r.choice(rules)
        depth = 2 if d < 0.5 else 3
        for _ in range(200):
            e1 = rnd(depth)
            e2 = e1
            for _ in range(r.randint(1, 3)):
                e2 = rewrite(e2)
            if decoy:
                e2 = rnd(depth)
            if show(e1) == show(e2):
                continue
            eq = all(ev(e1, dict(zip(S, b))) == ev(e2, dict(zip(S, b))) for b in it.product([False, True], repeat=3))
            if eq != decoy:
                return f'∀ (α : Type) (A B C : Set α), {show(e1)} = {show(e2)}'
    @_t(3)
    def f_exists(self, r, d, decoy):
        a = r.randint(1, int(3 + 10 * d)); x = r.randint(-15, 15); b = r.randint(-20, 20)
        if decoy:
            return f'∃ x : ℤ, {2 * a} * x + {2 * r.randint(-10, 10) + 1} = 0'
        return f'∃ x : ℤ, {a} * x + {lean_int(b)} = {a * x + b}'

    # ---- T4 inequalities
    @_t(4)
    def f_ineq(self, r, d, decoy):
        k = r.randint(1, int(1 + 5 * d))
        true_t = ['∀ a b : ℝ, {k} * (a ^ 2 + b ^ 2) ≥ {k2} * a * b',
                  '∀ a b : ℝ, (a + b) ^ 2 ≤ 2 * (a ^ 2 + b ^ 2)',
                  '∀ a b c : ℝ, a ^ 2 + b ^ 2 + c ^ 2 ≥ a * b + b * c + c * a',
                  '∀ x : ℝ, x ^ 2 - {k2} * x + {kk} ≥ 0',
                  '∀ a b : ℝ, 0 ≤ a → 0 ≤ b → a ^ 3 + b ^ 3 ≥ a ^ 2 * b + a * b ^ 2',
                  '∀ a b c d : ℝ, (a * c + b * d) ^ 2 ≤ (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2)',
                  '∀ x y : ℝ, x ^ 2 + y ^ 2 = {kk} → x * y ≤ {kk} / 2',
                  '∀ a b : ℝ, 0 < a → 0 < b → a / b + b / a ≥ 2']
        false_t = ['∀ a b : ℝ, a ^ 2 + b ^ 2 ≤ {k2} * a * b', '∀ x : ℝ, x ^ 2 - {k2} * x + {kk} > 0',
                   '∀ a b : ℝ, a ^ 3 + b ^ 3 ≥ a ^ 2 * b + a * b ^ 2', '∀ a b : ℝ, (a + b) ^ 2 ≤ a ^ 2 + b ^ 2']
        hi = 1 + int(d * (len(true_t) - 1) + 0.999)
        t = r.choice(false_t if decoy else true_t[:hi])
        return t.format(k=k, k2=2 * k, kk=k * k)
    @_t(4)
    def f_shifted_square(self, r, d, decoy):
        p, q = r.randint(1, int(2 + 10 * d)), r.randint(0, int(1 + 10 * d))
        c = p * p + q - (q + 1 if decoy else 0)
        return f'∀ x : ℝ, x ^ 2 - {2 * p} * x + {c} ≥ {q}'

    # ---- T5 number theory
    @_t(5)
    def f_modlin(self, r, d, decoy):
        m = r.randint(2, int(3 + 10 * d)); a, b, res = r.randint(1, 20), r.randint(0, 30), None
        rr = r.randint(0, m - 1)
        v = (a * rr + b) % m
        if decoy:
            v = (v + 1) % m
        return f'∀ n : ℕ, n % {m} = {rr} → ({a} * n + {b}) % {m} = {v}'
    @_t(5)
    def f_divpoly(self, r, d, decoy):
        true_t = ['∀ n : ℕ, 2 ∣ n * (n + 1)', '∀ n : ℤ, 2 ∣ n ^ 2 + n', '∀ n : ℕ, 3 ∣ n * (n + 1) * (n + 2)',
                  '∀ n : ℤ, 3 ∣ n ^ 3 - n', '∀ n : ℕ, n ^ 2 % 4 = 0 ∨ n ^ 2 % 4 = 1', '∀ n : ℕ, Even (n ^ 2 + n)',
                  '∀ n : ℤ, 6 ∣ n ^ 3 - n', '∀ n : ℤ, 5 ∣ n ^ 5 - n']
        false_t = ['∀ n : ℕ, 3 ∣ n * (n + 1)', '∀ n : ℤ, 4 ∣ n ^ 4 - n', '∀ n : ℕ, Even (n ^ 2 + 1)', '∀ n : ℕ, n ^ 2 % 4 = 1']
        hi = 2 + int(d * (len(true_t) - 2) + 0.999)
        return r.choice(false_t if decoy else true_t[:hi])
    @_t(5)
    def f_parity_comb(self, r, d, decoy):
        a, b = r.randint(1, int(3 + 10 * d)), r.randint(0, 20)
        # (2a) n + 2b is even; decoy: odd constant
        c = 2 * b + (1 if decoy else 0)
        return f'∀ n : ℕ, Even ({2 * a} * n + {c})'

    # ---- T6 induction
    @_t(6)
    def f_sum_lin(self, r, d, decoy):
        a, b = r.randint(0, int(1 + 6 * d)), r.randint(0, int(1 + 6 * d))
        if a == 0 and b == 0: a = 1
        rhs = f'(n + 1) * ({a} * n + {2 * b + (1 if decoy else 0)})'
        return f'∀ n : ℕ, 2 * ∑ i ∈ Finset.range (n + 1), ({a} * i + {b}) = {rhs}'
    @_t(6)
    def f_sum_classic(self, r, d, decoy):
        true_t = ['∀ n : ℕ, ∑ i ∈ Finset.range n, (2 * i + 1) = n ^ 2',
                  '∀ n : ℕ, 6 * ∑ i ∈ Finset.range (n + 1), i ^ 2 = n * (n + 1) * (2 * n + 1)',
                  '∀ n : ℕ, 4 * ∑ i ∈ Finset.range (n + 1), i ^ 3 = (n * (n + 1)) ^ 2']
        false_t = ['∀ n : ℕ, ∑ i ∈ Finset.range n, (2 * i + 1) = n ^ 2 + 1', '∀ n : ℕ, 6 * ∑ i ∈ Finset.range (n + 1), i ^ 2 = n * (n + 1) * (n + 2)']
        hi = 1 + int(d * (len(true_t) - 1) + 0.999)
        return r.choice(false_t if decoy else true_t[:hi])
    @_t(6)
    def f_geom(self, r, d, decoy):
        q = r.randint(2, int(3 + 7 * d))
        return f'∀ n : ℕ, {q - 1} * ∑ i ∈ Finset.range n, {q} ^ i + {1 + (1 if decoy else 0)} = {q} ^ n'
    @_t(6)
    def f_pow_div(self, r, d, decoy):
        m = r.randint(2, int(3 + 9 * d))
        if decoy:
            return f'∀ n : ℕ, {m} ∣ {m + 1} ^ n + {m}'   # n=0: 1+m not divisible by m (m>=2)
        return f'∀ n : ℕ, {m} ∣ {m + 1} ^ n + {m - 1}'
    @_t(6)
    def f_bern(self, r, d, decoy):
        q = r.randint(2, int(3 + 5 * d))
        if decoy:
            return f'∀ n : ℕ, {q} ^ n ≥ {q} * n + 1'   # fails at n=1: q < q+1
        return f'∀ n : ℕ, {q} ^ n ≥ {q - 1} * n + 1'

    # ---- T7 abstract algebra
    @_t(7)
    def f_group_word(self, r, d, decoy):
        L = ['a', 'b', 'c']
        def red(w):
            out = []
            for x in w:
                if out and out[-1][0] == x[0] and out[-1][1] == -x[1]: out.pop()
                else: out.append(x)
            return out
        def show(w):
            return ' * '.join(x + ('⁻¹' if e < 0 else '') for x, e in w) if w else '1'
        n = r.randint(2, int(3 + 6 * d))
        w = [(r.choice(L), r.choice([1, -1])) for _ in range(n)]
        mode = r.random()
        if mode < 0.4:   # inverse of a product
            lhs = f'({show(w)})⁻¹'; rhs_w = red([(x, -e) for x, e in reversed(w)])
            tgt = rhs_w
        else:            # insert cancelling pairs
            tgt = red(w); lhs_w = list(tgt)
            for _ in range(r.randint(1, 2 + int(2 * d))):
                i = r.randint(0, len(lhs_w)); x = r.choice(L); e = r.choice([1, -1])
                lhs_w[i:i] = [(x, e), (x, -e)]
            lhs = show(lhs_w)
        if decoy:
            t = list(tgt)
            idx = [i for i in range(len(t) - 1) if t[i][0] != t[i + 1][0]]
            if not idx: return None
            i = r.choice(idx); t[i], t[i + 1] = t[i + 1], t[i]
            if red(t) == red(tgt): return None
            tgt = t
        return f'∀ (G : Type) [Group G] (a b c : G), {lhs} = {show(tgt)}'
    @_t(7)
    def f_comm_ring(self, r, d, decoy):
        sp = sympy(); a, b = sp.symbols('a b')
        k, m = r.randint(1, 4), r.randint(1, 4)
        e1 = (k * a + m * b) * (k * a - m * b); e2 = sp.expand(e1)
        if decoy: e2 = e2 + a * b
        return f'∀ (R : Type) [CommRing R] (a b : R), ({poly_lean(k * a + m * b, [a, b])}) * ({poly_lean(k * a - m * b, [a, b])}) = {poly_lean(e2, [a, b])}'
    @_t(7)
    def f_abel(self, r, d, decoy):
        cs = [r.randint(-4, 4) for _ in range(4)]; vs = ['a', 'b', 'a', 'b']
        terms = ' + '.join(f'({c} : ℤ) • {v}' for c, v in zip(cs, vs))
        ca, cb = cs[0] + cs[2] + (1 if decoy else 0), cs[1] + cs[3]
        return f'∀ (G : Type) [AddCommGroup G] (a b : G), {terms} = ({ca} : ℤ) • a + ({cb} : ℤ) • b'
    @_t(7)
    def f_field(self, r, d, decoy):
        k = r.randint(1, int(2 + 8 * d))
        rhs = f'x + {k + (1 if decoy else 0)}'
        return f'∀ x : ℝ, x ≠ 0 → (x ^ 2 + {k} * x) / x = {rhs}'

    # ---- T8 analysis
    @_t(8)
    def f_cont(self, r, d, decoy):
        sp = sympy(); x = sp.Symbol('x')
        p = poly_lean(sp.expand(sum(r.randint(-5, 5) * x ** i for i in range(1, 4)) + 1), [x])
        wraps = ['{p}', 'Real.exp ({p})', 'Real.sin ({p}) + {p}', 'Real.exp x * ({p})', 'Real.cos (Real.exp ({p}))']
        w = r.choice(wraps[:1 + int(d * 4 + 0.999)]).format(p=p)
        if decoy:
            return 'Continuous (fun x : ℝ => if x < 0 then (0 : ℝ) else 1)'
        kind = r.choice(['Continuous', 'Differentiable ℝ'])
        return f'{kind} (fun x : ℝ => {w})'
    @_t(8)
    def f_abs(self, r, d, decoy):
        k = r.randint(1, int(2 + 8 * d))
        true_t = ['∀ a b : ℝ, |a + b| ≤ |a| + |b|', '∀ a b : ℝ, |a| - |b| ≤ |a - b|',
                  f'∀ x : ℝ, |x - {k}| < 1 → {k - 1} < x', f'∀ x : ℝ, |x| ≤ {k} → x ^ 2 ≤ {k * k}']
        false_t = ['∀ a b : ℝ, |a + b| = |a| + |b|', f'∀ x : ℝ, |x - {k}| < 1 → {k} < x']
        return r.choice(false_t if decoy else true_t)
    @_t(8)
    def f_amgm_x(self, r, d, decoy):
        k = r.randint(1, int(2 + 6 * d))
        return f'∀ x : ℝ, 0 < x → x + {k * k} / x ≥ {2 * k + (1 if decoy else 0)}'
    @_t(8)
    def f_mono(self, r, d, decoy):
        a, b = r.randint(1, int(2 + 8 * d)), r.randint(-10, 10)
        a = -a if decoy else a
        return f'StrictMono (fun x : ℝ => {a} * x + {b})'.replace('+ -', '- ')
    @_t(8)
    def f_inj(self, r, d, decoy):
        a, b = r.randint(1, int(2 + 8 * d)), r.randint(-10, 10)
        if decoy:
            return 'Function.Injective (fun x : ℝ => x ^ 2)'
        return f'Function.Injective (fun x : ℝ => {r.choice([-1, 1]) * a} * x + {b})'.replace('+ -', '- ')

    # ---- T9 olympiad-lite
    @_t(9)
    def f_olymp(self, r, d, decoy):
        true_t = ['∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a ^ 3 + b ^ 3 + c ^ 3 ≥ 3 * a * b * c',
                  '∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c',
                  '∀ a b : ℝ, 0 < a → 0 < b → a + b = 1 → a * b ≤ 1 / 4',
                  '∀ a b c : ℝ, a + b + c = 0 → a ^ 3 + b ^ 3 + c ^ 3 = 3 * a * b * c',
                  '∀ x y : ℝ, x ^ 2 + y ^ 2 = 2 → x + y ≤ 2',
                  '∀ x : ℝ, x ^ 4 - 4 * x + 3 ≥ 0',
                  '∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a / b + b / c + c / a ≥ 3',
                  '∀ a b : ℝ, 0 < a → 0 < b → 2 / (1 / a + 1 / b) ≤ (a + b) / 2']
        false_t = ['∀ a b c : ℝ, a ^ 3 + b ^ 3 + c ^ 3 ≥ 3 * a * b * c', '∀ x y : ℝ, x ^ 2 + y ^ 2 = 2 → x + y ≤ 1']
        return r.choice(false_t if decoy else true_t)
    @_t(9)
    def f_pow_div2(self, r, d, decoy):
        b = r.randint(2, 9); m = b - 1
        while m < 2:
            b += 1; m = b - 1
        k = r.randint(1, 4)
        if decoy:
            return f'∀ n : ℕ, {m + 1} ∣ {b} ^ n - 1'
        return f'∀ n : ℕ, {m} ∣ {b} ^ (n * {k}) - 1' if r.random() < 0.5 else f'∀ n : ℕ, {m} ∣ {b} ^ n - 1'

TIER_NAMES = ['arith', 'linear', 'algebra', 'logic/sets', 'inequal', 'numtheory', 'induction', 'abs-algebra', 'analysis', 'olympiad']

# ----------------------------------------------------------------------------- curriculum governor
class Governor:
    def __init__(self, tiers):
        self.tiers = tiers
        self.unlocked = 1
        self.ema = {t: 0.0 for t in tiers}
        self.n = {t: 0 for t in tiers}
        self.d = {t: 0.0 for t in tiers}

    def pick(self, rng):
        F = self.unlocked - 1
        u = rng.random()
        if u < 0.12 and F + 1 < len(self.tiers):
            return F + 1                              # probe the locked tier above
        if u < 0.62 or F == 0:
            return F
        return rng.randrange(0, F)                    # replay below the frontier

    def update(self, tier, solved):
        a = 0.15
        self.n[tier] += 1
        self.ema[tier] = (1 - a) * self.ema[tier] + a * float(solved) if self.n[tier] > 1 else float(solved)
        if self.ema[tier] >= 0.7 and self.n[tier] >= 8:
            self.d[tier] = min(1.0, self.d[tier] + 0.05)
        elif self.ema[tier] < 0.3 and self.n[tier] >= 8:
            self.d[tier] = max(0.0, self.d[tier] - 0.05)
        F = self.unlocked - 1
        if tier >= F and self.ema[tier] >= 0.5 and self.n[tier] >= 12 and self.unlocked < len(self.tiers):
            self.unlocked = max(self.unlocked, tier + 2)
            log(f'*** TIER UNLOCKED: {self.unlocked - 1} ({TIER_NAMES[self.unlocked - 1]}) ***')

# ----------------------------------------------------------------------------- training
class Trainer:
    def __init__(self, net):
        self.net = net
        self.opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        self.pol = collections.deque(maxlen=12000)   # (gfeats, [afeats], chosen)
        self.val = collections.deque(maxlen=12000)   # (gfeats, label)

    def add_proof(self, steps):
        for g, cands, ci in steps:
            gf = goal_feats(g)
            self.pol.append((gf, [act_feats(c) for c in cands], ci))
            self.val.append((gf, 1.0))

    def add_failures(self, goals):
        for g in goals:
            self.val.append((goal_feats(g), 0.0))

    def train(self, epochs=3, bs=128):
        if len(self.pol) < 8:
            return 0.0, 0.0
        net = self.net; net.train()
        pol, val = list(self.pol), list(self.val)
        tl = tv = 0.0; nb = 0
        for _ in range(epochs):
            random.shuffle(pol); random.shuffle(val)
            for i in range(0, len(pol), bs):
                batch = pol[i:i + bs]
                gv = net.goal_vec([b[0] for b in batch])                     # (B, D)
                lens = [len(b[1]) for b in batch]; C = max(lens)
                flat = [a for b in batch for a in b[1]]
                own = torch.tensor([j for j, n in enumerate(lens) for _ in range(n)])
                pos = torch.tensor([k for n in lens for k in range(n)])
                a = net.ae(*net.bag(flat))
                g = gv[own]
                s = net.head(torch.cat([g, a, g * a], -1)).squeeze(-1) - PRIOR_SLOPE * pos.float()
                S = torch.full((len(batch), C), -1e9)
                S[own, pos] = s
                tgt = torch.tensor([b[2] for b in batch])
                loss = nn.functional.cross_entropy(S, tgt)
                vb = val[(i // bs * bs) % max(1, len(val)):][:bs]
                if vb:
                    vg = net.goal_vec([b[0] for b in vb])
                    vl = nn.functional.binary_cross_entropy(net.value(vg).clamp(1e-4, 1 - 1e-4), torch.tensor([b[1] for b in vb]))
                    loss = loss + 0.5 * vl
                    tv += float(vl)
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                tl += float(loss); nb += 1
        net.eval()
        return tl / max(nb, 1), tv / max(nb, 1)

def policy_entropy(net, steps):
    if not steps:
        return float('nan')
    ents = []
    with torch.no_grad():
        for g, cands, _ in steps[-200:]:
            gv = net.goal_vec([goal_feats(g)])
            s = -PRIOR_SLOPE * torch.arange(len(cands), dtype=torch.float32) + net.scores(gv, [act_feats(c) for c in cands])
            p = torch.softmax(s, 0)
            ents.append(float(-(p * torch.log(p + 1e-12)).sum() / math.log(len(cands))))
    return float(np.mean(ents))

# ----------------------------------------------------------------------------- the engine
class Engine:
    def __init__(self, proj, repl_bin):
        self.rng = random.Random(CFG['SEED'])
        self.nprng = np.random.default_rng(CFG['SEED'])
        torch.manual_seed(CFG['SEED'])
        self.lib = Library()
        self.macros = Macros()
        self.fam = Families()
        self.gov = Governor(self.fam.tiers)
        self.net = PolicyNet(); self.net.eval()
        self.trainer = Trainer(self.net)
        self.policy = Policy(self.net)
        self.control = Policy(None)
        self.empty_lib, self.empty_macros = Library(), Macros()
        log(f'starting {CFG["WORKERS"]} Lean REPL workers (import Mathlib)...')
        self.workers = [None] * CFG['WORKERS']
        def boot(i):
            self.workers[i] = LeanREPL(i, proj, repl_bin, self.lib)
        ths = [threading.Thread(target=boot, args=(i,)) for i in range(CFG['WORKERS'])]
        [t.start() for t in ths]; [t.join() for t in ths]
        log('workers ready')
        self.proven = {}            # stmt -> dict(tactics, tier, family, gen)
        self.metrics = dict(config=CFG, gens=[], evals=[], breaches=[], tier_names=TIER_NAMES)
        self.recent_steps = []
        self.totals = collections.Counter()
        self.best_heldout, self.best_state = -1.0, None
        self.heldout = self.build_heldout()

    def build_heldout(self):
        r = random.Random(777)
        H = []
        for t in self.fam.tiers:
            got, tries = set(), 0
            while len(got) < CFG['HELDOUT_PER_TIER'] and tries < 200:
                tries += 1
                fam, s = self.fam.sample(t, r, 0.5)
                if s and s not in got:
                    got.add(s); H.append((t, fam, s))
        log(f'held-out: {len(H)} problems over {len(self.fam.tiers)} tiers (never trained on)')
        self.heldout_set = set(s for _, _, s in H)
        return H

    # --- run many searches in parallel; each worker thread pulls jobs
    def run_jobs(self, jobs, fn):
        q, out = queue.Queue(), [None] * len(jobs)
        for i, j in enumerate(jobs):
            q.put((i, j))
        def loop(w):
            while True:
                try:
                    i, j = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    if w.lib_count < len(self.lib.lemmas):
                        w.sync_library()
                    out[i] = fn(w, j)
                    w.maybe_recycle()
                except Exception as e:
                    log(f'[w{w.wid}] job error {type(e).__name__}: {str(e)[:200]}')
                    try:
                        w.start()
                    except Exception:
                        pass
        ths = [threading.Thread(target=loop, args=(w,)) for w in self.workers]
        [t.start() for t in ths]; [t.join() for t in ths]
        return out

    def propose(self, n):
        jobs, tries = [], 0
        while len(jobs) < n and tries < n * 20:
            tries += 1
            decoy = self.rng.random() < CFG['DECOY_RATE']
            t = self.gov.pick(self.rng)
            d = self.gov.d[t]
            fam, s = self.fam.sample(t, self.rng, d, decoy=decoy)
            if not s or s in self.heldout_set or (not decoy and s in self.proven):
                continue
            if any(j[2] == s for j in jobs):
                continue
            jobs.append((t, fam, s, decoy))
        return jobs

    def solve_job(self, w, job):
        t, fam, s, decoy = job
        rng = np.random.default_rng(self.rng.randrange(1 << 30))
        res = best_first(w, s, self.policy, self.lib, self.macros, CFG['TRAIN_BUDGET'], explore=0.6, rng=rng)
        check = None
        if res.ok:
            name = f'P{zlib.crc32(s.encode()) % 10 ** 8:08d}'
            check = w.check_theorem(name, s, res.tactics)
        return res, check

    def evaluate(self, gen):
        def arm(policy, lib, mac):
            def fn(w, job):
                t, fam, s = job
                if lib is self.empty_lib:
                    env = w.base_env
                else:
                    env = None
                # control arm must not see the learned library: start goals in the base env
                saved = w.lib_env
                if env is not None:
                    w.lib_env = env
                try:
                    r = best_first(w, s, policy, lib, mac, CFG['EVAL_BUDGET'], explore=0.0)
                    ok = False
                    if r.ok:
                        ok = w.check_theorem('Heval', s, r.tactics)[0]
                finally:
                    w.lib_env = saved
                return ok, r.calls
            return fn
        t0 = time.time()
        tr = self.run_jobs(self.heldout, arm(self.policy, self.lib, self.macros))
        ct = self.run_jobs(self.heldout, arm(self.control, self.empty_lib, self.empty_macros))
        def summarize(rs):
            per = collections.defaultdict(list); calls = []
            for (t, _, _), r in zip(self.heldout, rs):
                ok = bool(r and r[0]); per[t].append(ok)
                if ok: calls.append(r[1])
            tot = [ok for t in per for ok in per[t]]
            return dict(acc=100.0 * sum(tot) / max(1, len(tot)),
                        per_tier={t: 100.0 * sum(v) / len(v) for t, v in per.items()},
                        calls_per_solve=float(np.mean(calls)) if calls else None)
        e = dict(gen=gen, minutes=(time.time() - _t0) / 60, trained=summarize(tr), control=summarize(ct))
        top = lambda d: max([t for t, a in d['per_tier'].items() if a >= 50] or [-1])
        e['trained']['top_tier50'], e['control']['top_tier50'] = top(e['trained']), top(e['control'])
        self.metrics['evals'].append(e)
        log(f'HELD-OUT gen {gen}: trained {e["trained"]["acc"]:.1f}% vs control {e["control"]["acc"]:.1f}%  '
            f'| top tier≥50%: {e["trained"]["top_tier50"]} vs {e["control"]["top_tier50"]}  '
            f'| calls/solve {e["trained"]["calls_per_solve"]} vs {e["control"]["calls_per_solve"]}  ({time.time() - t0:.0f}s)')
        log('   per tier trained: ' + ' '.join(f'{t}:{e["trained"]["per_tier"].get(t, 0):.0f}' for t in self.fam.tiers))
        log('   per tier control: ' + ' '.join(f'{t}:{e["control"]["per_tier"].get(t, 0):.0f}' for t in self.fam.tiers))
        acc = e['trained']['acc']
        if acc > self.best_heldout:
            self.best_heldout = acc
            self.best_state = copy.deepcopy(self.net.state_dict())
        elif acc < self.best_heldout - 15 and self.best_state is not None:
            log(f'!!! held-out collapsed {acc:.1f} < best {self.best_heldout:.1f} - 15: ROLLBACK policy')
            self.net.load_state_dict(self.best_state)
            e['rollback'] = True
        return e

    def run(self):
        deadline = _t0 + CFG['MINUTES'] * 60
        self.evaluate(0)
        gen = 0
        while time.time() < deadline and gen < CFG['MAX_GENS']:
            gen += 1
            tg = time.time()
            jobs = self.propose(CFG['PROBS_PER_GEN'])
            results = self.run_jobs(jobs, self.solve_job)
            g = dict(gen=gen, solved=0, attempted=0, decoys=0, rejected=0, ill=0, restarts=0, new_lemmas=0,
                     lib_reuse=0, macro_uses=0, calls_solved=[], per_tier=collections.defaultdict(lambda: [0, 0]))
            proofs_this_gen = []
            for job, out in zip(jobs, results):
                t, fam, s, decoy = job
                if out is None:
                    continue
                res, check = out
                g['restarts'] += res.restarts
                if res.ill:
                    g['ill'] += 1; continue
                if decoy:
                    g['decoys'] += 1
                    if res.ok and check and check[0]:
                        msg = f'BREACH: decoy proven! {s} :: {res.tactics}'
                        log('!!!!!!!!', msg); self.metrics['breaches'].append(msg)
                    continue
                g['attempted'] += 1
                g['per_tier'][t][1] += 1
                ok = res.ok and check is not None and check[0]
                if res.ok and not ok:
                    g['rejected'] += 1
                    log(f'  verifier REJECTED search claim ({check[1] if check else "?"}): {s[:80]}')
                self.gov.update(t, ok)
                if ok:
                    g['solved'] += 1; g['per_tier'][t][0] += 1; g['calls_solved'].append(res.calls)
                    self.trainer.add_proof(res.steps)
                    self.recent_steps.extend(res.steps)
                    used = self.lib.note_uses(res.tactics)
                    g['lib_reuse'] += bool(used)
                    g['macro_uses'] += self.macros.note_uses(res.tactics)
                    self.proven[s] = dict(tactics=res.tactics, tier=t, family=fam, gen=gen)
                    proofs_this_gen.append(res.tactics)
                    nontrivial = len(res.tactics) >= 2 or t >= 5
                    if nontrivial:
                        self.lib.add(s, res.tactics, t, fam); g['new_lemmas'] += 1
                else:
                    self.trainer.add_failures(res.neg_states)
            self.recent_steps = self.recent_steps[-400:]
            # learn
            pl, vl = self.trainer.train()
            new_macros = self.macros.mine([p['tactics'] for p in self.proven.values()])
            ent = policy_entropy(self.net, self.recent_steps)
            self.totals.update(dict(solved=g['solved'], attempted=g['attempted'], rejected=g['rejected'], decoys=g['decoys']))
            rec = dict(gen=gen, minutes=(time.time() - _t0) / 60, secs=time.time() - tg, frontier=self.gov.unlocked - 1,
                       solved=g['solved'], attempted=g['attempted'], decoys=g['decoys'], rejected=g['rejected'], ill=g['ill'],
                       restarts=g['restarts'], library=len(self.lib.lemmas), new_lemmas=g['new_lemmas'], macros=len(self.macros.items),
                       new_macros=new_macros, lib_reuse=g['lib_reuse'], macro_uses=g['macro_uses'],
                       calls_per_solve=float(np.mean(g['calls_solved'])) if g['calls_solved'] else None,
                       per_tier={k: v for k, v in g['per_tier'].items()}, tier_d={k: round(v, 2) for k, v in self.gov.d.items()},
                       tier_ema={k: round(v, 2) for k, v in self.gov.ema.items()}, policy_loss=pl, value_loss=vl, entropy=ent,
                       total_proven=len(self.proven), breaches=len(self.metrics['breaches']))
            self.metrics['gens'].append(rec)
            pt = ' '.join(f'T{k}:{v[0]}/{v[1]}' for k, v in sorted(g['per_tier'].items()))
            log(f'gen {gen:3d} | frontier T{rec["frontier"]} {TIER_NAMES[rec["frontier"]]:<11} | solved {g["solved"]}/{g["attempted"]} [{pt}] '
                f'| lib {rec["library"]} (+{g["new_lemmas"]}, reuse {g["lib_reuse"]}) | macros {rec["macros"]} | '
                f'calls/solve {rec["calls_per_solve"] and round(rec["calls_per_solve"], 1)} | rej {g["rejected"]} | decoys {g["decoys"]} '
                f'breach {rec["breaches"]} | ent {ent:.2f} | {rec["secs"]:.0f}s')
            if gen % 2 == 0 and ent == ent and ent < 0.08:
                log('!!! policy entropy collapse alarm (<0.08)')
            if gen % CFG['EVAL_EVERY'] == 0:
                self.evaluate(gen)
            self.save()
        if not self.metrics['evals'] or self.metrics['evals'][-1]['gen'] != gen:
            self.evaluate(gen)
        self.save(final=True)

    def save(self, final=False):
        with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=1, default=str)
        with open(os.path.join(OUT_DIR, 'library.lean'), 'w') as f:
            f.write('import Mathlib\n\n/-! Theorems proven by LEAN-RSI. Library lemmas L####, then all other proven theorems. -/\n\n')
            for name, d in self.lib.decls_since(0):
                f.write(d + '\n\n')
            k = 0
            libstmts = set(l['stmt'] for l in self.lib.lemmas.values())
            for s, p in self.proven.items():
                if s in libstmts:
                    continue
                f.write(f'-- tier {p["tier"]} ({TIER_NAMES[p["tier"]]}) family {p["family"]} gen {p["gen"]}\n')
                f.write(theorem_text(f'P{k:05d}', s, p['tactics'], heartbeats=CFG['FINAL_HEARTBEATS']) + '\n\n')
                k += 1
        with open(os.path.join(OUT_DIR, 'macros.json'), 'w') as f:
            json.dump(self.macros.items, f, ensure_ascii=False, indent=1)
        try:
            plot(self.metrics, os.path.join(OUT_DIR, 'progress.png'))
        except Exception as e:
            log('plot error', e)

# ----------------------------------------------------------------------------- plotting
def plot(m, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.titlecolor': INK, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=SURF)
    for a in ax.flat:
        a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.6); a.spines[['top', 'right']].set_visible(False)
    E, G = m['evals'], m['gens']
    if E:
        x = [e['gen'] for e in E]
        a = ax[0, 0]
        a.plot(x, [e['trained']['acc'] for e in E], color=BLUE, lw=2, marker='o', ms=4, label='trained (learning + library + macros)')
        a.plot(x, [e['control']['acc'] for e in E], color=ORANGE, lw=2, marker='s', ms=4, label='control (frozen initial system)')
        a.set_title('Held-out theorems proven (fixed set, tight budget)'); a.set_ylabel('% proven'); a.set_xlabel('generation')
        a.set_ylim(0, 100); a.legend(frameon=False, loc='lower right')
        a.annotate(f'{E[-1]["trained"]["acc"]:.0f}%', (x[-1], E[-1]['trained']['acc']), textcoords='offset points', xytext=(6, 0), color=INK)
        a = ax[0, 1]
        tiers = sorted(set(int(t) for e in E for t in e['trained']['per_tier']))
        last = E[-1]
        w = 0.38
        a.bar([t - w / 2 for t in tiers], [last['trained']['per_tier'].get(t, last['trained']['per_tier'].get(str(t), 0)) for t in tiers], w, color=BLUE, label='trained')
        a.bar([t + w / 2 for t in tiers], [last['control']['per_tier'].get(t, last['control']['per_tier'].get(str(t), 0)) for t in tiers], w, color=ORANGE, label='control')
        a.set_xticks(tiers); a.set_xticklabels([f'{t}\n{TIER_NAMES[t]}' for t in tiers], fontsize=7)
        a.set_ylim(0, 100); a.set_title(f'Held-out by tier, gen {last["gen"]} (basic → advanced)'); a.set_ylabel('% proven')
        a.legend(frameon=False)
    if G:
        x = [g['gen'] for g in G]
        a = ax[1, 0]
        a.step(x, [g['frontier'] for g in G], where='post', color=BLUE, lw=2, label='curriculum frontier tier')
        a.set_ylim(-0.3, 9.5); a.set_yticks(range(10)); a.set_yticklabels([f'{i} {n}' for i, n in enumerate(TIER_NAMES)], fontsize=7)
        a.set_title('Curriculum frontier (self-proposed)'); a.set_xlabel('generation')
        a2 = a.twinx() if False else None
        a = ax[1, 1]
        a.plot(x, [g['total_proven'] for g in G], color=BLUE, lw=2, label='theorems proven (kernel-checked)')
        a.plot(x, [g['library'] for g in G], color=AQUA, lw=2, label='library lemmas')
        a.set_title(f'Growth | verifier rejections {sum(g["rejected"] for g in G)} | decoy breaches {len(m["breaches"])}')
        a.set_xlabel('generation'); a.set_ylabel('count'); a.legend(frameon=False, loc='upper left')
    fig.suptitle('LEAN-RSI on Lean 4 + Mathlib: self-proposed conjectures, kernel-verified proofs', x=0.01, ha='left', color=INK, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURF); plt.close(fig)

# ----------------------------------------------------------------------------- final independent re-check
def final_recheck(proj):
    """Re-elaborate the whole library.lean with the `lean` binary, outside the REPL."""
    path = os.path.join(OUT_DIR, 'library.lean')
    dst = os.path.join(proj, 'RsiLibrary.lean')
    shutil.copy(path, dst)
    t = time.time()
    r = subprocess.run(['lake', 'env', 'lean', dst], cwd=proj, env=lean_env(), capture_output=True, text=True, timeout=3600)
    errs = [l for l in (r.stdout + r.stderr).splitlines() if 'error' in l]
    log(f'FINAL RE-CHECK of library.lean with standalone lean: exit {r.returncode}, {len(errs)} error lines, {time.time() - t:.0f}s')
    for l in errs[:10]:
        log('   ', l[:200])
    return r.returncode == 0

def main():
    log('LEAN-RSI config', json.dumps(CFG))
    proj, repl = setup_lean()
    eng = Engine(proj, repl)
    try:
        eng.run()
    except KeyboardInterrupt:
        log('interrupted'); eng.save(final=True)
    ok = final_recheck(proj)
    eng.metrics['final_recheck_ok'] = ok
    eng.save(final=True)
    G, E = eng.metrics['gens'], eng.metrics['evals']
    log('=' * 90)
    log(f'SUMMARY: {len(G)} gens | proven {len(eng.proven)} theorems | library {len(eng.lib.lemmas)} | macros {len(eng.macros.items)}')
    log(f'  held-out: gen {E[0]["gen"]} trained {E[0]["trained"]["acc"]:.1f}% / control {E[0]["control"]["acc"]:.1f}%  ->  '
        f'gen {E[-1]["gen"]} trained {E[-1]["trained"]["acc"]:.1f}% / control {E[-1]["control"]["acc"]:.1f}%')
    log(f'  frontier tier reached: {eng.gov.unlocked - 1} ({TIER_NAMES[eng.gov.unlocked - 1]})')
    log(f'  verifier rejections {eng.totals["rejected"]} | decoys tried {eng.totals["decoys"]} | decoy breaches {len(eng.metrics["breaches"])} | '
        f'REPL restarts {sum(w.restarts for w in eng.workers)} | final standalone re-check {"PASS" if ok else "FAIL"}')
    for w in eng.workers:
        try: w.proc.kill()
        except Exception: pass

if __name__ == '__main__':
    main()
