"""The fifteen shots of AUTOPILOT OFF.

Each shot owns a patch of world space (`origin`), builds its geometry under a
root empty in local coordinates, keys its own object animation, and exposes
`cam(t)` -> (pos, target, lens, roll) in local coordinates, where t is beats
since the shot started. FX hits (pump/shake/flash/aberration) are registered
in absolute beats on the shared FX table.
"""
import math
import random

import bpy
from mathutils import Vector, noise

from lib import (F, B, lerp, prog, out_expo, in_expo, in_out_cubic, out_cubic, in_cubic,
                 in_out_expo, out_back, out_elastic, pulse, new_collection,
                 set_active_collection, empty, mesh_obj, text, word_letters, to_mesh_obj,
                 key, bake, m_cube, m_sphere, m_uvsphere, m_cyl, m_torus, m_plane, m_shard,
                 mat_principled, mat_emit, mat_rim, mat_dissolve, key_node_value,
                 bake_node_value, HEAVY, WIDE, MONO, MONO_R)

FX = {"pump": [], "shake": [], "flash": [], "ca": [], "tint": [], "batt": [], "lyric": []}

WHITE = (1, 1, 1)
RED = (1.0, 0.03, 0.06)
GREEN = (0.25, 1.0, 0.05)
PURPLE = (0.55, 0.15, 1.0)


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


class Shot:
    name = "shot"
    length = 8
    tint = (0.004, 0.004, 0.006)      # world/fog colour
    fog = 0.004                       # world volume density

    def __init__(self, b0, origin):
        self.b0 = b0
        self.b1 = b0 + self.length
        self.origin = Vector(origin)
        self.col = new_collection(f"{b0:03.0f}_{self.name}")
        set_active_collection(self.col)
        self.root = empty(f"{self.name}.root", loc=self.origin)
        self.rng = random.Random(hash(self.name) & 0xFFFF)

    # --- helpers -------------------------------------------------------------
    def f(self, t):
        return F(self.b0 + t)

    def t_of(self, frame):
        return B(frame) - self.b0

    def bake(self, obj, path, t0, t1, fn):
        """Bake fn(t) every frame for local beats t0..t1."""
        bake(obj, path, self.f(t0), self.f(t1), lambda fr: fn(self.t_of(fr)))

    def bake_mat(self, mat, node, t0, t1, fn, input_idx=None):
        bake_node_value(mat, node, self.f(t0), self.f(t1), lambda fr: fn(self.t_of(fr)), input_idx)

    def hit(self, t, kind, amt=1.0, decay=0.35):
        FX[kind].append((self.b0 + t, amt, decay))

    def lyric(self, t, line):
        FX["lyric"].append((self.b0 + t, line))

    def batt(self, t, v):
        FX["batt"].append((self.b0 + t, v))

    def show(self, obj, t_on=None, t_off=None):
        """Scale-gate an object: invisible before t_on / after t_off."""
        if t_on is not None:
            key(obj, "hide_render", 1, True, "CONSTANT")
            key(obj, "hide_render", self.f(t_on), False, "CONSTANT")
        if t_off is not None:
            key(obj, "hide_render", self.f(t_off), True, "CONSTANT")

    def text(self, *a, **k):
        k.setdefault("parent", self.root)
        return text(*a, **k)

    def light(self, name, kind, loc, energy, color=WHITE, rot=(0, 0, 0), size=0.5,
              spot=math.radians(45), blend=0.3):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        ld.color = color
        if kind == "SPOT":
            ld.spot_size = spot
            ld.spot_blend = blend
        if kind in ("POINT", "SPOT"):
            ld.shadow_soft_size = size
        if kind == "AREA":
            ld.size = size
        ld.use_shadow = False
        o = bpy.data.objects.new(name, ld)
        o.location = loc
        o.rotation_euler = rot
        o.parent = self.root
        self.col.objects.link(o)
        return o

    def aim(self, obj, target):
        d = Vector(target) - Vector(obj.location)
        obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

    def road(self, y0=-120, y1=260, width=14, lines=True, mat=None, line_col=WHITE):
        g = mesh_obj(f"{self.name}.ground", m_plane("ground", 400, (y1 - y0) / 2),
                     mat or GROUND(), loc=(0, (y0 + y1) / 2, 0), parent=self.root)
        if lines:
            lm = mat_emit(f"{self.name}.lane", line_col, 3.0)
            dash = m_cube("dash", (0.18, 3.0, 0.02))
            y = y0
            while y < y1:
                mesh_obj("dash", dash, lm, loc=(0, y, 0.01), parent=self.root)
                y += 9
            side = m_cube("side", (0.12, (y1 - y0), 0.02))
            for x in (-width / 2, width / 2):
                mesh_obj("side", side, lm, loc=(x, (y0 + y1) / 2, 0.01), parent=self.root)
        return g

    def build(self):
        pass

    def cam(self, t):
        return V(0, -20, 3), V(0, 0, 2), 35.0, 0.0

    def finish(self):
        from lib import visible_between
        visible_between(self.col, F(self.b0) - 1, F(self.b1) + 1)


_GROUND = {}


def GROUND(tint=(0.006, 0.006, 0.008)):
    k = tuple(tint)
    if k not in _GROUND:
        m = mat_principled("ground_wet", tint, metallic=0.0, rough=0.12, coat=0.0)
        # wet patches: noise-driven roughness
        nt = m.node_tree
        p = nt.nodes["Principled BSDF"]
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 0.08
        nz.inputs["Detail"].default_value = 4
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 0.04
        mr.inputs["To Max"].default_value = 0.45
        nt.links.new(nz.outputs["Fac"], mr.inputs[0])
        nt.links.new(mr.outputs[0], p.inputs["Roughness"])
        _GROUND[k] = m
    return _GROUND[k]


# =============================================================================
# 1. TESLA — headlights on, pull-back reveal, dive through the E/S gap
# =============================================================================
class S01Tesla(Shot):
    name = "tesla"
    length = 8

    def build(self):
        self.lyric(0, "TESLA I JUST PULLED OFF IN THAT FUCKING TESLA")
        self.batt(0, 12)
        self.road()
        rim = mat_rim("tesla.rim", rim=WHITE, rim_strength=8)
        root, glyphs, w = word_letters("TESLA", "TESLA", size=9, extrude=1.3, bevel=0.06,
                                       mat=rim, tracking=0.1, parent=self.root)
        self.glyphs = glyphs
        xs = [p.location.x for _, p, _ in glyphs]
        self.gap = (xs[1] + xs[2]) / 2
        # second "Tesla": glyphs bump in sequence
        for i, (_, piv, _) in enumerate(glyphs):
            ti = 6.0 + i * 0.12
            self.bake(piv, "scale", 5.8, 7.6,
                      lambda t, ti=ti: (1, 1, 1 + 0.22 * pulse(t, ti, 0.08, 0.4)))
        rs = rim.node_tree.nodes["RimStrength"]
        self.bake_mat(rim, "RimStrength", 0, 8,
                      lambda t: 3 + 8 * prog(t, 0.4, 0.9) + 14 * pulse(t, 6, 0, 1.2), input_idx=1)
        # headlights
        for x in (-1.3, 1.3):
            sp = self.light("head", "SPOT", (x, -15, 0.6), 0, WHITE, spot=math.radians(28), blend=0.6)
            self.aim(sp, (x * 2, 0, 3.5))
            self.bake(sp.data, "energy", 0, 8, lambda t: 9000 * prog(t, 0.45, 0.55))
        # red neon kerb strips
        neon = mat_emit("tesla.neon", RED, 12)
        for x in (-22, 22):
            mesh_obj("neon", m_cube("neon", (0.25, 300, 0.25)), neon, loc=(x, 80, 0.12), parent=self.root)
        red = self.light("redwash", "POINT", (0, -6, 8), 0, RED, size=6)
        self.bake(red.data, "energy", 3.8, 6, lambda t: 60000 * pulse(t, 4.5, 0, 0.6))
        self.hit(0.5, "flash", 0.6, 0.6)
        self.hit(2, "shake", 1.4, 0.6)
        self.hit(2, "pump", 1.5)
        self.hit(4.5, "flash", 2.5, 0.25)
        self.hit(4.5, "tint", 1.0, 0.5)
        self.hit(4.5, "shake", 1.0, 0.4)
        self.hit(4.5, "ca", 1.0, 0.4)
        self.hit(6, "pump", 1.2)
        self.hit(7.4, "shake", 0.6, 0.3)

    def cam(self, t):
        g = self.gap
        if t < 2:
            p = lerp(V(1.6, -13.5, 0.35), V(1.4, -12.2, 0.4), t / 2)
            return p, V(0.4, 0, 4.6), 20, 0.12
        if t < 6:
            a = out_expo(prog(t, 2, 4.6))
            p = lerp(V(1.4, -12.2, 0.4), V(0, -56, 20), a)
            p = lerp(p, V(0, -50, 17), in_out_cubic(prog(t, 4.6, 6)))
            tg = lerp(V(0.4, 0, 4.6), V(0, 0, 6), a)
            return p, tg, lerp(20, 35, a), lerp(0.12, 0, a)
        a = in_out_cubic(prog(t, 6, 7.35))
        p = lerp(V(0, -50, 17), V(g, -5, 4.2), a)
        tg = lerp(V(0, 0, 6), V(g, 40, 3.6), a)
        b = in_expo(prog(t, 7.35, 8))
        p = lerp(p, V(g, 60, 3), b)
        return p, tg + V(0, 60 * b, 0), lerp(35, 18, max(a * 0.5, b)), 0.0


# =============================================================================
# 2. TWO BAD HOES — mirrored monolith walls, barrel roll, then four pillars
# =============================================================================
class S02Two(Shot):
    name = "two"
    length = 4

    def build(self):
        self.lyric(0, "WITH TWO BAD HOES, TWO BAD HOES")
        self.road(lines=True)
        body = mat_rim("two.slab", rim=WHITE, rim_strength=4)
        txt = mat_emit("two.txt", WHITE, 6)
        red = mat_emit("two.red", RED, 10)

        def wall(name, x, y, length, height, size, side, t_on=None):
            piv = empty(name, loc=(x, y, 0), parent=self.root)
            mesh_obj(name + ".slab", m_cube("slab", (1.2, length, height)), body,
                     loc=(0, 0, height / 2), parent=piv)
            tx = text(name + ".t", "TWO BAD HOES", size=size, extrude=0.08, mat=txt,
                      align_y="CENTER", parent=piv, loc=(-side * 0.65, 0, height / 2))
            tx.rotation_euler = (math.pi / 2, 0, side * math.pi / 2 * -1 if side > 0 else math.pi / 2)
            mesh_obj(name + ".strip", m_cube("strip", (0.1, length, 0.12)), red,
                     loc=(-side * 0.65, 0, 0.4), parent=piv)
            if t_on is not None:
                self.show(piv, None)
                for c in [piv] + list(piv.children):
                    self.show(c, t_on)
                self.bake(piv, "scale", t_on - 0.05, t_on + 0.6,
                          lambda t: (1, 1, max(0.001, out_back(prog(t, t_on, t_on + 0.5), 2.2))))
            return piv

        # first pair: left faces +X, right faces -X
        wall("wL", -7, 15, 34, 9, 3.6, -1)
        wall("wR", 7, 15, 34, 9, 3.6, +1)
        # second generation: four pillars pop on the repeat
        for i, (x, y) in enumerate([(-4.5, 52), (4.5, 52), (-4.5, 72), (4.5, 72)]):
            wall(f"p{i}", x, y, 14, 6, 1.6, -1 if x < 0 else 1, t_on=2.0 + i * 0.1)
        self.hit(0, "pump", 1.0)
        self.hit(2, "shake", 0.8, 0.4)
        self.hit(2, "pump", 1.2)
        self.hit(2, "ca", 0.6, 0.3)

    def cam(self, t):
        if t < 2:
            a = in_out_cubic(t / 2)
            p = lerp(V(0, -24, 4.2), V(0, 36, 4.5), t / 2)
            roll = 2 * math.pi * in_out_expo(prog(t, 0.25, 1.75))
            return p, p + V(0, 30, -0.6), 22, roll
        y = lerp(36, 92, (t - 2) / 2)
        x = 1.2 * math.sin((t - 2) * math.pi * 2)
        p = V(x, y, 3.0)
        return p, p + V(-x * 0.6, 30, -0.4), 22, -0.42 * math.sin((t - 2) * math.pi * 2)


# =============================================================================
# 3. POP — chrome balloon letters, slow-mo burst
# =============================================================================
class S03Pop(Shot):
    name = "pop"
    length = 4

    def build(self):
        self.lyric(0, "AND THEY SAID LET'S GO POP A TESLA")
        self.road(lines=False)
        chrome = mat_principled("pop.chrome", (1.0, 0.12, 0.3), metallic=0.0, rough=0.18, coat=1.0,
                                emit=(1, 0.2, 0.45), emit_strength=0.35)
        root, glyphs, w = word_letters("POP", "POP", size=8, extrude=1.6, bevel=1.1,
                                       mat=chrome, tracking=0.12, parent=self.root, loc=(0, 0, 3))
        self.bake(root, "scale", 0, 2.6,
                  lambda t: (1 + 0.15 * out_cubic(t / 2.5) + 0.05 * math.sin(t * 9),) * 3)
        self.bake(root, "location", 0, 2.6, lambda t: (0, 0, 3 + 0.4 * math.sin(t * 3)))
        for _, piv, g in glyphs:
            self.show(g, None, 2.5)
        small = mat_emit("pop.small", WHITE, 4)
        tx = self.text("pop.small", "LET'S GO POP A TESLA", size=0.9, fnt=MONO, extrude=0.0,
                       mat=small, loc=(0, -4, 0.6))
        self.show(tx, None, 2.5)
        # neon ring for reflections
        cols = [RED, WHITE, (1, 0.2, 0.6), WHITE]
        for i in range(16):
            a = 2 * math.pi * i / 16
            if math.sin(a) < -0.3:
                continue
            m = mat_emit(f"pop.n{i % 4}", cols[i % 4], 14)
            mesh_obj("bar", m_cube("bar", (0.4, 0.4, 18)), m,
                     loc=(30 * math.cos(a), 30 * math.sin(a), 9), parent=self.root)
        # shards
        rng = self.rng
        shards = []
        for i in range(160):
            gi = glyphs[i % 3][1]
            p0 = V(gi.location.x + rng.uniform(-2.5, 2.5), rng.uniform(-1, 1), 3 + rng.uniform(0.5, 8))
            v = V(rng.uniform(-1, 1), rng.uniform(-1.8, 0.4), rng.uniform(-0.4, 1)).normalized() * rng.uniform(6, 22)
            s = mesh_obj("shard", m_shard("shard", rng.uniform(0.25, 0.8), rng), chrome, loc=p0, parent=self.root)
            spin = V(rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-6, 6))
            self.show(s, 2.5)
            self.bake(s, "location", 2.5, 4, lambda t, p0=p0, v=v: p0 + v * self._ramp(t - 2.5))
            self.bake(s, "rotation_euler", 2.5, 4, lambda t, spin=spin: spin * self._ramp(t - 2.5))
        for l in (self.light("key", "AREA", (0, -14, 14), 12000, WHITE, size=12),):
            self.aim(l, (0, 0, 4))
        self.hit(0, "pump", 0.8)
        self.hit(1.5, "pump", 0.8)
        self.hit(2.5, "flash", 1.6, 0.3)
        self.hit(2.5, "shake", 1.2, 0.5)
        self.hit(2.5, "ca", 1.0, 0.5)
        self.batt(2.5, 23)

    @staticmethod
    def _ramp(tau):
        """Burst, slow-mo hang, then release (beats -> distance factor)."""
        if tau < 0.12:
            return 5 * tau
        if tau < 1.2:
            return 0.6 + 0.45 * (tau - 0.12)
        return 0.6 + 0.486 + 4 * (tau - 1.2)

    def cam(self, t):
        if t < 2.5:
            a = in_out_cubic(t / 2.5)
            return lerp(V(0, -44, 5), V(0, -27, 6), a), V(0, 0, 7), lerp(32, 48, a), 0.0
        a = in_out_cubic(prog(t, 2.6, 4))
        p = lerp(V(0, -27, 6), V(1.5, 6, 6.5), a)
        return p, p + V(-0.5, 20, 0.2), lerp(48, 22, a), 0.25 * a


# =============================================================================
# 4. TALK IT OUT — speech-bubble blimp gets harpooned
# =============================================================================
class S04Talk(Shot):
    name = "talk"
    length = 8

    def build(self):
        self.lyric(0, "TRYNA TALK IT OUT")
        self.lyric(4, "FUCK THAT SHIT YEAH WE CAN POKE IT OUT")
        self.road(lines=False)
        self.bub_mat = mat_principled("bubble", (0.85, 0.85, 0.88), rough=0.35, coat=1.0,
                                      emit=(1, 1, 1), emit_strength=0.05)
        bc = self.bub_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"]
        for fr, c in ((self.f(3.9), (0.85, 0.85, 0.88, 1)), (self.f(4.1), (0.8, 0.01, 0.02, 1))):
            bc.default_value = c
            bc.keyframe_insert("default_value", frame=fr)
        self.bub = bub = empty("bubble", loc=(0, 0, 9), parent=self.root)
        mesh_obj("bubble.body", m_uvsphere("b", 1, 48, 24), self.bub_mat, parent=bub).scale = (9, 3.2, 5.2)
        tail = mesh_obj("bubble.tail", m_cyl("tail", 1.6, 4.5, 24, 0.0), self.bub_mat,
                        loc=(-5, 0, -5.6), parent=bub)
        tail.rotation_euler = (0, math.radians(-30), 0)
        ink = mat_principled("ink", (0.01, 0.01, 0.01), rough=0.4)
        t1 = text("talk.t", "TALK IT OUT", size=2.3, extrude=0.25, mat=ink, align_y="CENTER",
                  parent=bub, loc=(0, -3.0, 0))
        self.show(t1, None, 4)
        wht = mat_emit("talk.w", WHITE, 5)
        t2 = text("talk.f", "FUCK THAT SHIT", size=1.8, extrude=0.25, mat=wht, align_y="CENTER",
                  parent=bub, loc=(0, -3.0, 0))
        self.show(t2, 4)
        # deflate + fly away
        self.bake(bub, "scale", 6.6, 8, lambda t: (
            lerp(1, 0.35, out_cubic(prog(t, 6.75, 7.4))) * (1 + 0.15 * math.sin(t * 40) * prog(t, 6.75, 7)),
            1, lerp(1, 0.3, out_cubic(prog(t, 6.75, 7.4)))))
        self.bake(bub, "location", 6.6, 8, lambda t: V(0, 0, 9) + V(36, 20, 26) * in_expo(prog(t, 7.0, 8)))
        self.bake(bub, "rotation_euler", 6.6, 8, lambda t: (0, 9 * in_cubic(prog(t, 6.8, 8)), 4 * in_cubic(prog(t, 6.8, 8))))
        # spike
        spk_m = mat_principled("spike", (1, 0.02, 0.03), metallic=1, rough=0.15, emit=RED, emit_strength=2)
        self.spike = spk = empty("spike", parent=self.root)
        s = mesh_obj("spike.m", m_cyl("spk", 0.9, 18, 24, 0.0), spk_m, parent=spk)
        s.rotation_euler = (0, math.radians(90), 0)   # tip along +X
        s.location = (-9, 0, 0)                       # tip at pivot origin
        self.spike_path = lambda t: lerp(V(-90, -6, 9.5), V(-8.2, -1, 9), in_expo(prog(t, 5.6, 6.75)))
        self.bake(spk, "location", 5.4, 8, lambda t: self.spike_path(t) + (
            V(36, 20, 26) * in_expo(prog(t, 7.0, 8))))
        self.bake(spk, "rotation_euler", 5.4, 8, lambda t: (0, 0, math.atan2(5, 82)))
        self.show(spk, 5.5)
        for c in spk.children:
            self.show(c, 5.5)
        poke = mat_emit("poke", RED, 10)
        root, glyphs, w = word_letters("POKE", "POKE IT OUT", size=2.6, extrude=0.3, mat=poke,
                                       parent=self.root, loc=(0, -6, 0.2))
        for i, (_, piv, g) in enumerate(glyphs):
            ti = 6.75 + i * 0.04
            self.show(g, ti)
            self.bake(piv, "scale", ti, ti + 0.5, lambda t, ti=ti: (lerp(3, 1, out_expo(prog(t, ti, ti + 0.3))),) * 3)
        self.light("key", "AREA", (-10, -20, 22), 9000, WHITE, size=14)
        self.aim(self.col.objects[-1], (0, 0, 8))
        self.hit(0, "pump", 0.6)
        self.hit(4, "shake", 1.0, 0.4)
        self.hit(4, "flash", 0.8, 0.3)
        self.hit(4, "tint", 0.5, 0.6)
        self.hit(6.75, "shake", 1.8, 0.6)
        self.hit(6.75, "flash", 1.5, 0.3)
        self.hit(6.75, "ca", 1.0, 0.5)

    def cam(self, t):
        c = V(0, 0, 9)
        if t < 4:
            a = math.radians(lerp(-125, -70, in_out_cubic(t / 4)))
            p = c + V(26 * math.cos(a), 26 * math.sin(a), 1 + 0.5 * math.sin(t))
            return p, c, 35, 0.0
        a70 = math.radians(-70)
        p70 = c + V(26 * math.cos(a70), 26 * math.sin(a70), 1)
        if t < 5.6:
            a = in_out_cubic(prog(t, 4, 5.6))
            return lerp(p70, V(-30, -26, 11), a), lerp(c, V(-10, 0, 9), a), lerp(35, 28, a), 0.0
        tip = self.spike_path(t)
        if t < 6.75:
            p = tip + V(-9, -15, 3)
            return p, tip + V(12, 4, -0.5), 28, -0.12
        a = out_expo(prog(t, 6.75, 7.6))
        p0 = self.spike_path(6.75) + V(-9, -15, 3)
        return lerp(p0, V(-12, -34, 8), a), lerp(self.spike_path(6.75) + V(12, 4, -0.5), V(6, 0, 10), a), lerp(28, 30, a), lerp(-0.12, 0.05, a)


# =============================================================================
# 5. POLKA DOT — 9 dots, 81 dots, an infinity room
# =============================================================================
class S05Polka(Shot):
    name = "polka"
    length = 8
    tint = (0.006, 0.0, 0.001)
    fog = 0.0015

    def _grid(self, name, n, nz, spacing, r, mat, center, t_on, t_off=None):
        me = m_sphere(name, r, 2)
        sk = None
        o = mesh_obj(name, me, mat, parent=self.root)
        o.shape_key_add(name="Basis")
        big = o.shape_key_add(name="pump")
        for v in big.data:
            v.co *= 1.45
        for axis, cnt in ((0, n), (2, n), (1, nz)):
            if cnt <= 1:
                continue
            a = o.modifiers.new(f"arr{axis}", "ARRAY")
            a.count = cnt
            a.use_relative_offset = False
            a.use_constant_offset = True
            off = [0, 0, 0]
            off[axis] = spacing
            a.constant_offset_displace = off
        o.location = center - V(spacing * (n - 1) / 2, spacing * (nz - 1) / 2, spacing * (n - 1) / 2)
        self.show(o, t_on, t_off)
        # dots pop in, then pump on every kick
        kicks = [k for k in (4, 4.75, 5.33, 6, 6.67, 7, 7.33, 7.67) if k >= t_on]
        end = t_off if t_off is not None else 8
        self.bake(o, "scale", t_on, min(t_on + 0.6, end), lambda t: (out_back(prog(t, t_on, t_on + 0.4), 2.5),) * 3)
        self.bake(big, "value", t_on, end, lambda t: min(1, sum(pulse(t, k, 0, 0.3) for k in kicks)))
        return o

    def build(self):
        self.lyric(0, "PUT MY NUT ON HER FACE LIKE A POLKA DOT")
        self.lyric(4, "POLKA DOT")
        self.lyric(5.33, "POLKA DOT  POLKA DOT")
        self.lyric(6.67, "POLKA DOT  POLKA DOT  POLKA DOT")
        rim = mat_rim("polka.rim", rim=(1, 0.3, 0.35), rim_strength=7)
        root, glyphs, w = word_letters("POLKA", "POLKA DOT", size=6, extrude=0.9, bevel=0.05,
                                       mat=rim, parent=self.root, loc=(0, 0, 1))
        for i, (_, piv, g) in enumerate(glyphs):
            ti = 0.2 + i * 0.12
            self.bake(piv, "location", 0, 4.2, lambda t, p=piv.location.copy(), ti=ti: p + V(0, 0, -9 * (1 - out_expo(prog(t, ti, ti + 0.5)))))
            self.show(g, None, 4)
        self.road(lines=False)
        dot = mat_principled("dot", (1, 0.02, 0.04), metallic=0.6, rough=0.12, emit=RED, emit_strength=1.6)
        self._grid("dots9", 3, 1, 5.0, 1.3, dot, V(0, 0, 7), 4, 5.33)
        self._grid("dots81", 9, 1, 2.2, 0.62, dot, V(0, 0, 7), 5.33, 6.67)
        self._grid("dots3d", 13, 13, 3.4, 0.5, dot, V(0, 22, 7), 6.67)
        self.light("red", "POINT", (0, -10, 12), 20000, (1, 0.4, 0.4), size=4)
        self.light("fill", "AREA", (0, -20, 30), 8000, WHITE, size=30)
        for k in (4, 5.33, 6.67):
            self.hit(k, "pump", 1.4)
            self.hit(k, "shake", 0.7, 0.3)
        self.hit(6.67, "flash", 1.2, 0.3)
        self.hit(6.67, "ca", 0.8, 0.5)

    def cam(self, t):
        if t < 4:
            a = in_out_cubic(t / 4)
            return lerp(V(0, -44, 3), V(0, -32, 6), a), V(0, 0, 5), 35, 0.0
        if t < 6.67:
            a = out_expo(prog(t, 5.33, 6.2))
            return lerp(V(0, -30, 7), V(0, -17, 7), a), V(0, 0, 7), 35, 0.0
        a = in_out_cubic(prog(t, 6.67, 8))
        p = lerp(V(0, -17, 7), V(1.7, 26, 8.7), a)
        yaw = 1.2 * a
        tg = p + V(math.sin(yaw), math.cos(yaw), 0.15) * 10
        return p, tg, lerp(35, 18, a), 1.6 * in_cubic(a)


# =============================================================================
# 6. SLIME MY B — melts in four angles, SLATT erupts
# =============================================================================
class S06Slime(Shot):
    name = "slime"
    length = 8
    tint = (0.002, 0.012, 0.001)
    HITS = (0, 1.5, 3, 4.5)

    def build(self):
        for i, h in enumerate(self.HITS):
            self.lyric(h, "  ".join(["SLIME MY B"] * (i + 1)))
        self.lyric(6, "SLATT")
        self.road(lines=False, mat=GROUND((0.004, 0.012, 0.002)))
        goo = mat_principled("goo", (0.18, 1.0, 0.03), rough=0.06, coat=1.0, emit=GREEN, emit_strength=0.8)
        src = text("slime.src", "SLIME MY B", size=4.6, extrude=0.8, bevel=0.15, parent=None)
        o = to_mesh_obj(src, "slime", voxel=0.11, mat=goo)
        o.parent = self.root
        me = o.data
        xs = [v.co.x for v in me.vertices]
        zs = [v.co.z for v in me.vertices]
        h = max(zs)
        o.shape_key_add(name="Basis")
        for lvl, m in enumerate((0.12, 0.38, 0.72, 1.0)):
            sk = o.shape_key_add(name=f"melt{lvl}")
            for v, kv in zip(me.vertices, sk.data):
                x, y, z = v.co
                d = 0.5 + 0.5 * noise.noise(V(x * 0.7, 0.3, 4.1))
                zf = z * (1 - m * (0.6 + 0.4 * d))
                # drips: narrow columns sag further
                drip = max(0.0, noise.noise(V(x * 3.1, 1.7, 0.5)) - 0.15) * 2.2 * m * h * (z / h)
                zf = max(0.04, zf - drip * (1 - m))
                low = 1 - min(1.0, zf / max(0.5, h * (1 - 0.8 * m)))
                spread = 1 + 3.5 * m * low * low
                kv.co = V(x + 0.9 * m * low * noise.noise(V(x, y, 3.0)), y * spread, zf)
        keys = me.shape_keys.key_blocks
        hits = self.HITS

        def weights(t):
            # continuous melt level 0..4 advancing on each hit
            lvl = 0.0
            for i, hb in enumerate(hits):
                lvl = max(lvl, i + out_expo(prog(t, hb, hb + 0.9)) if t >= hb else lvl)
            lvl = min(4.0, lvl)
            w = [0.0] * 4
            i = int(math.floor(lvl - 1e-6)) if lvl > 0 else -1
            if i < 0:
                return w
            frac = lvl - i
            if i == 0:
                w[0] = frac
            else:
                w[i - 1] = 1 - frac
                w[i] = frac
            return w

        for k in range(4):
            self.bake(keys[f"melt{k}"], "value", 0, 6.2, lambda t, k=k: weights(t)[k])
        self.show(o, None, 6.3)
        # falling drops on each hit
        rng = self.rng
        dm = m_sphere("drop", 0.28, 2)
        for i in range(36):
            hb = hits[i % 4] + rng.uniform(0, 0.6)
            x = rng.choice(xs)
            z0 = h * rng.uniform(0.3, 0.9) * (1 - 0.2 * (i % 4))
            dp = mesh_obj("drop", dm, goo, loc=(x, rng.uniform(-0.6, 0.6), z0), parent=self.root)
            self.show(dp, hb, 6.3)
            self.bake(dp, "location", hb, hb + 0.9, lambda t, x=x, z0=z0, y=dp.location.y, hb=hb:
                      (x, y, max(0.05, z0 - z0 * in_cubic(prog(t, hb, hb + 0.55)))))
            self.bake(dp, "scale", hb, hb + 0.9, lambda t, hb=hb:
                      (1 + 1.5 * prog(t, hb + 0.55, hb + 0.7), 1 + 1.5 * prog(t, hb + 0.55, hb + 0.7),
                       lerp(1.3, 0.15, prog(t, hb + 0.5, hb + 0.65))))
        # SLATT
        slatt_m = mat_rim("slatt", base=(0.05, 0.3, 0.02), rim=(0.7, 1, 0.5), rim_strength=10, face_emit=0.6)
        sl = self.text("slatt", "SLATT", size=10, extrude=1.5, bevel=0.08, mat=slatt_m)
        self.show(sl, 6)
        self.bake(sl, "location", 6, 8, lambda t: (0, 0, lerp(-16, 0, out_expo(prog(t, 6, 6.6)))))
        self.light("g1", "POINT", (-8, -10, 9), 15000, GREEN, size=3)
        self.light("g2", "POINT", (10, 4, 6), 10000, (0.6, 1, 0.5), size=3)
        top = self.light("top", "AREA", (0, -4, 26), 9000, WHITE, size=20)
        for hb in hits:
            self.hit(hb, "pump", 1.2)
            self.hit(hb, "shake", 0.6, 0.3)
        self.hit(6, "flash", 3.0, 0.35)
        self.hit(6, "shake", 2.2, 0.8)
        self.hit(6, "ca", 1.5, 0.6)
        self.hit(6, "pump", 2.0)
        self.batt(6, 41)

    def cam(self, t):
        if t < 1.5:
            a = t / 1.5
            return lerp(V(0, -26, 3.5), V(0, -22, 3.2), a), V(0, 0, 2.6), 35, 0.0
        if t < 3:
            a = (t - 1.5) / 1.5
            ang = math.radians(lerp(-140, -128, a))
            return V(19 * math.cos(ang), 19 * math.sin(ang), 6), V(0, 0, 2), 45, 0.0
        if t < 4.5:
            a = (t - 3) / 1.5
            return V(0, -0.3, 34 - 4 * a), V(0, 0, 0), 35, 0.4 + 0.5 * a
        if t < 6:
            a = (t - 4.5) / 1.5
            return lerp(V(7, -12, 0.3), V(4, -10, 0.25), a), V(-1, 0, 0.5), 24, -0.08
        a = out_expo(prog(t, 6, 6.9))
        p = lerp(V(0, -30, 1.0), V(0, -44, 15), a)
        return p, lerp(V(0, 0, 1), V(0, 0, 7.5), a), 30, 0.0


# =============================================================================
# 7. FUCK YOUR TEAM / CREW / SWAG — stamps, 180 whip, domino collapse
# =============================================================================
class S07Wack(Shot):
    name = "wack"
    length = 8
    SLABS = ((0, "TEAM"), (25, "CREW"), (50, "SWAG"))

    def build(self):
        self.lyric(0, "AND IT'S FUCK YOUR TEAM")
        self.lyric(2, "FUCK YOUR CREW")
        self.lyric(4, "FUCK YOUR SWAG")
        self.lyric(6, "THAT SHIT WACK")
        self.road(y0=-60, y1=150)
        body = mat_rim("wack.slab", rim=WHITE, rim_strength=3.5)
        wtx = mat_emit("wack.w", WHITE, 5)
        stamp = mat_emit("wack.stamp", RED, 9)
        ink = mat_principled("wack.ink", (0.01, 0.01, 0.01))
        pieces = []
        for i, (y, word) in enumerate(self.SLABS):
            piv = empty(f"slab{i}", loc=(0, y + 0.6, 0), parent=self.root)
            mesh_obj("slab", m_cube("slab", (16, 1.2, 9)), body, loc=(0, -0.6, 4.5), parent=piv)
            text(f"slab{i}.t", word, size=4.2, extrude=0.1, mat=wtx, parent=piv, loc=(0, -1.25, 2.6))
            text(f"slab{i}.fy", "FUCK YOUR", size=1.0, fnt=MONO, extrude=0, mat=wtx, parent=piv, loc=(0, -1.25, 7.6))
            # stamp: red plate slaps onto the face on the beat
            st = empty(f"stamp{i}", parent=piv)
            mesh_obj("plate", m_cube("plate", (10, 0.3, 2.6), 0.08), stamp, parent=st)
            text("rej", "REJECTED", size=1.5, fnt=WIDE, extrude=0.02, mat=ink, align_y="CENTER",
                 parent=st, loc=(0, -0.2, 0))
            tb = 2 * i
            self.show(st, tb - 0.2)
            for c in st.children:
                self.show(c, tb - 0.2)
            self.bake(st, "location", tb - 0.2, tb + 0.1, lambda t, tb=tb:
                      (0.5, -1.45 - 0.6 - 7 * (1 - in_expo(prog(t, tb - 0.2, tb))), 4.2))
            st.rotation_euler = (0, math.radians(-11), 0)
            pieces.append((y, piv, 9))
            self.hit(tb, "shake", 1.0, 0.3)
            self.hit(tb, "pump", 1.3)
            self.hit(tb, "flash", 0.5, 0.2)
        # filler dominoes
        for y in [y for y in range(5, 50, 5) if y % 25]:
            piv = empty(f"dom{y}", loc=(0, y + 0.4, 0), parent=self.root)
            mesh_obj("dom", m_cube("dom", (12, 0.8, 6.5)), body, loc=(0, -0.4, 3.25), parent=piv)
            pieces.append((y, piv, 6.5))
        pieces.sort()
        for j, (y, piv, hgt) in enumerate(pieces):
            t0 = 6.15 + j * 0.075
            self.bake(piv, "rotation_euler", t0 - 0.05, t0 + 0.6, lambda t, t0=t0:
                      (-math.radians(84) * in_cubic(prog(t, t0, t0 + 0.38)) + math.radians(4) * pulse(t, t0 + 0.38, 0, 0.2), 0, 0))
        wack = mat_emit("wack.red", RED, 12)
        wk = self.text("wack", "WACK", size=7.5, extrude=0.8, mat=wack, loc=(0, 70, 0))
        wk.rotation_euler = (math.pi / 2, 0, math.pi)
        self.show(wk, 7.2)
        self.bake(wk, "location", 7.2, 8, lambda t: (0, 70, 30 * (1 - in_expo(prog(t, 7.2, 7.45)))))
        self.light("key", "AREA", (8, 30, 20), 20000, WHITE, size=40)
        self.hit(7.45, "shake", 2.0, 0.6)
        self.hit(7.45, "flash", 1.2, 0.3)
        self.hit(7.45, "ca", 1.0, 0.5)
        self.batt(7.45, 58)

    def cam(self, t):
        if t < 6:
            y = -19 + 12.5 * t
            p = V(19, y, 5.0)
            return p, V(0, y + 20, 4.0), 30, 0.03
        a = in_out_expo(prog(t, 6.0, 6.45))
        p = lerp(V(19, 56, 5.0), V(6, 84, 12), out_cubic(prog(t, 6.0, 6.7)))
        ang = math.pi * a
        fwd = V(-19, 20, -1.0)
        back = V(0, 20, 2) - V(6, 84, 12)
        tg = p + lerp(fwd.normalized(), back.normalized(), a) * 30
        if a > 0.999:
            tg = lerp(V(0, 20, 2), V(0, 60, 3), out_cubic(prog(t, 7.0, 7.6)))
        return p, tg, lerp(30, 26, a), 0.25 * math.sin(math.pi * a)


# =============================================================================
# 8. SMOKING EXOTIC — volumetric haze, word condenses and evaporates
# =============================================================================
class S08Exotic(Shot):
    name = "exotic"
    length = 4
    tint = (0.006, 0.002, 0.012)

    def build(self):
        self.lyric(0, "AND I'M SMOKING EXOTIC")
        self.road(lines=False, mat=GROUND((0.006, 0.003, 0.01)))
        vm = bpy.data.materials.new("smoke")
        vm.use_nodes = True
        nt = vm.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        pv = nt.nodes.new("ShaderNodeVolumePrincipled")
        pv.inputs["Color"].default_value = (0.7, 0.55, 1, 1)
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.noise_dimensions = "4D"
        nz.name = "Noise"
        nz.inputs["Scale"].default_value = 0.12
        nz.inputs["Detail"].default_value = 3
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.45
        mr.inputs["From Max"].default_value = 0.75
        mr.inputs["To Max"].default_value = 0.25
        nt.links.new(nz.outputs["Fac"], mr.inputs[0])
        nt.links.new(mr.outputs[0], pv.inputs["Density"])
        nt.links.new(pv.outputs[0], out.inputs["Volume"])
        box = mesh_obj("smokebox", m_cube("sb", (60, 50, 22)), vm, loc=(0, 5, 11), parent=self.root)
        self.bake_mat(vm, "Noise", 0, 4, lambda t: t * 0.35, input_idx=1)
        dm = mat_dissolve("exotic.mat", (0.85, 0.7, 1), 7, scale=1.6)
        tx = self.text("exotic", "EXOTIC", size=6.5, extrude=0.5, mat=dm, loc=(0, 0, 1.2))
        self.bake_mat(dm, "Thresh", 0, 4, lambda t: 0.98 * out_cubic(prog(t, 0.1, 1.6)) * (1 - in_cubic(prog(t, 2.8, 4))))
        self.bake_mat(dm, "Noise", 0, 4, lambda t: t * 0.4, input_idx=1)
        sm = mat_emit("exotic.small", WHITE, 3)
        self.text("exotic.s", "AND I'M SMOKING", size=0.8, fnt=MONO, extrude=0, mat=sm, loc=(0, -2, 8.6))
        for x, c in ((-12, PURPLE), (12, (0.9, 0.4, 1))):
            sp = self.light("beam", "SPOT", (x, 6, 20), 60000, c, spot=math.radians(35), blend=0.8)
            self.aim(sp, (x * 0.3, 0, 0))
        self.hit(0, "pump", 0.6)
        self.hit(2, "pump", 0.6)

    def cam(self, t):
        a = in_out_cubic(t / 4)
        p = lerp(V(-3, -42, 4.5), V(1, -22, 5.5), a)
        return p, V(0, 0, 4.5), 38, 0.06 * math.sin(t * 1.6)


# =============================================================================
# 9. FLAWLESS — faceted glass word inside a spectrum ring
# =============================================================================
class S09Flawless(Shot):
    name = "flawless"
    length = 4
    tint = (0.002, 0.002, 0.004)

    def build(self):
        self.lyric(0, "LOOK AT MY DIAMOND, THEY TOO DAMN FLAWLESS")
        self.road(lines=False)
        glass = mat_rim("diamond", base=(0.55, 0.7, 0.95), rim=(0.85, 0.92, 1), rim_strength=10,
                        metallic=1.0, rough=0.04, face_emit=0.3)
        src = text("flaw.src", "FLAWLESS", size=4.2, extrude=0.9, bevel=0.2, parent=None)
        o = to_mesh_obj(src, "flawless", voxel=0.16, mat=glass)
        dec = o.modifiers.new("facet", "DECIMATE")
        dec.ratio = 0.14
        o.parent = self.root
        for p in o.data.polygons:
            p.use_smooth = False
        bpy.context.view_layer.update()
        cx = sum(v.co.x for v in o.data.vertices) / len(o.data.vertices)
        piv = empty("flaw.piv", loc=(0, 0, 3), parent=self.root)
        o.parent = piv
        o.location = (-cx, 0, -2.2)
        self.bake(piv, "rotation_euler", 0, 4, lambda t: (0, 0, 0.25 * math.sin(t * 1.2)))
        # spectrum ring
        for i in range(24):
            a = 2 * math.pi * i / 24
            hue = i / 24
            c = _hsv(hue, 1, 1)
            m = mat_emit(f"spec{i}", c, 18)
            mesh_obj("bar", m_cube("bar", (0.5, 0.5, 14)), m, loc=(26 * math.cos(a), 26 * math.sin(a), 7),
                     parent=self.root)
        # sweeping light bar
        sw = mat_emit("sweep", WHITE, 40)
        bar = mesh_obj("sweepbar", m_cube("sw", (0.3, 0.3, 20)), sw, parent=self.root)
        self.bake(bar, "location", 0, 4, lambda t: (lerp(-20, 20, in_out_cubic(prog(t, 0.5, 2.5))), 7, 6))
        sm = mat_emit("flaw.small", WHITE, 3)
        self.text("flaw.s", "LOOK AT MY DIAMOND", size=0.8, fnt=MONO, extrude=0, mat=sm, loc=(0, 0, 8.2))
        for i, c in enumerate((WHITE, (0.4, 0.6, 1), (1, 0.5, 0.8))):
            a = 2 * math.pi * i / 3
            self.light(f"k{i}", "POINT", (12 * math.cos(a), 12 * math.sin(a), 10), 9000, c, size=1)
        self.hit(1.5, "flash", 0.6, 0.3)
        self.hit(1.5, "pump", 1.0)
        self.hit(3.6, "flash", 2.5, 0.25)

    def cam(self, t):
        c = V(0, 0, 3)
        ang = math.radians(lerp(-150, -20, in_out_cubic(prog(t, 0, 3.3))))
        r = lerp(30, 21, in_out_cubic(prog(t, 0, 3.3)))
        p = c + V(r * math.cos(ang), r * math.sin(ang), lerp(4, 8, prog(t, 0, 3)))
        b = in_expo(prog(t, 3.3, 4))
        p = lerp(p, c + V(0, 0, 0.5), b)
        return p, c, lerp(35, 20, b), 0.0


def _hsv(h, s, v):
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


# =============================================================================
# 10. MILEAGE -> MILEY — odometer drums, wrecking ball
# =============================================================================
class S10Miley(Shot):
    name = "miley"
    length = 8
    START = (4, 8, 2, 9, 1)
    LETTERS = "MILEY"

    def build(self):
        self.lyric(0, "ALL OF MY GIRLS AIN'T GOT NO MILEAGE")
        self.lyric(4, "MILEY CYRUS")
        self.lyric(6, "MILEY CYRUS")
        R = 3.2
        n = 11
        step = 2 * math.pi / n
        drum_m = mat_principled("drum", (0.015, 0.015, 0.016), rough=0.5)
        glyph_m = mat_emit("odo.glyph", WHITE, 5)
        letter_m = mat_emit("odo.letter", RED, 9)
        frame_m = mat_principled("odo.frame", (0.8, 0.8, 0.82), metallic=1, rough=0.12)
        panel = mat_rim("odo.panel", rim=(1, 0.2, 0.2), rim_strength=2)
        rng = self.rng
        self.parts = []
        dash = empty("dash", parent=self.root)
        mesh_obj("panel", m_cube("p", (26, 3, 12), 0.3), panel, loc=(0, 3.6, 0.5), parent=dash)
        for xs, zs, sx, sz in ((0, 1.6, 15, 1.4), (0, -1.6, 15, 1.4), (-7.4, 0, 0.5, 4.6), (7.4, 0, 0.5, 4.6)):
            fr = mesh_obj("frame", m_cube("fr", (sx, 1.0, sz), 0.15), frame_m, loc=(xs, -3.6, zs), parent=dash)
            self.parts.append(fr)
        ring = mesh_obj("ring", m_torus("ring", 10.5, 0.18, 96, 8), mat_emit("odo.ring", RED, 10),
                        loc=(0, -3.4, 0.5), parent=dash)
        ring.rotation_euler = (math.pi / 2, 0, 0)
        km = mat_emit("odo.km", WHITE, 3)
        text("km", "KM", size=1.0, fnt=MONO, extrude=0, mat=km, parent=dash, loc=(8.8, -3.7, -0.4))
        text("odo", "ODO", size=0.8, fnt=MONO, extrude=0, mat=km, parent=dash, loc=(-8.8, -3.7, 2.6))
        self.drums = []
        for d in range(5):
            x = (d - 2) * 2.7
            drum = empty(f"drum{d}", loc=(x, 0, 0), parent=self.root)
            cyl = mesh_obj("cyl", m_cyl("c", R * 0.96, 2.3, 48), drum_m, parent=drum)
            cyl.rotation_euler = (0, math.pi / 2, 0)
            for k in range(n):
                sp = empty(f"spoke{d}.{k}", parent=drum)
                sp.rotation_euler = (k * step, 0, 0)
                ch = str(k) if k < 10 else self.LETTERS[d]
                text(f"g{d}.{k}", ch, size=1.6, fnt=MONO, extrude=0.03, mat=letter_m if k == 10 else glyph_m,
                     align_y="CENTER", parent=sp, loc=(0, -R, 0))
            # angle schedule: start digit -> roll back to 0 (-2 turns) -> spin to letter
            s0 = -self.START[d] * step
            s1 = -2 * math.pi * 2                       # 0 after two reverse turns
            s2 = -10 * step - 2 * math.pi * (6 + d)     # spins on, lands on its letter
            land = 4.6 + d * 0.13

            def ang(t, s0=s0, s1=s1, s2=s2, land=land):
                if t < 3.4:
                    return lerp(s0, s1 + 0.0, in_out_cubic(prog(t, 0.3, 3.4)))
                return lerp(s1, s2, out_back(prog(t, 4.0, land), 0.8))
            self.drums.append((drum, ang))
        # explosion
        self.ball_pivot = V(0, -1, 32)
        L = 30
        self.ball_ang = lambda t: lerp(-1.25, 0.0, in_cubic(prog(t, 5.9, 6.6))) + 0.5 * out_cubic(prog(t, 6.6, 8))
        ball_m = mat_principled("ball", (0.05, 0.05, 0.055), metallic=1, rough=0.25)
        bp = empty("ballpivot", loc=self.ball_pivot, parent=self.root)
        mesh_obj("ball", m_sphere("ball", 3.6, 4), ball_m, loc=(0, 0, -L), parent=bp)
        link_m = m_torus("link", 0.6, 0.15, 16, 8)
        for i in range(16):
            lk = mesh_obj("link", link_m, ball_m, loc=(0, 0, -i * (L - 3.6) / 16), parent=bp)
            lk.rotation_euler = (0, math.pi / 2 if i % 2 else 0, 0)
        self.bake(bp, "rotation_euler", 5.7, 8, lambda t: (0, -self.ball_ang(t), 0))
        for c in [bp] + list(bp.children):
            self.show(c, 5.7)
        t_hit = 6.6
        for i, (drum, ang) in enumerate(self.drums):
            v = V(rng.uniform(-1.5, 1.5) + (i - 2) * 1.2, rng.uniform(-3, -0.5), rng.uniform(0.2, 1.6)).normalized() * rng.uniform(16, 28)
            spin = V(rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(-8, 8))
            p0 = drum.location.copy()

            def dloc(t, p0=p0, v=v):
                return p0 + v * self._ramp(t - t_hit) + V(0, 0, -6 * max(0, t - t_hit) ** 2)
            self.bake(drum, "location", t_hit - 0.05, 8, dloc)
            self.bake(drum, "rotation_euler", 0, 8, lambda t, ang=ang, spin=spin: V((ang(t), 0, 0)) + spin * self._ramp(t - t_hit))
        for part in [dash]:
            self.bake(part, "location", t_hit - 0.05, 8, lambda t: V(0, 1, 0) * 14 * self._ramp(t - t_hit) + V(0, 0, -3) * self._ramp(t - t_hit))
            self.bake(part, "rotation_euler", t_hit - 0.05, 8, lambda t: (0.6 * self._ramp(t - t_hit), 0, 0.3 * self._ramp(t - t_hit)))
        glass_m = mat_principled("odo.glass", (0.9, 0.9, 1), metallic=0.5, rough=0.05, emit=WHITE, emit_strength=0.3)
        for i in range(110):
            p0 = V(rng.uniform(-7, 7), -3.6, rng.uniform(-2.5, 2.5))
            v = V(rng.uniform(-1, 1), rng.uniform(-2, -0.3), rng.uniform(-0.3, 1.2)).normalized() * rng.uniform(10, 34)
            sp = V(rng.uniform(-9, 9), rng.uniform(-9, 9), rng.uniform(-9, 9))
            s = mesh_obj("glass", m_shard("gs", rng.uniform(0.15, 0.6), rng), glass_m if i % 3 else frame_m,
                         loc=p0, parent=self.root)
            self.show(s, t_hit)
            self.bake(s, "location", t_hit, 8, lambda t, p0=p0, v=v: p0 + v * self._ramp(t - t_hit))
            self.bake(s, "rotation_euler", t_hit, 8, lambda t, sp=sp: sp * self._ramp(t - t_hit))
        k = self.light("key", "AREA", (0, -12, 12), 5000, WHITE, size=10)
        self.aim(k, (0, 0, 0))
        r = self.light("redfill", "POINT", (0, -6, -4), 4000, RED, size=4)
        self.hit(4, "pump", 1.2)
        self.hit(4.6, "shake", 0.4, 0.3)
        self.hit(t_hit, "flash", 3.0, 0.4)
        self.hit(t_hit, "shake", 2.6, 0.9)
        self.hit(t_hit, "ca", 2.0, 0.8)
        self.hit(t_hit, "pump", 2.2)
        self.batt(t_hit, 66)

    @staticmethod
    def _ramp(tau):
        if tau <= 0:
            return 0.0
        if tau < 0.1:
            return 4 * tau
        if tau < 1.0:
            return 0.4 + 0.3 * (tau - 0.1)
        return 0.67 + 1.5 * (tau - 1.0)

    def cam(self, t):
        if t < 3.4:
            a = in_out_cubic(t / 3.4)
            x = lerp(-6.5, 6.5, a)
            return V(x, -12, 1.0 - 0.6 * a), V(x * 0.85, 0, 0), 70, -0.05
        if t < 5.8:
            a = out_expo(prog(t, 3.4, 4.2))
            p = lerp(V(6.5, -12, 0.4), V(0, -29, 3), a)
            p = p + V(0, 3 * prog(t, 4.2, 5.8), 0)
            return p, V(0, 0, 0.3), lerp(70, 50, a), 0.0
        a = out_cubic(prog(t, 5.8, 6.5))
        p = lerp(V(0, -24, 3), V(-3, -40, 6), a)
        b = in_out_cubic(prog(t, 6.6, 8))
        p = lerp(p, V(4, -30, 4), b)
        return p, V(0, -4, 1.5), lerp(50, 30, a), 0.35 * b


# =============================================================================
# 11. POP ONE POP TWO I'M GEEKED — vertigo dolly-zoom on a neon grid
# =============================================================================
class S11Geeked(Shot):
    name = "geeked"
    length = 4

    def build(self):
        self.lyric(0, "POP ONE, POP TWO, I'M GEEKED")
        grid_m = mat_emit("grid", RED, 4)
        g = mesh_obj("gridx", m_cube("gx", (0.06, 400, 0.02)), grid_m, loc=(-120, 60, 0), parent=self.root)
        a = g.modifiers.new("arr", "ARRAY")
        a.count = 61
        a.use_relative_offset = False
        a.use_constant_offset = True
        a.constant_offset_displace = (4, 0, 0)
        g2 = mesh_obj("gridy", m_cube("gy", (400, 0.06, 0.02)), grid_m, loc=(0, -140, 0), parent=self.root)
        a2 = g2.modifiers.new("arr", "ARRAY")
        a2.count = 101
        a2.use_relative_offset = False
        a2.use_constant_offset = True
        a2.constant_offset_displace = (0, 4, 0)
        mesh_obj("floor", m_plane("f", 300, 300), GROUND(), loc=(0, 60, -0.02), parent=self.root)
        pill = mat_rim("pillar", rim=WHITE, rim_strength=4)
        for i in range(10):
            for side in (-1, 1):
                mesh_obj("pillar", m_cube("pl", (1.5, 1.5, 30)), pill, loc=(side * 16, 10 + i * 14, 15), parent=self.root)
        w = mat_emit("geek.w", WHITE, 8)
        for tb, ch, x in ((0, "1", -7), (1, "2", 7)):
            o = self.text(f"n{ch}", ch, size=9, extrude=0.8, mat=mat_rim(f"n{ch}", rim=WHITE, rim_strength=9, face_emit=0.6),
                          loc=(x, 0, 0))
            self.show(o, tb, 2)
            self.bake(o, "scale", tb, 2, lambda t, tb=tb: (out_back(prog(t, tb, tb + 0.3), 3) * (1 - in_expo(prog(t, 1.8, 2))),) * 3)
            self.hit(tb, "pump", 1.5)
            self.hit(tb, "shake", 0.6, 0.3)
        root, glyphs, wd = word_letters("GEEKED", "GEEKED", size=6, extrude=0.6, mat=w, parent=self.root, loc=(0, 0, 1))
        rng = self.rng
        for i, (_, piv, gl) in enumerate(glyphs):
            self.show(gl, 2)
            p0 = piv.location.copy()
            seeds = [V(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(200)]
            self.bake(piv, "location", 2, 4, lambda t, p0=p0, seeds=seeds: p0 + seeds[int(t * 12) % 200] * 0.25 * (0.3 + prog(t, 2, 4)))
            self.bake(piv, "scale", 2, 4, lambda t, i=i: (1 + 0.1 * math.sin(t * 13 + i), 1, 1 + 0.18 * math.sin(t * 17 + i * 2)))
        self.hit(2, "flash", 1.5, 0.3)
        self.hit(2, "ca", 2.0, 1.5)
        self.hit(3, "ca", 1.5, 1.0)
        self.hit(2, "shake", 1.2, 0.5)

    def cam(self, t):
        if t < 2:
            return V(0, -36 + t, 5), V(0, 0, 4), 32, 0.0
        a = in_out_cubic(prog(t, 2, 4))
        d0, d1 = 20.0, 80.0
        d = lerp(d0, d1, a)
        return V(0, -d, 4.5), V(0, 0, 4), 26 * d / d0, math.radians(-22) * a


# =============================================================================
# 12. SHE SAID SHE LIKE WHAT SHE SEE — iris, rack focus
# =============================================================================
class S12See(Shot):
    name = "see"
    length = 4

    def build(self):
        self.lyric(0, "SHE SAID SHE LIKE WHAT SHE SEE")
        self.road(lines=True)
        w = mat_rim("see.she", rim=WHITE, rim_strength=8)
        self.text("she", "SHE", size=5, extrude=0.6, mat=w, loc=(-5, 4, 2))
        r = mat_emit("see.see", RED, 10)
        self.text("seex", "SEE", size=13, extrude=1.0, mat=r, loc=(7, 60, 3))
        blade = mat_principled("blade", (0.03, 0.03, 0.035), metallic=1, rough=0.3)
        ring = mesh_obj("iris.ring", m_torus("ir", 11, 0.6, 96, 12), mat_principled("iris.r", (0.6, 0.6, 0.62), metallic=1, rough=0.15),
                        loc=(0, -14, 6), parent=self.root)
        ring.rotation_euler = (math.pi / 2, 0, 0)
        for i in range(9):
            a = 2 * math.pi * i / 9
            piv = empty(f"blade{i}", loc=(11 * math.cos(a), -14, 6 + 11 * math.sin(a)), parent=self.root)
            bl = mesh_obj("bl", m_cube("bl", (12, 0.15, 6)), blade, loc=(-6, 0, 0), parent=piv)
            # closed: blade swings across the opening; open: tucked to the ring
            self.bake(piv, "rotation_euler", 0, 4, lambda t, a=a: (0, -(a + math.radians(lerp(75, 18, in_out_cubic(prog(t, 0.4, 2.0))))), 0))
        self.hit(0, "pump", 0.8)
        self.hit(2, "pump", 1.0)
        self.light("key", "AREA", (0, -24, 16), 6000, WHITE, size=16)

    def cam(self, t):
        if t < 2.0:
            a = in_out_cubic(t / 2)
            return lerp(V(0, -44, 6), V(0, -26, 6), a), V(0, 0, 5.5), 40, 0.0
        a = in_out_cubic(prog(t, 2.0, 3.0))
        p = lerp(V(0, -26, 6), V(1, -5, 5), a)
        return p, lerp(V(0, 0, 5.5), V(5, 60, 8), in_out_cubic(prog(t, 2.6, 3.6))), 40, 0.0

    def focus(self, t):
        """Focus distance from camera: iris/she first, then rack to SEE."""
        p, _, _, _ = self.cam(t)
        she = V(-3, 4, 4)
        see = V(7, 60, 8)
        a = in_out_cubic(prog(t, 2.8, 3.6))
        return lerp((she - p).length, (see - p).length, a)


# =============================================================================
# 13. HELL NAW — toll barrier, twice; the apostrophe falls off
# =============================================================================
class S13Naw(Shot):
    name = "naw"
    length = 8

    def build(self):
        self.lyric(0, "HELL NAW YOU CAN'T MY B")
        self.lyric(4, "HE'LL NAW YOU CAN'T MY B")
        self.road(y0=-140)
        post = mat_rim("post", rim=WHITE, rim_strength=3)
        for x in (-8, 8):
            mesh_obj("post", m_cube("po", (1.2, 1.2, 12)), post, loc=(x, 0, 6), parent=self.root)
        mesh_obj("beam", m_cube("bm", (17.2, 1.2, 1.2)), post, loc=(0, 0, 12.6), parent=self.root)
        board = mesh_obj("board", m_cube("bd", (16, 0.6, 3.6)), post, loc=(0, -0.6, 10.4), parent=self.root)
        sign_m = mat_emit("naw.sign", RED, 12)
        s1 = self.text("sign1", "HELL NAW", size=2.4, extrude=0.05, mat=sign_m, align_y="CENTER", loc=(0, -1.0, 10.4))
        self.show(s1, None, 4)
        root, glyphs, w = word_letters("sign2", "HE'LL NAW", size=2.4, extrude=0.05, mat=sign_m,
                                       parent=self.root, loc=(0, -1.0, 9.55))
        for _, piv, g in glyphs:
            self.show(g, 4)
        apo = [p for ch, p, g in glyphs if ch == "'"][0]
        a0 = apo.location.copy()
        ax = a0.x

        def apo_loc(t):
            if t < 6:
                return a0
            fall = in_cubic(prog(t, 6, 6.55))
            z = lerp(a0.z, -9.4, fall)
            bounce = 0.9 * abs(math.sin(math.pi * prog(t, 6.55, 7.0))) * (1 - prog(t, 6.55, 7.0))
            x = ax + 4.5 * out_cubic(prog(t, 6.55, 8))
            return V(x, a0.y - 1.2 * prog(t, 6, 6.55), z + bounce)
        self.bake(apo, "location", 5.9, 8, apo_loc)
        self.bake(apo, "rotation_euler", 5.9, 8, lambda t: (0, 7 * out_cubic(prog(t, 6.55, 8)) + 0.6 * prog(t, 6, 6.55), 0))
        # boom arm: red/white stripes, pivots on left post
        arm = empty("arm", loc=(-7.4, -1.2, 3.2), parent=self.root)
        stripe_r = mat_emit("arm.r", RED, 6)
        stripe_w = mat_principled("arm.w", (0.9, 0.9, 0.9), rough=0.3, emit=WHITE, emit_strength=1.5)
        for i in range(8):
            mesh_obj("st", m_cube("st", (1.85, 0.5, 0.5)), stripe_r if i % 2 == 0 else stripe_w,
                     loc=(1 + i * 1.85, 0, 0), parent=arm)

        def arm_rot(t):
            if t < 4:
                return lerp(math.radians(80), 0, out_elastic(prog(t, 0.85, 1.6)) if t >= 0.85 else 0)
            return lerp(math.radians(80), 0, out_elastic(prog(t, 4.85, 5.6)) if t >= 4.85 else 0)
        self.bake(arm, "rotation_euler", 0, 8, lambda t: (0, -arm_rot(t), 0))
        sm = mat_emit("naw.small", WHITE, 4)
        tx = self.text("naw.s", "YOU CAN'T MY B", size=1.3, fnt=MONO, extrude=0, mat=sm, loc=(0, -1.2, 0.3))
        self.light("key", "AREA", (0, -14, 18), 8000, WHITE, size=14)
        for tb in (1.0, 5.0):
            self.hit(tb, "shake", 1.6, 0.4)
            self.hit(tb, "pump", 1.6)
            self.hit(tb, "flash", 0.8, 0.25)
        self.hit(6.55, "pump", 0.6)

    def cam(self, t):
        def rush(u):
            a = out_expo(prog(u, 0, 1.0))
            return lerp(V(0, -110, 3.2), V(0, -15, 4.0), a)
        if t < 4:
            p = rush(t)
            return p, V(0, 0, 6.5), 30, 0.0
        if t < 6.1:
            p = rush(t - 4)
            return p, V(0, 0, 6.5), 30, 0.0
        # side-eye down at the fallen apostrophe
        a = in_out_cubic(prog(t, 6.1, 6.9))
        p = lerp(V(0, -15, 4.0), V(-0.5, -14, 3.6), a)
        apo_x = 2.0 + 4.5 * out_cubic(prog(t, 6.55, 8))
        tg = lerp(V(0, 0, 6.5), V(apo_x - 1, -2.2, 0.2), a)
        return p, tg, lerp(30, 45, a), 0.0


# =============================================================================
# 14. OLD HEADS — surfing the wave, then drowned
# =============================================================================
class S14Wave(Shot):
    name = "wave"
    length = 8
    tint = (0.0, 0.004, 0.01)

    def build(self):
        self.lyric(0, "OLD HEADS TRYNA HOP ON THE WAVE")
        self.lyric(4, "OLD HEADS THEY GON DROWN IN THIS WAVE")
        self.water = ocean(self, (0.0, 0.01, 0.025))
        self.bake(self.water, "location", 0, 8, lambda t: (0, 0, lerp(0, 7, in_out_cubic(prog(t, 4, 5.2)))))
        oc = self.water.modifiers["ocean"]
        oc.keyframe_insert("wave_scale", frame=self.f(4))
        oc.wave_scale = 3.2
        oc.keyframe_insert("wave_scale", frame=self.f(5))
        oc.wave_scale = 1.4
        rider = empty("rider", parent=self.root)
        board_m = mat_principled("board", (0.9, 0.9, 0.9), rough=0.25, coat=1, emit=WHITE, emit_strength=0.4)
        mesh_obj("board", m_cube("bd", (13, 3.2, 0.35), 0.15), board_m, loc=(0, 0, 0.6), parent=rider)
        oh = mat_rim("oldheads", rim=(0.6, 0.85, 1), rim_strength=7)
        self.text("oh", "OLD HEADS", size=2.4, extrude=0.4, mat=oh, loc=(0, 0, 0.8), parent=rider)

        def rider_loc(t):
            bob = 0.5 * math.sin(t * 3.1) + 0.3 * math.sin(t * 5.3)
            sink = -16 * in_cubic(prog(t, 4.6, 8))
            return V(0, 0, 0.6 + bob + sink)
        self.bake(rider, "location", 0, 8, rider_loc)
        self.bake(rider, "rotation_euler", 0, 8, lambda t: (0.12 * math.sin(t * 2.3) - 0.5 * prog(t, 4.6, 8),
                                                        0.15 * math.sin(t * 3.7) + 0.9 * in_cubic(prog(t, 4.6, 8)), 0))
        dr = mat_emit("drown", (0.4, 0.7, 1.0), 9)
        d = self.text("drown", "DROWN", size=5, extrude=0.4, mat=dr, loc=(0, 10, -6))
        self.show(d, 5.4)
        self.bake(d, "location", 5.4, 8, lambda t: (0, 10, lerp(-5, -10, prog(t, 5.4, 8))))
        # bubbles
        rng = self.rng
        bm = mat_principled("bubble", (0.8, 0.9, 1), rough=0.0, emit=(0.6, 0.8, 1), emit_strength=1.2)
        sph = m_sphere("bub", 0.18, 2)
        for i in range(60):
            t0 = rng.uniform(4.8, 7.6)
            x = rng.uniform(-6, 6)
            b = mesh_obj("bub", sph, bm, parent=self.root)
            sc = rng.uniform(0.6, 2.0)
            b.scale = (sc,) * 3
            self.show(b, t0)
            self.bake(b, "location", t0, 8, lambda t, x=x, t0=t0: V(x + 0.3 * math.sin(t * 6 + x), 0.0,
                                                                  rider_loc(t0).z + 9 * (t - t0)))
        sun = self.light("sun", "SPOT", (0, 6, 30), 200000, (0.6, 0.85, 1), spot=math.radians(40), blend=0.9)
        self.aim(sun, (0, 0, -12))
        self.light("moon", "AREA", (-10, -20, 20), 6000, (0.6, 0.8, 1), size=20)
        self.aim(self.col.objects[-1], (0, 0, 0))
        self.hit(4, "shake", 1.2, 0.8)
        self.hit(4, "pump", 1.2)
        self.hit(5.2, "flash", 0.8, 0.4)
        self.batt(4, 79)

    def cam(self, t):
        if t < 4:
            a = t / 4
            p = V(lerp(-12, 4, a), -21, 2.6 + 0.4 * math.sin(t * 2))
            return p, V(0, 0, 1.8), 32, 0.05 * math.sin(t * 1.5)
        a = in_out_cubic(prog(t, 4.8, 5.6))
        p = lerp(V(4, -21, 3.0), V(2, -16, -6), a)
        tg = lerp(V(0, 0, 2), V(0, 2, -9), in_out_cubic(prog(t, 4.6, 6.5)))
        return p, tg, 32, 0.08 * a


def ocean(shot, color, size=80, under=True):
    w = bpy.data.meshes.new("water")
    o = mesh_obj("water", w, mat_principled("water", color, rough=0.04, coat=0.0, metallic=0.0),
                 parent=shot.root)
    oc = o.modifiers.new("ocean", "OCEAN")
    oc.geometry_mode = "GENERATE"
    oc.spatial_size = size
    oc.resolution = 9
    oc.repeat_x = 3
    oc.repeat_y = 3
    oc.wave_scale = 1.4
    oc.choppiness = 1.2
    oc.wind_velocity = 18
    shot.bake(oc, "time", 0, shot.length, lambda t: 1 + t * 0.45)
    if under:
        vm = bpy.data.materials.new("underwater")
        vm.use_nodes = True
        nt = vm.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        pv = nt.nodes.new("ShaderNodeVolumePrincipled")
        pv.inputs["Color"].default_value = (0.2, 0.55, 1.0, 1)
        pv.inputs["Density"].default_value = 0.035
        pv.inputs["Absorption Color"].default_value = (0.0, 0.3, 0.6, 1)
        nt.links.new(pv.outputs[0], out.inputs["Volume"])
        mesh_obj("underwater", m_cube("uw", (240, 240, 60)), vm, loc=(0, 0, -31), parent=o)
    return o


# =============================================================================
# 15. NEW GENERATION / YOUNG SENSATION — the gallery from above, convergence
# =============================================================================
class S15Finale(Shot):
    name = "finale"
    length = 12
    WORDS = ["TESLA", "TWO BAD", "POP", "TALK IT OUT", "POLKA DOT", "SLIME", "SLATT", "TEAM",
             "CREW", "SWAG", "WACK", "EXOTIC", "FLAWLESS", "MILEY", "GEEKED", "SHE SEE",
             "HELL NAW", "WAVE"]

    def build(self):
        self.lyric(0, "WE THE NEW GENERATION")
        self.lyric(4, "I'M A YOUNG SENSATION")
        self.lyric(10, "")
        self.water = ocean(self, (0.0, 0.008, 0.02), under=True)
        self.show(self.water, None, 4)
        for c in self.water.children:
            self.show(c, None, 4)
        rng = self.rng
        n = len(self.WORDS)
        center = V(0, 0, 30)
        palette = [WHITE, RED, GREEN, (1, 0.3, 0.6), (0.4, 0.7, 1), PURPLE]
        for i, wtxt in enumerate(self.WORDS):
            ang = i * 2.4
            r = 14 + i * 2.2
            p0 = V(r * math.cos(ang), r * math.sin(ang), 1.5)
            m = mat_emit(f"fin{i}", palette[i % len(palette)], 7)
            o = self.text(f"w{i}", wtxt, size=2.6 + rng.uniform(0, 1.5), extrude=0.3, mat=m, stand=False, loc=p0)
            o.rotation_euler = (0, 0, ang + math.pi / 2)
            t0 = 1.6 + rng.uniform(0, 0.6)
            spin = V(rng.uniform(-6, 6), rng.uniform(-6, 6), rng.uniform(-6, 6))

            def loc(t, p0=p0, t0=t0, ang=ang, r=r):
                orb = ang + 0.15 * t
                base = V(r * math.cos(orb), r * math.sin(orb), 1.5 + 0.5 * math.sin(t * 2 + r))
                a = in_expo(prog(t, t0, 3.95))
                return lerp(base, center, a)
            self.bake(o, "location", 0, 4, loc)
            self.bake(o, "rotation_euler", 0, 4, lambda t, ang=ang, spin=spin, t0=t0:
                      V((0, 0, ang + math.pi / 2 + 0.15 * t)) + spin * in_cubic(prog(t, t0, 3.95)))
            self.bake(o, "scale", 0, 4, lambda t, t0=t0: (1 - 0.9 * in_expo(prog(t, t0, 3.95)),) * 3)
            self.show(o, None, 4)
        core = mesh_obj("core", m_sphere("core", 1, 3), mat_emit("core", WHITE, 40), loc=center, parent=self.root)
        self.show(core, 3.2, 4.2)
        self.bake(core, "scale", 3.2, 4.2, lambda t: (0.01 + 6 * in_expo(prog(t, 3.2, 4.0)),) * 3)
        ys = mat_emit("young", WHITE, 6)
        root, glyphs, w = word_letters("YOUNG", "YOUNG", size=7, extrude=0.6, mat=ys, parent=self.root,
                                       loc=(0, 0, 32))
        r2, g2, w2 = word_letters("SENS", "SENSATION", size=4.2, extrude=0.5, mat=ys, parent=self.root,
                                  loc=(0, 0, 26))
        for i, (_, piv, g) in enumerate(glyphs + g2):
            ti = 4.0 + i * 0.07
            self.show(g, ti)
            p0 = piv.location.copy()
            self.bake(piv, "location", ti, ti + 0.8, lambda t, p0=p0, ti=ti: p0 + V(0, -6 * (1 - out_expo(prog(t, ti, ti + 0.6))), 0))
        ap = mat_emit("ap.off", RED, 6)
        off = self.text("apoff", "AUTOPILOT: OFF", size=0.9, fnt=MONO, extrude=0, mat=ap, loc=(0, 0, 23.6))
        self.show(off, 9.5)
        self.light("key", "AREA", (0, -20, 40), 3000, WHITE, size=20)
        for k, v in ((0, 88), (1.5, 94), (2.5, 99), (3.0, 100), (3.4, 112), (3.7, 247), (3.95, -1)):
            self.batt(k, v)
        self.hit(4, "flash", 6.0, 0.9)
        self.hit(4, "shake", 2.5, 1.0)
        self.hit(4, "ca", 2.5, 1.0)
        self.hit(4, "pump", 2.5)
        self.hit(2, "pump", 1.0)
        self.hit(3, "pump", 1.2)

    def cam(self, t):
        center = V(0, 0, 30)
        if t < 4:
            a = in_out_cubic(prog(t, 0, 2.2))
            z = lerp(-7, 72, out_cubic(prog(t, 0.1, 2.4)))
            p = V(0, lerp(-4, -24, a), z)
            tg = lerp(V(0, 30, 10), V(0, 0, 0), in_out_cubic(prog(t, 0.2, 1.6)))
            tg = lerp(tg, center, in_out_cubic(prog(t, 2.4, 3.6)))
            p = lerp(p, V(0, -42, 44), in_out_cubic(prog(t, 2.4, 3.8)))
            return p, tg, lerp(18, 30, a), 0.0
        ang = math.radians(-90 + 14 * in_out_cubic(prog(t, 4, 12)))
        r = lerp(40, 30, out_cubic(prog(t, 4, 12)))
        p = V(r * math.cos(ang), r * math.sin(ang), 31)
        return p, V(0, 0, 29.5), 35, 0.0


SHOTS = [S01Tesla, S02Two, S03Pop, S04Talk, S05Polka, S06Slime, S07Wack, S08Exotic,
         S09Flawless, S10Miley, S11Geeked, S12See, S13Naw, S14Wave, S15Finale]
