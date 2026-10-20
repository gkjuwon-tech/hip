"""Line 1: "Tesla I just pulled off in that fucking Tesla".

One continuous road, one camera move. Every word is its own 3D object that
lands on its beat:

    t (beats)  word        object
    0 - 2      (pre-roll)  dark road, headlights flicker on
    2          TESLA       monolith letters punch up out of the asphalt
    3 / 3.5    I / JUST    road-paint words light up as the camera cranes over
    4          PULLED      paint word, camera launches down the road
    4.75       OFF         paint word flips up like a ramp, camera hops it
    6 / 6.5    IN / THAT   gantry sign lights
    7          FUCKING     red neon on the gantry strobes, camera passes under
    8          TESLA       second monolith slams down from the sky
    9.4 - 10.5 (tail)      dive through the E/S gap, white-out

Word timings are a guess at 140 BPM until the real track is in; every word's
beat lives in WORDS below so it can be retimed in one place.
"""
import math
import random

import bpy
from mathutils import Vector

from lib import (F, B, lerp, prog, out_expo, in_expo, in_out_cubic, out_cubic, in_cubic,
                 out_back, pulse, spline, new_collection, set_active_collection, empty, mesh_obj,
                 text, word_letters, key, bake, m_cube, m_plane, mat_principled, mat_emit,
                 mat_rim, HEAVY, MONO)

FX = {"pump": [], "shake": [], "flash": [], "ca": [], "tint": []}

WHITE = (1, 1, 1)
RED = (1.0, 0.03, 0.06)

PRE = 2.0  # beats of pre-roll before the vocal
WORDS = {
    "TESLA": 0.0, "I": 1.0, "JUST": 1.5, "PULLED": 2.0, "OFF": 2.75,
    "IN": 4.0, "THAT": 4.5, "FUCKING": 5.0, "TESLA2": 6.0,
}


def W(word):
    """Shot-local beat of a word (pre-roll included)."""
    return PRE + WORDS[word]


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


class Shot:
    name = "shot"
    length = 8
    tint = (0.004, 0.004, 0.006)
    fog = 0.004

    def __init__(self, b0, origin):
        self.b0 = b0
        self.b1 = b0 + self.length
        self.origin = Vector(origin)
        self.col = new_collection(f"{b0:03.0f}_{self.name}")
        set_active_collection(self.col)
        self.root = empty(f"{self.name}.root", loc=self.origin)
        self.rng = random.Random(7)
        self.headlights = []

    def f(self, t):
        return F(self.b0 + t)

    def t_of(self, frame):
        return B(frame) - self.b0

    def bake(self, obj, path, t0, t1, fn):
        bake(obj, path, self.f(t0), self.f(t1), lambda fr: fn(self.t_of(fr)))

    def bake_socket(self, sock, t0, t1, fn):
        for fr in range(self.f(t0), self.f(t1) + 1):
            sock.default_value = fn(self.t_of(fr))
            sock.keyframe_insert("default_value", frame=fr)

    def hit(self, t, kind, amt=1.0, decay=0.35):
        FX[kind].append((self.b0 + t, amt, decay))

    def show(self, obj, t_on=None, t_off=None):
        if t_on is not None:
            key(obj, "hide_render", 1, True, "CONSTANT")
            key(obj, "hide_render", self.f(t_on), False, "CONSTANT")
        if t_off is not None:
            key(obj, "hide_render", self.f(t_off), True, "CONSTANT")

    def light(self, name, kind, loc, energy, color=WHITE, size=0.5, spot=math.radians(45), blend=0.3,
              parent=None):
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
        o.parent = parent or self.root
        self.col.objects.link(o)
        return o

    def aim(self, obj, target):
        d = Vector(target) - Vector(obj.location)
        obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

    def finish(self):
        pass


def emit_mat(name, color, strength):
    """Emission material whose strength is driven by node 'Str' (keyable)."""
    m = mat_emit(name, color, strength)
    m.node_tree.nodes["Emission"].name = "Str"
    return m


class Tesla(Shot):
    name = "tesla"
    length = PRE + 8.5

    def build(self):
        self.ground()
        self.monolith()
        self.road_words()
        self.gantry()
        self.monolith2()
        self.lights()

    # --- world ----------------------------------------------------------------
    def ground(self):
        g = mat_principled("asphalt", (0.006, 0.006, 0.008), rough=0.12)
        nt = g.node_tree
        p = nt.nodes["Principled BSDF"]
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 0.09
        nz.inputs["Detail"].default_value = 5
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 0.04
        mr.inputs["To Max"].default_value = 0.5
        nt.links.new(nz.outputs["Fac"], mr.inputs[0])
        nt.links.new(mr.outputs[0], p.inputs["Roughness"])
        mesh_obj("ground", m_plane("ground", 300, 200), g, loc=(0, 120, 0), parent=self.root)
        lane = mat_emit("lane", WHITE, 2.5)
        dash = m_cube("dash", (0.2, 3.0, 0.02))
        for y in range(-40, 320, 9):
            mesh_obj("dash", dash, lane, loc=(0, y, 0.01), parent=self.root)
        side = m_cube("side", (0.14, 360, 0.02))
        for x in (-8, 8):
            mesh_obj("side", side, lane, loc=(x, 140, 0.01), parent=self.root)
        neon = mat_emit("kerb", RED, 10)
        for x in (-16, 16):
            mesh_obj("kerb", m_cube("kerb", (0.25, 360, 0.25)), neon, loc=(x, 140, 0.13), parent=self.root)
        # streetlight poles every 24 units: sells the speed
        pole = mat_principled("pole", (0.02, 0.02, 0.02), metallic=1, rough=0.3)
        lamp = mat_emit("lamp", (1, 0.85, 0.7), 14)
        for y in range(10, 300, 24):
            for x in (-20, 20):
                mesh_obj("pole", m_cube("pole", (0.3, 0.3, 11)), pole, loc=(x, y, 5.5), parent=self.root)
                mesh_obj("lamp", m_cube("lampbar", (3.2, 0.3, 0.12)), lamp,
                         loc=(x - math.copysign(1.6, x), y, 11), parent=self.root)

    # --- TESLA #1: rises out of the road --------------------------------------
    def monolith(self):
        t0 = W("TESLA")
        self.rim1 = mat_rim("tesla1", rim=WHITE, rim_strength=8, face_emit=0.12)
        root, glyphs, w = word_letters("T1", "TESLA", size=7, extrude=1.2, bevel=0.06,
                                       mat=self.rim1, tracking=0.1, parent=self.root, loc=(0, 14, 0))
        for i, (_, piv, g) in enumerate(glyphs):
            ti = t0 - 0.06 + i * 0.035
            self.bake(piv, "location", 0, t0 + 1, lambda t, p=piv.location.copy(), ti=ti:
                      p + V(0, 0, -7.5 * (1 - out_expo(prog(t, ti, ti + 0.35)))))
            self.bake(piv, "scale", t0, t0 + 1, lambda t, ti=ti: (1, 1, 1 + 0.12 * pulse(t, ti + 0.3, 0, 0.3)))
        self.bake_socket(self.rim1.node_tree.nodes["RimStrength"].inputs[1], 0, t0 + 2,
                         lambda t: 4 + 16 * pulse(t, t0 + 0.25, 0, 0.6))
        self.hit(t0, "pump", 1.6)
        self.hit(t0 + 0.25, "shake", 1.4, 0.5)
        self.hit(t0 + 0.25, "flash", 0.5, 0.3)

    # --- I JUST PULLED OFF: road paint ----------------------------------------
    def road_words(self):
        self.paint = {}
        for word, y, size in (("I", 28, 6.5), ("JUST", 40, 4.5), ("PULLED", 52, 4.2), ("OFF", 68, 6.0)):
            t0 = W(word)
            m = emit_mat(f"paint.{word}", WHITE, 0.25)
            piv = empty(f"paint.{word}", loc=(0, y, 0.03), parent=self.root)
            text(f"paint.{word}.t", word, size=size, extrude=0.02, mat=m, stand=False,
                 align_y="BOTTOM_BASELINE", parent=piv)
            self.bake_socket(m.node_tree.nodes["Str"].inputs[1], t0 - 0.1, t0 + 2,
                             lambda t, t0=t0: 0.25 + 22 * pulse(t, t0, 0, 0.35) + 4 * prog(t, t0, t0 + 0.01))
            self.bake(piv, "scale", t0 - 0.05, t0 + 0.6, lambda t, t0=t0:
                      (1 + 0.3 * pulse(t, t0, 0, 0.3),) * 3)
            self.paint[word] = piv
            self.hit(t0, "pump", 0.8)
        # OFF flips up like a ramp, then slaps back down once the camera has hopped it
        off = self.paint["OFF"]
        t0 = W("OFF")
        self.bake(off, "rotation_euler", t0 - 0.1, t0 + 1.4, lambda t: (
            math.radians(80) * out_back(prog(t, t0, t0 + 0.3), 2.2) * (1 - in_cubic(prog(t, t0 + 0.75, t0 + 1.0))), 0, 0))
        self.hit(t0, "shake", 0.8, 0.3)
        self.hit(t0 + 1.0, "shake", 0.6, 0.25)
        self.hit(W("PULLED"), "shake", 1.0, 0.4)

    # --- IN THAT FUCKING: gantry ----------------------------------------------
    def gantry(self):
        y = 132
        steel = mat_rim("gantry", rim=WHITE, rim_strength=3, face_emit=0.05)
        for x in (-12, 12):
            mesh_obj("gpost", m_cube("gp", (0.9, 0.9, 13)), steel, loc=(x, y, 6.5), parent=self.root)
        mesh_obj("gbeam", m_cube("gb", (25, 1.4, 4.6)), steel, loc=(0, y, 11.4), parent=self.root)
        for word, x, size, col in (("IN", -8.5, 1.6, WHITE), ("THAT", -4.0, 1.6, WHITE)):
            t0 = W(word)
            m = emit_mat(f"g.{word}", col, 0.0)
            text(f"g.{word}", word, size=size, fnt=MONO, extrude=0.02, mat=m, align_y="CENTER",
                 parent=self.root, loc=(x, y - 0.75, 12.6))
            self.bake_socket(m.node_tree.nodes["Str"].inputs[1], t0 - 0.1, t0 + 0.6,
                             lambda t, t0=t0: 8 * prog(t, t0, t0 + 0.02) + 20 * pulse(t, t0, 0, 0.25))
            self.hit(t0, "pump", 0.6)
        t0 = W("FUCKING")
        m = emit_mat("g.fucking", RED, 0.0)
        text("g.fucking", "FUCKING", size=3.3, extrude=0.25, mat=m, align_y="CENTER",
             parent=self.root, loc=(1.5, y - 0.9, 10.4))

        def strobe(t):
            if t < t0:
                return 0.0
            on = int((t - t0) * 8) % 2 == 0 if t < t0 + 0.75 else True
            return (30 if on else 1.5) + 10 * pulse(t, t0, 0, 0.4)
        self.bake_socket(m.node_tree.nodes["Str"].inputs[1], t0 - 0.1, t0 + 2.5, strobe)
        red = self.light("gantry.red", "AREA", (0, y - 6, 9), 0, RED, size=20)
        self.aim(red, (0, y - 20, 0))
        self.bake(red.data, "energy", t0 - 0.1, t0 + 2.5, lambda t: 0 if t < t0 else 0.4 * strobe(t) * 2000)
        self.hit(t0, "flash", 1.6, 0.25)
        self.hit(t0, "tint", 1.0, 0.6)
        self.hit(t0, "shake", 1.3, 0.5)
        self.hit(t0, "ca", 1.2, 0.5)
        self.hit(t0, "pump", 1.4)

    # --- TESLA #2: slams down from the sky ------------------------------------
    def monolith2(self):
        t0 = W("TESLA2")
        y = 205
        self.rim2 = mat_rim("tesla2", base=(0.05, 0.01, 0.012), rim=(1, 0.25, 0.25), rim_strength=9,
                            face_emit=0.12)
        root, glyphs, w = word_letters("T2", "TESLA", size=10, extrude=1.6, bevel=0.08,
                                       mat=self.rim2, tracking=0.45, parent=self.root, loc=(0, y, 0))
        xs = [p.location.x for _, p, _ in glyphs]
        self.gap = (xs[1] + xs[2]) / 2
        self.y2 = y
        for i, (_, piv, g) in enumerate(glyphs):
            ti = t0 - 0.2 + abs(i - 2) * 0.05
            self.show(g, ti - 0.05)
            self.bake(piv, "location", 0, t0 + 1, lambda t, p=piv.location.copy(), ti=ti:
                      p + V(0, 0, 45 * (1 - in_expo(prog(t, ti, ti + 0.22)))))
            self.bake(piv, "scale", t0 - 0.3, t0 + 1, lambda t, ti=ti:
                      (1 + 0.08 * pulse(t, ti + 0.22, 0, 0.3), 1, 1 - 0.14 * pulse(t, ti + 0.22, 0, 0.3)))
        land = t0 + 0.05
        self.hit(land, "shake", 2.2, 0.7)
        self.hit(land, "flash", 1.2, 0.3)
        self.hit(land, "ca", 1.0, 0.5)
        self.hit(land, "pump", 1.8)
        self.hit(PRE + 8.3, "flash", 5.0, 1.2)   # white-out as we punch through

    # --- lights ---------------------------------------------------------------
    def lights(self):
        # headlights: build.py parents these to the camera
        on = PRE - 0.6
        for x in (-1.3, 1.3):
            sp = self.light("head", "SPOT", (x, -0.8, 0.2), 0, WHITE, spot=math.radians(32), blend=0.7)
            sp.rotation_euler = (math.radians(-6), 0, 0)
            self.bake(sp.data, "energy", 0, self.length, lambda t: 14000 * (
                0.0 if t < on else 0.6 if t < on + 0.12 else 0.0 if t < on + 0.25 else 1.0))
            self.headlights.append(sp)
        k = self.light("key", "AREA", (0, -10, 26), 9000, WHITE, size=40)
        self.aim(k, (0, 20, 0))
        self.hit(PRE - 0.35, "flash", 0.4, 0.3)

    # --- camera ---------------------------------------------------------------
    def cam(self, t):
        g, y2 = self.gap, self.y2
        P = [
            (0.0, V(0.7, -14, 0.5)),
            (W("TESLA") - 0.1, V(0.6, -13.2, 0.5)),
            (W("TESLA") + 0.5, V(0.5, -12, 0.7)),
            (W("JUST") - 0.2, V(0.0, -3, 8.5)),
            (W("PULLED"), V(0.0, 8, 10.0)),
            (W("OFF"), V(0.0, 42, 5.0)),
            (W("OFF") + 0.55, V(0.0, 68, 6.0)),
            (W("IN"), V(0.0, 92, 3.0)),
            (W("FUCKING"), V(0.0, 116, 2.6)),
            (W("FUCKING") + 0.7, V(0.0, 134, 2.4)),
            (W("TESLA2"), V(0.0, y2 - 58, 3.2)),
            (W("TESLA2") + 0.9, V(0.0, y2 - 44, 4.8)),
            (PRE + 7.6, V(g * 0.6, y2 - 22, 4.5)),
            (PRE + 8.5, V(g, y2 + 30, 3.4)),
        ]
        T = [
            (0.0, V(0, 14, 2.4)),
            (W("TESLA") + 0.5, V(0, 14, 4.2)),
            (W("JUST") - 0.2, V(0, 34, 0.5)),
            (W("PULLED"), V(0, 52, 0.0)),
            (W("OFF"), V(0, 72, 0.5)),
            (W("IN"), V(0, 132, 9.0)),
            (W("FUCKING"), V(0, 132, 10.8)),
            (W("FUCKING") + 0.7, V(0, 150, 11.0)),
            (W("TESLA2"), V(0, y2, 6.0)),
            (W("TESLA2") + 0.9, V(0, y2, 5.5)),
            (PRE + 7.6, V(g, y2, 4.0)),
            (PRE + 8.5, V(g, y2 + 80, 3.4)),
        ]
        L = [
            (0.0, 18.0), (W("TESLA") + 0.5, 18.0), (W("JUST") - 0.2, 22.0), (W("PULLED"), 26.0),
            (W("OFF"), 20.0), (W("IN"), 22.0), (W("FUCKING") + 0.7, 16.0), (W("TESLA2"), 24.0),
            (W("TESLA2") + 0.9, 30.0), (PRE + 7.6, 24.0), (PRE + 8.5, 14.0),
        ]
        R = [(0.0, 0.10), (W("TESLA") + 0.5, 0.06), (W("JUST"), 0.0), (W("OFF"), -0.05),
             (W("FUCKING"), 0.0), (W("FUCKING") + 0.7, 0.18), (W("TESLA2"), 0.0), (PRE + 8.5, 0.0)]
        return spline(P, t), spline(T, t), spline(L, t), spline(R, t)


SHOTS = [Tesla]
