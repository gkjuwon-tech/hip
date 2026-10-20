"""Line 1, word 1: "Tesla".

White wall, black 8-bit type built from real pixel blocks, one hard shadow.
"Tes" and "la" slam in pixel by pixel on the syllables; the word hangs big on
the left diagonal; the camera breathes back, then dives through the counter
(the hole) of the "a" and out into white.

Everything moves on twos (12 fps steps on a 24 fps timeline) with no motion
blur, the way hand-made motion graphics do it: decisive poses, no floaty
in-betweens.
"""
import math
import random

import bpy
from mathutils import Vector

from lib import (F, B, lerp, prog, in_expo, in_out_cubic, out_cubic, new_collection,
                 set_active_collection, empty, mesh_obj, text, key, m_cube, m_plane,
                 mat_principled)

FX = {"pump": [], "shake": [], "flash": [], "ca": [], "tint": []}

FONT = "PressStart2P-Regular.ttf"
SIZE = 10.0
PX = None                  # pixel pitch, measured from the cap height of "T" (7 px)
DEPTH = 1.6                # half-depth of each pixel block
TILT = math.radians(14)    # left diagonal: baseline rises to the right
WALL = 7.0                 # distance from the word to the white wall behind it

# beats (shot-local)
TES = 0.5
LA = 1.0
BREATHE = 1.5              # camera eases back a touch (anticipation)
DIVE = 1.75                # ...then commits
THROUGH = 2.5              # crosses the face of the "a"
END = 3.0


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def twos(fr):
    """Snap a frame to the previous odd frame: hold every pose for 2 frames."""
    return fr - ((fr - 1) % 2)


class Shot:
    name = "shot"
    length = 8
    tint = (1.0, 1.0, 1.0)
    world_strength = 1.0
    on_twos = False
    motion_blur = True

    def __init__(self, b0, origin):
        self.b0 = b0
        self.b1 = b0 + self.length
        self.origin = Vector(origin)
        self.col = new_collection(f"{b0:03.0f}_{self.name}")
        set_active_collection(self.col)
        self.root = empty(f"{self.name}.root", loc=self.origin)
        self.rng = random.Random(7)

    def f(self, t):
        return F(self.b0 + t)

    def show(self, obj, frame):
        key(obj, "hide_render", 1, True, "CONSTANT")
        key(obj, "hide_render", frame, False, "CONSTANT")

    def finish(self):
        pass


def _glyph_tris(ch):
    src = text("_px", ch, size=SIZE, fnt=FONT, extrude=0.0, align_x="LEFT", stand=False, parent=None)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(src.evaluated_get(dg))
    bpy.data.objects.remove(src)
    me.calc_loop_triangles()
    tris = [[me.vertices[i].co.copy() for i in t.vertices] for t in me.loop_triangles]
    bpy.data.meshes.remove(me)
    return tris


def glyph_pixels(ch):
    """Rasterise one Press Start 2P glyph into its 8-bit pixel grid.
    Returns a set of (i, j) cells in pixel units, origin at the pen position."""
    global PX
    if PX is None:
        t = _glyph_tris("T")
        zs = [v.y for tri in t for v in tri]
        PX = (max(zs) - min(zs)) / 7
    tris = _glyph_tris(ch)
    if not tris:
        return set()
    x0 = round(min(v.x for tri in tris for v in tri) / PX)
    x1 = round(max(v.x for tri in tris for v in tri) / PX)
    y0 = round(min(v.y for tri in tris for v in tri) / PX)
    y1 = round(max(v.y for tri in tris for v in tri) / PX)

    def inside(px, py):
        for a, b, c in tris:
            d1 = (px - b.x) * (a.y - b.y) - (a.x - b.x) * (py - b.y)
            d2 = (px - c.x) * (b.y - c.y) - (b.x - c.x) * (py - c.y)
            d3 = (px - a.x) * (c.y - a.y) - (c.x - a.x) * (py - a.y)
            if not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0)):
                return True
        return False
    return {(i, j) for i in range(x0, x1) for j in range(y0, y1)
            if inside((i + 0.5) * PX, (j + 0.5) * PX)}


def enclosed_center(cells):
    """Centre (in cells) of the empty region fully enclosed by ink: the counter."""
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, x1, y0, y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
    outside, stack = set(), [(x0, y0)]
    while stack:
        c = stack.pop()
        if c in outside or c in cells or not (x0 <= c[0] <= x1 and y0 <= c[1] <= y1):
            continue
        outside.add(c)
        stack += [(c[0] + 1, c[1]), (c[0] - 1, c[1]), (c[0], c[1] + 1), (c[0], c[1] - 1)]
    hole = [(i, j) for i in range(x0, x1 + 1) for j in range(y0, y1 + 1)
            if (i, j) not in cells and (i, j) not in outside]
    return (sum(i for i, _ in hole) / len(hole) + 0.5, sum(j for _, j in hole) / len(hole) + 0.5)


def block_mat():
    """Flat graphic shading, no lights involved: faces looking at the camera are
    pure black, faces turned away go to mid grey, so the blocks read as solid."""
    m = bpy.data.materials.new("ink")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0, 0, 0, 1)
    els[1].position, els[1].color = 0.35, (0.42, 0.42, 0.42, 1)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(lw.outputs["Facing"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], em.inputs[0])
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


class Tesla(Shot):
    name = "tesla"
    length = END
    world_strength = 0.82      # ambient on the wall; the sun adds the rest
    on_twos = True
    motion_blur = False

    def build(self):
        ink = block_mat()
        word = "Tesla"
        cells_by_char = {ch: glyph_pixels(ch) for ch in word}
        cube = m_cube("px", (PX, 2 * DEPTH, PX))
        self.word = wroot = empty("word", parent=self.root)
        wroot.rotation_euler = (0, -TILT, 0)
        adv = 8 * PX                      # monospace advance: 8 pixels per glyph
        total = adv * len(word)
        rng = self.rng
        for k, ch in enumerate(word):
            cells = cells_by_char[ch]
            gx = -total / 2 + k * adv
            g = empty(f"g.{ch}", loc=(gx, 0, -3.5 * PX), parent=wroot)
            hit = TES if k < 3 else LA
            f0 = twos(self.f(hit))
            order = list(cells)
            rng.shuffle(order)
            for n, (i, j) in enumerate(order):
                # each block lands within 3 steps of the syllable, falling in from the camera side
                fi = f0 + 2 * min(2, int(3 * n / len(order)))
                b = mesh_obj("px", cube, ink, loc=((i + 0.5) * PX, 0, (j + 0.5) * PX), parent=g)
                self.show(b, fi)
                land = b.location.copy()
                key(b, "location", fi, land + V(0, -4.0, 0), "CONSTANT")
                key(b, "location", fi + 2, land, "CONSTANT")
            if ch == "a":
                ci, cj = enclosed_center(cells)
                self.hole_local = (g, V(ci * PX, 0, cj * PX))
        # the word punches once on "la"
        fl = twos(self.f(LA)) + 4
        for fr, s in ((1, 1.0), (fl, 1.035), (fl + 2, 1.0)):
            key(wroot, "scale", fr, (s, s, s), "CONSTANT")

        # white wall behind, catching one hard shadow
        paper = mat_principled("paper", (1, 1, 1), rough=1.0)
        wall = mesh_obj("wall", m_plane("wall", 400, 400), paper, loc=(0, DEPTH + WALL, 0), parent=self.root)
        wall.rotation_euler = (math.pi / 2, 0, 0)
        sd = bpy.data.lights.new("sun", "SUN")
        sd.energy = 0.75
        sd.angle = math.radians(0.2)
        sd.use_shadow = True
        sd.shadow_maximum_resolution = 0.001
        sun = bpy.data.objects.new("sun", sd)
        sun.parent = self.root
        sun.rotation_euler = V(-0.45, 1.0, -0.55).to_track_quat("-Z", "Y").to_euler()
        self.col.objects.link(sun)

        bpy.context.view_layer.update()
        g, p = self.hole_local
        self.hole = g.matrix_world @ p - self.root.matrix_world.translation
        self.width = total
        print(f"[tesla] counter of 'a' at {tuple(round(v, 2) for v in self.hole)}")

    def cam(self, t):
        h = self.hole
        # whole word big, hung off the left margin, filling ~85% of the width;
        # camera sits off-axis (right and above) so block depth and shadow show
        vis = self.width * math.cos(TILT)
        fw = vis / 0.85
        cx = -vis / 2 + fw * 0.44
        d = fw * 50 / 36
        far = V(cx + 0.22 * d, -d, 0.16 * d)
        look = V(cx, 0, 0)
        # anticipation: ease back ~7% and counter-roll a hair
        back = in_out_cubic(prog(t, BREATHE, DIVE)) * (1 - prog(t, DIVE, DIVE + 0.25))
        p = far + (far - look) * 0.07 * back
        roll = -0.03 * back
        # dive: constant-rate zoom (distance shrinks geometrically) that keeps
        # accelerating, so every 2-frame step reads as a step closer into the "a"
        s = prog(t, DIVE, THROUGH) ** 1.6
        mouth = V(h.x, -DEPTH - 0.3, h.z)
        a = 1.0 if t >= THROUGH else 1 - 0.004 ** s
        p = lerp(p, mouth, a)
        tgt = lerp(look, V(h.x, 0, h.z), in_out_cubic(prog(t, DIVE, THROUGH - 0.25)))
        roll = roll + TILT * in_out_cubic(prog(t, DIVE, THROUGH))
        lens = lerp(50.0, 24.0, in_expo(prog(t, DIVE + 0.25, THROUGH)))
        if t > THROUGH:
            p = mouth + V(0, 60 * (t - THROUGH), 0)
            tgt = V(h.x, 80, h.z)
        return p, tgt, lens, roll


SHOTS = [Tesla]
