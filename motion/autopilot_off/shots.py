"""Line 1, word 1: "Tesla".

White page, black pixel type (8-bit Press Start 2P or 16-bit DotGothic16),
nothing else. The word sits big and tilted on the left diagonal; "Tes" and
"la" cut on with the syllables, then the camera rolls level and flies
straight through the counter (the hole) of the "a".
"""
import math
import random

import bpy
from mathutils import Vector

from lib import (F, B, lerp, prog, in_expo, out_expo, in_out_cubic, spline, new_collection,
                 set_active_collection, empty, word_letters, key, bake, mat_emit, HEAVY)

FX = {"pump": [], "shake": [], "flash": [], "ca": [], "tint": []}

FONTS = {"8bit": "PressStart2P-Regular.ttf", "16bit": "DotGothic16-Regular.ttf"}
FONT = "8bit"
DEPTH = 1.6            # half-depth of the extruded pixel blocks

GRID = 0.25            # 16th note, in beats
TILT = math.radians(14)  # left-diagonal: baseline rises to the right

# beats (shot-local)
TES = 0.5
LA = 1.0
DIVE = 1.0             # camera starts its push on "la"
THROUGH = 2.0          # camera crosses the plane of the word
END = 2.5


def V(*a):
    return Vector(a[0] if len(a) == 1 else a)


def snap(t):
    return round(t / GRID) * GRID


class Shot:
    name = "shot"
    length = 8
    tint = (1.0, 1.0, 1.0)
    fog = 0.0

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

    def show(self, obj, t_on=None, t_off=None):
        if t_on is not None:
            key(obj, "hide_render", 1, True, "CONSTANT")
            key(obj, "hide_render", self.f(t_on), False, "CONSTANT")
        if t_off is not None:
            key(obj, "hide_render", self.f(t_off), True, "CONSTANT")

    def finish(self):
        pass


def counter_center(glyph, res=160):
    """World-space centre of the largest enclosed hole in a flat glyph."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(glyph.evaluated_get(dg))
    me.calc_loop_triangles()
    tris = [[me.vertices[i].co.copy() for i in t.vertices] for t in me.loop_triangles]
    xs = [v.x for t in tris for v in t]
    ys = [v.y for t in tris for v in t]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    def inside(px, py):
        for a, b, c in tris:
            d1 = (px - b.x) * (a.y - b.y) - (a.x - b.x) * (py - b.y)
            d2 = (px - c.x) * (b.y - c.y) - (b.x - c.x) * (py - c.y)
            d3 = (px - a.x) * (c.y - a.y) - (c.x - a.x) * (py - a.y)
            neg = d1 < 0 or d2 < 0 or d3 < 0
            pos = d1 > 0 or d2 > 0 or d3 > 0
            if not (neg and pos):
                return True
        return False

    W = res
    H = max(8, int(res * (y1 - y0) / (x1 - x0)))
    ink = [[inside(x0 + (i + 0.5) * (x1 - x0) / W, y0 + (j + 0.5) * (y1 - y0) / H) for i in range(W)]
           for j in range(H)]
    seen = [[False] * W for _ in range(H)]
    best = []
    for j in range(H):
        for i in range(W):
            if ink[j][i] or seen[j][i]:
                continue
            stack, region, open_edge = [(i, j)], [], False
            seen[j][i] = True
            while stack:
                ci, cj = stack.pop()
                region.append((ci, cj))
                if ci in (0, W - 1) or cj in (0, H - 1):
                    open_edge = True
                for ni, nj in ((ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1)):
                    if 0 <= ni < W and 0 <= nj < H and not seen[nj][ni] and not ink[nj][ni]:
                        seen[nj][ni] = True
                        stack.append((ni, nj))
            if not open_edge and len(region) > len(best):
                best = region
    bpy.data.meshes.remove(me)
    cx = x0 + (sum(p[0] for p in best) / len(best) + 0.5) * (x1 - x0) / W
    cy = y0 + (sum(p[1] for p in best) / len(best) + 0.5) * (y1 - y0) / H
    return glyph.matrix_world @ Vector((cx, cy, 0))


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

    def build(self):
        ink = block_mat()
        # word on the XZ plane facing the camera, tilted on the left diagonal
        self.word = word = empty("word", parent=self.root)
        word.rotation_euler = (0, -TILT, 0)
        root, glyphs, w = word_letters("Tesla", "Tesla", size=10, fnt=FONTS[FONT], extrude=DEPTH,
                                       mat=ink, tracking=0.0, parent=word, loc=(0, 0, -3.5))
        for ch, piv, g in glyphs:
            self.show(g, snap(TES) if ch in "Tes" else snap(LA))
        bpy.context.view_layer.update()
        a = [g for ch, piv, g in glyphs if ch == "a"][0]
        self.hole = counter_center(a) - self.root.matrix_world.translation
        self.width = w
        print(f"[tesla] counter of 'a' at {tuple(round(v, 2) for v in self.hole)}")

    def cam(self, t):
        h = self.hole
        # frame the whole word big, hung off the left margin, filling ~85% of the width;
        # camera sits off-axis (right and above) so the block depth shows
        vis = self.width * math.cos(TILT)
        fw = vis / 0.85
        cx = -vis / 2 + fw * 0.44
        d = fw * 50 / 36
        far = V(cx + 0.22 * d, -d, 0.16 * d)
        # hold, then a hard push that lands exactly on the 16th grid
        a = in_expo(prog(t, DIVE, THROUGH))
        mouth = V(h.x, -DEPTH - 0.4, h.z)            # entrance of the tunnel through the 'a'
        p = lerp(far, mouth, a)
        if t > THROUGH:
            p = mouth + V(0, 40 * (t - THROUGH), 0)
        tgt = lerp(V(cx, 0, 0.0), V(h.x, 0, h.z), in_out_cubic(prog(t, DIVE, THROUGH - 0.25)))
        if t > THROUGH:
            tgt = V(h.x, 50, h.z)
        roll = TILT * in_out_cubic(prog(t, DIVE, THROUGH))
        lens = lerp(50.0, 28.0, in_expo(prog(t, DIVE + 0.5, THROUGH)))
        return p, tgt, lens, roll


SHOTS = [Tesla]
