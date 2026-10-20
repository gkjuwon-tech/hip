"""Shared helpers: timing, easing, materials, text, keyframe baking."""
import math
import os
import random

import bpy
from mathutils import Vector, Euler, Matrix, Quaternion

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

# ----------------------------------------------------------------------------
# Timing. Everything is authored in beats; swap BPM / offset once the real
# audio is in and the whole piece retimes itself.
# ----------------------------------------------------------------------------
FPS = 24
BPM = 140.0
OFFSET_SEC = 0.0


def F(beat):
    """Beat -> frame (1-based)."""
    return 1 + int(round((OFFSET_SEC + beat * 60.0 / BPM) * FPS))


def B(frame):
    """Frame -> beat (float)."""
    return ((frame - 1) / FPS - OFFSET_SEC) * BPM / 60.0


# ----------------------------------------------------------------------------
# Easing
# ----------------------------------------------------------------------------
def clamp01(t):
    return 0.0 if t < 0 else 1.0 if t > 1 else t


def lerp(a, b, t):
    if isinstance(a, (tuple, list, Vector)):
        return Vector(a).lerp(Vector(b), t)
    return a + (b - a) * t


def prog(x, a, b):
    """Normalised progress of x through [a, b], clamped."""
    if b == a:
        return 1.0 if x >= b else 0.0
    return clamp01((x - a) / (b - a))


def out_expo(t):
    t = clamp01(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def in_expo(t):
    t = clamp01(t)
    return 0.0 if t <= 0 else 2 ** (10 * t - 10)


def in_out_cubic(t):
    t = clamp01(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def out_cubic(t):
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def in_cubic(t):
    return clamp01(t) ** 3


def in_out_expo(t):
    t = clamp01(t)
    if t in (0.0, 1.0):
        return t
    return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2


def out_back(t, s=1.70158):
    t = clamp01(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def out_elastic(t):
    t = clamp01(t)
    if t in (0.0, 1.0):
        return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi) / 3) + 1


def pulse(x, at, attack=0.0, decay=0.35):
    """1 at beat `at`, decaying to 0 over `decay` beats (x in beats)."""
    if x < at - attack:
        return 0.0
    if x < at:
        return (x - (at - attack)) / attack
    d = (x - at) / decay
    return math.exp(-4.5 * d) if d < 3 else 0.0


# ----------------------------------------------------------------------------
# Scene / collection bookkeeping
# ----------------------------------------------------------------------------
def new_collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


_ACTIVE = {"col": None}


def set_active_collection(col):
    _ACTIVE["col"] = col


def link(obj):
    col = _ACTIVE["col"] or bpy.context.scene.collection
    col.objects.link(obj)
    return obj


def empty(name, loc=(0, 0, 0), parent=None):
    o = bpy.data.objects.new(name, None)
    o.location = loc
    o.parent = parent
    return link(o)


def mesh_obj(name, mesh, mat=None, loc=(0, 0, 0), parent=None):
    o = bpy.data.objects.new(name, mesh)
    o.location = loc
    o.parent = parent
    if mat is not None:
        o.data.materials.append(mat)
    return link(o)


def visible_between(col, f_on, f_off):
    """Render-hide every object of a collection outside [f_on, f_off],
    preserving any finer show/hide keys the shot already set."""
    for o in list(col.objects):
        fc = next((c for c in fcurves(o) if c.data_path == "hide_render"), None)
        if fc is None or min(k.co.x for k in fc.keyframe_points) > f_on:
            for f, v in (((1, True), (f_on, False)) if f_on > 1 else ((1, False),)):
                o.hide_render = v
                o.keyframe_insert("hide_render", frame=f)
        o.hide_render = True
        o.keyframe_insert("hide_render", frame=f_off + 1)


# ----------------------------------------------------------------------------
# Primitive meshes built without bpy.ops (works fully in background mode).
# ----------------------------------------------------------------------------
import bmesh  # noqa: E402


def _bm_to_mesh(bm, name, smooth=False):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    return me


def m_cube(name, size=(1, 1, 1), bevel=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= size[0]
        v.co.y *= size[1]
        v.co.z *= size[2]
    if bevel > 0:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=3, affect="EDGES")
    return _bm_to_mesh(bm, name, smooth=bevel > 0)


def m_sphere(name, r=1.0, subdiv=3):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=r)
    return _bm_to_mesh(bm, name, smooth=True)


def m_uvsphere(name, r=1.0, seg=32, rings=16):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=r)
    return _bm_to_mesh(bm, name, smooth=True)


def m_cyl(name, r=1.0, depth=1.0, seg=32, r2=None):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r,
                          radius2=r if r2 is None else r2, depth=depth)
    me = _bm_to_mesh(bm, name)
    for p in me.polygons:
        p.use_smooth = len(p.vertices) == 4
    return me


def m_torus(name, R=1.0, r=0.25, seg=24, sides=12):
    bm = bmesh.new()
    verts = []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        row = []
        for j in range(sides):
            b = 2 * math.pi * j / sides
            x = (R + r * math.cos(b)) * math.cos(a)
            y = (R + r * math.cos(b)) * math.sin(a)
            z = r * math.sin(b)
            row.append(bm.verts.new((x, y, z)))
        verts.append(row)
    for i in range(seg):
        for j in range(sides):
            bm.faces.new((verts[i][j], verts[(i + 1) % seg][j],
                          verts[(i + 1) % seg][(j + 1) % sides], verts[i][(j + 1) % sides]))
    return _bm_to_mesh(bm, name, smooth=True)


def m_plane(name, sx=1.0, sy=1.0):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    for v in bm.verts:
        v.co.x *= sx * 2
        v.co.y *= sy * 2
    return _bm_to_mesh(bm, name)


def m_shard(name, s=1.0, rng=random):
    bm = bmesh.new()
    pts = [Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))) * s for _ in range(6)]
    bmesh.ops.convex_hull(bm, input=[bm.verts.new(p) for p in pts])
    return _bm_to_mesh(bm, name)


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------
def _new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return m, nt, out


def mat_principled(name, color=(0.02, 0.02, 0.02), metallic=0.0, rough=0.3,
                   emit=None, emit_strength=0.0, transmission=0.0, ior=1.45,
                   coat=0.0, alpha=1.0):
    m, nt, out = _new_mat(name)
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = rough
    p.inputs["IOR"].default_value = ior
    p.inputs["Transmission Weight"].default_value = transmission
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Alpha"].default_value = alpha
    if emit:
        p.inputs["Emission Color"].default_value = (*emit, 1)
        p.inputs["Emission Strength"].default_value = emit_strength
    nt.links.new(p.outputs[0], out.inputs[0])
    if transmission > 0:
        m.surface_render_method = "DITHERED"
        m.use_raytrace_refraction = True
    return m


def mat_emit(name, color=(1, 1, 1), strength=5.0):
    m, nt, out = _new_mat(name)
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs[0].default_value = (*color, 1)
    e.inputs[1].default_value = strength
    nt.links.new(e.outputs[0], out.inputs[0])
    return m


def mat_rim(name, base=(0.035, 0.035, 0.04), rim=(1, 1, 1), rim_strength=6.0,
            metallic=0.8, rough=0.25, blend=0.18, face_emit=0.14):
    """Glossy black body whose silhouette edges glow (the house look)."""
    m, nt, out = _new_mat(name)
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = (*base, 1)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = rough
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = blend
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.55
    ramp.color_ramp.elements[1].position = 0.95
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (*rim, 1)
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY_ADD"
    mul.inputs[1].default_value = rim_strength
    mul.inputs[2].default_value = face_emit
    mul.name = "RimStrength"
    nt.links.new(lw.outputs["Facing"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], em.inputs[1])
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(p.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs[0])
    return m


def mat_dissolve(name, color=(1, 1, 1), strength=6.0, scale=3.0):
    """Emission text masked by animated noise; key node 'Thresh' value 0..1 (1 = solid)."""
    m, nt, out = _new_mat(name)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = 6
    nz.name = "Noise"
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    thr = nt.nodes.new("ShaderNodeValue")
    thr.name = "Thresh"
    thr.outputs[0].default_value = 1.0
    # mask = smoothstep(noise - (1 - thresh))
    sub = nt.nodes.new("ShaderNodeMath")
    sub.operation = "SUBTRACT"
    nt.links.new(nz.outputs["Fac"], sub.inputs[0])
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(thr.outputs[0], inv.inputs[1])
    nt.links.new(inv.outputs[0], sub.inputs[1])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -0.08
    mr.inputs["From Max"].default_value = 0.08
    nt.links.new(sub.outputs[0], mr.inputs[0])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (*color, 1)
    em.inputs[1].default_value = strength
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mr.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    m.surface_render_method = "DITHERED"
    return m


def key_node_value(mat, node_name, frame, value, input_idx=None):
    n = mat.node_tree.nodes[node_name]
    sock = n.outputs[0] if input_idx is None else n.inputs[input_idx]
    sock.default_value = value
    sock.keyframe_insert("default_value", frame=frame)


def bake_node_value(mat, node_name, f0, f1, fn, input_idx=None):
    for f in range(f0, f1 + 1):
        key_node_value(mat, node_name, f, fn(f), input_idx)


# ----------------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------------
_FONTS = {}


def font(name):
    if name not in _FONTS:
        _FONTS[name] = bpy.data.fonts.load(os.path.join(FONT_DIR, name))
    return _FONTS[name]


HEAVY = "ArchivoBlack-Regular.ttf"
WIDE = "RubikMonoOne-Regular.ttf"
MONO = "SpaceMono-Bold.ttf"
MONO_R = "SpaceMono-Regular.ttf"


def text(name, body, size=1.0, fnt=HEAVY, extrude=0.1, bevel=0.0, mat=None,
         align_x="CENTER", align_y="BOTTOM_BASELINE", loc=(0, 0, 0), stand=True,
         parent=None, spacing=1.0, res=3):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.font = font(fnt)
    cu.size = size
    cu.extrude = extrude
    cu.bevel_depth = bevel
    cu.bevel_resolution = res
    cu.resolution_u = 4
    cu.align_x = align_x
    cu.align_y = align_y
    cu.space_character = spacing
    o = bpy.data.objects.new(name, cu)
    o.location = loc
    if stand:
        o.rotation_euler = (math.pi / 2, 0, 0)
    o.parent = parent
    if mat is not None:
        cu.materials.append(mat)
    return link(o)


def word_letters(name, body, size=1.0, fnt=HEAVY, extrude=0.1, bevel=0.0, mat=None,
                 tracking=0.06, parent=None, loc=(0, 0, 0)):
    """One object per glyph, laid out centred; each glyph is pivoted at its
    own bottom-centre so it can pop / topple individually.
    Returns (root_empty, [(char, pivot_empty, glyph_obj)], total_width)."""
    root = empty(name, loc=loc, parent=parent)
    items = []
    for i, ch in enumerate(body):
        if ch == " ":
            items.append((ch, None, None, size * 0.33))
            continue
        piv = empty(f"{name}.{i}.piv", parent=root)
        g = text(f"{name}.{i}", ch, size=size, fnt=fnt, extrude=extrude, bevel=bevel,
                 mat=mat, parent=piv)
        items.append((ch, piv, g, None))
    bpy.context.view_layer.update()
    widths = []
    for ch, piv, g, w in items:
        widths.append(w if w is not None else g.dimensions.x)
    total = sum(widths) + tracking * size * (len(widths) - 1)
    x = -total / 2
    out = []
    for (ch, piv, g, _), w in zip(items, widths):
        if piv is not None:
            piv.location = (x + w / 2, 0, 0)
            out.append((ch, piv, g))
        x += w + tracking * size
    return root, out, total


def to_mesh_obj(src, name, voxel=None, mat=None, keep_src=False, apply_transform=True):
    """Bake an object's evaluated geometry (optionally voxel-remeshed) into a mesh object."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(src.evaluated_get(dg))
    o = bpy.data.objects.new(name, me)
    if apply_transform:
        me.transform(src.matrix_world)
    else:
        o.matrix_world = src.matrix_world.copy()
    link(o)
    if voxel:
        r = o.modifiers.new("remesh", "REMESH")
        r.mode = "VOXEL"
        r.voxel_size = voxel
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        me2 = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
        o.modifiers.remove(r)
        o.data = me2
        for p in me2.polygons:
            p.use_smooth = True
    if mat is not None:
        o.data.materials.clear()
        o.data.materials.append(mat)
    if not keep_src:
        bpy.data.objects.remove(src)
    return o


# ----------------------------------------------------------------------------
# Keyframes
# ----------------------------------------------------------------------------
def fcurves(idb):
    """F-curves of this ID's own slot (actions may be shared between object and data)."""
    ad = getattr(idb, "animation_data", None)
    if not ad or not ad.action:
        return []
    from bpy_extras import anim_utils
    cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    return list(cb.fcurves) if cb else []


def key(obj, path, frame, value=None, interp=None):
    if value is not None:
        cur = getattr(obj, path)
        if hasattr(cur, "__len__") and not isinstance(value, (int, float)):
            setattr(obj, path, value)
        elif hasattr(cur, "__len__"):
            setattr(obj, path, (value,) * len(cur))
        else:
            setattr(obj, path, value)
    obj.keyframe_insert(path, frame=frame)
    if interp:
        for fc in fcurves(obj):
            if fc.data_path == path:
                for kp in fc.keyframe_points:
                    if int(round(kp.co.x)) == frame:
                        kp.interpolation = interp


def bake(obj, path, f0, f1, fn):
    """Key `path` every frame in [f0, f1] with fn(frame)."""
    for f in range(f0, f1 + 1):
        key(obj, path, f, fn(f))


def linearize(id_block):
    if True:
        for fc in fcurves(id_block):
            for kp in fc.keyframe_points:
                if kp.interpolation == "BEZIER":
                    kp.interpolation = "LINEAR"


def look_quat(pos, target, roll=0.0):
    d = Vector(target) - Vector(pos)
    q = d.to_track_quat("-Z", "Y")
    if roll:
        q = q @ Quaternion((0, 0, 1), roll)
    return q


def spline(keys, t):
    """Smooth (Catmull-Rom / Hermite) interpolation through [(time, value), ...].
    Values may be floats or Vectors; ends ease in/out; clamps outside the range."""
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    i = max(j for j in range(len(keys) - 1) if keys[j][0] <= t)
    t0, p0 = keys[i]
    t1, p1 = keys[i + 1]

    def tangent(k):
        if k == 0 or k == len(keys) - 1:
            return p0 * 0
        (ta, pa), (tb, pb) = keys[k - 1], keys[k + 1]
        return (pb - pa) / (tb - ta)
    m0, m1 = tangent(i), tangent(i + 1)
    h = t1 - t0
    s = (t - t0) / h
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return p0 * h00 + m0 * (h10 * h) + p1 * h01 + m1 * (h11 * h)
