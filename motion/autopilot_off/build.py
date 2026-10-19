"""AUTOPILOT OFF — build the whole piece in Blender and (optionally) render it.

    blender -b --factory-startup -P build.py -- [--render] [--res 1280x720]
        [--samples 16] [--frames 1-1029] [--out renders/frame_] [--save out.blend]
        [--bpm 140] [--offset 0.0]

Everything is authored in beats (see lib.F). Once the real track is in,
pass --bpm/--offset (or edit lib.BPM) and the whole cut retimes.
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector, Euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--render", action="store_true")
ap.add_argument("--res", default="1280x720")
ap.add_argument("--samples", type=int, default=16)
ap.add_argument("--frames", default=None)
ap.add_argument("--out", default=os.path.join(lib.HERE, "renders", "frame_"))
ap.add_argument("--save", default=None)
ap.add_argument("--bpm", type=float, default=None)
ap.add_argument("--offset", type=float, default=None)
ap.add_argument("--still", default=None, help="comma list of beats to render as stills")
args = ap.parse_args(argv)
if args.bpm:
    lib.BPM = args.bpm
if args.offset is not None:
    lib.OFFSET_SEC = args.offset

from lib import F, B, lerp, pulse, look_quat, fcurves, key, new_collection, set_active_collection  # noqa: E402
import shots  # noqa: E402
from shots import FX, SHOTS  # noqa: E402

# -----------------------------------------------------------------------------
# Fresh scene
# -----------------------------------------------------------------------------
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sc = bpy.context.scene
sc.name = "AUTOPILOT_OFF"

# -----------------------------------------------------------------------------
# Build shots, each in its own patch of space
# -----------------------------------------------------------------------------
built = []
beat = 0.0
for i, cls in enumerate(SHOTS):
    s = cls(beat, (i * 4000.0, 0, 0))
    s.build()
    built.append(s)
    beat = s.b1
END_BEAT = beat
F_END = F(END_BEAT)
sc.frame_start = 1
sc.frame_end = F_END
sc.render.fps = lib.FPS

# -----------------------------------------------------------------------------
# Global groove: generic trap grid until we have the real stems.
# Kick on 1 / the "and" of 2 / 3, snare on 2 & 4 (half-time feel at 140).
# -----------------------------------------------------------------------------
b = 0.0
while b < END_BEAT:
    for k in (0, 1.5, 2.5):
        FX["pump"].append((b + k, 0.35, 0.3))
    for sn in (1, 3):
        FX["shake"].append((b + sn, 0.18, 0.25))
    b += 4


def fx(kind, beat_now):
    return sum(a * pulse(beat_now, t, 0.0, d) for t, a, d in FX[kind])


def shot_at(beat_now):
    for s in built:
        if s.b0 <= beat_now < s.b1:
            return s
    return built[-1]


# -----------------------------------------------------------------------------
# Camera: baked every frame from the shots' cam(t) + shake + pump.
# -----------------------------------------------------------------------------
set_active_collection(None)
cam_d = bpy.data.cameras.new("CAM")
cam_d.sensor_width = 36
cam_d.clip_start = 0.05
cam_d.clip_end = 900
cam = bpy.data.objects.new("CAM", cam_d)
sc.collection.objects.link(cam)
sc.camera = cam

import random  # noqa: E402
rng = random.Random(7)
shake_dirs = {}


def shake_vec(beat_now):
    v = Vector((0, 0, 0))
    for t, a, d in FX["shake"]:
        p = pulse(beat_now, t, 0.0, d)
        if p <= 0:
            continue
        if t not in shake_dirs:
            shake_dirs[t] = [Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.3, 0.3)))
                             for _ in range(64)]
        dirs = shake_dirs[t]
        # jitter at ~2 frames per new offset, amplitude decaying
        idx = int((beat_now - t) * 60 / lib.BPM * lib.FPS / 1.5) % 64
        v += dirs[idx] * a * p * 0.35
    return v


prev_eul = None
prev_shot = None
cam_pos = {}
for fr in range(1, F_END + 1):
    bn = B(fr)
    s = shot_at(bn)
    pos, tgt, lens, roll = s.cam(bn - s.b0)
    pos = Vector(pos) + s.origin
    tgt = Vector(tgt) + s.origin
    q = look_quat(pos, tgt, roll)
    sv = shake_vec(bn)
    pos = pos + q @ Vector((sv.x, sv.y, 0))
    q = q @ Euler((0, 0, sv.z * 0.04)).to_quaternion()
    e = q.to_euler("XYZ", prev_eul) if (prev_eul is not None and s is prev_shot) else q.to_euler("XYZ")
    lens = lens * (1 - 0.045 * min(2.5, fx("pump", bn)))
    cam.location = pos
    cam.rotation_euler = e
    cam_d.lens = lens
    cam.keyframe_insert("location", frame=fr)
    cam.keyframe_insert("rotation_euler", frame=fr)
    cam_d.keyframe_insert("lens", frame=fr)
    cam_pos[fr] = (pos.copy(), s)
    prev_eul, prev_shot = e, s

for idb in (cam, cam_d):
    for fc in fcurves(idb):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
# no motion-blur smear across a cut: hold the last frame of every shot
for s in built[1:]:
    cut = F(s.b0)
    for idb in (cam, cam_d):
        for fc in fcurves(idb):
            for kp in fc.keyframe_points:
                if int(round(kp.co.x)) == cut - 1:
                    kp.interpolation = "CONSTANT"

# depth of field only for the rack-focus shot
see = [s for s in built if s.name == "see"][0]
cam_d.dof.use_dof = False
cam_d.dof.aperture_fstop = 1.2
cam_d.keyframe_insert("dof.use_dof", frame=1)
cam_d.dof.use_dof = True
cam_d.keyframe_insert("dof.use_dof", frame=F(see.b0))
for fr in range(F(see.b0), F(see.b1) + 1):
    cam_d.dof.focus_distance = see.focus(B(fr) - see.b0)
    cam_d.keyframe_insert("dof.focus_distance", frame=fr)
cam_d.dof.use_dof = False
cam_d.keyframe_insert("dof.use_dof", frame=F(see.b1))
for fc in fcurves(cam_d):
    if fc.data_path == "dof.use_dof":
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"

# -----------------------------------------------------------------------------
# World: near-black with volumetric haze, tinted per shot, flashes.
# -----------------------------------------------------------------------------
world = bpy.data.worlds.new("night")
sc.world = world
world.use_nodes = True
wn = world.node_tree
bg = wn.nodes["Background"]
bg.inputs[1].default_value = 1.0
vol = wn.nodes.new("ShaderNodeVolumeScatter")
vol.inputs["Anisotropy"].default_value = 0.6
wn.links.new(vol.outputs[0], wn.nodes["World Output"].inputs["Volume"])
RED = Vector((0.5, 0.0, 0.01))
for fr in range(1, F_END + 1):
    bn = B(fr)
    s = shot_at(bn)
    tint = Vector(s.tint)
    c = tint + RED * min(1.0, fx("tint", bn))
    bg.inputs[0].default_value = (*c, 1)
    bg.inputs[0].keyframe_insert("default_value", frame=fr)
    vol.inputs["Density"].default_value = s.fog
    vol.inputs["Density"].keyframe_insert("default_value", frame=fr)
    sc.view_settings.exposure = 2.2 * min(3.0, fx("flash", bn)) - 4.0 * lib.prog(bn, END_BEAT - 0.75, END_BEAT)
    sc.view_settings.keyframe_insert("exposure", frame=fr)

# -----------------------------------------------------------------------------
# HUD — parented to the camera, scale tracks lens so it stays screen-locked.
# -----------------------------------------------------------------------------
hud_col = new_collection("HUD")
set_active_collection(hud_col)
hud = lib.empty("HUD", loc=(0, 0, -1), parent=cam)
hud_m = lib.mat_emit("hud", (1, 1, 1), 1.6)
hud_red = lib.mat_emit("hud.red", (1, 0.05, 0.08), 3.0)
AR = 9 / 16


def hud_text(name, body, u, v, size, ax="LEFT", ay="CENTER", mat=hud_m, fnt=lib.MONO):
    return lib.text(name, body, size=size, fnt=fnt, extrude=0, mat=mat, align_x=ax, align_y=ay,
                    stand=False, parent=hud, loc=(u, v * AR, 0))


h_auto = hud_text("hud.auto", "AUTOPILOT  ON", -0.92, 0.86, 0.034)
h_batt = hud_text("hud.batt", "BATT 012%", 0.92, 0.86, 0.034, ax="RIGHT")
h_spd = hud_text("hud.spd", "000 KM/H", -0.92, -0.86, 0.034)
h_tc = hud_text("hud.tc", "00:00:00", 0.92, -0.86, 0.034, ax="RIGHT", fnt=lib.MONO_R)
h_lyr = hud_text("hud.lyric", "", 0.0, -0.70, 0.042, ax="CENTER")
# corner brackets
brk = lib.m_cube("brk", (1, 1, 1))
for sx in (-1, 1):
    for sy in (-1, 1):
        for horiz in (True, False):
            o = lib.mesh_obj("hud.brk", brk, hud_m, parent=hud)
            o.scale = (0.06, 0.003, 0.001) if horiz else (0.003, 0.06 * 1.0, 0.001)
            o.location = (sx * (0.955 - (0.03 if horiz else 0)), sy * (0.53 - (0 if horiz else 0.03)), 0)
lens_fc = [fc for fc in fcurves(cam_d) if fc.data_path == "lens"][0]
for fr in range(1, F_END + 1):
    hw = (cam_d.sensor_width / 2) / lens_fc.evaluate(fr)
    hud.scale = (hw, hw, hw)
    hud.keyframe_insert("scale", frame=fr)
for fc in fcurves(hud):
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

# per-frame HUD text, precomputed then applied by a frame handler
lyrics = sorted(FX["lyric"])
batts = sorted(FX["batt"])
HUD_FRAMES = {}
for fr in range(1, F_END + 1):
    bn = B(fr)
    ly = ""
    for t, line in lyrics:
        if bn >= t:
            ly = line
    bv = 12
    for t, v in batts:
        if bn >= t:
            bv = v
    p, s = cam_pos[fr]
    pp, ps = cam_pos.get(fr - 1, (p, s))
    spd = (p - pp).length * lib.FPS * 3.6 if ps is s else None
    HUD_FRAMES[fr] = (ly, bv, spd, bn)

_last_spd = [0.0]


def hud_update(scene, depsgraph=None):
    fr = scene.frame_current
    if fr not in HUD_FRAMES:
        return
    ly, bv, spd, bn = HUD_FRAMES[fr]
    h_lyr.data.body = ly
    h_batt.data.body = "BATT ERROR" if bv < 0 else f"BATT {bv:03d}%"
    h_batt.data.materials[0] = hud_red if (bv < 0 or bv > 100) else hud_m
    if spd is not None:
        _last_spd[0] = 0.7 * _last_spd[0] + 0.3 * min(spd, 999)
    h_spd.data.body = f"{int(_last_spd[0]):03d} KM/H"
    sec = (fr - 1) / lib.FPS
    h_tc.data.body = f"{int(sec // 60):02d}:{int(sec % 60):02d}:{(fr - 1) % lib.FPS:02d}"
    off = bn >= END_BEAT - 2.5
    h_auto.data.body = "AUTOPILOT  OFF" if off else "AUTOPILOT  ON"
    h_auto.data.materials[0] = hud_red if off else hud_m


bpy.app.handlers.frame_change_pre.append(hud_update)

# -----------------------------------------------------------------------------
# Per-shot visibility (keeps EEVEE from paying for 14 shots it can't see)
# -----------------------------------------------------------------------------
for s in built:
    s.finish()

# -----------------------------------------------------------------------------
# Render settings
# -----------------------------------------------------------------------------
rx, ry = (int(v) for v in args.res.split("x"))
r = sc.render
r.engine = "BLENDER_EEVEE_NEXT"
r.resolution_x, r.resolution_y = rx, ry
r.resolution_percentage = 100
r.use_motion_blur = True
r.motion_blur_shutter = 0.45
r.motion_blur_position = "START"
r.film_transparent = False
ee = sc.eevee
ee.taa_render_samples = args.samples
ee.use_raytracing = True
ee.ray_tracing_method = "SCREEN"
ee.ray_tracing_options.resolution_scale = "2"
ee.use_shadows = True
ee.volumetric_tile_size = "8"
ee.volumetric_samples = 48
ee.volumetric_start = 0.5
ee.volumetric_end = 160
ee.use_volumetric_shadows = False
ee.use_fast_gi = False
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = "AgX - Medium High Contrast"

# Compositor: bloom, animated chromatic aberration, vignette, grain-free.
sc.use_nodes = True
nt = sc.node_tree
nt.nodes.clear()
rl = nt.nodes.new("CompositorNodeRLayers")
glare = nt.nodes.new("CompositorNodeGlare")
glare.glare_type = "BLOOM"
glare.quality = "HIGH"
glare.inputs["Threshold"].default_value = 0.9
glare.inputs["Strength"].default_value = 0.6
glare.inputs["Size"].default_value = 0.7
lens_n = nt.nodes.new("CompositorNodeLensdist")
lens_n.inputs["Distortion"].default_value = -0.012
ell = nt.nodes.new("CompositorNodeEllipseMask")
ell.width, ell.height = 1.25, 1.25
blur = nt.nodes.new("CompositorNodeBlur")
blur.filter_type = "GAUSS"
blur.size_x = blur.size_y = 220
mix = nt.nodes.new("CompositorNodeMixRGB")
mix.blend_type = "MULTIPLY"
mix.inputs[0].default_value = 0.65
comp = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], glare.inputs["Image"])
nt.links.new(glare.outputs["Image"], lens_n.inputs["Image"])
nt.links.new(lens_n.outputs["Image"], mix.inputs[1])
nt.links.new(ell.outputs["Mask"], blur.inputs["Image"])
nt.links.new(blur.outputs["Image"], mix.inputs[2])
nt.links.new(mix.outputs["Image"], comp.inputs["Image"])
disp = lens_n.inputs["Dispersion"]
for fr in range(1, F_END + 1):
    disp.default_value = min(0.25, 0.012 + 0.07 * fx("ca", B(fr)))
    disp.keyframe_insert("default_value", frame=fr)

print(f"[autopilot] built {len(built)} shots, {F_END} frames @ {lib.FPS}fps, {lib.BPM} BPM")
for s in built:
    print(f"  {s.name:9s} beats {s.b0:5.1f}-{s.b1:5.1f}  frames {F(s.b0)}-{F(s.b1) - 1}")

if args.save:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save))

if args.still:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for bt in args.still.split(","):
        fr = F(float(bt))
        sc.frame_set(fr)
        r.filepath = f"{args.out}{fr:04d}.png"
        bpy.ops.render.render(write_still=True)
        print("[autopilot] still", fr, r.filepath)
elif args.render:
    if args.frames:
        a, bb = (int(v) for v in args.frames.split("-"))
        sc.frame_start, sc.frame_end = a, bb
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    r.filepath = args.out
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    bpy.ops.render.render(animation=True)
