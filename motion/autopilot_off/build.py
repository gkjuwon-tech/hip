"""AUTOPILOT OFF — build line 1 in Blender and (optionally) render it.

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
    if s.on_twos:  # hold each camera pose for two frames
        bn = B(shots.twos(fr))
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

# -----------------------------------------------------------------------------
# World: flat paper white. No fog, no flashes: the type does the work.
# -----------------------------------------------------------------------------
world = bpy.data.worlds.new("paper")
sc.world = world
world.use_nodes = True
wn = world.node_tree
bg = wn.nodes["Background"]
bg.inputs[0].default_value = (*built[0].tint, 1)
bg.inputs[1].default_value = built[0].world_strength
# the camera always sees pure paper white; only the lighting uses the dimmer value
seen = wn.nodes.new("ShaderNodeBackground")
seen.inputs[0].default_value = (*built[0].tint, 1)
lp = wn.nodes.new("ShaderNodeLightPath")
mx = wn.nodes.new("ShaderNodeMixShader")
wn.links.new(lp.outputs["Is Camera Ray"], mx.inputs[0])
wn.links.new(bg.outputs[0], mx.inputs[1])
wn.links.new(seen.outputs[0], mx.inputs[2])
wn.links.new(mx.outputs[0], wn.nodes["World Output"].inputs["Surface"])

# -----------------------------------------------------------------------------
# Render settings
# -----------------------------------------------------------------------------
rx, ry = (int(v) for v in args.res.split("x"))
r = sc.render
r.engine = "BLENDER_EEVEE_NEXT"
r.resolution_x, r.resolution_y = rx, ry
r.resolution_percentage = 100
r.use_motion_blur = built[0].motion_blur
r.motion_blur_shutter = 0.45
r.motion_blur_position = "START"
r.film_transparent = False
ee = sc.eevee
ee.taa_render_samples = args.samples
ee.use_raytracing = False
ee.ray_tracing_method = "SCREEN"
ee.ray_tracing_options.resolution_scale = "2"
ee.use_shadows = True
ee.shadow_resolution_scale = 1.0
ee.use_fast_gi = False
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"

# No compositing: what EEVEE draws is what you get.
sc.use_nodes = False

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
