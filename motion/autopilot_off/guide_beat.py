"""Synthesise a placeholder trap groove on the same grid the camera is cut to,
so the preview can be watched in time before the real track is dropped in.

    python3 guide_beat.py out.wav [--bpm 140] [--beats 100]
"""
import argparse
import array
import math
import random
import wave

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--bpm", type=float, default=140.0)
ap.add_argument("--beats", type=float, default=100.0)
args = ap.parse_args()

SR = 44100
spb = 60.0 / args.bpm
n = int((args.beats * spb + 1.5) * SR)
buf = [0.0] * n
rng = random.Random(1)


def add(t0, fn, dur):
    i0 = int(t0 * SR)
    for i in range(int(dur * SR)):
        if i0 + i >= n:
            break
        buf[i0 + i] += fn(i / SR)


def kick808(t):
    f = 45 + 110 * math.exp(-t * 28)
    return 0.9 * math.sin(2 * math.pi * f * t) * math.exp(-t * 3.2)


def snare(t):
    return (0.45 * (rng.random() * 2 - 1) * math.exp(-t * 18)
            + 0.3 * math.sin(2 * math.pi * 190 * t) * math.exp(-t * 25))


def hat(t):
    return 0.12 * (rng.random() * 2 - 1) * math.exp(-t * 90)


b = 0.0
while b < args.beats:
    for k in (0, 1.5, 2.5):
        add((b + k) * spb, kick808, 0.9)
    for s in (1, 3):
        add((b + s) * spb, snare, 0.3)
    for h in range(16):
        if h % 4 != 3 or rng.random() < 0.5:
            add((b + h * 0.25) * spb, hat, 0.06)
    if int(b / 4) % 2 == 1:   # hat roll every other bar
        for r in range(6):
            add((b + 3.5 + r / 12) * spb, hat, 0.04)
    b += 4

peak = max(abs(v) for v in buf) or 1.0
pcm = array.array("h", (int(32000 * math.tanh(1.2 * v / peak)) for v in buf))
with wave.open(args.out, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", args.out)
