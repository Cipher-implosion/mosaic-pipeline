#!/usr/bin/env python3
"""Generate a synthetic test clip for the mosaic pipeline.

A textured, ball-like ellipse moves across a textured background, deforms over
time (aspect ratio and rotation change), leaves the frame through the right
edge, stays out of view for a while and then re-enters. A sine-tone audio track
is muxed in so audio copy can be verified.

Outputs:
  OUT.mp4            the clip (H.264 + AAC)
  OUT.gt.npz         ground-truth target masks (packed bits) + metadata

Usage: python3 make_test_clip.py [OUT.mp4] [--w 640 --h 360 --fps 25 --seconds 4]
"""
import argparse
import json
import math
import subprocess
import sys

import cv2
import numpy as np


def background(w, h, rng):
    """Static, textured background (gradient + checker + noise) so that
    'unchanged background' and 'pixelated target' are measurable."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.zeros((h, w, 3), np.float32)
    bg[..., 0] = 90 + 60 * (xx / w)            # B
    bg[..., 1] = 110 + 50 * (yy / h)           # G
    bg[..., 2] = 70 + 30 * np.sin(xx / 37.0)   # R
    checker = (((xx // 20) + (yy // 20)) % 2) * 25.0
    bg += checker[..., None]
    bg += rng.normal(0, 6, size=bg.shape).astype(np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


def target_params(i, n, w, h):
    """Centre, axes and angle of the target for frame i. Path: enter at left,
    move right, exit right edge (~35% of clip), stay out, re-enter from right
    (~60%), move left to ~30% of width."""
    t = i / max(n - 1, 1)
    base = 0.13 * h
    # shape change: aspect ratio oscillates, slow rotation
    ax = base * (1.0 + 0.35 * math.sin(2 * math.pi * t * 1.5))
    ay = base * (1.0 - 0.25 * math.sin(2 * math.pi * t * 1.5))
    ang = 90.0 * t
    cy = h * (0.5 + 0.15 * math.sin(2 * math.pi * t))
    if t < 0.40:                       # 0.15w -> off right edge
        cx = w * (0.15 + (1.30 - 0.15) * (t / 0.40))
    elif t < 0.60:                     # out of frame
        cx = w * 1.30
    else:                              # re-enter from right, go left
        cx = w * (1.30 - (1.30 - 0.30) * ((t - 0.60) / 0.40))
    return cx, cy, ax, ay, ang


def draw_target(frame, mask, cx, cy, ax, ay, ang):
    c = (int(round(cx)), int(round(cy)))
    axes = (max(int(round(ax)), 1), max(int(round(ay)), 1))
    cv2.ellipse(mask, c, axes, ang, 0, 360, 255, -1)
    m = mask > 0
    if not m.any():
        return
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # orange ball with dark seams and shading -> high internal detail
    r = np.sqrt(((xx - cx) / axes[0]) ** 2 + ((yy - cy) / axes[1]) ** 2)
    shade = np.clip(1.15 - 0.55 * r, 0.4, 1.15)
    col = np.zeros_like(frame, np.float32)
    col[..., 0] = 30 * shade
    col[..., 1] = 120 * shade
    col[..., 2] = 245 * shade
    # seams: stripes rotating with the shape
    a = math.radians(ang)
    u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
    v = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
    seam = (np.abs(np.sin(u / 6.0)) < 0.25) | (np.abs(v) < 2.0)
    col[seam] *= 0.25
    frame[m] = np.clip(col[m], 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="test_clip.mp4")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    w, h, fps = args.w, args.h, args.fps
    n = int(round(args.seconds * fps))
    rng = np.random.default_rng(args.seed)
    bg = background(w, h, rng)

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={n / fps}",
           "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    masks, params = [], []
    for i in range(n):
        frame = bg.copy()
        mask = np.zeros((h, w), np.uint8)
        p = target_params(i, n, w, h)
        draw_target(frame, mask, *p)
        proc.stdin.write(frame.tobytes())
        masks.append(np.packbits(mask > 0))
        params.append([round(float(x), 2) for x in p])
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")

    areas = [int(np.unpackbits(m)[: w * h].sum()) for m in masks]
    gt_path = args.out.rsplit(".", 1)[0] + ".gt.npz"
    np.savez_compressed(gt_path, masks=np.stack(masks), shape=np.array([h, w]),
                        fps=fps, params=np.array(params), areas=np.array(areas))
    absent = [i for i, a in enumerate(areas) if a == 0]
    info = {"out": args.out, "gt": gt_path, "frames": n, "fps": fps, "size": [w, h],
            "absent_frames": [absent[0], absent[-1]] if absent else None,
            "frame0_center": [round(params[0][0]), round(params[0][1])]}
    print(json.dumps(info))


if __name__ == "__main__":
    main()
