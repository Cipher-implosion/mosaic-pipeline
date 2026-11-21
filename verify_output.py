#!/usr/bin/env python3
"""Verify a mosaic.py output against its input and the synthetic ground truth.

Checks
  * container: same resolution, fps, frame count, duration (+-1 frame); audio
    stream present and bit-identical to the input (packet MD5 of the copied stream)
  * per frame (GT = ground-truth mask from make_test_clip.py):
      - pixelation inside the target: mean per-block pixel std inside fully-covered
        grid blocks, input vs output (pixelated blocks are ~flat)
      - background unchanged: PSNR / mean abs diff outside GT dilated by
        (dilate + block + 4) px
      - coverage of GT by the pipeline mask (dilated like the renderer) and IoU
        of the raw pipeline mask vs GT; false-positive pixels when GT is empty
Writes a JSON report and a contact-sheet PNG.

Usage: python3 verify_output.py IN.mp4 OUT.mp4 GT.npz MASKS.npz [--block 12 --dilate 6]
       [--report report.json] [--sheet sheet.png] [--max-frames N]
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,"
         "duration,sample_rate,channels:format=duration", "-of", "json", path],
        check=True, capture_output=True, text=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    return {"width": v["width"], "height": v["height"], "fps": v["avg_frame_rate"],
            "frames": int(v["nb_read_frames"]), "v_duration": float(v.get("duration", 0)),
            "format_duration": float(j["format"]["duration"]),
            "audio": None if a is None else {"codec": a["codec_name"], "rate": a.get("sample_rate"),
                                             "channels": a.get("channels"),
                                             "duration": float(a.get("duration", 0))}}


def audio_md5(path, max_seconds=None):
    t = ["-t", str(max_seconds)] if max_seconds else []
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, *t, "-map", "0:a:0", "-c", "copy",
                        "-f", "md5", "-"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def read_frames(path, w, h, limit=None):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    n = 0
    while limit is None or n < limit:
        buf = p.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        n += 1
    p.stdout.close()
    p.wait()


def unpack(packed, h, w):
    return np.unpackbits(packed, count=h * w).reshape(h, w).astype(bool)


def block_std(img, sel_blocks, block):
    """Mean over selected grid blocks of per-block pixel std (all channels)."""
    h, w = img.shape[:2]
    nby, nbx = h // block, w // block
    g = img[:nby * block, :nbx * block].astype(np.float32)
    g = g.reshape(nby, block, nbx, block, 3).transpose(0, 2, 1, 3, 4).reshape(nby, nbx, -1, 3)
    stds = g.std(axis=2).mean(axis=-1)
    return float(stds[sel_blocks].mean())


def full_blocks(mask, block):
    h, w = mask.shape
    nby, nbx = h // block, w // block
    m = mask[:nby * block, :nbx * block].reshape(nby, block, nbx, block)
    return m.all(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("gt")
    ap.add_argument("masks")
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--dilate", type=int, default=6)
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--report", default=None)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--sheet-frames", default="0,20,31,50,75,90")
    args = ap.parse_args()

    pi, po = probe(args.inp), probe(args.out)
    gt = np.load(args.gt)
    pm = np.load(args.masks)
    h, w = (int(x) for x in gt["shape"])
    n_expected = len(pm["masks"])
    fps_num, fps_den = (int(x) for x in pi["fps"].split("/"))
    frame_dur = fps_den / fps_num
    exp_frames = pi["frames"] if not args.max_seconds else n_expected
    exp_dur = pi["format_duration"] if not args.max_seconds else args.max_seconds

    container = {
        "input": pi, "output": po,
        "same_resolution": (pi["width"], pi["height"]) == (po["width"], po["height"]),
        "same_fps": pi["fps"] == po["fps"],
        "same_frame_count": po["frames"] == exp_frames,
        "duration_diff_s": round(abs(po["format_duration"] - exp_dur), 4),
        "duration_ok": abs(po["format_duration"] - exp_dur) <= frame_dur + 1e-3,
    }
    if pi["audio"]:
        mi, mo = audio_md5(args.inp, args.max_seconds), audio_md5(args.out)
        container["audio_present"] = po["audio"] is not None
        container["audio_bit_identical"] = mi is not None and mi == mo
        container["audio_codec_same"] = po["audio"] is not None and po["audio"]["codec"] == pi["audio"]["codec"]

    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.dilate + 1,) * 2)
    bg_r = args.dilate + args.block + 4
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bg_r + 1,) * 2)
    k_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    rows, sheet_frames = [], []
    sheet_idx = {int(x) for x in args.sheet_frames.split(",")}
    for i, (fi, fo) in enumerate(zip(read_frames(args.inp, w, h, n_expected),
                                     read_frames(args.out, w, h, n_expected))):
        g = unpack(gt["masks"][i], h, w)
        p = unpack(pm["masks"][i], h, w)
        pd = cv2.dilate(p.astype(np.uint8), k_dil).astype(bool)
        bg = ~cv2.dilate((g | pd).astype(np.uint8), k_bg).astype(bool)
        d = np.abs(fi.astype(np.int16) - fo.astype(np.int16))
        mse = float((d[bg].astype(np.float32) ** 2).mean())
        row = {"frame": i, "gt_area": int(g.sum()), "pred_area": int(p.sum()),
               "bg_mad": round(float(d[bg].mean()), 3),
               "bg_psnr": round(10 * np.log10(255 ** 2 / max(mse, 1e-9)), 2)}
        if g.any():
            inter, union = int((g & p).sum()), int((g | p).sum())
            row["iou"] = round(inter / union, 4)
            row["coverage"] = round(float((g & pd).sum() / g.sum()), 4)
            core = cv2.erode(g.astype(np.uint8), k_core).astype(bool)
            fb = full_blocks(core & pd, args.block)
            if fb.any():
                row["blocks_inside"] = int(fb.sum())
                row["in_block_std_input"] = round(block_std(fi, fb, args.block), 2)
                row["in_block_std_output"] = round(block_std(fo, fb, args.block), 2)
            else:
                # too little of the target in frame for a full block
                row["blocks_inside"] = 0
        else:
            row["false_pos_px"] = int(p.sum())
        rows.append(row)
        if i in sheet_idx:
            sheet_frames.append((i, fi.copy(), fo.copy()))

    present = [r for r in rows if r["gt_area"] > 0]
    absent = [r for r in rows if r["gt_area"] == 0]
    with_blocks = [r for r in present if r.get("blocks_inside")]
    ratio = [r["in_block_std_output"] / max(r["in_block_std_input"], 1e-6) for r in with_blocks]
    summary = {
        "frames_compared": len(rows),
        "gt_present_frames": len(present), "gt_absent_frames": len(absent),
        "mean_iou_present": round(float(np.mean([r["iou"] for r in present])), 4) if present else None,
        "median_iou_present": round(float(np.median([r["iou"] for r in present])), 4) if present else None,
        "frames_coverage_ge_0.99": sum(r["coverage"] >= 0.99 for r in present),
        "frames_coverage_lt_0.5": [r["frame"] for r in present if r["coverage"] < 0.5],
        "mean_coverage_present": round(float(np.mean([r["coverage"] for r in present])), 4) if present else None,
        "in_block_std_input_mean": round(float(np.mean([r["in_block_std_input"] for r in with_blocks])), 2) if with_blocks else None,
        "in_block_std_output_mean": round(float(np.mean([r["in_block_std_output"] for r in with_blocks])), 2) if with_blocks else None,
        "in_block_std_ratio_max": round(float(np.max(ratio)), 3) if ratio else None,
        "bg_psnr_min": min(r["bg_psnr"] for r in rows),
        "bg_psnr_mean": round(float(np.mean([r["bg_psnr"] for r in rows])), 2),
        "bg_mad_max": max(r["bg_mad"] for r in rows),
        "false_pos_px_absent_frames_max": max((r["false_pos_px"] for r in absent), default=0),
    }
    report = {"container": container, "summary": summary, "per_frame": rows}
    print(json.dumps({"container": {k: v for k, v in container.items() if k not in ("input", "output")},
                      "summary": summary}, indent=2))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    if args.sheet and sheet_frames:
        tiles = []
        for i, fi, fo in sheet_frames:
            pair = np.hstack([fi, fo])
            cv2.putText(pair, f"frame {i}: input | output", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            tiles.append(cv2.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2)))
        ext = os.path.splitext(args.sheet)[1] or ".png"  # imencode+tofile: non-ASCII paths on Windows
        cv2.imencode(ext, np.vstack(tiles))[1].tofile(args.sheet)


if __name__ == "__main__":
    main()
