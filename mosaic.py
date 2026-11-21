#!/usr/bin/env python3
"""Video mosaic (pixelation) of a user-specified object/part.

Pipeline
  1. Decode frames (ffmpeg) to a temp JPEG dir that SAM 2 reads lazily.
  2. SAM 2.1 (hiera_tiny) video predictor segments + tracks the target from
     the user's point/box prompt(s). Forward pass from --frame to the end, then a
     reverse pass from --frame back to frame 0 (if --frame > 0).
     --reprompt FRAME:... adds extra conditioning prompts mid-video.
  3. When tracking is lost (mask area collapses, e.g. the object leaves the
     frame) and --detect-reacquire is on, YOLO (nano) scans subsequent frames
     inside a growing search window around the last known region, maps the
     target's last known position relative to its detection box onto the new
     detection, and re-prompts SAM 2 with that box.
  4. Each frame: dilate mask, pixelate pixels under it (grid-aligned blocks),
     pipe to ffmpeg (libx264), copy the original audio stream.

Example
  python3 mosaic.py in.mp4 out.mp4 --frame 0 --point 96,180
  python3 mosaic.py in.mp4 out.mp4 --frame 30 --box 100,80,220,200 \\
      --reprompt 240:410,150 --detect-reacquire --block 16
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict, deque
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
# keep third-party caches/config inside the project dir, silence SAM 2's tqdm bars
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(HERE, ".ultralytics"))
os.environ.setdefault("TQDM_DISABLE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_SAM_CKPT = os.path.join(HERE, "models", "sam2.1_hiera_tiny.pt")
DEFAULT_SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # resolved inside the sam2 package
DEFAULT_YOLO = os.path.join(HERE, "models", "yolo11n.pt")


def log(msg):
    print(f"[mosaic] {msg}", file=sys.stderr, flush=True)


def fmt_dur(s):
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Progress:
    """Throttled progress lines via log(): '<label>: done/total frames (pct), elapsed, ETA'.

    A line is printed when progress advanced by >= `step_frac` of the total or
    `max_gap` seconds passed since the last line, but never sooner than `min_gap`
    seconds after the previous line. finish() adds a final line if the last one
    did not already show the final count. ETA is shown only once `min_done`
    frames are done and some time has elapsed."""

    def __init__(self, label, total, step_frac=0.05, max_gap=15.0, min_gap=2.0, min_done=3,
                 clock=time.perf_counter):
        self.label, self.total = label, max(int(total), 1)
        self.step_frac, self.max_gap, self.min_gap, self.min_done = step_frac, max_gap, min_gap, min_done
        self.clock = clock
        self.t0 = self.last_t = clock()
        self.done = self.last_done = 0

    def update(self, done):
        self.done = done = min(max(int(done), self.done), self.total)
        now = self.clock()
        since = now - self.last_t
        if done == self.last_done or since < self.min_gap:
            return
        if (done - self.last_done) / self.total >= self.step_frac or since >= self.max_gap:
            self._emit(now)

    def finish(self):
        if self.done != self.last_done:
            self._emit(self.clock())

    def _emit(self, now):
        el = now - self.t0
        msg = (f"{self.label}: {self.done}/{self.total} frames "
               f"({100.0 * self.done / self.total:.1f}%), elapsed {fmt_dur(el)}")
        if self.done >= self.total:
            msg += f", {self.done / el:.2f} frames/s" if el > 0 else ""
        elif self.done >= self.min_done and el > 0:
            msg += f", ETA {fmt_dur((self.total - self.done) * el / self.done)}"
        log(msg)
        self.last_t, self.last_done = now, self.done


# --------------------------------------------------------------------------- CLI
def parse_xy(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 2:
        raise argparse.ArgumentTypeError(f"expected X,Y got {s!r}")
    return v


def parse_box(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]:
        raise argparse.ArgumentTypeError(f"expected X1,Y1,X2,Y2 with X2>X1,Y2>Y1, got {s!r}")
    return v


def parse_reprompt(s):
    """FRAME:X,Y (positive point), FRAME:-X,Y is NOT supported (ambiguous);
    use FRAME:neg:X,Y for a negative point, FRAME:X1,Y1,X2,Y2 for a box."""
    parts = s.split(":")
    try:
        frame = int(parts[0])
        if len(parts) == 3 and parts[1] == "neg":
            return frame, "neg", parse_xy(parts[2])
        if len(parts) != 2:
            raise ValueError
        nums = parts[1].split(",")
        if len(nums) == 2:
            return frame, "pos", parse_xy(parts[1])
        if len(nums) == 4:
            return frame, "box", parse_box(parts[1])
        raise ValueError
    except (ValueError, argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(
            f"bad --reprompt {s!r}; use FRAME:X,Y | FRAME:neg:X,Y | FRAME:X1,Y1,X2,Y2")


def default_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_argparser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--frame", type=int, default=0, help="frame index of the initial prompt")
    ap.add_argument("--point", type=parse_xy, action="append", default=[],
                    help="positive point X,Y on --frame (repeatable)")
    ap.add_argument("--neg-point", type=parse_xy, action="append", default=[],
                    help="negative (background) point X,Y on --frame (repeatable)")
    ap.add_argument("--box", type=parse_box, help="box X1,Y1,X2,Y2 on --frame")
    ap.add_argument("--reprompt", type=parse_reprompt, action="append", default=[],
                    help="extra prompt FRAME:X,Y | FRAME:neg:X,Y | FRAME:X1,Y1,X2,Y2 (repeatable)")
    ap.add_argument("--block", type=int, default=12, help="mosaic block size in px (default 12)")
    ap.add_argument("--dilate", type=int, default=6, help="mask dilation in px (default 6)")
    ap.add_argument("--detect-reacquire", action=argparse.BooleanOptionalAction, default=False,
                    help="use YOLO to re-acquire the target when tracking is lost (default off)")
    ap.add_argument("--detect-classes", default="person",
                    help="comma-separated YOLO class names used for re-acquisition (default person)")
    ap.add_argument("--detect-conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--detect-every", type=int, default=1,
                    help="while lost, run YOLO every N frames (default 1)")
    ap.add_argument("--lost-frac", type=float, default=0.05,
                    help="tracking counts as lost when mask area < this fraction of the recent "
                         "median area (default 0.05)")
    ap.add_argument("--min-area", type=int, default=16, help="mask area (px) below which = lost")
    ap.add_argument("--max-seconds", type=float, help="only process the first N seconds")
    ap.add_argument("--crf", type=int, default=18, help="x264 CRF of the output (default 18)")
    ap.add_argument("--preset", default="medium", help="x264 preset (default medium)")
    ap.add_argument("--sam-checkpoint", default=DEFAULT_SAM_CKPT)
    ap.add_argument("--sam-config", default=DEFAULT_SAM_CFG)
    ap.add_argument("--yolo-weights", default=DEFAULT_YOLO)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=default_device(),
                    help="device for SAM 2 (default: cuda if torch.cuda.is_available(), else cpu)")
    ap.add_argument("--bf16", action="store_true",
                    help="run SAM 2 under bfloat16 autocast on --device (on cpu ~1.6x faster "
                         "encoder only on AVX512-BF16 CPUs; slightly different masks)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--save-masks", help="write per-frame masks (packed bits) to this .npz")
    ap.add_argument("--stats-json", help="write run statistics to this JSON file")
    ap.add_argument("--keep-temp", action="store_true", help="keep the temp frame directory")
    return ap


# ----------------------------------------------------------------------- video IO
def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration"
         ":format=duration", "-of", "json", path],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise SystemExit(f"no video stream in {path}")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    fps = v.get("avg_frame_rate") or "0/0"
    if fps.startswith("0/") or fps.endswith("/0"):
        fps = v["r_frame_rate"]
    return {"fps": fps, "r_frame_rate": v["r_frame_rate"],
            "has_audio": has_audio, "duration": float(info["format"].get("duration", 0) or 0)}


def ffmpeg_trim_args(max_seconds):
    return ["-t", f"{max_seconds:.6f}"] if max_seconds else []


def imread(path):
    """cv2.imread that also works for non-ASCII paths on Windows (e.g. C:\\Users\\<Japanese name>)."""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def _extract_cmd(path, out_dir, max_seconds, vf=None):
    return ["ffmpeg", "-v", "error", "-nostdin", "-i", path, *ffmpeg_trim_args(max_seconds),
            "-map", "0:v:0", *(["-vf", vf] if vf else []),
            "-fps_mode", "passthrough", "-q:v", "2", "-start_number", "0",
            os.path.join(out_dir, "%06d.jpg")]


def _clear_jpegs(out_dir):
    for f in os.listdir(out_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(out_dir, f))


def extract_jpegs(path, out_dir, max_seconds, fps=None):
    """Decode every frame of the first video stream to out_dir/%06d.jpg.

    The first attempt is the original passthrough command (unchanged behaviour for
    well-formed input). Some inputs (VFR sources, `-c copy` cuts) carry duplicate or
    non-monotonic timestamps, which the mjpeg encoder / image2 muxer reject
    ("Invalid pts (X) <= last (X)"). Only then we retry once with every decoded frame
    re-stamped sequentially at the probed fps (setpts=N/(fps*TB)): no frame is dropped
    or duplicated, so the JPEG sequence still lines up 1:1 with the frames render()
    decodes. If the retry also fails, the original error is raised."""
    try:
        subprocess.run(_extract_cmd(path, out_dir, max_seconds), check=True)
    except subprocess.CalledProcessError as first_err:
        try:
            rate = Fraction(fps or ffprobe(path)["fps"])
        except (ValueError, ZeroDivisionError):
            rate = Fraction(0)
        if rate <= 0:
            raise  # no usable frame rate to re-stamp with: surface the original error
        log(f"frame extraction failed (ffmpeg exit {first_err.returncode}), probably due to "
            f"duplicate/non-monotonic timestamps; retrying with timestamps re-stamped at "
            f"{rate} fps")
        _clear_jpegs(out_dir)
        vf = f"setpts=N*{rate.denominator}/({rate.numerator}*TB)"
        try:
            subprocess.run(_extract_cmd(path, out_dir, max_seconds, vf), check=True)
        except subprocess.CalledProcessError as retry_err:
            log(f"retry with re-stamped timestamps also failed (ffmpeg exit {retry_err.returncode})")
            raise first_err from retry_err
    n = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    if n == 0:
        raise SystemExit("no frames decoded")
    h, w = imread(os.path.join(out_dir, "000000.jpg")).shape[:2]
    return n, w, h


class LazyFrames:
    """Stands in for SAM 2's preloaded frame tensor: loads + normalises a JPEG on
    demand with a tiny LRU cache, so memory does not grow with video length."""

    def __init__(self, frame_dir, n, image_size, cache=4):
        import torch
        self.torch = torch
        self.dir, self.n, self.size, self.cache_n = frame_dir, n, image_size, cache
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        self.cache = OrderedDict()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from PIL import Image
        i = int(i)
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        with Image.open(os.path.join(self.dir, f"{i:06d}.jpg")) as im:
            arr = np.asarray(im.convert("RGB").resize((self.size, self.size)), dtype=np.float32)
        t = self.torch.from_numpy(arr / 255.0).permute(2, 0, 1).contiguous()
        t = (t - self.mean) / self.std
        self.cache[i] = t
        if len(self.cache) > self.cache_n:
            self.cache.popitem(last=False)
        return t

    def bgr(self, i):
        return imread(os.path.join(self.dir, f"{int(i):06d}.jpg"))


# ------------------------------------------------------------------------ detector
class Detector:
    def __init__(self, weights, class_names, conf):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        names = self.model.names
        wanted = [c.strip() for c in class_names.split(",") if c.strip()]
        inv = {v: k for k, v in names.items()}
        missing = [c for c in wanted if c not in inv]
        if missing:
            raise SystemExit(f"unknown YOLO class(es) {missing}; known: {sorted(inv)}")
        self.classes = [inv[c] for c in wanted]
        self.conf = conf
        self.calls, self.seconds = 0, 0.0

    def __call__(self, bgr):
        t0 = time.perf_counter()
        r = self.model.predict(bgr, conf=self.conf, classes=self.classes, verbose=False)[0]
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return [(b.tolist(), float(s)) for b, s in zip(r.boxes.xyxy.cpu(), r.boxes.conf.cpu())]


def mask_bbox(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a, b):
    return box_area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


# ------------------------------------------------------------------------ tracking
class Tracker:
    KEEP_NONCOND = 24  # >= num_maskmem(7) and max_obj_ptrs_in_encoder(16)

    def __init__(self, args, frames, n, w, h):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        import sam2.sam2_video_predictor as svp

        self.args, self.frames_dir, self.n, self.w, self.h = args, frames, n, w, h
        torch.set_num_threads(args.threads)
        self.torch = torch
        # Re-prompts on already-tracked frames must become conditioning frames,
        # otherwise propagate_in_video() recomputes the frame and drops the prompt.
        self.pred = build_sam2_video_predictor(
            args.sam_config, args.sam_checkpoint, device=args.device,
            hydra_overrides_extra=["++model.add_all_frames_to_correct_as_cond=true"])
        self.lazy = LazyFrames(frames, n, self.pred.image_size)
        lazy = self.lazy
        orig_loader = svp.load_video_frames
        svp.load_video_frames = lambda **kw: (lazy, h, w)
        try:
            self.state = self.pred.init_state(video_path=frames, offload_video_to_cpu=True,
                                              offload_state_to_cpu=True)
        finally:
            svp.load_video_frames = orig_loader

        self.det = None
        if args.detect_reacquire:
            self.det = Detector(args.yolo_weights, args.detect_classes, args.detect_conf)
        self.masks = {}          # frame -> packed bool mask
        self.areas = {}
        self.events = []         # prompts / lost / reacquired log
        self.sam_frames, self.sam_seconds = 0, 0.0

    def autocast(self):
        return self.torch.autocast(self.args.device, dtype=self.torch.bfloat16,
                                   enabled=self.args.bf16)

    # -- prompts ----------------------------------------------------------------
    def add_prompt(self, frame, p):
        pts = p["pos"] + p["neg"]
        labels = [1] * len(p["pos"]) + [0] * len(p["neg"])
        kw = {}
        if pts:
            kw["points"] = np.array(pts, np.float32)
            kw["labels"] = np.array(labels, np.int32)
        if p["box"] is not None:
            kw["box"] = np.array(p["box"], np.float32)
        with self.torch.inference_mode(), self.autocast():
            _, _, vm = self.pred.add_new_points_or_box(self.state, frame_idx=frame, obj_id=1, **kw)
        area = int((vm[0, 0] > 0).sum())
        self.events.append({"frame": frame, "event": p.get("kind", "prompt"),
                            "prompt": {k: v for k, v in p.items() if k != "kind"},
                            "mask_area": area})
        log(f"prompt@{frame} ({p.get('kind', 'prompt')}): mask area {area}px")
        return area

    def prune(self, f, step):
        nc = self.state["output_dict_per_obj"][0]["non_cond_frame_outputs"]
        for k in [k for k in nc if (f - k) * step > self.KEEP_NONCOND]:
            del nc[k]

    # -- re-acquisition ---------------------------------------------------------
    def relative_geometry(self, frame, bbox):
        """Where the target sits inside its associated detection box (so a part of a
        person can be re-proposed from a new person box). None -> use det box."""
        dets = self.det(self.lazy.bgr(frame))
        best, best_frac = None, 0.0
        for b, _ in dets:
            frac = inter_area(b, bbox) / max(box_area(bbox), 1.0)
            if frac > best_frac:
                best, best_frac = b, frac
        if best is None or best_frac < 0.5:
            return None
        bw, bh = best[2] - best[0], best[3] - best[1]
        return [(bbox[0] - best[0]) / bw, (bbox[1] - best[1]) / bh,
                (bbox[2] - best[0]) / bw, (bbox[3] - best[1]) / bh]

    def search_window(self, last_bbox, frames_lost):
        x1, y1, x2, y2 = last_bbox
        size = max(x2 - x1, y2 - y1, 32.0)
        m = size * (1.0 + 0.25 * frames_lost)
        return [max(0.0, x1 - m), max(0.0, y1 - m), min(self.w, x2 + m), min(self.h, y2 + m)]

    def find_candidate(self, f, last_bbox, frames_lost, rel):
        win = self.search_window(last_bbox, frames_lost)
        dets = self.det(self.lazy.bgr(f))
        lc = ((last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2)
        diag = math.hypot(self.w, self.h)
        best, best_score = None, -1.0
        for b, s in dets:
            if inter_area(b, win) < 0.5 * box_area(b):
                continue  # outside the search region
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            score = s * math.exp(-math.hypot(c[0] - lc[0], c[1] - lc[1]) / (0.5 * diag))
            if score > best_score:
                best, best_score = (b, s), score
        if best is None:
            return None
        b, s = best
        if rel is not None:
            bw, bh = b[2] - b[0], b[3] - b[1]
            prop = [b[0] + rel[0] * bw, b[1] + rel[1] * bh, b[0] + rel[2] * bw, b[1] + rel[3] * bh]
        else:
            prop = list(b)
        prop = [min(max(prop[0], 0), self.w - 1), min(max(prop[1], 0), self.h - 1),
                min(max(prop[2], 1), self.w), min(max(prop[3], 1), self.h)]
        if prop[2] - prop[0] < 2 or prop[3] - prop[1] < 2:
            return None
        return {"det_box": [round(v, 1) for v in b], "det_conf": round(s, 3),
                "box": [round(v, 1) for v in prop], "window": [round(v) for v in win]}

    # -- one tracking pass ------------------------------------------------------
    def run_pass(self, start, step, prompts):
        """Track from `start` in direction `step` (+1/-1). `prompts`: frame->prompt for
        frames strictly after `start` in this direction."""
        end = self.n - 1 if step > 0 else 0
        stops = sorted((f for f in prompts if (f - start) * step > 0), key=lambda f: f * step)
        a = self.args
        ref = deque(maxlen=15)
        lost_since, last_bbox, last_good, rel, next_try = None, None, None, None, None
        prog = Progress(f"tracking {'forward' if step > 0 else 'reverse'}",
                        abs(end - start) + 1)
        t = start
        while True:
            nxt = stops[0] if stops else None
            last = end if nxt is None else nxt - step
            restart = None
            t0 = time.perf_counter()
            with self.torch.inference_mode(), self.autocast():
                gen = self.pred.propagate_in_video(
                    self.state, start_frame_idx=t, max_frame_num_to_track=abs(last - t),
                    reverse=step < 0)
                for f, _, vm in gen:
                    m = (vm[0, 0] > 0).cpu().numpy()
                    self.sam_frames += 1
                    self.prune(f, step)
                    area = int(m.sum())
                    self.masks[f] = np.packbits(m)
                    self.areas[f] = area
                    prog.update(abs(f - start) + 1)  # frames of this pass covered so far
                    thr =max(a.min_area, a.lost_frac * (float(np.median(ref)) if ref else area))
                    if area >= thr and area > 0:
                        if lost_since is not None:
                            self.events.append({"frame": f, "event": "tracking_resumed",
                                                "lost_frames": abs(f - lost_since), "mask_area": area})
                            log(f"tracking resumed @{f} after {abs(f - lost_since)} lost frames")
                        lost_since, rel = None, None
                        ref.append(area)
                        last_bbox = mask_bbox(m)
                        last_good = f
                        continue
                    # ---- lost ----
                    if lost_since is None:
                        lost_since = f
                        self.events.append({"frame": f, "event": "lost", "mask_area": area,
                                            "ref_area": float(np.median(ref)) if ref else None})
                        log(f"tracking lost @{f} (area {area}px)")
                        if self.det is not None and last_bbox is not None:
                            rel = self.relative_geometry(last_good, last_bbox)
                        next_try = f
                    if self.det is None or last_bbox is None or (f - next_try) * step < 0:
                        continue
                    next_try = f + step * a.detect_every
                    cand = self.find_candidate(f, last_bbox, abs(f - lost_since), rel)
                    if cand is not None:
                        restart = (f, cand)
                        break
                gen.close()
            self.sam_seconds += time.perf_counter() - t0
            if restart is not None:
                f, cand = restart
                area = self.add_prompt(f, {"pos": [], "neg": [], "box": cand["box"],
                                           "kind": "reacquire"})
                self.events[-1].update({k: cand[k] for k in ("det_box", "det_conf", "window")})
                if area < a.min_area:
                    # SAM rejected the proposal: keep scanning a bit later
                    next_try = f + step * max(a.detect_every, 5)
                t = f
                continue
            if nxt is None:
                prog.finish()
                break
            stops.pop(0)
            self.add_prompt(nxt, prompts[nxt])
            t = nxt

    def run(self, frame0, prompts):
        if not (0 <= frame0 < self.n):
            raise SystemExit(f"--frame {frame0} outside [0, {self.n - 1}]")
        area = self.add_prompt(frame0, prompts.pop(frame0))
        if area == 0:
            log("WARNING: initial prompt produced an empty mask; check coordinates")
        self.run_pass(frame0, +1, {f: p for f, p in prompts.items() if f > frame0})
        if frame0 > 0:
            self.run_pass(frame0, -1, {f: p for f, p in prompts.items() if f < frame0})

    def mask(self, f):
        p = self.masks.get(f)
        if p is None:
            return None
        return np.unpackbits(p, count=self.w * self.h).reshape(self.h, self.w).astype(bool)


# -------------------------------------------------------------------------- mosaic
def pixelate_masked(img, mask, block, kernel):
    """Replace pixels under (dilated) mask by the mean colour of their block.
    Blocks are aligned to the global image grid so they do not 'swim'."""
    if mask is None or not mask.any():
        return img
    m = cv2.dilate(mask.astype(np.uint8), kernel) if kernel is not None else mask.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(m)
    H, W = img.shape[:2]
    x0, y0 = (x // block) * block, (y // block) * block
    x1, y1 = min(-(-(x + w) // block) * block, W), min(-(-(y + h) // block) * block, H)
    roi = img[y0:y1, x0:x1].astype(np.float32)
    rh, rw = roi.shape[:2]
    nby, nbx = -(-rh // block), -(-rw // block)
    pad = np.zeros((nby * block, nbx * block, 3), np.float32)
    cnt = np.zeros((nby * block, nbx * block, 1), np.float32)
    pad[:rh, :rw], cnt[:rh, :rw] = roi, 1.0
    sums = pad.reshape(nby, block, nbx, block, 3).sum(axis=(1, 3))
    cnts = cnt.reshape(nby, block, nbx, block, 1).sum(axis=(1, 3))
    means = np.round(sums / cnts).astype(np.uint8)
    pix = np.repeat(np.repeat(means, block, axis=0), block, axis=1)[:rh, :rw]
    out = img.copy()
    sel = m[y0:y1, x0:x1].astype(bool)
    out[y0:y1, x0:x1][sel] = pix[sel]
    return out


def render(args, tracker, info, n, w, h):
    kernel = None
    if args.dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.dilate + 1,) * 2)
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", args.input, *ffmpeg_trim_args(args.max_seconds),
         "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE)
    enc_cmd = ["ffmpeg", "-v", "error", "-y", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-framerate", info["fps"],
               "-i", "-", "-i", args.input, "-map", "0:v:0"]
    if info["has_audio"]:
        enc_cmd += ["-map", "1:a?", "-c:a", "copy"]
    enc_cmd += ["-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
                "-pix_fmt", "yuv420p", "-map_metadata", "1", "-movflags", "+faststart",
                *ffmpeg_trim_args(args.max_seconds), args.output]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    fsize = w * h * 3
    i, masked = 0, 0
    t0 = time.perf_counter()
    prog = Progress("rendering", n)
    try:
        while i < n:
            buf = dec.stdout.read(fsize)
            if len(buf) < fsize:
                break
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            m = tracker.mask(i)
            if m is not None and m.any():
                masked += 1
                img = pixelate_masked(img, m, args.block, kernel)
            enc.stdin.write(img.tobytes())
            i += 1
            prog.update(i)
        prog.finish()
    finally:
        dec.stdout.close()
        dec.wait()
        enc.stdin.close()
        rc = enc.wait()
    if rc != 0:
        raise SystemExit("ffmpeg encode failed")
    if i != n:
        log(f"WARNING: decoded {i} frames for rendering but {n} were tracked")
    return {"frames_written": i, "frames_masked": masked, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------------- main
def peak_rss_mb():
    try:
        import resource  # POSIX only
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except ImportError:
        try:
            import psutil  # Windows: peak working set (psutil ships with ultralytics)
            mi = psutil.Process().memory_info()
            return round(getattr(mi, "peak_wset", mi.rss) / 2**20)
        except ImportError:
            return None


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if not args.point and args.box is None:
        raise SystemExit("give at least one --point or a --box for --frame")
    if args.block < 2:
        raise SystemExit("--block must be >= 2")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    log(f"device: {args.device}")
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            raise SystemExit(f"{exe} not found on PATH")

    t_start = time.perf_counter()
    info = ffprobe(args.input)
    prompts = {args.frame: {"pos": [list(p) for p in args.point],
                            "neg": [list(p) for p in args.neg_point],
                            "box": args.box, "kind": "user"}}
    for f, kind, val in args.reprompt:
        p = prompts.setdefault(f, {"pos": [], "neg": [], "box": None, "kind": "reprompt"})
        if kind == "box":
            p["box"] = val
        else:
            p[kind].append(val)

    tmp = tempfile.mkdtemp(prefix="mosaic_frames_", dir=HERE)
    try:
        t0 = time.perf_counter()
        log("extracting frames...")
        n, w, h = extract_jpegs(args.input, tmp, args.max_seconds, info["fps"])
        t_decode = time.perf_counter() - t0
        log(f"{n} frames {w}x{h} @ {info['fps']} fps, audio={info['has_audio']} "
            f"(decode {t_decode:.1f}s)")
        for f in prompts:
            if not 0 <= f < n:
                raise SystemExit(f"prompt frame {f} outside [0, {n - 1}]")

        t0 = time.perf_counter()
        tracker = Tracker(args, tmp, n, w, h)
        t_load = time.perf_counter() - t0
        log(f"models loaded in {t_load:.1f}s; tracking...")
        t0 = time.perf_counter()
        tracker.run(args.frame, prompts)
        t_track = time.perf_counter() - t0
        log(f"tracking done: {tracker.sam_frames} SAM frame steps in {t_track:.1f}s "
            f"({1000 * t_track / max(n, 1):.0f} ms/frame)")

        r = render(args, tracker, info, n, w, h)
        log(f"rendered {r['frames_written']} frames ({r['frames_masked']} with mask) "
            f"in {r['seconds']:.1f}s -> {args.output}")

        if args.save_masks:
            np.savez_compressed(
                args.save_masks, shape=np.array([h, w]),
                masks=np.stack([tracker.masks.get(i, np.zeros((h * w + 7) // 8, np.uint8))
                                for i in range(n)]),
                tracked=np.array([i in tracker.masks for i in range(n)]))
        total = time.perf_counter() - t_start
        stats = {
            "input": args.input, "output": args.output, "frames": n, "size": [w, h],
            "fps": info["fps"], "args": {k: v for k, v in vars(args).items()},
            "timing_s": {"decode": round(t_decode, 2), "model_load": round(t_load, 2),
                         "tracking": round(t_track, 2), "render_encode": round(r["seconds"], 2),
                         "total": round(total, 2)},
            "ms_per_frame": {"tracking_incl_yolo": round(1000 * t_track / n, 1),
                             "render_encode": round(1000 * r["seconds"] / n, 1),
                             "end_to_end_excl_model_load": round(1000 * (total - t_load) / n, 1)},
            "sam_frame_steps": tracker.sam_frames,
            "yolo_calls": tracker.det.calls if tracker.det else 0,
            "yolo_ms_per_call": round(1000 * tracker.det.seconds / tracker.det.calls, 1)
            if tracker.det and tracker.det.calls else None,
            "frames_masked": r["frames_masked"],
            "peak_rss_mb": peak_rss_mb(),
            "events": tracker.events,
        }
        log(f"total {total:.1f}s; events: " +
            ", ".join(f"{e['event']}@{e['frame']}" for e in tracker.events))
        if args.stats_json:
            with open(args.stats_json, "w") as fh:
                json.dump(stats, fh, indent=2, default=str)
    finally:
        if args.keep_temp:
            log(f"temp frames kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
