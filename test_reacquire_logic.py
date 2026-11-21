#!/usr/bin/env python3
"""Test of mosaic.py's YOLO re-acquisition control flow (lost -> detect ->
re-prompt -> restart propagation) on the synthetic clip.

Real YOLO and real frames are used; SAM 2 is replaced by a stub that returns the
ground-truth mask, except that after the target leaves the frame it stays
"blind" (returns empty masks) until it receives a box prompt overlapping the
target. Thus the target is only masked again if mosaic.Tracker re-acquires it
via YOLO. This isolates the pipeline logic from SAM 2's own (slow on this CPU)
re-identification, which in the real end-to-end run recovered by itself.

Usage: python3 test_reacquire_logic.py [test/test_clip.mp4 test/test_clip.gt.npz]
"""
import shutil
import sys
import tempfile
import types

import numpy as np
import torch

import mosaic


class StubSAM:
    def __init__(self, gt, blind_from):
        self.gt, self.blind_from = gt, blind_from
        self.h, self.w = (int(x) for x in gt["shape"])
        self.locked_from = None
        self.prompts = []

    def gt_mask(self, f):
        return np.unpackbits(self.gt["masks"][f], count=self.h * self.w).reshape(self.h, self.w).astype(bool)

    def out(self, f):
        m = self.gt_mask(f)
        if f >= self.blind_from and (self.locked_from is None or f < self.locked_from):
            m = np.zeros_like(m)
        return torch.from_numpy(np.where(m, 10.0, -10.0).astype(np.float32))[None, None]

    def add_new_points_or_box(self, state, frame_idx, obj_id, points=None, labels=None, box=None):
        self.prompts.append((frame_idx, None if box is None else [float(v) for v in box]))
        if box is not None and frame_idx >= self.blind_from:
            bb = mosaic.mask_bbox(self.gt_mask(frame_idx))
            if bb and mosaic.inter_area(bb, list(box)) > 0.3 * mosaic.box_area(bb):
                self.locked_from = frame_idx
        return frame_idx, [1], self.out(frame_idx)

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track, reverse=False):
        step = -1 if reverse else 1
        end = start_frame_idx + step * max_frame_num_to_track
        for f in range(start_frame_idx, end + step, step):
            yield f, [1], self.out(f)


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "test/test_clip.mp4"
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "test/test_clip.gt.npz"
    gt = np.load(gt_path)
    areas = gt["areas"]
    absent = np.nonzero(areas == 0)[0]
    exit_f, reenter_f = int(absent[0]), int(absent[-1]) + 1

    args = mosaic.build_argparser().parse_args(
        [clip, "/dev/null", "--point", "96,180", "--detect-reacquire",
         "--detect-classes", "sports ball"])
    tmp = tempfile.mkdtemp(prefix="reacq_test_")
    try:
        n, w, h = mosaic.extract_jpegs(clip, tmp, None)
        tr = mosaic.Tracker.__new__(mosaic.Tracker)
        tr.args, tr.frames_dir, tr.n, tr.w, tr.h, tr.torch = args, tmp, n, w, h, torch
        tr.pred = StubSAM(gt, blind_from=exit_f)
        tr.state = {"output_dict_per_obj": {0: {"non_cond_frame_outputs": {}}}}
        tr.lazy = mosaic.LazyFrames(tmp, n, 1024)
        tr.det = mosaic.Detector(args.yolo_weights, args.detect_classes, args.detect_conf)
        tr.masks, tr.areas, tr.events = {}, {}, []
        tr.sam_frames, tr.sam_seconds = 0, 0.0
        prompts = {0: {"pos": [[96.0, 180.0]], "neg": [], "box": None, "kind": "user"}}
        tr.run(0, prompts)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ev = tr.events
    reacq = [e for e in ev if e["event"] == "reacquire"]
    ok_reacq = [e for e in reacq if e["mask_area"] > 0]
    print("events:")
    for e in ev:
        print("  ", {k: e[k] for k in e if k != "prompt"} | ({"box": e["prompt"]["box"]} if "prompt" in e else {}))
    first = ok_reacq[0]["frame"] if ok_reacq else None
    after = [f for f in range(first or n, n) if areas[f] > 0]
    unmasked_after = [f for f in after if tr.areas.get(f, 0) == 0]
    missed = [f for f in range(reenter_f, first or n) if areas[f] > 0]
    res = {
        "exit_frame": exit_f, "reenter_frame": reenter_f,
        "lost_event_frame": next((e["frame"] for e in ev if e["event"] == "lost"), None),
        "reacquired_at": first,
        "reacquire_latency_frames": None if first is None else first - reenter_f,
        "frames_target_visible_but_unmasked_before_reacquire": missed,
        "visible_frames_unmasked_after_reacquire": unmasked_after,
        "failed_reacquire_attempts": len(reacq) - len(ok_reacq),
        "yolo_calls": tr.det.calls,
        "yolo_ms_per_call": round(1000 * tr.det.seconds / max(tr.det.calls, 1), 1),
    }
    print(res)
    assert res["lost_event_frame"] == exit_f, "loss not detected at exit"
    assert first is not None, "target never re-acquired"
    assert first >= reenter_f, "re-acquired before target re-entered (false positive)"
    assert not unmasked_after, "target unmasked after re-acquisition"
    print("PASS")


if __name__ == "__main__":
    main()
