#!/usr/bin/env python3
"""モザイク GUI — a Gradio front-end for mosaic.py.

The app never imports torch / sam2. It only calls ffprobe / ffmpeg (to inspect
the video and grab single frames for clicking) and runs mosaic.py as a
subprocess with the same Python interpreter (sys.executable).

Tested with gradio 5.50.0.

BRIEF-APP2 additions: frame stepping (±1 / ±10 / jump to frame number, synced
with the time sliders), an "additional fix" mode (input = a finished output,
result = *_fix.mp4, mosaic is layered on top), live alerts parsed from
mosaic.py's `[mosaic-event] key=value ...` stdout lines, a post-run summary
from --stats-json, click-to-jump from alert rows, and a preview run
(--max-seconds N, *_preview.mp4).

Coordinates: gr.Image's `select` event reports `evt.index = [x, y]` in the
*natural* pixel size of the displayed image (the browser rescales the click by
naturalWidth / displayed width and compensates for letter-boxing). We send the
frame at full resolution, so those are original-video pixel coordinates. As a
safety net, clicks are additionally rescaled by (frame size / image size) and
clamped, in case a downscaled preview is ever shown.
"""
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MOSAIC = HERE / "mosaic.py"
WORK = HERE / "_app_work"          # extracted frames for the picker (scratch)
LOG_TAIL_LINES = 200
IS_WIN = os.name == "nt"

POS, NEG = "対象", "対象じゃない"
POS_BGR, NEG_BGR = (0, 200, 0), (0, 0, 230)   # green = target, red = not target

# running subprocess per browser session (local single-user app, but be tidy)
RUNNING = {}


# ------------------------------------------------------------------ video I/O
def _no_window():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WIN else {}


def probe_video(path):
    """fps / size / duration via ffprobe (same fps choice as mosaic.py)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, **_no_window()).stdout
    info = json.loads(out.decode("utf-8", "replace"))
    if not info.get("streams"):
        raise ValueError("動画ストリームが見つかりません")
    v = info["streams"][0]
    fps_s = v.get("avg_frame_rate") or "0/0"
    if fps_s.startswith("0/") or fps_s.endswith("/0"):
        fps_s = v.get("r_frame_rate") or "0/0"
    try:
        fps = float(Fraction(fps_s))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    if fps <= 0:
        raise ValueError("フレームレートが取得できません")
    duration = float(info.get("format", {}).get("duration") or 0)
    try:
        nframes = int(v.get("nb_frames"))
    except (TypeError, ValueError):
        nframes = int(duration * fps)
    return {"fps": fps, "fps_str": fps_s, "width": int(v["width"]), "height": int(v["height"]),
            "duration": duration, "nframes": max(nframes, 1)}


def frame_index(seconds, fps):
    return int(round(float(seconds or 0) * fps))


def extract_frame(video, idx, fps, out_path):
    """Grab frame number `idx` (0-based, same numbering as mosaic.py's decode).

    Accurate input seek returns the first frame whose timestamp is >= the
    target; aiming half a frame early lands on frame idx even with rounding."""
    t = max(0.0, (idx - 0.5) / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{t:.6f}", "-i", str(video),
         "-map", "0:v:0", "-frames:v", "1", str(out_path)],
        check=True, capture_output=True, **_no_window())
    if not out_path.exists():
        raise RuntimeError("フレームを取り出せませんでした（動画の末尾を超えている可能性）")
    return out_path


def imread(path):
    """cv2.imread that works with non-ASCII (Japanese) paths on Windows; returns BGR."""
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"画像を読めません: {path}")
    return bgr


def draw_points(frame_path, points):
    """Redraw numbered markers on the pristine frame. Returns an RGB array."""
    img = imread(frame_path)
    h, w = img.shape[:2]
    r = max(6, round(min(w, h) / 90))
    th = max(2, r // 3)
    for i, (x, y, kind) in enumerate(points, 1):
        col = POS_BGR if kind == "pos" else NEG_BGR
        c = (int(round(x)), int(round(y)))
        cv2.circle(img, c, r + th, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, r, col, -1, cv2.LINE_AA)
        if kind == "neg":   # an "x" so the two kinds differ beyond colour
            d = int(r * 0.6)
            cv2.line(img, (c[0] - d, c[1] - d), (c[0] + d, c[1] + d), (255, 255, 255), th)
            cv2.line(img, (c[0] - d, c[1] + d), (c[0] + d, c[1] - d), (255, 255, 255), th)
        fs = r / 9
        org = (c[0] + r + th + 2, c[1] - r)
        cv2.putText(img, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def click_to_video_xy(evt_index, img_size, frame_size):
    """Map a select-event index (pixels of the displayed image's natural size)
    to frame pixels. With a full-resolution image the scale is 1."""
    x, y = float(evt_index[0]), float(evt_index[1])
    iw, ih = img_size
    fw, fh = frame_size
    if iw and ih and (iw, ih) != (fw, fh):
        x, y = x * fw / iw, y * fh / ih
    return int(min(max(round(x), 0), fw - 1)), int(min(max(round(y), 0), fh - 1))



def points_text(points, frame=None):
    head = f"開始フレーム {frame} に置いた点:\n" if frame is not None else ""
    if not points:
        return head + "（まだ点がありません。画像をクリックしてください）"
    return head + "\n".join(f"{i}. {'対象' if k == 'pos' else '対象じゃない'}  x={x}, y={y}"
                            for i, (x, y, k) in enumerate(points, 1))


def rp_points_text(points, frame, stored):
    if frame is None:
        return "（「このフレームを表示」かコマ送りでフレームを選んでください）"
    tag = "登録済み — 点を変えたら「再指定として追加／更新」" if stored else "未登録"
    head = f"フレーム {frame} の点（{tag}）:\n"
    if not points:
        return head + "（まだ点がありません。画像をクリックしてください）"
    return head + "\n".join(f"{i}. {'対象' if k == 'pos' else '対象じゃない'}  x={x}, y={y}"
                            for i, (x, y, k) in enumerate(points, 1))


def reprompts_text(reprompts, fps):
    if not reprompts:
        return "（再指定はまだありません）"
    lines = []
    for rp in reprompts:
        pts = ", ".join(f"{'+' if k == 'pos' else '−'}({x},{y})" for x, y, k in rp["points"])
        lines.append(f"フレーム {rp['frame']}（{rp['frame'] / fps:.2f} 秒）: {pts}")
    return "\n".join(lines)


RP_HEADERS = ["フレーム", "秒", "点"]


def reprompt_rows(reprompts, fps):
    """Rows for the stored re-prompt table (click a row = jump to that frame)."""
    return [[rp["frame"], f"{rp['frame'] / fps:.2f}",
             ", ".join(f"{'+' if k == 'pos' else '−'}({x},{y})" for x, y, k in rp["points"])]
            for rp in reprompts]


# ------------------------------------------------------------- command builder
MODE_NEW, MODE_FIX = "新規に処理", "追加修正（出力済み動画にモザイクを重ねる）"
FIX_RE = re.compile(r"^(.*)_fix(\d*)$")


def output_path_for(video_path, mode=MODE_NEW, preview=False):
    """Output file (always in the app folder, never the input itself).

    new:  <stem>_out.mp4            preview: <stem>_preview.mp4
    fix:  <stem>_fix.mp4 ; an input that is already <base>_fix / _fixN gives
          <base>_fix(N+1).mp4      preview: <that stem>_preview.mp4"""
    stem = Path(video_path).stem
    if mode == MODE_FIX:
        m = FIX_RE.match(stem)
        if m:
            name = f"{m.group(1)}_fix{int(m.group(2) or 1) + 1}"
        else:
            name = f"{stem}_fix"
    else:
        name = f"{stem}_out" if not preview else stem
    if preview:
        name += "_preview"
    out = HERE / f"{name}.mp4"
    if out.resolve() == Path(video_path).resolve():   # paranoia: never overwrite the input
        out = HERE / f"{name}_new.mp4"
    return out


def stats_path_for(out_path):
    return WORK / f"{Path(out_path).stem}.stats.json"


def build_command(video_path, fps, start_seconds, points, reprompts, dilate, block,
                  detect_reacquire, max_seconds, nframes=None, python=None, hold_frames=0,
                  mode=MODE_NEW, preview_seconds=None, start_frame=None, stats_json=True):
    """Return (argv, output_path, warnings). Pure function: no I/O.

    points:    [(x, y, "pos"|"neg"), ...] on the start frame
    reprompts: [{"frame": int, "points": [(x, y, kind), ...]}, ...]
    start_frame: start frame index (overrides start_seconds when given)
    mode:      MODE_NEW or MODE_FIX (input = a finished output; result *_fix.mp4)
    preview_seconds: "プレビュー実行" — process only the first N seconds
                     (overrides max_seconds) and write *_preview.mp4
    stats_json: add --stats-json <app work dir>/<output stem>.stats.json
    """
    warnings = []
    pos = [(x, y) for x, y, k in points if k == "pos"]
    neg = [(x, y) for x, y, k in points if k == "neg"]
    if not pos:
        raise ValueError("開始フレームに「対象」の点を 1 つ以上クリックしてください")
    start = int(start_frame) if start_frame is not None else frame_index(start_seconds, fps)
    preview = preview_seconds not in (None, "")
    if preview:
        if float(preview_seconds) <= 0:
            raise ValueError("プレビューの秒数は 0 より大きくしてください")
        max_seconds = preview_seconds
    limit = nframes
    if max_seconds not in (None, "") and float(max_seconds) > 0:
        max_seconds = float(max_seconds)
        limit = int(max_seconds * fps) if limit is None else min(limit, int(max_seconds * fps))
    else:
        max_seconds = None
    if limit is not None and start >= max(limit, 1):
        what = "プレビューの秒数" if preview else "「処理する秒数」"
        raise ValueError(f"開始フレーム {start} が処理範囲（{limit} フレーム）の外です。"
                         f"開始フレームを前にするか{what}を増やしてください")
    out = output_path_for(video_path, mode, preview)
    argv = [python or sys.executable, str(MOSAIC), str(video_path), str(out),
            "--frame", str(start)]
    for x, y in pos:
        argv += ["--point", f"{x},{y}"]
    for x, y in neg:
        argv += ["--neg-point", f"{x},{y}"]
    for rp in reprompts:
        f = int(rp["frame"])
        if limit is not None and f >= limit:
            warnings.append(f"フレーム {f} の再指定は処理範囲外なので使いません")
            continue
        for x, y, k in rp["points"]:
            argv += ["--reprompt", f"{f}:{x},{y}" if k == "pos" else f"{f}:neg:{x},{y}"]
    argv += ["--dilate", str(int(dilate)), "--block", str(int(block))]
    if detect_reacquire:
        argv.append("--detect-reacquire")
    try:
        hold_frames = int(hold_frames or 0)
    except (TypeError, ValueError):
        hold_frames = 0
    if hold_frames > 0:
        argv += ["--hold-frames", str(hold_frames)]
    if max_seconds:
        argv += ["--max-seconds", f"{max_seconds:g}"]
    if stats_json:
        argv += ["--stats-json", str(stats_path_for(out))]
    return argv, out, warnings


# --------------------------------------------------------------- run / parse
PROG_RE = re.compile(
    r"\[mosaic\] (tracking forward|tracking reverse|rendering): (\d+)/(\d+) frames "
    r"\(([\d.]+)%\), elapsed ([\d:]+)(?:, ETA ([\d:]+))?")
PHASE_JA = {"tracking forward": "追跡（前方向）", "tracking reverse": "追跡（逆方向）",
            "rendering": "モザイク書き出し"}
EVENT_TAG = "[mosaic-event]"
KV_RE = re.compile(r"(\w+)=(\S*)")


def status_from_line(line, cur):
    """Update the status string from one log line; returns (status, pct or None)."""
    m = PROG_RE.search(line)
    if m:
        phase, done, total, pct, el, eta = m.groups()
        s = f"{PHASE_JA[phase]}: {done}/{total} フレーム（{pct}%） 経過 {el}"
        if eta:
            s += f" ／ 残り約 {eta}"
        return s, float(pct)
    if "extracting frames" in line:
        return "動画をフレームに分解中…", None
    if "models loaded" in line:
        return "モデル読み込み完了、追跡を開始…", None
    if "[mosaic] device:" in line:
        return f"起動しました（{line.split('device:')[1].strip()}）。準備中…", None
    if "tracking done" in line:
        return "追跡完了。モザイクを書き出し中…", None
    if "WARNING" in line:
        return cur + "  ⚠ " + line.split("[mosaic]")[-1].strip(), None
    return cur, None


def parse_event(line):
    """'[mosaic-event] type=lost frame=12 area=0 pass=forward' -> dict, else None.
    Integer-looking values become int."""
    i = line.find(EVENT_TAG)
    if i < 0:
        return None
    ev = {}
    for k, v in KV_RE.findall(line[i + len(EVENT_TAG):]):
        ev[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
    if "type" not in ev:
        return None
    return ev


def parse_ranges(s):
    """'0-2,5' -> [(0, 2), (5, 5)]; '-' or '' -> []."""
    out = []
    for part in str(s or "").split(","):
        part = part.strip()
        if not part or part == "-":
            continue
        a, _, b = part.partition("-")
        try:
            out.append((int(a), int(b or a)))
        except ValueError:
            continue
    return out


PROBLEM_TYPES = {"lost", "reacquire_failed", "empty_streak", "unmasked"}
ALERT_HEADERS = ["フレーム", "種類", "内容"]
PASS_JA = {"forward": "前方向", "reverse": "逆方向"}
MAX_RANGE_ROWS = 50


def _rng(a, b):
    a, b = min(a, b), max(a, b)
    return f"f{a}" if a == b else f"f{a}〜f{b}"


def event_rows(events):
    """Alert table rows [frame, kind, detail] from parsed events (sorted by frame).
    An `empty_streak state=ended` replaces the `ongoing` row of the same streak."""
    rows, streaks = [], {}
    for ev in events:
        t, f = ev.get("type"), ev.get("frame")
        ps = PASS_JA.get(ev.get("pass"), "")
        ps = f"（{ps}）" if ps else ""
        if t == "lost":
            rows.append([f, "⚠ 追跡ロスト", f"マスク面積 {ev.get('area', '?')}px{ps}"])
        elif t == "resumed":
            rows.append([f, "ℹ 追跡復帰",
                         f"{ev.get('lost_frames', '?')} フレームぶりに復帰"
                         f"（f{ev.get('lost_since', '?')} から外れていた）{ps}"])
        elif t == "reacquire_failed":
            rows.append([f, "⚠ 自動再捕捉に失敗", f"人物検出の候補で戻れませんでした（面積 "
                                            f"{ev.get('area', '?')}px）{ps}"])
        elif t == "reacquired":
            rows.append([f, "ℹ 自動再捕捉", f"人物検出で追跡し直しました（面積 {ev.get('area', '?')}px）{ps}"])
        elif t == "empty_streak":
            key = (f, ev.get("pass"))
            last = ev.get("last", f)
            ongoing = ev.get("state") == "ongoing"
            row = [f, "⚠ マスクなしが連続",
                   f"{_rng(f, last)}（{ev.get('length', '?')} フレーム"
                   f"{'以上、継続中' if ongoing else ''}）{ps}"]
            if key in streaks:
                rows[streaks[key]] = row
            else:
                streaks[key] = len(rows)
                rows.append(row)
        elif t == "hold":
            rows.append([f, "ℹ マスク保持",
                         f"{ev.get('held_frames', '?')} を f{ev.get('source', '?')} のマスクで補完"
                         + ("（上限で打ち切り）" if ev.get("expired") else "")])
        elif t == "unmasked":
            rngs = parse_ranges(ev.get("ranges"))
            for a, b in rngs[:MAX_RANGE_ROWS]:
                rows.append([a, "⚠ モザイクなし", f"{_rng(a, b)}（{b - a + 1} フレーム）にモザイクがかかっていません"])
            if len(rngs) > MAX_RANGE_ROWS:
                rows.append([rngs[MAX_RANGE_ROWS][0], "⚠ モザイクなし",
                             f"ほか {len(rngs) - MAX_RANGE_ROWS} 区間（ログ参照）"])
    rows.sort(key=lambda r: (r[0] if isinstance(r[0], int) else 10**9))
    return rows


def has_problem(events):
    return any(ev.get("type") in PROBLEM_TYPES and
               (ev.get("type") != "unmasked" or ev.get("count", 0) > 0) for ev in events)


def summarize(stats, events):
    """Post-run summary from the stats-json (+ the unmasked event, which carries the
    exact ranges). Returns (markdown, problem: bool)."""
    if not stats:
        return "", has_problem(events)
    n = int(stats.get("frames", 0))
    masked = int(stats.get("frames_masked", n))
    sev = stats.get("events", [])
    n_lost = sum(1 for e in sev if e.get("event") == "lost")
    n_reacq = sum(1 for e in sev if e.get("event") == "reacquire")
    n_rp = sum(1 for e in sev if e.get("event") == "reprompt")
    um = next((e for e in reversed(events) if e.get("type") == "unmasked"), None)
    rngs = parse_ranges(um.get("ranges")) if um else []
    lines = [f"**総括:** 全 {n} フレーム中、モザイクあり {masked} ／ **モザイクなし {n - masked}** フレーム"]
    if rngs:
        shown = "、".join(_rng(a, b) for a, b in rngs[:10])
        more = f" ほか {len(rngs) - 10} 区間" if len(rngs) > 10 else ""
        lines.append(f"- モザイクなしの範囲: {shown}{more}")
    lines.append(f"- 追跡ロスト: {n_lost} 回" + (f"（自動再捕捉 {n_reacq} 回）" if n_reacq else ""))
    if n_rp:
        lines.append(f"- 使った再指定: {n_rp} フレーム")
    hold = stats.get("hold_frames")
    if hold:
        lines.append(f"- マスク保持: {len(hold)} フレーム")
    t = stats.get("timing_s", {}).get("total")
    if t:
        lines.append(f"- 処理時間: {t / 60:.1f} 分")
    problem = (n - masked) > 0 or n_lost > 0 or has_problem(events)
    return "\n".join(lines), problem


def banner_html(kind, text):
    col = {"warn": ("#fff7e0", "#d97706"), "ok": ("#ecfdf5", "#059669"),
           "info": ("#eff6ff", "#3b82f6")}[kind]
    return (f'<div style="background:{col[0]};border-left:6px solid {col[1]};padding:8px 12px;'
            f'border-radius:4px;font-size:15px;">{text}</div>')


def progress_html(pct, label=""):
    pct = 0 if pct is None else max(0.0, min(100.0, pct))
    return (f'<div style="border:1px solid #888;border-radius:6px;height:22px;position:relative;">'
            f'<div style="background:#3b82f6;height:100%;width:{pct:.1f}%;border-radius:5px;"></div>'
            f'<div style="position:absolute;top:1px;width:100%;text-align:center;font-size:13px;">'
            f'{label} {pct:.1f}%</div></div>')


def _reader(stream, q):
    for raw in iter(stream.readline, b""):
        q.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
    stream.close()
    q.put(None)


def kill_tree(proc):
    if proc.poll() is not None:
        return
    if IS_WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True, **_no_window())
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_job(argv, out_path, session="default", poll=0.3):
    """Run mosaic.py; yields (log_tail, status, pct, output_path_or_None, finished, events).

    `events` is the list of parsed [mosaic-event] lines so far (stdout and stderr of
    the child are merged into one pipe)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    before = set(HERE.glob("mosaic_frames_*"))
    kw = _no_window() if IS_WIN else {"start_new_session": True}
    if "--stats-json" in argv:   # a stale stats file must not be summarised after a failed run
        sp = Path(argv[argv.index("--stats-json") + 1])
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.unlink(missing_ok=True)
    proc = subprocess.Popen(argv, cwd=str(HERE), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **kw)
    RUNNING[session] = proc
    q = queue.Queue()
    threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()
    lines, events = [], []
    status, pct, t0 = "起動中…（モデルの読み込みに少し時間がかかります）", 0.0, time.time()
    yield "", status, pct, None, False, events
    eof = False
    try:
        while not eof:
            changed = False
            try:
                while True:
                    line = q.get(timeout=poll)
                    if line is None:
                        eof = True
                        break
                    lines.append(line)
                    ev = parse_event(line)
                    if ev is not None:
                        events.append(ev)
                    else:
                        status, p = status_from_line(line, status)
                        pct = p if p is not None else pct
                    changed = True
            except queue.Empty:
                pass
            if changed:
                del lines[:-LOG_TAIL_LINES]
                yield "\n".join(lines), status, pct, None, False, events
        rc = proc.wait()
    finally:
        RUNNING.pop(session, None)
        if proc.poll() is None:
            kill_tree(proc)
    el = time.time() - t0
    if rc == 0 and out_path.exists():
        status, pct = f"完了しました（{el / 60:.1f} 分）: {out_path.name}", 100.0
        yield "\n".join(lines), status, pct, str(out_path), True, events
    else:
        for d in set(HERE.glob("mosaic_frames_*")) - before:   # leftovers of a stopped run
            shutil.rmtree(d, ignore_errors=True)
        stopped = getattr(proc, "_stopped_by_user", False)
        last = next((ln for ln in reversed(lines) if ln.strip()), "")
        status = "停止しました" if stopped else f"エラーで終了しました（終了コード {rc}）: {last}"
        yield "\n".join(lines), status, pct, None, True, events


def read_stats(argv):
    if "--stats-json" not in argv:
        return None
    try:
        with open(argv[argv.index("--stats-json") + 1], encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ UI logic
def fmt_info(info):
    d = info["duration"]
    return (f"**{info['name']}** — {info['width']}×{info['height']} px, "
            f"{info['fps']:.3f} fps, 長さ {int(d // 60)}:{d % 60:05.2f}（約 {info['nframes']} フレーム、"
            f"番号 0〜{info['nframes'] - 1}）")


def clamp_frame(info, idx):
    try:
        idx = int(round(float(idx or 0)))
    except (TypeError, ValueError):
        idx = 0
    return max(0, min(idx, info["nframes"] - 1))


def load_frame(info, idx, name, points, video=None):
    """Extract frame `idx` (clamped) of the input (or `video`) and draw `points`.
    Returns (idx, frame_path, rgb_image, label_markdown)."""
    if not info:
        raise gr.Error("先に動画を読み込んでください")
    idx = clamp_frame(info, idx)
    try:
        fp = extract_frame(video or info["path"], idx, info["fps"], WORK / f"{name}.png")
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"フレーム {idx} を取り出せませんでした: {e}")
    label = (f"**フレーム {idx}** ／ 最後 {info['nframes'] - 1}"
             f"（{idx / info['fps']:.3f} 秒）")
    return idx, str(fp), draw_points(fp, points), label


def video_info(path):
    """Probe + extract frame 0. Returns (info, frame0_path) or raises ValueError."""
    path = Path(path)
    try:
        info = probe_video(path)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"動画を読み込めませんでした: {e}")
    info.update(path=str(path), name=path.name)
    WORK.mkdir(exist_ok=True)
    try:
        fp = extract_frame(path, 0, info["fps"], WORK / "start.png")
        h, w = imread(fp).shape[:2]
        info["width"], info["height"] = w, h   # decoded (rotation-applied) size = click space
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"フレームを取り出せませんでした: {e}")
    return info, fp


def on_stop(request: gr.Request = None):
    session = getattr(request, "session_hash", None) or "default"
    proc = RUNNING.get(session)
    if proc is None:
        return "**状態:** 実行中の処理はありません"
    proc._stopped_by_user = True
    kill_tree(proc)
    return "**状態:** 停止しています…（点と再指定はそのまま残ります）"


def _sec(idx, fps):
    return round(idx / fps, 6)


# ------------------------------------------------------------------- layout
NAV_STEPS = [(-10, "◀◀ −10"), (-1, "◀ −1"), (1, "+1 ▶"), (10, "+10 ▶▶")]
FIX_NOTE = ("**修正は出力済み動画にモザイクを重ねる方式です。** 結果（_out.mp4 など）を入力にして、"
            "漏れたフレームで対象をクリックし直して実行すると、`元の名前_fix.mp4` が新しく作られます"
            "（入力ファイルは上書きしません）。")


def nav_row():
    """Frame stepping controls. Returns ([(delta, button)...], number, jump_button)."""
    with gr.Row(equal_height=True):
        btns = [(d, gr.Button(lbl, size="sm", min_width=70)) for d, lbl in NAV_STEPS]
        num = gr.Number(value=0, precision=0, minimum=0, label="フレーム番号", min_width=110)
        jump = gr.Button("ジャンプ", size="sm", min_width=70)
    return btns, num, jump


def build_ui():
    with gr.Blocks(title="動画モザイク", analytics_enabled=False) as demo:
        info = gr.State(None)
        start_fp, start_pts, start_idx = gr.State(None), gr.State([]), gr.State(0)
        rp_fp, rp_frame, rp_pts, reprompts = gr.State(None), gr.State(None), gr.State([]), gr.State([])
        last_out, events_st = gr.State(None), gr.State([])

        gr.Markdown("# 動画モザイク\n動画を読み込み、モザイクをかけたい対象を画像上でクリックして指定します。"
                    "すべてこのパソコンの中で処理されます。")

        gr.Markdown("## ① 動画を読み込む")
        video_in = gr.File(label="動画ファイル（ドラッグ＆ドロップ可）", file_types=["video"],
                           type="filepath")
        mode = gr.Radio([MODE_NEW, MODE_FIX], value=MODE_NEW, label="処理の種類")
        gr.Markdown(FIX_NOTE)
        info_md = gr.Markdown("動画を選んでください")

        gr.Markdown("## ② 開始フレームで対象をクリック")
        start_slider = gr.Slider(0, 1, value=0, step=0.04, label="開始時刻（秒）— 動かして離すとフレームが変わります")
        st_btns, st_num, st_jump = nav_row()
        start_label = gr.Markdown("")
        with gr.Row():
            with gr.Column(scale=3):
                start_img = gr.Image(label="クリックして点を追加", interactive=False,
                                     type="numpy", format="png", show_download_button=False,
                                     show_fullscreen_button=False)
            with gr.Column(scale=1):
                start_kind = gr.Radio([POS, NEG], value=POS, label="クリックで追加する点の種類")
                gr.Markdown("緑の点 = 対象、赤の × = 対象じゃない（例: 隣の人）")
                start_list = gr.Textbox(label="点の一覧", lines=6, interactive=False,
                                        value=points_text([], 0))
                with gr.Row():
                    start_undo = gr.Button("1つ戻す")
                    start_clear = gr.Button("全部消す")

        with gr.Accordion("③（任意）途中で対象を指定し直す — 手などで隠れる場面・漏れたフレームがある場合",
                          open=False) as rp_acc:
            gr.Markdown("追跡が外れた少し後（対象がまた見えている）フレームを表示し、対象をクリックして"
                        "「再指定として追加／更新」を押します。⑤のアラート一覧のフレーム番号をクリックしても"
                        "ここに表示されます。")
            with gr.Row():
                rp_slider = gr.Slider(0, 1, value=0, step=0.04, label="時刻（秒）")
                rp_show = gr.Button("このフレームを表示")
            rp_btns, rp_num, rp_jump = nav_row()
            rp_label = gr.Markdown("")
            with gr.Row():
                with gr.Column(scale=3):
                    rp_img = gr.Image(label="クリックして点を追加", interactive=False, type="numpy",
                                      format="png", show_download_button=False,
                                      show_fullscreen_button=False)
                with gr.Column(scale=1):
                    rp_kind = gr.Radio([POS, NEG], value=POS, label="点の種類")
                    rp_list = gr.Textbox(label="このフレームの点", lines=4, interactive=False,
                                         value=rp_points_text([], None, False))
                    with gr.Row():
                        rp_undo = gr.Button("1つ戻す")
                        rp_add = gr.Button("再指定として追加／更新", variant="primary")
                    rp_del = gr.Button("このフレームの再指定を削除")
                    rp_stored = gr.Dataframe(value=[], headers=RP_HEADERS, label="登録済みの再指定（行をクリックでそのフレームへ）",
                                             interactive=False, wrap=True)
                    rp_clear = gr.Button("再指定を全部消す")

        gr.Markdown("## ④ オプション")
        with gr.Row():
            dilate = gr.Slider(0, 30, value=6, step=1, label="モザイクの広げ幅（px）")
            block = gr.Slider(4, 40, value=12, step=1, label="モザイクの粗さ（ブロックの大きさ px）")
        with gr.Row():
            reacq = gr.Checkbox(value=True, label="見失ったら人物検出で探し直す")
            max_s = gr.Number(value=None, label="処理する秒数（空欄 = 動画全体）", minimum=0)
            hold = gr.Number(value=0, precision=0, minimum=0,
                             label="マスク保持フレーム数（0 = 無効。手などで一瞬隠れたとき、"
                                   "直前のモザイクをこのフレーム数まで残してチラつきを抑える）")
        out_md = gr.Markdown("")

        gr.Markdown("## ⑤ 実行")
        with gr.Row(equal_height=True):
            run_btn = gr.Button("モザイク処理を開始", variant="primary")
            prev_btn = gr.Button("プレビュー実行（冒頭だけ）")
            prev_s = gr.Number(value=30, minimum=1, label="プレビューの秒数", min_width=120)
            stop_btn = gr.Button("停止", variant="stop", interactive=False)
        gr.Markdown("おかしいと思ったら「停止」→ アラートのフレームで再指定を追加 → 再実行できます"
                    "（点と再指定は停止・完了後も残ります）。")
        status_md = gr.Markdown("**状態:** 待機中")
        prog = gr.HTML(progress_html(0))
        banner = gr.HTML("")
        alerts_df = gr.Dataframe(value=[], headers=ALERT_HEADERS, interactive=False, wrap=True,
                                 label="アラート（行をクリックすると ③ にそのフレームを表示）")
        summary_md = gr.Markdown("")
        with gr.Row():
            with gr.Column(scale=3):
                video_out = gr.Video(label="結果（プレビュー・ダウンロード）", interactive=False)
            with gr.Column(scale=2):
                result_frame = gr.Image(label="結果動画の同じフレーム（アラートをクリックしたとき）",
                                        interactive=False, type="numpy", format="png",
                                        show_download_button=False)
        fix_btn = gr.Button("この結果を追加修正する（結果を入力に読み込む）", interactive=False)
        log_box = gr.Textbox(label="ログ（最後の部分）", lines=10, max_lines=20, interactive=False,
                             autoscroll=True)

        # ------------------------------------------------------------ handlers
        def reset_outputs(inf, fp, out_text, msg=None):
            """Everything that changes when a new input video is loaded."""
            fps = inf["fps"] if inf else 1
            maxs = round(max(inf["duration"] - 1.0 / fps, 0.0), 3) if inf else 1
            step = round(1.0 / fps, 6) if inf else 0.04
            return {
                info: inf, info_md: msg or (fmt_info(inf) if inf else "動画を選んでください"),
                start_slider: gr.update(maximum=maxs, value=0, step=step),
                start_fp: str(fp) if fp else None, start_idx: 0,
                start_img: draw_points(fp, []) if fp else None, start_pts: [],
                start_list: points_text([], 0), st_num: 0,
                start_label: f"**フレーム 0** ／ 最後 {inf['nframes'] - 1}（0.000 秒）" if inf else "",
                rp_slider: gr.update(maximum=maxs, value=min(maxs, 1.0), step=step), rp_num: 0,
                rp_fp: None, rp_img: None, rp_frame: None, rp_pts: [], rp_label: "",
                rp_list: rp_points_text([], None, False), reprompts: [], rp_stored: [],
                out_md: out_text, video_out: None, result_frame: None,
                alerts_df: [], banner: "", summary_md: "", events_st: [],
            }

        def out_text_for(inf, md):
            if not inf:
                return ""
            return (f"出力ファイル: {output_path_for(inf['path'], md)}"
                    f"（プレビューは {output_path_for(inf['path'], md, preview=True).name}）")

        def load_path(path, md):
            if not path:
                return reset_outputs(None, None, "")
            try:
                inf, fp = video_info(path)
            except ValueError as e:
                return reset_outputs(None, None, "", msg=str(e))
            return reset_outputs(inf, fp, out_text_for(inf, md))

        def on_upload(file, md):
            return load_path(file if (file is None or isinstance(file, str)) else file.name, md)

        def on_mode(inf, md):
            return out_text_for(inf, md)

        def on_load_result(out):
            if not out or not Path(out).exists():
                raise gr.Error("読み込める結果がありません")
            d = load_path(out, MODE_FIX)
            d[mode] = MODE_FIX
            d[video_in] = out
            d[last_out] = out
            gr.Info("結果を追加修正の入力に読み込みました。漏れたフレームへ移動して対象をクリックしてください")
            return d

        # -- start frame
        def start_goto(inf, idx, points):
            idx, fp, img, label = load_frame(inf, idx, "start", points)
            return {start_idx: idx, start_fp: fp, start_img: img, start_label: label,
                    start_slider: _sec(idx, inf["fps"]), st_num: idx,
                    start_list: points_text(points, idx)}

        st_outs = [start_idx, start_fp, start_img, start_label, start_slider, st_num, start_list]

        def on_click(frame_path, points, kind, inf, evt: gr.SelectData):
            if not frame_path or not inf:
                raise gr.Error("先にフレームを表示してください")
            img = imread(frame_path)
            h, w = img.shape[:2]
            x, y = click_to_video_xy(evt.index, (w, h), (inf["width"], inf["height"]))
            return points + [(x, y, "pos" if kind == POS else "neg")]

        def start_click(frame_path, points, kind, inf, idx, evt: gr.SelectData):
            points = on_click(frame_path, points, kind, inf, evt)
            return points, draw_points(frame_path, points), points_text(points, idx)

        def start_undo_f(frame_path, points, idx):
            points = points[:-1]
            return points, (draw_points(frame_path, points) if frame_path else None), points_text(points, idx)

        def start_clear_f(frame_path, idx):
            return [], (draw_points(frame_path, []) if frame_path else None), points_text([], idx)

        # -- re-prompt frame
        def rp_goto(inf, idx, rps):
            idx = clamp_frame(inf, idx) if inf else 0
            stored = next((r for r in rps if r["frame"] == idx), None)
            pts = list(stored["points"]) if stored else []
            idx, fp, img, label = load_frame(inf, idx, "reprompt", pts)
            return {rp_frame: idx, rp_fp: fp, rp_img: img, rp_label: label,
                    rp_slider: _sec(idx, inf["fps"]), rp_num: idx, rp_pts: pts,
                    rp_list: rp_points_text(pts, idx, stored is not None)}

        rp_outs = [rp_frame, rp_fp, rp_img, rp_label, rp_slider, rp_num, rp_pts, rp_list]

        def rp_click(frame_path, points, kind, inf, idx, rps, evt: gr.SelectData):
            points = on_click(frame_path, points, kind, inf, evt)
            stored = any(r["frame"] == idx for r in rps)
            return points, draw_points(frame_path, points), rp_points_text(points, idx, stored)

        def rp_undo_f(frame_path, points, idx, rps):
            points = points[:-1]
            stored = any(r["frame"] == idx for r in rps)
            return (points, draw_points(frame_path, points) if frame_path else None,
                    rp_points_text(points, idx, stored))

        def rp_add_f(inf, idx, points, rps):
            if not inf or idx is None:
                raise gr.Error("先に「このフレームを表示」かコマ送りでフレームを選んでください")
            if not any(k == "pos" for _, _, k in points):
                raise gr.Error("「対象」の点を 1 つ以上クリックしてください")
            rps = sorted([r for r in rps if r["frame"] != idx] +
                         [{"frame": int(idx), "points": list(points)}], key=lambda r: r["frame"])
            gr.Info(f"フレーム {idx} を再指定として登録しました")
            return rps, reprompt_rows(rps, inf["fps"]), rp_points_text(points, idx, True)

        def rp_del_f(inf, idx, points, rps):
            if idx is None or not any(r["frame"] == idx for r in rps):
                raise gr.Error("このフレームには再指定が登録されていません")
            rps = [r for r in rps if r["frame"] != idx]
            return rps, reprompt_rows(rps, inf["fps"]), rp_points_text(points, idx, False)

        def rp_clear_f():
            return [], [], gr.update()

        def frame_from_row(evt):
            row = getattr(evt, "row_value", None) or []
            try:
                return int(float(row[0]))
            except (IndexError, TypeError, ValueError):
                return None

        def jump_to(inf, idx, rps, out):
            """Alert/summary/stored-row click: show frame idx in ③ (+ the result's frame)."""
            d = rp_goto(inf, idx, rps)
            d[rp_acc] = gr.Accordion(open=True)
            res = None
            if out and Path(out).exists():
                try:
                    res = cv2.cvtColor(imread(extract_frame(out, d[rp_frame], inf["fps"],
                                                            WORK / "result.png")), cv2.COLOR_BGR2RGB)
                except Exception:  # noqa: BLE001  (e.g. preview output shorter than the input)
                    res = None
            d[result_frame] = res
            return d

        def on_alert_select(inf, rps, out, evt: gr.SelectData):
            idx = frame_from_row(evt)
            if idx is None or not inf:
                return {}
            return jump_to(inf, idx, rps, out)

        # -- run
        def run_impl(preview, inf, md, sidx, points, rps, dil, blk, rq, mx, hd, pvs, request):
            def ui(log, status, pct, video=None, running=True, events=(), extra=None):
                events = list(events)
                rows = event_rows(events)
                if running:
                    bn = (banner_html("warn", "⚠ 追跡に問題が出ています（下のアラート参照）。"
                                              "おかしければ「停止」→ そのフレームで再指定を追加 → 再実行できます。")
                          if has_problem(events) else "")
                    summ = ""
                else:
                    summ, bn = extra or ("", "")
                return {log_box: log, status_md: f"**状態:** {status}", prog: progress_html(pct),
                        video_out: video, run_btn: gr.update(interactive=not running),
                        prev_btn: gr.update(interactive=not running),
                        stop_btn: gr.update(interactive=running), alerts_df: rows,
                        banner: bn, summary_md: summ, events_st: events,
                        last_out: video if video else gr.update(),
                        fix_btn: gr.update(interactive=bool(video)) if not running else gr.update()}

            if not inf:
                raise gr.Error("先に動画を読み込んでください")
            try:
                argv, out, warns = build_command(
                    inf["path"], inf["fps"], None, points, rps, dil, blk, rq, mx,
                    nframes=inf["nframes"], hold_frames=hd, mode=md,
                    preview_seconds=(pvs or 30) if preview else None, start_frame=sidx)
            except ValueError as e:
                raise gr.Error(str(e))
            for w in warns:
                gr.Warning(w)
            session = getattr(request, "session_hash", None) or "default"
            if session in RUNNING:
                raise gr.Error("すでに処理中です")
            head = ("【プレビュー実行】" if preview else "") + "実行コマンド:\n" + \
                subprocess.list2cmdline(argv) + "\n\n"
            for log, status, pct, video, finished, events in run_job(argv, out, session):
                extra = None
                if finished:
                    summ, problem = summarize(read_stats(argv), events)
                    if video and problem:
                        bn = banner_html("warn", "⚠ モザイクが漏れた／追跡が外れた可能性のあるフレームがあります。"
                                                 "アラートの行をクリックして確認し、③で再指定を追加して再実行するか、"
                                                 "「この結果を追加修正する」で直せます。")
                    elif video:
                        bn = banner_html("ok", "✓ 問題のあるフレームは検出されませんでした（念のため再生して確認してください）")
                    elif has_problem(events):
                        bn = banner_html("warn", "⚠ 停止（または終了）前に問題が出ていました。アラートの行をクリックして"
                                                 "そのフレームで再指定を追加し、再実行できます。")
                    else:
                        bn = ""
                    if preview and video:
                        summ = (summ + "\n\n" if summ else "") + \
                            f"（プレビュー: 冒頭 {float(pvs or 30):g} 秒だけの結果です）"
                    extra = (summ, bn)
                yield ui(head + log, status, pct, video, running=not finished, events=events, extra=extra)

        run_outs = [log_box, status_md, prog, video_out, run_btn, prev_btn, stop_btn, alerts_df,
                    banner, summary_md, events_st, last_out, fix_btn]
        run_ins = [info, mode, start_idx, start_pts, reprompts, dilate, block, reacq, max_s, hold, prev_s]

        def on_run(inf, md, sidx, points, rps, dil, blk, rq, mx, hd, pvs, request: gr.Request = None):
            yield from run_impl(False, inf, md, sidx, points, rps, dil, blk, rq, mx, hd, pvs, request)

        def on_preview(inf, md, sidx, points, rps, dil, blk, rq, mx, hd, pvs, request: gr.Request = None):
            yield from run_impl(True, inf, md, sidx, points, rps, dil, blk, rq, mx, hd, pvs, request)

        # ------------------------------------------------------------ events
        load_outs = [info, info_md, start_slider, start_fp, start_idx, start_img, start_pts,
                     start_list, st_num, start_label, rp_slider, rp_num, rp_fp, rp_img, rp_frame,
                     rp_pts, rp_label, rp_list, reprompts, rp_stored, out_md, video_out,
                     result_frame, alerts_df, banner, summary_md, events_st]
        video_in.upload(on_upload, [video_in, mode], load_outs)
        video_in.clear(lambda md: load_path(None, md), mode, load_outs)
        mode.change(on_mode, [info, mode], out_md)
        fix_btn.click(on_load_result, last_out, load_outs + [mode, video_in, last_out])

        start_slider.release(lambda inf, s, p: start_goto(inf, frame_index(s, inf["fps"]) if inf else 0, p),
                             [info, start_slider, start_pts], st_outs)
        for d, b in st_btns:
            b.click(lambda inf, cur, p, d=d: start_goto(inf, (cur or 0) + d, p),
                    [info, start_idx, start_pts], st_outs)
        st_jump.click(start_goto, [info, st_num, start_pts], st_outs)
        st_num.submit(start_goto, [info, st_num, start_pts], st_outs)
        start_img.select(start_click, [start_fp, start_pts, start_kind, info, start_idx],
                         [start_pts, start_img, start_list])
        start_undo.click(start_undo_f, [start_fp, start_pts, start_idx], [start_pts, start_img, start_list])
        start_clear.click(start_clear_f, [start_fp, start_idx], [start_pts, start_img, start_list])

        rp_show.click(lambda inf, s, r: rp_goto(inf, frame_index(s, inf["fps"]) if inf else 0, r),
                      [info, rp_slider, reprompts], rp_outs)
        rp_slider.release(lambda inf, s, r: rp_goto(inf, frame_index(s, inf["fps"]) if inf else 0, r),
                          [info, rp_slider, reprompts], rp_outs)
        for d, b in rp_btns:
            b.click(lambda inf, cur, r, d=d: rp_goto(inf, (cur or 0) + d, r),
                    [info, rp_frame, reprompts], rp_outs)
        rp_jump.click(rp_goto, [info, rp_num, reprompts], rp_outs)
        rp_num.submit(rp_goto, [info, rp_num, reprompts], rp_outs)
        rp_img.select(rp_click, [rp_fp, rp_pts, rp_kind, info, rp_frame, reprompts],
                      [rp_pts, rp_img, rp_list])
        rp_undo.click(rp_undo_f, [rp_fp, rp_pts, rp_frame, reprompts], [rp_pts, rp_img, rp_list])
        rp_add.click(rp_add_f, [info, rp_frame, rp_pts, reprompts], [reprompts, rp_stored, rp_list])
        rp_del.click(rp_del_f, [info, rp_frame, rp_pts, reprompts], [reprompts, rp_stored, rp_list])
        rp_clear.click(rp_clear_f, None, [reprompts, rp_stored, rp_list])
        jump_outs = rp_outs + [rp_acc, result_frame]
        rp_stored.select(on_alert_select, [info, reprompts, last_out], jump_outs)
        alerts_df.select(on_alert_select, [info, reprompts, last_out], jump_outs)

        run_btn.click(on_run, run_ins, run_outs, concurrency_limit=1, concurrency_id="mosaic_run")
        prev_btn.click(on_preview, run_ins, run_outs, concurrency_limit=1, concurrency_id="mosaic_run")
        stop_btn.click(on_stop, None, status_md, concurrency_limit=None)
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, allowed_paths=[str(HERE)])
