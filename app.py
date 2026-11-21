#!/usr/bin/env python3
"""モザイク GUI — a Gradio front-end for mosaic.py.

The app never imports torch / sam2. It only calls ffprobe / ffmpeg (to inspect
the video and grab single frames for clicking) and runs mosaic.py as a
subprocess with the same Python interpreter (sys.executable).

Tested with gradio 5.50.0.

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


def points_text(points):
    if not points:
        return "（まだ点がありません。画像をクリックしてください）"
    return "\n".join(f"{i}. {'対象' if k == 'pos' else '対象じゃない'}  x={x}, y={y}"
                     for i, (x, y, k) in enumerate(points, 1))


def reprompts_text(reprompts, fps):
    if not reprompts:
        return "（再指定はまだありません）"
    lines = []
    for rp in reprompts:
        pts = ", ".join(f"{'+' if k == 'pos' else '−'}({x},{y})" for x, y, k in rp["points"])
        lines.append(f"フレーム {rp['frame']}（{rp['frame'] / fps:.2f} 秒）: {pts}")
    return "\n".join(lines)


# ------------------------------------------------------------- command builder
def output_path_for(video_path):
    return HERE / f"{Path(video_path).stem}_out.mp4"


def build_command(video_path, fps, start_seconds, points, reprompts, dilate, block,
                  detect_reacquire, max_seconds, nframes=None, python=None):
    """Return (argv, output_path, warnings). Pure function: no I/O.

    points:    [(x, y, "pos"|"neg"), ...] on the start frame
    reprompts: [{"frame": int, "points": [(x, y, kind), ...]}, ...]
    """
    warnings = []
    pos = [(x, y) for x, y, k in points if k == "pos"]
    neg = [(x, y) for x, y, k in points if k == "neg"]
    if not pos:
        raise ValueError("開始フレームに「対象」の点を 1 つ以上クリックしてください")
    start = frame_index(start_seconds, fps)
    limit = nframes
    if max_seconds not in (None, "") and float(max_seconds) > 0:
        max_seconds = float(max_seconds)
        limit = int(max_seconds * fps) if limit is None else min(limit, int(max_seconds * fps))
    else:
        max_seconds = None
    if limit is not None and start >= max(limit, 1):
        raise ValueError(f"開始フレーム {start} が処理範囲（{limit} フレーム）の外です。"
                         "開始時刻を前にするか「処理する秒数」を増やしてください")
    out = output_path_for(video_path)
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
    if max_seconds:
        argv += ["--max-seconds", f"{max_seconds:g}"]
    return argv, out, warnings


# --------------------------------------------------------------- run / parse
PROG_RE = re.compile(
    r"\[mosaic\] (tracking forward|tracking reverse|rendering): (\d+)/(\d+) frames "
    r"\(([\d.]+)%\), elapsed ([\d:]+)(?:, ETA ([\d:]+))?")
PHASE_JA = {"tracking forward": "追跡（前方向）", "tracking reverse": "追跡（逆方向）",
            "rendering": "モザイク書き出し"}


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
    """Run mosaic.py; yields (log_tail, status, pct, output_path_or_None, finished)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    before = set(HERE.glob("mosaic_frames_*"))
    kw = _no_window() if IS_WIN else {"start_new_session": True}
    proc = subprocess.Popen(argv, cwd=str(HERE), stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **kw)
    RUNNING[session] = proc
    q = queue.Queue()
    threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()
    lines, status, pct, t0 = [], "起動中…（モデルの読み込みに少し時間がかかります）", 0.0, time.time()
    yield "", status, pct, None, False
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
                    status, p = status_from_line(line, status)
                    pct = p if p is not None else pct
                    changed = True
            except queue.Empty:
                pass
            if changed:
                del lines[:-LOG_TAIL_LINES]
                yield "\n".join(lines), status, pct, None, False
        rc = proc.wait()
    finally:
        RUNNING.pop(session, None)
        if proc.poll() is None:
            kill_tree(proc)
    el = time.time() - t0
    if rc == 0 and out_path.exists():
        status, pct = f"完了しました（{el / 60:.1f} 分）: {out_path.name}", 100.0
        yield "\n".join(lines), status, pct, str(out_path), True
    else:
        for d in set(HERE.glob("mosaic_frames_*")) - before:   # leftovers of a stopped run
            shutil.rmtree(d, ignore_errors=True)
        stopped = getattr(proc, "_stopped_by_user", False)
        last = next((ln for ln in reversed(lines) if ln.strip()), "")
        status = "停止しました" if stopped else f"エラーで終了しました（終了コード {rc}）: {last}"
        yield "\n".join(lines), status, pct, None, True


# ------------------------------------------------------------------ UI logic
def fmt_info(info):
    d = info["duration"]
    return (f"**{info['name']}** — {info['width']}×{info['height']} px, "
            f"{info['fps']:.3f} fps, 長さ {int(d // 60)}:{d % 60:05.2f}（約 {info['nframes']} フレーム）")


def on_upload(file):
    empty = (None, "動画を選んでください", gr.update(maximum=1, value=0), None, None, [],
             points_text([]), gr.update(maximum=1, value=0), None, None, [], points_text([]),
             [], reprompts_text([], 1), "", gr.update(value=None))
    if not file:
        return empty
    path = Path(file if isinstance(file, str) else file.name)
    try:
        info = probe_video(path)
    except Exception as e:  # noqa: BLE001
        return (None, f"動画を読み込めませんでした: {e}",) + empty[2:]
    info.update(path=str(path), name=path.name)
    WORK.mkdir(exist_ok=True)
    try:
        fp = extract_frame(path, 0, info["fps"], WORK / "start.png")
        h, w = imread(fp).shape[:2]
        info["width"], info["height"] = w, h   # decoded (rotation-applied) size = click space
    except Exception as e:  # noqa: BLE001
        return (None, f"フレームを取り出せませんでした: {e}",) + empty[2:]
    maxs = max(info["duration"] - 1.0 / info["fps"], 0.0)
    step = round(1.0 / info["fps"], 6)
    return (info, fmt_info(info), gr.update(maximum=round(maxs, 3), value=0, step=step),
            str(fp), draw_points(fp, []), [], points_text([]),
            gr.update(maximum=round(maxs, 3), value=min(maxs, 1.0), step=step),
            None, None, [], points_text([]), [], reprompts_text([], info["fps"]),
            f"出力ファイル: {output_path_for(path)}", gr.update(value=None))


def show_frame(info, seconds, points, name):
    if not info:
        raise gr.Error("先に動画を読み込んでください")
    idx = frame_index(seconds, info["fps"])
    fp = extract_frame(info["path"], idx, info["fps"], WORK / f"{name}.png")
    return str(fp), draw_points(fp, points), f"フレーム {idx}（{idx / info['fps']:.3f} 秒）"


def on_start_time(info, seconds, points):
    return show_frame(info, seconds, points, "start")


def on_rp_time(info, seconds):
    fp, img, label = show_frame(info, seconds, [], "reprompt")
    return fp, img, label, frame_index(seconds, info["fps"]), [], points_text([])


def on_click(frame_path, points, kind, info, evt: gr.SelectData):
    if not frame_path or not info:
        raise gr.Error("先にフレームを表示してください")
    img = imread(frame_path)
    h, w = img.shape[:2]
    x, y = click_to_video_xy(evt.index, (w, h), (info["width"], info["height"]))
    points = points + [(x, y, "pos" if kind == POS else "neg")]
    return points, draw_points(frame_path, points), points_text(points)


def on_undo(frame_path, points):
    points = points[:-1]
    return points, (draw_points(frame_path, points) if frame_path else None), points_text(points)


def on_clear(frame_path):
    return [], (draw_points(frame_path, []) if frame_path else None), points_text([])


def on_add_reprompt(info, rp_frame, rp_points, reprompts, rp_frame_path):
    if not info or rp_frame is None:
        raise gr.Error("先に「このフレームを表示」を押してください")
    if not any(k == "pos" for _, _, k in rp_points):
        raise gr.Error("「対象」の点を 1 つ以上クリックしてください")
    reprompts = [r for r in reprompts if r["frame"] != rp_frame] + \
        [{"frame": int(rp_frame), "points": list(rp_points)}]
    reprompts.sort(key=lambda r: r["frame"])
    gr.Info(f"フレーム {rp_frame} を再指定として追加しました")
    return (reprompts, reprompts_text(reprompts, info["fps"]), [],
            draw_points(rp_frame_path, []) if rp_frame_path else None, points_text([]))


def on_clear_reprompts(info):
    return [], reprompts_text([], info["fps"] if info else 1)


def on_run(info, start_s, points, reprompts, dilate, block, reacq, max_s,
           request: gr.Request = None):
    def ui(log, status, pct, video=None, running=True):
        return (log, f"**状態:** {status}", progress_html(pct), video,
                gr.update(interactive=not running), gr.update(interactive=running))

    if not info:
        raise gr.Error("先に動画を読み込んでください")
    try:
        argv, out, warns = build_command(
            info["path"], info["fps"], start_s, points, reprompts, dilate, block, reacq,
            max_s, nframes=info["nframes"])
    except ValueError as e:
        raise gr.Error(str(e))
    for w in warns:
        gr.Warning(w)
    session = getattr(request, "session_hash", None) or "default"
    if session in RUNNING:
        raise gr.Error("すでに処理中です")
    head = "実行コマンド:\n" + subprocess.list2cmdline(argv) + "\n\n"
    for log, status, pct, video, finished in run_job(argv, out, session):
        yield ui(head + log, status, pct, video, running=not finished)


def on_stop(request: gr.Request = None):
    session = getattr(request, "session_hash", None) or "default"
    proc = RUNNING.get(session)
    if proc is None:
        return "**状態:** 実行中の処理はありません"
    proc._stopped_by_user = True
    kill_tree(proc)
    return "**状態:** 停止しています…"


# ------------------------------------------------------------------- layout
def build_ui():
    with gr.Blocks(title="動画モザイク", analytics_enabled=False) as demo:
        info = gr.State(None)
        start_fp, start_pts = gr.State(None), gr.State([])
        rp_fp, rp_frame, rp_pts, reprompts = gr.State(None), gr.State(None), gr.State([]), gr.State([])

        gr.Markdown("# 動画モザイク\n動画を読み込み、モザイクをかけたい対象を画像上でクリックして指定します。"
                    "すべてこのパソコンの中で処理されます。")

        gr.Markdown("## ① 動画を読み込む")
        video_in = gr.File(label="動画ファイル（ドラッグ＆ドロップ可）", file_types=["video"],
                           type="filepath")
        info_md = gr.Markdown("動画を選んでください")

        gr.Markdown("## ② 開始フレームで対象をクリック")
        start_slider = gr.Slider(0, 1, value=0, step=0.04, label="開始時刻（秒）— 動かして離すとフレームが変わります")
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
                                        value=points_text([]))
                with gr.Row():
                    start_undo = gr.Button("1つ戻す")
                    start_clear = gr.Button("全部消す")

        with gr.Accordion("③（任意）途中で対象を指定し直す — 手などで隠れる場面がある場合", open=False):
            gr.Markdown("追跡が外れそうな時刻のフレームを表示し、対象をクリックして「再指定として追加」を押します。")
            with gr.Row():
                rp_slider = gr.Slider(0, 1, value=0, step=0.04, label="時刻（秒）")
                rp_show = gr.Button("このフレームを表示")
            rp_label = gr.Markdown("")
            with gr.Row():
                with gr.Column(scale=3):
                    rp_img = gr.Image(label="クリックして点を追加", interactive=False, type="numpy",
                                      format="png", show_download_button=False,
                                      show_fullscreen_button=False)
                with gr.Column(scale=1):
                    rp_kind = gr.Radio([POS, NEG], value=POS, label="点の種類")
                    rp_list = gr.Textbox(label="このフレームの点", lines=4, interactive=False,
                                         value=points_text([]))
                    with gr.Row():
                        rp_undo = gr.Button("1つ戻す")
                        rp_add = gr.Button("再指定として追加", variant="primary")
                    rp_stored = gr.Textbox(label="登録済みの再指定", lines=4, interactive=False,
                                           value=reprompts_text([], 1))
                    rp_clear = gr.Button("再指定を全部消す")

        gr.Markdown("## ④ オプション")
        with gr.Row():
            dilate = gr.Slider(0, 30, value=6, step=1, label="モザイクの広げ幅（px）")
            block = gr.Slider(4, 40, value=12, step=1, label="モザイクの粗さ（ブロックの大きさ px）")
        with gr.Row():
            reacq = gr.Checkbox(value=True, label="見失ったら人物検出で探し直す")
            max_s = gr.Number(value=None, label="処理する秒数（空欄 = 動画全体、試しに短く処理したいとき用）",
                              minimum=0)
        out_md = gr.Markdown("")

        gr.Markdown("## ⑤ 実行")
        with gr.Row():
            run_btn = gr.Button("モザイク処理を開始", variant="primary")
            stop_btn = gr.Button("停止", variant="stop", interactive=False)
        status_md = gr.Markdown("**状態:** 待機中")
        prog = gr.HTML(progress_html(0))
        video_out = gr.Video(label="結果（プレビュー・ダウンロード）", interactive=False)
        log_box = gr.Textbox(label="ログ（最後の部分）", lines=10, max_lines=20, interactive=False,
                             autoscroll=True)

        # ---- events
        video_in.change(on_upload, video_in,
                        [info, info_md, start_slider, start_fp, start_img, start_pts, start_list,
                         rp_slider, rp_fp, rp_img, rp_pts, rp_list, reprompts, rp_stored, out_md,
                         video_out])
        start_slider.release(on_start_time, [info, start_slider, start_pts],
                             [start_fp, start_img, start_label])
        start_img.select(on_click, [start_fp, start_pts, start_kind, info],
                         [start_pts, start_img, start_list])
        start_undo.click(on_undo, [start_fp, start_pts], [start_pts, start_img, start_list])
        start_clear.click(on_clear, start_fp, [start_pts, start_img, start_list])

        rp_show.click(on_rp_time, [info, rp_slider], [rp_fp, rp_img, rp_label, rp_frame, rp_pts, rp_list])
        rp_img.select(on_click, [rp_fp, rp_pts, rp_kind, info], [rp_pts, rp_img, rp_list])
        rp_undo.click(on_undo, [rp_fp, rp_pts], [rp_pts, rp_img, rp_list])
        rp_add.click(on_add_reprompt, [info, rp_frame, rp_pts, reprompts, rp_fp],
                     [reprompts, rp_stored, rp_pts, rp_img, rp_list])
        rp_clear.click(on_clear_reprompts, info, [reprompts, rp_stored])

        run_btn.click(on_run, [info, start_slider, start_pts, reprompts, dilate, block, reacq, max_s],
                      [log_box, status_md, prog, video_out, run_btn, stop_btn],
                      concurrency_limit=1)
        stop_btn.click(on_stop, None, status_md, concurrency_limit=None)
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, allowed_paths=[str(HERE)])
