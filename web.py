"""统一视频工具平台 — Flask Web 界面
  页面1: 哔哩哔哩下载 (/)    页面2: 视频处理 (/process)
  启动: python web.py
"""

import sys
import os
import re
import json
import time
import glob
import random
import socket
import queue
import logging
import traceback
import threading
from pathlib import Path

from flask import Flask, Response, request, jsonify, render_template

from modules.bilibili.config import (
    PROJECT_ROOT, COOKIE_FILE, get_output_dir, set_output_dir as _set_output_dir, _get,
)
from modules.bilibili.utils import is_playlist_url, format_size, format_speed
from modules.bilibili.cookies import cookie_file_exists, cookie_has_sessdata
from modules.bilibili.downloader import BilibiliDownloader

# ── 项目根目录（web.py 就在根目录）──
PROJECT_ROOT_PATH = Path(__file__).resolve().parent


# ═══════════════════════════════════════
# 日志队列处理器
# ═══════════════════════════════════════
class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            self.log_queue.put(("log", record.levelname, self.format(record)))
        except Exception:
            pass


# ═══════════════════════════════════════
# Flask App
# ═══════════════════════════════════════
_WEB_DIR = PROJECT_ROOT_PATH / "web"
app = Flask(__name__, template_folder=str(_WEB_DIR),
            static_folder=str(_WEB_DIR), static_url_path="/static")

_event_queue = queue.Queue()       # SSE 事件队列
_cancelled = False
_running = False
_worker: threading.Thread = None   # 下载线程
_worker_process = None             # 视频处理子进程
_worker_bridge: threading.Thread = None

_log_handler = QueueLogHandler(_event_queue)
logging.getLogger().addHandler(_log_handler)
logging.getLogger().setLevel(logging.INFO)


# ═══════════════════════════════════════
# 路由：页面
# ═══════════════════════════════════════
@app.route("/")
def index():
    return render_template(
        "index.html",
        default_url="",
        default_dir="",
        cookie_ok=cookie_file_exists(),
    )


@app.route("/process")
def process_page():
    return render_template(
        "process.html",
        cookie_ok=cookie_file_exists(),
    )


# ═══════════════════════════════════════
# 路由：目录浏览（共享）
# ═══════════════════════════════════════
@app.route("/api/browse")
def browse_directory():
    path = request.args.get("path", "")
    if not path:
        if os.name == "nt":
            import string
            roots = [d + ":\\" for d in string.ascii_uppercase if os.path.exists(d + ":\\")]
            return jsonify({"path": "", "parent": None, "dirs": [], "roots": roots})
        else:
            roots = [str(Path.home())]
            volumes_dir = Path("/Volumes")
            if volumes_dir.exists():
                try:
                    for v in sorted(volumes_dir.iterdir()):
                        if v.is_dir() and not v.name.startswith("."):
                            roots.append(str(v))
                except PermissionError:
                    pass
            roots.append("/")
            return jsonify({"path": "", "parent": None, "dirs": roots, "roots": []})

    p = Path(path).resolve()
    if not p.exists() or not p.is_dir():
        return jsonify({"path": str(p), "parent": None, "dirs": [], "error": "目录不存在"})

    try:
        dirs = sorted([
            d.name for d in p.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
    except PermissionError:
        dirs = []

    parent = str(p.parent) if str(p) != str(p.parent) else None
    return jsonify({"path": str(p), "parent": parent, "dirs": dirs, "roots": []})


# ═══════════════════════════════════════
# 路由：扫描目录中的视频文件
# ═══════════════════════════════════════
VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.wmv')


@app.route("/api/process/scan")
def scan_videos():
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"files": [], "count": 0, "error": "请选择目录"})
    p = Path(path)
    if not p.exists():
        return jsonify({"files": [], "count": 0, "error": "目录不存在"})
    if p.is_file():
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            return jsonify({"files": [{"name": p.name, "path": str(p)}], "count": 1})
        return jsonify({"files": [], "count": 0, "error": "不是视频文件"})

    files = []
    for ext in VIDEO_EXTENSIONS:
        for f in p.rglob(f"*{ext}"):
            files.append({"name": f.name, "path": str(f)})
    files.sort(key=lambda x: x["name"])
    return jsonify({"files": files, "count": len(files)})


# ═══════════════════════════════════════
# 路由：获取/保存 视频处理配置
# ═══════════════════════════════════════
def _load_processor_config():
    import yaml
    cfg_path = PROJECT_ROOT_PATH / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


@app.route("/api/process/config")
def get_process_config():
    cfg = _load_processor_config()
    return jsonify({
        "asr_device": cfg.get("asr", {}).get("device", "cpu"),
        "asr_model": cfg.get("asr", {}).get("model_size", "base"),
        "sample_rate": cfg.get("audio", {}).get("sample_rate", 16000),
        "tts_voice": cfg.get("tts", {}).get("voice", "zh-CN-XiaoxiaoNeural"),
        "max_workers": cfg.get("global", {}).get("max_concurrency", {}).get("video_processor", 1),
    })


@app.route("/api/process/config", methods=["POST"])
def save_process_config():
    import yaml
    data = request.get_json(force=True)
    cfg_path = PROJECT_ROOT_PATH / "config" / "config.yaml"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}

    cfg.setdefault("asr", {})
    cfg.setdefault("audio", {})
    cfg.setdefault("tts", {})
    cfg.setdefault("global", {}).setdefault("max_concurrency", {})

    if "asr_device" in data:
        cfg["asr"]["device"] = data["asr_device"]
    if "asr_model" in data:
        cfg["asr"]["model_size"] = data["asr_model"]
    if "sample_rate" in data:
        cfg["audio"]["sample_rate"] = data["sample_rate"]
    if "tts_voice" in data:
        cfg["tts"]["voice"] = data["tts_voice"]

    cfg["global"]["max_concurrency"]["video_processor"] = 1  # 串行处理

    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    return jsonify({"ok": True, "message": "配置已保存"})


# ═══════════════════════════════════════
# 路由：Cookie
# ═══════════════════════════════════════
@app.route("/api/cookie/status")
def cookie_status():
    ok = cookie_file_exists()
    has_sess = cookie_has_sessdata() if ok else False
    return jsonify({"ok": ok, "has_sessdata": has_sess, "path": str(COOKIE_FILE)})


@app.route("/api/cookie/upload", methods=["POST"])
def cookie_upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "未选择文件"}), 400
    file = request.files["file"]
    if not file.filename or not file.filename.endswith(".txt"):
        return jsonify({"ok": False, "error": "仅支持 .txt 格式"}), 400
    try:
        content = file.read()
        if len(content) < 100:
            return jsonify({"ok": False, "error": "文件内容过短"}), 400
        COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
        COOKIE_FILE.write_bytes(content)
        ok = cookie_has_sessdata()
        return jsonify({"ok": ok, "has_sessdata": ok, "path": str(COOKIE_FILE),
                        "message": "Cookie 导入成功！" if ok else "Cookie 已保存，但未检测到 SESSDATA"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"保存失败: {e}"}), 500


# ═══════════════════════════════════════
# 路由：SSE 事件流
# ═══════════════════════════════════════
@app.route("/api/stream")
def stream():
    def generate():
        try:
            while True:
                try:
                    msg = _event_queue.get(timeout=1)
                    msg_type = msg[0]
                    if msg_type == "log":
                        yield f"event: log\ndata: {json.dumps({'level': msg[1], 'message': msg[2]}, ensure_ascii=False)}\n\n"
                    elif msg_type == "progress":
                        yield f"event: progress\ndata: {json.dumps(msg[1], ensure_ascii=False)}\n\n"
                    elif msg_type == "video_start":
                        data = {"index": msg[1], "total": msg[2], "title": msg[3]}
                        yield f"event: video_start\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    elif msg_type == "video_done":
                        data = {"success": msg[1], "index": msg[2], "total": msg[3]}
                        yield f"event: video_done\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    elif msg_type == "process_step":
                        data = {"step": msg[1], "total": msg[2], "label": msg[3]}
                        yield f"event: process_step\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    elif msg_type == "all_done":
                        data = {"success_count": msg[1], "total": msg[2],
                                "fail_list": msg[3], "output_dir": msg[4],
                                "elapsed": msg[5] if len(msg) > 5 else 0}
                        yield f"event: all_done\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════
# 路由：下载控制（哔哩哔哩）
# ═══════════════════════════════════════
@app.route("/api/download", methods=["POST"])
def start_download():
    global _cancelled, _running, _worker
    if _running:
        return jsonify({"error": "任务正在进行中"}), 409

    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    out_dir = data.get("out_dir", "").strip()
    quality = data.get("quality", "best")
    embed_subs = data.get("embed_subs", False)
    # 下载模式: full / subs_only / video_only
    download_mode = data.get("download_mode", "full")

    if not url:
        return jsonify({"error": "请输入视频链接"}), 400
    if not out_dir:
        return jsonify({"error": "请指定输出目录"}), 400

    is_playlist = is_playlist_url(url)

    _set_output_dir(Path(out_dir))
    _cancelled = False
    _running = True

    def _put_log(level, msg):
        _event_queue.put(("log", level, msg))

    def _put_progress(data):
        _event_queue.put(("progress", data))

    def _put_video_start(index, total, title):
        _event_queue.put(("video_start", index, total, title))

    def _put_video_done(success, index, total):
        _event_queue.put(("video_done", success, index, total))

    def _put_all_done(success_count, total, fail_list, output_dir, elapsed=0):
        _event_queue.put(("all_done", success_count, total, fail_list, output_dir, elapsed))

    _put_log("HEADER", "=" * 60)
    _put_log("HEADER", f"🎬 开始下载: {url}")
    _put_log("HEADER", f"📁 输出目录: {out_dir}")
    if download_mode == "subs_only":
        _put_log("HEADER", f"📝 模式: 仅下载字幕")
    elif download_mode == "video_only":
        _put_log("HEADER", f"🎨 画质: {quality}  |  模式: 仅下载视频")
    else:
        _put_log("HEADER", f"🎨 画质: {quality}  |  模式: 完整下载")
    _put_log("HEADER", "=" * 60)

    _worker = threading.Thread(
        target=_do_download,
        args=(url, is_playlist, quality, embed_subs, download_mode,
              _put_log, _put_progress, _put_video_start,
              _put_video_done, _put_all_done),
        daemon=True,
    )
    _worker.start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def stop_download():
    global _cancelled
    _cancelled = True
    _event_queue.put(("log", "WARNING", "⏹ 用户请求停止，正在结束当前任务..."))
    return jsonify({"status": "stopping"})


def _do_download(url, is_playlist, quality, embed_subs, download_mode,
                 put_log, put_progress, put_video_start,
                 put_video_done, put_all_done):
    global _cancelled, _running
    start_time = time.time()
    try:
        downloader = BilibiliDownloader(progress_callback=put_progress, log_callback=put_log)
    except SystemExit:
        put_log("ERROR", "FFmpeg 或 yt-dlp 不可用，请检查依赖")
        put_all_done(0, 0, [], "")
        _running = False
        return

    if is_playlist:
        try:
            import yt_dlp
            put_log("INFO", "正在获取合集视频列表...")
            extract_opts = {
                "quiet": False, "no_warnings": True, "noplaylist": False,
                "extract_flat": "in_playlist",
                "cookiefile": str(COOKIE_FILE) if COOKIE_FILE.exists() else None,
                "progress_hooks": [], "logger": logging.getLogger(),
            }
            with yt_dlp.YoutubeDL(extract_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            entries = info.get("entries", [])
            if not entries:
                put_log("ERROR", "未获取到任何视频")
                put_all_done(0, 0, [], "")
                _running = False
                return

            total = len(entries)
            playlist_title = re.sub(r'[<>:"/\\|?*]', '_', info.get("title", "合集").strip())
            output_dir = get_output_dir()
            playlist_dir = output_dir / playlist_title
            playlist_dir.mkdir(parents=True, exist_ok=True)
            put_log("INFO", f"合集「{playlist_title}」共 {total} 个视频")

            max_retries = _get("collection.max_retries", 3)
            retry_delays = _get("collection.retry_delays", [3, 8, 15])
            success_count = 0
            fail_list = []

            for i, entry in enumerate(entries, 1):
                if _cancelled:
                    put_log("WARNING", "用户取消下载")
                    break
                video_url = (entry.get("webpage_url") or entry.get("url") or entry.get("original_url"))
                video_id = entry.get("id") or entry.get("display_id") or ""
                video_title = entry.get("title", f"video_{i}")
                if not video_url:
                    if video_id and video_id.startswith("BV"):
                        video_url = f"https://www.bilibili.com/video/{video_id}"
                    else:
                        fail_list.append(f"#{i} {video_title} (无链接)")
                        continue
                put_video_start(i, total, video_title)
                put_log("INFO", f"[{i}/{total}] {video_title}")
                _set_output_dir(playlist_dir)
                ok = False
                for retry in range(max_retries):
                    if _cancelled:
                        break
                    try:
                        ok = downloader.download_video(
                            url=video_url, is_playlist=False, quality=quality,
                            info_only=False,
                            embed_subs=embed_subs, verbose=False,
                            download_mode=download_mode)
                        if ok:
                            break
                    except (downloader.yt_dlp.utils.DownloadError,
                            downloader.yt_dlp.utils.ExtractorError) as e:
                        err_s = str(e).lower()
                        if any(kw in err_s for kw in
                               ["private", "deleted", "copyright", "region", "removed",
                                "not found", "404", "forbidden", "unavailable"]):
                            put_log("ERROR", f"该视频不可用: {e}")
                            break
                        if retry < max_retries - 1:
                            delay = retry_delays[retry]
                            put_log("WARNING", f"重试 {retry+1}（{delay}s 后）: {e}")
                            time.sleep(delay)
                        else:
                            put_log("ERROR", f"重试 {max_retries} 次仍失败: {e}")
                    except Exception as e:
                        put_log("ERROR", f"下载异常: {e}")
                        if retry >= max_retries - 1:
                            break
                put_video_done(ok, i, total)
                if ok:
                    success_count += 1
                else:
                    fail_list.append(f"#{i} {video_title}")
                if i < total and not _cancelled:
                    delay = random.uniform(_get("collection.min_delay", 2.0),
                                           _get("collection.max_delay", 5.0))
                    put_log("INFO", f"等待 {delay:.1f}s 后继续...")
                    time.sleep(delay)
            elapsed = time.time() - start_time
            put_all_done(success_count, total, fail_list, str(playlist_dir), elapsed)
        except Exception as e:
            put_log("ERROR", f"合集下载失败: {e}")
            put_log("ERROR", traceback.format_exc())
            put_all_done(0, 0, [], "")
    else:
        try:
            ok = downloader.download_video(
                url=url, is_playlist=False, quality=quality,
                info_only=False,
                embed_subs=embed_subs, verbose=False,
                download_mode=download_mode)
            elapsed = time.time() - start_time
            put_all_done(1 if ok else 0, 1, [] if ok else ["下载失败"],
                         str(get_output_dir()), elapsed)
        except Exception as e:
            put_log("ERROR", f"下载失败: {e}")
            put_log("ERROR", traceback.format_exc())
            put_all_done(0, 1, ["下载失败"], "")
    _running = False


# ═══════════════════════════════════════
# 路由：视频处理控制
# ═══════════════════════════════════════
@app.route("/api/process/start", methods=["POST"])
def start_processing():
    global _running, _worker_process, _worker_bridge
    if _running:
        return jsonify({"error": "任务正在进行中"}), 409

    data = request.get_json(force=True)
    video_files = data.get("video_files", [])
    out_dir = data.get("out_dir", "").strip()
    mode = data.get("mode", "subtitle_only")
    batch_name = data.get("batch_name", "").strip() or None

    if not video_files:
        return jsonify({"error": "请选择视频文件"}), 400
    if not out_dir:
        return jsonify({"error": "请指定输出目录"}), 400

    config = _load_processor_config()
    config.setdefault("asr", {}).setdefault("device", "cpu")
    config.setdefault("asr", {}).setdefault("model_size", "base")
    config.setdefault("audio", {}).setdefault("sample_rate", 16000)
    config.setdefault("global", {}).setdefault("max_concurrency",
                                                {"video_processor": 1})

    from multiprocessing import Process, Queue as MPQueue
    mp_queue = MPQueue()

    _running = True
    _worker_process = Process(
        target=_video_worker_fn,
        args=(video_files, config, mode, video_files[0].rsplit(os.sep, 1)[0] if len(video_files) > 0 else "",
              out_dir, batch_name, str(PROJECT_ROOT_PATH), mp_queue),
        daemon=True,
    )
    _worker_process.start()

    _worker_bridge = threading.Thread(target=_bridge_mp_queue, args=(_worker_process, mp_queue), daemon=True)
    _worker_bridge.start()

    _event_queue.put(("log", "HEADER", "=" * 60))
    _event_queue.put(("log", "HEADER", f"🔧 开始视频处理: {len(video_files)} 个视频"))
    _event_queue.put(("log", "HEADER", f"📁 输出目录: {out_dir}"))
    _event_queue.put(("log", "HEADER", f"🎯 处理模式: {mode}"))
    _event_queue.put(("log", "HEADER", "=" * 60))

    return jsonify({"status": "started"})


@app.route("/api/process/stop", methods=["POST"])
def stop_processing():
    global _worker_process
    _event_queue.put(("log", "WARNING", "⏹ 用户请求停止视频处理..."))
    if _worker_process and _worker_process.is_alive():
        _worker_process.terminate()
        _worker_process.join(timeout=5)
    return jsonify({"status": "stopping"})


# ═══════════════════════════════════════
# 队列桥接：子进程 Queue → Flask 事件队列
# ═══════════════════════════════════════
def _bridge_mp_queue(proc, mp_queue):
    """从 multiprocessing.Queue 读取消息，检测 STEP 模式，推送到 SSE 队列"""
    global _running
    step_pattern = re.compile(r'\[STEP (\d+)/(\d+)\]')
    try:
        while True:
            try:
                msg = mp_queue.get(timeout=0.5)
                msg_type = msg[0]
                if msg_type == "log":
                    _event_queue.put(msg)
                    message = msg[2]
                    m = step_pattern.search(message)
                    if m:
                        step = int(m.group(1))
                        total = int(m.group(2))
                        label = message.split('] ', 1)[1] if '] ' in message else ""
                        _event_queue.put(("process_step", step, total, label))
                else:
                    _event_queue.put(msg)
            except Exception:
                if not proc.is_alive():
                    break
    finally:
        _running = False


# ═══════════════════════════════════════
# 视频处理 Worker（模块级函数，用于 macOS spawn）
# ═══════════════════════════════════════
def _video_worker_fn(video_files, config, mode, input_dir, output_dir,
                     batch_name, project_root_str, progress_q):
    """在独立子进程中运行，逐个处理视频"""
    import os as _os
    import sys as _sys
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    project_root = _Path(project_root_str)
    _os.chdir(str(project_root))
    if str(project_root) not in _sys.path:
        _sys.path.insert(0, str(project_root))

    from loguru import logger as _logger
    _logger.remove()

    def _log_sink(message):
        try:
            progress_q.put(("log", message.record["level"].name,
                           str(message).rstrip()))
        except Exception:
            pass

    _logger.add(_log_sink, format="{message}", level="INFO")

    import video_cli as _main

    total = len(video_files)
    start_time = _time.time()
    results = []
    fail_list = []
    success_count = 0

    for i, vf in enumerate(video_files, 1):
        vf_path = _Path(vf)
        progress_q.put(("video_start", i, total, vf_path.name))
        _logger.info(f"开始处理: {vf_path.name}")
        try:
            result = _main.process_video_unified(
                vf, config, mode, input_dir, output_dir, batch_name)
            ok = result.get("status") == "success"
            progress_q.put(("video_done", ok, i, total))
            results.append(result)
            if ok:
                success_count += 1
            else:
                fail_list.append(vf_path.name)
        except Exception as e:
            _logger.error(f"处理异常: {e}")
            import traceback as _tb
            _logger.error(_tb.format_exc())
            progress_q.put(("video_done", False, i, total))
            fail_list.append(vf_path.name)

    elapsed = _time.time() - start_time
    progress_q.put(("all_done", success_count, total, fail_list, output_dir, elapsed))


# ═══════════════════════════════════════
# 端口检测
# ═══════════════════════════════════════
def find_free_port(start=19999, end=20999):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start


# ═══════════════════════════════════════
# 入口
# ═══════════════════════════════════════
def main():
    port = find_free_port()
    print(f"\n{'='*55}")
    print(f"  🎬 统一视频工具平台")
    print(f"  浏览器访问: http://127.0.0.1:{port}")
    print(f"  📥 哔哩哔哩下载    — 首页")
    print(f"  🔧 视频处理        — /process")
    print(f"{'='*55}\n")

    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
