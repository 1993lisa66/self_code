#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
哔哩哔哩（Bilibili）视频下载脚本
=================================
功能：
  1. 单视频下载：支持通过 BV 号或完整 URL 下载单个视频
  2. 合集/列表下载：支持分P视频和播放列表批量下载
  3. 音画同步合并：自动下载最佳音轨+视轨，使用 FFmpeg 无损合并为 MP4

依赖：
  - yt-dlp (核心下载引擎)
  - FFmpeg (音视频合并，项目自带 ffmpeg/bin/ 下)
  - requests (cookie 相关)

安装方式：
  pip install yt-dlp
"""

import os
import sys
import re
import json
import time
import random
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

# ──────────────────────────────────────────────
# 项目路径配置
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
FFMPEG_DIR = PROJECT_ROOT / "ffmpeg" / "bin"
FFMPEG_BIN = FFMPEG_DIR / "ffmpeg_real"
FFPROBE_BIN = FFMPEG_DIR / "ffprobe_real"
COOKIE_FILE = PROJECT_ROOT / "config" / "bilibili_cookies.txt"

# 运行时输出目录（在 main() 中初始化，可通过命令行 -o 覆盖）
_output_dir: Optional[Path] = None

# ──────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "bilibili_download.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def check_ffmpeg() -> bool:
    """检查 FFmpeg 是否可用"""
    ffmpeg_path = str(FFMPEG_BIN) if FFMPEG_BIN.exists() else "ffmpeg"
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            logger.info(f"FFmpeg 可用: {version_line}")
            return True
    except FileNotFoundError:
        pass
    logger.error("FFmpeg 不可用，请确保 ffmpeg/bin/ffmpeg_real 存在或已安装系统 FFmpeg")
    return False


def is_valid_bv(bv: str) -> bool:
    """校验 BV 号格式"""
    return bool(re.fullmatch(r"BV[a-zA-Z0-9]{10}", bv))


def extract_bv_from_url(url: str) -> Optional[str]:
    """从 URL 中提取 BV 号"""
    patterns = [
        r"BV[a-zA-Z0-9]{10}",
        r"bilibili\.com/video/(BV[a-zA-Z0-9]{10})",
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1) if "(" in p else match.group(0)
    return None


def build_url(bv: str) -> str:
    """根据 BV 号构建完整 URL"""
    return f"https://www.bilibili.com/video/{bv}"


def set_output_dir(path: Path):
    """设置运行时输出目录"""
    global _output_dir
    _output_dir = Path(path)
    _output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {_output_dir}")


# ──────────────────────────────────────────────
# B站二维码扫码登录
# ──────────────────────────────────────────────
def qrcode_login() -> bool:
    """
    使用 B站 官方 API 实现二维码扫码登录。

    流程：
      1. 先访问 B站首页获取初始 Cookie / CSRF
      2. 请求生成二维码（获取 qrcode_key）
      3. 在终端打印二维码，用户使用 B站 APP 扫描
      4. 轮询等待用户确认登录
      5. 从登录成功的响应中提取 Cookie 并保存为 Netscape 格式

    返回:
        bool: 登录是否成功
    """
    print("\n" + "=" * 60)
    print("  🔐 B站二维码扫码登录")
    print("=" * 60)

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.bilibili.com/",
        "Origin": "https://www.bilibili.com",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    })

    # ── Step 0：先访问 B站首页，获取必要的前置 Cookie ──
    try:
        resp = session.get("https://www.bilibili.com/", timeout=15)
    except Exception as e:
        logger.warning(f"访问 B站首页失败: {e}，继续尝试...")

    # ── Step 1：生成二维码 ──
    generate_url = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
    try:
        resp = session.get(generate_url, timeout=15)
        content_type = resp.headers.get("Content-Type", "?")
        content_encoding = resp.headers.get("Content-Encoding", "?")
        if resp.status_code != 200:
            logger.error(
                f"生成二维码失败 - HTTP {resp.status_code}, "
                f"Content-Type: {content_type}, "
                f"Content-Encoding: {content_encoding}"
            )
            raw_preview = resp.content[:200]
            logger.error(f"响应内容 (hex): {raw_preview.hex()}")
            return False
        raw_text = resp.text.strip()
        if not raw_text:
            logger.error("生成二维码返回空响应")
            return False
        data = resp.json()
    except requests.exceptions.JSONDecodeError as e:
        logger.error(
            f"API 返回非 JSON 数据 - "
            f"Content-Type: {resp.headers.get('Content-Type', '?')}, "
            f"Content-Encoding: {resp.headers.get('Content-Encoding', '?')}"
        )
        raw_preview = resp.content[:200]
        logger.error(f"响应内容 (hex): {raw_preview.hex()}")
        return False
    except Exception as e:
        logger.error(f"获取二维码失败: {e}")
        return False

    if data.get("code") != 0:
        logger.error(f"B站 API 返回错误 [{data.get('code')}]: {data.get('message', '未知')}")
        return False

    qrcode_key = data["data"]["qrcode_key"]
    qrcode_url = data["data"]["url"]
    logger.info(f"二维码 Key: {qrcode_key}")

    # ── Step 2：显示二维码 ──
    # 同时在终端打印和提供可点击链接
    print(f"\n  🔗 点击链接在浏览器中查看二维码：")
    print(f"  https://tool.liumingye.cn/qrcode/?text={requests.utils.quote(qrcode_url)}")
    print()
    _print_qrcode(qrcode_url)

    print("\n  📱 请使用【Bilibili APP】扫描上方二维码并确认登录...")
    print('  💡 终端二维码模糊？复制上方 🔗 链接在浏览器中打开\n')

    # ── Step 3：轮询登录状态 ──
    poll_url = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"
    max_wait = 180  # 最多等待 180 秒（3 分钟）
    poll_interval = 3  # 每 3 秒轮询一次
    elapsed = 0

    status_messages = {
        86101: "⏳ 等待扫码...",
        86090: "📲 已扫码，请在手机上【确认登录】...",
        86038: "⚠️  二维码已失效，请重新运行",
    }

    last_msg = ""

    while elapsed < max_wait:
        try:
            resp = session.get(poll_url, params={"qrcode_key": qrcode_key}, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            logger.error(f"轮询失败: {e}")
            time.sleep(poll_interval)
            elapsed += poll_interval
            continue

        if result.get("code") != 0:
            logger.error(f"轮询 API 错误: {result.get('message', '')}")
            time.sleep(poll_interval)
            elapsed += poll_interval
            continue

        status_code = result["data"].get("code")

        if status_code == 0:
            # ✅ 登录成功！提取 Cookie
            print("\n  ✅ 扫码登录成功！正在保存 Cookie...")
            return _save_cookies_from_session(session, result["data"])

        # 显示状态变更
        msg = status_messages.get(status_code)
        if msg and msg != last_msg:
            print(f"\r  {msg}", end="", flush=True)
            last_msg = msg
        elif status_code not in status_messages:
            logger.warning(f"未知状态码: {status_code}, 响应: {result['data']}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    print("\n  ⏰ 登录超时（超过 3 分钟），请重新运行")
    return False


def _print_qrcode(url: str):
    """在终端打印 ASCII 二维码"""
    try:
        import qrcode
        from qrcode.image.terminal import TerminalImage

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        return
    except ImportError:
        pass

    # ── 降级方案：使用 qrcode_terminal ──
    try:
        import qrcode_terminal
        qrcode_terminal.draw(url)
        return
    except ImportError:
        pass

    # ── 最终降级：纯文本 URL ──
    print(f"\n  （未安装 qrcode 库，无法显示二维码图形）")
    print(f"  （请复制以下链接在浏览器中打开查看二维码：）")
    print(f"\n  🔗 https://tool.liumingye.cn/qrcode/?text={requests.utils.quote(url)}")


def _save_cookies_from_session(session: requests.Session, login_data: dict) -> bool:
    """
    从登录响应的 session 中提取 Cookie 并保存为 Netscape 格式文件。
    yt-dlp 使用标准的 Netscape cookie 格式。
    """
    cookies = session.cookies

    # 关键 Cookie 字段
    required_keys = ["SESSDATA", "bili_jct", "DedeUserID"]
    existing = [k for k in required_keys if k in cookies]

    if not existing:
        logger.error("登录响应中未找到必需的 Cookie（SESSDATA / bili_jct / DedeUserID）")
        return False

    # ── 写入 Netscape 格式 Cookie 文件 ──
    COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(COOKIE_FILE, "w", encoding="utf-8") as f:
        f.write("# Netscape HTTP Cookie File\n")
        f.write("# This is a generated file! Do not edit.\n\n")
        for cookie in cookies:
            domain = cookie.domain if cookie.domain.startswith(".") else f".{cookie.domain}"
            flag = "TRUE"
            path = cookie.path or "/"
            secure = "TRUE" if cookie.secure else "FALSE"
            expires = str(cookie.expires) if cookie.expires else "0"
            name = cookie.name
            value = cookie.value or ""
            f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")

    logger.info(f"✅ Cookie 已保存到: {COOKIE_FILE}")
    logger.info(f"   包含关键字段: {', '.join(existing)}")
    return True


def extract_cookies_from_browser(browser: str) -> bool:
    """
    使用 yt-dlp 从浏览器提取 B站 Cookie 并保存为 Netscape 格式文件。

    这是目前最可靠的方式，因为：
      - B站 API 对脚本请求开启了 WAF / 反爬保护
      - yt-dlp 内置的浏览器 Cookie 提取可以绕过加密存储

    Args:
        browser: 浏览器名称 (chrome, firefox, safari, edge, etc.)

    Returns:
        bool: 是否成功提取并保存
    """
    print("\n" + "=" * 60)
    print(f"  🍪 从 {browser.title()} 浏览器提取 B站 Cookie")
    print("=" * 60)
    print(f"\n  请确保你已在 {browser.title()} 浏览器中登录 bilibili.com")
    print("  如果弹出「钥匙串访问」/ Keychain 权限请求，请点「允许」\n")

    COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 先删除旧文件，防止 yt-dlp 报 file exists 错误
    if COOKIE_FILE.exists():
        COOKIE_FILE.unlink()

    # 直接用 yt-dlp 提取并保存 cookie
    # 需要用一个有效的视频页面 URL，不能用 bilibili.com 根域名
    cmd = [
        "yt-dlp",
        "--cookies-from-browser", browser,
        "--cookies", str(COOKIE_FILE),
        "--skip-download",
        "--no-warnings",
        "--print", "Cookies extracted successfully",
        "https://www.bilibili.com/video/BV1GJ411x7h7",
    ]

    print(f"  执行: {' '.join(cmd)}\n")
    print("  等待中", end="", flush=True)

    # 不捕获输出，让 yt-dlp 直接输出到终端（Keychain 弹窗需要用户交互）
    # 使用 Popen 以便在过程中显示进度点
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        logger.error("未找到 yt-dlp 命令，请确保已安装: pip install yt-dlp")
        return False

    # 等待完成（给充足的时间让用户处理 Keychain 弹窗）
    try:
        stdout, stderr = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error("Cookie 提取超时（超过3分钟）。请确认浏览器已关闭且 Keychain 弹窗已处理")
        return False

    print()  # 换行
    returncode = proc.returncode

    if returncode != 0:
        # 分析错误原因
        combined = (stderr + stdout).lower()
        if any(kw in combined for kw in ["chrome", "chromium"]) and "locked" in combined:
            logger.error(
                "Chrome 浏览器正在运行或被锁定。"
                "请完全关闭 Chrome 后重试。"
            )
            logger.error(
                "  提示: 打开「活动监视器」确保没有 Chrome 进程在运行"
            )
        elif "keychain" in combined or "keyring" in combined:
            logger.error(
                "无法访问系统钥匙串（Keychain）。"
                "请在弹出的对话框中选择「允许」。"
            )
        elif "firefox" in combined and "cookie" in combined:
            logger.error(
                "Firefox cookie 数据库读取失败。"
                "请确保 Firefox 已安装且已登录 bilibili.com"
            )
        elif "no cookies" in combined or "no valid cookies" in combined:
            logger.error(
                f"未在 {browser} 浏览器中找到 bilibili.com 的 Cookie。"
                f"请先在浏览器中打开 https://www.bilibili.com 并登录。"
            )
        elif "permission" in combined or "denied" in combined:
            logger.error(
                "权限被拒绝。请在 macOS 设置中允许终端访问浏览器的数据："
                "\n  系统设置 → 隐私与安全性 → 允许终端访问 Chrome"
            )
        else:
            # 打印原始错误信息
            if stderr:
                logger.error(f"Cookie 提取失败: {stderr.strip()}")
            if stdout:
                logger.error(f"yt-dlp 输出: {stdout.strip()}")
        return False

    # 验证文件是否生成且有内容
    if not COOKIE_FILE.exists() or COOKIE_FILE.stat().st_size < 100:
        logger.error("Cookie 文件为空或过小，提取可能失败")
        if stderr:
            logger.error(f"yt-dlp stderr: {stderr.strip()}")
        return False

    # 读取并检查关键字段
    content = COOKIE_FILE.read_text(encoding="utf-8")
    has_sessdata = "SESSDATA" in content
    has_bili_jct = "bili_jct" in content

    if has_sessdata and has_bili_jct:
        logger.info(f"✅ Cookie 已保存到: {COOKIE_FILE}")
        logger.info("   包含 SESSDATA ✅ | bili_jct ✅")
        return True
    elif has_sessdata:
        logger.info(f"✅ Cookie 已保存到: {COOKIE_FILE}")
        logger.info("   包含 SESSDATA ✅ (bili_jct 将在后续请求中自动获取)")
        return True
    else:
        logger.warning("⚠️  Cookie 已保存但缺少 SESSDATA，可能未登录或已过期")
        return False


# ──────────────────────────────────────────────
# 核心下载器
# ──────────────────────────────────────────────
class BilibiliDownloader:
    """
    哔哩哔哩视频下载器
    基于 yt-dlp 实现，支持音视频分离流的自动下载与 FFmpeg 合并。
    """

    def __init__(self):
        # 验证 yt-dlp 是否安装
        try:
            import yt_dlp

            self.yt_dlp = yt_dlp
        except ImportError:
            logger.error(
                "yt-dlp 未安装，请执行: pip install yt-dlp"
            )
            sys.exit(1)

        if not check_ffmpeg():
            logger.error("FFmpeg 不可用，无法合并音视频轨道")
            sys.exit(1)

    # -------------------------------------------------------
    # 构建 yt-dlp 下载选项
    # -------------------------------------------------------
    def _build_opts(
        self,
        is_playlist: bool = False,
        quality: str = "bestvideo+bestaudio",
        download_subs: bool = True,
        embed_subs: bool = False,
        verbose: bool = False,
    ) -> dict:
        """
        构建 yt-dlp 选项字典。

        参数:
            is_playlist:  是否为播放列表/合集下载
            quality:      视频质量选择，默认自动选最佳
            download_subs: 是否下载字幕（默认开启）
            embed_subs:   是否将字幕嵌入视频文件（默认不嵌入，保留独立 .srt 文件）
        """
        output_template = str(_output_dir)
        if is_playlist:
            # 播放列表/合集：按序号分目录
            output_template = str(
                _output_dir / "%(playlist_title)s" / "%(playlist_index)03d_%(title)s.%(ext)s"
            )
        else:
            output_template = str(
                _output_dir / "%(title)s_%(id)s.%(ext)s"
            )

        opts = {
            # ── 音视频分离流处理 ──
            "format": f"{quality}/best",
            "merge_output_format": "mp4",
            # ── 输出设置 ──
            "outtmpl": output_template,
            "overwrites": False,
            "paths": {"home": str(_output_dir)},
            # ── FFmpeg 设置 ──
            "ffmpeg_location": str(FFMPEG_BIN) if FFMPEG_BIN.exists() else None,
            # ── 进度显示 ──
            "progress_hooks": [self._progress_hook],
            "quiet": False,
            "no_warnings": False,
            # ── 网络设置 ──
            "retries": 10,
            "fragment_retries": 10,
            "ignoreerrors": is_playlist,  # 合集模式忽略单个视频错误，继续下载后续
            "continuedl": True,
            "concurrent_fragment_downloads": 5,
            # B站 限流保护：限制请求速度，避免被断开连接
            "sleep_interval_requests": 1,
            "sleep_interval": 1,
            "max_sleep_interval_requests": 3,
            "socket_timeout": 30,
            # ── 元数据 ──
            "writedescription": False,
            "writeinfojson": False,
            "restrictfilenames": False,
            # ── 登录态 ──
            "cookiefile": str(COOKIE_FILE) if COOKIE_FILE.exists() else None,
            # ── 合集/分P处理 ──
            "playlistend": 0 if not is_playlist else None,
            "noplaylist": not is_playlist,
            # ── 字幕 ──
            # 注意：不要手动指定 postprocessors，会覆盖 yt-dlp 默认的字幕处理器
            "writesubtitles": download_subs,
            "writeautomaticsub": download_subs,
            "subtitleslangs": ["all"],
            "subtitlesformat": "srt",
            # ── 用户代理 ──
            "user_agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        # 仅在需要嵌入字幕时才添加 embedsubs 选项
        if embed_subs:
            opts["embedsubs"] = True

        return opts

    # -------------------------------------------------------
    # 进度回调
    # -------------------------------------------------------
    def _progress_hook(self, d: dict):
        """
        yt-dlp 进度钩子。
        在下载过程中实时输出进度信息，帮助用户了解当前状态。
        """
        status = d.get("status")
        if status == "downloading":
            # 已下载 / 总大小 或 已下载字节
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            speed = d.get("speed") or 0
            eta = d.get("eta", "?")
            percent = d.get("_percent_str", "0%").strip()
            filename = os.path.basename(d.get("filename", ""))

            if speed:
                speed_str = self._format_speed(speed)
            else:
                speed_str = "N/A"

            if total:
                progress = f"{percent} [{self._format_size(downloaded)}/{self._format_size(total)}]"
            else:
                progress = f"{self._format_size(downloaded)} 已下载"

            print(
                f"\r  📥 {filename[:50]:<50} {progress} 速度:{speed_str} ETA:{eta}s",
                end="",
            )
        elif status == "finished":
            print()  # 换行
            logger.info(f"✅ 下载完成: {os.path.basename(d.get('filename', ''))}")

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"

    @staticmethod
    def _format_speed(speed_bytes: float) -> str:
        """格式化下载速度"""
        for unit in ["B/s", "KB/s", "MB/s", "GB/s"]:
            if speed_bytes < 1024:
                return f"{speed_bytes:.1f}{unit}"
            speed_bytes /= 1024
        return f"{speed_bytes:.1f}TB/s"

    # -------------------------------------------------------
    # 获取视频信息
    # -------------------------------------------------------
    def get_video_info(self, url: str, verbose: bool = False) -> dict:
        """
        预获取视频元信息（不下载文件）。
        可用于预览视频标题、分P数量、可用格式等。
        """
        opts = {
            "quiet": not verbose,
            "no_warnings": not verbose,
            "verbose": verbose,
            "cookiefile": str(COOKIE_FILE) if COOKIE_FILE.exists() else None,
            # 同时拉取字幕信息
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["all"],
        }
        with self.yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # 如果是播放列表，提取列表信息
            if "entries" in info and info.get("entries"):
                info["entry_count"] = len(info["entries"])
                entries = info["entries"]
                # 只展示前5个
                info["entries_preview"] = [
                    {
                        "title": e.get("title", "未知"),
                        "duration": e.get("duration", 0),
                        "id": e.get("id", ""),
                    }
                    for e in entries[:5]
                ]
            return info

    # -------------------------------------------------------
    # 核心下载方法
    # -------------------------------------------------------
    def download_video(
        self,
        url: str,
        is_playlist: bool = False,
        quality: str = "bestvideo+bestaudio",
        info_only: bool = False,
        download_subs: bool = True,
        embed_subs: bool = False,
        verbose: bool = False,
    ) -> bool:
        """
        下载视频。

        参数:
            url:          视频 URL 或 BV 号
            is_playlist:  是否下载整个播放列表/合集
            quality:      下载质量。可选值:
                          "bestvideo+bestaudio" — 最佳画质+最佳音质（默认）
                          "1080" — 最高 1080p
                          "720" — 最高 720p
            info_only:    仅显示视频信息，不下载
            download_subs: 是否下载字幕（默认开启）
            embed_subs:   是否将字幕嵌入视频文件
            verbose:      是否输出 yt-dlp 详细日志

        返回:
            bool: 下载是否成功
        """
        # ── URL 规范化 ──
        if is_valid_bv(url):
            url = build_url(url)
        elif not url.startswith("http"):
            logger.error(f"无效的输入: {url}，请输入有效 BV 号或 URL")
            return False

        # ── 质量选择 ──
        format_map = {
            "bestvideo+bestaudio": "bestvideo+bestaudio",
            "1080": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "720": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        }
        fmt = format_map.get(quality, quality)

        # ── 仅查看信息模式 ──
        if info_only:
            try:
                info = self.get_video_info(url, verbose=verbose)
                self._print_video_info(info)
                return True
            except Exception as e:
                logger.error(f"获取视频信息失败: {e}")
                return False

        # ── 开始下载 ──
        opts = self._build_opts(
            is_playlist=is_playlist,
            quality=fmt,
            download_subs=download_subs,
            embed_subs=embed_subs,
            verbose=verbose,
        )

        try:
            with self.yt_dlp.YoutubeDL(opts) as ydl:
                logger.info(f"开始下载: {url}")
                if is_playlist:
                    logger.info("模式: 播放列表/合集（下载所有分P）")
                else:
                    logger.info("模式: 单视频")

                ydl.download([url])
                logger.info("=" * 60)
                logger.info(f"下载成功！文件保存在: {_output_dir}")
                return True

        except self.yt_dlp.utils.DownloadError as e:
            logger.error(f"下载失败 (DownloadError): {e}")
        except self.yt_dlp.utils.ExtractorError as e:
            logger.error(f"解析失败 (ExtractorError): {e}")
            logger.error("提示: 该视频可能需要登录才能访问，请提供 Cookie 文件")
        except Exception as e:
            logger.error(f"未知错误: {type(e).__name__}: {e}")

        return False

    # -------------------------------------------------------
    # 打印视频信息
    # -------------------------------------------------------
    def _print_video_info(self, info: dict):
        """格式化打印视频信息"""
        print("\n" + "=" * 60)
        print("📺 视频信息")
        print("=" * 60)
        print(f"  标题    : {info.get('title', '未知')}")
        print(f"  ID      : {info.get('id', '未知')}")
        duration = info.get("duration", 0)
        if duration:
            print(f"  时长    : {duration // 60}分{duration % 60}秒")
        print(f"  上传者  : {info.get('uploader', '未知')}")

        # 播放列表信息
        if info.get("entry_count"):
            print(f"  分P数量 : {info['entry_count']}")
            print(f"  前5项预览:")
            for i, entry in enumerate(info.get("entries_preview", []), 1):
                d = entry.get("duration", 0)
                print(f"    {i}. {entry['title'][:40]}  ({d//60}:{d%60:02d})")

        # 格式信息
        formats = info.get("formats", [])
        if formats:
            print(f"\n  可用格式 (共{len(formats)}种):")
            shown = 0
            for f in formats:
                if shown >= 8:
                    break
                fid = f.get("format_id", "?")
                res = f.get("resolution") or f.get("height") or "?"
                vcodec = f.get("vcodec", "none")
                acodec = f.get("acodec", "none")
                fsize = f.get("filesize") or f.get("filesize_approx")
                size_str = self._format_size(fsize) if fsize else "未知"
                has_video = vcodec != "none"
                has_audio = acodec != "none"
                tag = ""
                if has_video and has_audio:
                    tag = "[音画合一]"
                elif has_video:
                    tag = "[仅视频]"
                elif has_audio:
                    tag = "[仅音频]"
                print(f"    {fid:<8} {str(res):>8} {tag:<12} {size_str:>10}")
                shown += 1
            if len(formats) > 8:
                print(f"    ... 还有 {len(formats) - 8} 种格式")

        # 字幕信息
        subtitles = info.get("subtitles") or {}
        auto_subtitles = info.get("automatic_captions") or {}
        if subtitles or auto_subtitles:
            print(f"\n  📝 可用字幕:")
            for lang, subs in subtitles.items():
                count = len(subs) if isinstance(subs, list) else "?"
                print(f"    手动字幕 [{lang}]: {count} 条")
            for lang, subs in auto_subtitles.items():
                count = len(subs) if isinstance(subs, list) else "?"
                print(f"    AI字幕 [{lang}]: {count} 条")
        else:
            print(f"\n  📝 字幕: 该视频不支持字幕")

        print("=" * 60)

    # -------------------------------------------------------
    # 下载合集（通过 collection ID）
    # -------------------------------------------------------
    def download_collection(self, collection_id: str, quality: str = "bestvideo+bestaudio") -> bool:
        """
        通过合集 ID 下载整个合集。

        参数:
            collection_id: B站合集 ID（数字），通常从 URL 中获取
                          例如 https://space.bilibili.com/xxx/channel/collectiondetail?sid=12345
            quality:      下载质量
        """
        # B站合集 URL 格式
        collection_url = f"https://space.bilibili.com/0/channel/collectiondetail?sid={collection_id}"
        logger.info(f"合集 ID: {collection_id}")
        logger.info(f"合集 URL: {collection_url}")

        return self.download_video(
            url=collection_url,
            is_playlist=True,
            quality=quality,
        )


# ──────────────────────────────────────────────
# Cookie 帮助函数
# ──────────────────────────────────────────────
def generate_cookie_instructions():
    """打印获取 Cookie 文件的指引"""
    print("""
╔══════════════════════════════════════════════════════════╗
║           B站 Cookie 获取指引                              ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  部分高清视频（如 1080p+）需要登录才能访问。              ║
║  获取 Cookie 的步骤如下：                                 ║
║                                                          ║
║  方法一：浏览器扩展导出                                  ║
║    1. 安装 Chrome 扩展 "Get cookies.txt LOCALLY"          ║
║    2. 登录 B站后，在 B站页面点击该扩展导出                ║
║    3. 保存到 config/bilibili_cookies.txt                 ║
║                                                          ║
║  方法二：手动创建                                         ║
║    1. 在浏览器中登录 B站                                  ║
║    2. F12 -> Application -> Cookies -> bilibili.com      ║
║    3. 复制以下字段的值：                                  ║
║       - SESSDATA                                          ║
║       - bili_jct                                          ║
║       - DedeUserID                                        ║
║    4. 在 config/bilibili_cookies.txt 中写入：            ║
║       .bilibili.com\tTRUE\t/\tFALSE\t0\tSESSDATA\t值     ║
║       .bilibili.com\tTRUE\t/\tFALSE\t0\tbili_jct\t值     ║
║       .bilibili.com\tTRUE\t/\tFALSE\t0\tDedeUserID\t值   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


def is_playlist_url(url: str) -> bool:
    """检测 URL 是否为合集/播放列表/番剧"""
    patterns = [
        r"space\.bilibili\.com/\d+/lists/\d+",     # 合集/系列
        r"bilibili\.com/bangumi/play/",              # 番剧播放列表
        r"bilibili\.com/video/BV[a-zA-Z0-9]{10}\?p=",  # 带分P参数
        r"channel/collectiondetail",                 # 合集详情
    ]
    return any(re.search(p, url) for p in patterns)


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────
def main():
    # ═══════════════════════════════════════════════════
    # 所有配置都在这里修改
    # ═══════════════════════════════════════════════════
    DEFAULT_URL = "https://space.bilibili.com/22550161/lists/8091832?type=season"
    DEFAULT_OUTPUT_DIR = Path("/Volumes/mvp/交易场/光子交易")
    DEFAULT_QUALITY = "bestvideo+bestaudio"

    # ── 初始化输出目录 ──
    global _output_dir
    _output_dir = DEFAULT_OUTPUT_DIR if DEFAULT_OUTPUT_DIR.exists() else (PROJECT_ROOT / "outputs" / "bilibili")
    _output_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════

    parser = argparse.ArgumentParser(
        description="哔哩哔哩（Bilibili）视频下载器 — 基于 yt-dlp + FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
使用示例:
  # 直接运行（使用默认链接和输出目录）
  python bilibili_downloader.py

  # 下载单个视频（通过 BV 号）
  python bilibili_downloader.py BV1xx411c7mD

  # 下载单个视频（通过完整 URL）
  python bilibili_downloader.py https://www.bilibili.com/video/BV1xx411c7mD

  # 下载整个播放列表（所有分P）
  python bilibili_downloader.py https://www.bilibili.com/video/BV1xx411c7mD --playlist

  # 指定画质（720p）
  python bilibili_downloader.py BV1xx411c7mD -q 720

  # 仅查看视频信息（不下载）
  python bilibili_downloader.py https://www.bilibili.com/video/BV1xx411c7mD --info

  # 从浏览器提取 Cookie（最可靠，推荐！）
  python bilibili_downloader.py --cookies-from-browser chrome

  # 二维码扫码登录（部分网络可能不可用）
  python bilibili_downloader.py --login

  # 自定义输出目录
  python bilibili_downloader.py -o /path/to/output

  # 强制单视频模式（合集 URL 默认下载全部，加此参数只下载第一个）
  python bilibili_downloader.py --single

  # 不下载字幕
  python bilibili_downloader.py --no-subs

  # 将字幕嵌入到视频文件中
  python bilibili_downloader.py --embed-subs

  当前默认配置:
    链接: {DEFAULT_URL}
    输出: {_output_dir}
    画质: {DEFAULT_QUALITY}
    字幕: 自动下载（独立 .srt 文件）
        """,
    )

    # ── 主要参数 ──
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help=f"视频 URL 或 BV 号（不填则使用默认链接）",
    )
    parser.add_argument(
        "-p",
        "--playlist",
        action="store_true",
        help="强制下载整个播放列表/所有分P",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="强制单视频模式（合集 URL 也只下载第一个视频）",
    )
    parser.add_argument(
        "-q",
        "--quality",
        default=DEFAULT_QUALITY,
        choices=["bestvideo+bestaudio", "1080", "720"],
        help=f"下载画质（默认: {DEFAULT_QUALITY}，即最佳画质）",
    )
    parser.add_argument(
        "--collection",
        metavar="COLLECTION_ID",
        help="通过合集 ID 下载整个合集",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="仅查看视频信息，不下载",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="使用 B站 APP 扫描二维码登录，自动保存 Cookie",
    )
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        choices=["chrome", "firefox", "safari", "edge", "opera", "brave", "chromium"],
        help="从浏览器提取 B站 Cookie（最可靠的方式）。"
             "支持: chrome / firefox / safari / edge。"
             "先在浏览器登录 bilibili.com，然后运行此命令",
    )
    parser.add_argument(
        "--cookie-help",
        action="store_true",
        help="显示 Cookie 获取指引",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        default=None,
        help=f"自定义输出目录（默认: {_output_dir}）",
    )
    parser.add_argument(
        "--no-subs",
        action="store_true",
        help="不下载字幕（默认会下载 B站字幕为独立 .srt 文件）",
    )
    parser.add_argument(
        "--embed-subs",
        action="store_true",
        help="将字幕嵌入到视频文件中（而不是保留独立 .srt 文件）",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="输出 yt-dlp 详细调试日志，帮助诊断字幕等问题",
    )

    args = parser.parse_args()

    # ── Cookie 帮助 ──
    if args.cookie_help:
        generate_cookie_instructions()
        return

    # ── 从浏览器提取 Cookie（最可靠的方式）──
    if args.cookies_from_browser:
        if not extract_cookies_from_browser(args.cookies_from_browser):
            sys.exit(1)
        print("\n  ✅ Cookie 提取成功！")
        # 如果只做登录，不下载，直接退出
        if not args.url and not args.collection:
            print(f"  💡 现在可以直接运行: python bilibili_downloader.py")
            return

    # ── 二维码登录 ──
    if args.login:
        if not qrcode_login():
            sys.exit(1)
        # 如果登录后没有指定下载目标，直接退出
        if not args.url and not args.collection:
            print("\n  ✅ 登录成功！Cookie 已保存，下次下载时将自动使用。")
            print(f"  💡 现在可以直接运行: python bilibili_downloader.py")
            return

    # ── 合集 ID 模式 ──
    if args.collection:
        downloader = BilibiliDownloader()
        if args.output:
            set_output_dir(Path(args.output))
        success = downloader.download_collection(args.collection, quality=args.quality)
        sys.exit(0 if success else 1)

    # ── 确定下载 URL ──
    url = args.url
    if not url:
        url = DEFAULT_URL
        logger.info(f"使用默认链接: {url}")

    # ── 智能判断播放列表模式 ──
    # 合集/列表 URL 默认下载全部，用户可加 --single 覆盖
    if args.single:
        is_playlist = False
        logger.info("模式: 单视频（--single 强制）")
    elif args.playlist or is_playlist_url(url):
        is_playlist = True
        logger.info("模式: 合集/播放列表（下载全部视频）")
    else:
        is_playlist = False
        logger.info("模式: 单视频")

    # ── 创建下载器 ──
    downloader = BilibiliDownloader()

    # ── 自定义输出目录 ──
    if args.output:
        set_output_dir(Path(args.output))

    # ── 合集/播放列表模式：提取视频列表，逐个按单视频下载 ──
    #     这样每个视频都能拿到完整的字幕信息
    if is_playlist and not args.info:
        _download_collection_as_individuals(
            downloader, url, args.quality,
            download_subs=not args.no_subs,
            embed_subs=args.embed_subs,
            verbose=args.verbose,
        )
        return

    # ── 执行下载 ──
    success = downloader.download_video(
        url=url,
        is_playlist=is_playlist,
        quality=args.quality,
        info_only=args.info,
        download_subs=not args.no_subs,
        embed_subs=args.embed_subs,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)


def _download_collection_as_individuals(
    downloader: "BilibiliDownloader",
    url: str,
    quality: str,
    download_subs: bool = True,
    embed_subs: bool = False,
    verbose: bool = False,
):
    """合集下载：先提取所有视频 URL，再逐个按单视频模式下载（确保字幕提取）"""
    global _output_dir
    logger.info("正在获取合集视频列表...")

    # 用扁平提取快速获取视频列表（只拿 URL/标题，不深入提取每个视频页面）
    extract_opts = {
        "quiet": not verbose,
        "no_warnings": not verbose,
        "verbose": verbose,
        "noplaylist": False,
        "extract_flat": "in_playlist",
        "cookiefile": str(COOKIE_FILE) if COOKIE_FILE.exists() else None,
    }

    try:
        with downloader.yt_dlp.YoutubeDL(extract_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.error(f"获取合集信息失败: {e}")
        logger.info("回退到常规合集下载模式...")
        success = downloader.download_video(
            url=url, is_playlist=True, quality=quality,
            info_only=False, download_subs=download_subs,
            embed_subs=embed_subs, verbose=verbose,
        )
        sys.exit(0 if success else 1)

    entries = info.get("entries", [])
    if not entries:
        logger.error("未从合集中获取到任何视频，请检查链接是否有效")
        sys.exit(1)

    total = len(entries)
    playlist_title = re.sub(r'[<>:"/\\|?*]', '_', info.get("title", "合集").strip())
    playlist_dir = _output_dir / playlist_title
    playlist_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"合集「{playlist_title}」共 {total} 个视频")
    logger.info(f"输出目录: {playlist_dir}")

    # 字幕需要登录态提醒
    if download_subs and not COOKIE_FILE.exists():
        logger.warning("⚠️  B站字幕需要登录才能下载，请先运行 --login 扫码登录")

    # 逐个下载
    success_count = 0
    fail_list = []

    for i, entry in enumerate(entries, 1):
        video_url = entry.get("webpage_url") or entry.get("url") or entry.get("original_url")
        video_id = entry.get("id") or entry.get("display_id") or ""
        video_title = entry.get("title", f"video_{i}")

        if not video_url:
            # 尝试用 id 构造视频URL
            if video_id:
                if video_id.startswith("BV"):
                    video_url = f"https://www.bilibili.com/video/{video_id}"
                else:
                    video_url = f"https://www.bilibili.com/video/av{video_id.replace('av', '')}"
            else:
                logger.warning(f"[{i}/{total}] 跳过：无法获取链接")
                fail_list.append(f"#{i} {video_title} (无链接)")
                continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{total}] {video_title}")
        logger.info(f"         {video_url}")
        logger.info(f"{'='*60}")

        # 将输出目录临时切到合集子目录
        orig_dir = _output_dir
        set_output_dir(playlist_dir)

        # 带重试的下载（B站容易限流断开，重试通常能成功）
        max_retries = 3
        retry_delays = [3, 8, 15]  # 递增等待时长
        ok = False
        last_error = None

        for retry in range(max_retries):
            try:
                ok = downloader.download_video(
                    url=video_url,
                    is_playlist=False,
                    quality=quality,
                    info_only=False,
                    download_subs=download_subs,
                    embed_subs=embed_subs,
                    verbose=verbose,
                )
                if ok:
                    break
                # 下载失败但没有抛异常（yt-dlp 内部错误）
                last_error = "下载未成功"
            except (downloader.yt_dlp.utils.DownloadError,
                    downloader.yt_dlp.utils.ExtractorError) as e:
                last_error = str(e)
                # 非网络错误不重试
                error_str = str(e).lower()
                if any(kw in error_str for kw in
                       ["private", "deleted", "copyright", "region", "removed",
                        "not found", "404", "forbidden", "unavailable"]):
                    logger.error(f"  该视频 {error_str.split()[0]}，跳过不重试")
                    ok = False
                    break
            except Exception as e:
                last_error = str(e)

            if retry < max_retries - 1:
                delay = retry_delays[retry]
                logger.info(f"  🔄 第 {retry+1} 次重试（{delay}s 后）: {last_error}")
                time.sleep(delay)
            else:
                logger.error(f"  ❌ 重试 {max_retries} 次后仍然失败: {last_error}")
        else:
            # 所有重试耗尽
            ok = False

        # 恢复全局输出目录
        _output_dir = orig_dir

        # 视频间延迟（防止触碰到 B站 的请求频率限制）
        if i < total:
            delay = random.uniform(2, 5)
            logger.info(f"  ⏸  等待 {delay:.1f}s 后继续下一视频...")
            time.sleep(delay)

        if ok:
            success_count += 1
        else:
            fail_list.append(f"#{i} {video_title}")

    # 总结
    logger.info(f"\n{'='*60}")
    logger.info(f"合集下载完成: {success_count}/{total} 个视频成功")
    if fail_list:
        logger.warning(f"失败列表:")
        for f in fail_list:
            logger.warning(f"  - {f}")
    logger.info(f"文件保存在: {playlist_dir}")
    logger.info(f"{'='*60}")

    sys.exit(0 if success_count > 0 else 1)


if __name__ == "__main__":
    main()
