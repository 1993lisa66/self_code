"""核心下载器 — BilibiliDownloader 类"""

import os
import sys
import re
import time
import random
import logging
from pathlib import Path

from .config import (
    FFMPEG_BIN, COOKIE_FILE, get_output_dir, set_output_dir as _set_output_dir,
    _get,
)
from .utils import (
    check_ffmpeg, is_valid_bv, build_url, format_size, format_speed,
)

logger = logging.getLogger(__name__)


class BilibiliDownloader:
    """
    哔哩哔哩视频下载器
    基于 yt-dlp 实现，支持音视频分离流的自动下载与 FFmpeg 合并。
    """

    def __init__(self):
        try:
            import yt_dlp
            self.yt_dlp = yt_dlp
        except ImportError:
            logger.error("yt-dlp 未安装，请执行: pip install yt-dlp")
            sys.exit(1)

        if not check_ffmpeg():
            logger.error("FFmpeg 不可用，无法合并音视频轨道")
            sys.exit(1)

    # ── 构建 yt-dlp 选项 ─────────────────────
    def _build_opts(
        self,
        is_playlist: bool = False,
        quality: str = "bestvideo+bestaudio",
        download_subs: bool = True,
        embed_subs: bool = False,
        verbose: bool = False,
    ) -> dict:
        _output_dir = get_output_dir()
        if is_playlist:
            output_template = str(
                _output_dir / "%(playlist_title)s" / "%(playlist_index)03d_%(title)s.%(ext)s"
            )
        else:
            output_template = str(_output_dir / "%(title)s_%(id)s.%(ext)s")

        opts = {
            "format": f"{quality}/best",
            "merge_output_format": "mp4",
            "outtmpl": output_template,
            "overwrites": False,
            "paths": {"home": str(_output_dir)},
            "ffmpeg_location": str(FFMPEG_BIN) if FFMPEG_BIN.exists() else None,
            "progress_hooks": [self._progress_hook],
            "quiet": False,
            "no_warnings": False,
            # 网络设置（从 config.yaml 读取）
            "retries": _get("network.retries", 10),
            "fragment_retries": _get("network.fragment_retries", 10),
            "ignoreerrors": is_playlist,
            "continuedl": True,
            "concurrent_fragment_downloads": _get("network.concurrent_fragment_downloads", 5),
            "sleep_interval_requests": _get("network.sleep_interval_requests", 1),
            "sleep_interval": _get("network.sleep_interval", 1),
            "max_sleep_interval_requests": _get("network.max_sleep_interval_requests", 3),
            "socket_timeout": _get("network.socket_timeout", 30),
            # 元数据
            "writedescription": False,
            "writeinfojson": False,
            "restrictfilenames": False,
            # 登录态
            "cookiefile": str(COOKIE_FILE) if COOKIE_FILE.exists() else None,
            # 合集/分P
            "playlistend": 0 if not is_playlist else None,
            "noplaylist": not is_playlist,
            # 字幕
            "writesubtitles": download_subs,
            "writeautomaticsub": download_subs,
            "subtitleslangs": _get("subtitle.languages", ["all"]),
            "subtitlesformat": _get("subtitle.format", "srt"),
            # 用户代理
            "user_agent": _get("user_agent", ""),
        }
        if embed_subs:
            opts["embedsubs"] = True
        return opts

    # ── 进度回调 ─────────────────────────────
    def _progress_hook(self, d: dict):
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            speed = d.get("speed") or 0
            eta = d.get("eta", "?")
            percent = d.get("_percent_str", "0%").strip()
            filename = os.path.basename(d.get("filename", ""))

            speed_str = format_speed(speed) if speed else "N/A"
            if total:
                progress = f"{percent} [{format_size(downloaded)}/{format_size(total)}]"
            else:
                progress = f"{format_size(downloaded)} 已下载"

            print(
                f"\r  📥 {filename[:50]:<50} {progress} 速度:{speed_str} ETA:{eta}s",
                end="",
            )
        elif status == "finished":
            print()
            logger.info(f"✅ 下载完成: {os.path.basename(d.get('filename', ''))}")

    # ── 获取视频信息 ─────────────────────────
    def get_video_info(self, url: str, verbose: bool = False) -> dict:
        opts = {
            "quiet": not verbose,
            "no_warnings": not verbose,
            "verbose": verbose,
            "cookiefile": str(COOKIE_FILE) if COOKIE_FILE.exists() else None,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["all"],
        }
        with self.yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if "entries" in info and info.get("entries"):
                info["entry_count"] = len(info["entries"])
                entries = info["entries"]
                info["entries_preview"] = [
                    {"title": e.get("title", "未知"),
                     "duration": e.get("duration", 0),
                     "id": e.get("id", "")}
                    for e in entries[:5]
                ]
            return info

    # ── 核心下载 ─────────────────────────────
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
        if is_valid_bv(url):
            url = build_url(url)
        elif not url.startswith("http"):
            logger.error(f"无效的输入: {url}")
            return False

        format_map = {
            "bestvideo+bestaudio": "bestvideo+bestaudio",
            "1080": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "720": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        }
        fmt = format_map.get(quality, quality)

        if info_only:
            try:
                info = self.get_video_info(url, verbose=verbose)
                self._print_video_info(info)
                return True
            except Exception as e:
                logger.error(f"获取视频信息失败: {e}")
                return False

        opts = self._build_opts(is_playlist=is_playlist, quality=fmt,
                                download_subs=download_subs,
                                embed_subs=embed_subs, verbose=verbose)

        try:
            with self.yt_dlp.YoutubeDL(opts) as ydl:
                logger.info(f"开始下载: {url}")
                mode = "合集" if is_playlist else "单视频"
                logger.info(f"模式: {mode}")
                ydl.download([url])
                logger.info("=" * 60)
                logger.info(f"下载成功！文件保存在: {get_output_dir()}")
                return True
        except self.yt_dlp.utils.DownloadError as e:
            logger.error(f"下载失败 (DownloadError): {e}")
        except self.yt_dlp.utils.ExtractorError as e:
            logger.error(f"解析失败 (ExtractorError): {e}")
            logger.error("提示: 该视频可能需要登录才能访问")
        except Exception as e:
            logger.error(f"未知错误: {type(e).__name__}: {e}")
        return False

    # ── 合集下载 ─────────────────────────────
    def download_collection(self, collection_id: str,
                            quality: str = "bestvideo+bestaudio") -> bool:
        collection_url = f"https://space.bilibili.com/0/channel/collectiondetail?sid={collection_id}"
        logger.info(f"合集 ID: {collection_id}")
        return self.download_video(url=collection_url, is_playlist=True, quality=quality)

    # ── 打印视频信息 ─────────────────────────
    def _print_video_info(self, info: dict):
        print("\n" + "=" * 60)
        print("📺 视频信息")
        print("=" * 60)
        print(f"  标题    : {info.get('title', '未知')}")
        print(f"  ID      : {info.get('id', '未知')}")
        duration = info.get("duration", 0)
        if duration:
            print(f"  时长    : {duration // 60}分{duration % 60}秒")
        print(f"  上传者  : {info.get('uploader', '未知')}")

        if info.get("entry_count"):
            print(f"  分P数量 : {info['entry_count']}")
            print(f"  前5项预览:")
            for i, entry in enumerate(info.get("entries_preview", []), 1):
                d = entry.get("duration", 0)
                print(f"    {i}. {entry['title'][:40]}  ({d//60}:{d%60:02d})")

        formats = info.get("formats", [])
        if formats:
            print(f"\n  可用格式 (共{len(formats)}种):")
            for i, f in enumerate(formats[:8]):
                fid = f.get("format_id", "?")
                res = f.get("resolution") or f.get("height") or "?"
                vcodec = f.get("vcodec", "none")
                acodec = f.get("acodec", "none")
                fsize = f.get("filesize") or f.get("filesize_approx")
                size_str = format_size(fsize) if fsize else "未知"
                tag = ""
                if vcodec != "none" and acodec != "none":
                    tag = "[音画合一]"
                elif vcodec != "none":
                    tag = "[仅视频]"
                elif acodec != "none":
                    tag = "[仅音频]"
                print(f"    {fid:<8} {str(res):>8} {tag:<12} {size_str:>10}")
            if len(formats) > 8:
                print(f"    ... 还有 {len(formats) - 8} 种格式")

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


# ──────────────────────────────────────────────
# 合集下载辅助函数
# ──────────────────────────────────────────────
def download_collection_as_individuals(
    downloader: BilibiliDownloader,
    url: str,
    quality: str,
    download_subs: bool = True,
    embed_subs: bool = False,
    verbose: bool = False,
):
    """合集下载：先提取所有视频 URL，再逐个按单视频模式下载（确保字幕提取）"""
    from .config import get_output_dir as _gdir, set_output_dir as _sdir, COOKIE_FILE, _get

    _output_dir = _gdir()
    logger.info("正在获取合集视频列表...")

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
        logger.error("未从合集中获取到任何视频")
        sys.exit(1)

    total = len(entries)
    playlist_title = re.sub(r'[<>:"/\\|?*]', '_', info.get("title", "合集").strip())
    playlist_dir = _output_dir / playlist_title
    playlist_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"合集「{playlist_title}」共 {total} 个视频")
    logger.info(f"输出目录: {playlist_dir}")

    if download_subs and not COOKIE_FILE.exists():
        logger.warning("⚠️  B站字幕需要登录才能下载，请先运行 --cookies-from-browser")

    success_count = 0
    fail_list = []

    for i, entry in enumerate(entries, 1):
        video_url = entry.get("webpage_url") or entry.get("url") or entry.get("original_url")
        video_id = entry.get("id") or entry.get("display_id") or ""
        video_title = entry.get("title", f"video_{i}")

        if not video_url:
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

        orig_dir = _gdir()
        _sdir(playlist_dir)

        max_retries = _get("collection.max_retries", 3)
        retry_delays = _get("collection.retry_delays", [3, 8, 15])
        ok = False
        last_error = None

        for retry in range(max_retries):
            try:
                ok = downloader.download_video(
                    url=video_url, is_playlist=False, quality=quality,
                    info_only=False, download_subs=download_subs,
                    embed_subs=embed_subs, verbose=verbose,
                )
                if ok:
                    break
                last_error = "下载未成功"
            except (downloader.yt_dlp.utils.DownloadError,
                    downloader.yt_dlp.utils.ExtractorError) as e:
                last_error = str(e)
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

        _sdir(orig_dir)

        if ok:
            success_count += 1
        else:
            fail_list.append(f"#{i} {video_title}")

        if i < total:
            min_d = _get("collection.min_delay", 2.0)
            max_d = _get("collection.max_delay", 5.0)
            delay = random.uniform(min_d, max_d)
            logger.info(f"  ⏸  等待 {delay:.1f}s 后继续下一视频...")
            time.sleep(delay)

    logger.info(f"\n{'='*60}")
    logger.info(f"合集下载完成: {success_count}/{total} 个视频成功")
    if fail_list:
        logger.warning("失败列表:")
        for f in fail_list:
            logger.warning(f"  - {f}")
    logger.info(f"文件保存在: {playlist_dir}")
    logger.info(f"{'='*60}")
    sys.exit(0 if success_count > 0 else 1)
