#!/usr/bin/env python3
"""B站视频下载器 — 命令行入口"""

import sys
import logging
from pathlib import Path

from modules.bilibili.config import (
    PROJECT_ROOT, LOG_FILE, get_output_dir, set_output_dir as _set_output_dir, _get,
)
from modules.bilibili.utils import is_playlist_url
from modules.bilibili.cookies import generate_cookie_instructions
from modules.bilibili.downloader import BilibiliDownloader, download_collection_as_individuals

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_URL = _get("default.url", "")
DEFAULT_OUTPUT_DIR_RAW = _get("default.output_dir", "")
DEFAULT_QUALITY = _get("default.quality", "bestvideo+bestaudio")


def main():
    import argparse

    if DEFAULT_OUTPUT_DIR_RAW and Path(DEFAULT_OUTPUT_DIR_RAW).exists():
        output_dir = Path(DEFAULT_OUTPUT_DIR_RAW)
    else:
        fallback = PROJECT_ROOT / _get("paths.output_dir", "outputs/bilibili")
        output_dir = Path(fallback)
    _set_output_dir(output_dir)

    parser = argparse.ArgumentParser(
        description="哔哩哔哩（Bilibili）视频下载器 — 基于 yt-dlp + FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
使用示例:
  # 直接运行（使用 bilibili_config.yaml 中的默认链接）
  python bilibili_cli.py

  # 下载单个视频
  python bilibili_cli.py BV1xx411c7mD

  # 仅下载字幕（不下载视频）
  python bilibili_cli.py BV1xx411c7mD --download-mode subs_only

  # 仅下载视频（不下载字幕）
  python bilibili_cli.py BV1xx411c7mD --download-mode video_only

  # 下载整个播放列表
  python bilibili_cli.py https://www.bilibili.com/video/BV1xx411c7mD --playlist

  # 仅查看视频信息
  python bilibili_cli.py --info

  # 自定义输出目录
  python bilibili_cli.py -o /path/to/output
        """,
    )

    parser.add_argument("url", nargs="?", default=None, help="视频 URL 或 BV 号")
    parser.add_argument("-p", "--playlist", action="store_true", help="强制下载整个播放列表")
    parser.add_argument("--single", action="store_true", help="强制单视频模式")
    parser.add_argument("-q", "--quality", default=DEFAULT_QUALITY,
                        choices=["bestvideo+bestaudio", "1080", "720"],
                        help=f"下载画质（默认: {DEFAULT_QUALITY}）")
    parser.add_argument("--collection", metavar="COLLECTION_ID", help="通过合集 ID 下载")
    parser.add_argument("--info", action="store_true", help="仅查看视频信息，不下载")
    parser.add_argument("--cookie-help", action="store_true", help="显示 Cookie 获取指引")
    parser.add_argument("-o", "--output", metavar="DIR", help="自定义输出目录")
    parser.add_argument("--download-mode", default="full",
                        choices=["full", "subs_only", "video_only"],
                        help="下载模式: full(完整) / subs_only(仅字幕) / video_only(仅视频)  (默认: full)")
    parser.add_argument("--embed-subs", action="store_true", help="将字幕嵌入到视频文件中")
    parser.add_argument("--danmaku", action="store_true", help="下载弹幕文件（默认不下载）")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细调试日志")

    args = parser.parse_args()

    if args.cookie_help:
        generate_cookie_instructions()
        return

    if args.collection:
        downloader = BilibiliDownloader()
        if args.output:
            _set_output_dir(Path(args.output))
        success = downloader.download_collection(args.collection, quality=args.quality)
        sys.exit(0 if success else 1)

    url = args.url or DEFAULT_URL
    if not args.url:
        logger.info(f"使用默认链接: {url}")

    if args.single:
        is_playlist = False
        logger.info("模式: 单视频（--single 强制）")
    elif args.playlist or is_playlist_url(url):
        is_playlist = True
        logger.info("模式: 合集/播放列表")
    else:
        is_playlist = False
        logger.info("模式: 单视频")

    downloader = BilibiliDownloader()

    if args.output:
        _set_output_dir(Path(args.output))

    if is_playlist and not args.info:
        download_collection_as_individuals(
            downloader, url, args.quality,
            embed_subs=args.embed_subs,
            skip_danmaku=not args.danmaku,
            verbose=args.verbose,
            download_mode=args.download_mode,
        )
        return

    success = downloader.download_video(
        url=url, is_playlist=is_playlist, quality=args.quality,
        info_only=args.info,
        embed_subs=args.embed_subs, skip_danmaku=not args.danmaku,
        verbose=args.verbose,
        download_mode=args.download_mode,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
