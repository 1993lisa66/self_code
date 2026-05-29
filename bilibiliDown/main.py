#!/usr/bin/env python3
"""B站视频下载器 — 命令行入口"""

import sys
import logging
from pathlib import Path

from .config import (
    PROJECT_ROOT, LOG_FILE, get_output_dir, set_output_dir as _set_output_dir, _get,
)
from .utils import is_playlist_url
from .cookies import generate_cookie_instructions
from .downloader import BilibiliDownloader, download_collection_as_individuals

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

# ── 从 YAML 配置中读取默认值 ──
DEFAULT_URL = _get("default.url", "")
DEFAULT_OUTPUT_DIR_RAW = _get("default.output_dir", "")
DEFAULT_QUALITY = _get("default.quality", "bestvideo+bestaudio")


def main():
    import argparse

    # ── 初始化输出目录 ──
    if DEFAULT_OUTPUT_DIR_RAW and Path(DEFAULT_OUTPUT_DIR_RAW).exists():
        output_dir = Path(DEFAULT_OUTPUT_DIR_RAW)
    else:
        fallback = PROJECT_ROOT / _get("paths.output_dir", "../outputs/bilibili")
        output_dir = Path(fallback)
    _set_output_dir(output_dir)

    parser = argparse.ArgumentParser(
        description="哔哩哔哩（Bilibili）视频下载器 — 基于 yt-dlp + FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
使用示例:
  # 直接运行（使用 config.yaml 中的默认链接）
  python -m bilibiliDown

  # 下载单个视频
  python -m bilibiliDown BV1xx411c7mD

  # 下载整个播放列表
  python -m bilibiliDown https://www.bilibili.com/video/BV1xx411c7mD --playlist

  # 仅查看视频信息
  python -m bilibiliDown --info

  # 自定义输出目录
  python -m bilibiliDown -o /path/to/output

  # 不下载字幕
  python -m bilibiliDown --no-subs

  # 查看 Cookie 获取指引
  python -m bilibiliDown --cookie-help

当前配置 (config.yaml):
  默认链接: {DEFAULT_URL}
  默认输出: {output_dir}
  默认画质: {DEFAULT_QUALITY}
  字幕:     {'自动下载' if _get('subtitle.download', True) else '不下载'}
  配置文件: {PROJECT_ROOT / 'config.yaml'}
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
    parser.add_argument("--no-subs", action="store_true", help="不下载字幕")
    parser.add_argument("--embed-subs", action="store_true", help="将字幕嵌入到视频文件中")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细调试日志")

    args = parser.parse_args()

    # ── Cookie 帮助指引 ──
    if args.cookie_help:
        generate_cookie_instructions()
        return

    # ── 合集 ID 模式 ──
    if args.collection:
        downloader = BilibiliDownloader()
        if args.output:
            _set_output_dir(Path(args.output))
        success = downloader.download_collection(args.collection, quality=args.quality)
        sys.exit(0 if success else 1)

    # ── 确定下载 URL ──
    url = args.url or DEFAULT_URL
    if not args.url:
        logger.info(f"使用默认链接: {url}")

    # ── 判断模式 ──
    if args.single:
        is_playlist = False
        logger.info("模式: 单视频（--single 强制）")
    elif args.playlist or is_playlist_url(url):
        is_playlist = True
        logger.info("模式: 合集/播放列表（下载全部视频）")
    else:
        is_playlist = False
        logger.info("模式: 单视频")

    downloader = BilibiliDownloader()

    if args.output:
        _set_output_dir(Path(args.output))

    # ── 合集模式：逐个下载 ──
    if is_playlist and not args.info:
        download_collection_as_individuals(
            downloader, url, args.quality,
            download_subs=not args.no_subs,
            embed_subs=args.embed_subs,
            verbose=args.verbose,
        )
        return

    # ── 单视频下载 ──
    success = downloader.download_video(
        url=url, is_playlist=is_playlist, quality=args.quality,
        info_only=args.info, download_subs=not args.no_subs,
        embed_subs=args.embed_subs, verbose=args.verbose,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
