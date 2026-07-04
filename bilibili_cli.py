#!/usr/bin/env python3
"""B站视频下载器"""

import sys, logging
from pathlib import Path

from modules.bilibili.config import PROJECT_ROOT, LOG_FILE, set_output_dir, _get
from modules.bilibili.utils import is_playlist_url
from modules.bilibili.downloader import BilibiliDownloader, download_collection_as_individuals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main(url: str, output: str = "", quality: str = "", download_mode: str = ""):
    output = output or _get("default.output_dir", "") or str(PROJECT_ROOT / "outputs/bilibili")
    set_output_dir(Path(output))

    quality = quality or _get("default.quality", "bestvideo+bestaudio")
    download_mode = download_mode or "full"

    downloader = BilibiliDownloader()

    if is_playlist_url(url):
        logger.info("模式: 合集/播放列表")
        download_collection_as_individuals(downloader, url, quality, download_mode=download_mode)
    else:
        logger.info("模式: 单视频")
        success = downloader.download_video(url=url, quality=quality, download_mode=download_mode)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    # 在这里设置要下载的视频链接和输出路径
    url = "https://space.bilibili.com/40201146/lists/8467061?type=season"
    output = "/Volumes/mvp/[00]交易场"

    main(url, output)
