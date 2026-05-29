"""B站下载器模块 — 配置 / 工具 / Cookie / 下载器"""

from .config import (
    PROJECT_ROOT, COOKIE_FILE, FFMPEG_BIN, FFPROBE_BIN,
    LOG_DIR, LOG_FILE, get_output_dir, set_output_dir, _get,
)
from .utils import (
    check_ffmpeg, is_valid_bv, extract_bv_from_url,
    build_url, is_playlist_url, format_size, format_speed,
)
from .cookies import (
    cookie_file_exists, cookie_has_sessdata, generate_cookie_instructions,
)
from .downloader import BilibiliDownloader, download_collection_as_individuals

__all__ = [
    "PROJECT_ROOT", "COOKIE_FILE", "FFMPEG_BIN", "FFPROBE_BIN",
    "LOG_DIR", "LOG_FILE", "get_output_dir", "set_output_dir", "_get",
    "check_ffmpeg", "is_valid_bv", "extract_bv_from_url",
    "build_url", "is_playlist_url", "format_size", "format_speed",
    "cookie_file_exists", "cookie_has_sessdata", "generate_cookie_instructions",
    "BilibiliDownloader", "download_collection_as_individuals",
]
