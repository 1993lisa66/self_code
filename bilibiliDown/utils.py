"""工具函数：FFmpeg 检查、BV 号处理、URL 解析"""

import re
import subprocess
import logging
from pathlib import Path
from typing import Optional

from .config import FFMPEG_BIN, PROJECT_ROOT

logger = logging.getLogger(__name__)


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


def is_playlist_url(url: str) -> bool:
    """检测 URL 是否为合集/播放列表/番剧"""
    patterns = [
        r"space\.bilibili\.com/\d+/lists/\d+",
        r"bilibili\.com/bangumi/play/",
        r"bilibili\.com/video/BV[a-zA-Z0-9]{10}\?p=",
        r"channel/collectiondetail",
    ]
    return any(re.search(p, url) for p in patterns)


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def format_speed(speed_bytes: float) -> str:
    """格式化下载速度"""
    for unit in ["B/s", "KB/s", "MB/s", "GB/s"]:
        if speed_bytes < 1024:
            return f"{speed_bytes:.1f}{unit}"
        speed_bytes /= 1024
    return f"{speed_bytes:.1f}TB/s"
