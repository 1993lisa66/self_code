"""项目配置 — 从 config.yaml 读取所有配置"""

import yaml
import sys
from pathlib import Path
from typing import Any, Optional

# ── 项目目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_YAML = PROJECT_ROOT / "config.yaml"

# ── 加载配置 ──
def _load_config() -> dict:
    """加载 YAML 配置文件"""
    if not CONFIG_YAML.exists():
        print(f"[FATAL] 配置文件不存在: {CONFIG_YAML}")
        sys.exit(1)
    try:
        with open(CONFIG_YAML, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[FATAL] 配置文件解析失败: {e}")
        sys.exit(1)

_cfg = _load_config()


def _get(key: str, default: Any = None) -> Any:
    """按点号路径读取配置，如 _get('network.retries')"""
    keys = key.split(".")
    val = _cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default


# ── 全局配置常量 ──
# 路径
PATHS = _get("paths", {})

_ffmpeg_bin_dir = PROJECT_ROOT / _get("paths.ffmpeg_dir", "../ffmpeg/bin")
FFMPEG_BIN = _ffmpeg_bin_dir / "ffmpeg_real"
FFPROBE_BIN = _ffmpeg_bin_dir / "ffprobe_real"

# Cookie 文件
COOKIE_FILE = PROJECT_ROOT / _get("cookie.file", "www.bilibili.com_cookies.txt")

# 日志
LOG_DIR = PROJECT_ROOT / _get("paths.log_dir", "../logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "bilibili_download.log"

# ── 运行时输出目录 ──
_output_dir: Optional[Path] = None


def get_output_dir() -> Path:
    return _output_dir


def set_output_dir(path: Path):
    global _output_dir
    _output_dir = Path(path)
    _output_dir.mkdir(parents=True, exist_ok=True)
