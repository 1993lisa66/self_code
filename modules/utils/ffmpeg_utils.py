import os
import shutil
from loguru import logger


def _is_executable_file(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _candidate_names(base_name: str) -> list[str]:
    if os.name == "nt":
        return [f"{base_name}.exe"]
    return [base_name]

def get_ffmpeg_exe():
    """
    自动寻找并返回 FFmpeg 可执行文件的路径。
    优先级:
    1. 项目本地 ffmpeg 目录
    2. 系统环境变量 (PATH)
    3. imageio_ffmpeg 库提供的二进制文件
    4. 默认返回 "ffmpeg"
    """
    env_ffmpeg = os.environ.get("FFMPEG_BIN")
    if env_ffmpeg and (os.path.exists(env_ffmpeg) or shutil.which(env_ffmpeg)):
        return env_ffmpeg

    project_root = _project_root()
    local_ffmpeg_paths: list[str] = []
    for name in _candidate_names("ffmpeg"):
        local_ffmpeg_paths.extend(
            [
                os.path.join(project_root, "ffmpeg", "bin", name),
                os.path.join(project_root, "ffmpeg", name),
                os.path.join(project_root, name),
            ]
        )

    for p in local_ffmpeg_paths:
        if _is_executable_file(p) or (os.name == "nt" and os.path.exists(p)):
            logger.info(f"使用项目本地 FFmpeg: {p}")
            return p
    
    # 1. 尝试系统 PATH
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    
    # 2. 尝试 imageio_ffmpeg 的 binaries
    try:
        import imageio_ffmpeg
        imageio_ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if imageio_ffmpeg_path and os.path.exists(imageio_ffmpeg_path):
            return imageio_ffmpeg_path
    except ImportError:
        pass
    
    return "ffmpeg"

def get_ffprobe_exe():
    """
    自动寻找并返回 FFprobe 可执行文件的路径。
    优先级与 get_ffmpeg_exe 相同
    """
    env_ffprobe = os.environ.get("FFPROBE_BIN")
    if env_ffprobe and (os.path.exists(env_ffprobe) or shutil.which(env_ffprobe)):
        return env_ffprobe

    project_root = _project_root()
    local_ffprobe_paths: list[str] = []
    for name in _candidate_names("ffprobe"):
        local_ffprobe_paths.extend(
            [
                os.path.join(project_root, "ffmpeg", "bin", name),
                os.path.join(project_root, "ffmpeg", name),
                os.path.join(project_root, name),
            ]
        )

    for p in local_ffprobe_paths:
        if _is_executable_file(p) or (os.name == "nt" and os.path.exists(p)):
            logger.info(f"使用项目本地 FFprobe: {p}")
            return p
    
    # 1. 尝试系统 PATH
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe
    
    # 2. 尝试 imageio_ffmpeg 的 binaries 目录
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path:
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            for name in _candidate_names("ffprobe"):
                ffprobe_path = os.path.join(ffmpeg_dir, name)
                if os.path.exists(ffprobe_path):
                    return ffprobe_path
    except ImportError:
        pass
    
    return "ffprobe"
