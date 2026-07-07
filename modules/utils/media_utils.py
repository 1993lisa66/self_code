"""
媒体文件处理工具模块
提供统一的媒体时长获取、FFmpeg 命令执行等功能
"""
import os
import re
import subprocess
from typing import Optional
from loguru import logger

from modules.utils.ffmpeg_utils import get_ffprobe_exe

# 初始化 ffprobe 路径
ffprobe_exe = get_ffprobe_exe()


def get_media_duration(media_path: str, use_pydub: bool = True) -> Optional[float]:
    """
    获取媒体文件（视频/音频）的时长（秒）
    
    Args:
        media_path: 媒体文件路径
        use_pydub: 是否优先使用 pydub（默认 True）
        
    Returns:
        媒体时长（秒），失败返回 None
        
    Examples:
        >>> duration = get_media_duration("video.mp4")
        >>> print(f"时长: {duration:.2f}s")
    """
    try:
        # 确保路径是绝对路径
        media_path = os.path.abspath(media_path)
        
        if not os.path.exists(media_path):
            logger.error(f"媒体文件不存在: {media_path}")
            return None
        
        # 检查文件大小
        file_size = os.path.getsize(media_path)
        if file_size == 0:
            logger.error(f"媒体文件为空: {media_path}")
            return None
        
        # 方法1: 尝试使用 pydub (更可靠)
        if use_pydub:
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(media_path)
                duration = len(audio) / 1000.0  # 转换为秒
                logger.debug(f"媒体文件时长 (pydub): {os.path.basename(media_path)} -> {duration:.2f}s")
                return duration
            except Exception as e:
                logger.debug(f"Pydub 获取时长失败: {e}，尝试 ffprobe...")
        
        # 方法2: 使用 ffprobe 的标准命令
        cmd = [
            ffprobe_exe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            media_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            logger.debug(f"媒体文件时长 (ffprobe): {os.path.basename(media_path)} -> {duration:.2f}s")
            return duration
        
        # 方法3: 从 ffprobe stderr 中解析时长（兼容性方案）
        logger.debug(f"ffprobe 标准命令失败，尝试解析 stderr...")
        cmd_fallback = [ffprobe_exe, "-i", media_path]
        result_fallback = subprocess.run(
            cmd_fallback,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', result_fallback.stderr)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = float(match.group(3))
            duration = hours * 3600 + minutes * 60 + seconds
            logger.debug(f"媒体文件时长 (解析): {os.path.basename(media_path)} -> {duration:.2f}s")
            return duration
        
        logger.error(f"无法获取媒体时长: {media_path}")
        return None
        
    except Exception as e:
        logger.error(f"获取媒体时长失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def run_ffmpeg_command(cmd: list, description: str = "FFmpeg 命令", 
                       capture_output: bool = True, check: bool = False) -> subprocess.CompletedProcess:
    """
    运行 FFmpeg/FFprobe 命令的统一接口
    
    Args:
        cmd: 命令列表
        description: 命令描述（用于日志）
        capture_output: 是否捕获输出
        check: 如果命令失败是否抛出异常
        
    Returns:
        subprocess.CompletedProcess 对象
        
    Examples:
        >>> cmd = ["ffmpeg", "-i", "input.mp4", "output.avi"]
        >>> result = run_ffmpeg_command(cmd, "视频转换")
        >>> if result.returncode != 0:
        ...     print("转换失败")
    """
    logger.debug(f"执行 {description}: {' '.join(cmd[:5])}...")
    
    # 为多进程环境准备环境变量
    env = os.environ.copy()
    
    # 确保 FFmpeg bin 目录在 PATH 中（程序使用包装脚本，已内置 DYLD 处理）
    ffmpeg_dir = os.path.dirname(cmd[0]) if cmd else ""
    if ffmpeg_dir and os.path.exists(ffmpeg_dir):
        env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    
    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,  # 始终使用文本模式，避免中文路径 bytes 编码问题
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        env=env  # 使用自定义环境变量
    )
    
    if result.returncode != 0:
        error_msg = f"{description} 失败 (返回码: {result.returncode})"
        if capture_output and result.stderr:
            stderr_text = result.stderr.decode('utf-8', errors='replace') if isinstance(result.stderr, bytes) else result.stderr
            # 输出末尾部分（跳过 FFmpeg 版本横幅，定位真正的错误）
            error_msg += f"\n错误信息（末尾 2000 字）:\n{stderr_text[-2000:]}"
        
        logger.error(error_msg)
        
        if check:
            raise RuntimeError(error_msg)
    
    return result


def validate_media_file(path: str, min_size: int = 100) -> bool:
    """
    验证媒体文件是否存在且有效
    
    Args:
        path: 文件路径
        min_size: 最小文件大小（字节）
        
    Returns:
        文件是否有效
    """
    if not path or not os.path.exists(path):
        return False
    
    file_size = os.path.getsize(path)
    return file_size >= min_size


def cleanup_directory(directory: str, keep_dir: bool = False) -> dict:
    """
    清理目录及其内容
    
    Args:
        directory: 要清理的目录路径
        keep_dir: 是否保留目录本身（只删除内容）
        
    Returns:
        包含清理信息的字典 {deleted_files, deleted_size_mb}
    """
    result = {
        'deleted_files': 0,
        'deleted_size_mb': 0.0,
        'success': False
    }
    
    if not os.path.exists(directory):
        return result
    
    try:
        # 计算大小和文件数
        total_size = 0
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
                file_count += 1
        
        result['deleted_files'] = file_count
        result['deleted_size_mb'] = total_size / (1024 * 1024)
        
        # 删除目录
        if keep_dir:
            # 只删除内容，保留目录
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
        else:
            # 删除整个目录
            import shutil
            shutil.rmtree(directory)
        
        result['success'] = True
        return result
        
    except Exception as e:
        logger.error(f"清理目录失败: {e}")
        result['success'] = False
        return result
