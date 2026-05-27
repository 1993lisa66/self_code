import os
import subprocess
from loguru import logger
from modules.utils.ffmpeg_utils import get_ffmpeg_exe

ffmpeg_exe = get_ffmpeg_exe()

def extract_audio(video_path, output_dir="cache/audio", sample_rate=16000):
    """
    从视频中提取音频并转换为 ASR 友好的格式 (16kHz, mono, wav)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.wav")

    # 缓存检查
    if os.path.exists(output_path):
        logger.info(f"使用已存在的音频缓存: {output_path}")
        return output_path

    logger.info(f"正在提取音频: {video_path} -> {output_path}")

    try:
        # 使用 ffmpeg 提取音频
        cmd = [
            ffmpeg_exe, "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-af", "loudnorm",
            output_path
        ]
        
        # 运行命令，捕获输出（使用字节模式避免编码问题）
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        if result.returncode != 0:
            # 手动解码错误输出
            stderr_text = result.stderr.decode('utf-8', errors='replace') if result.stderr else "Unknown error"
            logger.error(f"FFmpeg 报错: {stderr_text[:500]}")
            raise Exception(f"FFmpeg 提取音频失败")

        logger.success(f"音频提取完成: {output_path}")
        return output_path

    except FileNotFoundError:
        logger.error("系统未找到 ffmpeg 命令，请确保已安装并添加到环境变量。")
        raise Exception("ffmpeg not found")
    except Exception as e:
        logger.error(f"提取音频时发生未知错误: {e}")
        raise e
