import os
from loguru import logger
from modules.utils.ffmpeg_utils import get_ffmpeg_exe
from modules.utils.media_utils import run_ffmpeg_command

ffmpeg_exe = get_ffmpeg_exe()

def extract_audio(video_path, output_dir="cache/audio", sample_rate=16000, max_retries=2):
    """
    从视频中提取音频并转换为 ASR 友好的格式 (16kHz, mono, wav)
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        sample_rate: 采样率
        max_retries: 最大重试次数
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

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            # 首先检查视频是否有音频流
            from modules.utils.ffmpeg_utils import get_ffprobe_exe
            ffprobe_exe = get_ffprobe_exe()
            
            check_cmd = [
                ffprobe_exe, "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                video_path
            ]
            
            import subprocess
            check_result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                env=os.environ.copy()
            )
            
            has_audio = check_result.stdout.strip() != ""
            if not has_audio:
                logger.warning(f"视频文件没有音频流: {video_path}")
                logger.info("创建静音音频文件作为占位")
                
                # 获取视频时长
                duration_cmd = [
                    ffprobe_exe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    video_path
                ]
                duration_result = subprocess.run(
                    duration_cmd,
                    capture_output=True,
                    text=True,
                    env=os.environ.copy()
                )
                
                try:
                    duration = float(duration_result.stdout.strip())
                except:
                    duration = 60.0  # 默认 60 秒
                
                # 生成静音音频
                silence_cmd = [
                    ffmpeg_exe, "-y",
                    "-f", "lavfi",
                    "-i", f"anullsrc=r={sample_rate}:cl=mono",
                    "-t", str(duration),
                    "-ac", "1",
                    "-ar", str(sample_rate),
                    output_path
                ]
                
                silence_result = subprocess.run(
                    silence_cmd,
                    capture_output=True,
                    env=os.environ.copy()
                )
                
                if silence_result.returncode == 0 and os.path.exists(output_path):
                    logger.success(f"已创建静音音频占位文件: {output_path} (时长: {duration:.2f}s)")
                    return output_path
                else:
                    raise Exception("创建静音音频失败")
            
            # 使用 ffmpeg 提取音频（对路径进行转义处理）
            cmd = [
                ffmpeg_exe, "-y",
                "-i", video_path,
                "-vn",
                "-ac", "1",
                "-ar", str(sample_rate),
                "-af", "loudnorm",
                output_path
            ]
            
            # 运行命令，捕获输出
            result = run_ffmpeg_command(cmd, f"音频提取: {base_name}", capture_output=True)
            
            if result.returncode != 0:
                stderr_text = result.stderr.decode('utf-8', errors='replace') if isinstance(result.stderr, bytes) else result.stderr
                last_error = stderr_text
                
                if attempt < max_retries:
                    logger.warning(f"FFmpeg 提取失败 (尝试 {attempt + 1}/{max_retries + 1}): {stderr_text[:200]}")
                    # 清理可能的损坏文件
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                        except:
                            pass
                    import time
                    time.sleep(1)  # 等待1秒后重试
                    continue
                else:
                    logger.error(f"FFmpeg 报错详情:\n{stderr_text[:1000]}")
                    raise Exception(f"FFmpeg 提取音频失败 (返回码: {result.returncode})")

            logger.success(f"音频提取完成: {output_path}")
            return output_path

        except FileNotFoundError:
            logger.error("系统未找到 ffmpeg 命令，请确保已安装并添加到环境变量。")
            raise Exception("ffmpeg not found")
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                logger.warning(f"提取音频异常 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                import time
                time.sleep(1)
                continue
            else:
                logger.error(f"提取音频时发生未知错误: {e}")
                raise e
    
    # 所有重试都失败
    raise Exception(f"FFmpeg 提取音频失败，已重试 {max_retries} 次。最后错误: {last_error}")
