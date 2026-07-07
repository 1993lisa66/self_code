import os
from loguru import logger
from modules.utils.ffmpeg_utils import get_ffmpeg_exe, get_ffprobe_exe
from modules.utils.media_utils import get_media_duration, run_ffmpeg_command

ffmpeg_exe = get_ffmpeg_exe()
ffprobe_exe = get_ffprobe_exe()




def merge_video(video_path, tts_audio, output_dir="outputs", config=None, output_name=None):
    """
    使用 FFmpeg 合成最终视频：原视频画面 + 新配音（不烧录字幕）。
    增加了视频时长自动补全逻辑，解决音画不同步导致的视频冻结问题。

    Args:
        video_path: 原始视频路径
        tts_audio: TTS 配音音频路径
        output_dir: 输出目录
        config: 合并配置字典
        output_name: 输出文件名（不含扩展名），若提供则使用此名称而非视频原名
    """
    video_path = os.path.abspath(video_path)
    tts_audio = os.path.abspath(tts_audio)
    output_dir = os.path.abspath(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_name = output_name if output_name else os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.mp4")

    # 获取时长（优先用 ffprobe，中文路径 pydub 可能失败）
    v_dur = get_media_duration(video_path, use_pydub=False) or 0
    a_dur = get_media_duration(tts_audio, use_pydub=False) or 0
    
    logger.info(f"视频时长: {v_dur:.2f}s, 音频时长: {a_dur:.2f}s")
    
    # 构建滤镜：如果音频比视频长，补齐视频最后一张图 (tpad)
    video_filters = []
    
    if a_dur > v_dur + 0.1:
        pad_dur = a_dur - v_dur
        logger.info(f"音频比视频长 {pad_dur:.2f}s，正在补齐视频尾部...")
        video_filters.append(f"tpad=stop_mode=clone:stop_duration={pad_dur}")

    audio_mode = config.get('audio_mode', 'tts_only') if config else 'tts_only'

    # 构建滤镜链
    v_filter = ",".join(video_filters) if video_filters else "copy"
    
    # 构建 FFmpeg 命令
    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-i", tts_audio,
    ]
    
    if audio_mode == "mix":
        # 混合模式：处理视频补齐 + 处理音频混合（原声+配音）
        filter_str = ""
        if video_filters:
            filter_str += f"[0:v]{v_filter}[outv];"
        else:
            filter_str += "[0:v]null[outv];"
            
        filter_str += "[0:a]volume=0.3[orig];[1:a]volume=1.0[dub];[orig][dub]amix=inputs=2:duration=longest[outa]"
        
        cmd.extend([
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-map", "[outa]"
        ])
    else:
        # tts_only 模式：仅中文配音
        if video_filters:
            cmd.extend([
                "-filter_complex", f"[0:v]{v_filter}[outv]",
                "-map", "[outv]",
                "-map", "1:a"
            ])
        else:
            cmd.extend([
                "-map", "0:v",
                "-map", "1:a"
            ])
    
    cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        output_path
    ])

    try:
        logger.info(f"运行合成指令...")
        result = run_ffmpeg_command(cmd, "视频合成", capture_output=True)
        
        if result.returncode != 0:
            stderr_text = result.stderr.decode('utf-8', errors='replace') if isinstance(result.stderr, bytes) else result.stderr
            logger.error(f"FFmpeg 合成指令执行失败，返回码: {result.returncode}")
            logger.error(f"FFmpeg 错误输出（末尾 2000 字）:\n{stderr_text[-2000:]}")
            raise Exception(f"FFmpeg 合成失败")

        logger.success(f"视频合成成功: {output_path}")
        return output_path

    except FileNotFoundError:
        logger.error("未找到 ffmpeg，无法合成。")
        raise Exception("ffmpeg not found")
    except Exception as e:
        logger.error(f"合成过程中发生错误: {e}")
        raise e
