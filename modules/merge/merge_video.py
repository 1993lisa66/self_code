import os
import subprocess
from loguru import logger
from modules.utils.ffmpeg_utils import get_ffmpeg_exe, get_ffprobe_exe

ffmpeg_exe = get_ffmpeg_exe()
ffprobe_exe = get_ffprobe_exe()


def get_duration(path):
    """获取媒体时长"""
    try:
        cmd = [
            ffprobe_exe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode != 0:
            return 0
        return float(result.stdout.strip() or 0)
    except Exception:
        return 0

def merge_video(video_path, tts_audio, srt_path, output_dir="outputs", config=None):
    """
    使用 FFmpeg 合成最终视频：原视频画面 + 新配音 + 字幕。
    增加了视频时长自动补全逻辑，解决音画不同步导致的视频冻结或字幕不更新问题。
    """
    video_path = os.path.abspath(video_path)
    tts_audio = os.path.abspath(tts_audio)
    srt_path = os.path.abspath(srt_path)
    output_dir = os.path.abspath(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_translated.mp4")

    # 获取时长
    v_dur = get_duration(video_path)
    a_dur = get_duration(tts_audio)
    
    logger.info(f"视频时长: {v_dur:.2f}s, 音频时长: {a_dur:.2f}s")
    
    # 构建滤镜：如果音频比视频长，补齐视频最后一张图 (tpad)
    video_filters = []
    
    if a_dur > v_dur + 0.1:
        pad_dur = a_dur - v_dur
        logger.info(f"音频比视频长 {pad_dur:.2f}s，正在补齐视频尾部...")
        video_filters.append(f"tpad=stop_mode=clone:stop_duration={pad_dur}")
    
    # 读取配置
    burn_subtitles = config.get('burn_subtitles', True) if config else True
    subtitle_position = config.get('subtitle_position', 'bottom') if config else 'bottom'
    subtitle_font_size = config.get('subtitle_font_size', 18) if config else 18
    audio_mode = config.get('audio_mode', 'tts_only') if config else 'tts_only'
    
    # 字幕滤镜
    if burn_subtitles:
        srt_path_fixed = srt_path.replace("\\", "/").replace(":", "\\:")
        force_style = ""
        if subtitle_position == "top":
            force_style = ",MarginV=50"
        elif subtitle_position == "center":
            force_style = ",Alignment=5"
        
        video_filters.append(f"subtitles='{srt_path_fixed}':force_style='Fontsize={subtitle_font_size}{force_style}'")

    # 构建滤镜链
    v_filter = ",".join(video_filters) if video_filters else "copy"
    
    # 构建 FFmpeg 命令
    cmd = [
        ffmpeg_exe, "-y",
        "-i", video_path,
        "-i", tts_audio,
    ]
    
    # 复杂的滤镜组合
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
        result = subprocess.run(cmd, capture_output=True, text=False)
        
        if result.returncode != 0:
            stderr_text = result.stderr.decode('utf-8', errors='replace')
            logger.error(f"FFmpeg 合成指令执行失败，返回码: {result.returncode}")
            logger.error(f"FFmpeg 错误输出:\n{stderr_text[:500]}")
            
            # 如果 subtitles 滤镜失败，尝试不带字幕合成
            if burn_subtitles:
                logger.warning("字幕烧录失败，尝试不带字幕进行合成...")
                cmd_no_sub = []
                skip_next = False
                for i, c in enumerate(cmd):
                    if skip_next:
                        skip_next = False
                        continue
                    if c == "-vf":
                        skip_next = True
                        continue
                    cmd_no_sub.append(c)
                
                result_no_sub = subprocess.run(cmd_no_sub, capture_output=True, text=False)
                if result_no_sub.returncode == 0:
                    logger.warning("已生成不带字幕的视频作为回退。")
                    return output_path
                raise Exception(f"FFmpeg 合成完全失败")
            else:
                raise Exception(f"FFmpeg 合成失败")

        logger.success(f"视频合成成功: {output_path}")
        
        # 如果不烧录字幕，生成多个版本
        if not burn_subtitles:
            logger.info("生成多版本输出文件...")
            
            # 版本1: {视频名}_translated.mp4 - 无字幕版本（已生成）
            logger.info(f"✓ 无字幕版本: {output_path}")
            
            # 版本2: {视频名}_with_external_sub.mp4 - 带软字幕轨道
            external_sub_path = os.path.join(output_dir, f"{base_name}_with_external_sub.mp4")
            logger.info(f"正在生成带软字幕轨道的版本: {external_sub_path}")
            
            cmd_ext = [
                ffmpeg_exe, "-y",
                "-i", output_path,
                "-i", srt_path,
                "-c:v", "copy",
                "-c:a", "copy",
                "-c:s", "mov_text",
                "-metadata:s:s:0", "language=chi",
                external_sub_path
            ]
            
            try:
                result_ext = subprocess.run(cmd_ext, capture_output=True, text=False)
                if result_ext.returncode == 0:
                    logger.success(f"✓ 软字幕版本生成成功: {external_sub_path}")
                else:
                    stderr_ext = result_ext.stderr.decode('utf-8', errors='replace')
                    logger.warning(f"软字幕版本生成失败: {stderr_ext[:200]}")
            except Exception as e:
                logger.warning(f"生成软字幕版本时出错: {e}")
            
            # 版本3: {视频名}.srt - 外挂字幕文件（已存在）
            logger.info(f"✓ 外挂字幕文件: {srt_path}")
            
            logger.info("\n输出文件说明:")
            logger.info(f"  1. {base_name}_translated.mp4 - 无字幕版本")
            logger.info(f"  2. {base_name}_with_external_sub.mp4 - 带软字幕轨道（可在播放器中开关）")
            logger.info(f"  3. {base_name}.srt - 外挂字幕文件（可单独分发）")
        else:
            logger.info(f"✓ 烧录字幕版本: {output_path}")
        
        return output_path

    except FileNotFoundError:
        logger.error("未找到 ffmpeg，无法合成。")
        raise Exception("ffmpeg not found")
    except Exception as e:
        logger.error(f"合成过程中发生错误: {e}")
        raise e
