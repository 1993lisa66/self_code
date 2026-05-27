import os
import json
import asyncio
import aiofiles
import edge_tts
import subprocess
from loguru import logger

# 必须在导入 pydub 之前设置 FFmpeg 路径
from modules.utils.ffmpeg_utils import get_ffmpeg_exe, get_ffprobe_exe
from modules.utils.media_utils import get_media_duration, validate_media_file

ffmpeg_exe = get_ffmpeg_exe()
ffprobe_exe = get_ffprobe_exe()

logger.info(f"TTS 模块使用的 FFmpeg: {ffmpeg_exe}")
logger.info(f"TTS 模块使用的 FFprobe: {ffprobe_exe}")

def get_media_duration(media_path):
    """
    获取媒体文件（视频/音频）的时长（秒）
    """
    try:
        # 确保路径是绝对路径
        media_path = os.path.abspath(media_path)
        
        if not os.path.exists(media_path):
            logger.error(f"媒体文件不存在: {media_path}")
            return None
        
        # 方法1: 尝试使用 pydub (更可靠)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(media_path)
            duration = len(audio) / 1000.0  # 转换为秒
            logger.info(f"媒体文件时长: {os.path.basename(media_path)} -> {duration:.2f}s")
            return duration
        except Exception as e:
            logger.debug(f"Pydub 获取时长失败: {e}，尝试 ffprobe...")
        
        # 方法2: 使用 ffprobe 的简化命令（兼容性更好）
        cmd = [
            ffprobe_exe, "-v", "error",
            "-i", media_path,
            "-show_entries", "format=duration",
            "-of", "csv=p=0"
        ]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        if result.returncode != 0:
            logger.error(f"ffprobe 执行失败: {result.stderr[:200]}")
            # 方法3: 最后的备用方案 - 使用 ffprobe 的基本输出
            try:
                cmd_fallback = [ffprobe_exe, "-i", media_path]
                result_fallback = subprocess.run(
                    cmd_fallback,
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                # 从 stderr 中解析时长
                import re
                match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', result_fallback.stderr)
                if match:
                    hours = int(match.group(1))
                    minutes = int(match.group(2))
                    seconds = float(match.group(3))
                    duration = hours * 3600 + minutes * 60 + seconds
                    logger.info(f"媒体文件时长 (解析): {os.path.basename(media_path)} -> {duration:.2f}s")
                    return duration
            except Exception as e2:
                logger.error(f"备用方案也失败: {e2}")
            return None
        
        # 解码输出并转换为浮点数
        duration_str = result.stdout.strip()
        if duration_str:
            duration = float(duration_str)
            logger.info(f"媒体文件时长: {os.path.basename(media_path)} -> {duration:.2f}s")
            return duration
        else:
            logger.error("ffprobe 返回空结果")
            return None
    except Exception as e:
        logger.error(f"获取媒体时长失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

async def _generate_edge_tts(text, voice, path, max_retries=2):
    """单条语音合成任务（带重试机制）"""
    for attempt in range(max_retries + 1):
        try:
            abs_path = os.path.abspath(path)
            # 确保目录存在
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            
            # 检查文本是否为空
            if not text or not text.strip():
                logger.error(f"文本为空，跳过合成: {path}")
                return False
            
            logger.debug(f"开始合成片段 {os.path.basename(path)}: '{text[:50]}...'")
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(abs_path)
            
            if validate_media_file(abs_path):
                logger.debug(f"片段 {os.path.basename(path)} 合成成功: {os.path.getsize(abs_path)} bytes")
                return True
            else:
                file_size = os.path.getsize(abs_path) if os.path.exists(abs_path) else 0
                if attempt < max_retries:
                    logger.warning(f"Edge-TTS 生成文件无效 ({file_size} bytes)，重试 {attempt + 1}/{max_retries}: {abs_path}")
                    continue
                else:
                    logger.error(f"Edge-TTS 生成文件无效 ({file_size} bytes, 可能网络问题或文本为空): {abs_path}")
                    return False
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Edge-TTS 合成失败 ({voice})，重试 {attempt + 1}/{max_retries}: {e}")
                await asyncio.sleep(1)  # 等待 1 秒后重试
            else:
                logger.error(f"Edge-TTS 合成失败 ({voice}): {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return False
    return False

def merge_audio_with_pydub(temp_files, output_path):
    """
    使用 pydub 进行音频合成。
    这种方法能自动处理采样率不一致问题，且生成的音频流更平滑，无卡顿。
    """
    if not temp_files:
        return False
    
    try:
        # 按照时间轴顺序排列片段
        sorted_files = sorted(temp_files, key=lambda x: x[0]['start'])
        
        logger.info(f"正在使用 Pydub 合成音频 (共 {len(sorted_files)} 个片段)...")
        
        # 初始化空白音频 (44.1kHz, 16-bit, Stereo)
        combined = AudioSegment.silent(duration=0, frame_rate=44100)
        
        for seg, path in sorted_files:
            if seg.get('is_gap'):
                # 处理静音间隙
                duration_ms = int((seg['end'] - seg['start']) * 1000)
                if duration_ms > 0:
                    combined += AudioSegment.silent(duration=duration_ms, frame_rate=44100)
            elif path and os.path.exists(path):
                # 处理 TTS 片段
                try:
                    segment_audio = AudioSegment.from_file(path)
                    # 统一采样率为 44.1kHz，通道数为 2，确保合成平滑
                    if segment_audio.frame_rate != 44100:
                        segment_audio = segment_audio.set_frame_rate(44100)
                    if segment_audio.channels != 2:
                        segment_audio = segment_audio.set_channels(2)
                    
                    combined += segment_audio
                except Exception as e:
                    logger.warning(f"加载音频片段失败 {path}: {e}")
                    # 失败则补充对应时长的静音，维持时间轴同步
                    duration_ms = int((seg['end'] - seg['start']) * 1000)
                    combined += AudioSegment.silent(duration=duration_ms, frame_rate=44100)
        
        # 导出为高质量 MP3
        combined.export(output_path, format="mp3", bitrate="192k")
        
        # 验证时长
        actual_duration = len(combined) / 1000.0
        logger.info(f"Pydub 合成完成，实际时长: {actual_duration:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"Pydub 合成发生错误: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

async def generate_tts(translated_results, output_dir="cache/tts", voice="zh-CN-XiaoxiaoNeural", config=None, original_video_path=None):
    """
    异步合成所有 TTS 片段并合并。
    
    Args:
        translated_results: 翻译结果列表
        output_dir: 输出目录
        voice: TTS 语音选择
        config: TTS 配置
        original_video_path: 原始视频路径（用于时长对齐）
    """
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    tasks = []
    temp_files_to_merge = []
    
    # 降低并发数以提高稳定性（从 10 降到 5）
    sem = asyncio.Semaphore(5)

    async def sem_task(task_info):
        async with sem:
            idx, text, path = task_info
            return await _generate_edge_tts(text, voice, path)

    logger.info(f"开始异步合成 {len(translated_results)} 段语音...")
    
    # 构建任务列表
    for i, seg in enumerate(translated_results):
        text = seg.get("tts_text") or seg.get("translated_text") or seg.get("text")
        if not text or not text.strip():
            logger.warning(f"跳过空文本片段 {i}")
            continue
            
        temp_path = os.path.join(output_dir, f"temp_{i}.mp3")
        temp_files_to_merge.append((seg, temp_path))
        tasks.append((i, text, temp_path))
    
    total_count = len(tasks)  # 修复：在添加任务后计算总数
    logger.info(f"有效任务数: {total_count}")

    # 并发运行（带进度显示）
    import sys
    
    completed_count = 0  # 移到正确的位置
    
    async def task_with_progress(task_info, idx_in_tasks):
        result = await sem_task(task_info)
        nonlocal completed_count
        completed_count += 1
        # 每完成 10% 或至少每 10 个显示一次进度
        progress_pct = (completed_count / total_count * 100) if total_count > 0 else 0
        if completed_count % max(1, total_count // 10) == 0 or completed_count == total_count:
            logger.info(f"TTS 合成进度: {completed_count}/{total_count} ({progress_pct:.1f}%)")
        return result
    
    results = await asyncio.gather(*(
        task_with_progress(t, idx) for idx, t in enumerate(tasks)
    ))
    
    # 统计成功数
    success_count = sum(1 for r in results if r)
    logger.info(f"TTS 合成完成: {success_count}/{len(tasks)} 成功")
    
    # 关键优化：采用自然时长，更新时间轴并插入静音间隙以保持同步，彻底解决语速过快和卡顿问题
    valid_temp_files = []
    skipped_count = 0
    current_time = 0.0  # 用于追踪新的时间轴起点
    
    # 首先对片段按原始时间排序，确保处理顺序正确
    temp_files_to_merge.sort(key=lambda x: x[0]['start'])
    
    logger.info("正在重新构建同步时间轴并处理音频片段...")
    
    for i, (seg, path) in enumerate(temp_files_to_merge):
        # 计算与前一片段的原始间隙 (Gap)
        orig_start = seg['start']
        prev_orig_end = temp_files_to_merge[i-1][0]['end'] if i > 0 else 0.0
        gap = max(0, orig_start - prev_orig_end)
        
        # 1. 记录静音间隙：不再生成临时文件，由 Pydub 在合成时统一处理
        if gap > 0.01:
            gap_seg = {'start': current_time, 'end': current_time + gap, 'is_gap': True}
            valid_temp_files.append((gap_seg, None))
            current_time += gap

        # 2. 处理 TTS 音频片段
        audio_duration = 0.0
        if validate_media_file(path):
            audio_duration = get_media_duration(path)
            
        if audio_duration is not None and audio_duration > 0.1:
            # 关键：不再使用 atempo 强制压缩时长，而是使用自然时长
            # 更新 segment 时间轴，这会直接影响后续生成的 SRT 字幕
            seg['start'] = current_time
            seg['end'] = current_time + audio_duration
            valid_temp_files.append((seg, path))
            current_time = seg['end']
            logger.debug(f"片段 {i} 时间轴已重对齐: {orig_start:.2f}s -> {seg['start']:.2f}s, 时长: {audio_duration:.2f}s")
        else:
            # 如果音频无效，也标记为间隙（静音占位），保证流程不中断
            logger.warning(f"片段 {i} 音频无效，将使用静音占位")
            skipped_count += 1
            orig_duration = max(0.1, seg['end'] - seg['start'])
            gap_seg = {'start': current_time, 'end': current_time + orig_duration, 'is_gap': True}
            valid_temp_files.append((gap_seg, None))
            current_time += orig_duration
    
    if skipped_count > 0:
        logger.warning(f"共跳过 {skipped_count} 个无效片段，已用静音占位符替代")

    full_output_path = os.path.join(output_dir, "full_tts.mp3")
    
    if not valid_temp_files:
        logger.error("没有成功的 TTS 片段可供合并，生成基础静音文件")
        total_duration = current_time if current_time > 0 else 10.0
        from modules.utils.media_utils import run_ffmpeg_command
        cmd = [ffmpeg_exe, "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={total_duration}", "-t", f"{total_duration}", full_output_path]
        run_ffmpeg_command(cmd, "生成静音音频", check=True)
    else:
        # 使用 Pydub 进行合并，确保音频流连续无卡顿
        success = merge_audio_with_pydub(valid_temp_files, full_output_path)
        if not success:
            logger.error("音频合成失败")
            raise Exception("音频合成失败")
        
        # 再次验证最终合并后的音频时长
        actual_duration = get_media_duration(full_output_path)
        if actual_duration is not None:
            logger.info(f"TTS 音频合成完成，总时长: {actual_duration:.2f}s")
            logger.success(f"音画同步优化：已基于 Pydub 实现平滑过渡")

    logger.success(f"TTS 合成流程结束: {full_output_path}")
    return full_output_path
