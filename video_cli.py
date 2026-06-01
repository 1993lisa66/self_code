#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一视频处理工具
支持字幕生成、翻译、配音等多种模式
"""

import os
import sys
import glob
import json
import time
import warnings
import shutil
from pathlib import Path
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 抑制 numpy/faster_whisper 的 matmul 溢出/除零警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module="faster_whisper")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# 必须在导入任何 AI 库之前设置环境变量和应用补丁
os.environ["NLTK_DATA"] = os.path.join(os.getcwd(), "nltk_data")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["ONELOGGER_DISABLED"] = "1"
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

# 设置本地模型目录(优先使用项目目录下的 models/)
project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "models")
if os.path.exists(models_dir):
    hf_home = os.path.join(models_dir, "huggingface")
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_home
    os.environ["MODELSCOPE_CACHE"] = os.path.join(models_dir, "funasr")
    os.environ["WHISPERX_CACHE"] = os.path.join(models_dir, "whisperx")
    print(f"[OK] 使用本地模型目录: {models_dir}")

# 配置 pydub 的 FFmpeg 路径（必须在导入 pydub 之前）
from modules.utils.ffmpeg_utils import get_ffmpeg_exe, get_ffprobe_exe
ffmpeg_path = get_ffmpeg_exe()
ffprobe_path = get_ffprobe_exe()

if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")

import nltk
from modules.utils.patch_torch import apply_torch_patch
apply_torch_patch()

try:
    # 设置本地 nltk_data 目录以避开权限和路径问题
    nltk_data_dir = os.environ["NLTK_DATA"]
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path = [nltk_data_dir]
except Exception as e:
    pass

# 导入功能模块
from modules.audio.extract_audio import extract_audio
from modules.vad.vad_pipeline import run_vad
from modules.asr.multi_asr import run_multi_asr
from modules.llm.fuse_asr import fuse_asr_result
from modules.translate.translate_pipeline import translate_segments
from modules.subtitle.generate_srt import generate_srt
from modules.tts.tts_pipeline import generate_tts
from modules.merge.merge_video import merge_video


def load_config():
    """加载配置文件"""
    import yaml
    config_path = os.path.join(project_root, "config", "config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # 默认配置
        return {
            'paths': {
                'input_dir': 'input',
                'output_dir': 'outputs',
                'cache_dir': 'cache',
                'log_dir': 'logs',
                'prompts_dir': 'prompts'
            },
            'audio': {'sample_rate': 16000},
            'vad': {},
            'asr': {'device': 'cpu', 'model_size': 'base'},
            'llm': {
                'api_key': '',
                'api_base': 'https://api.siliconflow.cn/v1',
                'model': 'deepseek-ai/DeepSeek-V3'
            },
            'translate': {'target_language': 'zh'},
            'tts': {'provider': 'edge', 'voice': 'zh-CN-XiaoxiaoNeural'},
            'video': {
                'burn_subtitles': False,
                'subtitle_position': 'bottom',
                'audio_mode': 'tts_only'
            },
            'global': {
                'max_concurrency': {'video_processor': 2},
                'auto_cleanup_cache': True
            }
        }


def check_output_exists(video_path, mode, output_dir):
    """
    检查是否已存在输出文件
    
    Args:
        video_path: 视频文件路径
        mode: 处理模式
        output_dir: 输出目录
    
    Returns:
        bool: True 如果输出已存在，False 否则
    """
    base_name = Path(video_path).stem
    
    if mode in ['subtitle_only', 'subtitle_bilingual', 'subtitle_chinese']:
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        return os.path.exists(srt_path)
    elif mode == 'tts_no_subtitle':
        synthetic_video = os.path.join(output_dir, f"{base_name}.mp4")
        return os.path.exists(synthetic_video)
    
    return False


def _check_has_audio(video_path):
    """
    快速检查视频文件是否有音频流
    
    Returns:
        (has_audio: bool, has_video: bool)
    """
    import subprocess
    from modules.utils.ffmpeg_utils import get_ffprobe_exe
    ffprobe_exe = get_ffprobe_exe()
    
    try:
        result = subprocess.run(
            [ffprobe_exe, "-v", "error",
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0",
             video_path],
            capture_output=True, text=True,
            env=os.environ.copy()
        )
        lines = result.stdout.strip().splitlines()
        has_audio = any("audio" in line for line in lines)
        has_video = any("video" in line for line in lines)
        return has_audio, has_video
    except Exception:
        return True, True  # 检查失败时假定有音频


def process_video_unified(video_path, config, mode='subtitle_only', input_dir='', output_dir='', batch_name=None):
    """
    统一视频处理函数
    
    Args:
        video_path: 视频文件路径
        config: 配置字典
        mode: 处理模式
            - subtitle_only: 仅生成中文字幕
            - subtitle_bilingual: 生成中英双语字幕
            - subtitle_chinese: 生成中文字幕（翻译后）
            - tts_no_subtitle: 生成中文配音（默认烧录字幕到视频）
        input_dir: 输入目录
        output_dir: 输出目录
        batch_name: 批次名称（可选，用于加载批次专属提示词）
    
    Returns:
        dict: 处理结果
    """
    video_path = os.path.abspath(video_path)
    base_name = Path(video_path).stem
    
    # 计算相对路径用于保持目录结构
    if input_dir and os.path.isdir(input_dir):
        rel_path = os.path.relpath(video_path, input_dir)
        rel_dir = os.path.dirname(rel_path)
        target_output_dir = os.path.join(output_dir, rel_dir)
    else:
        target_output_dir = output_dir
    
    # 确保输出目录存在
    os.makedirs(target_output_dir, exist_ok=True)
    
    # 检查输出是否已存在
    if check_output_exists(video_path, mode, target_output_dir):
        logger.info(f"⏭️  跳过（输出已存在）: {os.path.basename(video_path)}")
        return {
            'video_path': video_path,
            'status': 'skipped',
            'message': f'输出已存在（模式: {mode}）'
        }
    
    # 快速检查视频是否有音频流
    has_audio, has_video = _check_has_audio(video_path)
    if not has_audio:
        logger.warning(f"⏭️  跳过（无音频流）: {os.path.basename(video_path)}")
        return {
            'video_path': video_path,
            'status': 'skipped',
            'message': '视频文件没有音频流'
        }
    
    logger.info(f"{'='*60}")
    logger.info(f"开始处理视频: {os.path.basename(video_path)}")
    logger.info(f"处理模式: {mode}")
    logger.info(f"{'='*60}")
    
    # 创建临时缓存目录
    cache_dir = os.path.join(project_root, "cache", "video_processor", base_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # STEP 1: 提取音频
        logger.info("\n[STEP 1/6] 提取音频...")
        audio_output_dir = os.path.join(cache_dir, "audio")
        audio_path = extract_audio(
            video_path,
            output_dir=audio_output_dir,
            sample_rate=config['audio']['sample_rate']
        )
        logger.success(f"音频提取完成: {audio_path}")
        
        # STEP 2: VAD 语音切片
        logger.info("\n[STEP 2/6] 语音活动检测 (VAD)...")
        vad_output_dir = os.path.join(cache_dir, "vad")
        segments = run_vad(
            audio_path,
            output_dir=vad_output_dir,
            device=config['asr']['device']
        )
        logger.success(f"VAD 完成，检测到 {len(segments)} 个语音片段")
        
        if not segments:
            logger.warning("未检测到语音片段，跳过后续处理")
            return {
                'video_path': video_path,
                'status': 'failed',
                'message': '未检测到语音片段'
            }
        
        # STEP 3: ASR 识别
        logger.info("\n[STEP 3/6] 自动语音识别 (ASR)...")
        asr_output_dir = os.path.join(cache_dir, "asr")
        asr_results = run_multi_asr(
            audio_path,
            segments,
            output_dir=asr_output_dir,
            device=config['asr']['device'],
            model_size=config['asr'].get('model_size', 'base')
        )
        
        # 提取 WhisperX 结果（包含时间轴）
        whisperx_segments = asr_results.get('whisperx', [])
        if not whisperx_segments:
            logger.error("ASR 识别结果为空")
            return {
                'video_path': video_path,
                'status': 'failed',
                'message': 'ASR 识别结果为空'
            }
        
        logger.success(f"ASR 识别完成，共 {len(whisperx_segments)} 个片段")
        
        # STEP 4: LLM 修正
        fused_results = whisperx_segments
        if config.get('llm', {}).get('api_key'):
            logger.info("\n[STEP 4/6] LLM 修正字幕文本...")
            try:
                # 确定提示词目录（优先使用批次提示词）
                if batch_name:
                    batches_dir = config.get('paths', {}).get('batches_dir', os.path.join(project_root, "config", "batches"))
                    batch_dir = os.path.join(batches_dir, batch_name)
                    prompts_dir = os.path.join(batch_dir, "prompts")
                    logger.info(f"  使用批次提示词: {batch_name}")
                else:
                    batch_dir = None
                    prompts_dir = config['paths'].get('prompts_dir', 'config/prompts')
                    logger.info(f"  使用全局提示词")
                
                asr_fix_path = os.path.join(prompts_dir, 'asr_fix.txt')
                prompt_template = ""
                if os.path.exists(asr_fix_path):
                    with open(asr_fix_path, 'r', encoding='utf-8') as f:
                        prompt_template = f.read()
                    logger.info(f"  加载提示词模板: {os.path.basename(asr_fix_path)}")
                
                # 注入批次术语表路径
                llm_config = dict(config.get('llm', {}))
                if batch_dir:
                    term_path = os.path.join(batch_dir, "terminology.json")
                    if os.path.exists(term_path):
                        llm_config['terminology_file'] = term_path
                        logger.info(f"  注入批次术语表: {term_path}")
                
                fused_results = fuse_asr_result(
                    asr_results,
                    config=llm_config,
                    prompt_template=prompt_template
                )
                logger.success(f"LLM 修正完成")
                
                # LLM 驱动的提示词进化（asr_fix.txt）
                if batch_dir and prompt_template and llm_config.get('api_key') and os.path.exists(asr_fix_path):
                    try:
                        from modules.translate.translate_pipeline import evolve_prompt
                        asr_samples = []
                        step = max(1, min(len(whisperx_segments), len(fused_results)) // 25)
                        for k in range(0, min(len(whisperx_segments), len(fused_results)), step):
                            orig = whisperx_segments[k].get('text', '')
                            corr = fused_results[k].get('text', '') if k < len(fused_results) else ''
                            if orig and corr:
                                asr_samples.append({'input': orig, 'output': corr})
                        if asr_samples:
                            evolve_prompt(asr_fix_path, asr_samples, "ASR文本修正", config=llm_config)
                    except Exception as e:
                        logger.warning(f"ASR提示词进化跳过（不影响主流程）: {e}")
            except Exception as e:
                logger.warning(f"LLM 修正失败，使用原始 ASR 结果: {e}")
                fused_results = whisperx_segments
        else:
            logger.info("\n[STEP 4/6] 跳过 LLM 修正（未配置 API Key）")
            llm_config = config.get('llm', {})
        
        # STEP 5: 翻译（所有需要中文的模式统一在此翻译）
        final_segments = fused_results
        srt_path = None
        tts_audio = None
        final_video = None
        translated_results = None
        
        if mode in ['subtitle_bilingual', 'subtitle_chinese', 'tts_no_subtitle']:
            logger.info("\n[STEP 5/6] 翻译字幕（→ 中文）...")
            
            # 确定提示词目录（优先使用批次提示词）
            if batch_name:
                batches_dir = config.get('paths', {}).get('batches_dir', os.path.join(project_root, "config", "batches"))
                batch_dir = os.path.join(batches_dir, batch_name)
                prompts_dir = os.path.join(batch_dir, "prompts")
            else:
                batch_dir = None
                prompts_dir = config['paths'].get('prompts_dir', 'config/prompts')
            
            translation_path = os.path.join(prompts_dir, 'translation.txt')
            prompt_template = ""
            if os.path.exists(translation_path):
                with open(translation_path, 'r', encoding='utf-8') as f:
                    prompt_template = f.read()
            
            # 注入批次术语表路径
            llm_config = dict(config.get('llm', {}))
            if batch_dir:
                term_path = os.path.join(batch_dir, "terminology.json")
                if os.path.exists(term_path):
                    llm_config['terminology_file'] = term_path
                    logger.info(f"  注入批次术语表: {term_path}")
            
            translated_results = translate_segments(
                fused_results,
                target_lang='zh',
                config=llm_config,
                prompt_template=prompt_template
            )
            
            # LLM 驱动的提示词进化（translation.txt）
            if batch_dir and prompt_template and llm_config.get('api_key') and os.path.exists(translation_path):
                try:
                    from modules.translate.translate_pipeline import evolve_prompt
                    trans_samples = []
                    sample_count = min(len(fused_results), len(translated_results))
                    step = max(1, sample_count // 25)
                    for k in range(0, sample_count, step):
                        orig = fused_results[k].get('text', '')
                        trans = translated_results[k].get('translated_text', '')
                        if orig and trans:
                            trans_samples.append({'input': orig, 'output': trans})
                    if trans_samples:
                        evolve_prompt(translation_path, trans_samples, "字幕翻译", config=llm_config)
                except Exception as e:
                    logger.warning(f"翻译提示词进化跳过（不影响主流程）: {e}")
            
            # 如果是双语模式，保留原文和译文
            if mode == 'subtitle_bilingual':
                for seg in translated_results:
                    original_text = seg.get('text', '')
                    translated_text = seg.get('translated_text', '')
                    seg['content'] = f"{original_text}\n{translated_text}"
                final_segments = translated_results
            else:
                final_segments = translated_results
            
            logger.success(f"翻译完成")
        
        # STEP 6: 生成 SRT 字幕
        if mode in ['subtitle_only', 'subtitle_bilingual', 'subtitle_chinese', 'tts_no_subtitle']:
            logger.info("\n[STEP 6/6] 生成 SRT 字幕...")
            srt_output_path = os.path.join(target_output_dir, f"{base_name}.srt")
            
            # 对于双语模式，使用特殊格式
            if mode == 'subtitle_bilingual':
                from datetime import timedelta
                import srt
                
                subtitles = []
                for i, seg in enumerate(final_segments):
                    sub = srt.Subtitle(
                        index=i + 1,
                        start=timedelta(seconds=seg["start"]),
                        end=timedelta(seconds=seg["end"]),
                        content=seg.get("content", seg.get("translated_text", seg["text"]))
                    )
                    subtitles.append(sub)
                
                srt_content = srt.compose(subtitles)
                with open(srt_output_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                srt_path = srt_output_path
            else:
                srt_path = generate_srt(
                    final_segments,
                    output_path=srt_output_path
                )
            
            logger.success(f"SRT 字幕生成完成: {srt_path}")
        
        # 如果需要 TTS 配音
        if mode == 'tts_no_subtitle':
            if translated_results is None:
                logger.error("TTS 模式需要翻译结果，但未找到")
                raise RuntimeError("翻译步骤缺失")
            
            logger.info("\n[额外步骤] TTS 文本预处理...")
            from modules.tts.text_processor import process_tts_text_batch
            
            # 确定提示词目录（优先使用批次提示词）
            if batch_name:
                batches_dir = config.get('paths', {}).get('batches_dir', os.path.join(project_root, "config", "batches"))
                prompts_dir = os.path.join(batches_dir, batch_name, "prompts")
            else:
                prompts_dir = config['paths'].get('prompts_dir', 'config/prompts')
            
            tts_prep_path = os.path.join(prompts_dir, 'tts_prep.txt')
            prompt_template = ""
            if os.path.exists(tts_prep_path):
                with open(tts_prep_path, 'r', encoding='utf-8') as f:
                    prompt_template = f.read()
            
            # 批量处理 TTS 文本（每批 20 条，内置缓存）
            total_segments = len(translated_results)
            tts_texts = [seg.get("translated_text", seg["text"]) for seg in translated_results]
            processed = process_tts_text_batch(
                tts_texts,
                config=llm_config,
                prompt_template=prompt_template,
                batch_size=20
            )
            for idx, tts_text in enumerate(processed):
                translated_results[idx]["tts_text"] = tts_text
            logger.success(f"TTS 文本预处理完成")
            
            # 生成 TTS 音频
            logger.info("生成 TTS 配音...")
            tts_output_dir = os.path.join(cache_dir, "tts")
            import asyncio
            tts_audio = asyncio.run(generate_tts(
                translated_results,
                output_dir=tts_output_dir,
                voice=config['tts']['voice'],
                config=config['tts']
            ))
            logger.success(f"TTS 配音生成完成: {tts_audio}")
            
            # 合并音频到视频（不烧录字幕）
            logger.info("\n[额外步骤] 合成最终视频...")
            
            merge_config = config.get('video', {}).copy()
            
            final_video = merge_video(
                video_path,
                tts_audio,
                output_dir=target_output_dir,
                config=merge_config
            )
            logger.success(f"最终视频生成完成: {final_video}")
        
        # 生成 chapters.txt / summary.txt / result.json
        # logger.info("\n[最终步骤] 生成处理报告...")
        # _generate_reports(
        #     video_path=video_path,
        #     target_output_dir=target_output_dir,
        #     base_name=base_name,
        #     srt_path=srt_path,
        #     mode=mode,
        #     config=config,
        #     batch_name=batch_name,
        #     segments=segments,
        #     asr_results=asr_results,
        #     fused_results=fused_results,
        #     translated_results=translated_results,
        #     final_segments=final_segments,
        # )
        
        logger.info(f"{'='*60}")
        logger.success(f"视频 [{base_name}] 处理完成！")
        logger.info(f"{'='*60}\n")
        
        return {
            'video_path': video_path,
            'status': 'success',
            'srt_path': srt_path,
            'tts_audio': tts_audio,
            'final_video': final_video,
            'message': '成功'
        }
        
    except Exception as e:
        logger.error(f"处理视频 {video_path} 时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'video_path': video_path,
            'status': 'failed',
            'message': str(e)
        }
        
    finally:
        # 清理缓存
        try:
            import shutil
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                logger.debug(f"已清理缓存目录: {cache_dir}")
        except Exception as e:
            logger.warning(f"清理缓存失败: {e}")


def _generate_reports(video_path, target_output_dir, base_name, srt_path, mode,
                     config, batch_name, segments, asr_results, fused_results,
                     translated_results, final_segments):
    """生成 chapters.txt / summary.txt / result.json 三个报告文件"""
    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # === chapters.txt ===
    chapters_content = ""
    if srt_path and os.path.exists(srt_path) and config.get('llm', {}).get('api_key'):
        try:
            from modules.utils.chapter_generator import generate_chapters
            
            # 确定提示词目录
            if batch_name:
                batches_dir = config.get('paths', {}).get('batches_dir', os.path.join(project_root_dir, "config", "batches"))
                prompts_dir = os.path.join(batches_dir, batch_name, "prompts")
            else:
                prompts_dir = config['paths'].get('prompts_dir', os.path.join(project_root_dir, 'config', 'prompts'))
            
            chapters_prompt_path = os.path.join(prompts_dir, 'chapters.txt')
            prompt_template = ""
            if os.path.exists(chapters_prompt_path):
                with open(chapters_prompt_path, 'r', encoding='utf-8') as f:
                    prompt_template = f.read()
            
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            chapters_content = generate_chapters(
                srt_content,
                config=config.get('llm', {}),
                prompt_template=prompt_template
            )
        except Exception as e:
            logger.warning(f"章节生成失败: {e}")
    
    chapters_path = os.path.join(target_output_dir, f"{base_name}_chapters.txt")
    with open(chapters_path, 'w', encoding='utf-8') as f:
        f.write(chapters_content)
    logger.info(f"  ✓ chapters.txt: {chapters_path}")
    
    # === summary.txt ===
    # 统计各项数据
    vad_count = len(segments) if segments else 0
    total_speech_duration = sum(s.get('end', 0) - s.get('start', 0) for s in segments) if segments else 0
    
    whisperx_count = len(asr_results.get('whisperx', [])) if asr_results else 0
    glm_count = len(asr_results.get('glm', [])) if asr_results else 0
    fused_count = len(fused_results) if fused_results else 0
    
    translated_count = len(translated_results) if translated_results else 0
    
    final_subtitle_count = len(final_segments) if final_segments else 0
    tts_total_duration = 0
    if final_segments and len(final_segments) > 0:
        tts_total_duration = final_segments[-1].get('end', 0)
    
    chapters_line_count = len([l for l in chapters_content.splitlines() if l.strip()]) if chapters_content else 0
    
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tts_model = config['tts'].get('provider', 'edge') if config.get('tts') else 'edge'
    
    summary_lines = [
        f"=== Video Processor 处理摘要 ===",
        "",
        f"源视频文件: {video_path}",
        f"输出目录: {target_output_dir}",
        f"处理时间: {now}",
        f"TTS模型: {tts_model}",
        "",
        "--- VAD (语音活动检测) ---",
        f"VAD片段数: {vad_count}",
        f"音频段落数: {vad_count}",
        f"总语音时长: {total_speech_duration:.2f} 秒",
        "",
        "--- ASR (自动语音识别) ---",
        f"WhisperX识别结果数: {whisperx_count}",
        f"GLM识别结果数: {glm_count}",
        f"最终ASR结果数: {fused_count}",
        "",
        "--- 段落处理 ---",
        f"合并段落数: {fused_count}",
        f"句子段落索引数: {fused_count}",
        "",
        "--- 翻译和润色 ---",
        f"翻译字幕数: {translated_count}",
        "",
        "--- TTS (文本转语音) ---",
        f"TTS输入数: {translated_count}",
        f"TTS音频文件数: {translated_count}",
        f"TTS最终时间轴数: {final_subtitle_count}",
        f"TTS总时长: {tts_total_duration:.2f} 秒",
        "",
        "--- 最终输出 ---",
        f"最终字幕数: {final_subtitle_count}",
        f"章节数: {chapters_line_count}",
        f"章节文件: {chapters_path}",
        ""
    ]
    
    summary_path = os.path.join(target_output_dir, f"{base_name}_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    logger.info(f"  ✓ summary.txt: {summary_path}")
    
    # === result.json ===
    # 构造 segments 为 vad_list 格式
    vad_list = segments if segments else []
    whisperx_segments = asr_results.get('whisperx', []) if asr_results else []
    glm_segments = asr_results.get('glm', []) if asr_results else []
    
    final_subtitles_list = []
    if final_segments:
        for seg in final_segments:
            text = seg.get('translated_text', seg.get('text', ''))
            final_subtitles_list.append({
                "text": text,
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "word_count": len(text)
            })
    
    result_data = {
        "tts_model_type": tts_model,
        "video_path": video_path,
        "output_dir": target_output_dir,
        "config": config,
        "vad_list": vad_list,
        "whisperx_asr_result": whisperx_segments,
        "glm_asr_result": glm_segments,
        "final_asr_result": fused_results if fused_results else [],
        "merged_asr_paragraphs": fused_results if fused_results else [],
        "sentence_paragraph_indices": [],
        "translated_subtitles": translated_results if translated_results else [],
        "final_subtitles": final_subtitles_list,
        "chapter_file_path": chapters_path if chapters_content else "",
        "metadata": {
            "video_path": video_path,
            "output_dir": target_output_dir,
            "tts_model_type": tts_model,
            "processed_time": now,
            "polish_status": "completed",
            "agent_state_version": "1.0"
        }
    }
    
    result_path = os.path.join(target_output_dir, f"{base_name}_result.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ result.json: {result_path}")


def process_single_video(args):
    """
    包装函数，用于多进程调用
    
    Args:
        args: (video_path, config, mode, input_dir, output_dir, batch_name) 元组
    
    Returns:
        dict: 处理结果
    """
    video_path, config, mode, input_dir, output_dir, batch_name = args
    return process_video_unified(video_path, config, mode, input_dir, output_dir, batch_name)


def main(input_path=None, output_dir=None, mode=None, batch_name=None):
    """
    主函数
    
    Args:
        input_path: 输入路径（可选，覆盖默认配置）
        output_dir: 输出目录（可选，覆盖默认配置）
        mode: 处理模式（可选，覆盖默认配置）
        batch_name: 批次名称（可选，用于加载批次专属配置）
    """
    # 清理上一次运行的缓存
    cache_video_dir = os.path.join(project_root, "cache", "video_processor")
    if os.path.exists(cache_video_dir):
        shutil.rmtree(cache_video_dir)
        logger.info(f"已清理上次缓存: {cache_video_dir}")
    
    # 配置日志（每天零点自动切割，保留 7 天）
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger.add(
        os.path.join(logs_dir, "video_processor_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="7 days",
        level="DEBUG"
    )
    
    print("\n" + "="*60)
    print("🎬 统一视频处理工具")
    print("="*60 + "\n")
    
    # ==================== 默认配置区域 ====================
    # 在这里定义默认的输入输出路径和处理模式
    
    DEFAULT_INPUT_PATH =  os.path.join(project_root, "input")
    DEFAULT_OUTPUT_DIR = os.path.join(project_root, "outputs")
    
    # 处理模式选择：
    # - subtitle_only: 仅生成中文字幕（ASR 识别）
    # - subtitle_bilingual: 生成中英双语字幕
    # - subtitle_chinese: 生成中文字幕（翻译后）
    # - tts_no_subtitle: 生成中文配音（默认烧录字幕到视频）
    DEFAULT_MODE = "subtitle_only"
    
    # ====================================================
    
    # 解析命令行参数（可选覆盖配置）
    import argparse
    parser = argparse.ArgumentParser(description='统一视频处理工具')
    parser.add_argument('input_path', nargs='?', default=None, help='输入路径（视频文件或目录）')
    parser.add_argument('--mode', '-m', choices=[
        'subtitle_only',
        'subtitle_bilingual',
        'subtitle_chinese',
        'tts_no_subtitle'
    ], default=None, help='处理模式')
    parser.add_argument('--output', '-o', default=None, help='输出目录')
    
    # 使用参数或默认配置
    final_input_path = input_path if input_path else DEFAULT_INPUT_PATH
    final_output_dir = output_dir if output_dir else DEFAULT_OUTPUT_DIR
    final_mode = mode if mode else DEFAULT_MODE
    final_batch_name = batch_name  # batch_name 为 None 时不使用批次
    
    # 显示配置信息
    print(f"📂 输入路径: {final_input_path}")
    print(f"📤 输出目录: {final_output_dir}")
    print(f"🔧 处理模式: {final_mode}")
    if final_batch_name:
        print(f"📦 批次名称: {final_batch_name}")
    
    mode_descriptions = {
        'subtitle_only': '仅生成中文字幕（ASR 识别）',
        'subtitle_bilingual': '生成中英双语字幕',
        'subtitle_chinese': '生成中文字幕（翻译后）',
        'tts_no_subtitle': '生成中文配音（默认烧录字幕到视频）'
    }
    print(f"💡 模式说明: {mode_descriptions[final_mode]}\n")
    
    # 验证输入路径
    if not os.path.exists(final_input_path):
        logger.error(f"错误: 找不到路径 {final_input_path}")
        print(f"\n❌ 错误: 找不到路径 {final_input_path}")
        return
    
    # 收集视频文件
    VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.wmv')
    video_files = []
    
    if os.path.isfile(final_input_path):
        if final_input_path.lower().endswith(VIDEO_EXTENSIONS):
            video_files.append(final_input_path)
        else:
            logger.error(f"不支持的文件格式: {final_input_path}")
            return
    elif os.path.isdir(final_input_path):
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(glob.glob(os.path.join(final_input_path, f"**/*{ext}"), recursive=True))
    
    if not video_files:
        logger.warning(f"在 {final_input_path} 中未找到视频文件")
        print(f"在 {final_input_path} 中未找到可处理的视频文件。")
        return
    
    print(f"共找到 {len(video_files)} 个视频文件待处理。\n")
    
    # 加载配置（需要在批次检查之前加载）
    config = load_config()
    
    # 初始化提示词和术语管理器
    from modules.utils.prompt_term_manager import PromptTermManager
    prompt_manager = PromptTermManager()
    
    # 如果指定了批次，确保批次目录和文件存在
    if final_batch_name:
        logger.info(f"\n检查批次配置: {final_batch_name}")
        
        # 从配置读取批次目录路径
        batches_dir = config.get('paths', {}).get('batches_dir', os.path.join(project_root, "config", "batches"))
        batch_dir = os.path.join(batches_dir, final_batch_name)
        
        # 如果批次目录不存在，创建它
        if not os.path.exists(batch_dir):
            logger.info(f"批次目录不存在，正在创建: {batch_dir}")
            
            # 获取视频主题（可以从输入路径推断或使用默认值）
            video_topics = [os.path.basename(final_input_path)] if os.path.isdir(final_input_path) else ["general"]
            
            # 创建批次配置
            batch_info = prompt_manager.create_batch_prompts(
                batch_name=final_batch_name,
                video_topics=video_topics
            )
            logger.success(f"批次配置已创建: {batch_info['batch_dir']}")
        else:
            logger.info(f"批次目录已存在: {batch_dir}")
        
        # 验证批次文件
        batch_prompts_dir = os.path.join(batch_dir, "prompts")
        batch_terminology_file = os.path.join(batch_dir, "terminology.json")
        
        if not os.path.exists(batch_prompts_dir):
            logger.warning(f"批次提示词目录不存在，正在创建: {batch_prompts_dir}")
            os.makedirs(batch_prompts_dir, exist_ok=True)
            # 复制基础提示词
            for prompt_name in prompt_manager.list_prompts():
                content = prompt_manager.load_prompt(prompt_name)
                prompt_manager._save_to_file(os.path.join(batch_prompts_dir, f"{prompt_name}.txt"), content)
        
        if not os.path.exists(batch_terminology_file):
            logger.warning(f"批次术语表不存在，正在创建: {batch_terminology_file}")
            # 复制全局术语表
            global_terms = prompt_manager.load_terminology()
            with open(batch_terminology_file, 'w', encoding='utf-8') as f:
                json.dump(global_terms, f, ensure_ascii=False, indent=2)
        
        logger.success(f"批次配置验证完成: {final_batch_name}\n")
    
    # 获取多进程配置
    max_workers = config.get('global', {}).get('max_concurrency', {}).get('video_processor', 2)
    
    # 显示配置信息
    logger.info(f"配置信息:")
    logger.info(f"  - 处理模式: {final_mode}")
    logger.info(f"  - ASR 设备: {config['asr'].get('device', 'cpu')}")
    logger.info(f"  - 模型大小: {config['asr'].get('model_size', 'base')}")
    logger.info(f"  - 采样率: {config['audio']['sample_rate']}")
    logger.info(f"  - 并行进程数: {max_workers}")
    logger.info(f"\n")
    
    # 准备参数列表（包含 batch_name）
    task_args = [(video_file, config, final_mode, final_input_path, final_output_dir, final_batch_name) for video_file in video_files]
    
    # 多进程处理（所有模式都使用多进程）
    print(f"🚀 启动多进程处理模式（{max_workers} 个进程）\n")
    
    start_time = time.time()
    results = []
    
    # macOS 需要设置启动方法为 spawn
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('spawn', force=True)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_video = {executor.submit(process_single_video, args): args[0] 
                          for args in task_args}
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_video):
            video_file = future_to_video[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # 显示进度
                progress_pct = (completed / len(video_files) * 100)
                status_icon = "✅" if result['status'] == 'success' else "⏭️" if result['status'] == 'skipped' else "❌"
                logger.info(f"\n进度: [{completed}/{len(video_files)}] ({progress_pct:.1f}%) {status_icon} {os.path.basename(video_file)}")
                
            except Exception as e:
                logger.error(f"处理视频 {os.path.basename(video_file)} 时发生异常: {e}")
                results.append({
                    'video_path': video_file,
                    'status': 'failed',
                    'message': str(e)
                })
                completed += 1
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\n{'='*60}")
    print(f"📊 处理完成统计:")
    print(f"   - 总计: {len(video_files)} 个视频")
    print(f"   - 成功: {success_count} 个")
    print(f"   - 跳过: {skipped_count} 个（输出已存在）")
    print(f"   - 失败: {failed_count} 个")
    print(f"   - 总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print(f"{'='*60}")
    
    # 显示失败的文件
    if failed_count > 0:
        print(f"\n❌ 失败的文件:")
        for r in results:
            if r['status'] == 'failed':
                print(f"   - {os.path.basename(r['video_path'])}: {r.get('message', '未知错误')}")


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 在这里定义输入输出路径和处理模式
    
    # 输入路径（可以是单个视频文件或包含视频的目录）
    INPUT_PATH = "/Volumes/mvp/交易场/Lathyrus Trading原始/The Lathyrus Files/Backtesting Sessions"
    
    # 输出目录
    OUTPUT_DIR = "/Volumes/mvp/交易场/Lathyrus Trading原始/The Lathyrus Files/Backtesting Sessions-中文"
    
    # 处理模式选择：
    # - subtitle_only: 仅生成中文字幕（ASR 识别）
    # - subtitle_bilingual: 生成中英双语字幕
    # - subtitle_chinese: 生成中文字幕（翻译后）
    # - tts_no_subtitle: 生成中文配音（默认烧录字幕到视频）
    PROCESS_MODE = "tts_no_subtitle"
    
    # 批次名称（为空则使用全局配置）
    BATCH_NAME = "Lathyrus_Trading"  # 例如: "ICT_Trading_Batch1"
    
    # ================================================
    
    # 解析命令行参数（可选覆盖配置）
    import argparse
    parser = argparse.ArgumentParser(description='统一视频处理工具')
    parser.add_argument('input_path', nargs='?', default=None, help='输入路径（视频文件或目录）')
    parser.add_argument('--mode', '-m', choices=[
        'subtitle_only',
        'subtitle_bilingual',
        'subtitle_chinese',
        'tts_no_subtitle'
    ], default=None, help='处理模式')
    parser.add_argument('--output', '-o', default=None, help='输出目录')
    parser.add_argument('--batch', '-b', default=None, help='批次名称')
    
    args = parser.parse_args()
    
    # 调用 main 函数，传入配置
    main(
        input_path=args.input_path if args.input_path else INPUT_PATH,
        output_dir=args.output if args.output else OUTPUT_DIR,
        mode=args.mode if args.mode else PROCESS_MODE,
        batch_name=args.batch if args.batch else BATCH_NAME
    )
