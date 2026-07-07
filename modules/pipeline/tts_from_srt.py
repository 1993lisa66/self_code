#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tts_from_srt 模式：从已有中文字幕直接生成中文配音并合成视频。
跳过 ASR 识别和翻译步骤，直接：SRT 解析 → TTS 文本预处理 → 合成配音 → 合并视频。
"""

import os
import asyncio
import shutil
from pathlib import Path
from loguru import logger

from .prompt_helper import get_project_root, load_prompt_template
from ..subtitle.srt_utils import find_srt_file, parse_srt_to_segments
from ..subtitle.generate_srt import generate_srt


def process_tts_from_srt(video_path, base_name, cache_dir, target_output_dir,
                          config, output_base_name=None):
    """
    tts_from_srt 模式的核心处理函数。

    Args:
        video_path: 视频文件路径
        base_name: 原始基础文件名（用于内部日志/缓存）
        cache_dir: 缓存目录
        target_output_dir: 输出目录
        config: 配置字典
        output_base_name: 翻译后的输出文件名（不含扩展名）

    Returns:
        dict: 处理结果
    """
    use_name = output_base_name if output_base_name else base_name
    logger.info("模式: tts_from_srt（从已有字幕生成配音）")

    try:
        # Step 1: 查找并解析 SRT 文件
        logger.info("\n[STEP 1/4] 查找中文字幕文件...")
        srt_path = find_srt_file(video_path, lang='zh')
        if not srt_path:
            return {
                'video_path': video_path, 'status': 'failed',
                'output_name': use_name,
                'message': '未找到匹配的 SRT 字幕文件（请确保字幕文件与视频同名，放在同一目录下）'
            }
        logger.info(f"  找到字幕: {srt_path}")

        segments = parse_srt_to_segments(srt_path)
        if not segments:
            return {
                'video_path': video_path, 'status': 'failed',
                'output_name': use_name, 'message': 'SRT 文件解析为空'
            }
        logger.success(f"字幕解析完成，共 {len(segments)} 条")

        # Step 2: TTS 文本预处理
        logger.info("\n[STEP 2/4] TTS 文本预处理...")
        prompts_dir = config['paths'].get('prompts_dir', os.path.join(get_project_root(), 'config', 'prompts'))
        prompt_template = load_prompt_template(prompts_dir, 'tts_prep.txt')
        if prompt_template:
            logger.info(f"  加载 TTS 预处理模板: tts_prep.txt")

        llm_config = dict(config.get('llm', {}))

        tts_texts = [seg['text'] for seg in segments]

        # 惰性导入 TTS 模块
        from ..tts.text_processor import process_tts_text_batch
        logger.info("  开始 TTS 文本预处理...")
        processed = process_tts_text_batch(
            tts_texts, config=llm_config, prompt_template=prompt_template
        )
        for idx, tts_text in enumerate(processed):
            segments[idx]["tts_text"] = tts_text
            segments[idx]["translated_text"] = tts_text  # 兼容 TTS 模块

        logger.success(f"TTS 文本预处理完成")

        # Step 3: 生成 TTS 配音
        logger.info("\n[STEP 3/4] 生成 TTS 中文配音...")
        tts_output_dir = os.path.join(cache_dir, "tts")
        from ..tts.tts_pipeline import generate_tts
        logger.info("  开始 TTS 语音合成...")
        tts_audio = asyncio.run(generate_tts(
            segments, output_dir=tts_output_dir, config=config['tts']
        ))
        logger.success(f"TTS 配音生成完成: {tts_audio}")

        # Step 4: 合并音频到视频
        logger.info("\n[STEP 4/4] 合成最终视频...")
        from ..merge.merge_video import merge_video
        merge_config = config.get('video', {}).copy()
        final_video = merge_video(
            video_path, tts_audio, output_dir=target_output_dir,
            config=merge_config, output_name=use_name
        )
        logger.success(f"最终视频生成完成: {final_video}")

        # 生成对齐后的 SRT
        logger.info("\n[额外步骤] 生成对齐后的 SRT 字幕...")
        srt_output_path = os.path.join(target_output_dir, f"{use_name}.srt")
        srt_path_out = generate_srt(segments, output_path=srt_output_path)
        logger.success(f"对齐版 SRT 生成完成: {srt_path_out}")

        logger.info(f"{'='*60}")
        logger.success(f"视频 [{use_name}] tts_from_srt 处理完成！")
        logger.info(f"{'='*60}\n")

        return {
            'video_path': video_path, 'status': 'success', 'output_name': use_name,
            'srt_path': srt_path_out, 'tts_audio': tts_audio, 'final_video': final_video,
            'message': '成功（从已有字幕生成配音）'
        }

    except Exception as e:
        logger.error(f"tts_from_srt 处理 {video_path} 时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'video_path': video_path, 'status': 'failed',
            'output_name': use_name, 'message': str(e)
        }
    # 缓存目录保留在输出目录中，供下次运行复用（不再自动清理）
