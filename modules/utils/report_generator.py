#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理报告生成器：生成 chapters.txt / summary.txt / result.json。
"""

import os
import json
import datetime
from loguru import logger


def generate_reports(video_path, target_output_dir, base_name, srt_path, mode,
                     config, segments, asr_results, fused_results,
                     translated_results, final_segments):
    """生成 chapters.txt / summary.txt / result.json 三个报告文件"""
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # === chapters.txt ===
    chapters_content = _generate_chapters(
        srt_path, config, project_root_dir
    )
    chapters_path = os.path.join(target_output_dir, f"{base_name}_chapters.txt")
    with open(chapters_path, 'w', encoding='utf-8') as f:
        f.write(chapters_content)
    logger.info(f"  ✓ chapters.txt: {chapters_path}")

    # === summary.txt ===
    _generate_summary(
        target_output_dir, base_name, video_path, config,
        segments, asr_results, fused_results, translated_results,
        final_segments, chapters_content, chapters_path
    )

    # === result.json ===
    _generate_result_json(
        target_output_dir, base_name, video_path, config,
        segments, asr_results, fused_results, translated_results,
        final_segments, chapters_content, chapters_path
    )


def _generate_chapters(srt_path, config, project_root_dir):
    """生成章节内容"""
    if not srt_path or not os.path.exists(srt_path) or not config.get('llm', {}).get('api_key'):
        return ""

    try:
        from ..utils.chapter_generator import generate_chapters
        prompts_dir = config['paths'].get('prompts_dir', os.path.join(project_root_dir, 'config', 'prompts'))
        chapters_prompt_path = os.path.join(prompts_dir, 'chapters.txt')
        prompt_template = ""
        if os.path.exists(chapters_prompt_path):
            with open(chapters_prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()

        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        return generate_chapters(srt_content, config=config.get('llm', {}),
                                 prompt_template=prompt_template)
    except Exception as e:
        logger.warning(f"章节生成失败: {e}")
        return ""


def _generate_summary(target_output_dir, base_name, video_path, config,
                      segments, asr_results, fused_results, translated_results,
                      final_segments, chapters_content, chapters_path):
    """生成 summary.txt"""
    vad_count = len(segments) if segments else 0
    total_speech_duration = sum(
        s.get('end', 0) - s.get('start', 0) for s in segments
    ) if segments else 0

    whisperx_count = len(asr_results.get('whisperx', [])) if asr_results else 0
    glm_count = len(asr_results.get('glm', [])) if asr_results else 0
    fused_count = len(fused_results) if fused_results else 0
    translated_count = len(translated_results) if translated_results else 0

    final_subtitle_count = len(final_segments) if final_segments else 0
    tts_total_duration = final_segments[-1].get('end', 0) if final_segments and len(final_segments) > 0 else 0

    chapters_line_count = len([l for l in chapters_content.splitlines() if l.strip()]) if chapters_content else 0

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tts_model = config['tts'].get('provider', 'edge') if config.get('tts') else 'edge'

    summary_lines = [
        "=== Video Processor 处理摘要 ===",
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


def _generate_result_json(target_output_dir, base_name, video_path, config,
                          segments, asr_results, fused_results, translated_results,
                          final_segments, chapters_content, chapters_path):
    """生成 result.json"""
    tts_model = config['tts'].get('provider', 'edge') if config.get('tts') else 'edge'

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

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
