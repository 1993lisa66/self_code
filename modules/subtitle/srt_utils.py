#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT 字幕工具集：查找、解析、构建、生成字幕文件。
"""

import os
import shutil
import unicodedata
from datetime import timedelta
from pathlib import Path
from loguru import logger


def find_srt_file(video_path, lang='original'):
    """
    查找视频文件对应的 SRT 字幕文件。

    lang='original'（默认）：优先查找英文/原文字幕
        .en.srt（英文原文字幕） > .srt（无标记） > .zh.srt 等中文变体

    lang='zh'：优先查找中文字幕
        .zh.srt > .chs.srt > .chi.srt ... > .srt > .en.srt
    """
    base_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    if lang == 'zh':
        candidates = [
            os.path.join(base_dir, f"{base_name}.zh.srt"),
            os.path.join(base_dir, f"{base_name}.chs.srt"),
            os.path.join(base_dir, f"{base_name}.chi.srt"),
            os.path.join(base_dir, f"{base_name}.zh-CN.srt"),
            os.path.join(base_dir, f"{base_name}.zh-Hans.srt"),
            os.path.join(base_dir, f"{base_name}.cn.srt"),
            os.path.join(base_dir, f"{base_name}.srt"),
            os.path.join(base_dir, f"{base_name}.en.srt"),
        ]
    else:
        candidates = [
            os.path.join(base_dir, f"{base_name}.en.srt"),
            os.path.join(base_dir, f"{base_name}.srt"),
            os.path.join(base_dir, f"{base_name}.chs.srt"),
            os.path.join(base_dir, f"{base_name}.chi.srt"),
            os.path.join(base_dir, f"{base_name}.zh.srt"),
            os.path.join(base_dir, f"{base_name}.zh-CN.srt"),
            os.path.join(base_dir, f"{base_name}.zh-Hans.srt"),
            os.path.join(base_dir, f"{base_name}.cn.srt"),
        ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def parse_srt_to_segments(srt_path):
    """解析 SRT 字幕文件，返回包含 start/end/text 的 segment 列表。"""
    import srt
    segments = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    for sub in srt.parse(content):
        segments.append({
            'start': sub.start.total_seconds(),
            'end': sub.end.total_seconds(),
            'text': sub.content.strip(),
        })
    logger.info(f"从 SRT 解析到 {len(segments)} 条字幕: {srt_path}")
    return segments


def build_subtitles(segments, lang='zh'):
    """
    构建 SRT 字幕列表（不修改原始 segment 数据）。

    lang='zh'       → 使用 translated_text（中文）
    lang='original' → 使用 text（原始语言）
    lang='bilingual' → translated_text\ntext（上中文下原语）
    """
    import srt
    subtitles = []
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        if i > 0:
            prev_seg = segments[i - 1]
            if start < prev_seg['end']:
                start = prev_seg['end']

        if lang == 'zh':
            content = seg.get("translated_text", seg.get("text", ""))
        elif lang == 'original':
            content = seg.get("text", "")
        elif lang == 'bilingual':
            zh = seg.get("translated_text", "")
            en = seg.get("text", "")
            content = f"{zh}\n{en}"
        else:
            content = seg.get("translated_text", seg.get("text", ""))

        subtitles.append(srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=start),
            end=timedelta(seconds=end),
            content=content
        ))
    return subtitles


def generate_all_srt_files(fused_results, translated_results, output_dir, base_name,
                           source_is_chinese=False):
    """
    生成 SRT 字幕文件。

    正常流程（英文源 → 翻译成中文）：
      - {base_name}.zh.srt → 纯中文字幕（translated_text）
      - {base_name}.en.srt → 原始英文/原语字幕（text）
      - {base_name}.srt    → 双语字幕（上中文下原语）

    源语言已是中文时（source_is_chinese=True）：不生成 .en.srt（无外语原文），
    .zh.srt 和 .srt 均为中文原文。

    Returns:
        str: 主 SRT 文件路径
    """
    import srt

    has_translation = translated_results is not None and len(translated_results) > 0

    # 1. 中文字幕（.zh.srt）—— 始终从 fused_results 取中文原文
    #    注意：不能从 translated_results 取，因为 source_is_chinese 时
    #    translated_text 被复制为 text 的值，与 .en.srt 完全一致。
    zh_path = os.path.join(output_dir, f"{base_name}.zh.srt")
    if source_is_chinese:
        # 源语言已是中文：直接用 fused_results（无翻译）
        zh_subs = build_subtitles(fused_results, lang='zh')
    elif has_translation:
        zh_subs = build_subtitles(translated_results, lang='zh')
    else:
        zh_subs = build_subtitles(fused_results, lang='zh')
    with open(zh_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(zh_subs))
    logger.info(f"  中文字幕: {zh_path}")

    # 2. 原始语言字幕（.en.srt）—— 始终从 fused_results 取原始 text
    #    源语言是中文时不生成（无外语原文可提取）
    en_path = os.path.join(output_dir, f"{base_name}.en.srt")
    if not source_is_chinese:
        en_subs = build_subtitles(fused_results, lang='original')
        with open(en_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(en_subs))
        logger.info(f"  原始字幕: {en_path}")
    else:
        logger.info(f"  跳过原始字幕（源语言已是中文，无外语原文）")

    # 3. 主字幕（.srt）
    main_path = os.path.join(output_dir, f"{base_name}.srt")
    if has_translation and not source_is_chinese:
        # 有翻译且非中文源 → 双语字幕
        bilingual_subs = build_subtitles(translated_results, lang='bilingual')
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(bilingual_subs))
        logger.info(f"  双语字幕: {main_path}")
    elif source_is_chinese:
        # 源语言中文 → 直接用原文
        main_subs = build_subtitles(fused_results, lang='zh')
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(main_subs))
        logger.info(f"  字幕(源语言中文): {main_path}")
    else:
        shutil.copy2(en_path, main_path)
        logger.info(f"  字幕(主): {main_path}")

    return main_path


def is_chinese_text(segments, threshold=0.3):
    """判断 SRT 段落文本是否主要为中文"""
    sample_text = ' '.join(s.get('text', '') for s in segments[:20])
    if not sample_text.strip():
        return False
    chinese_chars = sum(1 for c in sample_text if 'CJK' in unicodedata.name(c, ''))
    return chinese_chars / max(len(sample_text), 1) > threshold
