#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线各阶段处理函数：
  - asr_stage: 音频提取 → VAD → ASR
  - llm_fix_stage: LLM 修正 ASR 文本
  - translation_stage: 字幕翻译
  - tts_stage: TTS 文本预处理 + 语音合成
  - merge_stage: 合并配音到视频
"""

import os
import asyncio
from loguru import logger

from .prompt_helper import (
    load_prompt_template, get_project_root
)


def _get_lazy_import(name, import_fn):
    """惰性导入辅助（类级别的延迟加载）"""
    cache = {}
    def getter():
        if name not in cache:
            cache[name] = import_fn()
        return cache[name]
    return getter


# ── 惰性导入（仅在对应阶段调用时加载） ──
def _import_extract_audio():
    from ..audio.extract_audio import extract_audio
    return extract_audio

def _import_run_vad():
    from ..vad.vad_pipeline import run_vad
    return run_vad

def _import_run_multi_asr():
    from ..asr.multi_asr import run_multi_asr
    return run_multi_asr

def _import_fuse_asr():
    from ..llm.fuse_asr import fuse_asr_result
    return fuse_asr_result

def _import_translate():
    from ..translate.translate_pipeline import translate_segments
    return translate_segments

def _import_process_tts_text_batch():
    from ..tts.text_processor import process_tts_text_batch
    return process_tts_text_batch

def _import_generate_tts():
    from ..tts.tts_pipeline import generate_tts
    return generate_tts


# ── 音频流检查 ──

def check_has_audio(video_path):
    """快速检查视频文件是否有音频流。返回 (has_audio, has_video)"""
    import subprocess
    from ..utils.ffmpeg_utils import get_ffprobe_exe

    try:
        result = subprocess.run(
            [get_ffprobe_exe(), "-v", "error",
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True,
            env=os.environ.copy()
        )
        lines = result.stdout.strip().splitlines()
        return any("audio" in line for line in lines), any("video" in line for line in lines)
    except Exception:
        return True, True


# ── 输出文件存在性检查 ──

def check_output_exists(video_path, mode, output_dir, output_name=None):
    """检查是否已存在输出文件"""
    base_name = output_name if output_name else os.path.splitext(os.path.basename(video_path))[0]

    if mode in ['tts_no_subtitle', 'tts_from_srt', 'tts_with_review']:
        synthetic_video = os.path.join(output_dir, f"{base_name}.mp4")
        if not os.path.exists(synthetic_video):
            orig_name = os.path.splitext(os.path.basename(video_path))[0]
            synthetic_video = os.path.join(output_dir, f"{orig_name}.mp4")
        return os.path.exists(synthetic_video)

    return False


# ── 阶段函数 ──

# ── 分步子阶段函数（供 video_cli.py 批量编排使用）──


def extract_audio_only(video_path, cache_dir, config):
    """只提取音频（Phase-1/1），供分步批处理调用。extract_audio 内部已有缓存检查。"""
    logger.info("[Phase-1/1] 提取音频...")
    audio_output_dir = os.path.join(cache_dir, "audio")
    audio_path = _import_extract_audio()(
        video_path, output_dir=audio_output_dir,
        sample_rate=config['audio']['sample_rate']
    )
    logger.success(f"音频提取完成: {audio_path}")
    return {'audio_path': audio_path}


def vad_only(video_path, cache_dir, config):
    """只做 VAD（Phase-1/2），从缓存读取音频。run_vad 内部已有缓存检查。"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(cache_dir, "audio", f"{base_name}.wav")

    logger.info("[Phase-1/2] 语音活动检测 (VAD)...")
    vad_output_dir = os.path.join(cache_dir, "vad")
    segments = _import_run_vad()(
        audio_path, output_dir=vad_output_dir,
        device=config['asr']['device'],
        min_silence_duration_ms=config.get('vad', {}).get('min_silence_duration_ms', 500),
        min_speech_duration_ms=config.get('vad', {}).get('min_speech_duration_ms', 250)
    )
    logger.success(f"VAD 完成，检测到 {len(segments)} 个语音片段")
    return {'segments': segments}


def asr_only(video_path, cache_dir, config):
    """只做 ASR（Phase-1/3），从缓存读取 VAD 结果。ASR 缓存命中时自动跳过。

    Returns:
        whisperx_segments 列表，失败返回 None
    """
    import json as _json
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # 快速路径：ASR 缓存已存在则直接返回
    asr_output_dir = os.path.join(cache_dir, "asr")
    asr_cache_path = os.path.join(asr_output_dir, f"{base_name}_multi_asr.json")
    if os.path.exists(asr_cache_path):
        with open(asr_cache_path, 'r', encoding='utf-8') as f:
            asr_results = _json.load(f)
        whisperx_segments = asr_results.get('whisperx', [])
        logger.info(f"📦 命中 ASR 缓存，共 {len(whisperx_segments)} 个片段")
        return whisperx_segments if whisperx_segments else None

    audio_path = os.path.join(cache_dir, "audio", f"{base_name}.wav")

    # 读取缓存的 VAD 结果
    vad_json = os.path.join(cache_dir, "vad", f"{base_name}_segments.json")
    if not os.path.exists(vad_json):
        logger.error(f"VAD 缓存不存在: {vad_json}")
        return None
    with open(vad_json, 'r', encoding='utf-8') as f:
        segments = _json.load(f)

    logger.info("[Phase-1/3] 自动语音识别 (ASR)...")
    asr_results = _import_run_multi_asr()(
        audio_path, segments, output_dir=asr_output_dir,
        device=config['asr']['device'],
        model_size=config['asr'].get('model_size', 'base'),
        language=config['asr'].get('language') if config['asr'].get('language') != 'auto' else None,
    )

    whisperx_segments = asr_results.get('whisperx', [])
    if not whisperx_segments:
        logger.error("ASR 识别结果为空")
        return None
    logger.success(f"ASR 识别完成，共 {len(whisperx_segments)} 个片段")
    return whisperx_segments


# ── 步骤 4/5/6 分步子阶段函数（供 video_cli.py 批量编排使用）──


def _load_asr_whisperx(cache_dir):
    """从 ASR 缓存直接读取 whisperx_segments（不重新调用 asr_only 或加载模型）。"""
    import json as _json
    base_name = os.path.basename(cache_dir)
    asr_cache_path = os.path.join(cache_dir, "asr", f"{base_name}_multi_asr.json")
    if not os.path.exists(asr_cache_path):
        return None
    with open(asr_cache_path, 'r', encoding='utf-8') as f:
        asr_results = _json.load(f)
    return asr_results.get('whisperx', [])


def _load_asr_full(cache_dir):
    """从 ASR 缓存读取完整的 multi_asr 结果（含 whisperx + glm）。"""
    import json as _json
    base_name = os.path.basename(cache_dir)
    asr_cache_path = os.path.join(cache_dir, "asr", f"{base_name}_multi_asr.json")
    if not os.path.exists(asr_cache_path):
        return None
    with open(asr_cache_path, 'r', encoding='utf-8') as f:
        return _json.load(f)


def llm_fix_only(video_path, cache_dir, config, skip_llm_fix=False):
    """只做 LLM 修正（Phase-1/4），直接从 ASR 缓存读取结果。缓存命中时自动跳过。

    Returns:
        dict: {'fused_results': [...]}，失败返回 None
    """
    import json as _json
    fused_cache_path = os.path.join(cache_dir, "fused_results.json")
    if os.path.exists(fused_cache_path):
        with open(fused_cache_path, 'r', encoding='utf-8') as f:
            fused = _json.load(f)
        logger.info(f"📦 命中 LLM修正缓存，共 {len(fused)} 条片段")
        return {'fused_results': fused}

    # 直接从 ASR 缓存读取，不再重新调用 asr_only()（步骤 1.3 已完成 ASR）
    whisperx_segments = _load_asr_whisperx(cache_dir)
    if not whisperx_segments:
        logger.error("ASR 缓存不存在或为空，无法进行 LLM 修正")
        return None

    if skip_llm_fix:
        logger.info("[Phase-1/4] 跳过 LLM 修正（已配置 SKIP_LLM_FIX=True）")
        fused_results = whisperx_segments
        with open(fused_cache_path, 'w', encoding='utf-8') as f:
            _json.dump(fused_results, f, ensure_ascii=False, indent=2)
        logger.debug(f"已保存 fused_results 缓存: {fused_cache_path}")
        return {'fused_results': fused_results}

    logger.info("[Phase-1/4] LLM 修正字幕文本...")
    # 加载完整的双模型结果（含 whisperx + glm），支持投票融合
    full_asr = _load_asr_full(cache_dir)
    asr_results = full_asr if full_asr else {'whisperx': whisperx_segments}
    fused_results = llm_fix_stage(asr_results, config)

    with open(fused_cache_path, 'w', encoding='utf-8') as f:
        _json.dump(fused_results, f, ensure_ascii=False, indent=2)
    logger.debug(f"已保存 fused_results 缓存: {fused_cache_path}")

    return {'fused_results': fused_results}


def translation_only(video_path, cache_dir, config):
    """只做翻译（Phase-1/5），从缓存读取 LLM 修正结果。缓存命中时自动跳过。

    Returns:
        dict: {'translated_results': [...], 'source_is_chinese': bool}，失败返回 None
    """
    import json as _json
    from ..llm.resegmentation import semantic_resegment

    translated_cache_path = os.path.join(cache_dir, "translated_results.json")
    fused_cache_path = os.path.join(cache_dir, "fused_results.json")
    resegmented_cache_path = os.path.join(cache_dir, "resegmented_results.json")

    # ── 缓存命中路径 ──
    if os.path.exists(translated_cache_path) and os.path.exists(resegmented_cache_path):
        with open(translated_cache_path, 'r', encoding='utf-8') as f:
            translated = _json.load(f)
        from ..subtitle.srt_utils import is_chinese_text
        source_is_chinese = False
        if os.path.exists(fused_cache_path):
            with open(fused_cache_path, 'r', encoding='utf-8') as ff:
                fused_results = _json.load(ff)
            source_is_chinese = is_chinese_text(
                [{'text': seg.get('text', '')} for seg in fused_results]
            )
        logger.info(f"📦 命中翻译缓存，共 {len(translated)} 条片段")
        return {'translated_results': translated, 'source_is_chinese': source_is_chinese}

    # ── 加载 fused_results ──
    if not os.path.exists(fused_cache_path):
        logger.error(f"LLM 修正缓存不存在: {fused_cache_path}")
        return None
    with open(fused_cache_path, 'r', encoding='utf-8') as f:
        fused_results = _json.load(f)

    from ..subtitle.srt_utils import is_chinese_text
    source_is_chinese = is_chinese_text(
        [{'text': seg.get('text', '')} for seg in fused_results]
    )

    # ── 语义重切分：合并 VAD 断句碎片为完整句子（仅非中文源）──
    if not source_is_chinese and not os.path.exists(resegmented_cache_path):
        logger.info("[重切分] 合并断开片段为完整句子...")
        fused_results = semantic_resegment(fused_results, config=config.get('llm', {}))
        with open(resegmented_cache_path, 'w', encoding='utf-8') as rf:
            _json.dump(fused_results, rf, ensure_ascii=False, indent=2)
        with open(fused_cache_path, 'w', encoding='utf-8') as ff:
            _json.dump(fused_results, ff, ensure_ascii=False, indent=2)
        logger.success(f"语义重切分完成，片段数 → {len(fused_results)} 条")
        # 清空旧翻译缓存（片段数已变化）
        if os.path.exists(translated_cache_path):
            with open(translated_cache_path, 'w', encoding='utf-8') as tf:
                _json.dump([], tf)
    elif os.path.exists(resegmented_cache_path):
        # 已有重切分缓存，直接用
        with open(resegmented_cache_path, 'r', encoding='utf-8') as rf:
            fused_results = _json.load(rf)

    logger.info("[Phase-1/5] 翻译字幕（→ 中文）...")
    translated_results = translation_stage(
        fused_results, config, source_is_chinese=source_is_chinese
    )

    with open(translated_cache_path, 'w', encoding='utf-8') as f:
        _json.dump(translated_results, f, ensure_ascii=False, indent=2)
    logger.debug(f"已保存 translated_results 缓存: {translated_cache_path}")

    return {'translated_results': translated_results, 'source_is_chinese': source_is_chinese}


def srt_only(video_path, cache_dir, config, target_output_dir):
    """只生成 SRT 字幕（Phase-1/6），从缓存读取所有结果。

    Returns:
        dict: {'srt_path': ..., 'source_is_chinese': ..., 'output_base_name': ...}
    """
    import json as _json

    fused_cache_path = os.path.join(cache_dir, "fused_results.json")
    translated_cache_path = os.path.join(cache_dir, "translated_results.json")
    resegmented_cache_path = os.path.join(cache_dir, "resegmented_results.json")

    if not os.path.exists(fused_cache_path):
        logger.error(f"LLM 修正缓存不存在: {fused_cache_path}")
        return None
    # 优先使用语义重切分后的结果（与翻译对齐，时间边界一致）
    if os.path.exists(resegmented_cache_path):
        with open(resegmented_cache_path, 'r', encoding='utf-8') as f:
            fused_results = _json.load(f)
    else:
        with open(fused_cache_path, 'r', encoding='utf-8') as f:
            fused_results = _json.load(f)

    if not os.path.exists(translated_cache_path):
        logger.error(f"翻译缓存不存在: {translated_cache_path}")
        return None
    with open(translated_cache_path, 'r', encoding='utf-8') as f:
        translated_results = _json.load(f)

    from ..subtitle.srt_utils import is_chinese_text
    source_is_chinese = is_chinese_text(
        [{'text': seg.get('text', '')} for seg in fused_results]
    )

    from ..utils.filename_translator import translate_filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    llm_config = config.get('llm', {})
    output_base_name = translate_filename(base_name, llm_config)

    logger.info("[Phase-1/6] 生成 SRT 字幕（中文 / 原始语言 / 双语）...")
    from ..subtitle.srt_utils import generate_all_srt_files
    srt_path = generate_all_srt_files(
        fused_results=fused_results,
        translated_results=translated_results,
        output_dir=target_output_dir,
        base_name=output_base_name,
        source_is_chinese=source_is_chinese
    )
    logger.success(f"SRT 字幕生成完成: {srt_path}")
    return {
        'srt_path': srt_path, 'source_is_chinese': source_is_chinese,
        'output_base_name': output_base_name
    }


# ── 阶段二子步骤（供 video_cli.py 批量编排使用）──


def tts_prep_only(video_path, cache_dir, config, target_output_dir=None):
    """阶段二 Step 1：SRT 解析 + TTS 文本预处理。缓存命中时自动跳过。

    Returns:
        dict: {'segments': [...]}，失败返回 None
    """
    import json as _json
    prepped_cache_path = os.path.join(cache_dir, "tts_preprocessed.json")

    if os.path.exists(prepped_cache_path):
        with open(prepped_cache_path, 'r', encoding='utf-8') as f:
            prepped = _json.load(f)
        logger.info(f"📦 命中 TTS预处理缓存，共 {len(prepped)} 条片段")
        return {'segments': prepped}

    logger.info("[Phase-2/1] SRT 解析 + TTS 文本预处理...")

    from pathlib import Path as _Path
    from ..subtitle.srt_utils import find_srt_file, parse_srt_to_segments
    from ..utils.filename_translator import translate_filename as _translate_filename

    original_base_name = _Path(video_path).stem

    # 在输出目录中查找 .zh.srt 文件（按优先级：翻译名 > 原始名 > 同编号前缀）
    srt_path = None
    if target_output_dir:
        # 1) 优先用翻译后的文件名（与 srt_only 成功翻译时一致）
        output_base_name = _translate_filename(original_base_name, config.get('llm', {}))
        candidate = os.path.join(target_output_dir, f"{output_base_name}.zh.srt")
        if os.path.exists(candidate):
            srt_path = candidate
            logger.info(f"  在输出目录找到字幕: {srt_path}")

    if not srt_path and target_output_dir:
        # 2) 回退：原始文件名（翻译失败或缓存不一致时）
        candidate_orig = os.path.join(target_output_dir, f"{original_base_name}.zh.srt")
        if os.path.exists(candidate_orig):
            srt_path = candidate_orig
            logger.info(f"  在输出目录找到字幕(原始名): {srt_path}")

    if not srt_path and target_output_dir:
        # 3) 模糊匹配：同编号前缀的任一 .zh.srt（处理文件名被部分修改的极端情况）
        import re as _re
        prefix_match = _re.match(r'^(\d+\s*[-._]\s*)', original_base_name)
        if prefix_match:
            prefix = prefix_match.group(0)
            try:
                for fname in os.listdir(target_output_dir):
                    if fname.startswith(prefix) and fname.endswith('.zh.srt'):
                        srt_path = os.path.join(target_output_dir, fname)
                        logger.info(f"  在输出目录找到字幕(前缀匹配): {srt_path}")
                        break
            except OSError:
                pass

    # 回退：在视频所在目录查找
    if not srt_path:
        srt_path = find_srt_file(video_path, lang='zh')

    if not srt_path:
        logger.error("未找到中文字幕文件（SRT）")
        return None
    logger.info(f"  找到字幕: {srt_path}")

    segments = parse_srt_to_segments(srt_path)
    if not segments:
        logger.error("SRT 文件解析为空")
        return None
    logger.success(f"字幕解析完成，共 {len(segments)} 条")

    # ── 回填原文（原始语言，如英文）──
    translated_cache_path = os.path.join(cache_dir, "translated_results.json")
    if os.path.exists(translated_cache_path):
        with open(translated_cache_path, 'r', encoding='utf-8') as f:
            translated_results = _json.load(f)
        _backfill_original_text(segments, translated_results)

    # TTS 文本预处理（LLM 调用）
    prompts_dir = config['paths'].get('prompts_dir', os.path.join(get_project_root(), 'config', 'prompts'))
    prompt_template = load_prompt_template(prompts_dir, 'tts_prep.txt')
    llm_config = dict(config.get('llm', {}))
    # 合并 translate 子配置（batch_size 等关键参数）
    translate_cfg = config.get('translate', {})
    for k in ('batch_size', 'max_tokens', 'temperature'):
        if k in translate_cfg:
            llm_config.setdefault(k, translate_cfg[k])

    tts_texts = [seg['text'] for seg in segments]
    processed = _import_process_tts_text_batch()(
        tts_texts, config=llm_config, prompt_template=prompt_template
    )

    for idx, tts_text in enumerate(processed):
        segments[idx]["tts_text"] = tts_text
        segments[idx]["translated_text"] = tts_text

    logger.success("TTS 文本预处理完成")

    with open(prepped_cache_path, 'w', encoding='utf-8') as f:
        _json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.debug(f"已保存 tts_preprocessed 缓存: {prepped_cache_path}")

    return {'segments': segments}


def _backfill_original_text(segments, translated_results):
    """将 translated_results 中的原文（原始语言）回填到 SRT segments 的 original_text 字段。
    
    通过翻译后文本（translated_text）与 SRT 文本（text）做匹配，
    找到对应的原文并写入 original_text。
    """
    if not translated_results:
        return
    # 构建 translated_text → original_text 查找表
    lookup = {}
    for tr in translated_results:
        tt = (tr.get('translated_text') or '').strip()
        ot = (tr.get('text') or '').strip()
        if tt and ot and tt != ot:
            lookup[tt] = ot
    if not lookup:
        return

    matched = 0
    for i, seg in enumerate(segments):
        seg_text = (seg.get('text') or '').strip()
        if seg_text in lookup:
            seg['original_text'] = lookup[seg_text]
            matched += 1
        else:
            # 模糊匹配：翻译后文本可能被 TTS 预处理微调过
            found = False
            for tt_key, ot_val in lookup.items():
                if len(seg_text) > 3 and len(tt_key) > 3:
                    # 用最长公共前缀/后缀或相似度判断
                    if (seg_text[:6] == tt_key[:6] or seg_text[-6:] == tt_key[-6:]
                            or abs(len(seg_text) - len(tt_key)) < 5):
                        seg['original_text'] = ot_val
                        matched += 1
                        found = True
                        break
            if not found:
                # 按索引回退匹配
                if i < len(translated_results):
                    tr = translated_results[i]
                    ot = (tr.get('text') or '').strip()
                    tt = (tr.get('translated_text') or '').strip()
                    if ot and tt and ot != tt:
                        seg['original_text'] = ot
                        matched += 1

    if matched > 0:
        logger.info(f"  已回填原文 (original_text): {matched}/{len(segments)} 条")


def tts_generate_only(video_path, cache_dir, config):
    """阶段二 Step 2：TTS 语音合成。从缓存读取预处理结果，缓存命中时自动跳过。

    Returns:
        dict: {'tts_audio': path}，失败返回 None
    """
    import json as _json

    tts_output_dir = os.path.join(cache_dir, "tts")
    full_audio_path = os.path.join(tts_output_dir, "full_tts.mp3")

    if os.path.exists(full_audio_path):
        logger.info(f"📦 命中 TTS 音频缓存: {full_audio_path}")
        return {'tts_audio': full_audio_path}

    prepped_cache_path = os.path.join(cache_dir, "tts_preprocessed.json")
    if not os.path.exists(prepped_cache_path):
        logger.error(f"TTS 预处理缓存不存在: {prepped_cache_path}")
        return None
    with open(prepped_cache_path, 'r', encoding='utf-8') as f:
        segments = _json.load(f)

    logger.info("[Phase-2/2] 生成 TTS 语音...")
    tts_audio = asyncio.run(_import_generate_tts()(
        segments, output_dir=tts_output_dir, config=config['tts']
    ))

    logger.success(f"TTS 语音生成完成: {tts_audio}")
    return {'tts_audio': tts_audio}


def merge_only(video_path, cache_dir, config, target_output_dir):
    """阶段二 Step 3：合并视频 + 生成对齐 SRT。

    Returns:
        dict: {'final_video': ..., 'srt_path': ..., 'output_base_name': ...}
    """
    import json as _json

    tts_audio_path = os.path.join(cache_dir, "tts", "full_tts.mp3")
    if not os.path.exists(tts_audio_path):
        logger.error(f"TTS 音频缓存不存在: {tts_audio_path}")
        return None

    prepped_cache_path = os.path.join(cache_dir, "tts_preprocessed.json")
    if not os.path.exists(prepped_cache_path):
        logger.error(f"TTS 预处理缓存不存在: {prepped_cache_path}")
        return None
    with open(prepped_cache_path, 'r', encoding='utf-8') as f:
        segments = _json.load(f)

    from ..utils.filename_translator import translate_filename
    output_base_name = translate_filename(
        os.path.splitext(os.path.basename(video_path))[0],
        config.get('llm', {})
    )

    logger.info("[Phase-2/3] 合并视频 + 生成对齐 SRT...")
    from ..merge.merge_video import merge_video
    merge_config = config.get('video', {}).copy()
    final_video = merge_video(
        video_path, tts_audio_path, output_dir=target_output_dir,
        config=merge_config, output_name=output_base_name
    )
    logger.success(f"最终视频生成完成: {final_video}")

    from ..subtitle.generate_srt import generate_srt
    srt_output_path = os.path.join(target_output_dir, f"{output_base_name}.zh.srt")
    srt_path_out = generate_srt(segments, output_path=srt_output_path)
    logger.success(f"对齐版 SRT 生成完成: {srt_path_out}")

    return {
        'final_video': final_video, 'srt_path': srt_path_out,
        'output_base_name': output_base_name
    }


def asr_stage(video_path, cache_dir, config):
    """
    ASR 阶段（组合调用）：提取音频 → VAD → ASR 识别。

    Returns:
        (whisperx_segments, segments): ASR 识别结果和 VAD 片段
    """
    result = extract_audio_only(video_path, cache_dir, config)

    vad_result = vad_only(video_path, cache_dir, config)
    segments = vad_result.get('segments', [])
    if not segments:
        return None, None

    whisperx_segments = asr_only(video_path, cache_dir, config)
    if whisperx_segments is None:
        return None, segments

    return whisperx_segments, segments


def llm_fix_stage(asr_results, config):
    """
    LLM 修正阶段：使用 LLM 修正 ASR 识别的文本。

    Args:
        asr_results: ASR 多引擎结果字典（含 whisperx / glm）
        config: 完整配置

    Returns:
        fused_results: 修正后的 segments 列表
    """
    whisperx_segments = asr_results.get('whisperx', [])
    if not config.get('llm', {}).get('api_key'):
        logger.info("\n[STEP 4/6] 跳过 LLM 修正（未配置 API Key）")
        return whisperx_segments

    logger.info("\n[STEP 4/6] LLM 修正字幕文本...")
    prompts_dir = config['paths'].get('prompts_dir', os.path.join(get_project_root(), 'config', 'prompts'))
    prompt_template = load_prompt_template(prompts_dir, 'asr_fix.txt')
    if prompt_template:
        logger.info(f"  加载提示词模板: asr_fix.txt")

    llm_config = dict(config.get('llm', {}))
    # 合并 translate 子配置（batch_size 等关键参数）
    translate_cfg = config.get('translate', {})
    for k in ('batch_size', 'max_tokens', 'temperature'):
        if k in translate_cfg:
            llm_config.setdefault(k, translate_cfg[k])

    try:
        logger.info("  开始 LLM 修正...")
        fused_results = _import_fuse_asr()(
            asr_results, config=llm_config, prompt_template=prompt_template
        )
        logger.success(f"LLM 修正完成")
        return fused_results
    except Exception as e:
        logger.warning(f"LLM 修正失败，使用原始 ASR 结果: {e}")
        return whisperx_segments


def translation_stage(fused_results, config, source_is_chinese=False):
    """
    翻译阶段：将 ASR 文本翻译为中文。

    Returns:
        translated_results: 带 translated_text 字段的 segments 列表
    """
    if source_is_chinese:
        logger.info("\n[STEP 5/6] 源语言已是中文，跳过翻译步骤...")
        translated = [{**seg, 'translated_text': seg.get('text', '')} for seg in fused_results]
        logger.success(f"跳过翻译（源语言已是中文，共 {len(translated)} 条）")
        return translated

    logger.info("\n[STEP 5/6] 翻译字幕（→ 中文）...")
    prompts_dir = config['paths'].get('prompts_dir', os.path.join(get_project_root(), 'config', 'prompts'))
    prompt_template = load_prompt_template(prompts_dir, 'translation.txt')

    llm_config = dict(config.get('llm', {}))
    # 合并 translate 子配置（batch_size 等关键参数）
    translate_cfg = config.get('translate', {})
    for k in ('batch_size', 'max_tokens', 'temperature'):
        if k in translate_cfg:
            llm_config.setdefault(k, translate_cfg[k])

    logger.info("  开始翻译...")
    translated_results = _import_translate()(
        fused_results, target_lang='zh', config=llm_config, prompt_template=prompt_template
    )

    logger.success(f"翻译完成")
    return translated_results


def tts_stage(translated_results, cache_dir, config):
    """
    TTS 阶段：文本预处理 + 语音合成。

    Returns:
        tts_audio: TTS 音频文件路径
    """
    # TTS 文本预处理
    logger.info("\n[额外步骤] TTS 文本预处理...")
    prompts_dir = config['paths'].get('prompts_dir', os.path.join(get_project_root(), 'config', 'prompts'))
    prompt_template = load_prompt_template(prompts_dir, 'tts_prep.txt')
    llm_config = dict(config.get('llm', {}))
    # 合并 translate 子配置（batch_size 等关键参数）
    translate_cfg = config.get('translate', {})
    for k in ('batch_size', 'max_tokens', 'temperature'):
        if k in translate_cfg:
            llm_config.setdefault(k, translate_cfg[k])

    tts_texts = [seg.get("translated_text", seg["text"]) for seg in translated_results]
    logger.info("  开始 TTS 文本预处理...")
    processed = _import_process_tts_text_batch()(
        tts_texts, config=llm_config, prompt_template=prompt_template
    )
    for idx, tts_text in enumerate(processed):
        translated_results[idx]["tts_text"] = tts_text
    logger.success(f"TTS 文本预处理完成")

    # 生成 TTS 音频
    logger.info("生成 TTS 配音...")
    tts_output_dir = os.path.join(cache_dir, "tts")
    tts_audio = asyncio.run(_import_generate_tts()(
        translated_results, output_dir=tts_output_dir, config=config['tts']
    ))
    logger.success(f"TTS 配音生成完成: {tts_audio}")
    return tts_audio


def merge_stage(video_path, tts_audio, target_output_dir, config, output_base_name):
    """合并阶段：将 TTS 配音合成到原视频"""
    logger.info("\n[额外步骤] 合成最终视频...")
    from ..merge.merge_video import merge_video

    merge_config = config.get('video', {}).copy()
    final_video = merge_video(
        video_path, tts_audio, output_dir=target_output_dir,
        config=merge_config, output_name=output_base_name
    )
    logger.success(f"最终视频生成完成: {final_video}")
    return final_video
