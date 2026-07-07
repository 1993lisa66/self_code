#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一视频处理器：process_video_unified 核心逻辑。
编排 ASR、翻译、TTS、合并等各阶段，支持四种处理模式。
"""

import os
import shutil
from pathlib import Path
from loguru import logger

from .stages import (
    asr_stage, llm_fix_stage, translation_stage, tts_stage, merge_stage,
    check_output_exists, check_has_audio,
    extract_audio_only, vad_only, asr_only,
    llm_fix_only, translation_only, srt_only,
    tts_prep_only, tts_generate_only, merge_only,
)
from .tts_from_srt import process_tts_from_srt
from ..subtitle.srt_utils import (
    find_srt_file, parse_srt_to_segments, generate_all_srt_files, is_chinese_text,
)
from ..utils.filename_translator import translate_filename


def process_video_unified(video_path, config, mode='tts_no_subtitle',
                           input_dir='', output_dir='',
                           skip_llm_fix=False):
    """
    统一视频处理函数。

    Args:
        video_path: 视频文件路径
        config: 配置字典
        mode: 处理模式
            - tts_no_subtitle: 生成中文配音（全流程：ASR→翻译→TTS→合成）
            - tts_from_srt: 从已有中文字幕生成配音
            - tts_with_review: 带人工审核的配音生成（两阶段）
        input_dir: 输入目录
        output_dir: 输出目录

    Returns:
        dict: 处理结果 {video_path, status, output_name, srt_path, tts_audio, final_video, message}
    """
    video_path = os.path.abspath(video_path)
    base_name = Path(video_path).stem

    # 翻译输出文件名
    output_base_name = translate_filename(base_name, config.get('llm', {}))

    # 计算目标输出目录（保持目录结构）
    if input_dir and os.path.isdir(input_dir):
        rel_path = os.path.relpath(video_path, input_dir)
        rel_dir = os.path.dirname(rel_path)
        target_output_dir = os.path.join(output_dir, rel_dir)
    else:
        target_output_dir = output_dir

    os.makedirs(target_output_dir, exist_ok=True)

    # 检查输出是否已存在（tts_from_srt 模式下跳过，因为可能是审核流程的视频副本）
    if mode != 'tts_from_srt' and check_output_exists(video_path, mode, target_output_dir, output_name=output_base_name):
        logger.info(f"⏭️  跳过（输出已存在）: {os.path.basename(video_path)}")
        return {
            'video_path': video_path, 'status': 'skipped',
            'output_name': output_base_name,
            'message': f'输出已存在（模式: {mode}）'
        }

    # ── tts_with_review 模式：检查是否已生成 SRT 字幕 ──
    if mode == 'tts_with_review':
        review_srt = os.path.join(target_output_dir, f"{output_base_name}.zh.srt")
        if os.path.exists(review_srt):
            logger.info(f"⏸️  字幕已生成，等待人工审核: {os.path.basename(video_path)}")
            logger.info(f"   请检查并修正: {review_srt}，确认无误后使用 --mode tts_from_srt 继续")
            return {
                'video_path': video_path, 'status': 'skipped',
                'output_name': output_base_name,
                'message': '字幕已生成，等待人工审核（修正 SRT 后使用 --mode tts_from_srt 继续）'
            }
        logger.info("📝 审核模式 Phase 1: 生成字幕供人工审核")

    # ── 智能检测已有字幕 ──
    _skip_audio_pipeline = False
    _existing_srt_segments = None
    _existing_srt_is_chinese = False
    _source_is_chinese = False

    if mode in ['tts_no_subtitle', 'tts_with_review']:
        existing_srt_path = find_srt_file(video_path)
        if existing_srt_path:
            srt_dir = os.path.dirname(existing_srt_path)
            srt_base = os.path.splitext(os.path.basename(existing_srt_path))[0]
            # 去掉 .en / .zh 后缀得到纯 base_name
            for suffix in ('.en', '.zh', '.chs', '.chi'):
                if srt_base.endswith(suffix):
                    srt_base = srt_base[:-len(suffix)]
                    break
            # 检测是否是工具自身生成的：如果同级目录存在配套 .zh.srt，说明是上次输出
            companion_zh = os.path.join(srt_dir, f"{srt_base}.zh.srt")
            _is_self_generated = os.path.exists(companion_zh)
            if _is_self_generated:
                logger.warning(
                    f"检测到上次工具生成的 SRT（存在配套 .zh.srt），将忽略已有字幕，重新进行 ASR")
                # 不设置 _skip_audio_pipeline，走完整流程
            else:
                srt_filename = os.path.basename(existing_srt_path)
                _existing_srt_segments = parse_srt_to_segments(existing_srt_path)
                _existing_srt_is_chinese = is_chinese_text(_existing_srt_segments)
                _skip_audio_pipeline = True
                logger.info(f"📄 检测到已有字幕: {srt_filename}，将跳过音频提取与 ASR")

    # 快速检查视频是否有音频流
    if mode != 'tts_from_srt' and not _skip_audio_pipeline:
        has_audio, has_video = check_has_audio(video_path)
        if not has_audio:
            logger.warning(f"⏭️  跳过（无音频流）: {os.path.basename(video_path)}")
            return {
                'video_path': video_path, 'status': 'skipped',
                'output_name': output_base_name, 'message': '视频文件没有音频流'
            }

    logger.info(f"{'='*60}")
    logger.info(f"开始处理视频: {os.path.basename(video_path)}")
    logger.info(f"处理模式: {mode}")
    logger.info(f"{'='*60}")

    # ── 缓存目录（放在输出目录中，跨运行持久化）──
    cache_dir = os.path.join(target_output_dir, ".cache", base_name)
    os.makedirs(cache_dir, exist_ok=True)

    # ── 检查是否有缓存的处理结果 ──
    _cached_fused = None
    _cached_translated = None
    fused_cache_path = os.path.join(cache_dir, "fused_results.json")
    translated_cache_path = os.path.join(cache_dir, "translated_results.json")

    if os.path.exists(fused_cache_path):
        import json as _json
        with open(fused_cache_path, 'r', encoding='utf-8') as _f:
            _cached_fused = _json.load(_f)
        logger.info(f"📦 命中 ASR+LLM修正缓存，跳过音频提取/VAD/ASR/LLM修正阶段")
    if os.path.exists(translated_cache_path):
        import json as _json
        with open(translated_cache_path, 'r', encoding='utf-8') as _f:
            _cached_translated = _json.load(_f)
        logger.info(f"📦 命中翻译缓存，跳过翻译阶段")

    _use_full_cache = _cached_fused is not None

    # ── tts_from_srt 模式：直接委托 ──
    if mode == 'tts_from_srt':
        return process_tts_from_srt(
            video_path, base_name, cache_dir, target_output_dir,
            config, output_base_name=output_base_name
        )

    try:
        # ── 从缓存恢复 / 重新计算 ──
        if _use_full_cache:
            # 从 JSON 缓存直接恢复 fused_results 和 translated_results
            fused_results = _cached_fused
            translated_results = _cached_translated
            # 检测源语言
            _source_is_chinese = is_chinese_text(
                [{'text': seg.get('text', '')} for seg in fused_results]
            )
            if _source_is_chinese:
                logger.info("🔍 检测到缓存中 ASR 输出已是中文（源语言为中文），将跳过翻译")
            logger.success(f"从缓存恢复完成，共 {len(fused_results)} 条片段")
        else:
            # ── Phase 1: ASR（或使用已有字幕） ──
            if _skip_audio_pipeline:
                whisperx_segments = [
                    {'start': s['start'], 'end': s['end'], 'text': s['text']}
                    for s in _existing_srt_segments
                ]
                segments = [
                    {'start': s['start'], 'end': s['end']}
                    for s in _existing_srt_segments
                ]
                logger.success(f"已有字幕加载完成，共 {len(whisperx_segments)} 条（跳过 ASR）")
            else:
                whisperx_segments, vad_segments = asr_stage(video_path, cache_dir, config)
                if whisperx_segments is None:
                    return {
                        'video_path': video_path, 'status': 'failed',
                        'output_name': output_base_name,
                        'message': 'ASR 识别结果为空' if vad_segments else '未检测到语音片段'
                    }
                segments = vad_segments

            # ── Phase 2: LLM 修正 ──
            fused_results = whisperx_segments
            if _skip_audio_pipeline or skip_llm_fix:
                if skip_llm_fix:
                    logger.info("\n[STEP 4/6] 跳过 LLM 修正（已配置 SKIP_LLM_FIX=True）")
                else:
                    logger.info("\n[STEP 4/6] 跳过 LLM 修正（已有字幕）")
            else:
                # 从 ASR 缓存加载完整双模型结果（含 glm），传给 llm_fix_stage 做投票融合
                import json as _json
                asr_results = {'whisperx': whisperx_segments}
                multi_asr_path = os.path.join(cache_dir, "asr", f"{base_name}_multi_asr.json")
                if os.path.exists(multi_asr_path):
                    with open(multi_asr_path, 'r', encoding='utf-8') as _f:
                        asr_results = _json.load(_f)
                fused_results = llm_fix_stage(asr_results, config)

            # ── 检测源语言 ──
            if not _skip_audio_pipeline or _existing_srt_segments:
                _source_is_chinese = is_chinese_text(
                    [{'text': seg.get('text', '')} for seg in fused_results]
                )
                if _source_is_chinese:
                    logger.info("🔍 检测到 ASR 输出已是中文（源语言为中文），将跳过翻译")

            # ── Phase 3: 翻译 ──
            translated_results = None

            if mode in ['tts_no_subtitle', 'tts_with_review']:
                translated_results = translation_stage(
                    fused_results, config,
                    source_is_chinese=(_existing_srt_is_chinese or _source_is_chinese)
                )

            # ── 保存缓存（下次运行时直接复用）──
            import json as _json
            with open(fused_cache_path, 'w', encoding='utf-8') as _f:
                _json.dump(fused_results, _f, ensure_ascii=False, indent=2)
            logger.debug(f"已保存 fused_results 缓存: {fused_cache_path}")
            if translated_results is not None:
                with open(translated_cache_path, 'w', encoding='utf-8') as _f:
                    _json.dump(translated_results, _f, ensure_ascii=False, indent=2)
                logger.debug(f"已保存 translated_results 缓存: {translated_cache_path}")

        srt_path = None
        tts_audio = None
        final_video = None

        # ── Phase 4: 生成 SRT 文件 ──
        if mode in ['tts_no_subtitle', 'tts_with_review']:
            logger.info("\n[STEP 6/6] 生成 SRT 字幕（中文 / 原始语言 / 双语）...")
            srt_path = generate_all_srt_files(
                fused_results=fused_results,
                translated_results=translated_results,
                output_dir=target_output_dir,
                base_name=output_base_name,
                source_is_chinese=_source_is_chinese
            )
            logger.success(f"SRT 字幕生成完成: {srt_path}")

        # ── tts_with_review：复制视频到输出目录 ──
        if mode == 'tts_with_review':
            # 将原始视频复制到输出目录，使用翻译后的名字，与 SRT 保持同名同目录
            original_ext = os.path.splitext(video_path)[1]
            copied_video = os.path.join(target_output_dir, f"{output_base_name}{original_ext}")
            if not os.path.exists(copied_video):
                shutil.copy2(video_path, copied_video)
                logger.info(f"📁 视频已复制到输出目录: {copied_video}")
            else:
                logger.info(f"📁 视频已存在于输出目录，跳过复制: {copied_video}")

            review_file = f"{os.path.join(target_output_dir, output_base_name)}.zh.srt" \
                if not _source_is_chinese else f"{os.path.join(target_output_dir, output_base_name)}.srt"
            _print_review_instructions(review_file, output_base_name, target_output_dir)
            return {
                'video_path': video_path, 'status': 'review_pending',
                'output_name': output_base_name, 'srt_path': srt_path,
                'message': '字幕已生成，等待人工审核（修正 SRT 后使用 --mode tts_from_srt 继续）'
            }

        # ── Phase 5: TTS + 合并（tts_no_subtitle） ──
        if mode == 'tts_no_subtitle':
            if translated_results is None:
                logger.error("TTS 模式需要翻译结果，但未找到")
                raise RuntimeError("翻译步骤缺失")

            tts_audio = tts_stage(translated_results, cache_dir, config)
            final_video = merge_stage(
                video_path, tts_audio, target_output_dir, config, output_base_name
            )

        logger.info(f"{'='*60}")
        logger.success(f"视频 [{output_base_name}] 处理完成！")
        logger.info(f"{'='*60}\n")

        return {
            'video_path': video_path, 'status': 'success',
            'output_name': output_base_name,
            'srt_path': srt_path, 'tts_audio': tts_audio, 'final_video': final_video,
            'message': '成功'
        }

    except Exception as e:
        logger.error(f"处理视频 {video_path} 时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'video_path': video_path, 'status': 'failed',
            'output_name': output_base_name, 'message': str(e)
        }
    # 缓存目录保留在输出目录中，供下次运行复用（不再自动清理）


def _print_review_instructions(review_file, output_base_name, output_dir):
    """打印人工审核指引"""
    print(f"\n{'='*60}")
    print(f"📝 字幕已生成，请进行人工审核：")
    print(f"   1️⃣  检查并修正文件: {review_file}")
    print(f"   2️⃣  修正翻译内容（可在任意文本编辑器或 SRT 编辑器中操作）")
    print(f"   3️⃣  确认无误后运行以下命令继续合成:")
    print(f"      python video_cli.py --mode tts_from_srt \"{output_dir}\"")
    print(f"      （视频已随 SRT 一同复制到输出目录，名称一致可直接匹配）")
    print(f"{'='*60}\n")
    logger.info(f"⏸️  等待人工审核: {output_base_name}")


def process_substep(video_path, config, substep, input_dir='', output_dir='',
                    skip_llm_fix=False):
    """分步处理单一步骤，供 video_cli.py 的批量编排使用。

    将 6 个子步骤（提取音频 / VAD / ASR / LLM修正 / 翻译 / SRT字幕）
    拆分为独立批次：先对全部视频完成上一步，再全部进入下一步。

    Args:
        substep: '_extract_audio' | '_vad' | '_asr' | '_llm_fix' | '_translation' | '_srt'

    Returns:
        dict: 与 process_video_unified 兼容的处理结果
    """
    video_path = os.path.abspath(video_path)
    base_name = Path(video_path).stem

    # 计算缓存目录（与 process_video_unified 保持一致）
    if input_dir and os.path.isdir(input_dir):
        rel_path = os.path.relpath(video_path, input_dir)
        rel_dir = os.path.dirname(rel_path)
        target_output_dir = os.path.join(output_dir, rel_dir)
    else:
        target_output_dir = output_dir

    cache_dir = os.path.join(target_output_dir, ".cache", base_name)
    os.makedirs(cache_dir, exist_ok=True)

    substep_labels = {
        '_extract_audio': ('提取音频', lambda: extract_audio_only(video_path, cache_dir, config)),
        '_vad': ('VAD 语音检测', lambda: vad_only(video_path, cache_dir, config)),
        '_asr': ('ASR 语音识别', lambda: asr_only(video_path, cache_dir, config)),
        '_llm_fix': ('LLM 修正',
                     lambda: llm_fix_only(video_path, cache_dir, config,
                                          skip_llm_fix=skip_llm_fix)),
        '_translation': ('翻译',
                         lambda: translation_only(video_path, cache_dir, config)),
        '_srt': ('生成 SRT 字幕',
                 lambda: srt_only(video_path, cache_dir, config, target_output_dir)),
        '_tts_prep': ('SRT解析+TTS预处理',
                      lambda: tts_prep_only(video_path, cache_dir, config)),
        '_tts_generate': ('TTS 语音合成',
                          lambda: tts_generate_only(video_path, cache_dir, config)),
        '_merge': ('合并视频+对齐SRT',
                   lambda: merge_only(video_path, cache_dir, config, target_output_dir)),
    }

    label, fn = substep_labels.get(substep, (substep, None))
    if fn is None:
        return {
            'video_path': video_path, 'status': 'failed',
            'message': f'未知的分步子步骤: {substep}', 'output_name': base_name
        }

    try:
        logger.info(f"  [{label}] {base_name}")
        result = fn()

        if substep == '_extract_audio':
            return {
                'video_path': video_path, 'status': 'success',
                'message': '音频提取完成', 'output_name': base_name
            }
        elif substep == '_vad':
            segs = result.get('segments', [])
            if not segs:
                return {
                    'video_path': video_path, 'status': 'failed',
                    'message': '未检测到语音片段', 'output_name': base_name
                }
            return {
                'video_path': video_path, 'status': 'success',
                'message': f'VAD 完成，{len(segs)} 个片段', 'output_name': base_name
            }
        elif substep == '_asr':
            if result is None:
                return {
                    'video_path': video_path, 'status': 'failed',
                    'message': 'ASR 识别结果为空', 'output_name': base_name
                }
            return {
                'video_path': video_path, 'status': 'success',
                'message': f'ASR 完成，{len(result)} 个片段', 'output_name': base_name
            }
        elif substep == '_llm_fix':
            if result is None:
                return {
                    'video_path': video_path, 'status': 'failed',
                    'message': 'LLM 修正失败', 'output_name': base_name
                }
            fused_segs = result.get('fused_results', [])
            return {
                'video_path': video_path, 'status': 'success',
                'message': f'LLM 修正完成，{len(fused_segs)} 个片段', 'output_name': base_name
            }
        elif substep == '_translation':
            if result is None:
                return {
                    'video_path': video_path, 'status': 'failed',
                    'message': '翻译失败（LLM修正缓存不存在）', 'output_name': base_name
                }
            translated_segs = result.get('translated_results', [])
            # 翻译完成后确定输出文件名，记录到 processing_state.json 保证一致性
            output_base_name = translate_filename(base_name, config.get('llm', {}))
            return {
                'video_path': video_path, 'status': 'success',
                'message': f'翻译完成，{len(translated_segs)} 条', 'output_name': output_base_name
            }
        elif substep == '_srt':
            if result is None:
                return {
                    'video_path': video_path, 'status': 'failed',
                    'message': 'SRT 生成失败（缓存缺失）', 'output_name': base_name
                }
            output_base_name = translate_filename(base_name, config.get('llm', {}))
            return {
                'video_path': video_path, 'status': 'success',
                'message': 'SRT 字幕生成完成',
                'output_name': output_base_name,
                'srt_path': result.get('srt_path')
            }
        elif substep == '_tts_prep':
            if result is None:
                return {
                    'video_path': video_path, 'status': 'failed',
                    'message': 'SRT 解析或 TTS 预处理失败', 'output_name': base_name
                }
            segs = result.get('segments', [])
            return {
                'video_path': video_path, 'status': 'success',
                'message': f'TTS 预处理完成，{len(segs)} 条', 'output_name': base_name
            }
        elif substep == '_tts_generate':
            if result is None:
                return {
                    'video_path': video_path, 'status': 'failed',
                    'message': 'TTS 语音合成失败', 'output_name': base_name
                }
            return {
                'video_path': video_path, 'status': 'success',
                'message': 'TTS 语音合成完成', 'output_name': base_name,
                'tts_audio': result.get('tts_audio')
            }
        elif substep == '_merge':
            if result is None:
                return {
                    'video_path': video_path, 'status': 'failed',
                    'message': '合并视频失败', 'output_name': base_name
                }
            output_base_name = translate_filename(base_name, config.get('llm', {}))
            return {
                'video_path': video_path, 'status': 'success',
                'message': '合并视频完成',
                'output_name': output_base_name,
                'final_video': result.get('final_video'),
                'srt_path': result.get('srt_path')
            }
    except Exception as e:
        logger.error(f"分步处理 [{substep}] {base_name} 出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'video_path': video_path, 'status': 'failed',
            'message': str(e), 'output_name': base_name
        }
