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
from datetime import datetime
from loguru import logger
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 抑制 numpy/faster_whisper 的 matmul 溢出/除零警告（延迟导入，仅对应模式触发）
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

# 配置 pydub 的 FFmpeg 路径（必须在导入 pydub 之前）
from modules.utils.ffmpeg_utils import get_ffmpeg_exe, get_ffprobe_exe
from modules.utils.rate_limiter import wait_for_llm_api, mark_model_overloaded, is_model_overloaded, clear_overload
ffmpeg_path = get_ffmpeg_exe()
ffprobe_path = get_ffprobe_exe()

if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")

# ── 轻量模块：始终导入（不依赖 GPU/ML 框架） ──
from modules.subtitle.generate_srt import generate_srt
from modules.merge.merge_video import merge_video

# ── 重量级模块：按需延迟导入（tts_from_srt 模式不需要 ASR/VAD/翻译） ──
_extract_audio = None
_run_vad = None
_run_multi_asr = None
_fuse_asr_result = None
_translate_segments = None
_generate_tts = None
_process_tts_text_batch = None
_has_torch_patch = False


def _ensure_torch_patch():
    """确保 PyTorch 补丁已应用（仅在需要时触发）"""
    global _has_torch_patch
    if not _has_torch_patch:
        from modules.utils.patch_torch import apply_torch_patch
        apply_torch_patch()
        _has_torch_patch = True


def _get_extract_audio():
    global _extract_audio
    if _extract_audio is None:
        _ensure_torch_patch()
        from modules.audio.extract_audio import extract_audio as _fn
        _extract_audio = _fn
    return _extract_audio


def _get_run_vad():
    global _run_vad
    if _run_vad is None:
        _ensure_torch_patch()
        from modules.vad.vad_pipeline import run_vad as _fn
        _run_vad = _fn
    return _run_vad


def _get_run_multi_asr():
    global _run_multi_asr
    if _run_multi_asr is None:
        _ensure_torch_patch()
        from modules.asr.multi_asr import run_multi_asr as _fn
        _run_multi_asr = _fn
    return _run_multi_asr


def _get_fuse_asr_result():
    global _fuse_asr_result
    if _fuse_asr_result is None:
        from modules.llm.fuse_asr import fuse_asr_result as _fn
        _fuse_asr_result = _fn
    return _fuse_asr_result


def _get_translate_segments():
    global _translate_segments
    if _translate_segments is None:
        from modules.translate.translate_pipeline import translate_segments as _fn
        _translate_segments = _fn
    return _translate_segments


def _get_generate_tts():
    global _generate_tts
    if _generate_tts is None:
        from modules.tts.tts_pipeline import generate_tts as _fn
        _generate_tts = _fn
    return _generate_tts


def _get_process_tts_text_batch():
    global _process_tts_text_batch
    if _process_tts_text_batch is None:
        from modules.tts.text_processor import process_tts_text_batch as _fn
        _process_tts_text_batch = _fn
    return _process_tts_text_batch


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


# ──────────────────────────────────────────────
#  处理状态追踪器：记录哪些文件已处理 / 未处理 / 新增
# ──────────────────────────────────────────────

TRACKER_FILENAME = "processing_state.json"


class ProcessingTracker:
    """
    持久化处理状态追踪器。
    在输出目录保存 `processing_state.json`，记录每个视频文件
    的处理状态、时间、模式等信息。即使缓存被清除，也能知道哪些
    文件已处理、哪些是新增的、哪些之前失败了。
    """

    def __init__(self, output_dir, input_dir=None):
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.filepath = os.path.join(output_dir, TRACKER_FILENAME)
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": 1, "files": {}}

    def save(self):
        """保存追踪数据到磁盘"""
        self.data["last_updated"] = datetime.now().isoformat()
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _relpath(self, video_path):
        """计算视频相对路径作为追踪 key"""
        if self.input_dir and os.path.isdir(self.input_dir):
            try:
                return os.path.relpath(video_path, self.input_dir)
            except ValueError:
                pass
        return os.path.basename(video_path)

    def update(self, video_path, result, mode, output_name=None):
        """更新单个文件的处理记录"""
        rel = self._relpath(video_path)
        self.data["files"][rel] = {
            "status": result.get("status", "unknown"),
            "mode": mode,
            "output_name": output_name if output_name else Path(video_path).stem,
            "processed_at": datetime.now().isoformat(),
            "message": result.get("message", "")
        }

    def get_summary(self, video_files):
        """
        对比输入文件与历史记录，返回状态摘要。

        Returns:
            dict: {
                "total": 总数, "new": 新增, "completed": 已完成,
                "failed": 之前失败, "skipped": 需跳过
            }
        """
        tracked = set(self.data.get("files", {}).keys())
        current = {self._relpath(vf) for vf in video_files}

        # 已跟踪且成功的（输出文件仍存在）
        completed_success = set()
        # 已跟踪但失败的
        previously_failed = set()
        for k in tracked & current:
            entry = self.data["files"][k]
            if entry.get("status") == "success":
                completed_success.add(k)
            elif entry.get("status") == "failed":
                previously_failed.add(k)

        new_files = current - tracked

        return {
            "total": len(video_files),
            "new": sorted(new_files),
            "completed": sorted(completed_success),
            "previously_failed": sorted(previously_failed),
        }

    def get_entry(self, video_path):
        """获取单个文件的追踪记录，不存在返回 None"""
        rel = self._relpath(video_path)
        return self.data.get("files", {}).get(rel)


def check_output_exists(video_path, mode, output_dir, output_name=None):
    """
    检查是否已存在输出文件
    
    Args:
        video_path: 视频文件路径
        mode: 处理模式
        output_dir: 输出目录
        output_name: 翻译后的输出文件名（不含扩展名），若提供则优先检查此名称
    
    Returns:
        bool: True 如果输出已存在，False 否则
    """
    base_name = output_name if output_name else Path(video_path).stem
    
    if mode in ['subtitle_only', 'subtitle_bilingual', 'subtitle_chinese']:
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        # 同时检查原始名称（兼容历史输出）
        if not os.path.exists(srt_path):
            orig_name = Path(video_path).stem
            srt_path = os.path.join(output_dir, f"{orig_name}.srt")
        return os.path.exists(srt_path)
    elif mode in ['tts_no_subtitle', 'tts_from_srt']:
        synthetic_video = os.path.join(output_dir, f"{base_name}.mp4")
        if not os.path.exists(synthetic_video):
            orig_name = Path(video_path).stem
            synthetic_video = os.path.join(output_dir, f"{orig_name}.mp4")
        return os.path.exists(synthetic_video)
    
    return False


# ── 文件名翻译缓存（仅内存，不写文件） ──
_FILENAME_TRANSLATION_CACHE = {}  # {original_name: translated_name}


def _translate_filename(name, llm_config):
    """
    使用 LLM 将英文文件名翻译为中文。
    优先级：缓存命中 > API 翻译(带降级) > 原名。

    Args:
        name: 原始文件名（不含扩展名）
        llm_config: LLM 配置字典，包含 api_key, api_base, model

    Returns:
        str: 翻译后的文件名
    """
    if not name or not llm_config or not llm_config.get('api_key'):
        return name

    # 如果文件名已经主要是中文，无需翻译
    import unicodedata
    chinese_chars = sum(1 for c in name if 'CJK' in unicodedata.name(c, ''))
    if chinese_chars > len(name) * 0.3:
        return name

    # 缓存命中，直接返回
    if name in _FILENAME_TRANSLATION_CACHE:
        cached = _FILENAME_TRANSLATION_CACHE[name]
        logger.debug(f"  文件名翻译(缓存): \"{name}\" → \"{cached}\"")
        return cached

    import random
    import time
    import re
    max_retries = 2  # 文件名翻译优先级低，减少重试
    base_delay = 2
    fallback_model = llm_config.get('fallback_model') if llm_config else None
    primary_model = llm_config.get('model', 'deepseek-ai/DeepSeek-V3')
    
    # 提取编号前缀（如 "01 - ", "02-", "03.", "001_", 等），只翻译正文部分
    number_prefix_match = re.match(r'^(\d+\s*[-._]\s*)', name)
    number_prefix = number_prefix_match.group(0) if number_prefix_match else ''
    text_to_translate = name[number_prefix_match.end():].strip() if number_prefix_match else name
    
    # 如果去掉前缀后只剩空文本，无需翻译
    if not text_to_translate:
        return name
    
    # 全局过载检测：主模型已知不可用，直接启动备用模型
    if is_model_overloaded() and fallback_model:
        current_model = fallback_model
    else:
        current_model = primary_model

    for attempt in range(max_retries):
        # 首次失败即切换备用模型（文件名翻译优先级低，不浪费主模型配额）
        if attempt >= 1 and fallback_model and current_model != fallback_model:
            current_model = fallback_model

        try:
            wait_for_llm_api()  # 遵守全局速率限制
            from openai import OpenAI
            client = OpenAI(
                api_key=llm_config['api_key'],
                base_url=llm_config.get('api_base', 'https://api.siliconflow.cn/v1')
            )

            response = client.chat.completions.create(
                model=current_model,
                messages=[{
                    'role': 'user',
                    'content': (
                        f'请将以下英文标题翻译成简洁的中文（10字以内，意译优先，用词专业自然）：\n'
                        f'"{text_to_translate}"\n\n只输出翻译结果，不要任何解释或引号。'
                    )
                }],
                max_tokens=30,
                temperature=0.1
            )
            translated = response.choices[0].message.content.strip()
            translated = translated.strip('"\' 。，, \n\r\t')
            # 移除文件名不允许的字符
            translated = re.sub(r'[\\/:*?"<>|]', '', translated)
            if translated and len(translated) <= 50:
                # 拼回编号前缀
                full_translated = number_prefix + translated
                logger.info(f"  文件名翻译: \"{name}\" → \"{full_translated}\"")
                _FILENAME_TRANSLATION_CACHE[name] = full_translated
                # 如果用备用模型成功了且主模型之前被标记过载，尝试清除标记
                if current_model == fallback_model:
                    clear_overload()  # 主模型可能已恢复
                return full_translated
        except Exception as e:
            err_msg = str(e)
            if '429' in err_msg or 'rate' in err_msg.lower():
                mark_model_overloaded()  # 通知全局
                delay = (base_delay ** (attempt + 1)) * (0.5 + random.random())
                logger.debug(f"  翻译限流，{delay:.1f}s 后重试 ({attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            logger.warning(f"  文件名翻译失败（使用原名）: {e}")
            break
    else:
        logger.warning(f"  文件名翻译失败（{max_retries}次重试后仍失败，使用原名）")

    return name


def _find_srt_file(video_path):
    """
    查找视频文件对应的 SRT 字幕文件。
    按优先级搜索：同名 .srt、中文相关后缀 .chs.srt / .chi.srt / .zh.srt 等
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        str or None: SRT 文件路径
    """
    base_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    candidates = [
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


def _parse_srt_to_segments(srt_path):
    """
    解析 SRT 字幕文件，返回包含 start/end/text 的 segment 列表。
    
    Args:
        srt_path: SRT 文件路径
    
    Returns:
        list[dict]: [{'start': float, 'end': float, 'text': str}, ...]
    """
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


def process_video_unified(video_path, config, mode='subtitle_only', input_dir='', output_dir='', batch_name=None, skip_llm_fix=False):
    """
    统一视频处理函数
    
    Args:
        video_path: 视频文件路径
        config: 配置字典
        mode: 处理模式
            - subtitle_only: 仅生成中文字幕（ASR 识别）
            - subtitle_bilingual: 生成中英双语字幕
            - subtitle_chinese: 生成中文字幕（翻译后）
            - tts_no_subtitle: 生成中文配音（全流程：ASR→翻译→TTS→合成）
            - tts_from_srt: 从已有中文字幕生成配音（跳过ASR/翻译，直接SRT→TTS→合成）
        input_dir: 输入目录
        output_dir: 输出目录
        batch_name: 批次名称（可选，用于加载批次专属提示词）
    
    Returns:
        dict: 处理结果
    """
    video_path = os.path.abspath(video_path)
    base_name = Path(video_path).stem
    
    # 翻译输出文件名（如果 LLM 可用，将英文标题译为中文文件名）
    output_base_name = _translate_filename(base_name, config.get('llm', {}))
    
    # 计算相对路径用于保持目录结构
    if input_dir and os.path.isdir(input_dir):
        rel_path = os.path.relpath(video_path, input_dir)
        rel_dir = os.path.dirname(rel_path)
        target_output_dir = os.path.join(output_dir, rel_dir)
    else:
        target_output_dir = output_dir
    
    # 确保输出目录存在
    os.makedirs(target_output_dir, exist_ok=True)
    
    # 检查输出是否已存在（优先检查翻译后的文件名）
    if check_output_exists(video_path, mode, target_output_dir, output_name=output_base_name):
        logger.info(f"⏭️  跳过（输出已存在）: {os.path.basename(video_path)}")
        return {
            'video_path': video_path,
            'status': 'skipped',
            'output_name': output_base_name,
            'message': f'输出已存在（模式: {mode}）'
        }
    
    # 快速检查视频是否有音频流（tts_from_srt 模式依赖字幕而非原音频，跳过检查）
    if mode != 'tts_from_srt':
        has_audio, has_video = _check_has_audio(video_path)
        if not has_audio:
            logger.warning(f"⏭️  跳过（无音频流）: {os.path.basename(video_path)}")
            return {
                'video_path': video_path,
                'status': 'skipped',
                'output_name': output_base_name,
                'message': '视频文件没有音频流'
            }
    
    logger.info(f"{'='*60}")
    logger.info(f"开始处理视频: {os.path.basename(video_path)}")
    logger.info(f"处理模式: {mode}")
    logger.info(f"{'='*60}")
    
    # 创建临时缓存目录（内部使用原名，避免特殊字符问题）
    cache_dir = os.path.join(project_root, "cache", "video_processor", base_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    # ── tts_from_srt 模式：跳过 ASR/翻译，直接从已有字幕生成中文配音 ──
    if mode == 'tts_from_srt':
        return _process_tts_from_srt(
            video_path, base_name, cache_dir, target_output_dir,
            config, batch_name, output_base_name=output_base_name
        )
    
    try:
        # STEP 1: 提取音频
        logger.info("\n[STEP 1/6] 提取音频...")
        audio_output_dir = os.path.join(cache_dir, "audio")
        audio_path = _get_extract_audio()(
            video_path,
            output_dir=audio_output_dir,
            sample_rate=config['audio']['sample_rate']
        )
        logger.success(f"音频提取完成: {audio_path}")
        
        # STEP 2: VAD 语音切片
        logger.info("\n[STEP 2/6] 语音活动检测 (VAD)...")
        vad_output_dir = os.path.join(cache_dir, "vad")
        segments = _get_run_vad()(
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
                'output_name': output_base_name,
                'message': '未检测到语音片段'
            }
        
        # STEP 3: ASR 识别
        logger.info("\n[STEP 3/6] 自动语音识别 (ASR)...")
        asr_output_dir = os.path.join(cache_dir, "asr")
        asr_results = _get_run_multi_asr()(
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
                'output_name': output_base_name,
                'message': 'ASR 识别结果为空'
            }
        
        logger.success(f"ASR 识别完成，共 {len(whisperx_segments)} 个片段")
        
        # STEP 4: LLM 修正
        fused_results = whisperx_segments
        if config.get('llm', {}).get('api_key') and not skip_llm_fix:
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
                
                fused_results = _get_fuse_asr_result()(
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
            if skip_llm_fix:
                logger.info("\n[STEP 4/6] 跳过 LLM 修正（已配置 SKIP_LLM_FIX=True）")
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
            
            translated_results = _get_translate_segments()(
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
            srt_output_path = os.path.join(target_output_dir, f"{output_base_name}.srt")
            
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
            processed = _get_process_tts_text_batch()(
                tts_texts,
                config=llm_config,
                prompt_template=prompt_template,
                batch_size=30
            )
            for idx, tts_text in enumerate(processed):
                translated_results[idx]["tts_text"] = tts_text
            logger.success(f"TTS 文本预处理完成")
            
            # 生成 TTS 音频
            logger.info("生成 TTS 配音...")
            tts_output_dir = os.path.join(cache_dir, "tts")
            import asyncio
            tts_audio = asyncio.run(_get_generate_tts()(
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
                config=merge_config,
                output_name=output_base_name
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
        logger.success(f"视频 [{output_base_name}] 处理完成！")
        logger.info(f"{'='*60}\n")
        
        return {
            'video_path': video_path,
            'status': 'success',
            'output_name': output_base_name,
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
            'output_name': output_base_name,
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


def _process_tts_from_srt(video_path, base_name, cache_dir, target_output_dir, config, batch_name, output_base_name=None):
    """
    tts_from_srt 模式：从已有中文字幕直接生成中文配音并合成视频。
    
    跳过 ASR 识别和翻译步骤，直接：SRT 解析 → TTS 文本预处理 → 合成配音 → 合并视频。
    
    Args:
        video_path: 视频文件路径
        base_name: 原始基础文件名（用于内部日志/缓存）
        cache_dir: 缓存目录
        target_output_dir: 输出目录
        config: 配置字典
        batch_name: 批次名称
        output_base_name: 翻译后的输出文件名（不含扩展名），若提供则用于输出文件命名
    
    Returns:
        dict: 处理结果
    """
    # 使用翻译后的文件名替换输出名
    use_name = output_base_name if output_base_name else base_name
    logger.info("模式: tts_from_srt（从已有字幕生成配音）")
    
    try:
        # STEP 1: 查找并解析 SRT 文件
        logger.info("\n[STEP 1/4] 查找中文字幕文件...")
        srt_path = _find_srt_file(video_path)
        if not srt_path:
            logger.error(f"未找到匹配的 SRT 字幕文件: {video_path}")
            return {
                'video_path': video_path,
                'status': 'failed',
                'output_name': use_name,
                'message': f'未找到匹配的 SRT 字幕文件（请确保字幕文件与视频同名，放在同一目录下）'
            }
        logger.info(f"  找到字幕: {srt_path}")
        
        segments = _parse_srt_to_segments(srt_path)
        if not segments:
            logger.error("SRT 文件解析为空")
            return {
                'video_path': video_path,
                'status': 'failed',
                'output_name': use_name,
                'message': 'SRT 文件解析为空'
            }
        logger.success(f"字幕解析完成，共 {len(segments)} 条")
        
        # STEP 2: TTS 文本预处理
        logger.info("\n[STEP 2/4] TTS 文本预处理...")
        
        project_root_dir = os.path.dirname(os.path.abspath(__file__))
        if batch_name:
            batches_dir = config.get('paths', {}).get('batches_dir', os.path.join(project_root_dir, "config", "batches"))
            prompts_dir = os.path.join(batches_dir, batch_name, "prompts")
        else:
            prompts_dir = config['paths'].get('prompts_dir', os.path.join(project_root_dir, 'config', 'prompts'))
        
        tts_prep_path = os.path.join(prompts_dir, 'tts_prep.txt')
        prompt_template = ""
        if os.path.exists(tts_prep_path):
            with open(tts_prep_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            logger.info(f"  加载 TTS 预处理模板: {os.path.basename(tts_prep_path)}")
        
        llm_config = dict(config.get('llm', {}))
        if batch_name:
            batches_dir = config.get('paths', {}).get('batches_dir', os.path.join(project_root_dir, "config", "batches"))
            batch_dir = os.path.join(batches_dir, batch_name)
            term_path = os.path.join(batch_dir, "terminology.json")
            if os.path.exists(term_path):
                llm_config['terminology_file'] = term_path
                logger.info(f"  注入批次术语表: {term_path}")
        
        tts_texts = [seg['text'] for seg in segments]
        processed = _get_process_tts_text_batch()(
            tts_texts,
            config=llm_config,
            prompt_template=prompt_template,
            batch_size=20
        )
        for idx, tts_text in enumerate(processed):
            segments[idx]["tts_text"] = tts_text
            segments[idx]["translated_text"] = tts_text  # 兼容 TTS 模块
        
        logger.success(f"TTS 文本预处理完成")
        
        # STEP 3: 生成 TTS 配音
        logger.info("\n[STEP 3/4] 生成 TTS 中文配音...")
        tts_output_dir = os.path.join(cache_dir, "tts")
        import asyncio
        tts_audio = asyncio.run(_get_generate_tts()(
            segments,
            output_dir=tts_output_dir,
            voice=config['tts']['voice'],
            config=config['tts']
        ))
        logger.success(f"TTS 配音生成完成: {tts_audio}")
        
        # STEP 4: 合并音频到视频
        logger.info("\n[STEP 4/4] 合成最终视频...")
        merge_config = config.get('video', {}).copy()
        final_video = merge_video(
            video_path,
            tts_audio,
            output_dir=target_output_dir,
            config=merge_config,
            output_name=use_name
        )
        logger.success(f"最终视频生成完成: {final_video}")
        
        # 生成 SRT（重写时间轴为 TTS 实际时长对齐的版本）
        logger.info("\n[额外步骤] 生成对齐后的 SRT 字幕...")
        srt_output_path = os.path.join(target_output_dir, f"{use_name}.srt")
        srt_path_out = generate_srt(segments, output_path=srt_output_path)
        logger.success(f"对齐版 SRT 生成完成: {srt_path_out}")
        
        logger.info(f"{'='*60}")
        logger.success(f"视频 [{use_name}] tts_from_srt 处理完成！")
        logger.info(f"{'='*60}\n")
        
        return {
            'video_path': video_path,
            'status': 'success',
            'output_name': use_name,
            'srt_path': srt_path_out,
            'tts_audio': tts_audio,
            'final_video': final_video,
            'message': '成功（从已有字幕生成配音）'
        }
        
    except Exception as e:
        logger.error(f"tts_from_srt 处理 {video_path} 时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'video_path': video_path,
            'status': 'failed',
            'output_name': use_name,
            'message': str(e)
        }
    finally:
        # 清理缓存
        try:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                logger.debug(f"已清理缓存目录: {cache_dir}")
        except Exception as e:
            logger.warning(f"清理缓存失败: {e}")


def process_single_video(args):
    """
    包装函数，用于多进程调用
    
    Args:
        args: (video_path, config, mode, input_dir, output_dir, batch_name, skip_llm_fix) 元组
    
    Returns:
        dict: 处理结果
    """
    video_path, config, mode, input_dir, output_dir, batch_name, skip_llm_fix = args
    return process_video_unified(video_path, config, mode, input_dir, output_dir, batch_name, skip_llm_fix=skip_llm_fix)


def main(input_path=None, output_dir=None, mode=None, batch_name=None, skip_llm_fix=False):
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
    # - tts_no_subtitle: 生成中文配音（全流程：ASR→翻译→TTS→合成）
    # - tts_from_srt: 从已有中文字幕生成配音（跳过ASR/翻译，直接SRT→TTS→合成）
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
        'tts_no_subtitle',
        'tts_from_srt'
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
        'tts_no_subtitle': '生成中文配音（全流程：ASR→翻译→TTS→合成）',
        'tts_from_srt': '从已有中文字幕生成配音（跳过ASR/翻译，直接SRT→TTS→合成）'
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
        escaped_input = glob.escape(final_input_path)
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(glob.glob(os.path.join(escaped_input, f"**/*{ext}"), recursive=True))
    
    if not video_files:
        logger.warning(f"在 {final_input_path} 中未找到视频文件")
        print(f"在 {final_input_path} 中未找到可处理的视频文件。")
        return
    
    print(f"共找到 {len(video_files)} 个视频文件待处理。\n")
    
    # ── 加载处理状态追踪器，显示历史处理情况 ──
    tracker = ProcessingTracker(final_output_dir, input_dir=final_input_path if os.path.isdir(final_input_path) else None)
    
    if len(video_files) > 1 or os.path.isdir(final_input_path):
        summary = tracker.get_summary(video_files)
        
        if summary["completed"]:
            print(f"📋 已处理成功: {len(summary['completed'])} 个（将跳过）")
        if summary["previously_failed"]:
            print(f"⚠️  之前失败需重试: {len(summary['previously_failed'])} 个")
        if summary["new"]:
            print(f"🆕 新增待处理: {len(summary['new'])} 个")
        
        if not summary["new"] and not summary["previously_failed"] and summary["completed"]:
            print(f"✅ 所有 {len(video_files)} 个文件均已处理完成！")
            print(f"   如需重新处理，请删除输出文件后重试。")
            return
        
        print()
    
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
    
    # 准备参数列表（包含 batch_name 和 skip_llm_fix）
    task_args = [(video_file, config, final_mode, final_input_path, final_output_dir, final_batch_name, skip_llm_fix) for video_file in video_files]
    
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
    
    # ── 更新处理状态追踪器 ──
    updated_count = 0
    for result in results:
        if result['status'] in ('success', 'failed'):
            tracker.update(
                result['video_path'],
                result,
                final_mode,
                output_name=result.get('output_name')
            )
            updated_count += 1
    if updated_count > 0:
        tracker.save()
        print(f"\n📝 处理状态已保存: {tracker.filepath}（{updated_count} 条记录）")
        print(f"   下次运行时可查看 {tracker.filepath} 了解处理进度。")


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 在这里定义输入输出路径和处理模式
    
    # 输入路径（可以是单个视频文件或包含视频的目录）
    INPUT_PATH = "/Volumes/mvp/[00]交易场/edgeskool"
    
    # 输出目录
    OUTPUT_DIR = "/Volumes/mvp/[00]交易场/edgeskool-中文"
    
    # 处理模式选择：
    # - subtitle_only: 仅生成中文字幕（ASR 识别）
    # - subtitle_bilingual: 生成中英双语字幕
    # - subtitle_chinese: 生成中文字幕（翻译后）
    # - tts_no_subtitle: 生成中文配音（全流程：ASR→翻译→TTS→合成）
    # - tts_from_srt: 从已有中文字幕生成配音（跳过ASR/翻译，直接SRT→TTS→合成）
    PROCESS_MODE = "tts_no_subtitle"
    
    # 批次名称（为空则使用全局配置）
    BATCH_NAME = ""  # 例如: "ICT_Trading_Batch1"
    
    # 跳过 Step 4 LLM 修正（True=跳过，直接使用 ASR 原文进入翻译）
    SKIP_LLM_FIX = False
    
    # ================================================
    
    # 解析命令行参数（可选覆盖配置）
    import argparse
    parser = argparse.ArgumentParser(description='统一视频处理工具')
    parser.add_argument('input_path', nargs='?', default=None, help='输入路径（视频文件或目录）')
    parser.add_argument('--mode', '-m', choices=[
        'subtitle_only',
        'subtitle_bilingual',
        'subtitle_chinese',
        'tts_no_subtitle',
        'tts_from_srt'
    ], default=None, help='处理模式')
    parser.add_argument('--output', '-o', default=None, help='输出目录')
    parser.add_argument('--batch', '-b', default=None, help='批次名称')
    
    args = parser.parse_args()
    
    # 调用 main 函数，传入配置
    main(
        input_path=args.input_path if args.input_path else INPUT_PATH,
        output_dir=args.output if args.output else OUTPUT_DIR,
        mode=args.mode if args.mode else PROCESS_MODE,
        batch_name=args.batch if args.batch else BATCH_NAME,
        skip_llm_fix=SKIP_LLM_FIX
    )
