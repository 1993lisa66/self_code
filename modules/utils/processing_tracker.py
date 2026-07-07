#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持久化处理状态追踪器（步骤级追踪 v3）。

9 个处理步骤：
  1. extract_audio   → 提取音频
  2. vad             → VAD 语音检测
  3. asr             → ASR 语音识别
  4. llm_fix         → LLM 修正字幕文本
  5. translation     → 翻译（→ 中文）
  6. srt             → 生成 SRT 字幕
  7. tts_prep        → SRT 解析 + TTS 文本预处理
  8. tts_generate    → TTS 语音合成
  9. merge           → 合并视频 + 生成对齐 SRT

每步骤双重校验：tracker 记录 + 磁盘缓存文件 → 缺一即重做。
"""

import os
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

TRACKER_FILENAME = "processing_state.json"

# ── 步骤定义 ──
ALL_STEPS = [
    'extract_audio', 'vad', 'asr', 'llm_fix', 'translation', 'srt',
    'tts_prep', 'tts_generate', 'merge',
]

STEP_LABELS = {
    'extract_audio': '提取音频',
    'vad': 'VAD语音检测',
    'asr': 'ASR语音识别',
    'llm_fix': 'LLM修正',
    'translation': '翻译',
    'srt': '生成SRT字幕',
    'tts_prep': 'SRT解析+TTS预处理',
    'tts_generate': 'TTS语音合成',
    'merge': '合并视频+对齐SRT',
}

# 各步骤对应的子步骤标识（传给 process_substep）
STEP_SUBSTEP = {
    'extract_audio': '_extract_audio',
    'vad': '_vad',
    'asr': '_asr',
    'llm_fix': '_llm_fix',
    'translation': '_translation',
    'srt': '_srt',
    'tts_prep': '_tts_prep',
    'tts_generate': '_tts_generate',
    'merge': '_merge',
}

# 各模式需要的步骤
MODE_STEPS = {
    'tts_no_subtitle': ALL_STEPS,
    'tts_with_review': ALL_STEPS,
    'tts_from_srt': ['tts_prep', 'tts_generate', 'merge'],
}

# 步骤 7-8 使用输出目录中的视频（含字幕文件）；merge 用原始路径（从缓存读 TTS）
USE_OUTPUT_VIDEO = {'tts_prep', 'tts_generate'}

# ── 缓存/输出文件校验 ──
def _step_cache_files(step_name, cache_dir, base_name, output_dir=None):
    """返回该步骤完成后必须存在的文件列表（相对路径转绝对路径）。"""
    files = []
    if step_name == 'extract_audio':
        files.append(os.path.join(cache_dir, 'audio', f'{base_name}.wav'))
    elif step_name == 'vad':
        files.append(os.path.join(cache_dir, 'vad', f'{base_name}_segments.json'))
    elif step_name == 'asr':
        files.append(os.path.join(cache_dir, 'asr', f'{base_name}_multi_asr.json'))
    elif step_name == 'llm_fix':
        files.append(os.path.join(cache_dir, 'fused_results.json'))
    elif step_name == 'translation':
        files.append(os.path.join(cache_dir, 'translated_results.json'))
    elif step_name == 'srt':
        if output_dir:
            files.append(os.path.join(output_dir, f'{base_name}.zh.srt'))
            files.append(os.path.join(output_dir, f'{base_name}.srt'))
        else:
            files.append(os.path.join(cache_dir, f'{base_name}.zh.srt'))
    elif step_name == 'tts_prep':
        files.append(os.path.join(cache_dir, 'tts_preprocessed.json'))
    elif step_name == 'tts_generate':
        files.append(os.path.join(cache_dir, 'tts', 'full_tts.mp3'))
    elif step_name == 'merge':
        if output_dir:
            files.append(os.path.join(output_dir, f'{base_name}.mp4'))
    return files


class ProcessingTracker:
    """步骤级追踪器。每个文件记录 9 步状态，双重校验（记录 + 磁盘文件）。"""

    def __init__(self, output_dir, input_dir=None):
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.filepath = os.path.join(output_dir, TRACKER_FILENAME)
        self.data = self._load()
        self._migrate_old_formats()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": 3, "files": {}}

    def _migrate_old_formats(self):
        """将 v1/v2 格式自动升级到 v3（步骤级 tracking）。"""
        changed = False
        for key, entry in self.data.get("files", {}).items():
            if "steps" in entry:
                continue  # 已是 v3 格式

            # v2 格式：有 phases 字段
            phases = entry.get("phases", {})
            mode = entry.get("mode", "")
            steps = {}
            p1_steps = ['extract_audio', 'vad', 'asr', 'llm_fix', 'translation', 'srt']
            p2_steps = ['tts_prep', 'tts_generate', 'merge']

            if phases.get("phase1"):
                for s in p1_steps:
                    steps[s] = {"status": "done", "at": entry.get("processed_at", "")}
            else:
                for s in p1_steps:
                    # 检查旧格式：未完成但可能失败
                    old_status = entry.get("status", "")
                    if old_status == "failed":
                        steps[s] = {"status": "failed"}
                    else:
                        steps[s] = {"status": "pending"}

            if phases.get("phase2"):
                for s in p2_steps:
                    steps[s] = {"status": "done", "at": entry.get("processed_at", "")}
            else:
                for s in p2_steps:
                    steps[s] = {"status": "pending"}

            entry["steps"] = steps
            # 保留旧字段用于兼容
            changed = True
            logger.debug(f"已迁移旧格式: {key}")

        self.data["version"] = 3
        if changed:
            logger.info("🔄 已自动升级旧格式追踪文件到 v3（步骤级追踪）")

    def save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        if "mode" in self.data:
            del self.data["mode"]  # 清理顶层旧字段
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _relpath(self, video_path):
        if self.input_dir and os.path.isdir(self.input_dir):
            try:
                return os.path.relpath(video_path, self.input_dir)
            except ValueError:
                pass
        return os.path.basename(video_path)

    def _get_entry(self, video_path):
        rel = self._relpath(video_path)
        if rel not in self.data.get("files", {}):
            self.data["files"][rel] = {
                "output_name": Path(video_path).stem,
                "steps": {},
            }
        return self.data["files"][rel]

    # ── 步骤状态操作 ──

    def _ensure_steps_init(self, video_path, mode):
        """确保文件的所有步骤已初始化。"""
        entry = self._get_entry(video_path)
        needed = MODE_STEPS.get(mode, ALL_STEPS)
        for s in ALL_STEPS:
            if s not in entry.get("steps", {}):
                status = "skipped" if s not in needed else "pending"
                entry["steps"][s] = {
                    "status": status,
                    "reason": f"模式 {mode} 不需要此步骤" if status == "skipped" else ""
                }
        if entry.get("mode") != mode:
            entry["mode"] = mode

    def init_files(self, video_files, mode):
        """初始化所有文件的步骤状态。"""
        for vf in video_files:
            rel = self._relpath(vf)
            if rel not in self.data.get("files", {}):
                self.data["files"][rel] = {
                    "output_name": Path(vf).stem,
                    "steps": {},
                }
            self._ensure_steps_init(vf, mode)

    def mark_step_done(self, video_path, step_name, output_name=None, duration_seconds=None):
        """标记步骤完成。

        Args:
            duration_seconds: 该步骤执行耗时（秒），可选。传入后自动更新文件总耗时。
        """
        entry = self._get_entry(video_path)
        if output_name:
            entry["output_name"] = output_name
        step_record = {
            "status": "done",
            "at": datetime.now().isoformat(),
        }
        if duration_seconds is not None:
            step_record["duration_seconds"] = duration_seconds
        entry["steps"][step_name] = step_record
        self._update_total_duration(entry)

    def mark_step_failed(self, video_path, step_name, message=""):
        """标记步骤失败。"""
        entry = self._get_entry(video_path)
        entry["steps"][step_name] = {
            "status": "failed",
            "at": datetime.now().isoformat(),
            "message": message,
        }

    def mark_step_skipped(self, video_path, step_name, reason=""):
        """标记步骤跳过（模式不需要或已通过缓存跳过）。"""
        entry = self._get_entry(video_path)
        entry["steps"][step_name] = {
            "status": "skipped",
            "reason": reason,
        }

    def _update_total_duration(self, entry):
        """汇总所有已完成步骤的耗时，写入 entry['total_duration_seconds']。"""
        total = 0.0
        for step_info in entry.get("steps", {}).values():
            dur = step_info.get("duration_seconds")
            if isinstance(dur, (int, float)):
                total += dur
        entry["total_duration_seconds"] = round(total, 2)

    def get_step_status(self, video_path, step_name):
        """获取步骤状态字符串。"""
        entry = self._get_entry(video_path)
        return entry.get("steps", {}).get(step_name, {}).get("status", "pending")

    # ── 双重校验：tracker 记录 + 磁盘缓存文件 ──

    def is_step_synced(self, video_path, step_name, cache_dir, output_dir=None):
        """检查步骤是否同步（tracker 记录 done + 磁盘缓存都存在）。
        
        Returns True 表示不用重复执行。
        """
        status = self.get_step_status(video_path, step_name)
        if status != "done":
            return False

        # 检查磁盘文件
        # 缓存文件（步骤1-5,7-8）用原始文件名，输出文件（步骤6,9）用翻译后文件名
        entry = self._get_entry(video_path)
        if step_name in ('srt', 'merge'):
            base_name = entry.get("output_name", Path(video_path).stem)
        else:
            base_name = Path(video_path).stem
        cache_files = _step_cache_files(step_name, cache_dir, base_name, output_dir)

        if not cache_files:
            return True  # 没有缓存文件的步骤，只靠 tracker

        all_exist = all(os.path.exists(f) for f in cache_files)
        if not all_exist:
            logger.debug(f"步骤 {step_name} 缓存文件不完整（{base_name}），将重新执行")
        return all_exist

    def sync_from_disk(self, video_files, mode):
        """冷启动：扫描磁盘文件，推断各步骤完成状态。
        
        缓存文件始终按原始文件名组织在 .cache/<原始stem>/ 下，
        输出文件（srt/mp4）才使用翻译后的 output_name。
        """
        from .filename_translator import translate_filename

        for vf in video_files:
            rel = self._relpath(vf)
            if rel in self.data.get("files", {}):
                continue

            entry = self._get_entry(vf)
            base_name = Path(vf).stem
            cache_dir = os.path.join(self.output_dir, ".cache", base_name)

            # 冷启动：检查磁盘缓存文件是否存在（始终使用原始文件名）
            for step_name in ALL_STEPS:
                if self.get_step_status(vf, step_name) == "done":
                    continue
                cache_files = _step_cache_files(step_name, cache_dir, base_name, self.output_dir)
                if cache_files and all(os.path.exists(f) for f in cache_files):
                    entry["steps"][step_name] = {"status": "done", "at": datetime.now().isoformat()}
                    logger.debug(f"🔄 冷同步 step={step_name} file={base_name}")

            # 检查输出文件（srt/merge）是否存在，使用翻译后的文件名
            output_base_name = translate_filename(base_name, {})
            if output_base_name != base_name:
                if entry.get("output_name") == base_name:
                    entry["output_name"] = output_base_name
                # 只检查输出步骤（srt, merge），它们使用翻译后的文件名
                for step_name in ('srt', 'merge'):
                    if self.get_step_status(vf, step_name) == "done":
                        continue
                    cache_files = _step_cache_files(step_name, cache_dir, output_base_name, self.output_dir)
                    if cache_files and all(os.path.exists(f) for f in cache_files):
                        entry["steps"][step_name] = {"status": "done", "at": datetime.now().isoformat()}
                        logger.debug(f"🔄 冷同步 step={step_name} file={output_base_name} (translated)")

        self.save()

    # ── 摘要查询 ──

    def get_summary(self, video_files):
        """返回当前所有文件的全步骤状态摘要。"""
        tracked = set(self.data.get("files", {}).keys())
        current = {self._relpath(vf) for vf in video_files}

        fully_done = set()       # 所有 9 步 done
        in_progress = set()      # 部分步骤 done
        pending = set()          # 全部未开始
        previously_failed = set()

        for k in tracked & current:
            entry = self.data["files"][k]
            steps = entry.get("steps", {})
            mode = entry.get("mode", "")
            needed = MODE_STEPS.get(mode, ALL_STEPS)

            all_done = all(steps.get(s, {}).get("status") == "done" for s in needed)
            any_done = any(steps.get(s, {}).get("status") == "done" for s in needed)
            any_failed = any(v.get("status") == "failed" for v in steps.values())

            if any_failed:
                # 找到第一个失败的步骤
                failed_step = None
                for s in ALL_STEPS:
                    if steps.get(s, {}).get("status") == "failed":
                        failed_step = s
                        break
                previously_failed.add((k, failed_step))
            elif all_done:
                fully_done.add(k)
            elif any_done:
                in_progress.add(k)
            else:
                pending.add(k)

        new_files = current - tracked

        return {
            "total": len(video_files),
            "new": sorted(new_files),
            "completed": sorted(fully_done),
            "in_progress": sorted(in_progress),
            "previously_failed": [(k, s) for k, s in sorted(previously_failed)],
            "pending": sorted(pending),
        }

    def get_progress_summary(self, video_files, mode):
        """返回进度条式摘要，显示每个文件的各步骤完成情况。"""
        summary = self.get_summary(video_files)

        if len(video_files) <= 1:
            return summary

        needed = MODE_STEPS.get(mode, ALL_STEPS)
        step_counts = {}  # step_name → done/failed/pending 计数
        for k in {self._relpath(vf) for vf in video_files} & set(self.data.get("files", {}).keys()):
            entry = self.data["files"][k]
            for s in needed:
                if s not in step_counts:
                    step_counts[s] = {"done": 0, "failed": 0, "pending": 0}
                status = entry.get("steps", {}).get(s, {}).get("status", "pending")
                step_counts[s][status] = step_counts[s].get(status, 0) + 1

        logger.debug(f"步骤完成度: {json.dumps(step_counts, ensure_ascii=False)}")
        return summary

    def get_output_name(self, video_path):
        """获取输出名称（可能已翻译）。"""
        entry = self._get_entry(video_path)
        return entry.get("output_name", Path(video_path).stem)

    def set_output_name(self, video_path, output_name):
        """设置输出名称。"""
        entry = self._get_entry(video_path)
        entry["output_name"] = output_name
