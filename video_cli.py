#!/usr/bin/env python3
"""统一视频处理工具 —— 9 步断点续跑管道，三模式自适应。"""

from __future__ import annotations

import argparse, glob, json, multiprocessing, os, re, shutil, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from loguru import logger

# ═══════════════════════════════════════════════════════
#  环境补丁（AI 库导入前设置）
# ═══════════════════════════════════════════════════════
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
os.environ |= {
    "NLTK_DATA": str(ROOT / "nltk_data"),
    "HF_ENDPOINT": "https://hf-mirror.com",
    "TORCH_DISTRIBUTED_DEBUG": "OFF",
    "ONELOGGER_DISABLED": "1",
    "TORCHAUDIO_USE_TORCHCODEC": "0",
}
if (models := ROOT / "models").exists():
    os.environ |= {k: str(models / v) for k, v in [
        ("HF_HOME", "huggingface"), ("HUGGINGFACE_HUB_CACHE", "huggingface"),
        ("MODELSCOPE_CACHE", "funasr"), ("WHISPERX_CACHE", "whisperx"),
        ("TORCH_HOME", "torch"),
    ]}

from modules.utils.ffmpeg_utils import get_ffmpeg_exe
if (ff_dir := (Path(get_ffmpeg_exe()).parent if Path(get_ffmpeg_exe()).exists() else None)):
    os.environ["PATH"] = f"{ff_dir}{os.pathsep}{os.environ.get('PATH', '')}"

from modules.pipeline.processor import process_substep
from modules.utils.processing_tracker import (
    ALL_STEPS, STEP_LABELS, STEP_SUBSTEP, MODE_STEPS, USE_OUTPUT_VIDEO,
    ProcessingTracker,
)
from modules.utils.filename_translator import translate_filename

# ═══════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════
VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.wmv')
ICON = {'success': '✅', 'skipped': '⏭️', 'failed': '❌', 'done': '✅',
        'pending': '⏳', 'review_pending': '📝'}
SEP, WIDE = '─' * 60, '─' * 80
MODE_DESC = {
    'tts_no_subtitle': '自动配音（ASR→翻译→字幕→TTS→合成）',
    'tts_from_srt':    '从已有中文字幕生成配音（SRT→TTS→合成）',
    'tts_with_review': '带审核的配音（ASR→翻译→字幕→审核→TTS→合成）',
}
MODE_CHOICES = list(MODE_DESC)
DEFAULT_INPUT = str(ROOT / "input")
DEFAULT_OUTPUT = str(ROOT / "outputs")
DEFAULT_MODE = "tts_no_subtitle"

# ═══════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════

def load_config() -> dict:
    """加载 config.yaml，不存在则返回合理默认值。"""
    if (p := ROOT / "config" / "config.yaml").exists():
        return yaml.safe_load(p.read_text(encoding='utf-8'))
    return {
        'asr': {'device': 'cpu', 'model_size': 'base'},
        'llm': {'api_key': '', 'api_base': 'https://api.siliconflow.cn/v1',
                'model': 'deepseek-ai/DeepSeek-V3'},
        'tts': {'provider': 'edge', 'edge': {'voice': 'zh-CN-XiaoxiaoNeural'}},
        'video': {'burn_subtitles': False, 'audio_mode': 'tts_only'},
        'global': {'max_concurrency': {'video_processor': 2}},
    }


def target_dir(video_path: str, input_dir: str, output_dir: str) -> str:
    """保持输入目录结构映射到输出目录。"""
    try:
        return str(Path(output_dir) / Path(video_path).relative_to(input_dir).parent)
    except ValueError:
        return output_dir


def cache_path(video_path: str, output_dir: str, input_dir: str) -> str:
    """<output>/.cache/<stem>/"""
    return str(Path(target_dir(video_path, input_dir, output_dir))
               / ".cache" / Path(video_path).stem)


def collect_video_files(path: str) -> list[str] | None:
    """扫描路径，返回视频文件列表。"""
    if not os.path.exists(path):
        logger.error(f"路径不存在: {path}")
        return None
    if os.path.isfile(path):
        if path.lower().endswith(VIDEO_EXTS):
            return [path]
        logger.error(f"不支持的文件格式: {path}")
        return None
    dir_escaped = glob.escape(path)
    files = [f for ext in VIDEO_EXTS
             for f in glob.glob(os.path.join(dir_escaped, f"**/*{ext}"), recursive=True)]
    return files or None


# ═══════════════════════════════════════════════════════
#  多进程编排
# ═══════════════════════════════════════════════════════

def _worker(args: tuple) -> dict:
    vp, cfg, mode, inp, out, skip = args
    if mode.startswith('_'):
        return process_substep(vp, cfg, mode, inp, out, skip_llm_fix=skip)
    from modules.pipeline.processor import process_video_unified
    return process_video_unified(vp, cfg, mode, inp, out, skip_llm_fix=skip)


def batch_run(videos: list[str], config: dict, mode: str,
              input_dir: str, output_dir: str, skip_llm: bool,
              workers: int, label: str = "") -> tuple[list[dict], float]:
    """多进程执行一批视频。"""
    tasks = [(v, config, mode, input_dir, output_dir, skip_llm) for v in videos]
    if not tasks:
        return [], 0.0

    title = f"【{label}】" if label else ""
    print(f"🚀 {title}多进程 {workers} 路\n")
    t0, results, n, N = time.time(), [], 0, len(videos)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        fmap = {pool.submit(_worker, t): t[0] for t in tasks}
        for f in as_completed(fmap):
            vf = fmap[f]; n += 1
            try:
                r = f.result()
                results.append(r)
                s = r.get('status', 'failed')
                logger.info(f"  [{n}/{N}] ({n/N*100:.0f}%) {ICON.get(s, '❓')} {os.path.basename(vf)}")
            except Exception as e:
                logger.error(f"  异常 {os.path.basename(vf)}: {e}")
                results.append({'video_path': vf, 'status': 'failed',
                                'message': str(e), 'output_name': Path(vf).stem})
    return results, time.time() - t0


def step_stats(results: list[dict], total: int, elapsed: float, label: str = ""):
    """打印步骤统计并返回 {status: count}。"""
    cnt = {s: sum(1 for r in results if r['status'] == s)
           for s in ('success', 'skipped', 'failed')}
    tag = f"[{label}] " if label else ""
    print(f"\n{SEP}\n📊 {tag}{total}个 → ✅{cnt['success']} ⏭{cnt['skipped']} ❌{cnt['failed']}"
          f"  ⏱{elapsed:.0f}s\n{SEP}")
    if cnt['failed']:
        print("❌ 失败项:")
        for r in results:
            if r['status'] == 'failed':
                print(f"   - {os.path.basename(r['video_path'])}: {r.get('message', '?')}")
    return cnt


# ═══════════════════════════════════════════════════════
#  步骤调度
# ═══════════════════════════════════════════════════════

def _map_to_output_videos(originals: list[str], input_dir: str,
                          output_dir: str, config: dict) -> tuple[list[str], dict[str, str]]:
    """查找原始视频在输出目录中的副本，返回 (output_paths, {output→original} 映射)。"""
    videos, mapping = [], {}
    for vf in originals:
        ext = os.path.splitext(vf)[1]
        out_name = translate_filename(Path(vf).stem, config.get('llm', {}))
        p = os.path.join(target_dir(vf, input_dir, output_dir), f"{out_name}{ext}")
        if os.path.exists(p):
            videos.append(p)
            mapping[p] = vf
        else:
            logger.debug(f"未找到视频副本（将使用原始文件）: {p}")
    return videos, mapping


def _step_videos(video_files: list[str], step: str, mode: str,
                 input_dir: str, output_dir: str, config: dict,
                 tracker: ProcessingTracker) -> tuple[list[str] | None, str, dict[str, str]]:
    """筛选需要执行此步骤的文件。返回 (paths, input_dir, {path→original}) 或全 None。"""
    needed = MODE_STEPS.get(mode, ALL_STEPS)

    if step not in needed:
        for vf in video_files:
            if tracker.get_step_status(vf, step) not in ('done', 'skipped'):
                tracker.mark_step_skipped(vf, step, reason=f'{mode} 模式不需要此步骤')
        tracker.save()
        return None, "", {}

    work = [v for v in video_files
            if not tracker.is_step_synced(v, step, cache_path(v, output_dir, input_dir), output_dir)]
    if not work:
        return None, "", {}

    if step in USE_OUTPUT_VIDEO and mode != 'tts_from_srt':
        copies, mapping = _map_to_output_videos(work, input_dir, output_dir, config)
        if copies:
            # 始终用原始英文路径 → cache_dir 统一为英文名，TTS / Merge 一致
            orig_paths = [mapping[c] for c in copies]
            return (orig_paths, output_dir, mapping)
        return (work, input_dir, {v: v for v in work})

    return work, input_dir, {v: v for v in work}


def _print_progress(video_files: list[str], tracker: ProcessingTracker, mode: str):
    """打印步骤完成状态矩阵。"""
    needed = MODE_STEPS.get(mode, ALL_STEPS)
    if len(video_files) <= 1:
        return

    header = "文件" + "".join(f" │ {s[:3]}" for s in needed)
    print(f"\n{WIDE}\n📊 进度矩阵 ({len(video_files)} 文件)\n  {header}")

    for vf in video_files:
        steps = tracker.data.get("files", {}).get(tracker._relpath(vf), {}).get("steps", {})
        row = Path(vf).stem[:25]
        for s in needed:
            row += f" │ {ICON.get(steps.get(s, {}).get('status', 'pending'), '⬜')}"
        print(f"  {row}")
    print(f"{WIDE}\n")


def _consolidate_caches(output_dir: str):
    """合并 .cache 下分裂的中英文缓存文件夹（旧版本残留）。

    核心原则：缓存目录始终以原始英文文件名命名。
    检测到同一编号前缀存在中英文两套文件夹时，将中文文件夹中的
    文件合并到英文文件夹，然后删除中文文件夹。
    """
    cache_root = os.path.join(output_dir, ".cache")
    if not os.path.isdir(cache_root):
        return

    entries = [d for d in os.listdir(cache_root)
               if os.path.isdir(os.path.join(cache_root, d))]
    if len(entries) < 2:
        return

    # 提取编号前缀（如 "01 - "）并分组
    import unicodedata
    prefix_groups = {}
    for name in entries:
        m = re.match(r'^(\d+\s*[-._]\s*)', name)
        if not m:
            continue
        prefix = m.group(0)
        prefix_groups.setdefault(prefix, []).append(name)

    merged_count = 0
    for prefix, names in prefix_groups.items():
        if len(names) < 2:
            continue
        # 分为英文组（ASCII 为主）和中文组（含汉字）
        en_dirs, zh_dirs = [], []
        for n in names:
            has_cjk = any('CJK' in unicodedata.name(c, '') for c in n)
            zh_dirs.append(n) if has_cjk else en_dirs.append(n)

        if not zh_dirs or not en_dirs:
            continue

        # 将每个中文文件夹合并到对应的英文文件夹
        for zh_dir in zh_dirs:
            # 选择最匹配的英文文件夹（同前缀下）
            target = en_dirs[0]  # 简单策略：取第一个英文目录
            src_path = os.path.join(cache_root, zh_dir)
            dst_path = os.path.join(cache_root, target)

            for root, dirs, files in os.walk(src_path):
                # 跳过 macOS Apple Double 元数据文件（._ 前缀）
                dirs[:] = [d for d in dirs if not d.startswith('._')]
                for fname in files:
                    if fname.startswith('._'):
                        continue
                    rel = os.path.relpath(root, src_path)
                    dst_dir = os.path.join(dst_path, rel) if rel != '.' else dst_path
                    os.makedirs(dst_dir, exist_ok=True)
                    src_file = os.path.join(root, fname)
                    dst_file = os.path.join(dst_dir, fname)
                    if not os.path.exists(dst_file):
                        shutil.move(src_file, dst_file)
                        merged_count += 1
                    else:
                        # 目标已存在 → 比较大小，保留更大的
                        if os.path.getsize(src_file) > os.path.getsize(dst_file):
                            shutil.move(src_file, dst_file)
                            merged_count += 1
                        else:
                            os.remove(src_file)

            # 删除已清空的中文文件夹（递增尝试：rmtree → rmdir）
            for _fn_remove in (lambda: shutil.rmtree(src_path),
                              lambda: os.rmdir(src_path)):
                try:
                    _fn_remove()
                except Exception:
                    pass
            if os.path.isdir(src_path):
                logger.warning(f"   ⚠️  无法删除空目录: {src_path}")
            else:
                logger.info(f"🔗 已合并缓存: \"{zh_dir}\" → \"{target}\"")

    if merged_count:
        logger.info(f"✅ 共合并 {merged_count} 个缓存文件到统一目录")


def _review_prompt(video_files: list[str], input_dir: str,
                   output_dir: str, config: dict, tracker: ProcessingTracker):
    """tts_with_review：srt 完成后复制视频到输出目录并打印审核提示。"""
    import shutil
    llm_cfg = config.get('llm', {})
    print(f"\n{SEP}\n📝 SRT 字幕已生成，请人工审核：")
    for vf in video_files:
        out_name = tracker.get_output_name(vf) or translate_filename(Path(vf).stem, llm_cfg)
        tgt_dir = target_dir(vf, input_dir, output_dir)
        srt = os.path.join(tgt_dir, f"{out_name}.zh.srt")
        if os.path.exists(srt):
            print(f"   📄 {srt}")
        # 复制视频到输出目录（审核模式下视频和 SRT 放一起方便人工核对）
        original_ext = os.path.splitext(vf)[1]
        copied_video = os.path.join(tgt_dir, f"{out_name}{original_ext}")
        if not os.path.exists(copied_video):
            shutil.copy2(vf, copied_video)
            logger.info(f"📁 视频已复制到输出目录: {copied_video}")
    print(f"\n   审核完成后重新运行脚本，自动继续 TTS 配音合成。\n{SEP}\n")
    logger.info("⏸️  等待人工审核")


# ═══════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════

def main(input_path: str = DEFAULT_INPUT, output_dir: str = DEFAULT_OUTPUT,
         mode: str = DEFAULT_MODE, skip_llm_fix: bool = False):
    """9 步顺序管道：每步全量完成后推进，双校验（tracker+磁盘）自动断点续跑。"""

    # ── 日志 ──
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logs_dir = ROOT / "logs"; logs_dir.mkdir(exist_ok=True)
    logger.add(logs_dir / "video_processor_{time:YYYY-MM-DD}.log",
               rotation="00:00", retention="7 days", level="DEBUG")

    print(f"\n{SEP}\n🎬 统一视频处理工具\n{SEP}\n"
          f"📂 输入: {input_path}\n📤 输出: {output_dir}\n"
          f"🔧 模式: {mode} — {MODE_DESC.get(mode, mode)}\n")

    # ── 扫描视频 ──
    video_files = collect_video_files(input_path)
    if not video_files:
        logger.warning(f"未找到可处理的视频文件")
        return
    print(f"共找到 {len(video_files)} 个视频文件\n")

    # ── 追踪器 ──
    tracker = ProcessingTracker(output_dir,
                                input_dir=input_path if os.path.isdir(input_path) else None)
    tracker.sync_from_disk(video_files, mode)
    tracker.init_files(video_files, mode)
    tracker.save()

    # ── 合并旧版分裂缓存（中文文件夹 → 英文文件夹）──
    _consolidate_caches(output_dir)

    # ── 状态检查 ──
    s = tracker.get_summary(video_files)
    for key, label in [("completed", "📋 全部完成"), ("in_progress", "🔄 进行中"),
                       ("pending", "⏳ 待处理"), ("new", "🆕 新增")]:
        if s[key]:
            print(f"{label}: {len(s[key])}")
    if s["previously_failed"]:
        info = ", ".join(f"{k}({v})" for k, v in s["previously_failed"])
        print(f"⚠️  失败需重试: {info}")

    has_pending = s["new"] or s["pending"] or s["in_progress"] or s["previously_failed"]
    if s["completed"] and len(s["completed"]) == len(video_files) and not has_pending:
        print(f"✅ {len(video_files)} 个文件全部完成")
        return

    _print_progress(video_files, tracker, mode)

    # ── 配置 ──
    config = load_config()
    workers = config.get('global', {}).get('max_concurrency', {}).get('video_processor', 2)
    logger.info(f"mode={mode} workers={workers} "
                f"device={config.get('asr', {}).get('device', 'cpu')}")
    if sys.platform == 'darwin':
        multiprocessing.set_start_method('spawn', force=True)

    # ═══════════════════════════════════════════════════
    #  9 步骤管道
    # ═══════════════════════════════════════════════════
    total_t, review = 0.0, False
    needed = MODE_STEPS.get(mode, ALL_STEPS)

    for idx, step in enumerate(ALL_STEPS, 1):
        step_vids, inp_dir, v2orig = _step_videos(
            video_files, step, mode, input_path, output_dir, config, tracker)

        if step_vids is None:
            print(f"[{idx}/9] {STEP_LABELS[step]}: 全部跳过 ({len(video_files)})")
            continue

        print(f"\n{SEP}\n[{idx}/9] {STEP_LABELS[step]} — {len(step_vids)}/{len(video_files)} 文件\n{SEP}")
        results, dt = batch_run(step_vids, config, STEP_SUBSTEP[step],
                                inp_dir, output_dir, skip_llm_fix, workers, STEP_LABELS[step])
        total_t += dt
        cnt = step_stats(results, len(step_vids), dt, STEP_LABELS[step])

        # 更新 tracker
        for r in results:
            orig = v2orig.get(r['video_path'], r['video_path'])
            match r['status']:
                case 'success':  tracker.mark_step_done(orig, step, output_name=r.get('output_name'),
                                                        duration_seconds=r.get('elapsed'))
                case 'failed':   tracker.mark_step_failed(orig, step, message=r.get('message', ''))
                case 'skipped':  tracker.mark_step_skipped(orig, step, reason=r.get('message', ''))
        tracker.save()

        # 审核拦截：tts_with_review 在 srt 完成后暂停
        if mode == 'tts_with_review' and step == 'srt' and cnt['success'] > 0:
            _review_prompt(video_files, input_path, output_dir, config, tracker)
            review = True

    # ═══════════════════════════════════════════════════
    #  最终统计
    # ═══════════════════════════════════════════════════
    _print_progress(video_files, tracker, mode)

    done = sum(1 for vf in video_files
               if all(tracker.get_step_status(vf, s) == 'done' for s in needed))
    failed = sum(1 for vf in video_files
                 if any(tracker.get_step_status(vf, s) == 'failed' for s in ALL_STEPS))

    print(f"\n{SEP}\n📊 最终统计: ✅{done} ❌{failed}"
          f"  ⏱{total_t:.0f}s ({total_t/60:.0f}min)\n{SEP}")
    if review:
        print("📝 请审核字幕后重新运行，自动继续后续步骤。")

    tracker.save()
    print(f"\n📝 状态: {tracker.filepath}")


# ═══════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── 预设默认值（团队共享可直接改这里）──
    #'tts_no_subtitle': '自动配音（ASR→翻译→字幕→TTS→合成）',
    #'tts_from_srt':    '从已有中文字幕生成配音（SRT→TTS→合成）',
    #'tts_with_review': '带审核的配音（ASR→翻译→字幕→审核→TTS→合成）',
    PRESET = {
        "input": "/Volumes/mvp/[00]交易场/Mulham Trading/Full Trading Courses",
        "output": "/Volumes/mvp/[00]交易场/Mulham Trading/Full Trading Courses/output",
        "mode": "tts_no_subtitle "
    }

    parser = argparse.ArgumentParser(description='统一视频处理工具 — 9 步管道')
    parser.add_argument('input', nargs='?', help='输入路径（文件/目录）')
    parser.add_argument('-m', '--mode', choices=MODE_CHOICES, help='处理模式')
    parser.add_argument('-o', '--output', help='输出目录')
    parser.add_argument('--skip-llm-fix', action='store_true', help='跳过 LLM 修正')
    args = parser.parse_args()

    main(input_path=args.input or PRESET["input"],
         output_dir=args.output or PRESET["output"],
         mode=args.mode or PRESET["mode"],
         skip_llm_fix=args.skip_llm_fix)
