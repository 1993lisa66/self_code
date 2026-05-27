import os
import sys

# 必须在导入任何 AI 库之前设置环境变量和应用补丁
os.environ["NLTK_DATA"] = os.path.join(os.getcwd(), "nltk_data")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 抑制 PyTorch Distributed 和 OneLogger 警告
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["ONELOGGER_DISABLED"] = "1"

# 禁用 torchcodec，强制 torchaudio 使用 soundfile 后端
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

import glob
import nltk
from multiprocessing import Pool, cpu_count
from functools import partial
import time

from modules.utils.patch_torch import apply_torch_patch
apply_torch_patch()

try:
    # 设置本地 nltk_data 目录以避开权限和路径问题
    nltk_data_dir = os.environ["NLTK_DATA"]
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path = [nltk_data_dir]
    
    # 尝试下载
    # nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    # nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
except Exception as e:
    pass

from pipeline import VideoTranslationPipeline
from loguru import logger

def clear_directory(directory_path):
    """清空指定目录下的所有内容"""
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"清理 {item_path} 时出错: {e}")
        print(f"[OK] 已清空目录: {directory_path}")
    else:
        os.makedirs(directory_path, exist_ok=True)
        print(f"[OK] 已创建目录: {directory_path}")

def process_single_video(video_file, input_path, config_dict):
    """处理单个视频的函数（用于多进程）
    
    Args:
        video_file: 视频文件路径
        input_path: 输入目录路径
        config_dict: 配置字典（从主进程传递，避免重复加载）
    """
    from pipeline import VideoTranslationPipeline
    
    filename = os.path.basename(video_file)
    
    # 计算相对路径
    if os.path.isdir(input_path):
        rel_path = os.path.relpath(video_file, input_path)
        sub_dir = os.path.dirname(rel_path)
    else:
        sub_dir = ""

    print(f"\n>>> 正在处理 [{filename}] ...")
    try:
        # 使用传入的配置创建 Pipeline 实例
        # 注意：每个进程仍需创建自己的 Pipeline，因为模型不能跨进程共享
        # 但 Pipeline 内部使用了单例模式缓存模型，同一进程处理多个视频时会复用
        pipeline = VideoTranslationPipeline(config_dict=config_dict)
        # 运行 Pipeline，传入 sub_dir 以保持目录结构
        pipeline.run(video_file, sub_dir=sub_dir)
        print(f"✅ DONE: [{filename}] 处理完成！")
        return {"file": filename, "status": "success"}
    except Exception as e:
        logger.error(f"处理视频 {filename} 时发生异常: {e}")
        print(f"❌ FAILED: [{filename}] 处理失败，请查看日志。")
        return {"file": filename, "status": "failed", "error": str(e)}

def main():
    # 清空 cache 和 outputs 目录
    cache_dir = os.path.join(project_root, "cache")
    outputs_dir = os.path.join(project_root, "outputs")
    logs_dir = os.path.join(project_root, "logs")

    print("\n>>> 清理缓存和输出目录...")
    clear_directory(cache_dir)
    clear_directory(outputs_dir)
    clear_directory(logs_dir)
    print()

    # 默认视频扩展名
    VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.flv')

    # 0. 显示配置信息（只执行一次）
    print("="*60)
    print("📋 配置信息检查")
    print("="*60)
    
    # 创建 Pipeline 实例以加载配置（只加载一次）
    pipeline = VideoTranslationPipeline()
    config = pipeline.config
    
    input_dir = config['paths'].get('input_dir', 'input')
    output_dir = config['paths'].get('output_dir', 'outputs')
    cache_dir = config['paths'].get('cache_dir', 'cache')
    logs_dir = config['paths'].get('log_dir', 'logs')

    print(f"📂 输入目录: {input_dir}")
    print(f"📤 输出目录: {output_dir}")
    print(f"💾 缓存目录: {cache_dir}")
    print(f"💾 日志目录: {logs_dir}")

    # 检查目录是否存在（输入目录必须存在，输出和缓存目录会自动创建）
    if not os.path.exists(input_dir):
        print(f"\n⚠️  警告: 输入目录不存在: {input_dir}")
        print(f"   程序将尝试自动创建该目录...")
    else:
        print(f"✅ 输入目录已存在")
    
    print("="*60 + "\n")

    # 2. 从配置读取输入路径，如果没有则默认使用 "input"
    input_path = config['paths'].get('input_dir', 'input')
    
    # 3. 如果有命令行参数，则覆盖配置中的路径
    if len(sys.argv) > 1:
        input_path = sys.argv[1]

    # 4. 验证输入路径
    if not os.path.exists(input_path):
        logger.error(f"错误: 找不到路径 {input_path}")
        print(f"\n❌ 错误: 找不到路径 {input_path}")
        
        # 尝试创建目录（如果是默认路径或配置的路径）
        if input_path in [config['paths'].get('input_dir'), 'input']:
            try:
                os.makedirs(input_path, exist_ok=True)
                print(f"✅ 已自动创建目录: {input_path}")
                print(f"请将视频放入该目录后重试。\n")
            except Exception as e:
                print(f"无法创建目录: {e}\n")
        else:
            print(f"请检查配置文件 config.yaml 中的 paths.input_dir 设置是否正确。\n")
            print(f"当前配置: {config['paths'].get('input_dir')}")
        return

    # 收集待处理的视频文件
    video_files = []
    if os.path.isfile(input_path):
        if input_path.lower().endswith(VIDEO_EXTENSIONS):
            video_files.append(input_path)
        else:
            print(f"错误: 不支持的文件格式 {input_path}")
            return
    elif os.path.isdir(input_path):
        for ext in VIDEO_EXTENSIONS:
            # 使用递归搜索
            video_files.extend(glob.glob(os.path.join(input_path, f"**/*{ext}"), recursive=True))
    
    if not video_files:
        print(f"在 {input_path} 中未找到可处理的视频文件。")
        return

    print(f"共找到 {len(video_files)} 个视频文件待处理。")

    # 从配置文件读取并行处理参数（支持环境变量覆盖）
    video_config = config.get('global', {}).get('video_processing', {})
    max_workers = int(os.environ.get(
        'MAX_CONCURRENT_VIDEOS', 
        str(video_config.get('max_parallel_videos', 3))
    ))
    max_workers = min(max_workers, len(video_files), cpu_count())  # 不超过CPU核心数和文件数
    
    print(f"\n🚀 启动多进程处理，并行数: {max_workers}")
    print(f"💡 提示: 可在 config.yaml 中配置 global.video_processing.max_parallel_videos")
    
    # 配置 API 速率限制（优先从配置文件读取，支持环境变量覆盖）
    api_limits = video_config.get('api_rate_limits', {})
    llm_rate = int(os.environ.get(
        'LLM_API_RATE', 
        str(api_limits.get('llm_per_minute', 15))
    ))
    tts_rate = int(os.environ.get(
        'TTS_API_RATE', 
        str(api_limits.get('tts_per_minute', 30))
    ))
    
    from modules.utils.rate_limiter import set_llm_rate_limit, set_tts_rate_limit
    set_llm_rate_limit(max_calls=llm_rate, time_window=60.0)
    set_tts_rate_limit(max_calls=tts_rate, time_window=60.0)
    
    # 使用进程池并行处理
    start_time = time.time()
    
    # 创建偏函数，传递配置字典（避免子进程重复加载配置文件）
    process_func = partial(process_single_video, input_path=input_path, config_dict=config)
    
    with Pool(processes=max_workers) as pool:
        results = pool.map(process_func, video_files)
    
    elapsed_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\n{'='*60}")
    print(f"📊 处理完成统计:")
    print(f"   - 总计: {len(video_files)} 个视频")
    print(f"   - 成功: {success_count} 个")
    print(f"   - 失败: {failed_count} 个")
    print(f"   - 总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print(f"{'='*60}")
    
    # 显示失败的文件
    if failed_count > 0:
        print(f"\n❌ 失败的文件:")
        for r in results:
            if r['status'] == 'failed':
                print(f"   - {r['file']}: {r.get('error', '未知错误')}")

if __name__ == "__main__":
    main()
