import os
import json
from loguru import logger
from faster_whisper import WhisperModel

def run_vad(audio_path, output_dir="cache/vad", device="cuda", model_size="large-v3"):
    """
    使用 Faster-Whisper 的 VAD 功能进行语音切片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_segments.json")

    # 缓存检查
    if os.path.exists(output_path):
        logger.info(f"使用已存在的 VAD 缓存: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    logger.info(f"正在进行 VAD 切片: {audio_path}")

    try:
        # 根据设备选择计算类型
        compute_type = "float16" if device == "cuda" else "int8"
        
        # 初始化模型 (仅用于 VAD 时，可以使用 tiny 模型以节省显存)
        model = WhisperModel("tiny", device=device, compute_type=compute_type)
        
        # segments 是一个生成器
        segments, info = model.transcribe(
            audio_path, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        results = []
        for segment in segments:
            results.append({
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": ""  # VAD 阶段不关注文本，但保留字段以兼容后端
            })

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.success(f"VAD 切片完成，共 {len(results)} 段: {output_path}")
        return results

    except Exception as e:
        logger.error(f"VAD 处理失败: {e}")
        # 如果 GPU 失败，尝试回退到 CPU
        if device == "cuda":
            logger.warning("尝试回退到 CPU 进行 VAD...")
            return run_vad(audio_path, output_dir, device="cpu")
        raise e
