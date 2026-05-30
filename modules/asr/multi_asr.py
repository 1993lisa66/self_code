import os
import json
import time
import numpy as np
import librosa
import threading
from loguru import logger
from modules.utils.patch_torch import apply_torch_patch

apply_torch_patch()

class ASRModelLoader:
    """ASR 模型加载与缓存器 (单例模式)"""
    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ASRModelLoader, cls).__new__(cls)
        return cls._instance

    def get_fw_model(self, model_size="large-v3", device="cuda"):
        key = f"fw_{model_size}_{device}"
        if key not in self._models:
            from faster_whisper import WhisperModel
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info(f"加载 Faster-Whisper 模型: {model_size} ({device})")
            self._models[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
        return self._models[key]

    def get_funasr_model(self, device="cuda"):
        if "funasr" not in self._models:
            try:
                from funasr import AutoModel
                logger.info("加载 FunASR SenseVoiceSmall (iic/SenseVoiceSmall)...")
                model = AutoModel(
                    model="iic/SenseVoiceSmall", 
                    device=device,
                    disable_update=True
                )
                self._models["funasr"] = model
            except ImportError:
                logger.warning("未安装 funasr。如需使用 GLM/SenseVoice，请运行: pip install funasr modelscope")
                return None
            except Exception as e:
                logger.error(f"加载 FunASR 模型失败: {e}")
                logger.info("\n" + "="*70)
                logger.info("手动下载 FunASR 模型指南:")
                logger.info("="*70)
                logger.info("1. 访问 ModelScope: https://modelscope.cn/models/iic/SenseVoiceSmall")
                logger.info("2. 下载模型文件到以下目录:")
                logger.info(f"   {os.path.abspath('models/funasr/models/iic/SenseVoiceSmall')}")
                logger.info("3. 确保目录结构如下:")
                logger.info("   models/funasr/models/iic/SenseVoiceSmall/")
                logger.info("   ├── config.yaml")
                logger.info("   ├── configuration.json")
                logger.info("   ├── model.pt")
                logger.info("   ├── README.md")
                logger.info("   └── ...")
                logger.info("4. 重新运行程序")
                logger.info("="*70 + "\n")
                return None
        return self._models["funasr"]

def run_multi_asr(audio_path, segments, output_dir="cache/asr", device="cuda", model_size="large-v3"):
    """
    多 ASR 识别模块。使用双模型架构：
    1. WhisperX / Faster-Whisper (Ground Truth 时间轴)
    2. GLM / SenseVoice (FunASR)
    """
    # 确保路径是绝对路径
    audio_path = os.path.abspath(audio_path)
    output_dir = os.path.abspath(output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_multi_asr.json")

    if os.path.exists(output_path):
        logger.info(f"使用 Multi-ASR 缓存: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    loader = ASRModelLoader()
    multi_results = {"whisperx": [], "glm": []}

    try:
        # --- 1. WhisperX (Faster-Whisper) ---
        fw_model = loader.get_fw_model(model_size, device)
        logger.info("开始 WhisperX 转录...")
        segments_res, info = fw_model.transcribe(audio_path, beam_size=5)
        
        # 检测到的语言（如有）
        detected_lang = info.language if hasattr(info, 'language') else 'N/A'
        logger.info(f"  检测语言: {detected_lang}")
        
        whisper_segs = []
        ws_start = time.time()
        last_report = ws_start
        for i, s in enumerate(segments_res):
            whisper_segs.append({
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip()
            })
            # 每 100 个片段或每 30 秒报告一次进度
            now = time.time()
            if (i + 1) % 100 == 0 or now - last_report > 30:
                elapsed = now - ws_start
                last_report = now
                logger.info(f"  WhisperX 进度: {i + 1} 个片段已处理 ({elapsed:.0f}s / ~{elapsed / (i + 1):.1f}s per)")
        
        elapsed_total = time.time() - ws_start
        multi_results["whisperx"] = whisper_segs
        logger.success(f"WhisperX 完成，{len(whisper_segs)} 个片段 / 耗时 {elapsed_total:.0f}s")

        if not whisper_segs:
            return multi_results

        # 加载音频用于切片
        logger.info("加载音频数据用于 FunASR 切片对齐...")
        audio_data, sr = librosa.load(audio_path, sr=16000)

        # 准备切片音频
        audio_chunks = []
        for seg in whisper_segs:
            start_idx = int(seg["start"] * sr)
            end_idx = int(seg["end"] * sr)
            audio_chunks.append(audio_data[start_idx:end_idx])
        
        # --- 2. FunASR GLM ---
        funasr_model = loader.get_funasr_model(device)
        
        n_batch_size = 10
        threads = []
        errors = {}
        
        # ---- FunASR GLM ----
        def _run_funasr():
            try:
                logger.info("[Thread] FunASR 批量识别启动...")
                total_chunks = len(audio_chunks)
                fs_start = time.time()
                for i in range(0, total_chunks, n_batch_size):
                    batch_chunks = audio_chunks[i:i+n_batch_size]
                    res = funasr_model.generate(
                        input=batch_chunks, cache={}, language="auto",
                        use_itn=True, batch_size_s=0
                    )
                    for r in res:
                        text = r.get('text', '').strip() if isinstance(r, dict) else str(r or '').strip()
                        multi_results["glm"].append({"text": text})
                    # 每 500 个片段报告一次进度
                    processed = min(i + n_batch_size, total_chunks)
                    if processed % 500 == 0 or processed >= total_chunks:
                        elapsed = time.time() - fs_start
                        logger.info(f"  FunASR 进度: {processed}/{total_chunks} ({elapsed:.0f}s)")
                logger.success(f"[Thread] FunASR 完成，{len(multi_results['glm'])} 片段")
            except Exception as e:
                logger.error(f"[Thread] FunASR 失败: {e}")
                errors["glm"] = str(e)
        
        # 启动线程
        if funasr_model:
            t = threading.Thread(target=_run_funasr, name="FunASR")
            t.start()
            threads.append(t)
        else:
            logger.warning("FunASR 模型未加载，跳过 GLM 识别")
        
        logger.info(f"等待 {len(threads)} 个 ASR 模型线程完成...")
        for t in threads:
            t.join()
        
        if errors:
            logger.warning(f"部分模型识别失败: {list(errors.keys())}")

        # 补全逻辑：如果 FunASR 失败，用 WhisperX 结果填充
        if not multi_results["glm"]:
            logger.info("GLM 结果为空，使用 WhisperX 占位")
            multi_results["glm"] = [{"text": s["text"]} for s in whisper_segs]
        elif len(multi_results["glm"]) < len(whisper_segs):
            logger.info("GLM 结果长度不足，正在补齐...")
            while len(multi_results["glm"]) < len(whisper_segs):
                idx = len(multi_results["glm"])
                multi_results["glm"].append({"text": whisper_segs[idx]["text"]})

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(multi_results, f, ensure_ascii=False, indent=2)

        return multi_results

    except Exception as e:
        logger.error(f"多模型 ASR 核心流程失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

