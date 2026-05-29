import os
import json
import torch
import numpy as np
import librosa
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

    def get_nemo_models(self, device="cuda"):
        if "nemo" not in self._models:
            try:
                import nemo.collections.asr as nemo_asr
                logger.info("尝试加载 NeMo Parakeet (nvidia/parakeet-tdt-1.1b)...")
                
                # 增加重试机制和网络超时处理
                max_retries = 3
                parakeet = None
                for attempt in range(max_retries):
                    try:
                        parakeet = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-1.1b")
                        logger.info(f"Parakeet 模型加载成功，类型: {type(parakeet)}")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Parakeet 下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                            import time
                            time.sleep(2)  # 等待2秒后重试
                        else:
                            raise e
                
                logger.info("尝试加载 NeMo Canary (nvidia/canary-1b)...")
                canary = None
                for attempt in range(max_retries):
                    try:
                        canary = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b")
                        logger.info(f"Canary 模型加载成功，类型: {type(canary)}")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Canary 下载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                            import time
                            time.sleep(2)
                        else:
                            raise e
                
                if device == "cuda" and torch.cuda.is_available():
                    logger.info("将 NeMo 模型移动到 GPU...")
                    parakeet = parakeet.cuda()
                    canary = canary.cuda()
                else:
                    logger.info("使用 CPU 运行 NeMo 模型")
                
                # 设置为评估模式
                parakeet.eval()
                canary.eval()
                logger.info("NeMo 模型已设置为评估模式")
                
                self._models["nemo"] = {"parakeet": parakeet, "canary": canary}
                logger.success("NeMo 模型加载完成")
            except ImportError:
                logger.warning("未安装 nemo_toolkit。如需使用 Parakeet/Canary，请运行: pip install nemo_toolkit[all]")
                return None
            except Exception as e:
                logger.error(f"加载 NeMo 模型失败: {e}")
                logger.warning("跳过 NeMo 模型，将使用 WhisperX + FunASR 进行识别")
                import traceback
                logger.debug(traceback.format_exc())
                return None
        else:
            logger.info("使用缓存的 NeMo 模型")
        return self._models["nemo"]

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
    多 ASR 识别模块。使用 4 模型架构：
    1. WhisperX / Faster-Whisper (Ground Truth 时间轴)
    2. GLM / SenseVoice (FunASR)
    3. Parakeet-TDT 1.1B (NeMo) — 新增
    4. Canary-1B (NeMo) — 新增
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
            cached = json.load(f)
        # 兼容旧缓存：缺少 parakeet/canary 时降级重跑
        if "parakeet" not in cached or "canary" not in cached:
            logger.info("检测到旧版缓存（缺少 parakeet/canary），将重新识别...")
        else:
            return cached

    loader = ASRModelLoader()
    multi_results = {"whisperx": [], "glm": [], "parakeet": [], "canary": []}

    try:
        # --- 1. WhisperX (Faster-Whisper) ---
        fw_model = loader.get_fw_model(model_size, device)
        logger.info("开始 WhisperX 转录...")
        segments_res, _ = fw_model.transcribe(audio_path, beam_size=5)
        
        whisper_segs = []
        for s in segments_res:
            whisper_segs.append({
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip()
            })
        multi_results["whisperx"] = whisper_segs
        logger.success(f"WhisperX 完成，得到 {len(whisper_segs)} 个片段")

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
        
        # --- 3. FunASR GLM (SenseVoice) ---
        funasr_model = loader.get_funasr_model(device)
        if funasr_model:
            try:
                logger.info("开始 FunASR 批量识别...")
                # FunASR 支持批量输入数组列表
                # 为了防止内存溢出，我们按 10 个一批处理
                f_batch_size = 10
                total_processed = 0
                for i in range(0, len(audio_chunks), f_batch_size):
                    batch_chunks = audio_chunks[i:i+f_batch_size]
                    # logger.info(f"FunASR 处理批次 {i//f_batch_size + 1}/{(len(audio_chunks) + f_batch_size - 1)//f_batch_size} ({len(batch_chunks)} 个片段)...")
                    
                    # SenseVoiceSmall 推荐参数
                    res = funasr_model.generate(
                        input=batch_chunks, 
                        cache={}, 
                        language="auto", 
                        use_itn=True,
                        batch_size_s=0 # 禁用按秒批处理，直接按个数
                    )
                    
                    # logger.info(f"FunASR 批次结果类型: {type(res)}, 长度: {len(res) if hasattr(res, '__len__') else 'N/A'}")
                    for idx, r in enumerate(res):
                        if isinstance(r, dict):
                            text = r.get('text', '').strip()
                        else:
                            text = str(r).strip() if r else ''
                        
                        if not text:
                            logger.warning(f"FunASR 批次 {i//f_batch_size} 第 {idx} 个片段结果为空")
                        multi_results["glm"].append({"text": text})
                    
                    total_processed += len(batch_chunks)
                    # logger.info(f"FunASR 已处理 {total_processed}/{len(audio_chunks)} 个片段")
                    
                logger.success(f"FunASR 批量识别完成，共处理 {len(multi_results['glm'])} 个片段")
            except Exception as e:
                logger.error(f"FunASR 运行中出错: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # --- 4. NeMo Parakeet + Canary (1.1B 参数级模型) ---
        nemo_models = loader.get_nemo_models(device)
        if nemo_models:
            import tempfile
            import wave
            
            # 将切片保存为临时 WAV 文件供 NeMo 调用
            chunk_temp_dir = os.path.join(output_dir, "chunks_nemo")
            os.makedirs(chunk_temp_dir, exist_ok=True)
            chunk_paths = []
            try:
                for idx, chunk_data in enumerate(audio_chunks):
                    chunk_path = os.path.join(chunk_temp_dir, f"chunk_{idx}.wav")
                    with wave.open(chunk_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(sr)
                        wf.writeframes((chunk_data * 32767).astype(np.int16).tobytes())
                    chunk_paths.append(chunk_path)
                
                # 按批次处理，避免内存溢出
                n_batch_size = 10
                
                # 4a. Parakeet
                logger.info("开始 NeMo Parakeet 批量识别...")
                try:
                    total_p = 0
                    for i in range(0, len(chunk_paths), n_batch_size):
                        batch_paths = chunk_paths[i:i+n_batch_size]
                        res = nemo_models["parakeet"].transcribe(
                            paths2audio_files=batch_paths,
                            batch_size=min(n_batch_size, len(batch_paths)),
                            num_workers=0
                        )
                        for r in res:
                            multi_results["parakeet"].append({"text": r.strip() if isinstance(r, str) else str(r).strip()})
                        total_p += len(batch_paths)
                    logger.success(f"Parakeet 识别完成，共 {len(multi_results['parakeet'])} 个片段")
                except Exception as e:
                    logger.error(f"Parakeet 识别失败: {e}")
                    # 尝试逐条识别的备用方案
                    logger.info("Parakeet 回退到逐条识别...")
                    try:
                        for chunk_path in chunk_paths:
                            try:
                                res = nemo_models["parakeet"].transcribe(paths2audio_files=[chunk_path], batch_size=1, num_workers=0)
                                multi_results["parakeet"].append({"text": res[0].strip() if res else ""})
                            except Exception:
                                multi_results["parakeet"].append({"text": ""})
                        logger.success(f"Parakeet 逐条识别完成: {len(multi_results['parakeet'])}")
                    except Exception as e2:
                        logger.error(f"Parakeet 回退方案也失败: {e2}")
                
                # 4b. Canary
                logger.info("开始 NeMo Canary 批量识别...")
                try:
                    total_c = 0
                    for i in range(0, len(chunk_paths), n_batch_size):
                        batch_paths = chunk_paths[i:i+n_batch_size]
                        res = nemo_models["canary"].transcribe(
                            paths2audio_files=batch_paths,
                            batch_size=min(n_batch_size, len(batch_paths)),
                            num_workers=0
                        )
                        for r in res:
                            multi_results["canary"].append({"text": r.strip() if isinstance(r, str) else str(r).strip()})
                        total_c += len(batch_paths)
                    logger.success(f"Canary 识别完成，共 {len(multi_results['canary'])} 个片段")
                except Exception as e:
                    logger.error(f"Canary 识别失败: {e}")
                    logger.info("Canary 回退到逐条识别...")
                    try:
                        for chunk_path in chunk_paths:
                            try:
                                res = nemo_models["canary"].transcribe(paths2audio_files=[chunk_path], batch_size=1, num_workers=0)
                                multi_results["canary"].append({"text": res[0].strip() if res else ""})
                            except Exception:
                                multi_results["canary"].append({"text": ""})
                        logger.success(f"Canary 逐条识别完成: {len(multi_results['canary'])}")
                    except Exception as e2:
                        logger.error(f"Canary 回退方案也失败: {e2}")
                
            finally:
                # 清理临时 WAV 文件
                if os.path.exists(chunk_temp_dir):
                    import shutil
                    shutil.rmtree(chunk_temp_dir, ignore_errors=True)
        else:
            logger.warning("NeMo 模型加载失败，跳过 Parakeet/Canary 识别")

        # 补全逻辑：如果某模型失败，用 WhisperX 结果填充
        def _fill_if_needed(model_key):
            if not multi_results[model_key]:
                logger.info(f"模型 {model_key} 结果为空，使用 WhisperX 占位")
                multi_results[model_key] = [{"text": s["text"]} for s in whisper_segs]
            elif len(multi_results[model_key]) < len(whisper_segs):
                logger.info(f"模型 {model_key} 结果长度不足，正在补齐...")
                while len(multi_results[model_key]) < len(whisper_segs):
                    idx = len(multi_results[model_key])
                    multi_results[model_key].append({"text": whisper_segs[idx]["text"]})

        for key in ["glm", "parakeet", "canary"]:
            _fill_if_needed(key)

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(multi_results, f, ensure_ascii=False, indent=2)

        return multi_results

    except Exception as e:
        logger.error(f"多模型 ASR 核心流程失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

