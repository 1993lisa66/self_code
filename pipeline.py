import os
import yaml
import json
import time
import shutil
from loguru import logger

# 导入功能模块
from modules.audio.extract_audio import extract_audio
from modules.vad.vad_pipeline import run_vad
from modules.asr.multi_asr import run_multi_asr
from modules.llm.fuse_asr import fuse_asr_result
from modules.llm.resegmentation import semantic_resegment
from modules.translate.translate_pipeline import translate_segments
from modules.subtitle.generate_srt import generate_srt
from modules.tts.tts_pipeline import generate_tts
from modules.tts.text_processor import process_tts_text
from modules.merge.merge_video import merge_video
from modules.utils.chapter_generator import generate_chapters
import re

def split_into_subtitles(translated_results):
    """
    将长句子进一步拆分为多个短字幕，使屏幕显示更易读。
    参考 ex/summary.txt 中的逻辑：翻译数 < 字幕数。
    优化：去除连续重复的字幕内容
    """
    final_subtitles = []
    for seg in translated_results:
        text = seg.get("translated_text", seg["text"])
        
        # 如果文本为空或过短，跳过
        if not text or len(text.strip()) < 2:
            continue
            
        # 按标点符号拆分
        parts = re.split(r'([，。！？；])', text)
        
        # 重新组合，保持标点符号在句子末尾
        combined_parts = []
        for i in range(0, len(parts)-1, 2):
            part = parts[i] + parts[i+1]
            if part.strip():  # 只添加非空部分
                combined_parts.append(part)
        if len(parts) % 2 == 1 and parts[-1]:
            if parts[-1].strip():
                combined_parts.append(parts[-1])
            
        # 如果没有拆分成功，直接使用原文
        if len(combined_parts) <= 1:
            # 检查是否与上一个字幕重复
            if final_subtitles and final_subtitles[-1].get("translated_text") == text:
                continue  # 跳过重复内容
            final_subtitles.append(seg.copy())
            continue
            
        # 按字符数比例分配时长
        total_chars = sum(len(p) for p in combined_parts)
        total_duration = seg['end'] - seg['start']
        current_start = seg['start']
        
        for p in combined_parts:
            if not p.strip():  # 跳过空片段
                continue
                
            part_len = len(p)
            part_duration = (part_len / total_chars) * total_duration
            new_seg = seg.copy()
            new_seg["translated_text"] = p
            new_seg["start"] = current_start
            new_seg["end"] = current_start + part_duration
            
            # 检查是否与上一个字幕重复
            if final_subtitles and final_subtitles[-1].get("translated_text") == p:
                continue  # 跳过重复内容
                
            final_subtitles.append(new_seg)
            current_start += part_duration
            
    return final_subtitles

class VideoTranslationPipeline:
    def __init__(self, config_path="config/config.yaml", config_dict=None):
        """
        初始化 Pipeline
        
        Args:
            config_path: 配置文件路径（当 config_dict 为 None 时使用）
            config_dict: 配置字典（直接传入，避免重复加载文件）
        """
        # 加载 YAML 配置
        if config_dict is not None:
            # 直接使用传入的配置字典
            self.config = config_dict
        else:
            # 从文件加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        # 配置日志系统
        log_dir = self.config['paths']['log_dir']
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger.add(
            os.path.join(log_dir, "runtime_{time:YYYY-MM-DD}.log"),
            rotation="00:00",
            retention="7 days"
        )
        
        # 预加载提示词内容
        self.prompts = self._load_prompts()
        
        logger.info(f"Pipeline 初始化完成，项目名称: {self.config['project']['name']}")

    def _load_prompts(self):
        """从文件加载所有提示词模板"""
        prompts = {}
        prompts_dir = self.config['paths']['prompts_dir']
        prompt_files = self.config['llm']['prompts']
        
        for key, filename in prompt_files.items():
            path = os.path.join(prompts_dir, filename)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    prompts[key] = f.read()
            else:
                logger.warning(f"提示词文件不存在: {path}")
                prompts[key] = ""
        
        # 自动检测术语表：prompts 目录的父目录中是否有 terminology.json
        if 'terminology_file' not in self.config['llm']:
            parent_dir = os.path.dirname(prompts_dir)
            term_path = os.path.join(parent_dir, "terminology.json")
            if os.path.exists(term_path):
                self.config['llm']['terminology_file'] = term_path
                logger.info(f"自动检测到批次术语表: {term_path}")
        
        return prompts

    def run(self, video_path, sub_dir=""):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"开始处理视频: {video_path}")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 计算输出目录 (保持原有的目录结构 + 以文件名命名的独立文件夹)
        output_base_dir = os.path.abspath(self.config['paths']['output_dir'])
        target_output_dir = os.path.join(output_base_dir, sub_dir, base_name)
        if not os.path.exists(target_output_dir):
            try:
                os.makedirs(target_output_dir, exist_ok=True)
                logger.info(f"创建输出目录: {target_output_dir}")
            except Exception as e:
                logger.error(f"无法创建输出目录 {target_output_dir}: {e}")
                raise
            
        # 缓存目录也保持结构 (绝对路径 + 以文件名命名的独立文件夹)
        cache_base_dir = os.path.abspath(self.config['paths']['cache_dir'])
        target_cache_dir = os.path.join(cache_base_dir, sub_dir, base_name)
        if not os.path.exists(target_cache_dir):
            try:
                os.makedirs(target_cache_dir, exist_ok=True)
                logger.info(f"创建缓存目录: {target_cache_dir}")
            except Exception as e:
                logger.error(f"无法创建缓存目录 {target_cache_dir}: {e}")
                raise
            
        # 用于保存全量结果的字典
        final_result_data = {
            "video_path": video_path,
            "start_time": start_time,
            "config": self.config,
            "steps": {}
        }
        
        try:
            # STEP 1: 提取音频
            logger.info("STEP 1: 提取音频")
            audio_path = extract_audio(
                video_path, 
                output_dir=os.path.join(target_cache_dir, "audio"),
                sample_rate=self.config['audio']['sample_rate']
            )
            final_result_data["steps"]["audio_extraction"] = {"path": audio_path}
            
            # STEP 2: VAD 语音切片
            logger.info("STEP 2: VAD切片")
            segments = run_vad(
                audio_path,
                output_dir=os.path.join(target_cache_dir, "vad"),
                device=self.config['asr']['device']
            )
            final_result_data["steps"]["vad"] = {"count": len(segments), "segments": segments}
            
            # STEP 3: 多 ASR 识别
            logger.info("STEP 3: 多ASR识别")
            asr_results = run_multi_asr(
                audio_path, 
                segments,
                output_dir=os.path.join(target_cache_dir, "asr"),
                device=self.config['asr']['device'],
                model_size=self.config['asr'].get('model_size', 'large-v3'),
                language=self.config['asr'].get('language') if self.config['asr'].get('language') != 'auto' else None,
            )
            final_result_data["steps"]["asr"] = {"results": asr_results}
            
            # STEP 4: LLM 修正
            logger.info("STEP 4: LLM融合修正")
            fused_results = fuse_asr_result(
                asr_results, 
                config=self.config['llm'],
                prompt_template=self.prompts.get('asr_fix')
            )
            final_result_data["steps"]["llm_fusion"] = {"results": fused_results}
            
            # STEP 4.5: 语义重切分 (优化长句断句)
            logger.info("STEP 4.5: 语义重切分")
            resegmented_results = semantic_resegment(
                fused_results,
                config=self.config['llm'],
                prompt_template=self.prompts.get('semantic_segmentation')
            )
            final_result_data["steps"]["resegmentation"] = {"results": resegmented_results}
            
            # 工业级优化：回填融合后的文本到 VAD 缓存，方便用户查看和调试
            try:
                vad_cache_path = os.path.join(target_cache_dir, "vad", f"{base_name}_segments.json")
                if os.path.exists(vad_cache_path):
                    with open(vad_cache_path, 'w', encoding='utf-8') as f:
                        json.dump(fused_results, f, ensure_ascii=False, indent=2)
                    logger.info(f"已回填融合文本至 VAD 缓存: {vad_cache_path}")
            except Exception as e:
                logger.warning(f"回填 VAD 缓存失败 (不影响主流程): {e}")
            
            # STEP 5: 翻译
            logger.info("STEP 5: 翻译")
            # 合并翻译引擎配置（provider: llm/google）
            translate_cfg = self.config.get('translate', {})
            merged_config = dict(self.config.get('llm', {}))
            if translate_cfg:
                merged_config['provider'] = translate_cfg.get('provider', 'llm')
                merged_config['google_delay'] = translate_cfg.get('google_delay', 0.3)
            translated_results = translate_segments(
                resegmented_results, 
                target_lang=self.config['translate']['target_language'],
                config=merged_config,
                prompt_template=self.prompts.get('translation')
            )
            final_result_data["steps"]["translation"] = {"results": translated_results}
            
            # STEP 6: TTS 文本预处理 & 语音合成
            logger.info("STEP 6: TTS 文本处理与合成")
            # 文本预处理 (数字转中文) - 使用并发处理提高速度
            total_segments = len(translated_results)
            logger.info(f"开始 TTS 文本预处理，共 {total_segments} 个片段...")
            
            import asyncio
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_single_segment(idx_seg):
                """处理单个片段的函数"""
                idx, seg = idx_seg
                # 保存原始语言文本（如英文），供 tts_preprocessed.json 等下游使用
                original_text = seg.get("translated_text", seg["text"])
                tts_text = process_tts_text(
                    original_text, 
                    config=self.config['llm'], 
                    prompt_template=self.prompts.get('tts_prep')
                )
                return idx, tts_text, seg.get("text", "")  # 返回原文供回填
            
            # 使用线程池并发处理（最多 10 个并发）
            max_workers = min(10, total_segments)
            logger.info(f"使用 {max_workers} 个线程并发处理...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_single_segment, (idx, seg)): idx 
                          for idx, seg in enumerate(translated_results)}
                
                completed = 0
                for future in as_completed(futures):
                    idx, tts_text, original_text = future.result()
                    translated_results[idx]["tts_text"] = tts_text
                    # 回填原始语言文本（如英文），确保 tts_preprocessed.json 包含原文
                    if original_text and not translated_results[idx].get("original_text"):
                        translated_results[idx]["original_text"] = original_text
                    completed += 1
                    
                    # 每完成 10% 或至少每 10 个显示一次进度
                    if completed % max(1, total_segments // 10) == 0 or completed == total_segments:
                        progress_pct = (completed / total_segments * 100) if total_segments > 0 else 0
                        logger.info(f"TTS 文本预处理进度: {completed}/{total_segments} ({progress_pct:.1f}%)")
            
            tts_audio = asyncio.run(generate_tts(
                translated_results,
                output_dir=os.path.join(target_cache_dir, "tts"),
                config=self.config['tts']
            ))
            final_result_data["steps"]["tts"] = {"path": tts_audio}

            # STEP 7: 生成 SRT 字幕 (在 TTS 后生成，以匹配更新后的时间轴)
            logger.info("STEP 7: 生成字幕")
            # 工业级：将长句子拆分为多行字幕以提升观看体验
            final_subtitles = split_into_subtitles(translated_results)
            logger.info(f"字幕拆分完成: {len(translated_results)} 句 -> {len(final_subtitles)} 行")
            
            # 将拆分后的结果存入 final_result_data 供导出使用
            data_for_export = final_result_data.copy()
            data_for_export["final_subtitles_list"] = final_subtitles 

            srt_path = generate_srt(
                final_subtitles, 
                output_path=os.path.join(target_output_dir, f"{base_name}.srt")
            )
            final_result_data["steps"]["subtitle"] = {"path": srt_path, "final_subtitles": final_subtitles}
            
            # STEP 8: 生成章节
            logger.info("STEP 8: 生成章节")
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            chapters_content = generate_chapters(
                srt_content, 
                config=self.config['llm'], 
                prompt_template=self.prompts.get('chapters')
            )
            chapters_path = os.path.join(target_output_dir, f"{base_name}_chapters.txt")
            with open(chapters_path, 'w', encoding='utf-8') as f:
                f.write(chapters_content)
            final_result_data["steps"]["chapters"] = {"path": chapters_path, "content": chapters_content}
            
            # STEP 9: 视频最终合成
            logger.info("STEP 9: 视频最终合成")
            final_video = merge_video(
                video_path, 
                tts_audio, 
                output_dir=target_output_dir,
                config=self.config.get('video', {})
            )
            final_result_data["final_video"] = final_video
            
            # STEP 10: 导出全量结果与摘要
            self._export_final_outputs(base_name, final_result_data, target_output_dir)
            
            # STEP 11: 清理缓存（根据配置）
            if self.config.get('global', {}).get('auto_cleanup_cache', True):
                self._cleanup_cache(target_cache_dir, base_name)
            else:
                logger.info(f"缓存已保留: {target_cache_dir}")
            
            logger.success(f"视频 [{base_name}] 处理完成！")
            return final_video
            
        except Exception as e:
            logger.error(f"处理视频 {video_path} 时 Pipeline 崩溃: {e}")
            # 即使失败也尝试清理缓存（如果配置了自动清理）
            if self.config.get('global', {}).get('auto_cleanup_cache', True):
                try:
                    self._cleanup_cache(target_cache_dir, base_name, success=False)
                except:
                    pass
            raise e

    def _export_final_outputs(self, base_name, data, target_output_dir):
        """导出符合 ex/summary.txt 和 ex/result.json 格式的产物"""
        # 1. 导出 result.json
        result_path = os.path.join(target_output_dir, f"{base_name}_result.json")
        
        # 构造符合 ex/result.json 结构的字典
        clean_result = {
            "tts_model_type": self.config['tts'].get('provider', 'edge'),
            "video_path": data.get("video_path"),
            "output_dir": target_output_dir,
            "config": data.get("config"),
            "vad_list": data.get("steps", {}).get("vad", {}).get("segments", []),
            "audio_segments": [data.get("steps", {}).get("audio_extraction", {}).get("path")],
            "whisperx_asr_result": data.get("steps", {}).get("asr", {}).get("results", {}).get("whisperx", []),
            "glm_asr_result": data.get("steps", {}).get("asr", {}).get("results", {}).get("glm", []),
            "final_asr_result": data.get("steps", {}).get("llm_fusion", {}).get("results", []),
            "merged_asr_paragraphs": data.get("steps", {}).get("resegmentation", {}).get("results", []),
            "sentence_paragraph_indices": [], # 占位，如果需要可以从 resegmentation 逻辑中提取
            "translated_subtitles": data.get("steps", {}).get("translation", {}).get("results", []),
            "final_subtitles": [], # 将在下面填充
            "chapter_file_path": data.get("steps", {}).get("chapters", {}).get("path"),
            "metadata": {
                "video_path": data.get("video_path"),
                "output_dir": target_output_dir,
                "tts_model_type": self.config['tts'].get('provider', 'edge'),
                "processed_time": data.get("start_time"),
                "polish_status": "completed",
                "agent_state_version": "1.0"
            }
        }
        
        # 填充 final_subtitles (SRT 生成后的最终片段)
        if "subtitle" in data.get("steps", {}) and "final_subtitles" in data["steps"]["subtitle"]:
            for seg in data["steps"]["subtitle"]["final_subtitles"]:
                clean_result["final_subtitles"].append({
                    "text": seg.get("translated_text", ""),
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "word_count": len(seg.get("translated_text", ""))
                })

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(clean_result, f, ensure_ascii=False, indent=2)
            
        # 2. 导出 summary.txt
        summary_path = os.path.join(target_output_dir, f"{base_name}_summary.txt")
        
        # 计算时长
        total_audio_duration = 0
        if clean_result["vad_list"]:
            total_audio_duration = sum(s['end'] - s['start'] for s in clean_result["vad_list"])
            
        tts_duration = 0
        if clean_result["final_subtitles"]:
            tts_duration = clean_result["final_subtitles"][-1]["end"]

        summary_lines = [
            f"=== {self.config['project']['name']} 处理摘要 ===",
            "",
            f"源视频文件: {data['video_path']}",
            f"输出目录: {target_output_dir}",
            f"处理时间: {data['start_time']}",
            f"TTS模型: {self.config['tts']['provider']}",
            "",
            "--- VAD (语音活动检测) ---",
            f"VAD片段数: {len(clean_result['vad_list'])}",
            f"音频段落数: {len(clean_result['vad_list'])}",
            f"总语音时长: {total_audio_duration:.2f} 秒",
            "",
            "--- ASR (自动语音识别) ---",
            f"WhisperX识别结果数: {len(clean_result['whisperx_asr_result'])}",
            f"GLM识别结果数: {len(clean_result['glm_asr_result'])}",
            f"最终ASR结果数: {len(clean_result['final_asr_result'])}",
            "",
            "--- 段落处理 ---",
            f"合并段落数: {len(clean_result['merged_asr_paragraphs'])}",
            f"句子段落索引数: {len(clean_result['merged_asr_paragraphs'])}",
            "",
            "--- 翻译和润色 ---",
            f"翻译字幕数: {len(clean_result['translated_subtitles'])}",
            "",
            "--- TTS (文本转语音) ---",
            f"TTS输入数: {len(clean_result['translated_subtitles'])}",
            f"TTS音频文件数: {len(clean_result['translated_subtitles'])}",
            f"TTS最终时间轴数: {len(clean_result['final_subtitles'])}",
            f"TTS总时长: {tts_duration:.2f} 秒",
            "",
            "--- 最终输出 ---",
            f"最终字幕数: {len(clean_result['final_subtitles'])}",
            f"章节数: {len(data['steps']['chapters']['content'].splitlines()) if data['steps']['chapters'].get('content') else 0}",
            f"章节文件: {clean_result['chapter_file_path']}",
            ""
        ]
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))
        
        logger.info(f"已导出摘要和全量数据到 {target_output_dir}")

    def _cleanup_cache(self, cache_dir, video_name, success=True):
        """
        清理视频处理后的缓存目录
        
        Args:
            cache_dir: 缓存目录路径
            video_name: 视频名称（用于日志）
            success: 是否成功处理完成
        """
        if not os.path.exists(cache_dir):
            return
        
        try:
            # 计算缓存大小
            total_size = 0
            file_count = 0
            for dirpath, dirnames, filenames in os.walk(cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
                    file_count += 1
            
            size_mb = total_size / (1024 * 1024)
            
            if success:
                logger.info(f"STEP 11: 清理缓存")
                logger.info(f"  - 视频: {video_name}")
                logger.info(f"  - 缓存文件数: {file_count}")
                logger.info(f"  - 缓存大小: {size_mb:.2f} MB")
                logger.info(f"  - 缓存目录: {cache_dir}")
                
                # 删除整个缓存目录
                shutil.rmtree(cache_dir)
                logger.success(f"✅ 缓存已清理，释放 {size_mb:.2f} MB 空间")
            else:
                logger.warning(f"⚠️  视频处理失败，清理缓存")
                logger.info(f"  - 视频: {video_name}")
                logger.info(f"  - 缓存大小: {size_mb:.2f} MB")
                shutil.rmtree(cache_dir)
                logger.info(f"✅ 失败任务的缓存已清理")
                
        except Exception as e:
            logger.warning(f"清理缓存时出错: {e}（不影响主流程）")
