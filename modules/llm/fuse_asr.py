import os
import json
import asyncio
from loguru import logger
from openai import OpenAI
from modules.utils.rate_limiter import wait_for_llm_api

def fuse_asr_result(multi_asr_results, config=None, prompt_template=None):
    """
    使用 LLM 对多个 ASR 模型的结果进行“投票”与融合修正。
    优化策略：
    1. 恢复标准英文术语（专有名词、技术词汇）
    2. 上下文一致性检查
    3. 标点符号和格式规范化
    4. 长度压缩（去除冗余词）
    """
    if not multi_asr_results:
        return []

    # 兼容性处理
    if isinstance(multi_asr_results, list):
        multi_asr_results = {"whisperx": multi_asr_results}

    if not multi_asr_results.get("whisperx"):
        return []

    logger.info("开始多模型 ASR 结果投票与融合修正...")

    # 从配置读取 API Key
    api_key = config.get('api_key') if config else None
    if not api_key or "your-openai-api-key" in api_key:
        logger.warning("未配置有效的 API Key，跳过融合，默认使用 WhisperX 结果。")
        return multi_asr_results["whisperx"]

    model_name = config.get('model', 'deepseek-ai/DeepSeek-V3')
    base_url = config.get('api_base', 'https://api.siliconflow.cn/v1')
    client = OpenAI(api_key=api_key, base_url=base_url)

    whisper_segs = multi_asr_results["whisperx"]
    
    # 工业级：分批处理，每批 20 个片段（提高批次大小减少 API 调用）
    batch_size = 20
    final_results = [None] * len(whisper_segs)  # 预分配空间保持顺序
    total_batches = (len(whisper_segs) + batch_size - 1) // batch_size
    completed_batches = 0
    
    logger.info(f"总共 {len(whisper_segs)} 个片段，分 {total_batches} 批处理（每批 {batch_size} 个）")
    
    try:
        for i in range(0, len(whisper_segs), batch_size):
            batch_num = i // batch_size + 1
            batch = whisper_segs[i:i+batch_size]
            logger.info(f"正在融合 ASR 片段批次: {batch_num}/{total_batches} (片段 {i+1}-{min(i+len(batch), len(whisper_segs))})...")
            
            combined_context = ""
            for j, seg in enumerate(batch):
                global_idx = i + j
                combined_context += f"片段 {j+1}:\n"
                combined_context += f"- WhisperX: {seg['text']}\n"
                for m in ["parakeet", "canary", "glm"]:
                    if m in multi_asr_results and global_idx < len(multi_asr_results[m]):
                        combined_context += f"- {m.capitalize()}: {multi_asr_results[m][global_idx]['text']}\n"
                combined_context += "\n"

            # 加载术语表
            terminology = {}
            term_file = config.get('terminology_file', 'terminology.json') if config else 'terminology.json'
            if os.path.exists(term_file):
                try:
                    with open(term_file, 'r', encoding='utf-8') as f:
                        terminology = json.load(f)
                except Exception as e:
                    logger.warning(f"加载术语表失败: {e}")
            
            term_str = json.dumps(terminology, ensure_ascii=False) if terminology else "无特殊术语"
            
            prompt = f"""
你是一个专业的音频字幕融合专家。下面是多个 ASR 模型对同一段音频的识别结果。
请选出最准确的内容，并进行以下优化：

1. **术语修正**：参考术语表，确保专有名词和技术词汇准确无误
   - 术语表：{term_str}
2. **上下文一致性**：保持前后文的逻辑连贯性
3. **格式规范**：修正标点符号、大小写、空格等
4. **长度压缩**：去除冗余词、填充词（如 um, uh, you know），使句子更简洁
5. **语言标准化**：如果是英文，恢复为标准英文拼写和语法

必须严格遵守以下格式：
1. 每个片段返回一行，格式为 "数字: 修正后的文本"
2. 返回的行数必须正好是 {len(batch)} 行
3. 即使某些片段识别结果一致，也请重复返回
4. 不要返回任何额外解释或注释

待处理内容：
{combined_context}
"""

            try:
                # API 速率限制检查
                wait_for_llm_api()
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                
                content = response.choices[0].message.content.strip()
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                # 建立一个临时字典保存融合结果，防止行号错乱
                temp_map = {}
                import re
                for line in lines:
                    match = re.match(r'^(\d+)\s*[:：]\s*(.*)$', line)
                    if match:
                        idx = int(match.group(1)) - 1
                        temp_map[idx] = match.group(2).strip()
                    elif ":" in line:
                        try:
                            idx_str, content_part = line.split(":", 1)
                            idx = int(re.sub(r'\D', '', idx_str)) - 1
                            temp_map[idx] = content_part.strip()
                        except: pass

                for j, seg in enumerate(batch):
                    fused_text = temp_map.get(j)
                    if not fused_text and len(lines) == len(batch):
                        curr_line = lines[j]
                        if ":" in curr_line:
                            _, fused_text = curr_line.split(":", 1)
                            fused_text = fused_text.strip()
                        else:
                            fused_text = curr_line
                    
                    global_idx = i + j
                    final_results[global_idx] = {
                        "start": seg["start"], 
                        "end": seg["end"], 
                        "text": fused_text if fused_text else seg["text"]
                    }
                
                # 更新进度
                completed_batches += 1
                progress_pct = (completed_batches / total_batches * 100) if total_batches > 0 else 0
                logger.info(f"批次 {batch_num}/{total_batches} 融合完成 ({progress_pct:.1f}%)")
            except Exception as e:
                logger.error(f"批次融合失败 (批次 {batch_num}/{total_batches}): {e}")
                for j, seg in enumerate(batch):
                    global_idx = i + j
                    final_results[global_idx] = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                # 即使失败也要更新进度
                completed_batches += 1
                progress_pct = (completed_batches / total_batches * 100) if total_batches > 0 else 0
                logger.warning(f"批次 {batch_num}/{total_batches} 失败，使用原文 ({progress_pct:.1f}%)")

        logger.success(f"多模型融合修正完成，共 {len(final_results)} 段。")
        return final_results

    except Exception as e:
        logger.error(f"LLM 融合总流程失败: {e}")
        return whisper_segs
