import os
import json
from loguru import logger
from openai import OpenAI
from modules.utils.rate_limiter import wait_for_llm_api

def translate_segments(fused_results, target_lang="zh", config=None, prompt_template=None):
    """
    使用 LLM 进行翻译。
    优化策略：
    1. 术语一致性（参考术语表）
    2. 上下文连贯性（传递前文作为参考）
    3. 口语化表达（适合视频字幕）
    4. 长度控制（避免过长句子）
    5. 文化适配（本地化表达）
    """
    if not fused_results:
        return []

    logger.info(f"开始翻译任务 -> {target_lang}...")

    # 从配置读取 API Key
    api_key = config.get('api_key') if config else None
    if not api_key or "your-openai-api-key" in api_key:
        logger.warning("未配置有效的 API Key，返回占位翻译。")
        for seg in fused_results:
            seg["translated_text"] = f"[FIXME] {seg['text']}"
        return fused_results

    model_name = config.get('model', 'deepseek-ai/DeepSeek-V3')
    base_url = config.get('api_base', 'https://api.siliconflow.cn/v1')
    temperature = config.get('temperature', 0.1)
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 尝试加载术语表
    terminology = {}
    term_file = "terminology.json" # 默认路径
    if os.path.exists(term_file):
        try:
            with open(term_file, 'r', encoding='utf-8') as f:
                terminology = json.load(f)
        except: pass

    translated_results = []
    
    # 工业级：增大批次大小以减少 API 调用次数
    batch_size = 20  # 从 10 增加到 20，减少一半的 API 调用
    total_batches = (len(fused_results) + batch_size - 1) // batch_size
    completed_batches = 0
    
    logger.info(f"总共 {len(fused_results)} 个片段，分 {total_batches} 批处理（每批 {batch_size} 个）")
    
    for i in range(0, len(fused_results), batch_size):
        batch_num = i // batch_size + 1
        batch = fused_results[i:i+batch_size]
        
        # 显示当前批次进度
        logger.info(f"正在翻译批次 {batch_num}/{total_batches} (片段 {i+1}-{min(i+len(batch), len(fused_results))})...")
        
        # 构造批量翻译提示词
        texts_to_translate = "\n".join([f"{j+1}: {seg['text']}" for j, seg in enumerate(batch)])
        term_str = json.dumps(terminology, ensure_ascii=False) if terminology else "无"
        
        # 提取前文作为上下文参考（最近 5 个片段）
        context_start = max(0, i - 5)
        context_segments = fused_results[context_start:i]
        context_text = "\n".join([f"{idx+1}: {seg.get('translated_text', seg['text'])}" 
                                 for idx, seg in enumerate(context_segments)]) if context_segments else "无"
        
        if prompt_template:
            try:
                # 尝试格式化，支持 target_lang, text, terminology, context
                prompt = prompt_template.format(
                    target_lang=target_lang, 
                    text=texts_to_translate,
                    terminology=term_str,
                    context=context_text
                )
            except KeyError:
                # 如果格式化失败（例如缺少某些键），回退到默认
                prompt = f"请将以下内容翻译成 {target_lang}，参考术语表 {term_str}。如果原文不是 {target_lang}，请务必翻译成 {target_lang}：\n{texts_to_translate}"
        else:
            prompt = f"""
你是一个专业的视频翻译专家。请将以下文本翻译成{target_lang}。
注意：原文可能是任何语言（如意大利语、英语、日语等），请务必统一翻译为{target_lang}。

**术语约束**（必须严格遵守）：
{term_str}

**上下文参考**（前文翻译，保持连贯性）：
{context_text}

**翻译要求**：
1. **准确性**：严格遵循术语表，专有名词和技术词汇必须准确
2. **口语化**：使用自然的口语表达，适合视频字幕阅读
3. **简洁性**：句子长度控制在 20-30 字以内，避免过长
4. **连贯性**：参考上下文，保持逻辑和语气的一致性
5. **文化适配**：使用符合目标语言文化的表达方式
6. **格式规范**：严格按照输入格式返回，每行一个翻译结果，格式为 "数字: 翻译内容"
7. **数量一致**：返回的行数必须正好是 {len(batch)} 行
8. **无多余内容**：不要返回任何解释、注释或碎碎念
9. **容错处理**：如果无法翻译，也请保留行号返回原文

待翻译内容：
{texts_to_translate}
"""
        
        try:
            # API 速率限制检查
            wait_for_llm_api()
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            
            content = response.choices[0].message.content.strip()
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # 建立一个临时字典保存翻译结果，防止行号错乱
            temp_map = {}
            for line in lines:
                # 尝试匹配 "数字: 内容" 或 "数字：内容"
                import re
                match = re.match(r'^(\d+)\s*[:：]\s*(.*)$', line)
                if match:
                    idx = int(match.group(1)) - 1 # 1-based to 0-based
                    temp_map[idx] = match.group(2).strip()
                elif ":" in line:
                    try:
                        idx_str, content_part = line.split(":", 1)
                        idx = int(re.sub(r'\D', '', idx_str)) - 1
                        temp_map[idx] = content_part.strip()
                    except: pass

            for j, seg in enumerate(batch):
                # 优先从 map 取
                translated_text = temp_map.get(j)
                
                # 如果 map 中没有，尝试按顺序取 (仅当行数正好匹配时)
                if not translated_text and len(lines) == len(batch):
                    # 再次尝试从当前行提取内容，即使它不符合 "数字: 内容" 格式
                    curr_line = lines[j]
                    if ":" in curr_line:
                         _, translated_text = curr_line.split(":", 1)
                         translated_text = translated_text.strip()
                    else:
                         translated_text = curr_line
                
                seg["translated_text"] = translated_text if translated_text else seg["text"]
                translated_results.append(seg)
            
            # 更新批次完成进度
            completed_batches += 1
            progress_pct = (completed_batches / total_batches * 100) if total_batches > 0 else 0
            logger.info(f"批次 {batch_num}/{total_batches} 完成 ({progress_pct:.1f}%)")
                    
        except Exception as e:
            logger.error(f"批量翻译失败 (批次 {batch_num}/{total_batches}): {e}")
            for seg in batch:
                seg["translated_text"] = seg["text"]
                translated_results.append(seg)
            # 即使失败也要更新进度
            completed_batches += 1
            progress_pct = (completed_batches / total_batches * 100) if total_batches > 0 else 0
            logger.warning(f"批次 {batch_num}/{total_batches} 失败，使用原文 ({progress_pct:.1f}%)")

    logger.success(f"翻译完成，共 {len(translated_results)} 段")
    return translated_results
