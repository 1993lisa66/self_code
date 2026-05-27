import os
import json
import re
from loguru import logger
from openai import OpenAI
from modules.utils.rate_limiter import wait_for_llm_api

def semantic_resegment(fused_results, config=None, prompt_template=None):
    """
    使用 LLM 对 ASR 融合后的片段进行语义重切分。
    将断开的句子合并，并在自然的停顿点（如句号、问号）处切分。
    """
    if not fused_results:
        return []

    logger.info("开始语义重切分...")

    # 从配置读取 API Key
    api_key = config.get('api_key') if config else None
    if not api_key or "your-openai-api-key" in api_key:
        logger.warning("未配置有效的 API Key，跳过重切分。")
        return fused_results

    model_name = config.get('model', 'deepseek-ai/DeepSeek-V3')
    base_url = config.get('api_base', 'https://api.siliconflow.cn/v1')
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 批处理：每 15 个片段一组进行重切分
    batch_size = 15
    new_segments = []
    
    for i in range(0, len(fused_results), batch_size):
        batch = fused_results[i:i+batch_size]
        
        # 准备上下文：包含索引、文本和原始时长
        combined_text = ""
        for j, seg in enumerate(batch):
            combined_text += f"[{j+1}] {seg['text']}\n"
            
        # 使用模板格式化，如果不存在则使用默认
        if prompt_template:
            try:
                prompt = prompt_template.format(text=combined_text, count=len(batch))
            except Exception:
                prompt = f"请重切分以下文本：\n{combined_text}"
        else:
            prompt = f"""
你是一个专业的视频字幕专家。下面是一段 ASR 识别出的文本片段，由于语音识别的限制，很多句子在中间被切断了。
请根据语义，将这些片段重新组合成自然、完整的句子。

规则：
1. 必须保留所有原文内容，不得删减或增加。
2. 将断开的句子合并，并在自然语意结束处（如句号、问号、感叹号）断句。
3. 返回格式必须为每行一个完整句子，且每行开头保留对应的原始片段索引范围，格式为 "[起始索引-结束索引] 句子内容"。
   例如："[1-2] 这是合并后的第一句话。" 表示这句话涵盖了原始的第 1 和第 2 个片段。
   如果一句话只对应一个片段，格式为 "[1-1] 句子内容"。
4. 确保覆盖所有的原始索引（从 1 到 {len(batch)}）。

待处理文本：
{combined_text}
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
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            
            for line in lines:
                # 匹配 [1-2] 文本 或 [1-1] 文本
                match = re.match(r'^\[(\d+)-(\d+)\]\s*(.*)$', line)
                if match:
                    start_idx = int(match.group(1)) - 1
                    end_idx = int(match.group(2)) - 1
                    text = match.group(3).strip()
                    
                    # 边界检查
                    start_idx = max(0, min(start_idx, len(batch)-1))
                    end_idx = max(0, min(end_idx, len(batch)-1))
                    
                    # 计算新的时间轴
                    new_start = batch[start_idx]['start']
                    new_end = batch[end_idx]['end']
                    
                    new_segments.append({
                        "start": new_start,
                        "end": new_end,
                        "text": text
                    })
                else:
                    logger.warning(f"重切分输出格式不匹配: {line}")
                    
        except Exception as e:
            logger.error(f"批次重切分失败: {e}")
            # 失败则回退到原始片段
            new_segments.extend(batch)

    logger.success(f"语义重切分完成，片段数: {len(fused_results)} -> {len(new_segments)}")
    return new_segments
