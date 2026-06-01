import os
import json
import asyncio
import re
from loguru import logger
from openai import OpenAI

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
    
    # 在循环外加载术语表一次（术语表不变，没必要每批重新读取）
    terminology = {}
    term_file = config.get('terminology_file', 'terminology.json') if config else 'terminology.json'
    if os.path.exists(term_file):
        try:
            with open(term_file, 'r', encoding='utf-8') as f:
                terminology = json.load(f)
        except Exception as e:
            logger.warning(f"加载术语表失败: {e}")
            if term_file != 'terminology.json' and os.path.exists('terminology.json'):
                try:
                    with open('terminology.json', 'r', encoding='utf-8') as f:
                        terminology = json.load(f)
                except:
                    pass
    term_str = json.dumps(terminology, ensure_ascii=False) if terminology else "无特殊术语"
    
    # 内置默认 prompt 模板（极致精简版）
    _default_prompt_tpl = (
        "合并多个ASR结果，选出最准确的版本并优化：\n"
        f"术语表：{term_str}\n"
        "修正标点、大小写；去冗余词(um/uh/you know)；英文恢复标准拼写。\n\n"
        "格式：\"数字: 文本\" 每行一条，共{{count}}行，不解释。\n\n"
        "{{text}}"
    )
    
    try:
        llm_disabled = False  # 一旦遇到余额不足等致命错误，跳过后续所有批次
        for i in range(0, len(whisper_segs), batch_size):
            batch_num = i // batch_size + 1
            batch = whisper_segs[i:i+batch_size]

            if llm_disabled:
                logger.info(f"跳过批次 {batch_num}/{total_batches}（LLM 已禁用，直接使用原文）")
                for j, seg in enumerate(batch):
                    global_idx = i + j
                    final_results[global_idx] = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                continue

            logger.info(f"正在融合 ASR 片段批次: {batch_num}/{total_batches} (片段 {i+1}-{min(i+len(batch), len(whisper_segs))})...")
            
            combined_context = ""
            for j, seg in enumerate(batch):
                global_idx = i + j
                parts = [f"[{j+1}] W: {seg['text']}"]
                for m in ["glm"]:
                    if m in multi_asr_results and global_idx < len(multi_asr_results[m]):
                        parts.append(f"G: {multi_asr_results[m][global_idx]['text']}")
                combined_context += " | ".join(parts) + "\n"

            # 使用外部提示词模板（如果提供），否则使用内置默认模板
            if prompt_template:
                try:
                    prompt = prompt_template.format(
                        text=combined_context,
                        terminology=term_str,
                        count=len(batch)
                    )
                except KeyError:
                    prompt = _default_prompt_tpl.format(text=combined_context, count=len(batch))
            else:
                prompt = _default_prompt_tpl.format(text=combined_context, count=len(batch))

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1200
                )
                
                content = response.choices[0].message.content.strip()
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                # 建立一个临时字典保存融合结果，防止行号错乱
                temp_map = {}
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
                err_msg = str(e)
                logger.error(f"批次融合失败 (批次 {batch_num}/{total_batches}): {e}")
                # 检测致命错误（余额不足、key 无效等），禁用后续 LLM 调用
                if any(kw in err_msg.lower() for kw in ('balance', 'insufficient', 'invalid', 'unauthorized', '403', '401')):
                    llm_disabled = True
                    logger.warning(f"检测到致命 API 错误，将跳过后续所有批次的 LLM 融合，直接使用原文")
                for j, seg in enumerate(batch):
                    global_idx = i + j
                    final_results[global_idx] = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                completed_batches += 1
                progress_pct = (completed_batches / total_batches * 100) if total_batches > 0 else 0
                logger.info(f"批次 {batch_num}/{total_batches} 使用原文 ({progress_pct:.1f}%)")

        logger.success(f"多模型融合修正完成，共 {len(final_results)} 段。")
        return final_results

    except Exception as e:
        logger.error(f"LLM 融合总流程失败: {e}")
        return whisper_segs
