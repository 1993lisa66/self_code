import os
import json
import random
import re
import time
from loguru import logger
from openai import OpenAI
from modules.utils.rate_limiter import wait_for_llm_api, mark_model_overloaded, is_model_overloaded

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
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    whisper_segs = multi_asr_results["whisperx"]
    
    # 工业级：分批处理，每批 30 个片段
    batch_size = 30
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
    
    # 内置默认 prompt 模板（精简版）
    _default_prompt_tpl = (
        "合并多个ASR结果，选最优并修正：\n"
        f"术语表：{term_str}\n"
        "修正标点、大小写、去冗余词(um/uh/you know)、恢复英文标准拼写。\n\n"
        "格式：\"数字: 文本\" 每行一条，共{{count}}行，不解释。\n\n"
        "{{text}}"
    )
    
    try:
        llm_disabled = False  # 一旦遇到余额不足等致命错误，跳过后续所有批次
        fallback_model = config.get('fallback_model') if config else None
        current_model = model_name
        model_sticky = False  # 粘性降级：一旦切到备用模型就保持，每 N 批试探主模型是否恢复
        sticky_check_interval = 8  # 每 8 批检测一次主模型是否恢复
        
        for i in range(0, len(whisper_segs), batch_size):
            batch_num = i // batch_size + 1
            batch = whisper_segs[i:i+batch_size]

            if llm_disabled:
                logger.info(f"跳过批次 {batch_num}/{total_batches}（LLM 已禁用，直接使用原文）")
                for j, seg in enumerate(batch):
                    global_idx = i + j
                    final_results[global_idx] = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                continue

            # 全局过载检测：已确认主模型不可用，直接切备用
            if not model_sticky and is_model_overloaded() and fallback_model:
                logger.info(f"  检测到主模型全局过载，直接使用备用模型: {fallback_model.split('/')[-1]}")
                current_model = fallback_model
                model_sticky = True
            
            # 粘性降级：定期试探主模型是否恢复
            if model_sticky and fallback_model and batch_num % sticky_check_interval == 0:
                logger.info(f"  试探主模型 {model_name.split('/')[-1]} 是否恢复...")
                current_model = model_name
            elif model_sticky and fallback_model:
                current_model = fallback_model

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

            # ── 带重试 + 模型降级 + 粘性 ──
            max_retries = 3
            fused_this_batch = False
            
            for attempt in range(max_retries):
                try:
                    wait_for_llm_api()
                    response = client.chat.completions.create(
                        model=current_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=2000,
                        timeout=120
                    )
                    
                    content = response.choices[0].message.content.strip()
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    
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
                    
                    fused_this_batch = True
                    completed_batches += 1
                    progress_pct = (completed_batches / total_batches * 100) if total_batches > 0 else 0
                    logger.info(f"批次 {batch_num}/{total_batches} 融合完成 ({progress_pct:.1f}%)")
                    break  # 成功，跳出重试
                    
                except Exception as e:
                    err_msg_lower = str(e).lower()
                    # 致命错误
                    if any(kw in err_msg_lower for kw in ('balance', 'insufficient', 'invalid', 'unauthorized', '403', '401')):
                        llm_disabled = True
                        logger.error(f"批次融合致命错误 (批次 {batch_num}/{total_batches}): {e}")
                        logger.warning("检测到致命 API 错误，将跳过后续所有批次")
                        break
                    
                    # 429 限流 → 通知全局过载检测器
                    is_rate_limit = '429' in str(e) or 'rate' in err_msg_lower or 'too busy' in err_msg_lower
                    if is_rate_limit:
                        mark_model_overloaded()
                    if is_rate_limit and attempt < max_retries - 1:
                        # 尝试切换到备用模型
                        if fallback_model and current_model != fallback_model and attempt >= 1:
                            logger.info(f"  主模型受限，切换到备用模型: {fallback_model.split('/')[-1]}")
                            current_model = fallback_model
                            model_sticky = True  # 启用粘性，后续批次也用备用
                        delay = (2 ** (attempt + 1)) * (0.5 + random.random())  # 2~4s, 4~8s
                        model_label = current_model.split('/')[-1]
                        logger.warning(f"  ASR融合限流 [{model_label}]，{delay:.1f}s 后重试 ({attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    
                    logger.error(f"批次融合失败 (批次 {batch_num}/{total_batches}): {e}")
                    break
            
            # 降级：使用原文
            if not fused_this_batch:
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
