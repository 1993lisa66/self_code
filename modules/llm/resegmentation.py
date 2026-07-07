import os
import json
import re
from loguru import logger
from openai import OpenAI
from modules.utils.rate_limiter import wait_for_llm_api

def semantic_resegment(fused_results, config=None, prompt_template=None):
    """
    使用 LLM 对 ASR 融合后的片段进行语义重切分（滑窗重叠模式）。
    
    将断开的句子合并，并在自然的停顿点（句号、问号）处切分。
    
    滑窗策略：
    - 每批处理 batch_size=15 条，但窗口滑动步长 stride=12
    - 相邻窗口重叠 3 条，重叠区域的合并方案取在更多窗口中一致的结果
    - 彻底消除了旧版批次边界导致的"永久断层"问题
    """
    if not fused_results:
        return []

    logger.info("开始语义重切分（滑窗重叠模式）...")

    api_key = config.get('api_key') if config else None
    if not api_key or "your-openai-api-key" in api_key:
        logger.warning("未配置有效的 API Key，跳过重切分。")
        return fused_results

    stage_cfg = config.get('resegmenter', {}) if config else {}

    # ── 用户主动跳过 ──
    if stage_cfg.get('skip', False):
        logger.info("重切分已关闭（skip_resegment=true），保留原始片段")
        return fused_results

    # ── 启发式跳过：大部分片段已以句号/问号/感叹号结尾，说明 ASR 切分已经够好 ──
    _sentence_enders = re.compile(r'[.?!。？！]\s*$')
    _ended = sum(1 for seg in fused_results
                 if _sentence_enders.search((seg.get('text') or '').strip()))
    _completion_ratio = _ended / len(fused_results) if fused_results else 0
    _skip_threshold = stage_cfg.get('auto_skip_threshold', 0.70)  # 70% 已完成则跳过
    if _completion_ratio >= _skip_threshold:
        logger.info(
            f"跳过重切分：{_ended}/{len(fused_results)} ({_completion_ratio:.0%}) 片段已以标点结尾，"
            f"无需再合并"
        )
        return fused_results
    logger.info(f"  仅 {_ended}/{len(fused_results)} ({_completion_ratio:.0%}) 片段已完整，需要重切分")

    model_name = stage_cfg.get('model', config.get('model', 'deepseek-ai/DeepSeek-V3'))
    base_url = config.get('api_base', 'https://api.siliconflow.cn/v1')
    request_timeout = stage_cfg.get('request_timeout', config.get('default_request_timeout', 120))
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=max(request_timeout + 30, 120.0))

    temperature = stage_cfg.get('temperature', 0.1)
    max_tokens = stage_cfg.get('max_tokens', 4096)
    max_retries = stage_cfg.get('max_retries', 1)
    api_timeout = stage_cfg.get('request_timeout', config.get('default_request_timeout', 120))
    circuit_breaker_limit = stage_cfg.get('circuit_breaker_windows', 3)

    # 批次大小和重叠：加大 batch_size、缩小重叠=更少的窗口=更快完成
    batch_size = stage_cfg.get('batch_size', 20)   # 15→20，每窗口处理更多片段
    overlap = stage_cfg.get('overlap', 2)           # 3→2，相邻窗口重叠更少
    stride = batch_size - overlap
    
    # ── 收集每个原始片段在多次重切分中的归属方案 ──
    # segment_votes[global_idx] = [group_id_1, group_id_2, ...]
    # group_id = (merged_start, merged_end)，表示该片段被合并到的范围
    segment_plans = []  # [(start_global_idx, end_global_idx, merged_text), ...]
    covered_ranges = []  # 记录每个片段已被哪些方案覆盖
    
    total_windows = max(1, (len(fused_results) - batch_size) // stride + 1) if len(fused_results) > batch_size else 1
    logger.info(f"  共 {len(fused_results)} 个片段，分 {total_windows} 个滑窗处理 (batch_size={batch_size}, stride={stride})")
    
    consecutive_failures = 0  # 熔断计数器

    for w in range(0, len(fused_results), stride):
        window_num = w // stride + 1
        batch = fused_results[w:w + batch_size]
        logger.info(f"  窗口 {window_num}/{total_windows}: 处理片段 {w+1}-{w+len(batch)}")
        if not batch:
            break
        
        # 准备上下文
        combined_text = ""
        for j, seg in enumerate(batch):
            combined_text += f"[{j+1}] {seg['text']}\n"
        
        if prompt_template:
            try:
                prompt = prompt_template.format(text=combined_text, count=len(batch))
            except Exception:
                prompt = f"请重切分以下文本：\n{combined_text}"
        else:
            prompt = (
                "Merge broken ASR fragments into complete sentences."
                "Split at periods, question marks, exclamation marks.\n"
                "Do NOT add, remove, or change any words.\n"
                "Output format: [start-end] sentence text\n"
                f"Cover indices 1 to {len(batch)}.\n\n"
                f"{combined_text}"
            )

        window_got_valid = False
        retry_count = 0
        while retry_count <= max_retries:
            try:
                retry_prompt = prompt
                if retry_count > 0:
                    retry_prompt = (
                        "Combine these ASR fragments into complete sentences.\n"
                        "Rules:\n"
                        "- Keep ALL original words exactly as they are.\n"
                        "- Split sentences at periods, question marks, or exclamation marks.\n"
                        "- Output one line per sentence in this format:\n"
                        "  [start_num-end_num] the complete sentence\n"
                        f"- Cover every index from 1 to {len(batch)}.\n\n"
                        f"{combined_text}"
                    )

                wait_for_llm_api()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": retry_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=api_timeout
                )
                
                content = response.choices[0].message.content.strip()
                lines = [l.strip() for l in content.split('\n') if l.strip()]
                
                matched_count = 0
                bad_lines = []
                for line in lines:
                    match = re.match(r'^\[(\d+)(?:-(\d+))?\]\s*(.*)$', line)
                    if match:
                        start_idx = int(match.group(1)) - 1
                        end_idx = int(match.group(2)) - 1 if match.group(2) else start_idx
                        text = match.group(3).strip()
                        if not text:
                            bad_lines.append(line)
                            continue
                        
                        global_start = w + max(0, min(start_idx, len(batch) - 1))
                        global_end = w + max(0, min(end_idx, len(batch) - 1))
                        global_start = max(0, min(global_start, len(fused_results) - 1))
                        global_end = max(0, min(global_end, len(fused_results) - 1))
                        
                        segment_plans.append((global_start, global_end, text))
                        covered_ranges.append((global_start, global_end))
                        matched_count += 1
                    else:
                        bad_lines.append(line)
                
                if matched_count > 0:
                    window_got_valid = True

                # 有效行太少（< batch 1/3），且未达最大重试次数 → 重试
                min_valid = max(1, len(batch) // 3)
                if matched_count >= min_valid or retry_count >= max_retries:
                    if bad_lines:
                        sample_count = min(3, len(bad_lines))
                        for i in range(sample_count):
                            sample = bad_lines[i]
                            if len(sample) > 120:
                                sample = sample[:117] + "..."
                            logger.warning(f"重切分输出格式不匹配: {sample}")
                        if len(bad_lines) > sample_count:
                            logger.warning(
                                f"  ... 另有 {len(bad_lines) - sample_count} 行格式不匹配 "
                                f"(共 {len(bad_lines)}/{len(lines)} 行不匹配，{matched_count} 行有效)"
                            )
                    break
                else:
                    retry_count += 1
                    logger.warning(
                        f"  窗口 {window_num}: 仅 {matched_count}/{len(lines)} 行有效，"
                        f"重试 {retry_count}/{max_retries}..."
                    )
                    continue
                    
            except Exception as e:
                error_msg = str(e)
                # 超时 = API 拥塞，重试也没用，直接降级
                is_timeout = 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower()
                
                if is_timeout:
                    logger.warning(f"窗口 {window_num}/{total_windows} 请求超时（{api_timeout}s），跳过重试，保留原始片段")
                    for j, seg in enumerate(batch):
                        global_idx = w + j
                        if global_idx < len(fused_results):
                            segment_plans.append((global_idx, global_idx, seg['text']))
                            covered_ranges.append((global_idx, global_idx))
                    break
                
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"窗口 {window_num}/{total_windows} 重切分失败: {e}")
                    for j, seg in enumerate(batch):
                        global_idx = w + j
                        if global_idx < len(fused_results):
                            segment_plans.append((global_idx, global_idx, seg['text']))
                            covered_ranges.append((global_idx, global_idx))
                    break
                else:
                    logger.warning(f"  窗口 {window_num}: API 异常，重试 {retry_count}/{max_retries}: {e}")
                    continue

        # ── 熔断检查 ──
        if window_got_valid:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= circuit_breaker_limit:
                logger.warning(
                    f"熔断触发：连续 {consecutive_failures} 个窗口（#{window_num - consecutive_failures + 1}~#{window_num}）"
                    f"全部无有效输出，跳过剩余 {total_windows - window_num} 个窗口，保留原始片段"
                )
                # 剩余片段全部保留为独立条目
                for remaining_w in range(w + stride, len(fused_results), stride):
                    remaining_batch = fused_results[remaining_w:remaining_w + batch_size]
                    for j, seg in enumerate(remaining_batch):
                        global_idx = remaining_w + j
                        if global_idx < len(fused_results):
                            segment_plans.append((global_idx, global_idx, seg['text']))
                            covered_ranges.append((global_idx, global_idx))
                break

    # ── 冲突消解：对每个全局索引，选择更合理的合并方案 ──
    if not segment_plans:
        return fused_results
    
    # 按起始位置排序
    segment_plans.sort(key=lambda x: (x[0], x[1]))
    
    # 贪心选择：从前往后，选择覆盖当前位置的最优方案
    new_segments = []
    pos = 0
    n = len(fused_results)
    
    while pos < n:
        # 找到所有 start <= pos 的候选方案
        candidates = [(gs, ge, t) for gs, ge, t in segment_plans if gs <= pos <= ge]
        if not candidates:
            # 没有覆盖当前位置的方案，用原始片段
            seg = fused_results[pos]
            new_segments.append({
                "start": seg['start'],
                "end": seg['end'],
                "text": seg['text']
            })
            pos += 1
            continue
        
        # 优先选择结束位置最远（合并范围最大）的方案
        best = max(candidates, key=lambda x: x[1])
        
        # 限制合并范围，避免过度合并（单次最多合并 8 条）
        merge_limit = min(best[1], pos + 8)
        effective_end = min(best[1], merge_limit)
        
        new_segments.append({
            "start": fused_results[best[0]]['start'],
            "end": fused_results[effective_end]['end'],
            "text": best[2]
        })
        pos = effective_end + 1

    logger.success(
        f"语义重切分完成（滑窗重叠），片段数: {len(fused_results)} → {len(new_segments)}"
    )
    return new_segments
