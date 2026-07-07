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

    # 读取第二模型（GLM/SenseVoice）结果，用于投票对比
    glm_segs = multi_asr_results.get("glm", [])
    # 清洗 SenseVoice 标签：<|en|> <|NEUTRAL|> <|Speech|> <|withitn|> 等
    _tag_re = re.compile(r'<\|[^|]*\|>')
    for g in glm_segs:
        g["text"] = _tag_re.sub('', g.get("text", "")).strip()
    has_second_model = bool(glm_segs) and len(glm_segs) == len(whisper_segs)
    if has_second_model:
        logger.info(f"检测到第二模型结果（{len(glm_segs)} 条），将启用双模型投票融合")
    else:
        if glm_segs:
            logger.warning(f"第二模型结果数量不匹配（glm:{len(glm_segs)} vs whisperx:{len(whisper_segs)}），仅单模型修正")
        else:
            logger.info("未检测到第二模型结果，仅对 WhisperX 结果进行 LLM 修正")
    
    # 控制批次：太大容易超出 max_tokens 导致幻觉/截断（输出假"示例"文本）
    batch_size = config.get('batch_size', 15) if config else 15
    final_results = [None] * len(whisper_segs)  # 预分配空间保持顺序
    total_batches = (len(whisper_segs) + batch_size - 1) // batch_size
    
    logger.info(f"总共 {len(whisper_segs)} 个片段，分 {total_batches} 批处理（每批 {batch_size} 个）")
    
    # 在循环外加载术语表一次，批内按需过滤
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
    
    # 内置默认 prompt 模板（双模型投票 / 单模型修正）
    _default_prompt_tpl = {
        False: (
            # 单模型：仅修正
            "请修正以下英文ASR识别结果，修正标点、大小写、去语气词、恢复英文标准拼写：\n"
            "{term_line}"
            "格式：\"数字: 文本\" 每行一条，共{{count}}行，不解释。\n\n"
            "{{text}}"
        ),
        True: (
            # 双模型：投票融合
            "以下有两组ASR识别结果（模型A和模型B），请对比两组结果投票选出最优文本，"
            "合并修正标点、大小写、去语气词、恢复英文标准拼写：\n"
            "{term_line}"
            "格式：\"数字: 文本\" 每行一条，共{{count}}行，不解释。\n\n"
            "{{text}}"
        ),
    }

    # ── LLM 响应中常见的无效注释模式 ──
    _NOTE_PATTERN = re.compile(
        r'[\(\[（]\s*(?:Note|注|说明|注意|提示|Note\s*that)[：:].*?[\)\]）]',
        re.IGNORECASE | re.DOTALL,
    )

    def _clean_fuse_line(text: str) -> str:
        """清洗 LLM 返回行中的注释/碎碎念，返回空字符串表示该行应丢弃。"""
        return _NOTE_PATTERN.sub('', text).strip()

    
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
            
            # ── 内部函数：尝试融合一批（单次 API 调用） ──
            def _try_fuse_one(batch_segs, glm_batch_texts=None):
                """返回 (parsed_map, is_hallucination) 或抛出异常"""
                wait_for_llm_api()
                # 动态构建 prompt（术语按需过滤）
                sub_texts = [seg['text'] for seg in batch_segs]
                sub_term_line = ""
                if terminology:
                    ct = " ".join(sub_texts).lower()
                    rel = {k: v for k, v in terminology.items() if k.lower() in ct}
                    if rel:
                        sub_term_line = f"术语表：{';'.join(f'{k}={v}' for k,v in rel.items())}\n"

                # 构建输入文本：单模型或双模型对比
                if glm_batch_texts and len(glm_batch_texts) == len(batch_segs):
                    lines = []
                    for k in range(len(sub_texts)):
                        lines.append(f"{k+1}: A[{sub_texts[k]}] B[{glm_batch_texts[k]}]")
                    combined = "\n".join(lines)
                else:
                    combined = "\n".join(f"{k+1}:{t}" for k, t in enumerate(sub_texts))

                use_dual = bool(glm_batch_texts and len(glm_batch_texts) == len(batch_segs))
                tpl = _default_prompt_tpl.get(use_dual, _default_prompt_tpl[False])
                if prompt_template:
                    try:
                        sub_prompt = prompt_template.format(text=combined, count=len(batch_segs))
                    except KeyError:
                        sub_prompt = tpl.format(
                            term_line=sub_term_line, text=combined, count=len(batch_segs))
                else:
                    sub_prompt = tpl.format(
                        term_line=sub_term_line, text=combined, count=len(batch_segs))

                input_chars = sum(len(t) for t in sub_texts)
                dynamic_max_tokens = max(3000, int(input_chars * 2.5))
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": sub_prompt}],
                    temperature=0.1,
                    max_tokens=dynamic_max_tokens,
                    timeout=120
                )
                content = response.choices[0].message.content.strip()
                lines = [l.strip() for l in content.split('\n') if l.strip()]

                # 幻觉检测
                _hallucination_kw = ['修正文本示例', '修正示例', '示例文本', '输出示例', '输出格式',
                                    'incomplete', 'provide full context', 'needs more context']
                if len(lines) <= len(batch_segs) and any(kw in content.lower() for kw in _hallucination_kw):
                    return None, True  # is_hallucination

                # 解析结果
                parsed = {}
                for line in lines:
                    m = re.match(r'^(\d+)\s*[:：]\s*(.*)$', line)
                    if m:
                        cleaned = _clean_fuse_line(m.group(2).strip())
                        if cleaned:
                            parsed[int(m.group(1)) - 1] = cleaned
                    elif ":" in line:
                        try:
                            idx_str, cp = line.split(":", 1)
                            cleaned = _clean_fuse_line(cp.strip())
                            if cleaned:
                                parsed[int(re.sub(r'\D', '', idx_str)) - 1] = cleaned
                        except: pass
                for j in range(len(batch_segs)):
                    if j not in parsed and len(lines) == len(batch_segs):
                        if ":" in lines[j]:
                            _, parsed[j] = lines[j].split(":", 1)
                            parsed[j] = parsed[j].strip()
                        else:
                            parsed[j] = lines[j]
                return parsed, False  # success, no hallucination

            # ── 递归拆分融合（幻觉时自动切半重试） ──
            MIN_SUB_BATCH = 5  # 最小子批次，再小直接回退原文
            def _fuse_with_split(batch_segs, base_idx, glm_texts=None, depth=0):
                """融合一批片段，幻觉时拆分重试；返回填充 final_results 的数量"""
                size = len(batch_segs)
                if size <= 1:
                    # 单个片段，直接保留原文
                    final_results[base_idx] = {
                        "start": batch_segs[0]["start"], "end": batch_segs[0]["end"],
                        "text": batch_segs[0]["text"]
                    }
                    return 1

                prefix = f"(d{depth})" if depth > 0 else ""
                try:
                    parsed, hallucinated = _try_fuse_one(batch_segs, glm_texts)
                    if hallucinated:
                        if size <= MIN_SUB_BATCH:
                            logger.warning(
                                f"  {prefix}批次幻觉且已达最小拆分({size}条)，回退原文")
                            for j, seg in enumerate(batch_segs):
                                final_results[base_idx + j] = {
                                    "start": seg["start"], "end": seg["end"],
                                    "text": seg["text"]
                                }
                            return size
                        # 拆半递归（同时拆分 glmm 文本）
                        mid = size // 2
                        glm_left = glm_texts[:mid] if glm_texts else None
                        glm_right = glm_texts[mid:] if glm_texts else None
                        logger.info(
                            f"  {prefix}批次幻觉，拆分重试: {size}→{mid}+{size-mid}")
                        n1 = _fuse_with_split(batch_segs[:mid], base_idx, glm_left, depth + 1)
                        n2 = _fuse_with_split(batch_segs[mid:], base_idx + mid, glm_right, depth + 1)
                        return n1 + n2
                    else:
                        for j, seg in enumerate(batch_segs):
                            fused_text = parsed.get(j)
                            final_results[base_idx + j] = {
                                "start": seg["start"], "end": seg["end"],
                                "text": fused_text if fused_text else seg["text"]
                            }
                        return size
                except Exception as e:
                    err_msg_lower = str(e).lower()
                    if any(kw in err_msg_lower for kw in
                           ('balance', 'insufficient', 'invalid', 'unauthorized', '403', '401')):
                        nonlocal llm_disabled
                        llm_disabled = True
                        logger.error(f"  {prefix}致命 API 错误，后续批次回退原文: {e}")
                    elif '429' in str(e) or 'rate' in err_msg_lower or 'too busy' in err_msg_lower:
                        mark_model_overloaded()
                        logger.warning(f"  {prefix}限流，回退原文")
                    else:
                        logger.warning(f"  {prefix}融合失败，回退原文: {e}")
                    for j, seg in enumerate(batch_segs):
                        final_results[base_idx + j] = {
                            "start": seg["start"], "end": seg["end"],
                            "text": seg["text"]
                        }
                    return size

            # ── 调用拆分融合（传入双模型文本） ──
            glm_batch_texts = None
            if has_second_model:
                glm_batch = glm_segs[i:i + len(batch)]
                glm_batch_texts = [g.get('text', '') for g in glm_batch]
            _fuse_with_split(batch, i, glm_batch_texts)
            logger.info(f"批次 {batch_num}/{total_batches} 完成 ({batch_num*100//total_batches}%)")

        logger.success(f"多模型融合修正完成，共 {len(final_results)} 段。")
        return final_results

    except Exception as e:
        logger.error(f"LLM 融合总流程失败: {e}")
        return whisper_segs
