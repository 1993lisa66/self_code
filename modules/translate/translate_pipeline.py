import os
import json
import random
import re
import time
import shutil
from loguru import logger
from openai import OpenAI
from modules.utils.rate_limiter import wait_for_llm_api, mark_model_overloaded, is_model_overloaded


def _load_terminology(config=None):
    """统一加载术语表：优先使用 config 中的 terminology_file 路径"""
    terminology = {}
    # 1. 从 config 中获取指定的术语表路径
    term_file = config.get('terminology_file') if config else None
    if term_file and os.path.exists(term_file):
        try:
            with open(term_file, 'r', encoding='utf-8') as f:
                terminology = json.load(f)
            logger.info(f"已加载批次术语表: {term_file} ({len(terminology)} 条)")
            return terminology
        except Exception as e:
            logger.warning(f"加载指定术语表失败 ({term_file}): {e}")
    # 2. 回退到全局术语表
    if os.path.exists("terminology.json"):
        try:
            with open("terminology.json", 'r', encoding='utf-8') as f:
                terminology = json.load(f)
        except:
            pass
    return terminology


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
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    # 加载术语表并转换为字符串（仅在进化后重新加载）
    terminology = _load_terminology(config)
    term_str = json.dumps(terminology, ensure_ascii=False) if terminology else "无"
    
    translated_results = []
    
    # 工业级：增大批次大小以减少 API 调用次数
    batch_size = 30  # 每批 30 个片段
    total_batches = (len(fused_results) + batch_size - 1) // batch_size
    completed_batches = 0
    last_evolved_batch = 0  # 追踪上次进化的批次号
    evolve_every = 12  # 每 N 批进行一次中期术语进化（合理频率，避免过度消耗）
    evolve_disabled = False  # 收敛后停止进化，节省 token
    
    # 粘性降级：主模型限流后切换到备用，后续批次直接用备用，定期试探主模型
    fallback_model = config.get('fallback_model') if config else None
    current_model = model_name
    model_sticky = False
    sticky_check_interval = 8  # 每 8 批试探一次主模型
    
    logger.info(f"总共 {len(fused_results)} 个片段，分 {total_batches} 批处理（每批 {batch_size} 个）")
    
    for i in range(0, len(fused_results), batch_size):
        batch_num = i // batch_size + 1
        batch = fused_results[i:i+batch_size]
        
        # 全局过载检测：已确认主模型不可用，直接切备用
        if not model_sticky and is_model_overloaded() and fallback_model:
            logger.info(f"  检测到主模型全局过载，直接使用备用模型: {fallback_model.split('/')[-1]}")
            current_model = fallback_model
            model_sticky = True
        
        # 粘性降级：定期试探主模型
        if model_sticky and fallback_model and batch_num % sticky_check_interval == 0:
            logger.info(f"  试探主模型 {model_name.split('/')[-1]} 是否恢复...")
            current_model = model_name
        elif model_sticky and fallback_model:
            current_model = fallback_model
        
        # 显示当前批次进度
        logger.info(f"正在翻译批次 {batch_num}/{total_batches} (片段 {i+1}-{min(i+len(batch), len(fused_results))})...")
        
        # 构造批量翻译提示词
        texts_to_translate = "\n".join([f"{j+1}: {seg['text']}" for j, seg in enumerate(batch)])
        
        # 提取前文作为上下文参考（最近 5 个片段）— 空则跳过，不发送"无"
        context_start = max(0, i - 5)
        context_segments = fused_results[context_start:i]
        context_text = ""
        context_block = ""
        if context_segments:
            context_text = "\n".join([f"{idx+1}: {seg.get('translated_text', seg['text'])}" 
                                     for idx, seg in enumerate(context_segments)])
            context_block = f"\n**上下文参考**（前文翻译，保持连贯性）：\n{context_text}"
        
        if prompt_template:
            # 安全格式化：只替换模板中实际存在的占位符
            try:
                prompt = prompt_template.replace('{target_lang}', target_lang)
                prompt = prompt.replace('{text}', texts_to_translate)
                prompt = prompt.replace('{terminology}', term_str)
                prompt = prompt.replace('{context}', context_text)
            except Exception:
                prompt = (
                    f"请将以下英文翻译成中文，口语化、适合字幕。"
                    f"原文一定是外语，必须翻译成中文。"
                    f"参考术语表：{term_str}\n"
                    f"{context_block}\n\n"
                    f"格式: 数字: 翻译 每行一条, 共{len(batch)}行 不解释.\n\n"
                    f"{texts_to_translate}"
                )
        else:
            term_block = f"\n术语表：{term_str}" if terminology else ""
            prompt = (
                f"翻译成{target_lang}，口语化适合字幕。"
                f"{term_block}{context_block}\n"
                f"格式：\"数字: 翻译\" 每行一条，共{len(batch)}行 不解释。\n\n"
                f"{texts_to_translate}"
            )
        
        # ── 带重试 + 模型降级 + 粘性的 API 调用 ──
        max_retries = 3
        base_delay = 3  # 初始等待秒数
        translated_this_batch = False

        for attempt in range(max_retries):
            # 第 2 次失败后切换备用模型，并启用粘性
            if attempt >= 2 and fallback_model and current_model != fallback_model:
                logger.info(f"  └ 主模型 {current_model} 持续限流，切换到备用模型: {fallback_model}")
                current_model = fallback_model
                model_sticky = True  # 后续批次直接用备用模型

            try:
                wait_for_llm_api()
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=2000,
                    timeout=120
                )

                content = response.choices[0].message.content.strip()
                lines = [line.strip() for line in content.split('\n') if line.strip()]

                # 建立一个临时字典保存翻译结果，防止行号错乱
                temp_map = {}
                for line in lines:
                    # 尝试匹配 "数字: 内容" 或 "数字：内容"
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

                translated_this_batch = True
                break  # 成功后跳出重试循环

            except Exception as e:
                err_msg = str(e).lower()
                # 检测致命错误（余额不足、key 无效等），立即返回
                if any(kw in err_msg for kw in ('balance', 'insufficient', 'invalid', 'unauthorized', '403', '401')):
                    logger.error(f"翻译批次 {batch_num}/{total_batches} API 错误: {e}")
                    logger.warning("检测到致命 API 错误（如余额不足），跳过后续所有翻译批次")
                    # 剩余片段使用原文
                    remaining_start = i + len(batch)
                    for seg in fused_results[remaining_start:]:
                        seg["translated_text"] = seg["text"]
                        translated_results.append(seg)
                    translated_this_batch = True  # 标记为已处理，跳出外层
                    break

                # 429 限流 / 服务繁忙 → 通知全局过载检测器
                is_rate_limit = '429' in str(e) or 'rate' in err_msg or 'too busy' in err_msg
                if is_rate_limit:
                    mark_model_overloaded()
                if is_rate_limit and attempt < max_retries - 1:
                    delay = (base_delay ** (attempt + 1)) * (0.5 + random.random())  # 加抖动: 3±1.5s, 9±4.5s...
                    model_label = current_model.split('/')[-1]
                    logger.warning(f"翻译限流 [{model_label}]，{delay:.1f}s 后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue

                # 非限流错误，或重试次数耗尽
                logger.error(f"批量翻译失败 (批次 {batch_num}/{total_batches}): {e}")
                break

        # ── 如果重试耗尽仍未成功，降级使用原文 ──
        if not translated_this_batch:
            for seg in batch:
                seg["translated_text"] = seg["text"]
                translated_results.append(seg)
            completed_batches += 1
            progress_pct = (completed_batches / total_batches * 100) if total_batches > 0 else 0
            logger.warning(f"批次 {batch_num}/{total_batches} 失败，使用原文 ({progress_pct:.1f}%)")
        else:
            # 更新批次完成进度
            completed_batches += 1
            progress_pct = (completed_batches / total_batches * 100) if total_batches > 0 else 0
            logger.info(f"批次 {batch_num}/{total_batches} 完成 ({progress_pct:.1f}%)")
        
        # ── 中期术语表进化（每 evolve_every 批触发一次，收敛后自动停止）──
        if (not evolve_disabled and config and config.get('api_key') 
                and config.get('terminology_file') and translated_results 
                and batch_num % evolve_every == 0 and batch_num != last_evolved_batch):
            try:
                prev_count = len(terminology) if terminology else 0
                logger.info(f"  └ 触发术语表中途进化（第 {batch_num}/{total_batches} 批）")
                result = evolve_terminology(translated_results, config=config)
                last_evolved_batch = batch_num
                # 立即重新加载术语表并更新缓存
                reloaded = _load_terminology(config)
                new_count = len(reloaded) if reloaded else 0
                if new_count != prev_count:
                    logger.info(f"  └ 术语表已刷新: {prev_count} → {new_count} 条")
                terminology = reloaded
                term_str = json.dumps(terminology, ensure_ascii=False) if terminology else "无"
                # 收敛检测：术语表已较丰富且本轮无新增，停止后续进化
                if new_count > 5 and new_count == prev_count:
                    evolve_disabled = True
                    logger.info(f"  └ 术语表已收敛（{new_count} 条），停止后续进化")
            except Exception as e:
                logger.warning(f"  └ 中期术语进化跳过（不影响主流程）: {e}")

    # ── 最终术语表进化（仅在未收敛且最后一轮未触发中期进化时执行）──
    if (not evolve_disabled and config and config.get('api_key') 
            and config.get('terminology_file') and translated_results 
            and last_evolved_batch < total_batches):
        try:
            prev_count = len(terminology) if terminology else 0
            logger.info("触发术语表最终进化...")
            result = evolve_terminology(translated_results, config=config)
            reloaded = _load_terminology(config)
            new_count = len(reloaded) if reloaded else 0
            if new_count != prev_count:
                logger.info(f"术语表最终更新: {prev_count} → {new_count} 条")
        except Exception as e:
            logger.warning(f"最终术语进化跳过（不影响主流程）: {e}")

    logger.success(f"翻译完成，共 {len(translated_results)} 段")
    return translated_results


def evolve_terminology(translated_results, config=None, prompt_template=None):
    """
    LLM 驱动的术语表进化。
    分析翻译结果中的原文→译文对照，发现新术语或优化现有术语定义，
    然后将新术语合并保存回批次术语表。
    
    每次运行都会提升后续翻译/TTS的术语一致性和流畅度。
    """
    if not translated_results or not config:
        return {}
    
    api_key = config.get('api_key')
    if not api_key:
        return {}
    
    term_file = config.get('terminology_file')
    if not term_file:
        logger.info("未指定术语表文件路径，跳过术语进化")
        return {}
    
    model_name = config.get('model', 'deepseek-ai/DeepSeek-V3')
    base_url = config.get('api_base', 'https://api.siliconflow.cn/v1')
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    
    # 加载现有术语表
    existing_terms = _load_terminology(config)
    
    # 采样翻译结果（最多取 30 段代表性样本，60条太大没必要）
    sample_size = min(30, len(translated_results))
    step = max(1, len(translated_results) // sample_size)
    samples = translated_results[::step][:sample_size]
    
    # 构造原文→译文对照
    pairs_text = "\n".join([
        f"{i+1}. EN: {s.get('text', '')}\n   ZH: {s.get('translated_text', '')}"
        for i, s in enumerate(samples) if s.get('text') and s.get('translated_text')
    ])
    
    existing_terms_str = json.dumps(existing_terms, ensure_ascii=False) if existing_terms else "（空）"
    
    prompt = (
        f"分析翻译对照，找出遗漏或不当的专业术语。\n\n"
        f"现有术语表：{existing_terms_str}\n\n"
        f"原文→译文样本：\n{pairs_text}\n\n"
        "返回JSON：{{\"英文术语\":\"中文翻译\"}}，只包含需新增/修正的条目。"
        "保留正确的、删除错误的、不解释。"
    )
    
    try:
        wait_for_llm_api()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
            timeout=120
        )
        content = response.choices[0].message.content.strip()
        
        # 提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            new_terms = json.loads(json_match.group())
            
            # 合并：新术语表 + 保留旧的不冲突条目
            merged = {}
            for k, v in existing_terms.items():
                if k not in new_terms:
                    merged[k] = v
            merged.update(new_terms)
            
            # 保存回文件
            os.makedirs(os.path.dirname(term_file), exist_ok=True)
            with open(term_file, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            
            added = set(new_terms.keys()) - set(existing_terms.keys())
            modified = {k for k in new_terms if k in existing_terms and new_terms[k] != existing_terms[k]}
            removed = set(existing_terms.keys()) - set(merged.keys())
            
            logger.success(f"术语表已进化: {term_file}")
            if added:
                logger.info(f"  新增术语 ({len(added)}): {', '.join(sorted(added))}")
            if modified:
                logger.info(f"  修正术语 ({len(modified)}): {', '.join(sorted(modified))}")
            if removed:
                logger.info(f"  移除术语 ({len(removed)}): {', '.join(sorted(removed))}")
            
            return merged
        else:
            logger.warning("术语进化：LLM 返回中未找到 JSON，跳过")
            return existing_terms
            
    except json.JSONDecodeError as e:
        logger.warning(f"术语进化：JSON 解析失败: {e}")
        return existing_terms
    except Exception as e:
        logger.warning(f"术语进化失败（不影响主流程）: {e}")
        return existing_terms


def _should_evolve_prompt(prompt_path, cooldown_hours=6):
    """
    检查提示词是否应该进化（基于冷却期）。
    在 prompt_path 同目录下保存 .last_evolved.json 记录上次进化时间和内容哈希。
    """
    track_path = os.path.join(os.path.dirname(prompt_path), ".last_evolved.json")
    try:
        if os.path.exists(track_path):
            with open(track_path, 'r', encoding='utf-8') as f:
                track = json.load(f)
            prompt_name = os.path.basename(prompt_path)
            if prompt_name in track:
                last_time = track[prompt_name].get('time', 0)
                elapsed_hours = (time.time() - last_time) / 3600
                if elapsed_hours < cooldown_hours:
                    logger.info(f"提示词进化 [{prompt_name}]：冷却中（{elapsed_hours:.1f}h < {cooldown_hours}h），跳过")
                    return False
    except Exception:
        pass
    return True


def _record_evolve_prompt(prompt_path):
    """记录提示词进化时间"""
    track_path = os.path.join(os.path.dirname(prompt_path), ".last_evolved.json")
    try:
        track = {}
        if os.path.exists(track_path):
            with open(track_path, 'r', encoding='utf-8') as f:
                track = json.load(f)
        prompt_name = os.path.basename(prompt_path)
        track[prompt_name] = {'time': time.time()}
        with open(track_path, 'w', encoding='utf-8') as f:
            json.dump(track, f)
    except Exception:
        pass


def evolve_prompt(prompt_path, samples, step_name, config=None):
    """
    LLM 驱动的提示词自动进化。
    分析实际输入→输出结果的质量，让 LLM 审视并优化提示词，
    使后续运行产生更准确、更一致的输出。

    Args:
        prompt_path: 提示词文件完整路径
        samples:  输入输出样本列表，每项为 {'input': str, 'output': str}
        step_name: 步骤描述（如 "ASR文本修正"、"字幕翻译"）
        config:   LLM 配置字典

    Returns:
        bool: 是否成功进化
    """
    if not os.path.exists(prompt_path) or not samples or not config:
        return False

    # 冷却期检查：同一提示词短时间内不重复进化
    if not _should_evolve_prompt(prompt_path):
        return False

    api_key = config.get('api_key')
    if not api_key:
        return False

    # 只取有实质差异的样本（至少 5 个才触发进化，避免过拟合）
    meaningful = [s for s in samples if s.get('input') and s.get('output') and s['input'].strip() != s['output'].strip()]
    if len(meaningful) < 5:
        logger.info(f"提示词进化 [{step_name}]：有效样本不足（{len(meaningful)} 个），跳过")
        return False

    # 读取当前提示词
    with open(prompt_path, 'r', encoding='utf-8') as f:
        current_prompt = f.read().strip()

    if not current_prompt:
        logger.warning(f"提示词进化 [{step_name}]：当前文件为空，跳过")
        return False

    model_name = config.get('model', 'deepseek-ai/DeepSeek-V3')
    base_url = config.get('api_base', 'https://api.siliconflow.cn/v1')
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    # 去重后最多取 15 个样本
    seen = set()
    unique_samples = []
    for s in meaningful:
        key = (s['input'].strip()[:60], s['output'].strip()[:60])
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)
    sampled = unique_samples[:15]

    sample_text = "\n---\n".join([
        f"输入:\n{s['input'][:300]}\n输出:\n{s['output'][:300]}"
        for s in sampled
    ])

    evolution_prompt = f"""你是提示词工程师。请根据以下样本迭代优化提示词。

**当前提示词**（"{step_name}" 步骤）：
```
{current_prompt}
```

**输入→输出样本**：
{sample_text}

**优化准则**：
1. 分析输出的不足（术语错误、格式混乱、漏翻），追溯提示词根因
2. 修补模糊指令，将隐式规则显式化
3. 删除无效或矛盾的指令

**硬性约束**：
- 必须完整保留所有占位符变量（花括号词，如 {{{{text}}}}、{{{{terminology}}}}、{{{{count}}}}）
- 不要添加不存在的占位符
- 如果提示词已足够好，返回原文
- 只返回优化后的提示词，不要解释

优化后的提示词："""

    try:
        wait_for_llm_api()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": evolution_prompt}],
            temperature=0.25,
            max_tokens=1500,
            timeout=120
        )
        optimized = response.choices[0].message.content.strip()

        # 去除可能的 markdown 代码块包裹
        if optimized.startswith("```"):
            lines = optimized.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            optimized = "\n".join(lines).strip()

        # 比对是否有实质变化
        if optimized == current_prompt:
            logger.info(f"提示词进化 [{step_name}]：无需改动，当前版本已最优")
            _record_evolve_prompt(prompt_path)
            return True

        # 安全检查：优化后的提示词必须包含原提示词的关键占位符
        import re as _re
        orig_placeholders = set(_re.findall(r'\{(\w+)\}', current_prompt))
        new_placeholders = set(_re.findall(r'\{(\w+)\}', optimized))
        missing = orig_placeholders - new_placeholders
        if missing:
            logger.warning(f"提示词进化 [{step_name}]：优化后丢失占位符 {missing}，已拒绝")
            return False

        # 备份旧版本
        backup_path = prompt_path + ".bak"
        try:
            shutil.copy2(prompt_path, backup_path)
        except Exception:
            pass

        # 写入新版本
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(optimized)

        logger.success(f"提示词已进化 [{step_name}]: {prompt_path}")
        logger.info(f"  旧版本备份: {backup_path}")
        _record_evolve_prompt(prompt_path)
        return True

    except Exception as e:
        logger.warning(f"提示词进化失败 [{step_name}]（不影响主流程）: {e}")
        return False
