import os
import json
import re
import shutil
from loguru import logger
from openai import OpenAI


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
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 加载术语表
    terminology = _load_terminology(config)

    translated_results = []
    
    # 工业级：增大批次大小以减少 API 调用次数
    batch_size = 20  # 从 10 增加到 20，减少一半的 API 调用
    total_batches = (len(fused_results) + batch_size - 1) // batch_size
    completed_batches = 0
    last_evolved_batch = 0  # 追踪上次进化的批次号
    evolve_every = 5  # 每 N 批进行一次中期术语进化
    
    logger.info(f"总共 {len(fused_results)} 个片段，分 {total_batches} 批处理（每批 {batch_size} 个）")
    
    for i in range(0, len(fused_results), batch_size):
        batch_num = i // batch_size + 1
        batch = fused_results[i:i+batch_size]
        
        # 每批开始时重新加载术语表（上一批进化可能写入了新术语）
        if batch_num > 1:
            reloaded = _load_terminology(config)
            if len(reloaded) != len(terminology):
                logger.info(f"  └ 术语表已刷新: {len(terminology)} → {len(reloaded)} 条")
            terminology = reloaded
        
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
        
        # ── 中期术语表进化（每 evolve_every 批触发一次）──
        if (config and config.get('api_key') and config.get('terminology_file') 
                and translated_results and batch_num % evolve_every == 0 
                and batch_num != last_evolved_batch):
            try:
                logger.info(f"  └ 触发术语表中途进化（第 {batch_num}/{total_batches} 批）")
                evolve_terminology(translated_results, config=config)
                last_evolved_batch = batch_num
                # 立即重新加载，让下一批使用最新的术语表
                reloaded = _load_terminology(config)
                if len(reloaded) != len(terminology):
                    logger.info(f"  └ 术语表已刷新: {len(terminology)} → {len(reloaded)} 条")
                terminology = reloaded
            except Exception as e:
                logger.warning(f"  └ 中期术语进化跳过（不影响主流程）: {e}")

    # ── 最终术语表进化（如果最后一轮没有触发中期进化）──
    if (config and config.get('api_key') and config.get('terminology_file') 
            and translated_results and last_evolved_batch < total_batches):
        try:
            logger.info("触发术语表最终进化...")
            evolve_terminology(translated_results, config=config)
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
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 加载现有术语表
    existing_terms = _load_terminology(config)
    
    # 采样翻译结果（最多取 60 段代表性样本）
    sample_size = min(60, len(translated_results))
    step = max(1, len(translated_results) // sample_size)
    samples = translated_results[::step][:sample_size]
    
    # 构造原文→译文对照
    pairs_text = "\n".join([
        f"{i+1}. EN: {s.get('text', '')}\n   ZH: {s.get('translated_text', '')}"
        for i, s in enumerate(samples) if s.get('text') and s.get('translated_text')
    ])
    
    existing_terms_str = json.dumps(existing_terms, ensure_ascii=False) if existing_terms else "（空）"
    
    prompt = f"""
你是一个专业的金融/技术翻译术语专家。请分析以下视频字幕的原文→译文对照，找出被遗漏或翻译不当的专业术语。

**现有术语表**：
{existing_terms_str}

**原文→译文样本**：
{pairs_text}

**任务**：
1. 从样本中识别所有专业术语（金融、交易、技术等领域），尤其是现有术语表中没有的
2. 检查现有术语表中每个术语的翻译是否准确、是否在样本中出现
3. 如果有术语翻译不当或不够精准，给出修正建议
4. 用 JSON 格式输出最终术语表，格式为 {{"英文术语": "中文翻译"}}

**要求**：
- 只返回一个 JSON 对象，不要任何解释
- 保留现有术语表中正确的条目
- 新增遗漏的关键术语
- 修正翻译不准确的条目
- 删除完全不相关或错误的条目
- JSON 中所有 key 为英文原词，value 为中文翻译
"""
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
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
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 去重后最多取 25 个样本
    seen = set()
    unique_samples = []
    for s in meaningful:
        key = (s['input'].strip()[:60], s['output'].strip()[:60])
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)
    sampled = unique_samples[:25]

    sample_text = "\n---\n".join([
        f"输入:\n{s['input'][:600]}\n输出:\n{s['output'][:600]}"
        for s in sampled
    ])

    evolution_prompt = f"""你是顶级提示词工程师。请根据以下提示词在实际任务中的执行结果，迭代优化提示词。

**当前提示词**（"{step_name}" 步骤）：
```
{current_prompt}
```

**实际输入→输出样本**（由当前提示词驱动生成）：
{sample_text}

**优化准则**：
1. 分析输出中的不足（术语错误、格式混乱、漏翻、冗余等），追溯提示词中的根因
2. 修补模糊指令，将"潜规则"显式化
3. 添加缺失的约束（如：不要漏行、不要添加解释、控制输出长度）
4. 删除无效或矛盾的指令
5. 强化对关键质量维度的要求

**硬性约束**：
- **必须完整保留**所有占位符变量（花括号包裹的词，如 {{{{text}}}}、{{{{terminology}}}}、{{{{count}}}}、{{{{target_lang}}}}、{{{{context}}}}）
- 切勿添加不存在的占位符变量
- 输出结构必须与输入完全一致，只优化指令部分
- 如果当前提示词已经足够好，返回原文（不要强行改动）
- **只返回优化后的完整提示词**，不要任何前言、后记、解释或代码块包裹

优化后的提示词："""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": evolution_prompt}],
            temperature=0.25
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
        return True

    except Exception as e:
        logger.warning(f"提示词进化失败 [{step_name}]（不影响主流程）: {e}")
        return False
