import os
import json
import random
import re
import time
import shutil
import asyncio
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


def _filter_terms_by_text(terminology, texts):
    """只保留批内文本中出现的术语，其余术语不影响翻译质量，节省 token"""
    if not terminology or not texts:
        return "无"
    combined = " ".join(texts).lower()
    relevant = {k: v for k, v in terminology.items() if k.lower() in combined}
    if not relevant:
        return "无"
    return ";".join(f"{k}={v}" for k, v in relevant.items())


# ── LLM 响应中常见的无效注释模式 ──
_NOTE_PATTERN = re.compile(
    r'[\(\[（]\s*(?:Note|注|说明|注意|提示|Note\s*that)[：:].*?[\)\]）]',
    re.IGNORECASE | re.DOTALL,
)


def _clean_llm_line(text: str) -> str:
    """清洗 LLM 返回行中的注释/碎碎念。返回清洗后的文本，若整行都是注释则返回空字符串。"""
    text = _NOTE_PATTERN.sub('', text).strip()
    # 整行就是注释，没有实际内容
    if not text:
        return ''
    # 语言检测占位符（LLM 拒绝翻译时的典型输出）
    if re.fullmatch(r'[（(]?注[：:]?\s*行\s*\d+\s*(?:不完整|信息不足|需要上下文)[）)]?', text):
        return ''
    return text


def translate_segments(fused_results, target_lang="zh", config=None, prompt_template=None):
    """
    统一翻译入口：根据配置选择 LLM 或 Google 翻译引擎。

    Args:
        fused_results: ASR 融合结果列表
        target_lang: 目标语言
        config: 配置字典，需包含 translate.provider（"llm" / "google"）
        prompt_template: LLM 提示词模板（仅 llm 模式使用）

    Returns:
        包含 translated_text 字段的结果列表
    """
    if not fused_results:
        return []

    provider = config.get('provider', 'llm') if config else 'llm'

    if provider == 'google':
        return _translate_segments_google(fused_results, target_lang, config)
    else:
        return _translate_segments_llm(fused_results, target_lang, config, prompt_template)


def _google_translate_single(text, dest='zh-cn', max_retries=2, translator_instance=None):
    """
    用 Google 翻译单条文本。失败时返回 None（由调用方决定降级策略）。

    googletrans 的 translate() 是异步协程，每次用独立 event loop 执行。
    注意：translator_instance 在连续多次失败后会被重置，外部无需关心。
    """
    if not text or not text.strip():
        return text, translator_instance

    _text = text.strip()

    for attempt in range(max_retries):
        _loop = None
        try:
            if translator_instance is None:
                from googletrans import Translator
                translator_instance = Translator()
            _loop = asyncio.new_event_loop()
            result = _loop.run_until_complete(
                translator_instance.translate(_text, dest=dest)
            )
            translated = result.text
            if translated and translated != _text:
                return translated, translator_instance
            # Google 可能原样返回（限流或无法翻译），再试一次
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            err_msg = str(e).lower()
            # googletrans 内部 session 绑定了旧 loop，重置 translator
            if 'event loop' in err_msg or 'loop is closed' in err_msg:
                translator_instance = None
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
        finally:
            if _loop is not None and not _loop.is_closed():
                _loop.close()
    return None, translator_instance


def _parse_llm_translation_response(content, batch_size):
    """解析 LLM 翻译响应，返回 {batch_index: translated_text} 字典。"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    temp_map = {}
    for line in lines:
        match = re.match(r'^(\d+)\s*[:：]\s*(.*)$', line)
        if match:
            idx = int(match.group(1)) - 1
            cleaned = _clean_llm_line(match.group(2).strip())
            if cleaned:
                temp_map[idx] = cleaned
        elif ":" in line:
            try:
                idx_str, content_part = line.split(":", 1)
                idx = int(re.sub(r'\D', '', idx_str)) - 1
                cleaned = _clean_llm_line(content_part.strip())
                if cleaned:
                    temp_map[idx] = cleaned
            except:
                pass
    # 行数匹配时按顺序补齐缺失条目
    for j in range(batch_size):
        if j not in temp_map and len(lines) == batch_size:
            if ":" in lines[j]:
                _, temp_map[j] = lines[j].split(":", 1)
                temp_map[j] = temp_map[j].strip()
            else:
                temp_map[j] = lines[j]
    return temp_map


def _commit_translated_segments(batch, temp_map, batch_size, batch_num, total_batches):
    """将解析后的译文提交到结果列表，缺失条目用 Google 翻译兜底。"""
    results = []
    for j in range(min(batch_size, len(batch))):
        seg = batch[j]
        translated_text = temp_map.get(j)
        if not translated_text:
            logger.warning(
                f"翻译批次 {batch_num}/{total_batches} 片段 {j+1} 译文缺失，"
                f"尝试 Google 翻译: {seg['text'][:60]}"
            )
            translated_text, _ = _google_translate_single(seg['text'], dest='zh-cn')
            if translated_text:
                logger.info(f"  └ Google 翻译成功: {seg['text'][:40]} → {translated_text[:40]}")
            else:
                logger.warning(f"  └ Google 翻译也失败，降级使用原文: {seg['text'][:60]}")
                translated_text = seg['text']
        results.append({**seg, "translated_text": translated_text})
    return results


def _translate_segments_llm(fused_results, target_lang="zh", config=None, prompt_template=None):
    """
    使用 LLM 进行翻译（滑窗重叠 + 上下文连贯）。
    
    核心优化：
    1. 上下文来自已翻译的中文结果（而非英文原文），确保语义连贯
    2. 批次间有重叠窗口，每个片段在两次不同上下文中被翻译，取第二次（更多参考）
    3. 上下文窗口扩展到 6 条，覆盖约 30 秒的前文内容
    4. 术语一致性、口语化表达、长度控制
    """
    if not fused_results:
        return []

    logger.info(f"开始 LLM 翻译任务 -> {target_lang}...")

    api_key = config.get('api_key') if config else None
    if not api_key or "your-openai-api-key" in api_key:
        logger.warning("未配置有效的 API Key，返回占位翻译。")
        return [{**seg, "translated_text": f"[FIXME] {seg['text']}"} for seg in fused_results]

    stage_cfg = config.get('translator', {}) if config else {}
    model_name = stage_cfg.get('model', config.get('model', 'deepseek-ai/DeepSeek-V3'))
    base_url = config.get('api_base', 'https://api.siliconflow.cn/v1')
    temperature = stage_cfg.get('temperature', config.get('temperature', 0.1))
    request_timeout = stage_cfg.get('request_timeout', config.get('default_request_timeout', 120))
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=max(request_timeout + 30, 120.0))

    terminology = _load_terminology(config)
    full_term_str = "无"
    if terminology:
        full_term_str = ";".join(f"{k}={v}" for k, v in terminology.items())
    
    translated_results = []  # 已确认的翻译结果（中文），也是上下文来源
    
    batch_size = config.get('batch_size', 20) if config else 20
    overlap = max(2, batch_size // 8)  # 每批末尾重叠 2~3 条，在下一批重译
    context_window = stage_cfg.get('context_window', config.get('translate_context_window', 8))
    total_batches = (len(fused_results) + batch_size - 1) // batch_size
    completed_batches = 0
    last_evolved_batch = 0
    evolve_every = 12
    evolve_disabled = False
    
    fallback_model = config.get('fallback_model') if config else None
    current_model = model_name
    model_sticky = False
    sticky_check_interval = config.get('sticky_check_interval', 8) if config else 8
    
    logger.info(
        f"总共 {len(fused_results)} 个片段，{total_batches} 批（每批 {batch_size} 条，"
        f"重叠 {overlap} 条，上下文窗口 {context_window} 条）"
    )
    
    i = 0  # 当前批次在 fused_results 中的起始索引
    while i < len(fused_results):
        batch_num = (i // batch_size) + 1
        batch_display_num = (i // batch_size) + 1
        
        # 动态计算本批重叠量（最后一批可能没有足够数据做重叠）
        actual_overlap = overlap if i + batch_size + overlap <= len(fused_results) else 0
        batch_end = min(i + batch_size + actual_overlap, len(fused_results))
        batch = fused_results[i:batch_end]
        commit_count = min(batch_size, len(batch) - actual_overlap) if actual_overlap > 0 else len(batch)
        
        # 全局过载检测
        if not model_sticky and is_model_overloaded() and fallback_model:
            logger.info(f"  检测到主模型全局过载，直接使用备用模型: {fallback_model.split('/')[-1]}")
            current_model = fallback_model
            model_sticky = True
        
        if model_sticky and fallback_model and batch_display_num % sticky_check_interval == 0:
            logger.info(f"  试探主模型 {model_name.split('/')[-1]} 是否恢复...")
            current_model = model_name
        elif model_sticky and fallback_model:
            current_model = fallback_model
        
        logger.info(
            f"正在翻译批次 {batch_display_num}/{total_batches} "
            f"(原文片段 {i+1}-{batch_end}，提交前 {commit_count} 条)..."
        )
        
        # ── 构造批量翻译提示词 ──
        texts_to_translate = "\n".join([f"{j+1}: {seg['text']}" for j, seg in enumerate(batch)])
        batch_term_str = _filter_terms_by_text(terminology, [seg['text'] for seg in batch]) if terminology else "无"
        
        # ── 上下文：从已翻译结果取最近 context_window 条（原文→译文对照）──
        context_text = ""
        context_block = ""
        if translated_results:
            ctx_start = max(0, len(translated_results) - context_window)
            ctx_segs = translated_results[ctx_start:]
            context_lines = []
            for idx, seg in enumerate(ctx_segs):
                src = seg.get('text', '')
                zh = seg.get('translated_text', '')
                if src and zh and src != zh:
                    # 原文→译文对照：让模型理解翻译风格
                    context_lines.append(f"{idx+1}: {src}  →  {zh}")
                elif zh:
                    context_lines.append(f"{idx+1}: {zh}")
            if context_lines:
                context_text = "\n".join(context_lines)
                context_block = (
                    f"**前文参考**（原文→译文对照，保持术语一致、语气连贯）：\n"
                    f"{context_text}"
                )
        
        if prompt_template:
            try:
                prompt = prompt_template.replace('{target_lang}', target_lang)
                prompt = prompt.replace('{text}', texts_to_translate)
                prompt = prompt.replace('{terminology}', batch_term_str)
                prompt = prompt.replace('{context}', context_block)
                prompt = prompt.replace('{count}', str(len(batch)))
            except Exception:
                prompt = (
                    f"请将以下英文翻译成中文，口语化、适合字幕。"
                    f"原文一定是外语，必须翻译成中文。"
                    f"参考术语表：{batch_term_str}\n"
                    f"{context_block}\n\n"
                    f"格式: 数字: 翻译 每行一条, 共{len(batch)}行 不解释.\n\n"
                    f"{texts_to_translate}"
                )
        else:
            term_block = f"\n术语表：{batch_term_str}" if batch_term_str != "无" else ""
            prompt = (
                f"翻译成{target_lang}，口语化适合字幕。"
                f"{term_block}{context_block}\n"
                f"格式：\"数字: 翻译\" 每行一条，共{len(batch)}行 不解释。\n\n"
                f"{texts_to_translate}"
            )
        
        # ── 带重试 + 模型降级 + 粘性的 API 调用 ──
        max_retries = stage_cfg.get('max_retries', 3)
        base_delay = config.get('retry_base_delay', 3) if config else 3
        translated_this_batch = False
        temp_map = {}

        for attempt in range(max_retries):
            if attempt >= 2 and fallback_model and current_model != fallback_model:
                logger.info(f"  └ 主模型 {current_model} 持续限流，切换到备用模型: {fallback_model}")
                current_model = fallback_model
                model_sticky = True

            try:
                wait_for_llm_api()
                input_chars = sum(len(seg['text']) for seg in batch)
                dynamic_max_tokens = max(3000, int(input_chars * 3))
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=dynamic_max_tokens,
                    timeout=request_timeout
                )

                content = response.choices[0].message.content.strip()
                temp_map = _parse_llm_translation_response(content, len(batch))
                translated_this_batch = True
                break

            except Exception as e:
                err_msg = str(e).lower()
                err_str = str(e)
                if any(kw in err_msg for kw in ('balance', 'insufficient', 'invalid', 'unauthorized', '403', '401')):
                    logger.error(f"翻译批次 {batch_display_num}/{total_batches} API 错误: {e}")
                    raise RuntimeError(
                        f"翻译 API 致命错误（余额不足/Key无效），已处理 {completed_batches}/{total_batches} 批。"
                        f"请检查 API Key 和账户余额后重试。"
                    ) from e

                is_rate_limit = '429' in err_str or 'rate' in err_msg or 'too busy' in err_msg
                is_server_error = any(f'{code}' in err_str for code in range(500, 600))
                is_timeout_err = 'timeout' in err_msg or 'timed out' in err_msg
                is_retryable = is_rate_limit or is_server_error or is_timeout_err

                if is_rate_limit:
                    mark_model_overloaded()
                if is_retryable and attempt < max_retries - 1:
                    delay = (base_delay ** (attempt + 1)) * (0.5 + random.random())
                    model_label = current_model.split('/')[-1]
                    reason = "限流" if is_rate_limit else ("超时" if is_timeout_err else f"服务端错误({err_str[:80]})")
                    logger.warning(f"翻译{reason} [{model_label}]，{delay:.1f}s 后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue

                logger.error(f"批量翻译失败 (批次 {batch_display_num}/{total_batches}): {e}")
                break

        # ── 提交本批结果 ──
        if not translated_this_batch:
            logger.warning(
                f"翻译批次 {batch_display_num}/{total_batches} LLM 失败（{max_retries} 次重试耗尽），"
                f"使用 Google 翻译兜底 {len(batch)} 个片段"
            )
            for j, seg in enumerate(batch):
                gt, _ = _google_translate_single(seg['text'], dest='zh-cn')
                if gt:
                    translated_results.append({**seg, "translated_text": gt})
                    logger.info(f"  └ Google 兜底 [{j+1}/{len(batch)}]: {seg['text'][:40]} → {gt[:40]}")
                else:
                    logger.warning(f"  └ Google 兜底也失败 [{j+1}/{len(batch)}]，使用原文: {seg['text'][:60]}")
                    translated_results.append({**seg, "translated_text": seg['text']})
            completed_batches += 1
            progress_pct = (completed_batches / total_batches * 100) if total_batches > 0 else 0
            logger.info(f"批次 {batch_display_num}/{total_batches} 完成（Google 兜底）({progress_pct:.1f}%)")
        else:
            # 仅提交前 commit_count 条，剩余的 overlap 交由下一批重译
            batch_results = _commit_translated_segments(
                batch, temp_map, commit_count, batch_display_num, total_batches
            )
            translated_results.extend(batch_results)
            completed_batches += 1
            progress_pct = (completed_batches / total_batches * 100) if total_batches > 0 else 0
            logger.info(f"批次 {batch_display_num}/{total_batches} 完成 ({progress_pct:.1f}%)")
        
        # ── 中期术语表进化 ──
        if (not evolve_disabled and config and config.get('api_key') 
                and config.get('terminology_file') and translated_results 
                and batch_display_num % evolve_every == 0 and batch_display_num != last_evolved_batch):
            try:
                prev_count = len(terminology) if terminology else 0
                logger.info(f"  └ 触发术语表中途进化（第 {batch_display_num}/{total_batches} 批）")
                result = evolve_terminology(translated_results, config=config)
                last_evolved_batch = batch_display_num
                reloaded = _load_terminology(config)
                new_count = len(reloaded) if reloaded else 0
                if new_count != prev_count:
                    logger.info(f"  └ 术语表已刷新: {prev_count} → {new_count} 条")
                terminology = reloaded
                if terminology:
                    full_term_str = ";".join(f"{k}={v}" for k, v in terminology.items())
                else:
                    full_term_str = "无"
                if new_count > 5 and new_count == prev_count:
                    evolve_disabled = True
                    logger.info(f"  └ 术语表已收敛（{new_count} 条），停止后续进化")
            except Exception as e:
                logger.warning(f"  └ 中期术语进化跳过（不影响主流程）: {e}")
        
        # 前进 batch_size（而非 batch_size+overlap），重叠部分下批重译
        i += batch_size

    # ── 最终术语表进化 ──
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

    # ── 翻译质量校验：检测并修复漏翻/空翻译 ──
    untranslated = 0
    fixed_count = 0
    for seg in translated_results:
        src_text = (seg.get('text') or '').strip()
        tr_text = (seg.get('translated_text') or '').strip()
        if not tr_text or tr_text == src_text:
            untranslated += 1
            # 自动尝试 Google 翻译兜底
            gt, _ = _google_translate_single(src_text, dest='zh-cn')
            if gt and gt != src_text:
                seg['translated_text'] = gt
                fixed_count += 1
    if untranslated > 0:
        if fixed_count > 0:
            logger.info(f"翻译质量校验: Google 兜底修复 {fixed_count}/{untranslated} 条空/漏翻译")
        if untranslated > fixed_count:
            logger.warning(
                f"翻译质量校验: {untranslated - fixed_count} 条片段仍无有效翻译，"
                f"请检查原文或人工修正"
            )

    logger.success(f"LLM 翻译完成，共 {len(translated_results)} 段")
    return translated_results


def _translate_segments_google(fused_results, target_lang="zh", config=None):
    """
    使用 Google Translate（googletrans）进行翻译。

    优点：免费、快速、无需 API Key
    缺点：不支持术语表、可能被限速、翻译质量不如专业 LLM

    每次翻译一个片段，间隔短暂延迟避免被 Google 封禁。
    失败时降级使用原文。

    Args:
        fused_results: ASR 融合结果列表
        target_lang: 目标语言（zh→zh-cn, ja, ko, en）
        config: 配置字典（可包含 translate.google_delay 控制请求间隔）

    Returns:
        包含 translated_text 字段的结果列表
    """
    if not fused_results:
        return []

    try:
        from googletrans import Translator
    except ImportError:
        raise ImportError(
            "请安装 googletrans: pip install googletrans>=3.1.0a0\n"
            "或使用 LLM 翻译：在 config.yaml 中设置 translate.provider: llm"
        )

    translator = Translator()

    # 语言代码映射
    lang_map = {'zh': 'zh-cn', 'en': 'en', 'ja': 'ja', 'ko': 'ko', 'fr': 'fr',
                'de': 'de', 'es': 'es', 'pt': 'pt', 'ru': 'ru', 'ar': 'ar'}
    dest = lang_map.get(target_lang, target_lang)

    # Google 翻译延迟（秒），控制请求频率避免被 ban
    sleep_interval = config.get('google_delay', 0.3) if config else 0.3

    translated_results = []
    total = len(fused_results)
    failed_count = 0

    logger.info(f"开始 Google 翻译 -> {dest}，共 {total} 个片段（间隔 {sleep_interval}s）")

    for idx, seg in enumerate(fused_results):
        text = seg.get('text', '').strip()
        if not text:
            # 空文本直接保留
            translated_results.append({**seg, 'translated_text': ''})
            continue

        # 使用带重试的单条翻译（复用 translator 实例，失败时自动重建）
        gt, translator = _google_translate_single(text, dest=dest, max_retries=2,
                                                   translator_instance=translator)
        if gt:
            translated_results.append({**seg, 'translated_text': gt})
        else:
            # Google 翻译失败 → 降级使用原文（最后手段）
            failed_count += 1
            logger.warning(f"Google 翻译片段 {idx+1}/{total} 失败，降级使用原文")
            translated_results.append({**seg, 'translated_text': text})

        # 进度显示
        if (idx + 1) % 20 == 0 or idx == total - 1:
            pct = (idx + 1) / total * 100
            logger.info(f"Google 翻译进度: {idx+1}/{total} ({pct:.0f}%)")

        # 请求间隔，避免被 Google 限速
        if idx < total - 1:
            time.sleep(sleep_interval)

    if failed_count > 0:
        logger.warning(f"Google 翻译完成，{failed_count}/{total} 个片段降级使用原文")
    else:
        logger.success(f"Google 翻译完成，共 {len(translated_results)} 段")

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
    
    stage_cfg = config.get('translator', {}) if config else {}
    model_name = stage_cfg.get('model', config.get('model', 'deepseek-ai/DeepSeek-V3'))
    base_url = config.get('api_base', 'https://api.siliconflow.cn/v1')
    request_timeout = stage_cfg.get('request_timeout', config.get('default_request_timeout', 120))
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=max(request_timeout + 30, 120.0))
    
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
    
    # 紧凑格式术语表
    if existing_terms:
        existing_terms_str = ";".join(f"{k}={v}" for k, v in existing_terms.items())
    else:
        existing_terms_str = "（空）"
    
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
            timeout=request_timeout
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

    stage_cfg = config.get('translator', {}) if config else {}
    model_name = stage_cfg.get('model', config.get('model', 'deepseek-ai/DeepSeek-V3'))
    base_url = config.get('api_base', 'https://api.siliconflow.cn/v1')
    request_timeout = stage_cfg.get('request_timeout', config.get('default_request_timeout', 120))
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=max(request_timeout + 30, 120.0))

    # 去重后最多取 15 个样本
    seen = set()
    unique_samples = []
    for s in meaningful:
        key = (s['input'].strip()[:60], s['output'].strip()[:60])
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)
    sampled = unique_samples[:10]

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
            timeout=request_timeout
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
