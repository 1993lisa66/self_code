import os
import json
import hashlib
import re
import time
import random
from loguru import logger
from openai import OpenAI
from modules.utils.rate_limiter import wait_for_llm_api, mark_model_overloaded


def _get_cache_path():
    """获取 TTS 文本缓存文件路径"""
    from pathlib import Path
    cache_dir = Path(__file__).parent.parent.parent.parent / "cache" / "tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "text_cache.json"


def _load_cache():
    """加载 TTS 文本缓存"""
    cache_path = _get_cache_path()
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache_dict):
    """保存 TTS 文本缓存"""
    cache_path = _get_cache_path()
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存 TTS 文本缓存失败: {e}")


def _text_hash(text):
    """计算文本哈希值作为缓存键"""
    return hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()


def _make_client(config):
    """创建 OpenAI 客户端"""
    api_key = config.get('api_key') if config else None
    base_url = config.get('api_base', 'https://api.siliconflow.cn/v1') if config else 'https://api.siliconflow.cn/v1'
    stage_cfg = config.get('tts_processor', {}) if config else {}
    model_name = stage_cfg.get('model', config.get('model', 'deepseek-ai/DeepSeek-V3')) if config else 'deepseek-ai/DeepSeek-V3'
    if not api_key or "your-openai-api-key" in api_key:
        return None, None
    timeout = stage_cfg.get('request_timeout', config.get('default_request_timeout', 120)) if config else 120
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=max(timeout + 30, 120.0))
    return client, model_name


def _clean_response(text):
    """清洗 LLM 响应中的碎碎念"""
    text = re.sub(r'[\(（](注|Note).*?[\)）]', '', text, flags=re.DOTALL)
    text = "\n".join([line.strip() for line in text.split('\n') if line.strip()])
    return text.strip()


_IGNORE_RESPONSES = [
    "原样返回即可", "无需转换", "不需要转换",
    "The text does not contain", "No conversion needed"
]


def _is_noop_response(text):
    """检查 LLM 是否返回了 '无需处理' 类提示语"""
    return any(resp in text for resp in _IGNORE_RESPONSES)


def process_tts_text(text, config=None, prompt_template=None):
    """
    将文本中的数字、日期等转换为适合 TTS 朗读的中文格式。
    （单句模式，保留向后兼容）
    """
    if not text:
        return ""

    client, model_name = _make_client(config)
    if client is None:
        return text

    if prompt_template:
        prompt = prompt_template.format(current_sentence=text)
    else:
        prompt = (
            f"数字、日期、符号→TTS中文口语。只返回结果，不解释。无需转换则原文。\n\n"
            f"{text}"
        )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        processed_text = response.choices[0].message.content.strip()
        processed_text = _clean_response(processed_text)
        
        if not processed_text:
            logger.warning(f"LLM 返回空文本，使用原文: {text[:50]}...")
            return text
        if _is_noop_response(processed_text):
            logger.debug(f"LLM 返回提示语，使用原文: {text[:50]}...")
            return text
        
        return processed_text
    except Exception as e:
        logger.error(f"TTS 文本预处理失败: {e}")
        return text


def process_tts_text_batch(texts, config=None, prompt_template=None, batch_size=None):
    # 批次太大容易导致 LLM 截断返回（只输出编号不输出内容）
    # 优先级：显式参数 > config['batch_size'] > config['tts_processor']['batch_size'] > 默认 8
    if batch_size is None:
        if config:
            batch_size = config.get('batch_size') or config.get('tts_processor', {}).get('batch_size')
        batch_size = batch_size or 8
    """
    批量处理 TTS 文本，一次 LLM 调用处理多条，大幅降低 API 调用次数和 token 消耗。
    同时内置文件缓存，相同输入文本不会重复调用 LLM。

    Args:
        texts: 待处理的文本列表
        config: LLM 配置字典
        prompt_template: 可选提示词模板
        batch_size: 每批处理的文本数量，默认 20

    Returns:
        处理后的文本列表，与输入一一对应
    """
    if not texts:
        return []

    client, model_name = _make_client(config)
    if client is None:
        return texts

    # 1. 加载缓存，过滤已缓存的文本
    cache = _load_cache()
    results = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []
    cache_hits = 0

    for i, text in enumerate(texts):
        if not text:
            results[i] = ""
            continue
        key = _text_hash(text)
        if key in cache:
            results[i] = cache[key]
            cache_hits += 1
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if cache_hits > 0:
        logger.info(f"TTS 缓存命中: {cache_hits}/{len(texts)} 条")

    # 2. 对未缓存的文本分批调用 LLM
    if not uncached_texts:
        logger.success("TTS 文本预处理完成（全部命中缓存）")
        return results

    total_batches = (len(uncached_texts) + batch_size - 1) // batch_size
    logger.info(f"TTS 文本预处理: {len(uncached_texts)} 条未缓存，分 {total_batches} 批处理（每批 {batch_size} 条）")

    new_cache_entries = {}
    for batch_idx in range(0, len(uncached_texts), batch_size):
        batch_texts = uncached_texts[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        # 构建编号格式的输入
        numbered = "\n".join(f"{j+1}: {t}" for j, t in enumerate(batch_texts))

        if prompt_template:
            try:
                prompt = prompt_template.format(text=numbered, count=len(batch_texts))
            except (KeyError, IndexError):
                prompt = _default_batch_prompt(numbered, len(batch_texts))
        else:
            prompt = _default_batch_prompt(numbered, len(batch_texts))

        max_retries = config.get('tts_processor', {}).get('max_retries', 3) if config else 3
        base_delay = config.get('retry_base_delay', 3) if config else 3
        batch_ok = False

        for attempt in range(max_retries):
            try:
                logger.info(f"TTS 批次 {batch_num}/{total_batches} ({len(batch_texts)} 条)...")
                wait_for_llm_api()
                # 动态 max_tokens：输入字符数 × 3（输出中文 token 开销大于输入英文）
                # 最少 2000，防止模型因 token 不足而截断行尾内容
                input_chars = sum(len(t) for t in batch_texts)
                dynamic_max_tokens = max(2000, int(input_chars * 3))
                logger.debug(f"TTS 批次 max_tokens={dynamic_max_tokens} (输入 {input_chars} 字符)")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=dynamic_max_tokens
                )
                content = response.choices[0].message.content.strip()
                lines = [line.strip() for line in content.split('\n') if line.strip()]

                # 解析编号格式的响应
                parsed = _parse_numbered_response(lines, len(batch_texts))

                for j, orig_idx in enumerate(range(batch_idx, batch_idx + len(batch_texts))):
                    original_text = uncached_texts[orig_idx]
                    processed = parsed.get(j, "")
                    if processed:
                        processed = _clean_response(processed)
                        if _is_noop_response(processed):
                            processed = original_text
                    else:
                        processed = original_text

                    # 质量验证：结果太短（可能是 LLM 截断），回退到原文
                    if processed and len(processed) < 5 and len(original_text) > 10:
                        logger.warning(
                            f"TTS 预处理结果疑似截断（结果{len(processed)}字 vs 原文{len(original_text)}字），"
                            f"回退原文: '{processed[:50]}' → '{original_text[:50]}...'"
                        )
                        processed = original_text

                    global_idx = uncached_indices[orig_idx]
                    results[global_idx] = processed
                    key = _text_hash(original_text)
                    new_cache_entries[key] = processed

                logger.info(f"TTS 批次 {batch_num}/{total_batches} 完成 ({batch_num*100//total_batches}%)")
                batch_ok = True
                break  # 成功，跳出重试循环

            except Exception as e:
                err_msg = str(e).lower()
                err_str = str(e)
                is_rate_limit = '429' in err_str or 'rate' in err_msg or 'too busy' in err_msg
                is_server_error = any(f'{code}' in err_str for code in range(500, 600))
                is_retryable = is_rate_limit or is_server_error

                if is_retryable and attempt < max_retries - 1:
                    if is_rate_limit:
                        mark_model_overloaded()
                    delay = (base_delay ** (attempt + 1)) * (0.5 + random.random())
                    reason = "限流" if is_rate_limit else f"服务端错误({err_str[:80]})"
                    logger.warning(f"TTS 预处理{reason}，{delay:.1f}s 后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue

                logger.error(f"TTS 批次 {batch_num}/{total_batches} 失败: {e}")
                break

        # 所有重试都失败 → 回退到原文（不缓存，下次重试）
        if not batch_ok:
            for j, orig_idx in enumerate(range(batch_idx, batch_idx + len(batch_texts))):
                global_idx = uncached_indices[orig_idx]
                results[global_idx] = uncached_texts[orig_idx]

    # 3. 更新缓存
    if new_cache_entries:
        cache.update(new_cache_entries)
        _save_cache(cache)
        logger.info(f"TTS 缓存已更新: +{len(new_cache_entries)} 条")

    logger.success(f"TTS 文本预处理完成: {len(texts)} 条")
    return results


def _default_batch_prompt(numbered_text, count):
    """默认批量提示词"""
    return (
        "数字→中文口语，如11,234→一万一千两百三十四，"
        f"50%→百分之五十。格式：\"数字: 结果\"，共{count}行，不解释。\n\n"
        f"{numbered_text}"
    )


def _parse_numbered_response(lines, expected_count):
    """解析 LLM 返回的编号格式响应，返回 {序号: 文本} 字典"""
    result = {}
    for line in lines:
        match = re.match(r'^(\d+)\s*[:：]\s*(.*)$', line)
        if match:
            idx = int(match.group(1)) - 1
            content = match.group(2).strip()
            if content:  # 只保留非空内容
                result[idx] = content
        elif ':' in line or '：' in line:
            sep = ':' if ':' in line else '：'
            idx_str, _, content = line.partition(sep)
            try:
                idx = int(re.sub(r'\D', '', idx_str)) - 1
                content = content.strip()
                if content:
                    result[idx] = content
            except ValueError:
                pass

    # 如果解析到的条目不够，尝试按行号对齐
    # 关键修复：必须剥离编号前缀，否则 TTS 引擎会读出行号数字
    if len(result) < expected_count and len(lines) == expected_count:
        for i, line in enumerate(lines):
            if i not in result:
                # 尝试剥离可能的编号前缀再使用
                stripped = re.sub(r'^\d+\s*[:：]\s*', '', line).strip()
                result[i] = stripped if stripped else line

    return result
