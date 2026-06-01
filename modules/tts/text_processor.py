import os
import json
import hashlib
import re
from loguru import logger
from openai import OpenAI


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
    base_url = config.get('api_base', 'https://api.openai.com/v1') if config else 'https://api.openai.com/v1'
    model_name = config.get('model', 'gpt-4o') if config else 'gpt-4o'
    if not api_key or "your-openai-api-key" in api_key:
        return None, None
    client = OpenAI(api_key=api_key, base_url=base_url)
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


def process_tts_text_batch(texts, config=None, prompt_template=None, batch_size=20):
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

        try:
            logger.info(f"TTS 批次 {batch_num}/{total_batches} ({len(batch_texts)} 条)...")
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            content = response.choices[0].message.content.strip()
            lines = [line.strip() for line in content.split('\n') if line.strip()]

            # 解析编号格式的响应
            parsed = _parse_numbered_response(lines, len(batch_texts))

            for j, orig_idx in enumerate(range(batch_idx, batch_idx + len(batch_texts))):
                processed = parsed.get(j, "")
                if processed:
                    processed = _clean_response(processed)
                    if _is_noop_response(processed):
                        processed = uncached_texts[orig_idx]
                else:
                    processed = uncached_texts[orig_idx]

                global_idx = uncached_indices[orig_idx]
                results[global_idx] = processed
                key = _text_hash(uncached_texts[orig_idx])
                new_cache_entries[key] = processed

            logger.info(f"TTS 批次 {batch_num}/{total_batches} 完成 ({batch_num*100//total_batches}%)")

        except Exception as e:
            logger.error(f"TTS 批次 {batch_num}/{total_batches} 失败: {e}")
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
        "数字、日期、符号→TTS中文口语。格式：\"数字: 结果\"，"
        f"共{count}行。不解释。无需转换则原文。\n\n"
        f"{numbered_text}"
    )


def _parse_numbered_response(lines, expected_count):
    """解析 LLM 返回的编号格式响应，返回 {序号: 文本} 字典"""
    result = {}
    for line in lines:
        match = re.match(r'^(\d+)\s*[:：]\s*(.*)$', line)
        if match:
            idx = int(match.group(1)) - 1
            result[idx] = match.group(2).strip()
        elif ':' in line or '：' in line:
            idx_str, _, content = line.partition(':') if ':' in line else line.partition('：')
            try:
                idx = int(re.sub(r'\D', '', idx_str)) - 1
                result[idx] = content.strip()
            except ValueError:
                pass

    # 如果解析到的条目不够，尝试按行号对齐
    if len(result) < expected_count and len(lines) == expected_count:
        for i, line in enumerate(lines):
            if i not in result:
                result[i] = line

    return result
