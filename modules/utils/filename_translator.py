#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名翻译工具：使用 Google Translate 将英文文件名翻译为中文。
"""

import asyncio
import re
import unicodedata
from loguru import logger

# Emoji / 图标 Unicode 范围（保留不被翻译）
_ICON_PATTERN = re.compile(
    '[\U0001F300-\U0001F5FF'   # 杂项符号和象形文字
    '\U0001F600-\U0001F64F'    # 表情符号
    '\U0001F680-\U0001F6FF'    # 交通和地图
    '\U0001F700-\U0001F77F'    # 炼金术符号
    '\U0001F780-\U0001F7FF'    # 几何形状扩展
    '\U0001F800-\U0001F8FF'    # 补充箭头-C
    '\U0001F900-\U0001F9FF'    # 补充符号和象形文字
    '\U0001FA00-\U0001FA6F'    # 象棋符号
    '\U0001FA70-\U0001FAFF'    # 扩展-A 符号和象形文字
    '\U0001F000-\U0001F02F'    # 麻将牌
    '\U0001F0A0-\U0001F0FF'    # 扑克牌
    '\U0001F100-\U0001F1FF'    # 封闭式字母数字补充
    '\U0001F200-\U0001F2FF'    # 封闭式表意文字补充
    '\u2600-\u26FF'            # 杂项符号
    '\u2700-\u27BF'            # 装饰符号
    '\u2300-\u23FF'            # 杂项技术
    '\u2B00-\u2BFF'            # 杂项符号和箭头
    '\uFE0F\u200D'             # 变体选择符 & 零宽连字
    ']'
)

_FILENAME_TRANSLATION_CACHE = {}
_GOOGLE_TRANSLATOR = None


def _get_translator():
    """懒加载 Google Translator 实例。"""
    global _GOOGLE_TRANSLATOR
    if _GOOGLE_TRANSLATOR is None:
        try:
            from googletrans import Translator
        except ImportError:
            raise ImportError(
                "请安装 googletrans: pip install googletrans>=3.1.0a0"
            )
        _GOOGLE_TRANSLATOR = Translator()
    return _GOOGLE_TRANSLATOR


def translate_filename(name, config=None):
    """
    使用 Google Translate 将英文文件名翻译为中文。
    优先级：缓存命中 > Google 翻译 > 原名。

    Args:
        name: 原始文件名（不含扩展名）
        config: 可选配置字典（保留兼容性，支持 google_delay 字段）

    Returns:
        str: 翻译后的文件名
    """
    if not name:
        return name or ""

    # 如果文件名已经主要是中文，无需翻译
    chinese_chars = sum(1 for c in name if 'CJK' in unicodedata.name(c, ''))
    if chinese_chars > len(name) * 0.3:
        return name

    # 缓存命中
    if name in _FILENAME_TRANSLATION_CACHE:
        cached = _FILENAME_TRANSLATION_CACHE[name]
        logger.debug(f"  文件名翻译(缓存): \"{name}\" → \"{cached}\"")
        return cached

    # 提取编号前缀（如 "01 - ", "02-", "03."），只翻译正文部分
    number_prefix_match = re.match(r'^(\d+\s*[-._]\s*)', name)
    number_prefix = number_prefix_match.group(0) if number_prefix_match else ''
    text_to_translate = name[number_prefix_match.end():].strip() if number_prefix_match else name

    # 提取图标/emoji 前缀（如 🔥 或 🎯），保留到翻译后的文件名中
    icon_prefix = ''
    icon_match = _ICON_PATTERN.match(text_to_translate)
    if icon_match:
        icon_prefix = icon_match.group(0)
        text_to_translate = text_to_translate[icon_match.end():].strip()

    if not text_to_translate:
        return name

    max_retries = 2
    _loop = None
    for attempt in range(max_retries):
        try:
            translator = _get_translator()
            # googletrans 4.0.0rc1 的 translate 是异步协程，每次用独立 event loop
            _loop = asyncio.new_event_loop()
            result = _loop.run_until_complete(
                translator.translate(text_to_translate, src='en', dest='zh-cn')
            )
            translated = result.text.strip()
            translated = translated.strip('"\' 。，, \n\r\t')
            translated = re.sub(r'[\\/:*?"<>|]', '', translated)

            if translated:
                # 编号 + 图标 + 翻译文本
                full_translated = number_prefix + icon_prefix + translated
                logger.info(f"  文件名翻译: \"{name}\" → \"{full_translated}\"")
                _FILENAME_TRANSLATION_CACHE[name] = full_translated
                return full_translated
            return name  # 翻译结果为空
        except Exception as e:
            # googletrans 内部 aiohttp session 绑定到旧 loop，loop关闭后复用时出错
            # 重置 Translator 使其在下一次尝试中用新 session
            err_msg = str(e)
            if 'event loop' in err_msg.lower() or 'loop is closed' in err_msg.lower():
                global _GOOGLE_TRANSLATOR
                _GOOGLE_TRANSLATOR = None
            if attempt == max_retries - 1:
                logger.warning(f"  文件名翻译失败（使用原名）: {e}")
            else:
                logger.debug(f"  文件名翻译重试 ({attempt + 1}/{max_retries}): {e}")
        finally:
            if _loop is not None and not _loop.is_closed():
                _loop.close()

    return name
