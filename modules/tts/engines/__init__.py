# -*- coding: utf-8 -*-
"""
TTS 引擎工厂模块。
通过 config.yaml 的 tts.provider 字段切换引擎。
"""

from loguru import logger


def create_tts_engine(tts_config: dict, voice: str = None):
    """
    根据配置创建 TTS 引擎实例。

    Args:
        tts_config: 完整的 tts 配置字典，包含 provider 字段和引擎专属配置
        voice: 可选的全局语音覆盖（仅 Edge TTS 使用）

    Returns:
        BaseTTSEngine 子类实例

    Raises:
        ValueError: 不支持的 provider
    """
    provider = tts_config.get('provider', 'edge')
    engine_config = tts_config.get(provider, {})

    if voice is not None and provider == 'edge':
        engine_config = {**engine_config, 'voice': voice}

    if provider == 'edge':
        from modules.tts.engines.edge_tts_engine import EdgeTTSEngine
        return EdgeTTSEngine(engine_config)
    else:
        raise ValueError(
            f"不支持的 TTS provider: {provider}，"
            f"可选: edge"
        )
