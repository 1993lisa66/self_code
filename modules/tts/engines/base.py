# -*- coding: utf-8 -*-
"""
TTS 引擎抽象基类。
所有 TTS 引擎必须继承此类并实现 synthesize 方法。
"""

import os
from loguru import logger
from modules.utils.media_utils import validate_media_file


class BaseTTSEngine:
    """TTS 引擎基类"""

    def __init__(self, engine_config: dict):
        """
        Args:
            engine_config: 引擎专属配置字典（如 tts.edge 等）
        """
        self.config = engine_config

    async def synthesize(self, text: str, output_path: str, max_retries: int = 2) -> bool:
        """
        合成单段语音。

        Args:
            text: 待合成的文本
            output_path: 输出音频文件路径
            max_retries: 最大重试次数

        Returns:
            bool: 合成是否成功
        """
        raise NotImplementedError("子类必须实现 synthesize 方法")

    def validate_output(self, path: str) -> bool:
        """验证输出文件是否有效（非空、可播放）"""
        return validate_media_file(path)

    def _ensure_output_dir(self, path: str):
        """确保输出目录存在"""
        abs_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(abs_dir, exist_ok=True)

    def cleanup(self):
        """释放引擎资源（模型卸载等）。子类可按需重写。"""
        pass
