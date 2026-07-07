# -*- coding: utf-8 -*-
"""
Edge TTS 引擎（微软免费在线 TTS）。
使用 edge-tts 库调用微软 Edge 浏览器的语音合成接口。
"""

import os
import asyncio
import edge_tts
from loguru import logger

from modules.tts.engines.base import BaseTTSEngine


class EdgeTTSEngine(BaseTTSEngine):
    """Edge TTS（微软免费在线语音合成）"""

    def __init__(self, engine_config: dict):
        super().__init__(engine_config)
        self.voice = engine_config.get('voice', 'zh-CN-YunyangNeural')

    async def synthesize(self, text: str, output_path: str, max_retries: int = 2) -> bool:
        """调用 Edge TTS 合成单段语音，支持自动重试"""
        self._ensure_output_dir(output_path)

        if not text or not text.strip():
            logger.error(f"文本为空，跳过合成: {output_path}")
            return False

        for attempt in range(max_retries + 1):
            try:
                abs_path = os.path.abspath(output_path)
                logger.debug(
                    f"Edge TTS 合成: '{text[:50]}...' → {os.path.basename(output_path)}"
                )
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(abs_path)

                if self.validate_output(abs_path):
                    logger.debug(
                        f"Edge TTS 成功: {os.path.basename(output_path)} "
                        f"({os.path.getsize(abs_path)} bytes)"
                    )
                    return True
                else:
                    file_size = os.path.getsize(abs_path) if os.path.exists(abs_path) else 0
                    if attempt < max_retries:
                        logger.warning(
                            f"Edge TTS 生成文件无效 ({file_size} bytes)，"
                            f"重试 {attempt + 1}/{max_retries}: {abs_path}"
                        )
                        continue
                    else:
                        logger.error(
                            f"Edge TTS 生成文件无效 ({file_size} bytes): {abs_path}"
                        )
                        return False

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Edge TTS 合成失败，重试 {attempt + 1}/{max_retries}: {e}"
                    )
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Edge TTS 合成失败 (已重试 {max_retries} 次): {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return False

        return False
