#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨进程 API 调用互斥锁：确保多进程场景下，LLM / TTS 等远程 API 调用串行执行，
避免触发 API 提供商的限速机制。本地计算（ASR、音频处理、FFmpeg 合成）不受影响。

用法：
    from modules.utils.api_lock import api_lock

    with api_lock():
        # 调用 LLM API / Edge TTS 等远程服务
        response = client.chat.completions.create(...)
"""

import fcntl
import os
from pathlib import Path


def _get_default_lock_dir():
    """获取默认锁目录（项目根下的 cache 目录）"""
    project_root = Path(__file__).resolve().parent.parent.parent
    lock_dir = project_root / "cache" / ".api_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return str(lock_dir)


class ApiLock:
    """基于 fcntl.flock 的跨进程互斥锁，用于串行化远程 API 调用。"""

    def __init__(self, lock_dir=None, lock_name="api_call"):
        """
        Args:
            lock_dir: 锁文件目录（默认 cache/.api_locks）
            lock_name: 锁名称（用于区分不同用途的锁）
        """
        self._lock_dir = lock_dir or _get_default_lock_dir()
        self._lock_path = os.path.join(self._lock_dir, f".{lock_name}.lock")
        self._fd = None

    def acquire(self):
        """阻塞式获取锁。"""
        self._fd = open(self._lock_path, "w")
        fcntl.flock(self._fd, fcntl.LOCK_EX)

    def release(self):
        """释放锁。"""
        if self._fd:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


# 全局单例：LLM API 锁（LLM修正、翻译、TTS文本预处理等共用）
api_lock = ApiLock(lock_name="api_call")

# TTS 锁：Edge TTS / 其他 TTS 提供商使用，与 LLM API 互不阻塞
tts_lock = ApiLock(lock_name="tts_call")
