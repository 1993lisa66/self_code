"""
API 调用速率限制器
用于控制 LLM API 和 TTS API 的调用频率，避免触发限流
"""
import time
import threading
from typing import Optional


class RateLimiter:
    """简单的令牌桶速率限制器"""
    
    def __init__(self, max_calls: int = 10, time_window: float = 60.0):
        """
        初始化速率限制器
        
        Args:
            max_calls: 时间窗口内允许的最大调用次数
            time_window: 时间窗口（秒）
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def wait(self):
        """等待直到可以发起新的API调用"""
        with self.lock:
            now = time.time()
            
            # 清理过期的调用记录
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            # 如果达到限制，等待
            if len(self.calls) >= self.max_calls:
                oldest_call = min(self.calls)
                wait_time = self.time_window - (now - oldest_call) + 0.1  # 额外0.1秒缓冲
                
                if wait_time > 0:
                    print(f"⏳ API 速率限制：等待 {wait_time:.2f} 秒...")
                    time.sleep(wait_time)
            
            # 记录本次调用
            self.calls.append(time.time())


# ── 全局过载检测器 ──
# 任意模块触发 429 时标记，所有模块检查此标记后可跳过主模型或可选步骤
_overload_lock = threading.Lock()
_overload_until = 0.0  # 过载状态过期时间戳（0 表示未过载）
_OVERLOAD_DURATION = 300  # 过载标记持续 5 分钟后自动清除


def mark_model_overloaded():
    """标记主模型过载（429），通知所有模块降级"""
    global _overload_until
    with _overload_lock:
        _overload_until = time.time() + _OVERLOAD_DURATION


def is_model_overloaded():
    """查询主模型是否处于过载状态"""
    global _overload_until
    with _overload_lock:
        if _overload_until and time.time() < _overload_until:
            return True
        _overload_until = 0.0  # 过期自动清除
        return False


def clear_overload():
    """手动清除过载标记（主模型恢复后调用）"""
    global _overload_until
    with _overload_lock:
        _overload_until = 0.0


# 全局速率限制器实例
# LLM API: 默认每分钟最多60次调用（SiliconFlow DeepSeek-V3 实际额度远高于此）
llm_rate_limiter = RateLimiter(max_calls=60, time_window=60.0)

# TTS API: 默认每分钟最多30次调用（Edge TTS 限制较宽松）
tts_rate_limiter = RateLimiter(max_calls=30, time_window=60.0)


def wait_for_llm_api():
    """等待 LLM API 可用"""
    llm_rate_limiter.wait()


def wait_for_tts_api():
    """等待 TTS API 可用"""
    tts_rate_limiter.wait()


def set_llm_rate_limit(max_calls: int, time_window: float = 60.0):
    """设置 LLM API 速率限制"""
    global llm_rate_limiter
    llm_rate_limiter = RateLimiter(max_calls=max_calls, time_window=time_window)
    print(f"✅ LLM API 速率限制已设置为: {max_calls} 次/{time_window}秒")


def set_tts_rate_limit(max_calls: int, time_window: float = 60.0):
    """设置 TTS API 速率限制"""
    global tts_rate_limiter
    tts_rate_limiter = RateLimiter(max_calls=max_calls, time_window=time_window)
    print(f"✅ TTS API 速率限制已设置为: {max_calls} 次/{time_window}秒")
