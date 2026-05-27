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


# 全局速率限制器实例
# LLM API: 默认每分钟最多15次调用（根据 SiliconFlow 免费配额调整）
llm_rate_limiter = RateLimiter(max_calls=15, time_window=60.0)

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
