import os
import time
import random
from yt_dlp import YoutubeDL



# Invest with Henry
# https://www.youtube.com/watch?v=sCzdZCb0LlI

def is_playlist(url):
    """检测 URL 是播放列表还是单个视频"""
    return any(kw in url.lower() for kw in ["playlist", "list="])


def download_videos(url, output_dir, cookie_browser="chrome"):
    """
    下载 YouTube 视频或播放列表，始终下载最高画质。
    自动识别 URL 类型，使用对应的命名模板。

    Args:
        url:          YouTube 视频或播放列表 URL
        output_dir:   输出目录
        cookie_browser: 导出 Cookie 的浏览器 (chrome / firefox / edge 等)
    """
    playlist = is_playlist(url)

    if playlist:
        print(f"📋 播放列表: {url}")
        outtmpl = os.path.join(output_dir, '%(playlist_index)02d - %(title)s.%(ext)s')
    else:
        print(f"🎬 单个视频: {url}")
        outtmpl = os.path.join(output_dir, '%(title)s.%(ext)s')

    # 随机延时，模拟人类行为
    delay = random.uniform(3, 8)
    print(f"⏳ 等待 {delay:.1f} 秒后开始下载...")
    time.sleep(delay)

    ydl_opts = {
        'outtmpl': outtmpl,
        # ====================== 最高画质 ======================
        'format': (
            'bestvideo[height>=2160][ext=mp4]+bestaudio[ext=m4a]/'  # 4K mp4
            'bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/'  # 1080p mp4
            'bestvideo[ext=mp4]+bestaudio[ext=m4a]/'                # mp4 可用最高
            'bestvideo+bestaudio/'                                   # 任意格式最高
            'best'                                                   # 兜底
        ),
        'merge_output_format': 'mp4',
        'format_sort': ['res:2160', 'codec:h264'],

        # 实时从浏览器读取 Cookie（每次请求都获取最新，不会过期）
        'cookiesfrombrowser': (cookie_browser,),

        # ====================== 字幕设置 ======================
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'zh'],
        'subtitlesformat': 'srt/best',

        'ignoreerrors': True,
        'retries': 30,
        'fragment_retries': 30,
        'concurrent_fragment_downloads': 4,
        'sleep_interval': 3
    }

    os.makedirs(output_dir, exist_ok=True)

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("\n🎉 下载完成！")
        return True
    except Exception as e:
        print(f"❌ 下载出错: {e}")
        return False


# ====================== 默认配置 ======================
# 不传参数时使用的默认值，可直接修改这里
DEFAULT_URL = "https://www.youtube.com/playlist?list=PLOk5U2Eu5On0XCn5x6deWmIFngHAyTgP3"
DEFAULT_OUTPUT = "/Volumes/mvp/[00]交易场/JustinWerlein/Trade School"
COOKIE_BROWSER = "chrome"
# ====================================================


# ====================== 观察列表 ======================
# https://www.youtube.com/playlist?list=PLguWwLNVYKWfGzKcW358QivkQceAtW5B-

if __name__ == "__main__":
    # 直接修改上方 DEFAULT_URL / DEFAULT_OUTPUT 即可运行
    print(f"📥 目标: {DEFAULT_URL}")
    print(f"📂 输出: {DEFAULT_OUTPUT}")
    download_videos(DEFAULT_URL, DEFAULT_OUTPUT, COOKIE_BROWSER)