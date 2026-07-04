import os
import sys
import time
import subprocess
from yt_dlp import YoutubeDL

# ====================== 配置 ======================
URL = "https://www.youtube.com/playlist?list=PL2w6zJPse6NHmHnvXJo3_ceLK4CZBiX7U"
OUTPUT_DIR = "/Volumes/mvp/[00]交易场/Full Trading Courses"
COOKIE_FILE = "www.youtube.com_cookies.txt"


# ================================================

def refresh_cookies():
    """自动刷新 Cookie"""
    print("🔄 Cookie 已失效，正在自动重新导出...")
    try:
        cmd = f'yt-dlp --cookies-from-browser chrome --cookies "{COOKIE_FILE}" "https://www.youtube.com"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if os.path.exists(COOKIE_FILE) and os.path.getsize(COOKIE_FILE) > 500:
            print(f"✅ Cookie 刷新成功 ({os.path.getsize(COOKIE_FILE)} bytes)")
            return True
        else:
            print("❌ Cookie 导出失败")
            return False
    except Exception as e:
        print(f"❌ 刷新失败: {e}")
        return False


def download_playlist():
    print("🚀 开始下载播放列表 (1080p + 字幕)\n")

    while True:  # 自动重试循环
        ydl_opts = {
            'outtmpl': os.path.join(OUTPUT_DIR, '%(playlist_index)02d - %(title)s.%(ext)s'),
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            'merge_output_format': 'mp4',

            'cookiefile': COOKIE_FILE,

            # 字幕
            'writesubtitles': True,
            'writeautomaticsub': True,
            'allsubtitles': True,
            # 'subtitleslangs': ['en', 'zh'],
            'subtitlesformat': 'srt/best',

            'extractor_args': {'youtube': {'player_client': ['web']}},
            'ignoreerrors': True,
            'retries': 20,
            'fragment_retries': 20,
            'concurrent_fragment_downloads': 3,
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([URL])
            print("\n🎉 下载完成！")
            break

        except Exception as e:
            err_str = str(e).lower()
            if "cookies are no longer valid" in err_str or "sign in to confirm" in err_str:
                print("⚠️  Cookie 失效，尝试自动刷新...")
                if not refresh_cookies():
                    print("❌ 自动刷新失败，请手动在 Chrome 中登录 YouTube 后重试")
                    break
                time.sleep(3)  # 等待一下
            else:
                print(f"❌ 其他错误: {e}")
                break


if __name__ == "__main__":
    download_playlist()