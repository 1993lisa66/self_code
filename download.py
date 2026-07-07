import os
import time
import subprocess
from yt_dlp import YoutubeDL

# ====================== 默认配置 ======================
# 不传参数时使用的默认值，可直接修改这里
DEFAULT_URL = "https://www.youtube.com/watch?v=t8JhvEbifR8"
DEFAULT_OUTPUT = "/Volumes/mvp/[00]交易场/edgeskool"
DEFAULT_QUALITY = "2160p"  # 360p / 480p / 720p / 1080p / 1440p / 2160p
COOKIE_FILE = "www.youtube.com_cookies.txt"
# ====================================================


def is_playlist(url):
    """检测 URL 是播放列表还是单个视频"""
    return any(kw in url.lower() for kw in ["playlist", "list="])


def refresh_cookies(cookie_file=COOKIE_FILE, browser="chrome"):
    """自动刷新 YouTube Cookies"""
    print(f"🔄 Cookie 已失效，正在从 {browser} 重新导出...")
    try:
        cmd = (
            f'yt-dlp --cookies-from-browser {browser} '
            f'--cookies "{cookie_file}" '
            f'"https://www.youtube.com"'
        )
        subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        if os.path.exists(cookie_file) and os.path.getsize(cookie_file) > 500:
            print(f"✅ Cookie 刷新成功 ({os.path.getsize(cookie_file)} bytes)")
            return True
        else:
            print("❌ Cookie 导出失败")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Cookie 导出超时")
        return False
    except Exception as e:
        print(f"❌ 刷新失败: {e}")
        return False


def download_videos(url, output_dir, cookie_file=COOKIE_FILE,
                    quality="1080p", cookie_browser="chrome"):
    """
    下载 YouTube 视频或播放列表。
    自动识别 URL 类型（单视频 / 播放列表），使用对应的命名模板。

    Args:
        url:          YouTube 视频或播放列表 URL
        output_dir:   输出目录
        cookie_file:  Cookie 文件路径
        quality:      视频质量 (如 1080p)
        cookie_browser: 导出 Cookie 的浏览器 (chrome / firefox / edge 等)
    """
    playlist = is_playlist(url)

    if playlist:
        print(f"📋 播放列表: {url}")
        outtmpl = os.path.join(output_dir, '%(playlist_index)02d - %(title)s.%(ext)s')
    else:
        print(f"🎬 单个视频: {url}")
        outtmpl = os.path.join(output_dir, '%(title)s.%(ext)s')

    height = quality.rstrip("p")

    ydl_opts = {
        'outtmpl': outtmpl,
        'format': f'bestvideo[height<={height}]+bestaudio/best[height<={height}]',
        'merge_output_format': 'mp4',

        'cookiefile': cookie_file,

        # 字幕
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'zh'],
        'subtitlesformat': 'srt/best',

        'extractor_args': {'youtube': {'player_client': ['web']}},
        'ignoreerrors': True,
        'retries': 20,
        'fragment_retries': 20,
        'concurrent_fragment_downloads': 3,
    }

    os.makedirs(output_dir, exist_ok=True)

    max_cookie_retries = 3
    for attempt in range(max_cookie_retries):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print("\n🎉 下载完成！")
            return True

        except Exception as e:
            err_str = str(e).lower()
            if "cookies are no longer valid" in err_str or "sign in to confirm" in err_str:
                print(f"⚠️  Cookie 失效 (第 {attempt+1}/{max_cookie_retries} 次尝试)")
                if attempt < max_cookie_retries - 1:
                    if not refresh_cookies(cookie_file, cookie_browser):
                        print("❌ 自动刷新失败，请手动在浏览器中登录 YouTube 后重试")
                        return False
                    time.sleep(3)
                else:
                    print("❌ Cookie 刷新已达最大重试次数，下载失败")
                    return False
            else:
                print(f"❌ 下载出错: {e}")
                return False

    return False


if __name__ == "__main__":
    # 直接修改上方 DEFAULT_URL / DEFAULT_OUTPUT 即可运行
    print(f"📥 目标: {DEFAULT_URL}")
    print(f"📂 输出: {DEFAULT_OUTPUT}")
    print(f"🎨 画质: {DEFAULT_QUALITY}\n")
    download_videos(DEFAULT_URL, DEFAULT_OUTPUT, COOKIE_FILE, DEFAULT_QUALITY)