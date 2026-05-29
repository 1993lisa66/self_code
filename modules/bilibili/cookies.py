"""B站 Cookie 管理 — 读取 / 验证 / 帮助指引"""

import logging
from pathlib import Path

from .config import COOKIE_FILE

logger = logging.getLogger(__name__)


def cookie_file_exists() -> bool:
    return COOKIE_FILE.exists() and COOKIE_FILE.stat().st_size > 100


def cookie_has_sessdata() -> bool:
    if not cookie_file_exists():
        return False
    content = COOKIE_FILE.read_text(encoding="utf-8")
    return "SESSDATA" in content


def generate_cookie_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              B站 Cookie 获取指引                             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  B站字幕下载需要登录态，请按以下步骤获取 Cookie 文件：       ║
║                                                              ║
║  方法一：浏览器扩展导出（推荐）                              ║
║    1. 安装 Chrome/Edge 扩展 "Get cookies.txt LOCALLY"         ║
║    2. 在浏览器中登录 bilibili.com                            ║
║    3. 打开任意 B站 页面，点击该扩展图标                      ║
║    4. 点击 "Export" 导出 cookies.txt                         ║
║    5. 将导出的文件放到项目根目录:                             ║
║       www.bilibili.com_cookies.txt                           ║
║                                                              ║
║  方法二：浏览器开发者工具手动复制                            ║
║    1. 在浏览器中登录 bilibili.com                            ║
║    2. 按 F12 打开开发者工具                                  ║
║    3. 切换到 Application → Cookies → bilibili.com            ║
║    4. 找到 SESSDATA 和 bili_jct，记下其值                   ║
║    5. 按以下格式创建文件（域名/字段用 Tab 分隔）：           ║
║       # Netscape HTTP Cookie File                            ║
║       .bilibili.com  TRUE  /  FALSE  0  SESSDATA  你的值     ║
║       .bilibili.com  TRUE  /  FALSE  0  bili_jct  你的值     ║
║                                                              ║
║  关键字段说明：                                              ║
║    SESSDATA   — 登录凭证（必需）                             ║
║    bili_jct   — CSRF Token（必需）                           ║
║    DedeUserID — 用户 ID（可选，用于验证）                    ║
║                                                              ║
║  Cookie 文件位置:                                            ║
║    www.bilibili.com_cookies.txt（项目根目录）                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
