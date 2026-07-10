#!/bin/bash
# Bilibili Video Downloader - Chrome Extension API Server
# Start script for macOS/Linux

echo "=================================="
echo "B站视频下载器 Chrome扩展 API服务"
echo "=================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# Check dependencies
echo "检查依赖..."
if ! python3 -c "import flask" &> /dev/null; then
    echo "📦 安装 Flask..."
    pip3 install flask flask-cors
fi

if ! python3 -c "import yt_dlp" &> /dev/null; then
    echo "📦 安装 yt-dlp..."
    pip3 install yt-dlp
fi

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg 已安装"
else
    echo "⚠️  FFmpeg 未安装，视频合并可能失败"
fi

# Start the server
echo ""
echo "🚀 启动API服务..."
echo "   访问 http://localhost:5000 查看服务状态"
echo "   按 Ctrl+C 停止服务"
echo ""

python3 api_server.py