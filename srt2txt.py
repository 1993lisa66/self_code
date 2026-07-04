#!/usr/bin/env python3
"""将指定文件夹中的 SRT 字幕文件转化为 TXT 文件，仅保留文字部分，合并为一个文件"""

import os
import re

FOLDER = "/Volumes/mvp/交易场/ict/合集·【二创】TradingHub3.0"


def read_file_auto(path: str) -> str:
    """自动检测编码读取文件"""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "gb18030", "gbk", "gb2312", "big5", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_srt(srt_path: str) -> str:
    """解析 SRT 文件，返回纯文字内容"""
    content = read_file_auto(srt_path)

    blocks = re.split(r"\n\s*\n", content.strip())
    lines = []
    for block in blocks:
        block_lines = block.strip().splitlines()
        text_lines = []
        for line in block_lines:
            if re.match(r"^\d+$", line.strip()):
                continue
            if re.match(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}$", line.strip()):
                continue
            text_lines.append(line.strip())
        if text_lines:
            lines.append("".join(text_lines))

    return "\n".join(lines)


def convert_folder(folder: str):
    """将文件夹中所有 .srt 文件的字幕合并为一个 txt"""
    if not os.path.isdir(folder):
        print(f"错误: {folder} 不是有效目录")
        return

    folder_name = os.path.basename(folder.rstrip("/"))
    txt_path = os.path.join(folder, f"{folder_name}.txt")

    all_text = []
    srt_files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(".srt") and not f.startswith("._"))

    if not srt_files:
        print("未找到 SRT 文件")
        return

    for fname in srt_files:
        srt_path = os.path.join(folder, fname)
        text = parse_srt(srt_path)
        if text:
            all_text.append(text)
            print(f"已提取: {fname}")

    merged = "\n".join(all_text)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(merged)

    print(f"\n合并完成: {txt_path} (共 {len(srt_files)} 个字幕文件)")


if __name__ == "__main__":
    convert_folder(FOLDER)
