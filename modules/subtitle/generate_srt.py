import os
import srt
from datetime import timedelta
from loguru import logger

def generate_srt(translated_results, output_path="outputs/final.srt"):
    """
    将翻译结果转换为 SRT 字幕文件。
    """
    logger.info(f"正在生成 SRT 字幕: {output_path}")

    subtitles = []
    total_duration = 0
    
    for i, seg in enumerate(translated_results):
        # 关键修复：确保时间轴严格递增且无重叠
        if i > 0:
            prev_seg = translated_results[i-1]
            if seg['start'] < prev_seg['end']:
                logger.warning(f"片段 {i} 与片段 {i-1} 时间重叠: [{prev_seg['start']:.2f}-{prev_seg['end']:.2f}] vs [{seg['start']:.2f}-{seg['end']:.2f}]")
                # 调整当前片段的开始时间为上一个片段的结束时间
                seg['start'] = prev_seg['end']
        
        segment_duration = seg['end'] - seg['start']
        total_duration += segment_duration
        
        # 创建 srt.Subtitle 对象
        sub = srt.Subtitle(
            index=i + 1,
            start=timedelta(seconds=seg["start"]),
            end=timedelta(seconds=seg["end"]),
            content=seg.get("translated_text", seg["text"])
        )
        subtitles.append(sub)
    
    logger.info(f"字幕总时长: {total_duration:.2f}s, 共 {len(subtitles)} 个片段")

    # 序列化为字符串
    srt_content = srt.compose(subtitles)

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    logger.success(f"SRT 字幕生成完成: {output_path}")
    return output_path
