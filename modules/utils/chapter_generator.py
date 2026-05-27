import os
from loguru import logger
from openai import OpenAI

def generate_chapters(srt_content, config=None, prompt_template=None):
    """
    根据字幕内容生成视频章节。
    """
    if not srt_content:
        return ""

    logger.info("开始生成视频章节...")

    # 从配置读取 API Key
    api_key = config.get('api_key') if config else None
    if not api_key or "your-openai-api-key" in api_key:
        logger.warning("未配置有效的 API Key，跳过章节生成。")
        return ""

    model_name = config.get('model', 'gpt-4o') if config else 'gpt-4o'
    base_url = config.get('api_base', 'https://api.openai.com/v1') if config else 'https://api.openai.com/v1'
    client = OpenAI(api_key=api_key, base_url=base_url)

    if prompt_template:
        prompt = prompt_template.format(srt_text=srt_content[:10000])
    else:
        prompt = f"根据以下字幕内容生成视频章节列表，格式为 HH:MM:SS 标题：\n\n{srt_content[:10000]}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        chapters = response.choices[0].message.content.strip()
        logger.success("章节生成完成。")
        return chapters

    except Exception as e:
        logger.error(f"章节生成失败: {e}")
        return ""
