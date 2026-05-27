import os
from loguru import logger
from openai import OpenAI

def process_tts_text(text, config=None, prompt_template=None):
    """
    将文本中的数字、日期等转换为适合 TTS 朗读的中文格式。
    """
    if not text:
        return ""

    # 从配置读取 API Key
    api_key = config.get('api_key') if config else None
    if not api_key or "your-openai-api-key" in api_key:
        return text

    model_name = config.get('model', 'gpt-4o') if config else 'gpt-4o'
    base_url = config.get('api_base', 'https://api.openai.com/v1') if config else 'https://api.openai.com/v1'
    client = OpenAI(api_key=api_key, base_url=base_url)

    if prompt_template:
        prompt = prompt_template.format(current_sentence=text)
    else:
        prompt = f"""
        请将以下文本中的数字、日期、符号等转换为适合 TTS 朗读的中文口语格式。
        
        要求：
        1. 只返回处理后的文本内容。
        2. 严禁包含任何解释、注脚、引言或“修正如下”等字样。
        3. 如果文本不包含需要转换的内容，原样返回即可。
        
        待处理文本：
        {text}
        """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        processed_text = response.choices[0].message.content.strip()
        
        # 工业级：二次清洗，移除常见的 LLM “碎碎念”
        import re
        # 移除 （注：...） 或 (Note: ...) 及其内容
        processed_text = re.sub(r'[\(（](注|Note).*?[\)）]', '', processed_text, flags=re.DOTALL)
        # 移除空行
        processed_text = "\n".join([line.strip() for line in processed_text.split('\n') if line.strip()])
        
        processed_text = processed_text.strip()
        
        # 关键修复：如果处理后的文本为空或与原文相同且不需要转换，直接返回原文
        if not processed_text:
            logger.warning(f"LLM 返回空文本，使用原文: {text[:50]}...")
            return text
        
        # 检查是否只是返回了“原样返回”等提示语
        common_responses = [
            "原样返回即可", "无需转换", "不需要转换",
            "The text does not contain", "No conversion needed"
        ]
        if any(resp in processed_text for resp in common_responses):
            logger.debug(f"LLM 返回提示语，使用原文: {text[:50]}...")
            return text
        
        return processed_text
    except Exception as e:
        logger.error(f"TTS 文本预处理失败: {e}")
        return text
