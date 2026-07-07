#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线辅助工具：提示词目录解析、LLM 配置构建。
"""

import os


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_llm_config(config):
    """
    构建带有翻译引擎配置的 LLM 配置。

    Args:
        config: 完整配置字典

    Returns:
        dict: 合并后的 LLM 配置
    """
    llm_config = dict(config.get('llm', {}))
    translate_cfg = config.get('translate', {})
    if translate_cfg:
        llm_config['provider'] = translate_cfg.get('provider', 'llm')
        llm_config['google_delay'] = translate_cfg.get('google_delay', 0.3)
    return llm_config


def load_prompt_template(prompts_dir, prompt_name):
    """加载指定名称的提示词模板"""
    prompt_path = os.path.join(prompts_dir, prompt_name)
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""
