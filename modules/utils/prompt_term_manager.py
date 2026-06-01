#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示词和术语管理系统
支持为大模型维护和更新 prompts 和 terminology 文件
"""

import os
import json
import yaml
from pathlib import Path
from loguru import logger
from openai import OpenAI


class PromptTermManager:
    """提示词和术语管理器"""
    
    def __init__(self, config_path=None):
        """
        初始化管理器
        
        Args:
            config_path: 配置文件路径（相对于项目根目录）
        """
        # ── 项目根目录（modules/utils/prompt_term_manager.py → 上级3层）──
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        if config_path is None:
            config_path = os.path.join("config", "config.yaml")
        
        self.config = self._load_config(config_path)
        
        # 初始化 LLM 客户端
        llm_config = self.config.get('llm', {})
        api_key = llm_config.get('api_key', '')
        api_base = llm_config.get('api_base', 'https://api.siliconflow.cn/v1')
        self.model = llm_config.get('model', 'deepseek-ai/DeepSeek-V3')
        
        if api_key and api_key != "your-openai-api-key":
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.llm_available = True
        else:
            self.client = None
            self.llm_available = False
            logger.warning("未配置有效的 API Key，LLM 功能不可用")
        
        # 路径配置（从配置文件读取或使用默认值）
        self.config_dir = os.path.join(self.project_root, "config")
        
        # 从配置文件读取路径
        paths_config = self.config.get('paths', {})
        self.prompts_dir = paths_config.get('prompts_dir', os.path.join(self.config_dir, "prompts"))
        self.terminology_file = paths_config.get('terminology_file', os.path.join(self.config_dir, "terminology.json"))
        self.batches_dir = paths_config.get('batches_dir', os.path.join(self.config_dir, "batches"))
        
        # 确保目录存在
        os.makedirs(self.prompts_dir, exist_ok=True)
        os.makedirs(self.batches_dir, exist_ok=True)
    
    def _load_config(self, config_path):
        """加载配置文件"""
        config_path = os.path.join(self.project_root, config_path)
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_terminology(self):
        """加载术语表"""
        if os.path.exists(self.terminology_file):
            with open(self.terminology_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_terminology(self, terminology):
        """保存术语表"""
        with open(self.terminology_file, 'w', encoding='utf-8') as f:
            json.dump(terminology, f, ensure_ascii=False, indent=2)
        logger.success(f"术语表已保存到: {self.terminology_file}")
    
    def load_prompt(self, prompt_name):
        """
        加载提示词模板
        
        Args:
            prompt_name: 提示词名称（不含 .txt 后缀）
        
        Returns:
            str: 提示词内容
        """
        prompt_file = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def save_prompt(self, prompt_name, content):
        """
        保存提示词模板
        
        Args:
            prompt_name: 提示词名称（不含 .txt 后缀）
            content: 提示词内容
        """
        prompt_file = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.success(f"提示词已保存到: {prompt_file}")
    
    def list_prompts(self):
        """列出所有提示词文件"""
        if not os.path.exists(self.prompts_dir):
            return []
        
        prompts = []
        for file in os.listdir(self.prompts_dir):
            if file.endswith('.txt'):
                prompts.append(file[:-4])  # 去除 .txt 后缀
        return sorted(prompts)
    
    def extract_terms_from_text(self, text, domain="general"):
        """
        从文本中提取专业术语
        
        Args:
            text: 待分析的文本
            domain: 领域（如：trading, technology, medical 等）
        
        Returns:
            dict: 提取的术语字典 {英文: 中文}
        """
        if not self.llm_available:
            logger.error("LLM 不可用，无法提取术语")
            return {}
        
        prompt = (
            f"从文本提取专业术语→中文。忽略常见词。领域：{domain}\n"
            f'返回JSON：{{"英文术语":"中文翻译"}}，无术语返回{{}}。\n\n'
            f"{text[:2000]}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            # 尝试解析 JSON
            try:
                # 查找 JSON 代码块
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = content
                
                terms = json.loads(json_str)
                logger.info(f"成功提取 {len(terms)} 个术语")
                return terms
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析失败: {e}")
                logger.debug(f"原始响应: {content}")
                return {}
                
        except Exception as e:
            err_msg = str(e)
            # 检测致命 API 错误，禁用 LLM 功能
            if any(kw in err_msg.lower() for kw in ('balance', 'insufficient', 'invalid', 'unauthorized', '403', '401')):
                self.llm_available = False
                logger.warning("检测到致命 API 错误（余额不足/Key无效），已禁用后续 LLM 调用")
            logger.error(f"术语提取失败: {e}")
            return {}
    
    def optimize_prompt(self, prompt_name, feedback="", examples=""):
        """
        优化提示词模板
        
        Args:
            prompt_name: 提示词名称
            feedback: 用户反馈（当前提示词的问题）
            examples: 期望的输出示例
        
        Returns:
            str: 优化后的提示词
        """
        if not self.llm_available:
            logger.error("LLM 不可用，无法优化提示词")
            return ""
        
        current_prompt = self.load_prompt(prompt_name)
        
        prompt = (
            "优化提示词模板，使指令清晰、具体、无歧义。删除冗余。保留所有占位符变量。\n\n"
            f"当前：\n{current_prompt}\n\n"
            f"反馈：{feedback or '无'}\n期望示例：{examples or '无'}\n\n"
            "只返回优化后的提示词，不解释。"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            optimized = response.choices[0].message.content.strip()
            logger.success("提示词优化完成")
            return optimized
            
        except Exception as e:
            logger.error(f"提示词优化失败: {e}")
            return ""
    
    def create_batch_prompts(self, batch_name, video_topics=None):
        """
        为特定批次视频创建专属的提示词和术语
        
        Args:
            batch_name: 批次名称（如：ICT_Trading_Batch1）
            video_topics: 视频主题列表
        
        Returns:
            dict: 创建的批处理配置
        """
        # 创建批处理目录（使用优化后的目录结构）
        batch_dir = os.path.join(self.batches_dir, batch_name)
        os.makedirs(batch_dir, exist_ok=True)
        
        # 复制基础提示词
        base_prompts = self.list_prompts()
        batch_prompts_dir = os.path.join(batch_dir, "prompts")
        os.makedirs(batch_prompts_dir, exist_ok=True)
        
        for prompt_name in base_prompts:
            content = self.load_prompt(prompt_name)
            self._save_to_file(os.path.join(batch_prompts_dir, f"{prompt_name}.txt"), content)
        
        # 创建批次专属术语表
        batch_terminology = {}
        if video_topics:
            # 根据主题生成相关术语
            topics_text = "\n".join(video_topics)
            extracted_terms = self.extract_terms_from_text(topics_text, domain="trading")
            batch_terminology.update(extracted_terms)
        
        # 合并全局术语表
        global_terminology = self.load_terminology()
        batch_terminology.update(global_terminology)
        
        # 保存批次术语表
        batch_terminology_file = os.path.join(batch_dir, "terminology.json")
        with open(batch_terminology_file, 'w', encoding='utf-8') as f:
            json.dump(batch_terminology, f, ensure_ascii=False, indent=2)
        
        # 创建批次配置文件
        batch_config = {
            "batch_name": batch_name,
            "created_at": "",
            "video_topics": video_topics or [],
            "prompts_dir": "prompts",
            "terminology_file": "terminology.json",
            "description": f"Batch: {batch_name}"
        }
        
        batch_config_file = os.path.join(batch_dir, "batch_config.yaml")
        with open(batch_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(batch_config, f, allow_unicode=True, default_flow_style=False)
        
        logger.success(f"批处理配置已创建: {batch_dir}")
        return {
            "batch_dir": batch_dir,
            "prompts_count": len(base_prompts),
            "terminology_count": len(batch_terminology)
        }
    
    def merge_terminology(self, new_terms, auto_save=True):
        """
        合并新术语到全局术语表
        
        Args:
            new_terms: 新术语字典
            auto_save: 是否自动保存
        
        Returns:
            dict: 合并后的术语表
        """
        current_terms = self.load_terminology()
        
        # 合并术语（新术语优先）
        merged = {**current_terms, **new_terms}
        
        if auto_save:
            self.save_terminology(merged)
        
        logger.info(f"术语表合并完成: 原有 {len(current_terms)} 个，新增 {len(new_terms)} 个，总计 {len(merged)} 个")
        return merged
    
    def review_and_add_terms(self, text, domain="general", auto_save=True):
        """
        审查文本并添加新术语
        
        Args:
            text: 待审查的文本
            domain: 领域
            auto_save: 是否自动保存
        
        Returns:
            dict: 新增的术语
        """
        # 提取术语
        extracted = self.extract_terms_from_text(text, domain)
        
        if not extracted:
            logger.info("未提取到新术语")
            return {}
        
        # 过滤已存在的术语
        current_terms = self.load_terminology()
        new_terms = {k: v for k, v in extracted.items() if k not in current_terms}
        
        if new_terms:
            logger.info(f"发现 {len(new_terms)} 个新术语:")
            for term, translation in new_terms.items():
                logger.info(f"  - {term}: {translation}")
            
            # 合并并保存
            self.merge_terminology(new_terms, auto_save=auto_save)
        else:
            logger.info("所有术语已存在于术语表中")
        
        return new_terms
    
    def _save_to_file(self, file_path, content):
        """保存内容到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    """主函数 - 演示用法"""
    manager = PromptTermManager()
    
    print("\n" + "="*60)
    print("📝 提示词和术语管理系统")
    print("="*60 + "\n")
    
    # 1. 列出所有提示词
    print("📋 当前提示词列表:")
    prompts = manager.list_prompts()
    for p in prompts:
        print(f"  - {p}")
    print()
    
    # 2. 查看术语表
    print("📖 当前术语表:")
    terminology = manager.load_terminology()
    for term, trans in list(terminology.items())[:5]:
        print(f"  - {term}: {trans}")
    if len(terminology) > 5:
        print(f"  ... 共 {len(terminology)} 个术语")
    print()
    
    # 3. 从文本中提取术语（示例）
    sample_text = """
    In ICT trading concepts, we look for Fair Value Gap (FVG) and Order Blocks.
    The Market Maker often creates liquidity pools at Premium and Discount zones.
    Candle patterns help identify potential reversals in S&P and NASDAQ indices.
    """
    
    print("🔍 从示例文本中提取术语:")
    extracted = manager.extract_terms_from_text(sample_text, domain="trading")
    if extracted:
        for term, trans in extracted.items():
            print(f"  - {term}: {trans}")
    print()
    
    # 4. 创建批处理配置（示例）
    print("📦 创建批处理配置示例:")
    batch_info = manager.create_batch_prompts(
        batch_name="ICT_Trading_Demo",
        video_topics=["ICT Concepts", "Fair Value Gap", "Order Blocks"]
    )
    print(f"  - 批处理目录: {batch_info['batch_dir']}")
    print(f"  - 提示词数量: {batch_info['prompts_count']}")
    print(f"  - 术语数量: {batch_info['terminology_count']}")
    print()
    
    print("="*60)
    print("✅ 演示完成")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
