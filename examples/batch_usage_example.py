#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批次管理使用示例
演示如何创建和管理视频处理批次
"""

from prompt_term_manager import PromptTermManager


def example_basic_usage():
    """基础用法示例"""
    print("\n" + "="*60)
    print("示例 1: 基础用法")
    print("="*60 + "\n")
    
    manager = PromptTermManager()
    
    # 1. 查看当前提示词
    print("📋 当前提示词列表:")
    prompts = manager.list_prompts()
    for p in prompts:
        print(f"  - {p}")
    print()
    
    # 2. 查看术语表
    print("📖 当前术语表（前5个）:")
    terms = manager.load_terminology()
    for i, (term, trans) in enumerate(list(terms.items())[:5], 1):
        print(f"  {i}. {term}: {trans}")
    print(f"  ... 共 {len(terms)} 个术语\n")


def example_create_batch():
    """创建批次示例"""
    print("\n" + "="*60)
    print("示例 2: 创建批次配置")
    print("="*60 + "\n")
    
    manager = PromptTermManager()
    
    # 创建批次
    batch_name = "ICT_Trading_Batch1"
    video_topics = [
        "ICT Concepts Introduction",
        "Fair Value Gap Trading Strategy",
        "Order Block Identification"
    ]
    
    print(f"📦 创建批次: {batch_name}")
    print(f"📝 视频主题:")
    for topic in video_topics:
        print(f"  - {topic}")
    print()
    
    batch_info = manager.create_batch_prompts(
        batch_name=batch_name,
        video_topics=video_topics
    )
    
    print(f"✅ 批次创建成功!")
    print(f"  - 目录: {batch_info['batch_dir']}")
    print(f"  - 提示词数量: {batch_info['prompts_count']}")
    print(f"  - 术语数量: {batch_info['terminology_count']}\n")


def example_extract_terms():
    """从文本提取术语示例"""
    print("\n" + "="*60)
    print("示例 3: 从视频转录中提取术语")
    print("="*60 + "\n")
    
    manager = PromptTermManager()
    
    # 模拟视频转录文本
    transcript = """
    Today we'll learn about ICT trading concepts. 
    We look for Fair Value Gap (FVG) in the market structure.
    Order Blocks are key areas where institutional orders are placed.
    The Market Maker often creates liquidity pools at Premium and Discount zones.
    Candle patterns help identify potential reversals in S&P and NASDAQ indices.
    """
    
    print("📄 分析文本:")
    print(f"  {transcript[:100]}...\n")
    
    print("🔍 提取术语...")
    new_terms = manager.review_and_add_terms(
        transcript, 
        domain="trading",
        auto_save=False  # 不自动保存，先预览
    )
    
    if new_terms:
        print(f"\n✅ 发现 {len(new_terms)} 个新术语:")
        for term, trans in new_terms.items():
            print(f"  - {term}: {trans}")
        
        # 询问是否保存
        print("\n💡 提示: 设置 auto_save=True 可自动保存到术语表\n")
    else:
        print("\nℹ️  未提取到新术语\n")


def example_optimize_prompt():
    """优化提示词示例"""
    print("\n" + "="*60)
    print("示例 4: 优化提示词模板")
    print("="*60 + "\n")
    
    manager = PromptTermManager()
    
    prompt_name = "translation"
    
    print(f"📝 优化提示词: {prompt_name}")
    print(f"💬 反馈: 翻译结果过于直译，需要更口语化\n")
    
    # 注意: 这需要有效的 API Key
    if manager.llm_available:
        optimized = manager.optimize_prompt(
            prompt_name=prompt_name,
            feedback="翻译结果过于直译，需要更自然的口语表达",
            examples="原文: 'How are you?' -> 译文: '你好吗？'（而非'你怎么样？'）"
        )
        
        if optimized:
            print("✅ 优化完成!")
            print("\n优化后的提示词（前200字符）:")
            print(f"  {optimized[:200]}...\n")
            print("💡 提示: 使用 save_prompt() 保存优化后的提示词\n")
    else:
        print("⚠️  LLM 不可用，请配置 API Key 后重试\n")


def example_batch_workflow():
    """完整批次工作流程示例"""
    print("\n" + "="*60)
    print("示例 5: 完整批次工作流程")
    print("="*60 + "\n")
    
    manager = PromptTermManager()
    
    # 步骤 1: 创建批次
    batch_name = "Trading_Tutorial_Series"
    print(f"步骤 1: 创建批次 '{batch_name}'")
    batch_info = manager.create_batch_prompts(
        batch_name=batch_name,
        video_topics=["Trading Basics", "Technical Analysis"]
    )
    print(f"  ✅ 批次已创建: {batch_info['batch_dir']}\n")
    
    # 步骤 2: 处理视频并提取术语
    print("步骤 2: 模拟处理视频并提取术语")
    video_transcript = "We use candlestick patterns to identify market trends."
    new_terms = manager.review_and_add_terms(
        video_transcript,
        domain="trading",
        auto_save=False
    )
    if new_terms:
        print(f"  ✅ 提取到 {len(new_terms)} 个新术语\n")
    
    # 步骤 3: 优化批次提示词
    print("步骤 3: 为批次定制提示词")
    print("  💡 可以针对批次特点优化提示词\n")
    
    print("✅ 批次工作流程完成!\n")
    print("📌 下一步:")
    print(f"  python main.py /path/to/videos --batch {batch_name}\n")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("📦 批次管理系统 - 使用示例")
    print("="*60)
    
    try:
        example_basic_usage()
        example_create_batch()
        example_extract_terms()
        example_optimize_prompt()
        example_batch_workflow()
        
        print("\n" + "="*60)
        print("✅ 所有示例运行完成")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
