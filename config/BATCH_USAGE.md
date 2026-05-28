# 批次提示词使用说明

## 📋 问题解答

### Q1: `subtitle_only` 模式下是否会使用批次提示词？

**✅ 是的，现在会正确使用批次提示词！**

修复后的代码逻辑：
```python
# 确定提示词目录（优先使用批次提示词）
if batch_name:
    batches_dir = config.get('paths', {}).get('batches_dir', ...)
    prompts_dir = os.path.join(batches_dir, batch_name, "prompts")
    logger.info(f"  使用批次提示词: {batch_name}")
else:
    prompts_dir = config['paths'].get('prompts_dir', 'config/prompts')
    logger.info(f"  使用全局提示词")
```

### Q2: 是否会调用大模型修正识别的字幕？

**✅ 是的，会调用大模型修正字幕！**

在 `subtitle_only` 模式下的处理流程（STEP 4）：

1. **检查 API Key 配置**（第 245 行）
   ```python
   if config.get('llm', {}).get('api_key'):
   ```

2. **加载提示词模板**（第 248-260 行）
   - 如果指定了批次：使用 `config/batches/{batch_name}/prompts/asr_fix.txt`
   - 如果没有批次：使用 `config/prompts/asr_fix.txt`

3. **调用 LLM 修正**（第 262-266 行）
   ```python
   fused_results = fuse_asr_result(
       asr_results,
       config=config.get('llm', {}),
       prompt_template=prompt_template
   )
   ```

4. **修正内容**：
   - ✅ 联系上下文修正错别字
   - ✅ 统一术语翻译
   - ✅ 优化标点符号
   - ✅ 改善句子流畅度

### Q3: 系统什么时候创建批次目录？

**批次目录在 main 函数启动时自动创建**（第 575-617 行）：

#### 创建时机：
```
用户运行命令 → main() 函数启动 → 检查批次名称 → 创建批次目录
```

#### 具体流程：

1. **检查是否指定批次**（第 575 行）
   ```python
   if final_batch_name:
       logger.info(f"\n检查批次配置: {final_batch_name}")
   ```

2. **如果批次目录不存在，创建它**（第 583-594 行）
   ```python
   if not os.path.exists(batch_dir):
       logger.info(f"批次目录不存在，正在创建: {batch_dir}")
       
       # 创建批次配置
       batch_info = prompt_manager.create_batch_prompts(
           batch_name=final_batch_name,
           video_topics=video_topics
       )
   ```

3. **验证并补全批次文件**（第 598-617 行）
   - 检查 `prompts/` 目录是否存在
   - 检查 `terminology.json` 是否存在
   - 如果缺失，自动从全局配置复制

#### 创建的目录结构：
```
config/batches/Photon_Trading/
├── prompts/                    # 从 config/prompts/ 复制
│   ├── asr_fix.txt            # ASR 修正提示词
│   ├── translation.txt        # 翻译提示词
│   ├── tts_prep.txt          # TTS 预处理提示词
│   ├── chapters.txt          # 章节生成提示词
│   ├── resegmentation.txt    # 重分段提示词
│   └── semantic_segmentation.txt
├── terminology.json           # 从 config/terminology.json 复制
└── batch_config.yaml          # 批次配置文件
```

## 🚀 使用方式

### 方式 1: 在代码中配置（推荐）

编辑 [main.py](file:///Users/seven/seven_home/code/main.py) 第 706 行：
```python
BATCH_NAME = "Photon_Trading"  # 设置批次名称
```

然后运行：
```bash
python main.py
```

### 方式 2: 使用命令行参数

```bash
# 使用批次
python main.py /path/to/videos --batch Photon_Trading

# 不使用批次（使用全局配置）
python main.py /path/to/videos
```

## 📊 完整处理流程

### `subtitle_only` 模式流程：

```
视频文件
  ↓
[STEP 1] 提取音频
  ↓
[STEP 2] VAD 语音切片
  ↓
[STEP 3] ASR 识别 (WhisperX)
  ↓
[STEP 4] LLM 修正字幕文本 ⭐
  ├─ 加载批次提示词: config/batches/Photon_Trading/prompts/asr_fix.txt
  ├─ 调用 DeepSeek-V3 API
  └─ 修正错别字、优化表达
  ↓
[STEP 5] 跳过（不需要翻译）
  ↓
[STEP 6] 生成 SRT 字幕
  ↓
输出: outputs/video_name.srt
```

### 其他模式的额外步骤：

- **`subtitle_bilingual`**: STEP 5 翻译为中英双语
- **`subtitle_chinese`**: STEP 5 翻译为纯中文
- **`tts_no_subtitle`**: 额外生成 TTS 配音
- **`tts_with_subtitle`**: 生成 TTS + 合并字幕到视频

## ✨ 批次提示词的优势

### 1. 差异化定制
每个批次可以有专属的提示词：
```
Batch1 (教育培训): 强调专业术语准确性
Batch2 (娱乐休闲): 强调口语化表达
Batch3 (科技产品): 强调技术名词规范
```

### 2. 独立术语表
每个批次维护自己的术语：
```json
// config/batches/Photon_Trading/terminology.json
{
  "ICT": "内圈交易者",
  "FVG": "公允价值缺口",
  "OB": "订单块"
}
```

### 3. 迭代优化
可以针对特定批次不断优化提示词，不影响其他批次。

## 🔍 日志示例

运行时你会看到这样的日志：

```
🎬 统一视频处理工具
============================================================

📂 输入路径: /Volumes/mvp/交易场/光子交易/合集·【光子交易】从0到考试资助账号 4.0
📤 输出目录: /Volumes/mvp/交易场/光子交易/合集·【光子交易】从0到考试资助账号 4.0
🔧 处理模式: subtitle_only
📦 批次名称: Photon_Trading
💡 模式说明: 仅生成中文字幕（ASR 识别）

共找到 10 个视频文件待处理。

检查批次配置: Photon_Trading
批次目录已存在: /Users/seven/seven_home/code/config/batches/Photon_Trading
批次配置验证完成: Photon_Trading

配置信息:
  - 处理模式: subtitle_only
  - ASR 设备: cpu
  - 模型大小: base
  - 采样率: 16000
  - 并行进程数: 2

🚀 启动多进程处理模式（2 个进程）

[STEP 1/6] 提取音频...
[STEP 2/6] 语音活动检测 (VAD)...
[STEP 3/6] 自动语音识别 (ASR)...
[STEP 4/6] LLM 修正字幕文本...
  使用批次提示词: Photon_Trading
  加载提示词模板: asr_fix.txt
LLM 修正完成
[STEP 6/6] 生成 SRT 字幕...
SRT 字幕生成完成: outputs/video1.srt
```

## 📝 修改批次提示词

如果需要优化某个批次的提示词：

1. **编辑批次提示词**：
   ```bash
   vim config/batches/Photon_Trading/prompts/asr_fix.txt
   ```

2. **添加批次专属术语**：
   ```bash
   vim config/batches/Photon_Trading/terminology.json
   ```

3. **重新运行处理**：
   ```bash
   python main.py --batch Photon_Trading
   ```

## ⚠️ 注意事项

1. **API Key 必须配置**
   - 如果没有配置 `llm.api_key`，会跳过 LLM 修正步骤
   - 在 [config.yaml](file:///Users/seven/seven_home/code/config.yaml) 第 52 行配置

2. **批次提示词优先级**
   - 有批次：使用 `config/batches/{batch_name}/prompts/`
   - 无批次：使用 `config/prompts/`

3. **自动创建机制**
   - 首次使用批次时自动创建目录
   - 后续运行直接使用，不会覆盖已有文件

4. **多进程支持**
   - 所有批次处理都使用多进程
   - 并行数由 `global.max_concurrency.video_processor` 控制

## 🎯 总结

✅ **`subtitle_only` 模式会使用批次提示词**  
✅ **会调用大模型修正字幕（需配置 API Key）**  
✅ **批次目录在 main 函数启动时自动创建**  
✅ **优先使用批次提示词，其次使用全局提示词**  
