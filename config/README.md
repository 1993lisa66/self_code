# 配置目录说明

## 目录结构

```
config/
├── prompts/                    # 基础提示词模板
│   ├── asr_fix.txt            # ASR 修正提示词
│   ├── translation.txt        # 翻译提示词
│   ├── tts_prep.txt          # TTS 文本预处理提示词
│   ├── chapters.txt          # 章节生成提示词
│   ├── resegmentation.txt    # 重分段提示词
│   └── semantic_segmentation.txt  # 语义分段提示词
│
├── terminology.json           # 全局术语表（所有批次共享）
│
└── batches/                   # 批次配置目录
    ├── Batch1/               # 批次1
    │   ├── prompts/          # 批次专属提示词（从基础复制）
    │   ├── terminology.json  # 批次专属术语表（全局+批次特有）
    │   └── batch_config.yaml # 批次配置文件
    ├── Batch2/               # 批次2
    │   └── ...
    └── ...
```

## 使用说明

### 1. 基础配置（全局）

- **提示词模板**: `config/prompts/*.txt`
  - 所有批次共享的基础提示词
  - 修改后会影响到新创建的批次
  
- **术语表**: `config/terminology.json`
  - 全局术语，所有批次共享
  - 格式: `{"英文术语": "中文翻译"}`

### 2. 批次配置

每个批次有独立的配置目录，位于 `config/batches/{batch_name}/`

**创建批次**:
```python
from prompt_term_manager import PromptTermManager

manager = PromptTermManager()
manager.create_batch_prompts(
    batch_name="ICT_Trading_Batch1",
    video_topics=["ICT Concepts", "FVG", "Order Blocks"]
)
```

**使用批次处理视频**:
```bash
# 命令行方式
python main.py /path/to/videos --batch ICT_Trading_Batch1

# 或在 main.py 中配置
BATCH_NAME = "ICT_Trading_Batch1"
```

### 3. 工作流程

1. **维护基础配置**
   - 编辑 `config/prompts/` 中的提示词模板
   - 更新 `config/terminology.json` 添加通用术语

2. **创建批次**
   - 为每批视频创建专属配置
   - 系统自动复制基础提示词和术语表
   - 可根据视频主题提取专属术语

3. **处理视频**
   - 指定批次名称
   - 系统自动加载批次配置
   - 处理过程中动态更新批次术语表

## 优势

- ✅ **分层管理**: 基础配置 + 批次专属配置
- ✅ **灵活定制**: 每个批次可以有独特的提示词和术语
- ✅ **易于维护**: 清晰的目录结构，便于查找和修改
- ✅ **自动同步**: 创建批次时自动复制基础配置
- ✅ **动态更新**: 处理过程中可实时更新术语表
