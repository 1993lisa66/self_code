# 配置文件说明 (config.yaml)

## ✅ 已优化的配置

### 1. 多进程配置清理

**已删除的无用配置**:
- ❌ `asr_selector: 15` - 旧 pipeline.py 使用
- ❌ `english_translator: 15` - 旧 pipeline.py 使用
- ❌ `tts_text_processor: 20` - 旧 pipeline.py 使用
- ❌ `chapter_generator: 1` - 旧 pipeline.py 使用
- ❌ `subtitle_generator: 5` - generate_cn_subtitles.py 使用（已删除）

**保留的配置**:
- ✅ `video_processor: 2` - main.py 统一视频处理器使用

```yaml
global:
  max_concurrency:
    video_processor: 2  # 视频处理器的并行进程数（1=单进程，>1=多进程）
  auto_cleanup_cache: true  # 处理完每个视频后自动清理缓存
```

### 2. 路径配置优化

所有路径都更新为新的目录结构：

```yaml
paths:
  input_dir: input
  output_dir: outputs
  cache_dir: cache
  log_dir: logs
  prompts_dir: config/prompts          # 优化后的提示词目录
  assets_dir: assets
  models_dir: models                   # 本地模型存储目录
  batches_dir: config/batches          # 批次配置目录
  terminology_file: config/terminology.json  # 全局术语表
```

### 3. ASR 术语表路径更新

```yaml
asr:
  terminology_file: config/terminology.json  # 使用新路径
```

## 📋 完整配置说明

### 项目配置
```yaml
project:
  name: TransVoice
```

### 路径配置
- `input_dir`: 输入视频目录
- `output_dir`: 输出结果目录
- `cache_dir`: 缓存目录
- `log_dir`: 日志目录
- `prompts_dir`: 提示词模板目录
- `assets_dir`: 资源文件目录
- `models_dir`: 模型存储目录
- `batches_dir`: 批次配置目录
- `terminology_file`: 全局术语表路径

### 多进程配置
- `video_processor`: 控制同时处理的视频数量
  - `1`: 单进程顺序处理（适合内存有限的机器）
  - `2-5`: 多进程并行处理（推荐值，根据 CPU 核心数调整）
  - `>5`: 高并发处理（需要强大的硬件支持）

### 音频配置
```yaml
audio:
  sample_rate: 16000  # 采样率
  channels: 1         # 声道数
  normalize: true     # 音频标准化
```

### VAD（语音活动检测）配置
```yaml
vad:
  min_speech_duration_ms: 250   # 最小语音持续时间
  min_silence_duration_ms: 300  # 最小静音持续时间
```

### ASR（语音识别）配置
```yaml
asr:
  model_id: "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
  device: cpu
  model_size: base
  voting_models:
    - whisperx
    - glm
  terminology_file: config/terminology.json  # 术语表路径
```

### WhisperX 配置
```yaml
whisperx:
  model: base
  device: cpu
  compute_type: float16
  batch_size: 16
  diarization: true        # 说话人分离
  min_speakers: 1
  max_speakers: 10
  align: true             # 对齐
```

### LLM（大语言模型）配置
```yaml
llm:
  api_key: "sk-xxx"
  api_base: "https://api.siliconflow.cn/v1"
  model: "deepseek-ai/DeepSeek-V3"
  
  # 翻译配置
  translator:
    model: "deepseek-ai/DeepSeek-V3"
    temperature: 0.1
    max_retries: 3
  
  # TTS 文本预处理配置
  tts_processor:
    model: "deepseek-ai/DeepSeek-V3"
    temperature: 0.1
  
  # 章节生成配置
  chapter_generator:
    model: "deepseek-ai/DeepSeek-V3"
    temperature: 0.1
  
  # 提示词文件映射
  prompts:
    asr_fix: asr_fix.txt
    translation: translation.txt
    chapters: chapters.txt
    tts_prep: tts_prep.txt
    resegmentation: resegmentation.txt
    semantic_segmentation: semantic_segmentation.txt
```

### 翻译配置
```yaml
translate:
  target_language: zh  # 目标语言
```

### TTS（文本转语音）配置
```yaml
tts:
  provider: edge  # TTS 提供商
  voice: zh-CN-XiaoxiaoNeural  # 语音选择
  speed_limit: 1.5  # 最大语速倍数
```

**可用语音选项**:
- **女声**: XiaoxiaoNeural, XiaoyiNeural, Luna, XiaochenNeural, XiaohanNeural, 等
- **男声**: YunxiNeural, YunjianNeural, YunfengNeural, YunhaoNeural, 等

**视频类型推荐**:
- 教育培训: XiaoxiaoNeural
- 科技产品: YunxiNeural
- 儿童内容: XiaoyiNeural
- 纪录片: YunfengNeural
- 新闻资讯: XiaochenNeural
- 娱乐休闲: XiaoyiNeural / YunyeNeural

### 视频配置
```yaml
video:
  subtitle_font_size: 18
  codec: libx264
  crf: 23
  preset: fast
  burn_subtitles: false  # 是否烧录字幕到视频中
  subtitle_position: "bottom"  # 字幕位置
  audio_mode: "tts_only"  # 音频模式
```

**音频模式选项**:
- `tts_only`: 仅中文配音
- `mix`: 混合原声+配音
- `bgm_mix`: 背景音乐+配音
- `original`: 仅原声

## 🔧 配置建议

### 性能优化
1. **CPU 较弱**: 设置 `video_processor: 1`
2. **CPU 中等**: 设置 `video_processor: 2-3`
3. **CPU 强大**: 设置 `video_processor: 4-5`

### 内存优化
如果处理大视频时内存不足：
- 减少 `video_processor` 的值
- 减小 `whisperx.batch_size`
- 设置 `whisperx.compute_type: int8_float16`

### 质量优化
如果需要更高质量的字幕：
- 增加 `llm.translator.max_retries`
- 降低 `llm.translator.temperature`
- 使用更大的 ASR 模型

## 📝 版本历史

### v2.0 (当前版本)
- ✅ 删除无用的多进程配置项
- ✅ 优化路径配置，使用统一的 config 目录
- ✅ 更新术语表路径为新结构
- ✅ 只保留 video_processor 配置

### v1.0 (旧版本)
- ❌ 包含多个已废弃的配置项
- ❌ 路径分散在不同位置
