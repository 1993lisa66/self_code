# 配置文件说明 (config.yaml)

## 目录结构配置

```yaml
paths:
  input_dir: input                      # 输入视频目录
  output_dir: outputs                   # 输出目录
  cache_dir: cache                      # 缓存目录
  log_dir: logs                        # 日志目录
  prompts_dir: config/prompts          # 提示词模板目录
  assets_dir: assets                   # 资源文件目录
  models_dir: models                   # 模型存储目录
  batches_dir: config/batches          # 批次配置目录
  terminology_file: config/terminology.json  # 全局术语表
```

## 多进程配置

```yaml
global:
  max_concurrency:
    video_processor: 2  # 视频处理并行进程数（1=单进程，>1=多进程）
  auto_cleanup_cache: true  # 自动清理缓存
```

**说明**:
- `video_processor`: 控制同时处理的视频数量
  - 设置为 1: 单进程顺序处理（适合内存有限的机器）
  - 设置为 2-5: 多进程并行处理（适合高性能机器）
  - 建议值: CPU 4核以下设为 1-2，8核以上可设为 3-5

## ASR 配置

```yaml
asr:
  model_id: "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
  device: cpu          # 运行设备: cpu 或 cuda
  model_size: base     # Whisper 模型大小: tiny, base, small, medium, large-v3
  voting_models:
    - whisperx
    - glm
  terminology_file: terminology.json
```

## LLM 配置

```yaml
llm:
  api_key: "your-api-key"              # API 密钥
  api_base: "https://api.siliconflow.cn/v1"  # API 地址
  model: "deepseek-ai/DeepSeek-V3"     # 默认模型
  
  translator:                          # 翻译专用配置
    model: "deepseek-ai/DeepSeek-V3"
    temperature: 0.1
    max_retries: 3
  
  tts_processor:                       # TTS 文本处理配置
    model: "deepseek-ai/DeepSeek-V3"
    temperature: 0.1
  
  chapter_generator:                   # 章节生成配置
    model: "deepseek-ai/DeepSeek-V3"
    temperature: 0.1
  
  prompts:                             # 提示词文件名映射
    asr_fix: asr_fix.txt
    translation: translation.txt
    chapters: chapters.txt
    tts_prep: tts_prep.txt
    resegmentation: resegmentation.txt
    semantic_segmentation: semantic_segmentation.txt
```

## TTS 配置

```yaml
tts:
  provider: edge                      # TTS 提供商: edge (Microsoft Edge TTS)
  voice: zh-CN-XiaoxiaoNeural         # 语音选择
  speed_limit: 1.5                    # 最大语速倍数
```

**可用语音**:
- 女声: zh-CN-XiaoxiaoNeural (推荐), zh-CN-XiaoyiNeural, zh-CN-lunaNeural 等
- 男声: zh-CN-YunxiNeural (推荐), zh-CN-YunjianNeural, zh-CN-YunfengNeural 等

## 视频配置

```yaml
video:
  subtitle_font_size: 18              # 字幕字体大小
  codec: libx264                      # 视频编码器
  crf: 23                            # 视频质量 (18-28, 越小质量越高)
  preset: fast                       # 编码速度: ultrafast, fast, medium, slow
  burn_subtitles: false              # 是否烧录字幕到视频
  subtitle_position: "bottom"        # 字幕位置: bottom, top, center
  audio_mode: "tts_only"             # 音频模式: tts_only, mix, bgm_mix, original
```
