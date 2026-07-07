# 删除 `subtitle_bilingual` 和 `subtitle_chinese` 模式变更记录

**日期**: 2026-07-05  
**文件**: `video_cli.py`

## 变更说明

删除了两种已废弃的处理模式，简化代码逻辑。这两个模式与 `subtitle_only` 功能重叠，实际业务中不再使用。

- **`subtitle_bilingual`**: 生成中英双语字幕
- **`subtitle_chinese`**: 生成中文字幕（翻译后）

## 变更详情

### 1. `check_output_exists` 检查输出函数 (L347)

```diff
- if mode in ['subtitle_only', 'subtitle_bilingual', 'subtitle_chinese']:
+ if mode in ['subtitle_only']:
```

### 2. `process_video_unified` 函数 docstring (L894-895)

移除了对 `subtitle_bilingual` 和 `subtitle_chinese` 模式的文档说明。

### 3. 翻译触发条件 (L1113)

```diff
- if mode in ['subtitle_bilingual', 'subtitle_chinese', 'tts_no_subtitle', 'tts_with_review']:
+ if mode in ['tts_no_subtitle', 'tts_with_review']:
```

移除后，翻译步骤仅在 `tts_no_subtitle` 和 `tts_with_review` 模式下触发。

### 4. SRT 字幕生成触发条件 (L1167)

```diff
- if mode in ['subtitle_only', 'subtitle_bilingual', 'subtitle_chinese', 'tts_no_subtitle', 'tts_with_review']:
+ if mode in ['subtitle_only', 'tts_no_subtitle', 'tts_with_review']:
```

### 5. `main` 函数注释 (L1670-1675)

移除了对应模式的注释说明。

### 6. `mode_descriptions` 字典 (L1707-1713)

移除了两个模式的描述条目：
```diff
- 'subtitle_bilingual': '生成中英双语字幕',
- 'subtitle_chinese': '生成中文字幕（翻译后）',
```

### 7. 第一处 argparse choices (L1682-1689)

```diff
  choices=[
      'subtitle_only',
-     'subtitle_bilingual',
-     'subtitle_chinese',
      'tts_no_subtitle',
      'tts_from_srt',
      'tts_with_review'
  ]
```

### 8. 第二处 argparse choices — `__main__` 块 (L1960-1967)

同第一处，移除了 CLI 参数中的 `subtitle_bilingual` 和 `subtitle_chinese` 选项。

### 9. `__main__` 块注释 (L1944-1948)

移除了对应模式的注释说明。

## 保留的模式

| 模式 | 说明 |
|------|------|
| `subtitle_only` | 仅生成中文字幕（ASR 识别） |
| `tts_no_subtitle` | 生成中文配音（全流程：ASR→翻译→TTS→合成） |
| `tts_from_srt` | 从已有中文字幕生成配音（跳过 ASR/翻译） |
| `tts_with_review` | 带人工审核的配音生成（两阶段） |
