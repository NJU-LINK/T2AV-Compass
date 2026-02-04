# 视频/音频与文本相似度批量推理指南

## 概述

这两个脚本用于从 JSON 文件中读取文本提示词（prompt），并计算生成的视频/音频与对应文本的相似度。

## 脚本说明

### 1. `batch_inference_video_text.py`
计算**视频-文本相似度**

- 从 JSON 中读取 `video_prompt` 字段
- 根据 `index` 查找对应的视频文件
- 计算视频内容与文本描述的相似度

### 2. `batch_inference_audio_text.py`
计算**音频-文本相似度**

- 从 JSON 中读取 `audio_prompt` 字段
- 根据 `index` 查找对应的音频文件
- 计算音频内容与文本描述的相似度

## JSON 格式要求

输入的 JSON 文件应包含以下字段：

```json
[
    {
        "index": 1,
        "video_prompt": "视频内容的文本描述...",
        "audio_prompt": "音频内容的文本描述..."
    },
    {
        "index": 2,
        "video_prompt": "...",
        "audio_prompt": "..."
    }
]
```

## 文件命名规则

脚本会自动查找以下命名格式的文件（按优先级）：

### 视频文件
- `sample_{index:04d}.mp4` (如 `sample_0001.mp4`)
- `sample_{index}.mp4` (如 `sample_1.mp4`)
- `{index:04d}.mp4` (如 `0001.mp4`)
- `{index}.mp4` (如 `1.mp4`)
- `video_{index:04d}.mp4`
- `video_{index}.mp4`

支持的视频格式：`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

### 音频文件
- `sample_{index:04d}.wav` (如 `sample_0001.wav`)
- `sample_{index}.wav` (如 `sample_1.wav`)
- `{index:04d}.wav` (如 `0001.wav`)
- `{index}.wav` (如 `1.wav`)
- `audio_{index:04d}.wav`
- `audio_{index}.wav`

支持的音频格式：`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`

## 使用方法

### 视频-文本相似度

```bash
python batch_inference_video_text.py \
    --json_file /root/bayes-tmp/data/av_data.json \
    --video_dir /root/bayes-tmp/data/videos/sora2 \
    --output_file results_sora2_video_text.json \
    --device cuda:0
```

### 音频-文本相似度

```bash
python batch_inference_audio_text.py \
    --json_file /root/bayes-tmp/data/av_data.json \
    --audio_dir /root/bayes-tmp/data/audios/sora2 \
    --output_file results_sora2_audio_text.json \
    --device cuda:0
```

## 参数说明

| 参数 | 说明 | 必填 | 默认值 |
|------|------|------|--------|
| `--json_file` | 包含提示词的 JSON 文件路径 | 是 | - |
| `--video_dir` / `--audio_dir` | 视频/音频文件所在目录 | 是 | - |
| `--output_file` | 输出结果的 JSON 文件路径 | 否 | `results_video_text.json` / `results_audio_text.json` |
| `--device` | 计算设备 | 否 | `cuda:0` |

## 输出格式

输出的 JSON 文件结构：

```json
{
  "statistics": {
    "timestamp": "2025-12-06T12:34:56.789123",
    "total_count": 38,
    "successful_count": 35,
    "failed_count": 3,
    "similarity": {
      "mean": 0.7234,
      "std": 0.0856,
      "min": 0.5123,
      "max": 0.8976,
      "median": 0.7456
    }
  },
  "results": [
    {
      "index": 1,
      "video_path": "/path/to/video.mp4",
      "video_prompt": "视频描述文本...",
      "success": true,
      "error": null,
      "similarity": 0.7234,
      "video_embedding_shape": [1, 1024],
      "text_embedding_shape": [1, 1024]
    },
    ...
  ]
}
```

## 相似度解释

相似度范围：`[-1, 1]`（基于余弦相似度）

| 相似度范围 | 含义 |
|-----------|------|
| 0.8 - 1.0 | 🟢 非常匹配（内容高度相关） |
| 0.6 - 0.8 | 🟡 较好匹配 |
| 0.4 - 0.6 | 🟠 一般匹配 |
| 0.2 - 0.4 | 🔴 弱匹配 |
| < 0.2 | ⚫ 基本无关 |

## 完整工作流示例

假设你有以下目录结构：

```
/root/bayes-tmp/
├── data/
│   ├── av_data.json          # 包含提示词的 JSON
│   └── videos/
│       ├── sora2/
│       │   ├── sample_0001.mp4
│       │   ├── sample_0002.mp4
│       │   └── ...
│       └── veo3/
│           ├── sample_0001.mp4
│           └── ...
```

### 1. 评估 SORA2 视频-文本相似度

```bash
cd /root/bayes-tmp/eval/text_similarity/ImageBind-main

python batch_inference_video_text.py \
    --json_file /root/bayes-tmp/data/av_data.json \
    --video_dir /root/bayes-tmp/data/videos/sora2 \
    --output_file ./outputs/sora2_video_text.json \
    --device cuda:0
```

### 2. 评估 VEO3 视频-文本相似度

```bash
python batch_inference_video_text.py \
    --json_file /root/bayes-tmp/data/av_data.json \
    --video_dir /root/bayes-tmp/data/videos/veo3 \
    --output_file ./outputs/veo3_video_text.json \
    --device cuda:0
```

### 3. 对比不同模型

使用 `compare_models.py` 对比结果：

```bash
python compare_models.py \
    --sora2_results ./outputs/sora2_video_text.json \
    --veo3_results ./outputs/veo3_video_text.json \
    --output comparison_report.txt
```

## 性能建议

1. **GPU 使用**：推荐使用 GPU 加速（`--device cuda:0`）
2. **批量大小**：脚本会逐个处理，避免内存溢出
3. **文件查找**：确保文件命名符合支持的格式

## 故障排除

### 问题：视频/音频文件未找到

**解决方案**：
1. 检查文件命名是否符合支持的格式
2. 确认 `index` 字段与文件名匹配
3. 查看控制台警告信息

### 问题：相似度很低

**可能原因**：
1. 生成的内容与文本描述不符
2. 提示词（prompt）不准确
3. 模型对某些内容类型的理解有限

### 问题：CUDA 内存不足

**解决方案**：
```bash
# 使用 CPU
python batch_inference_video_text.py \
    --json_file ... \
    --video_dir ... \
    --device cpu
```

## 注意事项

1. **首次运行**：会自动下载 ImageBind 模型权重（约 2.4GB）
2. **视频格式**：建议使用 `.mp4` 格式
3. **音频格式**：建议使用 `.wav` 格式
4. **文本编码**：JSON 文件使用 UTF-8 编码
5. **空提示词**：如果 `video_prompt` 或 `audio_prompt` 为空，该条目会被标记为失败

## 扩展功能

如果需要支持其他文件命名格式，可以修改脚本中的 `find_video_file()` 或 `find_audio_file()` 函数。

例如，添加新的命名模式：

```python
patterns = [
    f'sample_{index:04d}',
    f'my_custom_name_{index}',  # 添加自定义格式
    # ...
]
```

## 相关脚本

- `batch_inference.py` - 原始的音频-视频一致性评估脚本
- `batch_pairs_test.py` - 音频-视频配对批量测试脚本
- `compare_models.py` - 模型结果对比脚本


