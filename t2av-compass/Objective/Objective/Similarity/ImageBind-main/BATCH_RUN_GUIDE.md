# ImageBind 批量推理使用指南

## 概述

ImageBind 提供了三种批量推理模式，可以评估视频和文本、音频和文本、以及音频和视频之间的相似度/一致性。

## 三种推理模式

### 1. 视频-文本相似度 (video_text)
评估视频内容与对应文本提示词的相似度。

### 2. 音频-文本相似度 (audio_text)
评估音频内容与对应文本提示词的相似度。

### 3. 音频-视频一致性 (av_consistency)
评估音频和视频内容的匹配度/一致性。

## 方式一：使用 run.sh 脚本（推荐）

最简单的批量推理方式，类似于 DOVER 和 Synchformer 的 run.sh。

### 基本用法

```bash
cd /root/bayes-tmp/eval/text_similarity/ImageBind-main
bash run.sh
```

### 脚本说明

`run.sh` 脚本可以：
1. 批量测试多个视频目录
2. 支持三种模式（视频-文本、音频-文本、音频-视频一致性）
3. 自动保存结果到 JSON 文件

## 方式二：使用 batch_test_all_videos.py（推荐）

功能最全面的批量推理脚本，支持所有三种模式。

### 基本用法

#### 1. 测试所有模式

```bash
python batch_test_all_videos.py \
    --mode all \
    --video_dirs /root/bayes-tmp/all_videos/kling/videos /root/bayes-tmp/all_videos/ovi \
    --json_file /root/bayes-tmp/data/av_data.json \
    --output_dir ./output/all_results \
    --device cuda:0 \
    --batch_size 4
```

#### 2. 只测试视频-文本相似度

```bash
python batch_test_all_videos.py \
    --mode video_text \
    --video_dirs /root/bayes-tmp/all_videos/kling/videos /root/bayes-tmp/all_videos/ovi \
    --json_file /root/bayes-tmp/data/av_data.json \
    --output_dir ./output/video_text \
    --device cuda:0
```

#### 3. 只测试音频-文本相似度

```bash
python batch_test_all_videos.py \
    --mode audio_text \
    --video_dirs /root/bayes-tmp/all_videos/kling/videos /root/bayes-tmp/all_videos/ovi \
    --json_file /root/bayes-tmp/data/av_data.json \
    --output_dir ./output/audio_text \
    --device cuda:0
```

#### 4. 只测试音频-视频一致性

```bash
python batch_test_all_videos.py \
    --mode av_consistency \
    --video_dirs /root/bayes-tmp/all_videos/kling/videos /root/bayes-tmp/all_videos/ovi \
    --output_dir ./output/av_consistency \
    --device cuda:0 \
    --batch_size 4
```

### 参数说明

- `--mode`: 测试模式
  - `video_text`: 视频-文本相似度
  - `audio_text`: 音频-文本相似度
  - `av_consistency`: 音频-视频一致性
  - `all`: 所有模式（默认）

- `--video_dirs`: 视频目录列表，用空格分隔（必需）

- `--audio_dirs`: 音频目录列表（可选，仅用于 av_consistency 模式）
  - 如果未指定，假设音频在视频目录的 `audio` 子目录中

- `--json_file`: JSON 数据文件路径（用于 video_text 和 audio_text 模式）

- `--output_dir`: 输出目录路径（必需）

- `--device`: 设备（默认: `cuda:0`）

- `--batch_size`: 批处理大小（仅用于 av_consistency 模式，默认: 4）

- `--retrieval_mode`: 检索模式（仅用于 av_consistency 模式）

- `--no-resume`: 不从中断处继续（重新开始所有测试）

## 方式三：使用单个脚本

### 视频-文本相似度

```bash
python batch_inference_video_text.py \
    --json_file /root/bayes-tmp/data/av_data.json \
    --video_dir /root/bayes-tmp/all_videos/kling/videos \
    --output_file ./output/kling_video_text.json \
    --device cuda:0
```

### 音频-文本相似度

```bash
python batch_inference_audio_text.py \
    --json_file /root/bayes-tmp/data/av_data.json \
    --audio_dir /root/bayes-tmp/all_videos/kling/audio \
    --output_file ./output/kling_audio_text.json \
    --device cuda:0
```

### 音频-视频一致性

```bash
python batch_inference.py \
    --audio_dir /root/bayes-tmp/all_videos/kling/audio \
    --video_dir /root/bayes-tmp/all_videos/kling/videos \
    --output_dir ./output/kling_av_consistency \
    --device cuda:0 \
    --batch_size 4
```

## 输出格式

### JSON 输出格式

#### 视频-文本相似度

```json
{
  "video_dir": "/path/to/video/dir",
  "json_file": "/path/to/json/file",
  "statistics": {
    "timestamp": "2024-01-01T12:00:00",
    "total_count": 100,
    "successful_count": 95,
    "failed_count": 5,
    "similarity": {
      "mean": 0.7234,
      "std": 0.1456,
      "min": 0.5123,
      "max": 0.9876,
      "median": 0.7456
    }
  },
  "results": [
    {
      "index": 1,
      "video_path": "/path/to/video1.mp4",
      "video_prompt": "a cat playing",
      "success": true,
      "similarity": 0.8234,
      "error": null
    }
  ]
}
```

#### 音频-文本相似度

格式类似，但字段为 `audio_path` 和 `audio_prompt`。

#### 音频-视频一致性

```json
{
  "video_dir": "/path/to/video/dir",
  "audio_dir": "/path/to/audio/dir",
  "audio_count": 100,
  "video_count": 100,
  "metrics": {
    "paired_similarity_mean": 0.8234,
    "paired_similarity_std": 0.0987,
    "paired_similarity_min": 0.5123,
    "paired_similarity_max": 0.9876,
    "similarity_matrix_mean": 0.7234,
    "similarity_matrix_std": 0.1456
  }
}
```

### 输出文件结构

```
output/
├── video_text/
│   ├── kling_video_text.json
│   ├── ovi_video_text.json
│   └── ...
├── audio_text/
│   ├── kling_audio_text.json
│   ├── ovi_audio_text.json
│   └── ...
├── av_consistency/
│   ├── kling_av_consistency.json
│   ├── kling_av_consistency/
│   │   ├── audio_embeddings.npy
│   │   ├── video_embeddings.npy
│   │   ├── similarity_matrix.npy
│   │   ├── metrics.json
│   │   └── consistency_report.txt
│   └── ...
└── summary.json
```

## 使用示例

### 示例1：测试单个目录的所有模式

```bash
python batch_test_all_videos.py \
    --mode all \
    --video_dirs /root/bayes-tmp/all_videos/kling/videos \
    --json_file /root/bayes-tmp/data/av_data.json \
    --output_dir ./output/kling_all \
    --device cuda:0 \
    --batch_size 4
```

### 示例2：测试多个目录（使用 run.sh）

编辑 `run.sh`，修改视频目录列表，然后运行：

```bash
bash run.sh
```

### 示例3：指定音频目录

如果音频不在视频目录的 `audio` 子目录中：

```bash
python batch_test_all_videos.py \
    --mode av_consistency \
    --video_dirs /root/bayes-tmp/all_videos/kling/videos /root/bayes-tmp/all_videos/ovi \
    --audio_dirs /root/bayes-tmp/all_videos/kling/audio /root/bayes-tmp/all_videos/ovi/audio \
    --output_dir ./output/av_consistency \
    --device cuda:0 \
    --batch_size 4
```

## 注意事项

1. **JSON 文件格式**：
   - 用于 `video_text` 和 `audio_text` 模式
   - 需要包含 `index`、`video_prompt` 或 `audio_prompt` 字段
   - 视频/音频文件命名需要与 `index` 对应

2. **文件命名**：
   - 视频文件：`sample_0001.mp4`、`sample_1.mp4`、`1.mp4` 等
   - 音频文件：`sample_0001.wav`、`sample_1.wav`、`1.wav` 等

3. **音频目录**：
   - 如果未指定 `--audio_dirs`，脚本会假设音频在视频目录的 `audio` 子目录中
   - 例如：`/path/to/videos/audio/`

4. **批处理大小**：
   - 视频处理需要更多显存，建议使用较小的 `batch_size`（默认 4）
   - GPU (24GB+): `batch_size=4-8`
   - GPU (12GB): `batch_size=2-4`
   - GPU (8GB): `batch_size=1-2`

5. **断点续传**：
   - 脚本支持断点续传，已完成的测试会自动跳过
   - 使用 `--no-resume` 可以重新开始所有测试

6. **模型下载**：
   - 首次运行会自动下载模型权重
   - 确保网络连接正常

## 故障排除

### 问题1：找不到视频/音频文件

**解决**：检查文件命名格式，确保与 JSON 中的 `index` 对应

### 问题2：显存不足

**解决**：减小 `--batch_size` 参数，或使用 CPU（不推荐）

### 问题3：JSON 文件格式错误

**解决**：确保 JSON 文件包含必需的字段（`index`、`video_prompt` 或 `audio_prompt`）

### 问题4：音频目录不存在

**解决**：使用 `--audio_dirs` 参数指定音频目录，或确保音频在视频目录的 `audio` 子目录中

## 性能优化建议

1. **使用 GPU**：强烈建议使用 GPU 加速（`--device cuda:0`）
2. **批处理大小**：根据 GPU 显存调整 `batch_size`
3. **分批处理**：对于大量视频，可以分多次运行
4. **使用 SSD**：视频文件较大，使用 SSD 可以加快 I/O

## 输出指标说明

### 相似度指标

- `mean`: 平均相似度（0-1，越高越好）
- `std`: 标准差（越小表示越稳定）
- `min/max`: 最小/最大相似度
- `median`: 中位数相似度

### 一致性指标

- `paired_similarity_mean`: 平均配对相似度（配对模式）
- `similarity_matrix_mean`: 平均相似度（检索模式）
- `recall_at_k`: 检索性能指标（检索模式）

## 相关文档

- `BATCH_INFERENCE.md`: 详细的批量推理说明
- `TEXT_SIMILARITY_GUIDE.md`: 文本相似度评估指南
- `BATCH_TEST_GUIDE.md`: 批量测试指南


