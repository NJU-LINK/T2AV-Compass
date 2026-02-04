# ImageBind 音频-视频一致性批量推理使用指南

本文档介绍如何使用 ImageBind 进行音频-视频一致性批量推理，专门用于评估音频和视频的匹配度/一致性。

## 项目概述

ImageBind 是 Meta AI 开发的多模态联合嵌入模型，能够将多种模态统一映射到同一个嵌入空间。本脚本专门用于：

- **音频-视频一致性评估**: 评估音频和视频内容的匹配程度
- **跨模态检索**: 通过音频检索匹配的视频，或反之
- **批量处理**: 支持大规模音频-视频对的一致性测试

## 核心功能

### 1. 两种工作模式

#### 配对模式 (Pairing Mode) - 默认
- 自动匹配文件名相同的音频和视频文件
- 计算配对的一致性指标
- 适用于评估生成的音频-视频对的质量

#### 检索模式 (Retrieval Mode)
- 计算所有音频-视频对的相似度矩阵
- 评估检索性能（Recall@K, 平均排名等）
- 适用于音频-视频检索任务

### 2. 批量输入处理

#### 音频输入
- **目录扫描**: 使用 `--audio_dir` 参数，自动扫描目录下的所有音频文件
- **路径列表文件**: 使用 `--audio_file` 参数，从文件中读取音频路径列表
- **直接指定路径**: 使用 `--audio_paths` 参数，直接指定多个音频路径

支持的音频格式: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`

#### 视频输入
- **目录扫描**: 使用 `--video_dir` 参数，自动扫描目录下的所有视频文件
- **路径列表文件**: 使用 `--video_file` 参数，从文件中读取视频路径列表
- **直接指定路径**: 使用 `--video_paths` 参数，直接指定多个视频路径

支持的视频格式: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.webm`

### 3. 一致性评估指标

脚本会自动计算以下指标：

- **配对相似度**: 配对音频-视频的平均相似度、标准差、最小值、最大值
- **检索性能**: 平均排名、中位数排名、Recall@1/5/10
- **相似度矩阵**: 所有音频-视频对的相似度矩阵

## 使用方法

### 基本用法

#### 1. 配对模式（默认）

当音频和视频文件名相同（不含扩展名）时，会自动配对：

```bash
# 目录结构示例:
# ./audios/
#   ├── video1.wav
#   ├── video2.wav
#   └── video3.wav
# ./videos/
#   ├── video1.mp4
#   ├── video2.mp4
#   └── video3.mp4

python batch_inference.py \
    --audio_dir ./audios \
    --video_dir ./videos \
    --output_dir ./outputs
```

#### 2. 检索模式

计算所有音频-视频对的相似度，用于检索任务：

```bash
python batch_inference.py \
    --audio_dir ./audios \
    --video_dir ./videos \
    --output_dir ./outputs \
    --retrieval_mode
```

#### 3. 从文件列表读取

```bash
# audio_list.txt 内容:
# /path/to/audio1.wav
# /path/to/audio2.wav

# video_list.txt 内容:
# /path/to/video1.mp4
# /path/to/video2.mp4

python batch_inference.py \
    --audio_file audio_list.txt \
    --video_file video_list.txt \
    --output_dir ./outputs
```

#### 4. 直接指定路径

```bash
python batch_inference.py \
    --audio_paths ./audio1.wav ./audio2.wav \
    --video_paths ./video1.mp4 ./video2.mp4 \
    --output_dir ./outputs
```

### 高级用法

#### 调整批处理大小

视频处理需要更多显存，建议使用较小的 batch_size：

```bash
# GPU (24GB+): batch_size=4-8
# GPU (12GB): batch_size=2-4
# GPU (8GB): batch_size=1-2
# CPU: batch_size=1

python batch_inference.py \
    --audio_dir ./audios \
    --video_dir ./videos \
    --batch_size 2 \
    --output_dir ./outputs
```

#### 使用 CPU

如果 GPU 不可用或想使用 CPU：

```bash
python batch_inference.py \
    --audio_dir ./audios \
    --video_dir ./videos \
    --device cpu \
    --batch_size 1 \
    --output_dir ./outputs
```

## 输出文件说明

运行脚本后，在输出目录中会生成以下文件：

### 嵌入向量文件 (.npy)
- `audio_embeddings.npy`: 音频嵌入向量，shape `(N, 1024)`
- `video_embeddings.npy`: 视频嵌入向量，shape `(M, 1024)`
- `similarity_matrix.npy`: 相似度矩阵，shape `(N, M)`

### 文件名列表 (.txt)
- `audio_names.txt`: 音频文件名列表（不含扩展名）
- `video_names.txt`: 视频文件名列表（不含扩展名）
- `pairs.txt`: 配对信息（配对模式），格式：`音频路径\t视频路径`

### 评估指标文件
- `metrics.json`: JSON 格式的评估指标
- `consistency_report.txt`: 可读的一致性评估报告

### 报告内容示例

```
================================================================================
音频-视频一致性评估报告
================================================================================

音频数量: 100
视频数量: 100
配对数量: 100

相似度统计:
  平均相似度: 0.7234
  标准差: 0.1456

配对一致性:
  平均配对相似度: 0.8234 ± 0.0987
  最小值: 0.5123
  最大值: 0.9876

检索性能:
  平均排名: 2.34
  中位数排名: 1.00
  Recall@1: 0.7500
  Recall@5: 0.9500
  Recall@10: 0.9800
```

## 代码示例

### Python API 使用

```python
from batch_inference import batch_inference_av_consistency

# 批量推理
results = batch_inference_av_consistency(
    audio_paths=["./audio1.wav", "./audio2.wav"],
    video_paths=["./video1.mp4", "./video2.mp4"],
    device="cuda:0",
    batch_size=4,
    output_dir="./outputs",
    retrieval_mode=False
)

# 访问结果
audio_emb = results['audio_embeddings']  # shape: (N, 1024)
video_emb = results['video_embeddings']  # shape: (M, 1024)
metrics = results['metrics']

# 查看指标
print(f"平均相似度: {metrics['similarity_matrix_mean']:.4f}")
if 'paired_similarity_mean' in metrics:
    print(f"配对相似度: {metrics['paired_similarity_mean']:.4f}")
```

### 计算自定义一致性指标

```python
import torch
import numpy as np

# 加载保存的嵌入向量
audio_emb = np.load("./outputs/audio_embeddings.npy")
video_emb = np.load("./outputs/video_embeddings.npy")

# 转换为 torch 张量
audio_emb = torch.from_numpy(audio_emb)
video_emb = torch.from_numpy(video_emb)

# 归一化
audio_norm = torch.nn.functional.normalize(audio_emb, p=2, dim=1)
video_norm = torch.nn.functional.normalize(video_emb, p=2, dim=1)

# 计算余弦相似度
similarity = audio_norm @ video_norm.T  # (N, M)

# 对于配对的数据，提取对角线元素
if len(audio_emb) == len(video_emb):
    paired_similarities = torch.diag(similarity)
    print(f"配对相似度: {paired_similarities.mean().item():.4f}")
```

## 批量输入输出流程

```
音频/视频文件/目录
    ↓
读取文件列表
    ↓
匹配文件（配对模式）
    ↓
分批处理 (batch_size)
    ↓
数据预处理 (data.py)
    - load_and_transform_audio_data() → 梅尔频谱图
    - load_and_transform_video_data() → 视频帧序列
    ↓
模型推理 (imagebind_model.py)
    - forward() → 嵌入向量
    ↓
计算一致性指标
    - 配对相似度
    - 检索性能
    - 相似度矩阵
    ↓
保存结果
    - .npy 文件 (嵌入向量和相似度矩阵)
    - .json 文件 (评估指标)
    - .txt 文件 (报告和文件名列表)
```

## 注意事项

1. **GPU 显存**: 视频处理需要更多显存，建议使用较小的 `batch_size`（默认 4）
2. **文件匹配**: 配对模式基于文件名（不含扩展名）匹配，确保文件名一致
3. **文件格式**: 
   - 音频: 支持常见格式，建议使用 `.wav` 或 `.mp3`
   - 视频: 支持常见格式，建议使用 `.mp4`
4. **模型下载**: 首次运行会自动下载模型权重到 `.checkpoints/imagebind_huge.pth`
5. **错误处理**: 脚本会自动跳过无法处理的文件，并显示警告信息

## 性能优化建议

### 批处理大小建议

| GPU 显存 | batch_size | 说明 |
|---------|-----------|------|
| 24GB+ | 4-8 | 视频处理显存需求较大 |
| 12GB | 2-4 | 推荐设置 |
| 8GB | 1-2 | 最小设置 |
| CPU | 1 | 非常慢，不推荐 |

### 其他优化技巧

1. **预处理数据**: 如果重复测试，可以预先提取嵌入向量并保存
2. **使用 SSD**: 视频文件较大，使用 SSD 可以加快 I/O
3. **多进程**: 可以手动并行处理多个目录

## 常见问题

**Q: 音频和视频数量不一致怎么办？**
A: 配对模式只会匹配文件名相同的文件。如果数量不一致，未匹配的文件会被忽略。检索模式会计算所有配对。

**Q: 如何判断音频-视频是否匹配？**
A: 查看 `consistency_report.txt` 中的配对相似度。相似度越高（接近 1.0），匹配度越好。一般相似度 > 0.7 可以认为是匹配的。

**Q: Recall@K 是什么意思？**
A: Recall@K 表示正确配对在前 K 个检索结果中的比例。Recall@1=0.8 表示 80% 的音频能在检索到的前 1 个视频中找到正确配对。

**Q: 输出文件的维度是什么？**
A: 所有模态的嵌入向量都是 1024 维。相似度矩阵的维度是 `(音频数量, 视频数量)`。

**Q: 如何处理大量文件？**
A: 使用 `--audio_file` 和 `--video_file` 从文件列表读取，脚本会自动分批处理。对于非常大的数据集，可以考虑分多次运行。

**Q: 配对模式 vs 检索模式的区别？**
A: 
- 配对模式：假设音频和视频已经配对，评估配对的质量
- 检索模式：评估从所有视频中检索匹配音频的能力

## 依赖要求

确保已安装以下依赖：

```bash
pip install torch torchvision torchaudio
pip install pytorchvideo
pip install timm ftfy regex einops iopath numpy tqdm
```

或者使用项目的 requirements.txt：

```bash
pip install -r requirements.txt
pip install tqdm  # 批量推理脚本额外需要
```

## 应用场景

1. **音频-视频生成质量评估**: 评估生成的音频和视频是否匹配
2. **跨模态检索**: 评估音频检索视频或视频检索音频的性能
3. **一致性基准测试**: 建立音频-视频一致性评估基准
4. **数据清洗**: 识别不匹配的音频-视频对
