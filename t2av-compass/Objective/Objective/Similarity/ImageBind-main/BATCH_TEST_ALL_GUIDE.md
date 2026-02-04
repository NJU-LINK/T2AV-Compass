# 批量测试脚本使用指南

## 概述

`batch_test_all_models.sh` 是一个自动化批量测试脚本，可以一次性评估多个模型的：
1. **视频-文本相似度**
2. **音频-文本相似度**
3. **音频-视频一致性**

## 快速开始

### 1. 基本用法

```bash
cd /root/bayes-tmp/eval/text_similarity/ImageBind-main
bash batch_test_all_models.sh
```

### 2. 默认配置

脚本已预配置以下 6 个模型，支持**智能音频目录检测**：

| 模型名称 | 视频目录 | 音频目录检测 |
|---------|---------|-------------|
| javis | T2av_Results_2/JavisDit/samples | 自动检测 |
| mtv | mtv_output | mtv_output_speech（指定） |
| av1 | AVPipe1/av_output | 自动检测 |
| ovi | OVI_10s | 自动检测 |
| sora2 | sora2 | 自动检测 |
| veo3 | veo3 | 自动检测 |

**音频目录检测规则**（当设置为 `auto` 时）：
1. 优先检查：`视频目录/audio/`
2. 其次检查：`视频目录/`（与视频同目录）
3. 如果都不存在，跳过音频相关测试

## 配置说明

### 修改配置

编辑脚本中的配置区域：

```bash
# JSON 数据文件路径
JSON_FILE="/root/bayes-tmp/data/av_data.json"

# 基础路径
BASE_VIDEO_DIR="/root/bayes-tmp/data/videos"
BASE_AUDIO_DIR="/root/bayes-tmp/data/videos"

# 输出根目录
OUTPUT_ROOT="./batch_test_results"

# GPU 设备
DEVICE="cuda:0"

# 模型列表配置
MODELS=(
    "模型名称:视频子目录:音频子目录"
    "javis:T2av_Results_2/JavisDit/samples:auto"  # auto = 自动检测
    # ... 添加更多模型
)
```

### 音频目录配置选项

有三种方式配置音频目录：

#### 1. 自动检测模式（推荐）

```bash
MODELS=(
    "javis:video_path:auto"
)
```

脚本会自动检测：
- 优先：`${BASE_VIDEO_DIR}/video_path/audio/`
- 其次：`${BASE_VIDEO_DIR}/video_path/`

#### 2. 明确指定路径

```bash
MODELS=(
    "mtv:mtv_output:mtv_output_speech"  # 明确指定不同目录
)
```

#### 3. 留空（使用视频目录）

```bash
MODELS=(
    "model:video_path:"  # 音频与视频同目录
)
```

### 添加新模型

根据你的音频文件位置，选择合适的配置方式：

#### 情况 1: 音频在 video_dir/audio/ 或 video_dir/

```bash
MODELS=(
    "my_model:my_videos:auto"  # 使用自动检测
)
```

#### 情况 2: 音频在完全不同的目录

```bash
MODELS=(
    "my_model:my_videos:my_separate_audio_dir"
)
```

#### 情况 3: 音频与视频混在一起

```bash
MODELS=(
    "my_model:my_videos:"  # 留空，使用视频目录
)

## 输出结构

运行完成后，会在 `./batch_test_results/` 生成以下结构：

```
batch_test_results/
├── summary_report.txt                 # 汇总报告（重要！）
├── javis/
│   ├── video_text_similarity.json    # 视频-文本相似度结果
│   ├── audio_text_similarity.json    # 音频-文本相似度结果
│   ├── av_consistency/               # 音频-视频一致性结果
│   │   ├── metrics.json
│   │   ├── consistency_report.txt
│   │   ├── audio_embeddings.npy
│   │   └── video_embeddings.npy
│   ├── video_text_log.txt            # 视频-文本测试日志
│   ├── audio_text_log.txt            # 音频-文本测试日志
│   └── av_consistency_log.txt        # 音频-视频测试日志
├── mtv/
│   └── ... (相同结构)
├── av1/
│   └── ... (相同结构)
├── ovi/
│   └── ... (相同结构)
├── sora2/
│   └── ... (相同结构)
└── veo3/
    └── ... (相同结构)
```

## 结果文件说明

### 1. `summary_report.txt` - 汇总报告

包含所有模型的测试摘要：

```
【javis】
  ✓ 视频-文本相似度: 平均值: 0.7234, 成功: 35/38
  ✓ 音频-文本相似度: 平均值: 0.6856, 成功: 35/38
  ✓ 音频-视频一致性: 平均相似度: 0.8123
```

### 2. `video_text_similarity.json` - 视频-文本相似度

```json
{
  "statistics": {
    "similarity": {
      "mean": 0.7234,
      "std": 0.0856,
      "min": 0.5123,
      "max": 0.8976
    }
  },
  "results": [...]
}
```

### 3. `audio_text_similarity.json` - 音频-文本相似度

结构同上，包含音频-文本相似度统计。

### 4. `av_consistency/metrics.json` - 音频-视频一致性

```json
{
  "paired_similarity_mean": 0.8123,
  "paired_similarity_std": 0.0654,
  "similarity_matrix_mean": 0.7856
}
```

## 运行流程

脚本会按以下顺序执行：

1. **检查环境**
   - 验证 JSON 文件是否存在
   - 验证 Python 脚本是否存在
   - 创建输出目录

2. **逐个处理模型**
   - 检查视频/音频目录是否存在
   - 运行三种评测（如果路径有效）
   - 保存结果和日志

3. **生成汇总报告**
   - 汇总所有模型的测试结果
   - 生成可读的文本报告

## 注意事项

### 1. 路径要求

- 视频文件命名：`sample_0001.mp4`, `sample_1.mp4` 等
- 音频文件命名：`sample_0001.wav`, `sample_1.wav` 等
- 文件索引需要与 JSON 中的 `index` 字段匹配

### 2. 资源需求

- GPU 内存：建议 ≥ 8GB
- 磁盘空间：每个模型约 1-2GB（包含嵌入向量）
- 运行时间：每个模型约 5-15 分钟（取决于样本数量）

### 3. 错误处理

- 如果某个目录不存在，会跳过相关测试
- 如果某个测试失败，会继续处理其他测试
- 所有错误信息都会记录在日志文件中

## 高级用法

### 仅测试特定模型

修改 `MODELS` 数组，注释掉不需要的模型：

```bash
MODELS=(
    "javis:T2av_Results_2/JavisDit/samples:T2av_Results_2/JavisDit/samples/audio"
    # "mtv:mtv_output:mtv_output_speech"  # 注释掉
    # "av1:AVPipe1/av_output:AVPipe1/av_output/audio"  # 注释掉
)
```

### 使用不同 GPU

修改 `DEVICE` 变量：

```bash
DEVICE="cuda:1"  # 使用第二张 GPU
```

### 自定义输出目录

修改 `OUTPUT_ROOT` 变量：

```bash
OUTPUT_ROOT="./my_test_results_$(date +%Y%m%d)"
```

## 故障排除

### 问题 1: 找不到视频/音频文件

**症状**：输出显示 "视频文件未找到"

**解决方案**：
1. 检查 `BASE_VIDEO_DIR` 和 `BASE_AUDIO_DIR` 路径
2. 检查模型的子目录配置
3. 验证文件命名格式

### 问题 2: CUDA 内存不足

**症状**：脚本运行时 GPU 内存溢出

**解决方案**：
```bash
# 使用 CPU
DEVICE="cpu"

# 或者在运行前清理 GPU
nvidia-smi
# 找到占用 GPU 的进程并杀掉
```

### 问题 3: Python 依赖缺失

**症状**：ImportError 或 ModuleNotFoundError

**解决方案**：
```bash
conda activate imagebind
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## 查看结果

### 1. 查看汇总报告

```bash
cat ./batch_test_results/summary_report.txt
```

### 2. 查看单个模型详细结果

```bash
# 查看 javis 的视频-文本相似度
python -c "import json; print(json.dumps(json.load(open('./batch_test_results/javis/video_text_similarity.json')), indent=2))"

# 或使用 jq（如果已安装）
cat ./batch_test_results/javis/video_text_similarity.json | jq '.statistics'
```

### 3. 对比多个模型

使用现有的 `compare_models.py`：

```bash
python compare_models.py \
    --model1_results ./batch_test_results/javis/video_text_similarity.json \
    --model2_results ./batch_test_results/sora2/video_text_similarity.json \
    --output comparison_javis_vs_sora2.txt
```

## 示例输出

运行成功后，控制台会显示：

```
================================================================================
批量测试脚本 - 所有模型评测
================================================================================

[INFO] 配置信息:
  JSON 文件: /root/bayes-tmp/data/av_data.json
  输出目录: ./batch_test_results
  设备: cuda:0
  模型数量: 6

================================================================================
[INFO] 处理模型: javis
================================================================================
[INFO] [javis] 开始视频-文本相似度测试...
[SUCCESS] [javis] 视频-文本相似度测试完成
[INFO] [javis] 开始音频-文本相似度测试...
[SUCCESS] [javis] 音频-文本相似度测试完成
[INFO] [javis] 开始音频-视频一致性测试...
[SUCCESS] [javis] 音频-视频一致性测试完成
[SUCCESS] 模型 javis 测试完成！

... (其他模型类似输出)

================================================================================
[SUCCESS] 所有测试完成！
================================================================================

总耗时: 45 分钟 32 秒
输出目录: ./batch_test_results
```

## 性能优化建议

1. **并行处理**：如果有多张 GPU，可以修改脚本支持并行
2. **缓存嵌入向量**：避免重复计算
3. **使用 SSD**：加快文件 I/O

## 相关文档

- `TEXT_SIMILARITY_GUIDE.md` - 文本相似度评测详细指南
- `BATCH_INFERENCE.md` - 音频-视频一致性评测指南
- `BATCH_TEST_GUIDE.md` - 配对测试指南

