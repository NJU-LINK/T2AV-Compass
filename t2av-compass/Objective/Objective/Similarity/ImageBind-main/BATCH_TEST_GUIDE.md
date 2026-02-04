# 音频-视频配对批量测试指南

## 概述

`batch_pairs_test.py` 是一个专门用于批量测试多个音频-视频配对的脚本。支持：

✅ 多个音频-视频配对的批量处理  
✅ CSV 和 TXT 格式的配对文件  
✅ 自动生成对比报告（文本 + CSV）  
✅ 相似度统计和失败追踪  

## 快速开始

### 1. 准备配对文件

#### 方式 A: CSV 格式（推荐）

创建 `pairs.csv`:
```csv
pair_id,audio_path,video_path,description
pair_1,/path/to/audio1.wav,/path/to/video1.mp4,测试样本1
pair_2,/path/to/audio2.wav,/path/to/video2.mp4,测试样本2
pair_3,/path/to/audio3.wav,/path/to/video3.mp4,测试样本3
```

**优点:**
- 清晰易读
- 支持描述字段
- 可用 Excel 编辑

#### 方式 B: TXT 格式

创建 `pairs.txt`:
```
# 注释行
/path/to/audio1.wav,/path/to/video1.mp4,描述1
/path/to/audio2.wav,/path/to/video2.mp4,描述2
/path/to/audio3.wav,/path/to/video3.mp4,描述3
```

**优点:**
- 轻量级
- 易于脚本生成

### 2. 运行批量测试

#### 基础用法

```bash
python batch_pairs_test.py --pairs_file pairs.csv --output_dir ./results
```

#### 指定 GPU 设备

```bash
# 使用 cuda:0
python batch_pairs_test.py --pairs_file pairs.csv --output_dir ./results --device cuda:0

# 使用 cuda:1
python batch_pairs_test.py --pairs_file pairs.csv --output_dir ./results --device cuda:1

# 使用 CPU
python batch_pairs_test.py --pairs_file pairs.csv --output_dir ./results --device cpu
```

### 3. 查看结果

脚本会在输出目录生成三个文件：

#### 📊 `batch_results.json`
完整的 JSON 格式结果，包含所有配对的详细信息：
```json
{
  "timestamp": "2025-11-07T12:34:56.789123",
  "device": "cuda:0",
  "results": [
    {
      "pair_id": "pair_1",
      "audio_name": "audio1",
      "video_name": "video1",
      "success": true,
      "metrics": {
        "similarity": 0.8234,
        "audio_embedding_shape": [1, 1024],
        "video_embedding_shape": [1, 1024]
      }
    }
  ]
}
```

#### 📄 `batch_comparison_report.txt`
人类易读的文本报告，包含：
- 测试汇总（总数、成功、失败）
- 相似度统计（平均、标准差、最小/最大值）
- 详细结果表格
- 失败详情

示例输出：
```
============================================================================================
音频-视频配对批量测试对比报告
============================================================================================

测试时间: 2025-11-07T12:34:56.789123
计算设备: cuda:0
总配对数: 5
有效配对: 5
无效配对: 0

处理结果汇总:
  成功: 5
  失败: 0

相似度统计:
  平均相似度: 0.7542
  标准差: 0.0832
  最小值: 0.6234
  最大值: 0.8756
  中位数: 0.7634
```

#### 📑 `batch_results_summary.csv`
CSV 格式汇总表，便于 Excel 分析：

| 配对ID | 音频文件 | 视频文件 | 描述 | 状态 | 相似度 | 错误信息 |
|--------|---------|---------|------|------|--------|---------|
| pair_1 | audio1  | video1  | 样本1 | 成功 | 0.8234 |         |
| pair_2 | audio2  | video2  | 样本2 | 成功 | 0.7456 |         |

## 实际使用案例

### 场景 1: 测试多个模型的输出

比较不同生成模型（SORA、VEO、LUMA 等）生成的视频质量：

```csv
pair_id,audio_path,video_path,description
sora_test1,music1.wav,sora_output1.mp4,SORA生成
veo_test1,music1.wav,veo_output1.mp4,VEO生成
luma_test1,music1.wav,luma_output1.mp4,LUMA生成
```

### 场景 2: A/B 对比测试

测试同一音乐不同视频版本的匹配度：

```csv
pair_id,audio_path,video_path,description
v1,music.wav,video_version1.mp4,版本1
v2,music.wav,video_version2.mp4,版本2
v3,music.wav,video_version3.mp4,版本3
```

### 场景 3: 数据集评估

对整个数据集的所有配对进行评估：

```csv
pair_id,audio_path,video_path,description
pair_1,dataset/audio/sample1.wav,dataset/video/sample1.mp4,数据集样本1
pair_2,dataset/audio/sample2.wav,dataset/video/sample2.mp4,数据集样本2
...
```

## 输出文件详解

### 目录结构

```
./results/
├── batch_results.json              # JSON 格式完整结果
├── batch_comparison_report.txt     # 文本格式报告
├── batch_results_summary.csv       # CSV 汇总表
├── pair_1/                         # 每个配对的嵌入向量
│   ├── audio_embedding.npy
│   └── video_embedding.npy
├── pair_2/
│   ├── audio_embedding.npy
│   └── video_embedding.npy
└── ...
```

### 相似度解释

相似度范围 [-1, 1]，基于余弦相似度计算：

| 范围 | 含义 |
|------|------|
| 0.8 - 1.0 | 🟢 非常匹配（高度相关） |
| 0.6 - 0.8 | 🟡 较好匹配 |
| 0.4 - 0.6 | 🟠 一般匹配 |
| 0.2 - 0.4 | 🔴 弱匹配 |
| < 0.2 | ⚫ 基本无关 |

## 常见问题

### Q: 如何处理很多配对（1000+）？

A: 脚本支持任意数量配对，但建议：
- 使用 GPU（cuda）而非 CPU
- 配对过多时可分批处理
- 检查磁盘空间（每个配对会保存两个 npy 文件）

### Q: 能否跳过保存嵌入向量？

A: 当前实现会保存，如果需要节省空间，可修改代码或删除 npy 文件。

### Q: 如何与其他评估指标结合？

A: 可以：
1. 使用 batch_results.json 作为输入
2. 结合运动质量指标（jerk、flow等）
3. 生成综合评估报告

### Q: 支持多 GPU 并行处理吗？

A: 当前版本单 GPU，后续可扩展支持分布式处理。

## 完整工作流示例

```bash
# 1. 准备配对文件
cat > pairs.csv << EOF
pair_id,audio_path,video_path,description
pair_1,/path/to/audio1.wav,/path/to/video1.mp4,样本1
pair_2,/path/to/audio2.wav,/path/to/video2.mp4,样本2
EOF

# 2. 运行批量测试
python batch_pairs_test.py --pairs_file pairs.csv --output_dir ./results --device cuda:0

# 3. 查看结果
cat ./results/batch_comparison_report.txt

# 4. 用 CSV 进行进一步分析
python -c "import pandas as pd; df = pd.read_csv('./results/batch_results_summary.csv'); print(df)"
```

## 扩展建议

1. **多 GPU 支持**: 使用 DistributedDataParallel
2. **进度保存**: 支持断点续传
3. **缓存机制**: 避免重复处理相同文件
4. **统计分析**: 生成相似度分布图表
5. **自动配对**: 从目录结构自动生成配对

---

**需要帮助？** 查看 `batch_pairs_test.py` 的帮助信息：

```bash
python batch_pairs_test.py --help
```



