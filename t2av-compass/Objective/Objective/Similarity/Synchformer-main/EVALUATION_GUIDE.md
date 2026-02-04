# Synchformer 视频评测指南

本指南介绍如何使用 Synchformer 对视频进行音视频同步评测。

## 目录

1. [评测方法概览](#评测方法概览)
2. [方法一：单视频快速评测](#方法一单视频快速评测)
3. [方法二：批量视频评测](#方法二批量视频评测)
4. [方法三：完整数据集评估](#方法三完整数据集评估)
5. [方法四：可同步性评估](#方法四可同步性评估)
6. [评测指标说明](#评测指标说明)
7. [常见问题](#常见问题)

---

## 评测方法概览

Synchformer 提供四种评测方式：

| 方法 | 脚本 | 适用场景 | 输出 |
|------|------|----------|------|
| 单视频评测 | `example.py` | 快速测试单个视频 | 控制台输出 |
| 批量评测 | `batch_inference.py` | 处理多个视频 | JSON/CSV 文件 |
| 数据集评估 | `main.py` | 完整数据集评估 | 详细指标和日志 |
| 可同步性评估 | `scripts/test_syncability.py` | 判断是否可同步 | ROC 曲线和指标 |

---

## 方法一：单视频快速评测

### 用途
快速测试单个视频的音视频同步情况。

### 使用方法

```bash
python example.py \
  --exp_name "24-01-04T16-39-21" \
  --vid_path "./data/video.mp4" \
  --offset_sec 1.6 \
  --v_start_i_sec 0.0 \
  --device cuda:0
```

### 参数说明

- `--exp_name`: 预训练模型ID（格式：`xx-xx-xxTxx-xx-xx`）
  - `24-01-04T16-39-21`: AudioSet 预训练模型（推荐）
  - `24-01-02T10-00-53`: VGGSound 预训练模型
  - `23-12-23T18-33-57`: LRS3 预训练模型
- `--vid_path`: 视频文件路径
- `--offset_sec`: 音频偏移秒数（测试用，0.0 表示无偏移）
- `--v_start_i_sec`: 视频起始秒数（默认 0.0）
- `--device`: 设备（`cuda:0` 或 `cpu`）

### 输出示例

```
Ground Truth offset (sec): 1.60 (18)

Prediction Results:
p=0.8076 (11.5469), "1.60" (18)
p=0.1234 (8.2345), "1.50" (17)
...
```

### 注意事项

1. **视频格式要求**：
   - 视频帧率：25 fps
   - 音频采样率：16000 Hz
   - 最小边：256 像素
   - 如果格式不符，脚本会自动重新编码

2. **模型下载**：首次运行会自动下载预训练模型

---

## 方法二：批量视频评测

### 用途
批量处理多个视频，生成结构化的评测结果。

### 使用方法

#### 方式 1：使用视频列表文件

```bash
# 创建视频列表文件 video_list.txt
# 每行一个视频路径
echo "/path/to/video1.mp4" > video_list.txt
echo "/path/to/video2.mp4" >> video_list.txt
echo "/path/to/video3.mp4" >> video_list.txt

# 运行批量推理
python batch_inference.py \
  --exp_name "24-01-04T16-39-21" \
  --video_list video_list.txt \
  --output results.json \
  --output_format json \
  --offset_sec 0.0 \
  --v_start_i_sec 0.0 \
  --device cuda:0
```

#### 方式 2：使用逗号分隔的路径

```bash
python batch_inference.py \
  --exp_name "24-01-04T16-39-21" \
  --video_list "/path/to/video1.mp4,/path/to/video2.mp4,/path/to/video3.mp4" \
  --output results.csv \
  --output_format csv \
  --device cuda:0
```

### 参数说明

- `--exp_name`: 预训练模型ID
- `--video_list`: 视频列表文件路径或逗号分隔的路径
- `--output`: 输出文件路径
- `--output_format`: 输出格式（`json` 或 `csv`）
- `--offset_sec`: 音频偏移秒数（默认 0.0）
- `--v_start_i_sec`: 视频起始秒数（默认 0.0）
- `--device`: 设备
- `--batch_size`: 批量大小（当前实现为 1，可扩展）

### 输出格式

#### JSON 格式

```json
[
  {
    "video_path": "/path/to/video1.mp4",
    "predicted_offset_sec": 1.60,
    "predicted_class": 18,
    "predicted_probability": 0.8076,
    "ground_truth_offset_sec": null,
    "ground_truth_class": null,
    "topk_predictions": [
      {
        "class": 18,
        "offset_sec": 1.60,
        "probability": 0.8076,
        "logit": 11.5469
      }
    ],
    "success": true,
    "error": null
  }
]
```

#### CSV 格式

```csv
video_path,predicted_offset_sec,predicted_class,predicted_probability,ground_truth_offset_sec,ground_truth_class,success,error
/path/to/video1.mp4,1.60,18,0.8076,,,True,
```

### 结果字段说明

- `predicted_offset_sec`: 预测的偏移秒数（主要输出）
- `predicted_class`: 预测的类别索引（0-20，对应 -2.0 到 +2.0 秒）
- `predicted_probability`: 预测置信度（0-1）
- `topk_predictions`: Top-5 预测结果
- `success`: 是否成功处理
- `error`: 错误信息（如果失败）

---

## 方法三：完整数据集评估

### 用途
在完整数据集上评估模型性能，生成详细的评估指标。

### 使用方法

```bash
# 设置模型ID
S2_CKPT_ID="24-01-04T16-39-21"

# 运行评估
torchrun --standalone --nnodes=1 --nproc-per-node=1 --master_addr=localhost --master_port=1234 \
main.py \
    config="./logs/sync_models/$S2_CKPT_ID/cfg-$S2_CKPT_ID.yaml" \
    ckpt_path="./logs/sync_models/$S2_CKPT_ID/$S2_CKPT_ID.pt" \
    logging.logdir="./logs" \
    training.finetune="False" \
    training.run_test_only="True" \
    data.iter_times="5" \
    data.dataset.params.load_fixed_offsets_on="[]" \
    logging.log_code_state=False \
    model.params.afeat_extractor.params.ckpt_path=null \
    model.params.vfeat_extractor.params.ckpt_path=null \
    training.base_batch_size=8 \
    logging.log_frequency=4 \
    logging.use_wandb=False
```

### 参数说明

- `config`: 模型配置文件路径
- `ckpt_path`: 模型检查点路径
- `training.run_test_only`: 设置为 `True` 表示仅运行测试
- `data.iter_times`: 迭代次数（小数据集建议 5-25 次）
- `data.dataset.params.load_fixed_offsets_on`: 固定偏移的数据集（设为 `[]` 表示不固定）
- `training.base_batch_size`: 批量大小

### 评估不同数据集

```bash
# 评估 VGGSound Sparse 数据集
torchrun --standalone --nnodes=1 --nproc-per-node=1 --master_addr=localhost --master_port=1234 \
main.py \
    config="./logs/sync_models/$S2_CKPT_ID/cfg-$S2_CKPT_ID.yaml" \
    ckpt_path="./logs/sync_models/$S2_CKPT_ID/$S2_CKPT_ID.pt" \
    data.dataset.target=dataset.vggsound.VGGSoundSparsePickedCleanTestFixedOffsets \
    data.vids_path="/path/to/vggsound/h264_video_25fps_256side_16000hz_aac/" \
    training.run_test_only="True" \
    ...
```

### 输出指标

评估会输出以下指标：

- `accuracy_1`: 精确匹配准确率（Acc@1）
- `accuracy_1_tolerance_1`: 允许 ±1 类误差的准确率（Acc@1 ±1）
- 其他详细指标会保存在日志文件中

### 迭代次数建议

- **LRS3**: `data.iter_times="2"`
- **VGGSound-Sparse**: `data.iter_times="25"`
- **其他小数据集**: `data.iter_times="5"`

---

## 方法四：可同步性评估

### 用途
评估音视频是否可同步（synchronizability），输出 ROC 曲线和 AUC 指标。

### 使用方法

```bash
S3_CKPT_ID="24-01-22T20-34-52"

torchrun --standalone --nnodes=1 --nproc-per-node=1 --master_addr=localhost --master_port=1234 \
scripts/test_syncability.py \
    config_sync="./logs/sync_models/${S3_CKPT_ID}/cfg-${S3_CKPT_ID}.yaml" \
    ckpt_path_sync="./logs/sync_models/${S3_CKPT_ID}/${S3_CKPT_ID}_best.pt" \
    logging.logdir="./logs" \
    training.finetune=False \
    training.run_test_only=True \
    data.dataset.target=dataset.vggsound.VGGSoundSparsePickedCleanTest \
    data.vids_path="/path/to/vggsound/h264_video_25fps_256side_16000hz_aac/" \
    data.n_segments=14 \
    data.dataset.params.iter_times=25 \
    data.dataset.params.load_fixed_offsets_on="[]" \
    logging.log_code_state=False \
    model.params.afeat_extractor.params.ckpt_path=null \
    model.params.vfeat_extractor.params.ckpt_path=null \
    training.base_batch_size=8 \
    logging.log_frequency=4 \
    logging.use_wandb=False
```

### 跨阈值评估

如果需要评估不同置信度阈值下的性能：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 --master_addr=localhost --master_port=1234 \
scripts/test_syncability.py \
    config_sync="./logs/sync_models/${S3_CKPT_ID}/cfg-${S3_CKPT_ID}.yaml" \
    ckpt_path_sync="./logs/sync_models/${S3_CKPT_ID}/${S3_CKPT_ID}_best.pt" \
    config_off="./logs/sync_models/${S2_CKPT_ID}/cfg-${S2_CKPT_ID}.yaml" \
    ckpt_path_off="./logs/sync_models/${S2_CKPT_ID}/${S2_CKPT_ID}.pt" \
    ...
```

### 输出

- ROC 曲线数据（保存为 `.pkl` 文件）
- AUC-ROC 分数
- 不同置信度阈值下的指标（0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99）

---

## 评测指标说明

### 同步任务指标

1. **Accuracy@1 (Acc@1)**
   - 精确匹配准确率
   - 预测类别与真实类别完全一致的比例

2. **Accuracy@1 ±1 (Acc@1 ±1)**
   - 允许 ±1 类误差的准确率
   - 更宽松的评估标准

3. **偏移类别映射**
   - 21 个类别对应 -2.0 到 +2.0 秒
   - 步长：0.1 秒
   - 类别 0: -2.0 秒
   - 类别 10: 0.0 秒（同步）
   - 类别 20: +2.0 秒

### 可同步性任务指标

1. **AUC-ROC**
   - ROC 曲线下面积
   - 范围：0-1，越高越好

2. **准确率 (Accuracy)**
   - 二分类准确率（可同步/不可同步）

3. **置信度阈值**
   - 不同置信度阈值下的性能
   - 用于平衡精确率和召回率

---

## 常见问题

### 1. 视频格式不匹配

**问题**：视频格式不符合要求（帧率、采样率、分辨率）

**解决**：
- 脚本会自动重新编码视频
- 或手动使用 ffmpeg 转换：
```bash
ffmpeg -i input.mp4 -vf "fps=25,scale=iw*256/min(iw,ih):ih*256/min(iw,ih)" -ar 16000 output.mp4
```

### 2. 模型下载失败

**问题**：首次运行时模型下载失败

**解决**：
- 检查网络连接
- 手动下载模型到 `./logs/sync_models/` 目录
- 模型链接见 README.md

### 3. 显存不足

**问题**：GPU 显存不足

**解决**：
- 减小 `training.base_batch_size`
- 使用 CPU 推理（`--device cpu`，速度较慢）
- 使用半精度推理（已在配置中启用）

### 4. 评估结果不准确

**问题**：评估结果与预期不符

**解决**：
- 检查 `av` 库版本（建议 8.1.0 或 9.0.0）
- 确保视频格式正确
- 增加 `data.iter_times` 以获得更稳定的结果

### 5. 批量处理速度慢

**问题**：批量处理速度慢

**解决**：
- 当前实现为逐个处理
- 可以修改 `batch_inference.py` 实现真正的批量处理
- 使用多 GPU 并行处理

---

## 快速开始示例

### 示例 1：快速测试单个视频

```bash
python example.py \
  --exp_name "24-01-04T16-39-21" \
  --vid_path "./data/test_video.mp4" \
  --offset_sec 0.0
```

### 示例 2：批量处理视频列表

```bash
# 创建视频列表
ls /path/to/videos/*.mp4 > video_list.txt

# 运行批量推理
python batch_inference.py \
  --exp_name "24-01-04T16-39-21" \
  --video_list video_list.txt \
  --output results.json \
  --output_format json
```

### 示例 3：评估完整数据集

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
main.py \
    config="./logs/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml" \
    ckpt_path="./logs/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt" \
    training.run_test_only="True" \
    data.iter_times="5"
```

---

## 更多资源

- [README.md](README.md) - 项目主文档
- [BATCH_INFERENCE.md](BATCH_INFERENCE.md) - 批量推理详细说明
- [项目主页](https://www.robots.ox.ac.uk/~vgg/research/synchformer/)
- [论文](https://arxiv.org/abs/2401.16423)

---

## 技术支持

如遇问题，请检查：
1. 环境配置是否正确（`conda_env.yml`）
2. 依赖版本是否匹配（特别是 `av` 库）
3. 视频格式是否符合要求
4. 模型文件是否完整下载


