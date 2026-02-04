# 模型评测执行指南

## 已创建的测试脚本

已为以下三个模型创建了测试脚本：

1. **MTV**: `test_mtv.sh`
2. **SORA2**: `test_sora2.sh`  
3. **VEO3**: `test_veo3.sh`

每个脚本都会执行三种评测：
- 视频-文本相似度
- 音频-文本相似度
- 音频-视频一致性

## 执行方法

### 方法 1: 使用一键执行脚本（推荐）

```bash
cd /root/bayes-tmp/eval/text_similarity/ImageBind-main

# 激活环境并执行所有测试
source $(conda info --base)/etc/profile.d/conda.sh
conda activate imagebind
bash execute_tests_and_generate_readme.sh
```

这个脚本会：
1. 自动激活 imagebind 环境
2. 依次执行三个模型的测试
3. 测试完成后自动生成 README 报告

### 方法 2: 单独执行每个测试

```bash
cd /root/bayes-tmp/eval/text_similarity/ImageBind-main

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate imagebind

# 执行单个测试
bash test_mtv.sh
bash test_sora2.sh
bash test_veo3.sh

# 测试完成后生成 README
python3 generate_results_readme.py
```

## 输出位置

所有测试结果将保存在：
```
./batch_test_results/
├── mtv/
│   ├── video_text_similarity.json
│   ├── audio_text_similarity.json
│   └── av_consistency/
├── sora2/
│   └── ...
├── veo3/
│   └── ...
└── README.md  # 汇总报告（执行完成后生成）
```

## 配置说明

### MTV 模型
- **视频目录**: `/root/bayes-tmp/data/videos/mtv`
- **音频目录**: `/root/bayes-tmp/data/videos/mtv` (音视频同目录)
- **文件命名**: `video_X.mp4` / `video_X.wav`

### SORA2 模型
- **视频目录**: `/root/bayes-tmp/data/videos/sora2/50`
- **音频目录**: `/root/bayes-tmp/data/videos/sora2/50/audio`
- **文件命名**: `sora2_uid_XXX.mp4` / `sora2_uid_XXX.wav`

### VEO3 模型
- **视频目录**: `/root/bayes-tmp/data/videos/veo3/50`
- **音频目录**: `/root/bayes-tmp/data/videos/veo3/50/audio`
- **文件命名**: `veo3_uid_XXX.mp4` / `veo3_uid_XXX.wav`

## 注意事项

1. **执行时间**: 每个模型的测试可能需要 10-30 分钟（取决于样本数量和 GPU 性能）
2. **GPU 要求**: 建议使用 GPU（cuda:0），CPU 会很慢
3. **磁盘空间**: 每个模型约需要 1-2GB 空间存储结果

## 查看结果

测试完成后，查看汇总报告：

```bash
cat ./batch_test_results/README.md
```

或者查看单个模型的详细结果：

```bash
# 查看 MTV 的视频-文本相似度
cat ./batch_test_results/mtv/video_text_similarity.json | python3 -m json.tool
```

