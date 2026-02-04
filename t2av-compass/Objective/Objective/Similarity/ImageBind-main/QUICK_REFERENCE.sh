#!/bin/bash
# 快速参考 - 批量测试命令

# ============================================================================
# 1. 完整批量测试（所有模型 + 所有评测）
# ============================================================================
cd /root/bayes-tmp/eval/text_similarity/ImageBind-main
bash batch_test_all_models.sh


# ============================================================================
# 2. 查看结果
# ============================================================================

# 查看汇总报告
cat ./batch_test_results/summary_report.txt

# 查看特定模型的详细结果
cat ./batch_test_results/javis/video_text_similarity.json | python -m json.tool


# ============================================================================
# 3. 单独运行某个评测（如需要）
# ============================================================================

# 视频-文本相似度（单个模型）
python batch_inference_video_text.py \
    --json_file /root/bayes-tmp/data/av_data.json \
    --video_dir /root/bayes-tmp/data/videos/sora2 \
    --output_file ./outputs/sora2_vt.json \
    --device cuda:0

# 音频-文本相似度（单个模型）
python batch_inference_audio_text.py \
    --json_file /root/bayes-tmp/data/av_data.json \
    --audio_dir /root/bayes-tmp/data/videos/sora2/audio \
    --output_file ./outputs/sora2_at.json \
    --device cuda:0

# 音频-视频一致性（单个模型）
python batch_inference.py \
    --audio_dir /root/bayes-tmp/data/videos/sora2/audio \
    --video_dir /root/bayes-tmp/data/videos/sora2 \
    --output_dir ./outputs/sora2_av \
    --device cuda:0


# ============================================================================
# 4. 模型对比
# ============================================================================

# 对比两个模型的视频-文本相似度
python compare_models.py \
    --sora2_results ./batch_test_results/sora2/video_text_similarity.json \
    --veo3_results ./batch_test_results/veo3/video_text_similarity.json \
    --output comparison_sora2_vs_veo3.txt


# ============================================================================
# 5. 配置的 6 个模型（支持智能音频目录检测）
# ============================================================================

# 1. javis    - T2av_Results_2/JavisDit/samples          [音频: 自动检测]
# 2. mtv      - mtv_output                               [音频: mtv_output_speech]
# 3. av1      - AVPipe1/av_output                        [音频: 自动检测]
# 4. ovi      - OVI_10s                                  [音频: 自动检测]
# 5. sora2    - sora2                                    [音频: 自动检测]
# 6. veo3     - veo3                                     [音频: 自动检测]

# 音频目录自动检测规则（当设置为 "auto" 时）：
# 1. 优先: video_dir/audio/
# 2. 其次: video_dir/
# 3. 都不存在: 跳过音频测试


# ============================================================================
# 6. 输出目录结构
# ============================================================================

# batch_test_results/
# ├── summary_report.txt              # 汇总报告（最重要）
# ├── javis/
# │   ├── video_text_similarity.json  # 视频-文本相似度
# │   ├── audio_text_similarity.json  # 音频-文本相似度
# │   └── av_consistency/             # 音频-视频一致性
# ├── mtv/
# ├── av1/
# ├── ovi/
# ├── sora2/
# └── veo3/


# ============================================================================
# 7. 常见问题
# ============================================================================

# Q: CUDA 内存不足？
# A: 修改脚本中的 DEVICE="cpu"

# Q: 找不到文件？
# A: 检查路径配置和文件命名格式

# Q: 想跳过某些模型？
# A: 编辑脚本，注释掉 MODELS 数组中对应的行

