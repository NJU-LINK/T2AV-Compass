#!/bin/bash
# Synchformer批量推理脚本
# 可以测试多个模型对多个视频目录的评估结果

# 设置输出目录
OUTPUT_DIR="./output"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 定义模型列表（可以测试多个模型）
# 常用模型：
# 24-01-04T16-39-21: AudioSet 模型（推荐）
# 24-01-02T10-00-53: VGGSound 模型
# 23-12-23T18-33-57: LRS3 模型
EXP_NAMES=("24-01-04T16-39-21")

# 批量测试单个模型对多个视频目录
# 使用 batch_test_folder.py 脚本

python batch_test_folder.py --folder '/root/bayes-tmp/all_videos/seedance1.5/#U5373#U68a6 short_prompt/seed' --exp_name 24-01-04T16-39-21 --output ${OUTPUT_DIR}/seed-synchformer.json --device cuda:0
python batch_test_folder.py --folder /root/bayes-tmp/all_videos/wan2.6/wan-2.6 --exp_name 24-01-04T16-39-21 --output ${OUTPUT_DIR}/wan2.6-synchformer.json --device cuda:0
python batch_test_folder.py --folder /root/bayes-tmp/all_videos/javis/javis --exp_name 24-01-04T16-39-21 --output ${OUTPUT_DIR}/javis-synchformer.json --device cuda:0
# python batch_test_folder.py --folder /root/bayes-tmp/all_videos/ovi --exp_name 24-01-04T16-39-21 --output ${OUTPUT_DIR}/ovi-synchformer.json --device cuda:0
# python batch_test_folder.py --folder /root/bayes-tmp/all_videos/pixverse --exp_name 24-01-04T16-39-21 --output ${OUTPUT_DIR}/pixverse-synchformer.json --device cuda:0
# python batch_test_folder.py --folder /root/bayes-tmp/all_videos/sora2 --exp_name 24-01-04T16-39-21 --output ${OUTPUT_DIR}/sora2-synchformer.json --device cuda:0
# python batch_test_folder.py --folder /root/bayes-tmp/all_videos/veo --exp_name 24-01-04T16-39-21 --output ${OUTPUT_DIR}/veo-synchformer.json --device cuda:0
# python batch_test_folder.py --folder /root/bayes-tmp/all_videos/wan2.5/wan2.5-720-10s-500#U4e2a --exp_name 24-01-04T16-39-21 --output ${OUTPUT_DIR}/wan2.5-synchformer.json --device cuda:0

# 如果需要测试多个模型，可以使用 batch_test_multiple_models.py 脚本
# python batch_test_multiple_models.py \
#     --video_dirs /root/bayes-tmp/all_videos/kling/videos /root/bayes-tmp/all_videos/ovi \
#     --exp_names 24-01-04T16-39-21 24-01-02T10-00-53 \
#     --output_dir ${OUTPUT_DIR}/multi_model_results \
#     --device cuda:0

