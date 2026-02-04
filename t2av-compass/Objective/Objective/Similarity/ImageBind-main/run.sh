#!/bin/bash
# ImageBind批量推理脚本
# 只测试音频-视频一致性

# 设置输出目录
OUTPUT_DIR="./output"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
# /root/bayes-tmp/all_videos/avpipe1
# 批量测试音频-视频一致性
python batch_test_all_videos.py \
    --mode av_consistency \
    --video_dirs \
        /root/bayes-tmp/all_videos/kling/videos \
        /root/bayes-tmp/all_videos/ovi \
    --output_dir ${OUTPUT_DIR}/av_consistency \
    --device cuda:0 \
    --batch_size 8

# 如果需要指定音频目录（音频不在视频目录的 audio 子目录中）
# python batch_test_all_videos.py \
#     --mode av_consistency \
#     --video_dirs /root/bayes-tmp/all_videos/kling/videos /root/bayes-tmp/all_videos/ovi \
#     --audio_dirs /root/bayes-tmp/all_videos/kling/audio /root/bayes-tmp/all_videos/ovi/audio \
#     --output_dir ${OUTPUT_DIR}/av_consistency \
#     --device cuda:0 \
#     --batch_size 4


