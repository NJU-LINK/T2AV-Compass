#!/bin/bash
################################################################################
# 批量测试脚本 - 评估所有模型的视频-文本、音频-文本和音频-视频相似度
#
# 用法:
#   bash batch_test_all_models.sh
#
# 说明:
#   - 自动测试所有配置的模型
#   - 三种评测: 视频-文本、音频-文本、音频-视频一致性
#   - 结果输出到 ./batch_test_results/ 目录下，按模型名称分文件夹
################################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 配置区域 - 根据实际情况修改
# ============================================================================

# JSON 数据文件路径
JSON_FILE="/root/bayes-tmp/data/av_data.json"

# 基础路径
BASE_VIDEO_DIR="/root/bayes-tmp/data/videos"
BASE_AUDIO_DIR="/root/bayes-tmp/data/videos"  # 音频通常在视频目录的子文件夹

# 输出根目录
OUTPUT_ROOT="./batch_test_results"

# GPU 设备
DEVICE="cuda:0"

# 模型列表配置
# 格式: "模型名称:视频子目录:音频子目录"
# 音频子目录说明：
#   - 可以指定具体路径（如 "mtv_output_speech"）
#   - 可以设置为 "auto" - 自动检测（优先 video_dir/audio，其次 video_dir）
#   - 可以留空 - 使用视频目录
MODELS=(
    "javis:T2av_Results_2/JavisDit/samples:auto"
    "mtv:mtv"
    "av1:AVPipe1/av_output:auto"
    "ovi:OVI_10s:auto"
    "sora2:sora2:auto"
    "veo3:veo3:auto"
)

# ============================================================================
# 函数定义
# ============================================================================

# 打印带颜色的消息
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_separator() {
    echo "================================================================================"
}

# 检查文件/目录是否存在
check_path() {
    local path=$1
    local desc=$2
    if [ ! -e "$path" ]; then
        print_error "$desc 不存在: $path"
        return 1
    fi
    return 0
}

# 智能检测音频目录
detect_audio_dir() {
    local video_dir=$1
    local audio_subdir=$2
    
    # 如果明确指定了音频子目录且不是 "auto"
    if [ -n "$audio_subdir" ] && [ "$audio_subdir" != "auto" ]; then
        echo "${BASE_AUDIO_DIR}/${audio_subdir}"
        return
    fi
    
    # 自动检测模式
    # 1. 优先检查 video_dir/audio
    local audio_in_subdir="${video_dir}/audio"
    if [ -d "$audio_in_subdir" ]; then
        # 将信息输出到 stderr，避免影响返回值
        echo "[检测] 找到音频子目录: audio/" >&2
        echo "$audio_in_subdir"
        return
    fi
    
    # 2. 其次使用视频目录本身
    if [ -d "$video_dir" ]; then
        # 将信息输出到 stderr，避免影响返回值
        echo "[检测] 使用视频目录（音视频同目录）" >&2
        echo "$video_dir"
        return
    fi
    
    # 3. 如果都不存在，返回空
    echo ""
}

# 处理单个模型
process_model() {
    local model_info=$1
    
    # 解析模型信息
    IFS=':' read -r model_name video_subdir audio_subdir <<< "$model_info"
    
    local video_dir="${BASE_VIDEO_DIR}/${video_subdir}"
    local output_dir="${OUTPUT_ROOT}/${model_name}"
    
    # 智能检测音频目录
    local audio_dir=$(detect_audio_dir "$video_dir" "$audio_subdir")
    
    print_separator
    print_info "处理模型: $model_name"
    print_separator
    print_info "  视频目录: $video_dir"
    print_info "  音频目录: $audio_dir"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 检查路径
    local has_video=true
    local has_audio=true
    
    if ! check_path "$video_dir" "视频目录"; then
        has_video=false
        print_error "跳过 $model_name 的视频相关测试"
    fi
    
    if [ -z "$audio_dir" ] || ! check_path "$audio_dir" "音频目录"; then
        has_audio=false
        print_error "跳过 $model_name 的音频相关测试"
    fi
    
    # ========================================================================
    # 1. 视频-文本相似度测试
    # ========================================================================
    if [ "$has_video" = true ]; then
        print_info "[$model_name] 开始视频-文本相似度测试..."
        
        python batch_inference_video_text.py \
            --json_file "$JSON_FILE" \
            --video_dir "$video_dir" \
            --output_file "${output_dir}/video_text_similarity.json" \
            --device "$DEVICE" \
            2>&1 | tee "${output_dir}/video_text_log.txt"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            print_success "[$model_name] 视频-文本相似度测试完成"
        else
            print_error "[$model_name] 视频-文本相似度测试失败"
        fi
        echo ""
    fi
    
    # ========================================================================
    # 2. 音频-文本相似度测试
    # ========================================================================
    if [ "$has_audio" = true ]; then
        print_info "[$model_name] 开始音频-文本相似度测试..."
        
        python batch_inference_audio_text.py \
            --json_file "$JSON_FILE" \
            --audio_dir "$audio_dir" \
            --output_file "${output_dir}/audio_text_similarity.json" \
            --device "$DEVICE" \
            2>&1 | tee "${output_dir}/audio_text_log.txt"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            print_success "[$model_name] 音频-文本相似度测试完成"
        else
            print_error "[$model_name] 音频-文本相似度测试失败"
        fi
        echo ""
    fi
    
    # ========================================================================
    # 3. 音频-视频一致性测试
    # ========================================================================
    if [ "$has_video" = true ] && [ "$has_audio" = true ]; then
        print_info "[$model_name] 开始音频-视频一致性测试..."
        
        python batch_inference.py \
            --audio_dir "$audio_dir" \
            --video_dir "$video_dir" \
            --output_dir "${output_dir}/av_consistency" \
            --device "$DEVICE" \
            2>&1 | tee "${output_dir}/av_consistency_log.txt"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            print_success "[$model_name] 音频-视频一致性测试完成"
        else
            print_error "[$model_name] 音频-视频一致性测试失败"
        fi
        echo ""
    fi
    
    print_success "模型 $model_name 测试完成！"
    echo ""
}

# 生成汇总报告
generate_summary() {
    local summary_file="${OUTPUT_ROOT}/summary_report.txt"
    
    print_info "生成汇总报告..."
    
    {
        echo "================================================================================"
        echo "批量测试汇总报告"
        echo "================================================================================"
        echo ""
        echo "测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "JSON 文件: $JSON_FILE"
        echo "输出目录: $OUTPUT_ROOT"
        echo ""
        echo "================================================================================"
        echo "模型列表及测试结果"
        echo "================================================================================"
        echo ""
        
        for model_info in "${MODELS[@]}"; do
            IFS=':' read -r model_name _ _ <<< "$model_info"
            local output_dir="${OUTPUT_ROOT}/${model_name}"
            
            echo "【$model_name】"
            echo "  输出目录: $output_dir"
            
            # 视频-文本相似度
            if [ -f "${output_dir}/video_text_similarity.json" ]; then
                local vt_stats=$(python -c "import json; data=json.load(open('${output_dir}/video_text_similarity.json')); stats=data.get('statistics',{}).get('similarity',{}); print(f\"平均值: {stats.get('mean','N/A'):.4f}, 成功: {data['statistics']['successful_count']}/{data['statistics']['total_count']}\" if stats else 'N/A')" 2>/dev/null || echo "解析失败")
                echo "  ✓ 视频-文本相似度: $vt_stats"
            else
                echo "  ✗ 视频-文本相似度: 未完成"
            fi
            
            # 音频-文本相似度
            if [ -f "${output_dir}/audio_text_similarity.json" ]; then
                local at_stats=$(python -c "import json; data=json.load(open('${output_dir}/audio_text_similarity.json')); stats=data.get('statistics',{}).get('similarity',{}); print(f\"平均值: {stats.get('mean','N/A'):.4f}, 成功: {data['statistics']['successful_count']}/{data['statistics']['total_count']}\" if stats else 'N/A')" 2>/dev/null || echo "解析失败")
                echo "  ✓ 音频-文本相似度: $at_stats"
            else
                echo "  ✗ 音频-文本相似度: 未完成"
            fi
            
            # 音频-视频一致性
            if [ -f "${output_dir}/av_consistency/metrics.json" ]; then
                local av_stats=$(python -c "import json; data=json.load(open('${output_dir}/av_consistency/metrics.json')); print(f\"平均相似度: {data.get('paired_similarity_mean','N/A'):.4f}\")" 2>/dev/null || echo "解析失败")
                echo "  ✓ 音频-视频一致性: $av_stats"
            else
                echo "  ✗ 音频-视频一致性: 未完成"
            fi
            
            echo ""
        done
        
        echo "================================================================================"
        echo "详细结果请查看各模型目录下的 JSON 和日志文件"
        echo "================================================================================"
        
    } | tee "$summary_file"
    
    print_success "汇总报告已保存到: $summary_file"
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    print_separator
    echo "批量测试脚本 - 所有模型评测"
    print_separator
    echo ""
    
    # 检查 JSON 文件
    if ! check_path "$JSON_FILE" "JSON 数据文件"; then
        print_error "JSON 文件不存在，请检查配置"
        exit 1
    fi
    
    # 检查 Python 脚本
    local scripts=("batch_inference_video_text.py" "batch_inference_audio_text.py" "batch_inference.py")
    for script in "${scripts[@]}"; do
        if ! check_path "$script" "Python 脚本 $script"; then
            print_error "请确保在正确的目录下运行此脚本"
            exit 1
        fi
    done
    
    # 创建输出根目录
    mkdir -p "$OUTPUT_ROOT"
    
    print_info "配置信息:"
    echo "  JSON 文件: $JSON_FILE"
    echo "  视频基础目录: $BASE_VIDEO_DIR"
    echo "  音频基础目录: $BASE_AUDIO_DIR"
    echo "  输出目录: $OUTPUT_ROOT"
    echo "  设备: $DEVICE"
    echo "  模型数量: ${#MODELS[@]}"
    echo ""
    
    # 处理每个模型
    local start_time=$(date +%s)
    
    for model_info in "${MODELS[@]}"; do
        process_model "$model_info"
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 生成汇总报告
    generate_summary
    
    # 打印完成信息
    print_separator
    print_success "所有测试完成！"
    print_separator
    echo ""
    echo "总耗时: $((duration / 60)) 分钟 $((duration % 60)) 秒"
    echo "输出目录: $OUTPUT_ROOT"
    echo ""
    echo "结果文件结构:"
    echo "  $OUTPUT_ROOT/"
    echo "  ├── summary_report.txt           # 汇总报告"
    echo "  ├── <model_name>/"
    echo "  │   ├── video_text_similarity.json      # 视频-文本相似度结果"
    echo "  │   ├── audio_text_similarity.json      # 音频-文本相似度结果"
    echo "  │   ├── av_consistency/                 # 音频-视频一致性结果"
    echo "  │   │   ├── metrics.json"
    echo "  │   │   └── consistency_report.txt"
    echo "  │   └── *.log                           # 日志文件"
    echo ""
}

# 运行主程序
main "$@"

