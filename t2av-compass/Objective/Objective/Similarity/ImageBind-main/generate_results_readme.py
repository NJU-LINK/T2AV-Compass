#!/usr/bin/env python3
"""
生成评测结果 README 汇总报告
"""
import json
import os
from pathlib import Path
from datetime import datetime

def load_json_safe(file_path):
    """安全加载 JSON 文件"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载 {file_path}: {e}")
    return None

def extract_statistics(data, metric_type):
    """提取统计信息"""
    if not data:
        return None
    
    stats = data.get('statistics', {})
    
    if metric_type == 'similarity':
        sim_stats = stats.get('similarity', {})
        if sim_stats:
            return {
                'count': stats.get('successful_count', 0),
                'total': stats.get('total_count', 0),
                'mean': sim_stats.get('mean', 0),
                'std': sim_stats.get('std', 0),
                'min': sim_stats.get('min', 0),
                'max': sim_stats.get('max', 0),
                'median': sim_stats.get('median', 0)
            }
    elif metric_type == 'consistency':
        if 'paired_similarity_mean' in stats:
            return {
                'mean': stats.get('paired_similarity_mean', 0),
                'std': stats.get('paired_similarity_std', 0),
                'min': stats.get('paired_similarity_min', 0),
                'max': stats.get('paired_similarity_max', 0)
            }
        elif 'similarity_matrix_mean' in stats:
            return {
                'mean': stats.get('similarity_matrix_mean', 0),
                'std': stats.get('similarity_matrix_std', 0)
            }
    
    return None

def generate_readme(output_dir="./batch_test_results"):
    """生成 README 汇总报告"""
    
    models = ['mtv', 'sora2', 'veo3', 'av1']
    base_dir = Path(output_dir)
    
    readme_lines = []
    
    # 标题
    readme_lines.append("# 模型评测结果汇总报告")
    readme_lines.append("")
    readme_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    readme_lines.append("")
    readme_lines.append("---")
    readme_lines.append("")
    
    # 目录
    readme_lines.append("## 目录")
    readme_lines.append("")
    readme_lines.append("- [评测概述](#评测概述)")
    readme_lines.append("- [模型评测结果](#模型评测结果)")
    for model in models:
        readme_lines.append(f"  - [{model.upper()}](#{model}-评测结果)")
    readme_lines.append("- [结果文件位置](#结果文件位置)")
    readme_lines.append("")
    readme_lines.append("---")
    readme_lines.append("")
    
    # 评测概述
    readme_lines.append("## 评测概述")
    readme_lines.append("")
    readme_lines.append("本报告汇总了以下模型的三种评测结果：")
    readme_lines.append("")
    readme_lines.append("1. **视频-文本相似度**：评估生成的视频内容与文本描述的匹配程度")
    readme_lines.append("2. **音频-文本相似度**：评估生成的音频内容与文本描述的匹配程度")
    readme_lines.append("3. **音频-视频一致性**：评估音频和视频的同步性和一致性")
    readme_lines.append("")
    readme_lines.append("### 相似度评分说明")
    readme_lines.append("")
    readme_lines.append("| 相似度范围 | 含义 |")
    readme_lines.append("|-----------|------|")
    readme_lines.append("| 0.8 - 1.0 | 🟢 非常匹配（内容高度相关） |")
    readme_lines.append("| 0.6 - 0.8 | 🟡 较好匹配 |")
    readme_lines.append("| 0.4 - 0.6 | 🟠 一般匹配 |")
    readme_lines.append("| 0.2 - 0.4 | 🔴 弱匹配 |")
    readme_lines.append("| < 0.2 | ⚫ 基本无关 |")
    readme_lines.append("")
    readme_lines.append("---")
    readme_lines.append("")
    
    # 各模型结果
    readme_lines.append("## 模型评测结果")
    readme_lines.append("")
    
    summary_table = []
    summary_table.append("| 模型 | 视频-文本相似度 | 音频-文本相似度 | 音频-视频一致性 |")
    summary_table.append("|------|----------------|----------------|----------------|")
    
    for model in models:
        model_dir = base_dir / model
        
        readme_lines.append(f"### {model.upper()} 评测结果")
        readme_lines.append("")
        
        # 视频-文本相似度
        vt_file = model_dir / "video_text_similarity.json"
        vt_data = load_json_safe(vt_file)
        vt_stats = extract_statistics(vt_data, 'similarity')
        
        readme_lines.append("#### 1. 视频-文本相似度")
        readme_lines.append("")
        if vt_stats:
            readme_lines.append(f"- **成功样本数**: {vt_stats['count']}/{vt_stats['total']}")
            readme_lines.append(f"- **平均相似度**: {vt_stats['mean']:.6f}")
            readme_lines.append(f"- **标准差**: {vt_stats['std']:.6f}")
            readme_lines.append(f"- **最小值**: {vt_stats['min']:.6f}")
            readme_lines.append(f"- **最大值**: {vt_stats['max']:.6f}")
            readme_lines.append(f"- **中位数**: {vt_stats['median']:.6f}")
            vt_mean_str = f"{vt_stats['mean']:.4f}"
        else:
            readme_lines.append("- ❌ 测试未完成或数据不可用")
            vt_mean_str = "N/A"
        readme_lines.append("")
        
        # 音频-文本相似度
        at_file = model_dir / "audio_text_similarity.json"
        at_data = load_json_safe(at_file)
        at_stats = extract_statistics(at_data, 'similarity')
        
        readme_lines.append("#### 2. 音频-文本相似度")
        readme_lines.append("")
        if at_stats:
            readme_lines.append(f"- **成功样本数**: {at_stats['count']}/{at_stats['total']}")
            readme_lines.append(f"- **平均相似度**: {at_stats['mean']:.6f}")
            readme_lines.append(f"- **标准差**: {at_stats['std']:.6f}")
            readme_lines.append(f"- **最小值**: {at_stats['min']:.6f}")
            readme_lines.append(f"- **最大值**: {at_stats['max']:.6f}")
            readme_lines.append(f"- **中位数**: {at_stats['median']:.6f}")
            at_mean_str = f"{at_stats['mean']:.4f}"
        else:
            readme_lines.append("- ❌ 测试未完成或数据不可用")
            at_mean_str = "N/A"
        readme_lines.append("")
        
        # 音频-视频一致性
        av_file = model_dir / "av_consistency" / "metrics.json"
        av_data = load_json_safe(av_file)
        av_stats = extract_statistics(av_data, 'consistency')
        
        readme_lines.append("#### 3. 音频-视频一致性")
        readme_lines.append("")
        if av_stats:
            readme_lines.append(f"- **平均配对相似度**: {av_stats['mean']:.6f}")
            if 'std' in av_stats:
                readme_lines.append(f"- **标准差**: {av_stats['std']:.6f}")
            if 'min' in av_stats:
                readme_lines.append(f"- **最小值**: {av_stats['min']:.6f}")
            if 'max' in av_stats:
                readme_lines.append(f"- **最大值**: {av_stats['max']:.6f}")
            av_mean_str = f"{av_stats['mean']:.4f}"
        else:
            readme_lines.append("- ❌ 测试未完成或数据不可用")
            av_mean_str = "N/A"
        readme_lines.append("")
        
        readme_lines.append("---")
        readme_lines.append("")
        
        # 添加到汇总表
        summary_table.append(f"| {model.upper()} | {vt_mean_str} | {at_mean_str} | {av_mean_str} |")
    
    # 在开头插入汇总表
    summary_idx = readme_lines.index("## 模型评测结果")
    readme_lines.insert(summary_idx + 2, "### 快速对比")
    readme_lines.insert(summary_idx + 3, "")
    readme_lines.insert(summary_idx + 4, "\n".join(summary_table))
    readme_lines.insert(summary_idx + 5, "")
    readme_lines.insert(summary_idx + 6, "> 注：数值为平均相似度，范围 [-1, 1]，越高越好")
    readme_lines.insert(summary_idx + 7, "")
    readme_lines.insert(summary_idx + 8, "---")
    readme_lines.insert(summary_idx + 9, "")
    
    # 结果文件位置
    readme_lines.append("## 结果文件位置")
    readme_lines.append("")
    readme_lines.append("所有评测结果保存在以下目录结构：")
    readme_lines.append("")
    readme_lines.append("```")
    readme_lines.append("batch_test_results/")
    for model in models:
        readme_lines.append(f"├── {model}/")
        readme_lines.append(f"│   ├── video_text_similarity.json      # 视频-文本相似度结果")
        readme_lines.append(f"│   ├── audio_text_similarity.json      # 音频-文本相似度结果")
        readme_lines.append(f"│   ├── av_consistency/                 # 音频-视频一致性结果")
        readme_lines.append(f"│   │   ├── metrics.json")
        readme_lines.append(f"│   │   └── consistency_report.txt")
        readme_lines.append(f"│   └── *.log                           # 日志文件")
    readme_lines.append("└── README.md                               # 本报告")
    readme_lines.append("```")
    readme_lines.append("")
    
    # 保存 README
    readme_path = base_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(readme_lines))
    
    print(f"✓ README 已生成: {readme_path}")
    return readme_path

if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./batch_test_results"
    generate_readme(output_dir)

