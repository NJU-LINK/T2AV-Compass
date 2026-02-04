#!/usr/bin/env python3
"""
模型对比分析脚本

用于对比不同模型（SORA2 vs VEO3等）的音频-视频一致性相似度结果。

用法:
  python compare_models.py --sora2_results batch_results_sora2/batch_results.json \
                           --veo3_results batch_results_veo3/batch_results.json \
                           --output comparison_report.txt
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


def load_batch_results(json_file: str) -> Dict:
    """加载批量测试结果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_similarities(results: Dict) -> List[float]:
    """从结果中提取相似度列表"""
    similarities = []
    for result in results['results']:
        if result['success'] and 'similarity' in result['metrics']:
            similarities.append(result['metrics']['similarity'])
    return similarities


def calculate_stats(similarities: List[float]) -> Dict:
    """计算统计数据"""
    if not similarities:
        return {}
    
    return {
        'count': len(similarities),
        'mean': float(np.mean(similarities)),
        'std': float(np.std(similarities)),
        'min': float(np.min(similarities)),
        'max': float(np.max(similarities)),
        'median': float(np.median(similarities)),
        'q1': float(np.percentile(similarities, 25)),
        'q3': float(np.percentile(similarities, 75))
    }


def compare_models(model_results: Dict[str, Dict]) -> Dict:
    """对比多个模型"""
    comparison = {}
    
    for model_name, results in model_results.items():
        similarities = extract_similarities(results)
        comparison[model_name] = {
            'results': results,
            'similarities': similarities,
            'stats': calculate_stats(similarities)
        }
    
    return comparison


def generate_comparison_report(comparison: Dict, output_file: str = None) -> str:
    """生成对比报告"""
    report = []
    report.append("=" * 100)
    report.append("音频-视频一致性模型对比分析报告")
    report.append("=" * 100)
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 汇总表
    report.append("=" * 100)
    report.append("模型对比汇总")
    report.append("=" * 100)
    report.append("")
    
    # 创建对比表格
    headers = ["模型", "样本数", "平均相似度", "标准差", "最小值", "最大值", "中位数"]
    report.append(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<15} {headers[3]:<12} {headers[4]:<12} {headers[5]:<12} {headers[6]:<10}")
    report.append("-" * 100)
    
    stats_list = []
    for model_name in sorted(comparison.keys()):
        data = comparison[model_name]
        stats = data['stats']
        
        if stats:
            row = [
                model_name[:14],
                str(stats['count']),
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                f"{stats['median']:.4f}"
            ]
            report.append(f"{row[0]:<15} {row[1]:<10} {row[2]:<15} {row[3]:<12} {row[4]:<12} {row[5]:<12} {row[6]:<10}")
            stats_list.append((model_name, stats))
    
    report.append("")
    report.append("")
    
    # 详细统计
    report.append("=" * 100)
    report.append("详细统计分析")
    report.append("=" * 100)
    report.append("")
    
    for model_name, stats in stats_list:
        report.append(f"【{model_name}】")
        report.append(f"  样本数:       {stats['count']}")
        report.append(f"  平均相似度:   {stats['mean']:.4f}")
        report.append(f"  标准差:       {stats['std']:.4f}")
        report.append(f"  最小值:       {stats['min']:.4f}")
        report.append(f"  最大值:       {stats['max']:.4f}")
        report.append(f"  中位数:       {stats['median']:.4f}")
        report.append(f"  Q1 (25%):     {stats['q1']:.4f}")
        report.append(f"  Q3 (75%):     {stats['q3']:.4f}")
        report.append(f"  四分位差:     {stats['q3'] - stats['q1']:.4f}")
        report.append("")
    
    # 对比分析
    if len(stats_list) > 1:
        report.append("=" * 100)
        report.append("对比分析")
        report.append("=" * 100)
        report.append("")
        
        # 找出最好和最差的模型
        sorted_by_mean = sorted(stats_list, key=lambda x: x[1]['mean'], reverse=True)
        best_model = sorted_by_mean[0]
        worst_model = sorted_by_mean[-1]
        
        report.append(f"🏆 最佳模型: {best_model[0]}")
        report.append(f"   平均相似度: {best_model[1]['mean']:.4f}")
        report.append("")
        
        report.append(f"⚠️  最差模型: {worst_model[0]}")
        report.append(f"   平均相似度: {worst_model[1]['mean']:.4f}")
        report.append("")
        
        # 性能差异
        improvement = ((best_model[1]['mean'] - worst_model[1]['mean']) / abs(worst_model[1]['mean']) * 100) if worst_model[1]['mean'] != 0 else 0
        report.append(f"性能差异: {best_model[0]} 比 {worst_model[0]} 高出 {improvement:.1f}%")
        report.append("")
        
        # 质量分布对比
        report.append("质量分布对比 (相似度分档):")
        report.append("")
        
        quality_brackets = [
            (0.8, 1.0, "优秀 (0.8-1.0)"),
            (0.6, 0.8, "良好 (0.6-0.8)"),
            (0.4, 0.6, "中等 (0.4-0.6)"),
            (0.2, 0.4, "一般 (0.2-0.4)"),
            (0.0, 0.2, "差 (0.0-0.2)")
        ]
        
        for model_name, data in sorted(comparison.items()):
            report.append(f"  {model_name}:")
            similarities = data['similarities']
            
            for lower, upper, label in quality_brackets:
                count = sum(1 for s in similarities if lower <= s < upper)
                pct = (count / len(similarities) * 100) if similarities else 0
                bar = "█" * int(pct / 5)
                report.append(f"    {label:<18} {count:>3} ({pct:>5.1f}%) {bar}")
            report.append("")
    
    # 原始数据详情
    report.append("=" * 100)
    report.append("原始数据详情")
    report.append("=" * 100)
    report.append("")
    
    for model_name, data in sorted(comparison.items()):
        report.append(f"【{model_name}】 - 各样本相似度:")
        report.append("")
        
        results = data['results']['results']
        similarities = data['similarities']
        
        for i, result in enumerate(results):
            if result['success'] and i < len(similarities):
                sim = similarities[i]
                # 根据相似度评级
                if sim >= 0.8:
                    level = "🟢 优秀"
                elif sim >= 0.6:
                    level = "🟡 良好"
                elif sim >= 0.4:
                    level = "🟠 中等"
                elif sim >= 0.2:
                    level = "🟤 一般"
                else:
                    level = "🔴 差"
                
                pair_id = result['pair_id']
                audio_name = result['audio_name']
                video_name = result['video_name']
                description = result.get('description', '')
                
                report.append(f"  {pair_id:<12} {sim:.4f}  {level}  {description}")
        
        report.append("")
    
    report_text = "\n".join(report)
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ 报告已保存: {output_file}")
    
    return report_text


def generate_csv_comparison(comparison: Dict, output_file: str = None) -> pd.DataFrame:
    """生成CSV对比表"""
    rows = []
    
    for model_name, data in sorted(comparison.items()):
        results = data['results']['results']
        
        for i, result in enumerate(results):
            if result['success']:
                row = {
                    '模型': model_name,
                    '配对ID': result['pair_id'],
                    '音频': result['audio_name'],
                    '视频': result['video_name'],
                    '描述': result.get('description', ''),
                    '相似度': result['metrics']['similarity'],
                    '状态': '✓'
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ CSV已保存: {output_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="模型对比分析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 对比两个模型
  python compare_models.py --sora2_results batch_results_sora2/batch_results.json \
                           --veo3_results batch_results_veo3/batch_results.json
  
  # 指定输出文件
  python compare_models.py --sora2_results batch_results_sora2/batch_results.json \
                           --veo3_results batch_results_veo3/batch_results.json \
                           --output comparison_report.txt \
                           --csv comparison_results.csv
        """
    )
    
    parser.add_argument("--sora2_results", type=str, help="SORA2结果JSON文件")
    parser.add_argument("--veo3_results", type=str, help="VEO3结果JSON文件")
    parser.add_argument("--model1_results", type=str, help="模型1结果JSON文件")
    parser.add_argument("--model2_results", type=str, help="模型2结果JSON文件")
    parser.add_argument("--model3_results", type=str, help="模型3结果JSON文件")
    parser.add_argument("--output", type=str, default="comparison_report.txt", help="输出报告文件")
    parser.add_argument("--csv", type=str, help="输出CSV文件")
    
    args = parser.parse_args()
    
    # 收集所有模型结果
    model_results = {}
    
    if args.sora2_results and Path(args.sora2_results).exists():
        model_results['SORA2'] = load_batch_results(args.sora2_results)
    
    if args.veo3_results and Path(args.veo3_results).exists():
        model_results['VEO3'] = load_batch_results(args.veo3_results)
    
    if args.model1_results and Path(args.model1_results).exists():
        model_results['模型1'] = load_batch_results(args.model1_results)
    
    if args.model2_results and Path(args.model2_results).exists():
        model_results['模型2'] = load_batch_results(args.model2_results)
    
    if args.model3_results and Path(args.model3_results).exists():
        model_results['模型3'] = load_batch_results(args.model3_results)
    
    if not model_results:
        print("错误: 未找到任何结果文件!")
        print("请使用 --sora2_results, --veo3_results 或 --model1_results 等参数指定结果文件")
        return
    
    print(f"加载了 {len(model_results)} 个模型的结果\n")
    
    # 进行对比
    comparison = compare_models(model_results)
    
    # 生成报告
    print("生成对比报告...")
    report = generate_comparison_report(comparison, args.output)
    print(report)
    
    # 生成CSV
    if args.csv:
        df = generate_csv_comparison(comparison, args.csv)
        print(f"\n详细数据预览:")
        print(df.head(20))


if __name__ == "__main__":
    main()



