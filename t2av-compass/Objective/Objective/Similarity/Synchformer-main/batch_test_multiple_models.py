"""
批量测试多个模型和多个视频目录的脚本
可以测试多个模型（exp_name）对多个视频目录的评估结果
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time

from batch_test_folder import batch_test_folder


def batch_test_multiple_models(
    video_dirs: List[str],
    exp_names: List[str],
    output_dir: str,
    device: str = 'cuda:0',
    offset_sec: float = 0.0,
    v_start_i_sec: float = 0.0,
    recursive: bool = False
) -> Dict[str, Any]:
    """
    批量测试多个模型和多个视频目录
    
    Args:
        video_dirs: 视频目录列表
        exp_names: 模型实验名称列表
        output_dir: 输出目录
        device: 设备
        offset_sec: 音频偏移秒数
        v_start_i_sec: 视频起始秒数
        recursive: 是否递归搜索子文件夹
        
    Returns:
        包含所有测试结果的字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': len(video_dirs) * len(exp_names),
        'completed_tests': 0,
        'failed_tests': 0,
        'results': []
    }
    
    print(f"开始批量测试:")
    print(f"  视频目录数: {len(video_dirs)}")
    print(f"  模型数: {len(exp_names)}")
    print(f"  总测试数: {all_results['total_tests']}")
    print(f"  输出目录: {output_dir}\n")
    
    # 遍历每个模型和每个视频目录
    for exp_name in exp_names:
        print(f"\n{'='*60}")
        print(f"测试模型: {exp_name}")
        print(f"{'='*60}")
        
        for video_dir in video_dirs:
            video_dir_name = Path(video_dir).name
            output_filename = f"{video_dir_name}_{exp_name}.json"
            output_file = output_path / output_filename
            
            # 检查是否已存在结果（断点续传）
            if output_file.exists():
                print(f"\n跳过已存在的测试: {video_dir_name} (模型: {exp_name})")
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_result = json.load(f)
                    all_results['results'].append({
                        'video_dir': video_dir,
                        'exp_name': exp_name,
                        'output_file': str(output_file),
                        'status': 'skipped',
                        'result': existing_result
                    })
                    all_results['completed_tests'] += 1
                    continue
                except:
                    print(f"  警告: 无法读取已有结果，将重新测试")
            
            print(f"\n处理视频目录: {video_dir}")
            print(f"  输出文件: {output_file}")
            
            try:
                # 批量测试
                result = batch_test_folder(
                    folder_path=video_dir,
                    exp_name=exp_name,
                    device=device,
                    offset_sec=offset_sec,
                    v_start_i_sec=v_start_i_sec,
                    recursive=recursive
                )
                
                # 保存单个结果
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # 记录结果
                all_results['results'].append({
                    'video_dir': video_dir,
                    'exp_name': exp_name,
                    'output_file': str(output_file),
                    'status': 'success',
                    'result': result
                })
                all_results['completed_tests'] += 1
                
                # 打印统计信息
                stats = result.get('statistics', {})
                print(f"  完成! 成功: {stats.get('successful_videos', 0)}, "
                      f"失败: {stats.get('failed_videos', 0)}")
                if stats.get('successful_videos', 0) > 0:
                    print(f"  平均偏移: {stats.get('average_offset_sec', 0):.4f} 秒")
                    print(f"  平均置信度: {stats.get('average_probability', 0):.4f}")
                
            except Exception as e:
                print(f"  错误: {str(e)}")
                all_results['results'].append({
                    'video_dir': video_dir,
                    'exp_name': exp_name,
                    'output_file': str(output_file),
                    'status': 'failed',
                    'error': str(e)
                })
                all_results['failed_tests'] += 1
    
    # 保存汇总结果
    summary_file = output_path / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"批量测试完成!")
    print(f"  总测试数: {all_results['total_tests']}")
    print(f"  完成数: {all_results['completed_tests']}")
    print(f"  失败数: {all_results['failed_tests']}")
    print(f"  汇总结果: {summary_file}")
    print(f"{'='*60}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='批量测试多个模型和多个视频目录')
    
    parser.add_argument(
        '--video_dirs',
        nargs='+',
        required=True,
        help='视频目录列表，用空格分隔'
    )
    
    parser.add_argument(
        '--exp_names',
        nargs='+',
        required=True,
        help='模型实验名称列表，用空格分隔（格式: xx-xx-xxTxx-xx-xx）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='设备（默认: cuda:0）'
    )
    
    parser.add_argument(
        '--offset_sec',
        type=float,
        default=0.0,
        help='音频偏移秒数（默认: 0.0）'
    )
    
    parser.add_argument(
        '--v_start_i_sec',
        type=float,
        default=0.0,
        help='视频起始秒数（默认: 0.0）'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='递归搜索子文件夹'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='不从中断处继续（重新开始所有测试）'
    )
    
    args = parser.parse_args()
    
    # 如果指定了不续传，删除已有结果
    if args.no_resume:
        output_path = Path(args.output_dir)
        if output_path.exists():
            for file in output_path.glob('*.json'):
                if file.name != 'summary.json':
                    file.unlink()
            print("已清除已有结果，将重新开始所有测试")
    
    # 批量测试
    batch_test_multiple_models(
        video_dirs=args.video_dirs,
        exp_names=args.exp_names,
        output_dir=args.output_dir,
        device=args.device,
        offset_sec=args.offset_sec,
        v_start_i_sec=args.v_start_i_sec,
        recursive=args.recursive
    )


if __name__ == '__main__':
    main()


