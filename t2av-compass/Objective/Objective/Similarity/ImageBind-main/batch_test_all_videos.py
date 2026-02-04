"""
批量测试多个视频目录的 ImageBind 推理脚本
支持三种模式：
1. 视频-文本相似度
2. 音频-文本相似度
3. 音频-视频一致性
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import batch_inference_video_text
import batch_inference_audio_text
from batch_inference import batch_inference_av_consistency


def load_json_data(json_path: str) -> List[Dict]:
    """加载 JSON 数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def batch_test_video_text(
    video_dirs: List[str],
    json_file: str,
    output_dir: str,
    device: str = 'cuda:0'
) -> Dict[str, Any]:
    """批量测试视频-文本相似度"""
    print(f"\n{'='*60}")
    print("批量测试视频-文本相似度")
    print(f"{'='*60}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载 JSON 数据
    json_data = load_json_data(json_file)
    
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'video_text',
        'json_file': json_file,
        'total_tests': len(video_dirs),
        'completed_tests': 0,
        'failed_tests': 0,
        'results': []
    }
    
    for video_dir in video_dirs:
        video_dir_name = Path(video_dir).name
        output_file = output_path / f"{video_dir_name}_video_text.json"
        
        # 检查是否已存在结果
        if output_file.exists():
            print(f"\n跳过已存在的测试: {video_dir_name}")
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                all_results['results'].append({
                    'video_dir': video_dir,
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
            # 批量推理
            results = batch_inference_video_text.batch_inference(
                json_data=json_data,
                video_dir=video_dir,
                device=device
            )
            
            # 计算统计信息
            statistics = batch_inference_video_text.calculate_statistics(results)
            
            # 保存结果
            output_data = {
                'video_dir': video_dir,
                'json_file': json_file,
                'statistics': statistics,
                'results': results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            all_results['results'].append({
                'video_dir': video_dir,
                'output_file': str(output_file),
                'status': 'success',
                'result': output_data
            })
            all_results['completed_tests'] += 1
            
            # 打印统计信息
            stats = statistics
            print(f"  完成! 成功: {stats.get('successful_count', 0)}, "
                  f"失败: {stats.get('failed_count', 0)}")
            if 'similarity' in stats:
                sim_stats = stats['similarity']
                print(f"  平均相似度: {sim_stats.get('mean', 0):.6f}")
            
        except Exception as e:
            print(f"  错误: {str(e)}")
            all_results['results'].append({
                'video_dir': video_dir,
                'output_file': str(output_file),
                'status': 'failed',
                'error': str(e)
            })
            all_results['failed_tests'] += 1
    
    return all_results


def batch_test_audio_text(
    audio_dirs: List[str],
    json_file: str,
    output_dir: str,
    device: str = 'cuda:0'
) -> Dict[str, Any]:
    """批量测试音频-文本相似度"""
    print(f"\n{'='*60}")
    print("批量测试音频-文本相似度")
    print(f"{'='*60}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载 JSON 数据
    json_data = load_json_data(json_file)
    
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'audio_text',
        'json_file': json_file,
        'total_tests': len(audio_dirs),
        'completed_tests': 0,
        'failed_tests': 0,
        'results': []
    }
    
    for audio_dir in audio_dirs:
        audio_dir_name = Path(audio_dir).name
        output_file = output_path / f"{audio_dir_name}_audio_text.json"
        
        # 检查是否已存在结果
        if output_file.exists():
            print(f"\n跳过已存在的测试: {audio_dir_name}")
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                all_results['results'].append({
                    'audio_dir': audio_dir,
                    'output_file': str(output_file),
                    'status': 'skipped',
                    'result': existing_result
                })
                all_results['completed_tests'] += 1
                continue
            except:
                print(f"  警告: 无法读取已有结果，将重新测试")
        
        print(f"\n处理音频目录: {audio_dir}")
        print(f"  输出文件: {output_file}")
        
        try:
            # 批量推理
            results = batch_inference_audio_text.batch_inference(
                json_data=json_data,
                audio_dir=audio_dir,
                device=device
            )
            
            # 计算统计信息
            statistics = batch_inference_audio_text.calculate_statistics(results)
            
            # 保存结果
            output_data = {
                'audio_dir': audio_dir,
                'json_file': json_file,
                'statistics': statistics,
                'results': results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            all_results['results'].append({
                'audio_dir': audio_dir,
                'output_file': str(output_file),
                'status': 'success',
                'result': output_data
            })
            all_results['completed_tests'] += 1
            
            # 打印统计信息
            stats = statistics
            print(f"  完成! 成功: {stats.get('successful_count', 0)}, "
                  f"失败: {stats.get('failed_count', 0)}")
            if 'similarity' in stats:
                sim_stats = stats['similarity']
                print(f"  平均相似度: {sim_stats.get('mean', 0):.6f}")
            
        except Exception as e:
            print(f"  错误: {str(e)}")
            all_results['results'].append({
                'audio_dir': audio_dir,
                'output_file': str(output_file),
                'status': 'failed',
                'error': str(e)
            })
            all_results['failed_tests'] += 1
    
    return all_results


def batch_test_av_consistency(
    video_dirs: List[str],
    audio_dirs: Optional[List[str]],
    output_dir: str,
    device: str = 'cuda:0',
    batch_size: int = 4,
    retrieval_mode: bool = False
) -> Dict[str, Any]:
    """批量测试音频-视频一致性"""
    print(f"\n{'='*60}")
    print("批量测试音频-视频一致性")
    print(f"{'='*60}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'av_consistency',
        'total_tests': len(video_dirs),
        'completed_tests': 0,
        'failed_tests': 0,
        'results': []
    }
    
    for i, video_dir in enumerate(video_dirs):
        video_dir_name = Path(video_dir).name
        
        # 确定音频目录
        if audio_dirs and i < len(audio_dirs):
            audio_dir = audio_dirs[i]
        else:
            # 默认使用视频目录作为音频目录（音频和视频在同一目录）
            audio_dir = video_dir
        
        output_subdir = output_path / f"{video_dir_name}_av_consistency"
        output_file = output_path / f"{video_dir_name}_av_consistency.json"
        
        # 检查是否已存在结果
        if output_file.exists():
            print(f"\n跳过已存在的测试: {video_dir_name}")
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_result = json.load(f)
                all_results['results'].append({
                    'video_dir': video_dir,
                    'audio_dir': audio_dir,
                    'output_file': str(output_file),
                    'status': 'skipped',
                    'result': existing_result
                })
                all_results['completed_tests'] += 1
                continue
            except:
                print(f"  警告: 无法读取已有结果，将重新测试")
        
        print(f"\n处理视频目录: {video_dir}")
        print(f"  音频目录: {audio_dir}")
        print(f"  输出目录: {output_subdir}")
        
        # 检查目录是否存在
        if not os.path.exists(video_dir):
            print(f"  警告: 视频目录不存在，跳过")
            all_results['failed_tests'] += 1
            continue
        
        if not os.path.exists(audio_dir):
            print(f"  警告: 音频目录不存在，跳过")
            all_results['failed_tests'] += 1
            continue
        
        try:
            # 获取音频和视频文件列表
            from batch_inference import get_files_from_dir
            
            audio_paths = get_files_from_dir(audio_dir, ['.wav', '.mp3', '.flac', '.m4a', '.ogg'])
            video_paths = get_files_from_dir(video_dir, ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'])
            
            if not audio_paths:
                print(f"  警告: 音频目录中没有找到音频文件")
                all_results['failed_tests'] += 1
                continue
            
            if not video_paths:
                print(f"  警告: 视频目录中没有找到视频文件")
                all_results['failed_tests'] += 1
                continue
            
            # 批量推理
            results = batch_inference_av_consistency(
                audio_paths=audio_paths,
                video_paths=video_paths,
                device=device,
                batch_size=batch_size,
                output_dir=str(output_subdir),
                retrieval_mode=retrieval_mode
            )
            
            # 保存结果摘要
            # 清理 metrics，排除无法 JSON 序列化的 ndarray 字段
            metrics = results.get('metrics', {}) if results else {}
            metrics_clean = {k: v for k, v in metrics.items() if k != 'similarity_matrix'}
            
            output_data = {
                'video_dir': video_dir,
                'audio_dir': audio_dir,
                'audio_count': len(audio_paths),
                'video_count': len(video_paths),
                'metrics': metrics_clean
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            all_results['results'].append({
                'video_dir': video_dir,
                'audio_dir': audio_dir,
                'output_file': str(output_file),
                'status': 'success',
                'result': output_data
            })
            all_results['completed_tests'] += 1
            
            # 打印统计信息
            if results and 'metrics' in results:
                metrics = results['metrics']
                if 'paired_similarity_mean' in metrics:
                    print(f"  完成! 平均配对相似度: {metrics['paired_similarity_mean']:.4f}")
                elif 'similarity_matrix_mean' in metrics:
                    print(f"  完成! 平均相似度: {metrics['similarity_matrix_mean']:.4f}")
            
        except Exception as e:
            print(f"  错误: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results['results'].append({
                'video_dir': video_dir,
                'audio_dir': audio_dir,
                'output_file': str(output_file),
                'status': 'failed',
                'error': str(e)
            })
            all_results['failed_tests'] += 1
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='批量测试多个视频目录的 ImageBind 推理')
    
    # 模式选择
    parser.add_argument(
        '--mode',
        type=str,
        choices=['video_text', 'audio_text', 'av_consistency', 'all'],
        default='all',
        help='测试模式 (default: all)'
    )
    
    # 输入
    parser.add_argument(
        '--video_dirs',
        nargs='+',
        required=True,
        help='视频目录列表，用空格分隔'
    )
    
    parser.add_argument(
        '--audio_dirs',
        nargs='+',
        help='音频目录列表（仅用于 av_consistency 模式），如果未指定则假设在视频目录的 audio 子目录中'
    )
    
    parser.add_argument(
        '--json_file',
        type=str,
        help='JSON 数据文件路径（用于 video_text 和 audio_text 模式）'
    )
    
    # 输出
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    
    # 其他参数
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='设备（默认: cuda:0）'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批处理大小（仅用于 av_consistency 模式，默认: 4）'
    )
    
    parser.add_argument(
        '--retrieval_mode',
        action='store_true',
        help='检索模式（仅用于 av_consistency 模式）'
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
                file.unlink()
            print("已清除已有结果，将重新开始所有测试")
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_summaries = {}
    
    # 根据模式执行测试
    if args.mode in ['video_text', 'all']:
        if not args.json_file:
            print("错误: video_text 模式需要 --json_file 参数")
            return
        summary = batch_test_video_text(
            video_dirs=args.video_dirs,
            json_file=args.json_file,
            output_dir=str(output_path / 'video_text'),
            device=args.device
        )
        all_summaries['video_text'] = summary
    
    if args.mode in ['audio_text', 'all']:
        if not args.json_file:
            print("错误: audio_text 模式需要 --json_file 参数")
            return
        # 对于 audio_text 模式，使用 audio_dirs 或 video_dirs
        audio_dirs = args.audio_dirs if args.audio_dirs else args.video_dirs
        summary = batch_test_audio_text(
            audio_dirs=audio_dirs,
            json_file=args.json_file,
            output_dir=str(output_path / 'audio_text'),
            device=args.device
        )
        all_summaries['audio_text'] = summary
    
    if args.mode in ['av_consistency', 'all']:
        summary = batch_test_av_consistency(
            video_dirs=args.video_dirs,
            audio_dirs=args.audio_dirs,
            output_dir=str(output_path / 'av_consistency'),
            device=args.device,
            batch_size=args.batch_size,
            retrieval_mode=args.retrieval_mode
        )
        all_summaries['av_consistency'] = summary
    
    # 保存汇总结果
    summary_file = output_path / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("批量测试完成!")
    print(f"{'='*60}")
    for mode, summary in all_summaries.items():
        print(f"\n{mode}:")
        print(f"  总测试数: {summary.get('total_tests', 0)}")
        print(f"  完成数: {summary.get('completed_tests', 0)}")
        print(f"  失败数: {summary.get('failed_tests', 0)}")
    print(f"\n汇总结果: {summary_file}")


if __name__ == '__main__':
    main()

