#!/usr/bin/env python3
"""
音频-视频配对批量测试脚本

支持多个音频-视频配对的批量评估，并生成汇总对比报告。

用法示例:
1. 从CSV文件读取配对（推荐）：
   python batch_pairs_test.py --pairs_file pairs.csv --output_dir ./batch_results

2. 从TXT文件读取配对（每行格式: /path/to/audio.wav,/path/to/video.mp4）：
   python batch_pairs_test.py --pairs_file pairs.txt --output_dir ./batch_results

3. 指定设备和批处理大小：
   python batch_pairs_test.py --pairs_file pairs.csv --output_dir ./batch_results --device cuda:0 --batch_size 2

配对文件格式（CSV）:
  pair_id,audio_path,video_path,description
  pair_1,/path/to/audio1.wav,/path/to/video1.mp4,测试样本1
  pair_2,/path/to/audio2.wav,/path/to/video2.mp4,测试样本2

配对文件格式（TXT）:
  /path/to/audio1.wav,/path/to/video1.mp4
  /path/to/audio2.wav,/path/to/video2.mp4
"""

import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from datetime import datetime

try:
    from batch_inference import (
        batch_inference_av_consistency,
        compute_consistency_metrics
    )
    from imagebind import data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在 ImageBind 目录下运行此脚本")


def load_pairs_from_csv(csv_path: str) -> List[Dict]:
    """从CSV文件读取配对信息"""
    df = pd.read_csv(csv_path)
    pairs = []
    for idx, row in df.iterrows():
        pair = {
            'pair_id': str(row.get('pair_id', f'pair_{idx+1}')),
            'audio_path': str(row['audio_path']),
            'video_path': str(row['video_path']),
            'description': str(row.get('description', ''))
        }
        pairs.append(pair)
    return pairs


def load_pairs_from_txt(txt_path: str) -> List[Dict]:
    """从TXT文件读取配对信息（每行: audio_path,video_path）"""
    pairs = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                pair = {
                    'pair_id': f'pair_{idx+1}',
                    'audio_path': parts[0].strip(),
                    'video_path': parts[1].strip(),
                    'description': parts[2].strip() if len(parts) > 2 else ''
                }
                pairs.append(pair)
    return pairs


def load_pairs(pairs_file: str) -> List[Dict]:
    """自动判断文件格式并读取配对"""
    if pairs_file.endswith('.csv'):
        return load_pairs_from_csv(pairs_file)
    else:
        return load_pairs_from_txt(pairs_file)


def validate_pairs(pairs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """验证配对文件是否存在"""
    valid_pairs = []
    invalid_pairs = []
    
    for pair in pairs:
        audio_path = pair['audio_path']
        video_path = pair['video_path']
        
        if not os.path.exists(audio_path):
            invalid_pairs.append({**pair, 'error': f'音频文件不存在: {audio_path}'})
        elif not os.path.exists(video_path):
            invalid_pairs.append({**pair, 'error': f'视频文件不存在: {video_path}'})
        else:
            valid_pairs.append(pair)
    
    return valid_pairs, invalid_pairs


def process_single_pair(
    pair: Dict,
    model,
    device: str,
    output_dir: str
) -> Dict:
    """处理单个音频-视频配对"""
    pair_id = pair['pair_id']
    audio_path = pair['audio_path']
    video_path = pair['video_path']
    
    result = {
        'pair_id': pair_id,
        'audio_path': audio_path,
        'video_path': video_path,
        'audio_name': Path(audio_path).stem,
        'video_name': Path(video_path).stem,
        'description': pair.get('description', ''),
        'success': False,
        'error': None,
        'metrics': {}
    }
    
    try:
        # 加载音频
        audio_inputs = data.load_and_transform_audio_data([audio_path], device)
        with torch.no_grad():
            audio_outputs = model({ModalityType.AUDIO: audio_inputs})
            audio_embedding = audio_outputs[ModalityType.AUDIO]
        
        # 加载视频
        video_inputs = data.load_and_transform_video_data([video_path], device)
        with torch.no_grad():
            video_outputs = model({ModalityType.VISION: video_inputs})
            video_embedding = video_outputs[ModalityType.VISION]
        
        # 计算相似度
        audio_norm = torch.nn.functional.normalize(audio_embedding, p=2, dim=1)
        video_norm = torch.nn.functional.normalize(video_embedding, p=2, dim=1)
        similarity = (audio_norm[0] * video_norm[0]).sum().item()
        
        # 保存嵌入向量
        pair_output_dir = os.path.join(output_dir, pair_id)
        os.makedirs(pair_output_dir, exist_ok=True)
        
        np.save(os.path.join(pair_output_dir, 'audio_embedding.npy'), audio_embedding.cpu().numpy())
        np.save(os.path.join(pair_output_dir, 'video_embedding.npy'), video_embedding.cpu().numpy())
        
        result['metrics'] = {
            'similarity': float(similarity),
            'audio_embedding_shape': list(audio_embedding.shape),
            'video_embedding_shape': list(video_embedding.shape)
        }
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
    
    return result


def batch_process_pairs(
    pairs: List[Dict],
    device: str = "cuda:0",
    output_dir: str = "./batch_results"
) -> Dict:
    """批量处理多个音频-视频配对"""
    
    # 检查设备
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = "cpu"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"正在加载 ImageBind 模型 (设备: {device})...")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    print("模型加载完成!\n")
    
    # 验证配对
    print(f"验证 {len(pairs)} 个配对...")
    valid_pairs, invalid_pairs = validate_pairs(pairs)
    print(f"有效配对: {len(valid_pairs)}, 无效配对: {len(invalid_pairs)}\n")
    
    if invalid_pairs:
        print("⚠️  无效配对:")
        for pair in invalid_pairs:
            print(f"  - {pair['pair_id']}: {pair['error']}")
        print()
    
    # 处理配对
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'total_pairs': len(pairs),
        'valid_pairs': len(valid_pairs),
        'invalid_pairs': len(invalid_pairs),
        'results': []
    }
    
    print(f"处理 {len(valid_pairs)} 个配对...")
    for pair in tqdm(valid_pairs, desc="处理配对"):
        result = process_single_pair(pair, model, device, output_dir)
        all_results['results'].append(result)
    
    # 统计
    successful = sum(1 for r in all_results['results'] if r['success'])
    failed = len(all_results['results']) - successful
    all_results['summary'] = {
        'successful': successful,
        'failed': failed
    }
    
    return all_results


def save_batch_results(all_results: Dict, output_dir: str):
    """保存批量测试结果"""
    
    # 保存JSON结果
    with open(os.path.join(output_dir, 'batch_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 生成对比报告
    report_path = os.path.join(output_dir, 'batch_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("音频-视频配对批量测试对比报告\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"测试时间: {all_results['timestamp']}\n")
        f.write(f"计算设备: {all_results['device']}\n")
        f.write(f"总配对数: {all_results['total_pairs']}\n")
        f.write(f"有效配对: {all_results['valid_pairs']}\n")
        f.write(f"无效配对: {all_results['invalid_pairs']}\n\n")
        
        f.write("处理结果汇总:\n")
        f.write(f"  成功: {all_results['summary']['successful']}\n")
        f.write(f"  失败: {all_results['summary']['failed']}\n\n")
        
        # 提取相似度数据
        similarities = []
        for result in all_results['results']:
            if result['success'] and 'similarity' in result['metrics']:
                similarities.append(result['metrics']['similarity'])
        
        if similarities:
            f.write("相似度统计:\n")
            f.write(f"  平均相似度: {np.mean(similarities):.4f}\n")
            f.write(f"  标准差: {np.std(similarities):.4f}\n")
            f.write(f"  最小值: {np.min(similarities):.4f}\n")
            f.write(f"  最大值: {np.max(similarities):.4f}\n")
            f.write(f"  中位数: {np.median(similarities):.4f}\n\n")
        
        # 详细结果表格
        f.write("=" * 100 + "\n")
        f.write("详细结果:\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'ID':<15} {'音频':<30} {'视频':<30} {'相似度':<12} {'状态':<10} {'描述':<20}\n")
        f.write("-" * 100 + "\n")
        
        for result in all_results['results']:
            pair_id = result['pair_id'][:14]
            audio_name = result['audio_name'][:29]
            video_name = result['video_name'][:29]
            
            if result['success']:
                similarity = f"{result['metrics']['similarity']:.4f}"
                status = "✓ 成功"
            else:
                similarity = "N/A"
                status = "✗ 失败"
            
            description = result['description'][:19]
            
            f.write(f"{pair_id:<15} {audio_name:<30} {video_name:<30} {similarity:<12} {status:<10} {description:<20}\n")
        
        # 失败详情
        failed_results = [r for r in all_results['results'] if not r['success']]
        if failed_results:
            f.write("\n" + "=" * 100 + "\n")
            f.write("失败详情:\n")
            f.write("=" * 100 + "\n\n")
            
            for result in failed_results:
                f.write(f"配对ID: {result['pair_id']}\n")
                f.write(f"  错误: {result['error']}\n")
                f.write(f"  音频: {result['audio_path']}\n")
                f.write(f"  视频: {result['video_path']}\n\n")


def generate_csv_summary(all_results: Dict, output_dir: str):
    """生成CSV汇总表"""
    rows = []
    for result in all_results['results']:
        row = {
            '配对ID': result['pair_id'],
            '音频文件': result['audio_name'],
            '视频文件': result['video_name'],
            '描述': result['description'],
            '状态': '成功' if result['success'] else '失败',
            '相似度': result['metrics'].get('similarity', 'N/A') if result['success'] else 'N/A',
            '错误信息': result['error'] if not result['success'] else ''
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'batch_results_summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存CSV汇总: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="音频-视频配对批量测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从CSV文件读取配对
  python batch_pairs_test.py --pairs_file pairs.csv --output_dir ./batch_results
  
  # 从TXT文件读取配对
  python batch_pairs_test.py --pairs_file pairs.txt --output_dir ./batch_results
  
  # 指定设备
  python batch_pairs_test.py --pairs_file pairs.csv --output_dir ./batch_results --device cuda:1

配对文件格式示例:
  CSV: pair_id,audio_path,video_path,description
  TXT: /path/to/audio.wav,/path/to/video.mp4[,描述]
        """
    )
    
    parser.add_argument(
        "--pairs_file",
        type=str,
        required=True,
        help="配对列表文件 (CSV或TXT格式)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./batch_results",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备 (default: cuda:0)"
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.pairs_file):
        print(f"错误: 配对文件不存在: {args.pairs_file}")
        return
    
    # 读取配对
    print(f"读取配对文件: {args.pairs_file}\n")
    pairs = load_pairs(args.pairs_file)
    print(f"共读取 {len(pairs)} 个配对\n")
    
    # 批量处理
    all_results = batch_process_pairs(
        pairs=pairs,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # 保存结果
    print("\n保存结果...")
    save_batch_results(all_results, args.output_dir)
    generate_csv_summary(all_results, args.output_dir)
    
    # 打印汇总
    print("\n" + "=" * 100)
    print("批量测试完成!")
    print("=" * 100)
    print(f"✓ 成功: {all_results['summary']['successful']}")
    print(f"✗ 失败: {all_results['summary']['failed']}")
    print(f"\n结果已保存到: {args.output_dir}")
    print(f"  - batch_results.json (详细JSON结果)")
    print(f"  - batch_comparison_report.txt (文本报告)")
    print(f"  - batch_results_summary.csv (CSV汇总表)")


if __name__ == "__main__":
    main()



