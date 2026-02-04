#!/usr/bin/env python3
"""
音频-文本相似度批量推理脚本

从 JSON 文件读取 audio_prompt，计算音频和对应文本的相似度。

用法示例:
python batch_inference_audio_text.py \
    --json_file /root/bayes-tmp/data/av_data.json \
    --audio_dir /root/bayes-tmp/data/audios/model_name \
    --output_file results_audio_text.json \
    --device cuda:0
"""

import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def load_json_data(json_path: str) -> List[Dict]:
    """加载 JSON 数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_audio_file(audio_dir: str, index: int, extensions: List[str] = None) -> str:
    """
    根据索引查找音频文件
    
    支持的命名格式:
    - sample_{index:04d}.wav (如 sample_0001.wav)
    - {index}.wav
    - audio_{index}.wav
    等
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    audio_dir = Path(audio_dir)
    
    # 尝试多种命名格式
    patterns = [
        f'sample_{index:04d}',
        f'sample_{index}',
        f'{index:04d}',
        f'{index}',
        f'audio_{index:04d}',
        f'audio_{index}',
    ]
    
    for pattern in patterns:
        for ext in extensions:
            audio_path = audio_dir / f'{pattern}{ext}'
            if audio_path.exists():
                return str(audio_path)
    
    return None


def process_single_item(
    item: Dict,
    audio_path: str,
    model,
    device: str
) -> Dict[str, Any]:
    """处理单个音频-文本对"""
    
    result = {
        'index': item['index'],
        'audio_path': audio_path,
        'audio_prompt': item.get('audio_prompt', ''),
        'success': False,
        'error': None,
        'similarity': None,
        'audio_embedding_shape': None,
        'text_embedding_shape': None
    }
    
    try:
        # 加载音频
        audio_inputs = data.load_and_transform_audio_data([audio_path], device)
        
        # 加载文本
        text_prompt = item.get('audio_prompt', '')
        if not text_prompt:
            raise ValueError("audio_prompt is empty")
        
        text_inputs = data.load_and_transform_text([text_prompt], device)
        
        # 模型推理
        with torch.no_grad():
            embeddings = model({
                ModalityType.AUDIO: audio_inputs,
                ModalityType.TEXT: text_inputs
            })
        
        audio_embedding = embeddings[ModalityType.AUDIO]
        text_embedding = embeddings[ModalityType.TEXT]
        
        # 计算相似度（余弦相似度）
        audio_norm = torch.nn.functional.normalize(audio_embedding, p=2, dim=1)
        text_norm = torch.nn.functional.normalize(text_embedding, p=2, dim=1)
        similarity = (audio_norm[0] * text_norm[0]).sum().item()
        
        result['similarity'] = float(similarity)
        result['audio_embedding_shape'] = list(audio_embedding.shape)
        result['text_embedding_shape'] = list(text_embedding.shape)
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
    
    return result


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算统计信息"""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_count': len(results),
        'successful_count': 0,
        'failed_count': 0
    }
    
    # 过滤出成功的结果
    successful_results = [r for r in results if r['success']]
    stats['successful_count'] = len(successful_results)
    stats['failed_count'] = len(results) - len(successful_results)
    
    if not successful_results:
        return stats
    
    # 提取相似度
    similarities = [r['similarity'] for r in successful_results 
                   if r['similarity'] is not None]
    
    if similarities:
        stats['similarity'] = {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities))
        }
    
    return stats


def batch_inference(
    json_data: List[Dict],
    audio_dir: str,
    device: str = 'cuda:0'
) -> Dict[str, Any]:
    """批量推理"""
    
    # 检查设备
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = "cpu"
    
    # 加载模型
    print(f"正在加载 ImageBind 模型 (设备: {device})...")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    print("模型加载完成!\n")
    
    # 处理每个条目
    results = []
    skipped_count = 0
    
    print(f"处理 {len(json_data)} 个条目...")
    for item in tqdm(json_data, desc="处理音频-文本对"):
        # 查找音频文件
        audio_path = find_audio_file(audio_dir, item['index'])
        
        if audio_path is None:
            skipped_count += 1
            result = {
                'index': item['index'],
                'audio_path': None,
                'audio_prompt': item.get('audio_prompt', ''),
                'success': False,
                'error': f"音频文件未找到 (index: {item['index']})",
                'similarity': None
            }
            results.append(result)
            continue
        
        # 处理音频-文本对
        result = process_single_item(item, audio_path, model, device)
        results.append(result)
    
    if skipped_count > 0:
        print(f"\n警告: {skipped_count} 个音频文件未找到")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='音频-文本相似度批量推理',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--json_file',
        type=str,
        required=True,
        help='包含 audio_prompt 的 JSON 文件路径'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        required=True,
        help='音频文件所在目录'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='results_audio_text.json',
        help='输出 JSON 文件路径'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='计算设备 (default: cuda:0)'
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.json_file):
        print(f"错误: JSON 文件不存在: {args.json_file}")
        return
    
    if not os.path.exists(args.audio_dir):
        print(f"错误: 音频目录不存在: {args.audio_dir}")
        return
    
    # 加载 JSON 数据
    print(f"读取 JSON 文件: {args.json_file}")
    json_data = load_json_data(args.json_file)
    print(f"共读取 {len(json_data)} 个条目\n")
    
    # 批量推理
    results = batch_inference(
        json_data=json_data,
        audio_dir=args.audio_dir,
        device=args.device
    )
    
    # 计算统计信息
    statistics = calculate_statistics(results)
    
    # 保存结果
    output_data = {
        'statistics': statistics,
        'results': results
    }
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {args.output_file}")
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print("统计摘要")
    print(f"{'='*80}")
    print(f"总数: {statistics['total_count']}")
    print(f"成功: {statistics['successful_count']}")
    print(f"失败: {statistics['failed_count']}")
    
    if 'similarity' in statistics:
        sim_stats = statistics['similarity']
        print(f"\n音频-文本相似度:")
        print(f"  平均值: {sim_stats['mean']:.6f}")
        print(f"  标准差: {sim_stats['std']:.6f}")
        print(f"  最小值: {sim_stats['min']:.6f}")
        print(f"  最大值: {sim_stats['max']:.6f}")
        print(f"  中位数: {sim_stats['median']:.6f}")


if __name__ == '__main__':
    main()


