#!/usr/bin/env python3
"""
ImageBind 音频-视频一致性批量推理脚本

专门用于测试音频和视频的一致性/匹配度。

支持两种模式：
1. 配对模式：音频和视频文件一一对应（文件名相同或指定配对文件）
2. 检索模式：计算所有音频-视频对的相似度矩阵

用法示例：
1. 配对模式（文件一一对应）：
   python batch_inference.py --audio_dir ./audios --video_dir ./videos --output_dir ./outputs

2. 检索模式（计算所有配对）：
   python batch_inference.py --audio_dir ./audios --video_dir ./videos --output_dir ./outputs --retrieval_mode

3. 从文件列表读取：
   python batch_inference.py --audio_file audio_list.txt --video_file video_list.txt --output_dir ./outputs
"""

import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def load_path_list(path_file: str) -> List[str]:
    """从文件中读取路径列表"""
    with open(path_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def get_files_from_dir(directory: str, extensions: List[str]) -> List[str]:
    """从目录中获取指定扩展名的所有文件"""
    directory = Path(directory)
    files = []
    for ext in extensions:
        files.extend(list(directory.glob(f"**/*{ext}")))
    return sorted([str(f) for f in files])


def match_audio_video_files(audio_paths: List[str], video_paths: List[str]) -> List[Tuple[str, str]]:
    """
    根据文件名匹配音频和视频文件
    
    Args:
        audio_paths: 音频文件路径列表
        video_paths: 视频文件路径列表
    
    Returns:
        匹配的 (音频路径, 视频路径) 对列表
    """
    pairs = []
    audio_dict = {Path(p).stem: p for p in audio_paths}
    video_dict = {Path(p).stem: p for p in video_paths}
    
    # 找到共同的文件名（不包含扩展名）
    common_stems = set(audio_dict.keys()) & set(video_dict.keys())
    
    for stem in sorted(common_stems):
        pairs.append((audio_dict[stem], video_dict[stem]))
    
    print(f"找到 {len(pairs)} 对匹配的音频-视频文件")
    if len(pairs) < len(audio_paths) or len(pairs) < len(video_paths):
        print(f"警告: 有 {len(audio_paths) - len(pairs)} 个音频文件未匹配")
        print(f"警告: 有 {len(video_paths) - len(pairs)} 个视频文件未匹配")
    
    return pairs


def compute_consistency_metrics(
    audio_embeddings: torch.Tensor,
    video_embeddings: torch.Tensor,
    pairs: Optional[List[Tuple[str, str]]] = None
) -> Dict:
    """
    计算音频-视频一致性指标
    
    Args:
        audio_embeddings: 音频嵌入向量 (N, 1024)
        video_embeddings: 视频嵌入向量 (M, 1024)
        pairs: 配对列表，如果为 None 则计算所有配对
    
    Returns:
        包含各种指标的字典
    """
    metrics = {}
    
    # 归一化嵌入向量（用于余弦相似度）
    audio_norm = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
    video_norm = torch.nn.functional.normalize(video_embeddings, p=2, dim=1)
    
    if pairs is not None and len(audio_embeddings) == len(video_embeddings):
        # 配对模式：只计算配对对的相似度
        # 在配对模式下，音频和视频顺序一致，直接计算配对对的点积
        paired_similarities = []
        n_pairs = min(len(audio_embeddings), len(video_embeddings))
        for i in range(n_pairs):
            # 只计算配对对的相似度（不计算完整矩阵）
            similarity = (audio_norm[i] * video_norm[i]).sum().item()
            paired_similarities.append(similarity)
        
        if paired_similarities:
            metrics['paired_similarity_mean'] = np.mean(paired_similarities)
            metrics['paired_similarity_std'] = np.std(paired_similarities)
            metrics['paired_similarity_min'] = np.min(paired_similarities)
            metrics['paired_similarity_max'] = np.max(paired_similarities)
            metrics['paired_similarities'] = paired_similarities
            
            # 配对模式下，相似度统计就是配对相似度的统计
            metrics['similarity_matrix_mean'] = metrics['paired_similarity_mean']
            metrics['similarity_matrix_std'] = metrics['paired_similarity_std']
            # 保存配对相似度数组（形状为 (n_pairs,)）
            metrics['similarity_matrix'] = np.array(paired_similarities)
    else:
        # 检索模式：计算完整的相似度矩阵
        similarity_matrix = audio_norm @ video_norm.T  # (N, M)
        
        # 整体统计
        metrics['similarity_matrix_mean'] = similarity_matrix.mean().item()
        metrics['similarity_matrix_std'] = similarity_matrix.std().item()
        metrics['similarity_matrix'] = similarity_matrix.cpu().numpy()
        
        # 检索模式：计算检索指标
        # 对于每个音频，找到最相似的视频的排名
        retrieval_ranks = []
        for i in range(len(audio_embeddings)):
            similarities = similarity_matrix[i].cpu().numpy()
            # 对于检索模式，假设正确配对是第 i 个视频（如果有配对信息）
            correct_idx = i if i < len(video_embeddings) else 0
            sorted_indices = np.argsort(similarities)[::-1]  # 降序排列
            rank = np.where(sorted_indices == correct_idx)[0][0] + 1  # 排名从1开始
            retrieval_ranks.append(rank)
        
        if retrieval_ranks:
            metrics['mean_rank'] = np.mean(retrieval_ranks)
            metrics['median_rank'] = np.median(retrieval_ranks)
            metrics['recall_at_1'] = np.mean([1 if r == 1 else 0 for r in retrieval_ranks])
            metrics['recall_at_5'] = np.mean([1 if r <= 5 else 0 for r in retrieval_ranks])
            metrics['recall_at_10'] = np.mean([1 if r <= 10 else 0 for r in retrieval_ranks])
    
    return metrics


def save_results(
    audio_embeddings: torch.Tensor,
    video_embeddings: torch.Tensor,
    audio_names: List[str],
    video_names: List[str],
    metrics: Dict,
    output_dir: str,
    pairs: Optional[List[Tuple[str, str]]] = None
):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存嵌入向量
    audio_np = audio_embeddings.cpu().numpy()
    video_np = video_embeddings.cpu().numpy()
    
    np.save(os.path.join(output_dir, "audio_embeddings.npy"), audio_np)
    np.save(os.path.join(output_dir, "video_embeddings.npy"), video_np)
    print(f"\n已保存嵌入向量:")
    print(f"  音频: {audio_np.shape}")
    print(f"  视频: {video_np.shape}")
    
    # 保存文件名列表
    with open(os.path.join(output_dir, "audio_names.txt"), 'w', encoding='utf-8') as f:
        for name in audio_names:
            f.write(f"{name}\n")
    
    with open(os.path.join(output_dir, "video_names.txt"), 'w', encoding='utf-8') as f:
        for name in video_names:
            f.write(f"{name}\n")
    
    # 保存配对信息
    if pairs:
        with open(os.path.join(output_dir, "pairs.txt"), 'w', encoding='utf-8') as f:
            for audio_path, video_path in pairs:
                f.write(f"{audio_path}\t{video_path}\n")
    
    # 保存相似度矩阵/数组
    similarity_data = metrics['similarity_matrix']
    np.save(os.path.join(output_dir, "similarity_matrix.npy"), similarity_data)
    
    # 保存指标
    metrics_to_save = {k: v for k, v in metrics.items() if k != 'similarity_matrix'}
    with open(os.path.join(output_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    
    # 保存可读的报告
    with open(os.path.join(output_dir, "consistency_report.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("音频-视频一致性评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"音频数量: {len(audio_names)}\n")
        f.write(f"视频数量: {len(video_names)}\n")
        if pairs:
            f.write(f"配对数量: {len(pairs)}\n")
        f.write("\n")
        
        if 'paired_similarity_mean' in metrics:
            # 配对模式：显示配对一致性
            f.write("配对一致性:\n")
            f.write(f"  平均配对相似度: {metrics['paired_similarity_mean']:.4f}\n")
            f.write(f"  标准差: {metrics['paired_similarity_std']:.4f}\n")
            f.write(f"  最小值: {metrics['paired_similarity_min']:.4f}\n")
            f.write(f"  最大值: {metrics['paired_similarity_max']:.4f}\n")
            f.write("\n")
            
            # 显示每个配对对的相似度
            f.write("各配对对的相似度:\n")
            for i, (audio_path, video_path) in enumerate(pairs):
                if i < len(metrics['paired_similarities']):
                    sim = metrics['paired_similarities'][i]
                    f.write(f"  {i+1}. {Path(audio_path).stem} <-> {Path(video_path).stem}: {sim:.4f}\n")
        else:
            # 检索模式：显示相似度矩阵统计
            f.write("相似度统计:\n")
            f.write(f"  平均相似度: {metrics['similarity_matrix_mean']:.4f}\n")
            f.write(f"  标准差: {metrics['similarity_matrix_std']:.4f}\n")
            f.write("\n")
        
        if 'mean_rank' in metrics:
            f.write("检索性能:\n")
            f.write(f"  平均排名: {metrics['mean_rank']:.2f}\n")
            f.write(f"  中位数排名: {metrics['median_rank']:.2f}\n")
            f.write(f"  Recall@1: {metrics['recall_at_1']:.4f}\n")
            f.write(f"  Recall@5: {metrics['recall_at_5']:.4f}\n")
            f.write(f"  Recall@10: {metrics['recall_at_10']:.4f}\n")
        
        # 显示相似度数据预览
        f.write("\n" + "=" * 80 + "\n")
        if pairs and similarity_data.ndim == 1:
            # 配对模式：显示配对相似度数组
            f.write("配对相似度 (前20个):\n")
            f.write("=" * 80 + "\n")
            preview = similarity_data[:20]
            np.savetxt(f, preview.reshape(-1, 1), fmt='%.4f', delimiter='\t')
        else:
            # 检索模式：显示相似度矩阵
            f.write("相似度矩阵 (前10x10):\n")
            f.write("=" * 80 + "\n")
            if similarity_data.ndim == 2:
                matrix_preview = similarity_data[:10, :10]
                np.savetxt(f, matrix_preview, fmt='%.4f', delimiter='\t')
    
    print(f"\n已保存结果到: {output_dir}")
    print(f"  - 嵌入向量: audio_embeddings.npy, video_embeddings.npy")
    print(f"  - 相似度矩阵: similarity_matrix.npy")
    print(f"  - 评估指标: metrics.json")
    print(f"  - 一致性报告: consistency_report.txt")


def batch_inference_av_consistency(
    audio_paths: List[str],
    video_paths: List[str],
    device: str = "cuda:0",
    batch_size: int = 4,
    output_dir: Optional[str] = None,
    retrieval_mode: bool = False,
):
    """
    音频-视频一致性批量推理
    
    Args:
        audio_paths: 音频文件路径列表
        video_paths: 视频文件路径列表
        device: 计算设备
        batch_size: 批处理大小（视频处理需要较小批次）
        output_dir: 输出目录
        retrieval_mode: 是否为检索模式（False=配对模式，True=检索模式）
    """
    # 检查设备
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        device = "cpu"
    
    # 加载模型
    print(f"正在加载 ImageBind 模型 (设备: {device})...")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    print("模型加载完成!")
    
    # 匹配音频-视频文件
    pairs = None
    if not retrieval_mode:
        pairs = match_audio_video_files(audio_paths, video_paths)
        if not pairs:
            print("警告: 未找到匹配的音频-视频对，切换到检索模式")
            retrieval_mode = True
        else:
            # 使用配对的文件
            audio_paths = [p[0] for p in pairs]
            video_paths = [p[1] for p in pairs]
    
    # 处理音频
    print(f"\n处理 {len(audio_paths)} 个音频文件...")
    valid_audio_paths = [p for p in audio_paths if os.path.exists(p)]
    if len(valid_audio_paths) != len(audio_paths):
        print(f"警告: {len(audio_paths) - len(valid_audio_paths)} 个音频文件不存在")
    
    audio_embeddings_all = []
    audio_names = []
    for i in tqdm(range(0, len(valid_audio_paths), batch_size), desc="音频批处理"):
        batch_paths = valid_audio_paths[i:i+batch_size]
        try:
            batch_inputs = data.load_and_transform_audio_data(batch_paths, device)
            
            with torch.no_grad():
                batch_outputs = model({ModalityType.AUDIO: batch_inputs})
                audio_embeddings_all.append(batch_outputs[ModalityType.AUDIO])
            
            audio_names.extend([Path(p).stem for p in batch_paths])
        except Exception as e:
            print(f"处理音频批次 {i//batch_size + 1} 时出错: {e}")
            continue
    
    if not audio_embeddings_all:
        print("错误: 没有成功处理任何音频文件")
        return
    
    audio_embeddings = torch.cat(audio_embeddings_all, dim=0)
    print(f"音频嵌入 shape: {audio_embeddings.shape}")
    
    # 处理视频
    print(f"\n处理 {len(video_paths)} 个视频文件...")
    valid_video_paths = [p for p in video_paths if os.path.exists(p)]
    if len(valid_video_paths) != len(video_paths):
        print(f"警告: {len(video_paths) - len(valid_video_paths)} 个视频文件不存在")
    
    video_embeddings_all = []
    video_names = []
    for i in tqdm(range(0, len(valid_video_paths), batch_size), desc="视频批处理"):
        batch_paths = valid_video_paths[i:i+batch_size]
        try:
            batch_inputs = data.load_and_transform_video_data(batch_paths, device)
            
            with torch.no_grad():
                batch_outputs = model({ModalityType.VISION: batch_inputs})
                video_embeddings_all.append(batch_outputs[ModalityType.VISION])
            
            video_names.extend([Path(p).stem for p in batch_paths])
        except Exception as e:
            print(f"处理视频批次 {i//batch_size + 1} 时出错: {e}")
            continue
    
    if not video_embeddings_all:
        print("错误: 没有成功处理任何视频文件")
        return
    
    video_embeddings = torch.cat(video_embeddings_all, dim=0)
    print(f"视频嵌入 shape: {video_embeddings.shape}")
    
    # 计算一致性指标
    print("\n计算音频-视频一致性指标...")
    metrics = compute_consistency_metrics(
        audio_embeddings, video_embeddings, pairs if not retrieval_mode else None
    )
    
    # 打印关键指标
    print("\n" + "=" * 80)
    print("一致性评估结果:")
    print("=" * 80)
    
    if 'paired_similarity_mean' in metrics:
        # 配对模式：只显示配对相似度
        print(f"配对一致性:")
        print(f"  平均配对相似度: {metrics['paired_similarity_mean']:.4f} ± {metrics['paired_similarity_std']:.4f}")
        print(f"  范围: [{metrics['paired_similarity_min']:.4f}, {metrics['paired_similarity_max']:.4f}]")
        print(f"  配对数量: {len(metrics['paired_similarities'])}")
    else:
        # 检索模式：显示完整矩阵统计
        print(f"平均相似度: {metrics['similarity_matrix_mean']:.4f} ± {metrics['similarity_matrix_std']:.4f}")
    
    if 'mean_rank' in metrics:
        print(f"\n检索性能:")
        print(f"  平均排名: {metrics['mean_rank']:.2f}")
        print(f"  中位数排名: {metrics['median_rank']:.2f}")
        print(f"  Recall@1: {metrics['recall_at_1']:.4f}")
        print(f"  Recall@5: {metrics['recall_at_5']:.4f}")
        print(f"  Recall@10: {metrics['recall_at_10']:.4f}")
    
    # 保存结果
    if output_dir:
        save_results(
            audio_embeddings, video_embeddings,
            audio_names, video_names,
            metrics, output_dir, pairs if not retrieval_mode else None
        )
    
    return {
        'audio_embeddings': audio_embeddings,
        'video_embeddings': video_embeddings,
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description="ImageBind 音频-视频一致性批量推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 配对模式（文件一一对应）
  python batch_inference.py --audio_dir ./audios --video_dir ./videos --output_dir ./outputs
  
  # 检索模式（计算所有配对）
  python batch_inference.py --audio_dir ./audios --video_dir ./videos --output_dir ./outputs --retrieval_mode
  
  # 从文件列表读取
  python batch_inference.py --audio_file audio_list.txt --video_file video_list.txt --output_dir ./outputs
        """
    )
    
    # 输入选项
    parser.add_argument("--audio_dir", type=str, help="音频目录路径")
    parser.add_argument("--audio_file", type=str, help="音频路径列表文件，每行一个路径")
    parser.add_argument("--audio_paths", type=str, nargs="+", help="音频路径列表")
    parser.add_argument("--video_dir", type=str, help="视频目录路径")
    parser.add_argument("--video_file", type=str, help="视频路径列表文件，每行一个路径")
    parser.add_argument("--video_paths", type=str, nargs="+", help="视频路径列表")
    
    # 处理选项
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备 (default: cuda:0)")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小 (default: 4，视频处理建议较小)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录 (default: ./outputs)")
    parser.add_argument("--retrieval_mode", action="store_true", help="检索模式：计算所有音频-视频对的相似度")
    
    args = parser.parse_args()
    
    # 解析音频输入
    audio_paths = None
    if args.audio_paths:
        audio_paths = args.audio_paths
    elif args.audio_file:
        audio_paths = load_path_list(args.audio_file)
    elif args.audio_dir:
        audio_paths = get_files_from_dir(args.audio_dir, ['.wav', '.mp3', '.flac', '.m4a', '.ogg'])
    
    # 解析视频输入
    video_paths = None
    if args.video_paths:
        video_paths = args.video_paths
    elif args.video_file:
        video_paths = load_path_list(args.video_file)
    elif args.video_dir:
        video_paths = get_files_from_dir(args.video_dir, ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'])
    
    # 检查输入
    if not audio_paths:
        parser.print_help()
        print("\n错误: 请提供音频输入!")
        return
    
    if not video_paths:
        parser.print_help()
        print("\n错误: 请提供视频输入!")
        return
    
    # 执行批量推理
    results = batch_inference_av_consistency(
        audio_paths=audio_paths,
        video_paths=video_paths,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        retrieval_mode=args.retrieval_mode,
    )
    
    if results:
        print("\n音频-视频一致性评估完成!")


if __name__ == "__main__":
    main()
