"""
批量推理脚本 - 批量处理多个视频文件并输出结果
"""
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import torch
import torchvision
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset.dataset_utils import get_video_and_audio
from dataset.transforms import make_class_grid, quantize_offset
from utils.utils import check_if_file_exists_else_download, which_ffmpeg
from scripts.train_utils import get_model, get_transforms, prepare_inputs


def reencode_video(path, vfps=25, afps=16000, in_size=256):
    """重新编码视频到指定格式"""
    assert which_ffmpeg() != '', 'Is ffmpeg installed?'
    new_path = Path.cwd() / 'vis' / f'{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.mp4'
    new_path.parent.mkdir(exist_ok=True)
    new_path = str(new_path)
    cmd = f'{which_ffmpeg()}'
    cmd += ' -hide_banner -loglevel panic'
    cmd += f' -y -i {path}'
    cmd += f" -vf fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -ar {afps}"
    cmd += f' {new_path}'
    import subprocess
    subprocess.call(cmd.split())
    return new_path


def patch_config(cfg):
    """修复配置"""
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    cfg.model.params.transformer.target = cfg.model.params.transformer.target\
                                             .replace('.modules.feature_selector.', '.sync_model.')
    return cfg


def process_single_video(vid_path: str, model, transforms, grid, device, cfg, 
                         offset_sec: float = 0.0, v_start_i_sec: float = 0.0) -> Dict[str, Any]:
    """
    处理单个视频文件
    
    Args:
        vid_path: 视频路径
        model: 模型
        transforms: 数据变换
        grid: 偏移类别网格
        device: 设备
        cfg: 配置
        offset_sec: 偏移秒数
        v_start_i_sec: 视频起始秒数
        
    Returns:
        包含预测结果的字典
    """
    vfps = 25
    afps = 16000
    in_size = 256
    
    try:
        # 检查视频格式
        v, _, info = torchvision.io.read_video(vid_path, pts_unit='sec')
        _, H, W, _ = v.shape
        if info['video_fps'] != vfps or info['audio_fps'] != afps or min(H, W) != in_size:
            vid_path = reencode_video(vid_path, vfps, afps, in_size)
        
        # 加载视频和音频
        rgb, audio, meta = get_video_and_audio(vid_path, get_meta=True)
        
        # 创建item
        item = dict(
            video=rgb, 
            audio=audio, 
            meta=meta, 
            path=vid_path, 
            split='test',
            targets={
                'v_start_i_sec': v_start_i_sec, 
                'offset_sec': offset_sec,
            },
        )
        
        # 应用变换
        item = transforms['test'](item)
        
        # 准备批量输入
        batch = torch.utils.data.default_collate([item])
        aud, vid, targets = prepare_inputs(batch, device)
        
        # 前向推理
        with torch.set_grad_enabled(False):
            with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                _, logits = model(vid, aud)
        
        # 解码预测结果
        off_probs = torch.softmax(logits, dim=-1)
        k = min(off_probs.shape[-1], 5)
        topk_logits, topk_preds = torch.topk(logits, k)
        
        # 提取结果
        pred_class = topk_preds[0, 0].item()
        pred_offset_sec = grid[pred_class].item()
        pred_prob = off_probs[0, pred_class].item()
        
        # Top-k预测
        topk_results = []
        for i in range(k):
            class_idx = topk_preds[0, i].item()
            topk_results.append({
                'class': class_idx,
                'offset_sec': grid[class_idx].item(),
                'probability': off_probs[0, class_idx].item(),
                'logit': topk_logits[0, i].item()
            })
        
        # 获取真实标签（如果有）
        gt_offset_sec = None
        gt_class = None
        if 'offset_label' in item['targets']:
            gt_class = item['targets']['offset_label'].item()
            gt_offset_sec = grid[gt_class].item()
        
        return {
            'video_path': str(vid_path),
            'predicted_offset_sec': pred_offset_sec,
            'predicted_class': pred_class,
            'predicted_probability': pred_prob,
            'ground_truth_offset_sec': gt_offset_sec,
            'ground_truth_class': gt_class,
            'topk_predictions': topk_results,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'video_path': str(vid_path),
            'predicted_offset_sec': None,
            'predicted_class': None,
            'predicted_probability': None,
            'ground_truth_offset_sec': None,
            'ground_truth_class': None,
            'topk_predictions': [],
            'success': False,
            'error': str(e)
        }


def batch_inference(video_paths: List[str], exp_name: str, device: str = 'cuda:0',
                   batch_size: int = 1, offset_sec: float = 0.0, 
                   v_start_i_sec: float = 0.0, output_format: str = 'json') -> List[Dict[str, Any]]:
    """
    批量推理
    
    Args:
        video_paths: 视频路径列表
        exp_name: 实验名称
        device: 设备
        batch_size: 批量大小（当前实现为1，可扩展）
        offset_sec: 偏移秒数
        v_start_i_sec: 视频起始秒数
        output_format: 输出格式 ('json' 或 'csv')
        
    Returns:
        预测结果列表
    """
    vfps = 25
    afps = 16000
    in_size = 256
    
    # 加载配置和模型
    cfg_path = f'./logs/sync_models/{exp_name}/cfg-{exp_name}.yaml'
    ckpt_path = f'./logs/sync_models/{exp_name}/{exp_name}.pt'
    
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)
    
    cfg = OmegaConf.load(cfg_path)
    cfg = patch_config(cfg)
    
    device = torch.device(device)
    _, model = get_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # 获取变换和网格
    transforms = get_transforms(cfg, ['test'])
    max_off_sec = cfg.data.max_off_sec
    num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
    grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)
    
    # 批量处理
    results = []
    for vid_path in tqdm(video_paths, desc='Processing videos'):
        result = process_single_video(
            vid_path, model, transforms, grid, device, cfg,
            offset_sec=offset_sec, v_start_i_sec=v_start_i_sec
        )
        results.append(result)
    
    return results


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算统计信息
    
    Args:
        results: 结果列表
        
    Returns:
        包含统计信息的字典
    """
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
    
    # 提取预测偏移量
    predicted_offsets = [r['predicted_offset_sec'] for r in successful_results 
                        if r['predicted_offset_sec'] is not None]
    if predicted_offsets:
        stats['predicted_offset_sec'] = {
            'mean': float(np.mean(predicted_offsets)),
            'std': float(np.std(predicted_offsets)),
            'min': float(np.min(predicted_offsets)),
            'max': float(np.max(predicted_offsets)),
            'median': float(np.median(predicted_offsets))
        }
    
    # 提取预测概率
    predicted_probs = [r['predicted_probability'] for r in successful_results 
                      if r['predicted_probability'] is not None]
    if predicted_probs:
        stats['predicted_probability'] = {
            'mean': float(np.mean(predicted_probs)),
            'std': float(np.std(predicted_probs)),
            'min': float(np.min(predicted_probs)),
            'max': float(np.max(predicted_probs)),
            'median': float(np.median(predicted_probs))
        }
    
    # 提取真实偏移量（如果存在）
    gt_offsets = [r['ground_truth_offset_sec'] for r in successful_results 
                 if r['ground_truth_offset_sec'] is not None]
    if gt_offsets:
        stats['ground_truth_offset_sec'] = {
            'mean': float(np.mean(gt_offsets)),
            'std': float(np.std(gt_offsets)),
            'min': float(np.min(gt_offsets)),
            'max': float(np.max(gt_offsets)),
            'median': float(np.median(gt_offsets))
        }
        
        # 计算误差（如果同时有预测值和真实值）
        if len(predicted_offsets) == len(gt_offsets):
            errors = [abs(pred - gt) for pred, gt in zip(predicted_offsets, gt_offsets)]
            stats['absolute_error'] = {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'min': float(np.min(errors)),
                'max': float(np.max(errors)),
                'median': float(np.median(errors)),
                'rmse': float(np.sqrt(np.mean([e**2 for e in errors])))
            }
    
    return stats


def save_results(results: List[Dict[str, Any]], output_path: str, format: str = 'json', 
                 exp_name: str = None, device: str = None):
    """保存结果到文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # 计算统计信息
        statistics = calculate_statistics(results)
        
        # JSON格式输出，包含配置、统计信息和结果
        output_data = {
            'config': {
                'exp_name': exp_name,
                'device': device,
                'total_videos': len(results)
            },
            'statistics': statistics,
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    elif format == 'csv':
        # CSV格式输出
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if len(results) == 0:
                return
            
            writer = csv.DictWriter(f, fieldnames=[
                'video_path', 'predicted_offset_sec', 'predicted_class', 
                'predicted_probability', 'ground_truth_offset_sec', 
                'ground_truth_class', 'success', 'error'
            ])
            writer.writeheader()
            
            for result in results:
                row = {
                    'video_path': result['video_path'],
                    'predicted_offset_sec': result['predicted_offset_sec'],
                    'predicted_class': result['predicted_class'],
                    'predicted_probability': result['predicted_probability'],
                    'ground_truth_offset_sec': result['ground_truth_offset_sec'],
                    'ground_truth_class': result['ground_truth_class'],
                    'success': result['success'],
                    'error': result['error']
                }
                writer.writerow(row)
    
    print(f'Results saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='批量推理音视频同步模型')
    parser.add_argument('--exp_name', required=True, help='实验名称，格式: xx-xx-xxTxx-xx-xx')
    parser.add_argument('--video_list', required=True, help='视频文件路径列表（每行一个路径）或逗号分隔的路径')
    parser.add_argument('--output', required=True, help='输出文件路径')
    parser.add_argument('--output_format', choices=['json', 'csv'], default='json', help='输出格式')
    parser.add_argument('--offset_sec', type=float, default=0.0, help='音频偏移秒数')
    parser.add_argument('--v_start_i_sec', type=float, default=0.0, help='视频起始秒数')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小（当前实现为1）')
    
    args = parser.parse_args()
    
    # 读取视频路径列表
    video_list_path = Path(args.video_list)
    
    # 视频文件扩展名列表
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.3gp', '.ts', '.mts'}
    
    if video_list_path.exists():
        # 检查是否是视频文件
        if video_list_path.suffix.lower() in video_extensions:
            # 如果是视频文件，直接作为单个视频路径处理
            video_paths = [str(video_list_path)]
        else:
            # 从文本文件读取，尝试多种编码方式
            video_paths = []
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    with open(video_list_path, 'r', encoding=encoding, errors='replace') as f:
                        video_paths = [line.strip() for line in f if line.strip()]
                    # 验证读取的内容是否看起来像文件路径（简单检查：至少有一行以/开头或包含常见视频扩展名）
                    if video_paths and (any(p.startswith('/') for p in video_paths) or 
                                       any(Path(p).suffix.lower() in video_extensions for p in video_paths)):
                        break
                    else:
                        video_paths = []
                except UnicodeDecodeError:
                    continue
            if not video_paths:
                # 如果所有编码都失败，可能是二进制文件，提示用户
                print(f'Warning: Cannot read {video_list_path} as text file. '
                      f'If it is a video file, use direct path. '
                      f'If it is a text file, check file encoding.')
    else:
        # 从命令行参数读取（逗号分隔）
        video_paths = [p.strip() for p in args.video_list.split(',')]
    
    # 验证视频文件存在
    valid_paths = []
    for path in video_paths:
        if Path(path).exists():
            valid_paths.append(path)
        else:
            print(f'Warning: Video file not found: {path}')
    
    if len(valid_paths) == 0:
        print('Error: No valid video files found')
        return
    
    print(f'Processing {len(valid_paths)} videos...')
    
    # 批量推理
    results = batch_inference(
        valid_paths,
        args.exp_name,
        device=args.device,
        batch_size=args.batch_size,
        offset_sec=args.offset_sec,
        v_start_i_sec=args.v_start_i_sec,
        output_format=args.output_format
    )
    
    # 保存结果
    save_results(results, args.output, format=args.output_format, 
                exp_name=args.exp_name, device=args.device)
    
    # 计算并打印统计信息
    statistics = calculate_statistics(results)
    successful = statistics['successful_count']
    failed = statistics['failed_count']
    
    print(f'\n{"="*80}')
    print('统计摘要')
    print(f'{"="*80}')
    print(f'总数: {statistics["total_count"]}')
    print(f'成功: {successful}')
    print(f'失败: {failed}')
    
    # 打印详细统计信息
    if successful > 0:
        if 'predicted_offset_sec' in statistics:
            offset_stats = statistics['predicted_offset_sec']
            print(f'\n预测偏移量 (秒):')
            print(f'  平均值: {offset_stats["mean"]:.6f}')
            print(f'  标准差: {offset_stats["std"]:.6f}')
            print(f'  最小值: {offset_stats["min"]:.6f}')
            print(f'  最大值: {offset_stats["max"]:.6f}')
            print(f'  中位数: {offset_stats["median"]:.6f}')
        
        if 'predicted_probability' in statistics:
            prob_stats = statistics['predicted_probability']
            print(f'\n预测置信度:')
            print(f'  平均值: {prob_stats["mean"]:.6f}')
            print(f'  标准差: {prob_stats["std"]:.6f}')
            print(f'  最小值: {prob_stats["min"]:.6f}')
            print(f'  最大值: {prob_stats["max"]:.6f}')
            print(f'  中位数: {prob_stats["median"]:.6f}')
        
        if 'absolute_error' in statistics:
            error_stats = statistics['absolute_error']
            print(f'\n绝对误差 (秒):')
            print(f'  平均值: {error_stats["mean"]:.6f}')
            print(f'  RMSE: {error_stats["rmse"]:.6f}')
            print(f'  标准差: {error_stats["std"]:.6f}')
            print(f'  最小值: {error_stats["min"]:.6f}')
            print(f'  最大值: {error_stats["max"]:.6f}')
        
        print(f'\n说明: 偏移量越接近 0 表示音视频同步越好')


if __name__ == '__main__':
    main()

