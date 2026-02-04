"""
批量测试文件夹下所有视频的脚本
自动扫描文件夹，批量处理视频，计算平均结果并保存到JSON
"""
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import torch
import torchvision
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


def find_video_files(folder_path: str, recursive: bool = False) -> List[str]:
    """
    查找文件夹下的所有视频文件
    
    Args:
        folder_path: 文件夹路径
        recursive: 是否递归搜索子文件夹
        
    Returns:
        视频文件路径列表
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    # 支持的视频格式
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.3gp', '.ts', '.mts'}
    
    video_files = []
    if recursive:
        # 递归搜索
        for ext in video_extensions:
            video_files.extend(folder.rglob(f'*{ext}'))
            video_files.extend(folder.rglob(f'*{ext.upper()}'))
    else:
        # 只搜索当前文件夹
        for ext in video_extensions:
            video_files.extend(folder.glob(f'*{ext}'))
            video_files.extend(folder.glob(f'*{ext.upper()}'))
    
    # 转换为字符串并排序
    video_files = sorted([str(f) for f in video_files])
    return video_files


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
        
        # 提取结果 - 确保索引是整数
        pred_class = int(topk_preds[0, 0].item())
        pred_offset_sec = float(grid[pred_class].item())
        pred_prob = float(off_probs[0, pred_class].item())
        
        # Top-k预测
        topk_results = []
        for i in range(k):
            class_idx = int(topk_preds[0, i].item())
            topk_results.append({
                'class': class_idx,
                'offset_sec': float(grid[class_idx].item()),
                'probability': float(off_probs[0, class_idx].item()),
                'logit': float(topk_logits[0, i].item())
            })
        
        # 获取真实标签（如果有）
        gt_offset_sec = None
        gt_class = None
        if 'offset_label' in item['targets']:
            gt_class = int(item['targets']['offset_label'].item())
            gt_offset_sec = float(grid[gt_class].item())
        
        return {
            'video_path': str(vid_path),
            'predicted_offset_sec': float(pred_offset_sec),
            'predicted_class': int(pred_class),
            'predicted_probability': float(pred_prob),
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


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算统计信息
    
    Args:
        results: 所有视频的预测结果列表
        
    Returns:
        统计信息字典
    """
    from datetime import datetime
    
    # 过滤出成功的结果
    successful_results = [r for r in results if r['success']]
    failed_count = len(results) - len(successful_results)
    
    # 基础统计
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_count': len(results),
        'successful_count': len(successful_results),
        'failed_count': failed_count
    }
    
    if len(successful_results) == 0:
        return stats
    
    # 提取偏移和概率
    offsets = [r['predicted_offset_sec'] for r in successful_results]
    probabilities = [r['predicted_probability'] for r in successful_results]
    
    # 偏移量统计
    stats['predicted_offset_sec'] = {
        'mean': float(np.mean(offsets)),
        'std': float(np.std(offsets)),
        'min': float(np.min(offsets)),
        'max': float(np.max(offsets)),
        'median': float(np.median(offsets))
    }
    
    # 概率统计
    stats['predicted_probability'] = {
        'mean': float(np.mean(probabilities)),
        'std': float(np.std(probabilities)),
        'min': float(np.min(probabilities)),
        'max': float(np.max(probabilities)),
        'median': float(np.median(probabilities))
    }
    
    # 偏移分布（按类别统计）
    offset_distribution = {}
    for r in successful_results:
        offset = r['predicted_offset_sec']
        # 四舍五入到0.1秒
        offset_rounded = round(offset, 1)
        if offset_rounded not in offset_distribution:
            offset_distribution[offset_rounded] = 0
        offset_distribution[offset_rounded] += 1
    
    stats['offset_distribution'] = offset_distribution
    
    # 置信度分布
    high_confidence = sum(1 for p in probabilities if p >= 0.8)
    medium_confidence = sum(1 for p in probabilities if 0.5 <= p < 0.8)
    low_confidence = sum(1 for p in probabilities if p < 0.5)
    
    stats['confidence_distribution'] = {
        'high_>=0.8': high_confidence,
        'medium_0.5-0.8': medium_confidence,
        'low_<0.5': low_confidence
    }
    
    # 同步质量评级（基于偏移绝对值）
    abs_offsets = [abs(offset) for offset in offsets]
    excellent = sum(1 for o in abs_offsets if o <= 0.1)  # 偏移 <= 0.1秒
    good = sum(1 for o in abs_offsets if 0.1 < o <= 0.2)  # 0.1-0.2秒
    acceptable = sum(1 for o in abs_offsets if 0.2 < o <= 0.5)  # 0.2-0.5秒
    poor = sum(1 for o in abs_offsets if o > 0.5)  # > 0.5秒
    
    stats['sync_quality_distribution'] = {
        'excellent_<=0.1s': excellent,
        'good_0.1-0.2s': good,
        'acceptable_0.2-0.5s': acceptable,
        'poor_>0.5s': poor
    }
    
    return stats


def batch_test_folder(folder_path: str, exp_name: str, device: str = 'cuda:0',
                      offset_sec: float = 0.0, v_start_i_sec: float = 0.0,
                      recursive: bool = False) -> Dict[str, Any]:
    """
    批量测试文件夹下的所有视频
    
    Args:
        folder_path: 视频文件夹路径
        exp_name: 实验名称（模型ID）
        device: 设备
        offset_sec: 偏移秒数
        v_start_i_sec: 视频起始秒数
        recursive: 是否递归搜索子文件夹
        
    Returns:
        包含所有结果和统计信息的字典
    """
    vfps = 25
    afps = 16000
    in_size = 256
    
    # 查找所有视频文件
    print(f"正在扫描文件夹: {folder_path}")
    video_files = find_video_files(folder_path, recursive=recursive)
    print(f"找到 {len(video_files)} 个视频文件")
    
    if len(video_files) == 0:
        print("警告: 未找到任何视频文件")
        return {
            'folder_path': folder_path,
            'total_videos': 0,
            'results': [],
            'statistics': {
                'total_videos': 0,
                'successful_videos': 0,
                'failed_videos': 0
            }
        }
    
    # 加载配置和模型
    cfg_path = f'./logs/sync_models/{exp_name}/cfg-{exp_name}.yaml'
    ckpt_path = f'./logs/sync_models/{exp_name}/{exp_name}.pt'
    
    print(f"加载模型: {exp_name}")
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
    print(f"开始处理 {len(video_files)} 个视频...")
    results = []
    for vid_path in tqdm(video_files, desc='Processing videos'):
        result = process_single_video(
            vid_path, model, transforms, grid, device, cfg,
            offset_sec=offset_sec, v_start_i_sec=v_start_i_sec
        )
        results.append(result)
    
    # 计算统计信息
    print("计算统计信息...")
    statistics = calculate_statistics(results)
    
    # 组装最终结果
    output = {
        'config': {
            'folder_path': folder_path,
            'exp_name': exp_name,
            'device': str(device),
            'recursive': recursive,
            'model_config': cfg_path,
            'model_checkpoint': ckpt_path
        },
        'statistics': statistics,
        'results': results
    }
    
    return output


def main():
    parser = argparse.ArgumentParser(description='批量测试文件夹下所有视频并计算平均结果')
    parser.add_argument('--folder', required=True, help='视频文件夹路径')
    parser.add_argument('--exp_name', required=True, help='实验名称，格式: xx-xx-xxTxx-xx-xx')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--offset_sec', type=float, default=0.0, help='音频偏移秒数')
    parser.add_argument('--v_start_i_sec', type=float, default=0.0, help='视频起始秒数')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--recursive', action='store_true', help='递归搜索子文件夹')
    
    args = parser.parse_args()
    
    # 批量测试
    output = batch_test_folder(
        folder_path=args.folder,
        exp_name=args.exp_name,
        device=args.device,
        offset_sec=args.offset_sec,
        v_start_i_sec=args.v_start_i_sec,
        recursive=args.recursive
    )
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")
    print(f"\n{'='*80}")
    print("统计摘要")
    print(f"{'='*80}")
    print(f"总数: {output['statistics']['total_count']}")
    print(f"成功: {output['statistics']['successful_count']}")
    print(f"失败: {output['statistics']['failed_count']}")
    
    if output['statistics']['successful_count'] > 0 and 'predicted_offset_sec' in output['statistics']:
        offset_stats = output['statistics']['predicted_offset_sec']
        prob_stats = output['statistics']['predicted_probability']
        
        print(f"\n预测偏移量 (秒):")
        print(f"  平均值: {offset_stats['mean']:.6f}")
        print(f"  标准差: {offset_stats['std']:.6f}")
        print(f"  最小值: {offset_stats['min']:.6f}")
        print(f"  最大值: {offset_stats['max']:.6f}")
        print(f"  中位数: {offset_stats['median']:.6f}")
        
        print(f"\n预测置信度:")
        print(f"  平均值: {prob_stats['mean']:.6f}")
        print(f"  标准差: {prob_stats['std']:.6f}")
        print(f"  最小值: {prob_stats['min']:.6f}")
        print(f"  最大值: {prob_stats['max']:.6f}")
        print(f"  中位数: {prob_stats['median']:.6f}")
        
        if 'sync_quality_distribution' in output['statistics']:
            sync_dist = output['statistics']['sync_quality_distribution']
            print(f"\n同步质量分布:")
            print(f"  优秀 (≤0.1s): {sync_dist['excellent_<=0.1s']}")
            print(f"  良好 (0.1-0.2s): {sync_dist['good_0.1-0.2s']}")
            print(f"  可接受 (0.2-0.5s): {sync_dist['acceptable_0.2-0.5s']}")
            print(f"  较差 (>0.5s): {sync_dist['poor_>0.5s']}")
        
        print(f"\n说明: 偏移量越接近 0 表示音视频同步越好")


if __name__ == '__main__':
    main()

