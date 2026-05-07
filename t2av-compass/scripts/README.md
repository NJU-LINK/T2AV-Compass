# T2AV-Compass Evaluation Scripts

This directory contains evaluation scripts for all objective metrics in T2AV-Compass.

## 📁 Script Overview

### Complete Evaluation

| Script | Description | Usage |
|--------|-------------|-------|
| `eval_all_metrics.sh` | Run all objective metrics in sequence | `bash eval_all_metrics.sh <input_dir> <prompts_json> <output_dir>` |

### Video Quality Metrics

| Script | Metric | Description | Usage |
|--------|--------|-------------|-------|
| `eval_video_aesthetic.sh` | VA | Video Aesthetic - High-level perceptual attributes via LAION-Aesthetic V2.5 | `bash eval_video_aesthetic.sh <input_dir> <output_dir> [num_frames]` |
| `eval_video_technical.sh` | VT | Video Technical - Low-level visual integrity via DOVER++ | `bash eval_video_technical.sh <input_dir> <output_dir>` |

### Audio Quality Metrics

| Script | Metric | Description | Usage |
|--------|--------|-------------|-------|
| `eval_audio_aesthetic.sh` | AA | Audio Aesthetic - Mean of PQ (Perceptual Quality) and CU (Content Usefulness) | `bash eval_audio_aesthetic.sh <input_dir> <output_dir>` |
| `eval_speech_quality.sh` | SQ | Speech Quality - Speech quality assessment via NISQA | `bash eval_speech_quality.sh <input_dir> <output_dir>` |

### Cross-Modal Alignment Metrics

| Script | Metric | Description | Usage |
|--------|--------|-------------|-------|
| `eval_text_video_alignment.sh` | T-V | Text-Video Alignment - Semantic alignment via ImageBind | `bash eval_text_video_alignment.sh <input_dir> <prompts_json> <output_dir>` |
| `eval_text_audio_alignment.sh` | T-A | Text-Audio Alignment - Semantic alignment via ImageBind | `bash eval_text_audio_alignment.sh <input_dir> <prompts_json> <output_dir>` |
| `eval_audio_video_alignment.sh` | A-V | Audio-Video Alignment - Semantic alignment via ImageBind (text-independent) | `bash eval_audio_video_alignment.sh <input_dir> <output_dir>` |
| `eval_av_sync.sh` | DeSync | Audio-Video Synchronization - Temporal sync via Synchformer | `bash eval_av_sync.sh <input_dir> <output_dir>` |
| `eval_lipsync.sh` | LS | Lip-Sync Quality - Lip-sync evaluation via LatentSync (for talking-face videos) | `bash eval_lipsync.sh <input_dir> <output_dir>` |

### Helper Scripts

| Script | Description |
|--------|-------------|
| `run_audiobox_batch.py` | Batch AudioBox aesthetics scoring (called by eval_audio_aesthetic.sh) |

## 🚀 Quick Start

### Run All Metrics

```bash
# Run complete evaluation pipeline
bash eval_all_metrics.sh input Data/prompts.json Output
```

### Run Individual Metrics

```bash
# Video quality
bash eval_video_aesthetic.sh input Output 10
bash eval_video_technical.sh input Output

# Audio quality
bash eval_audio_aesthetic.sh input Output
bash eval_speech_quality.sh input Output

# Cross-modal alignment
bash eval_text_video_alignment.sh input Data/prompts.json Output
bash eval_text_audio_alignment.sh input Data/prompts.json Output
bash eval_audio_video_alignment.sh input Output
bash eval_av_sync.sh input Output
bash eval_lipsync.sh input Output  # For talking-face videos
```

## 📋 Parameters

### Common Parameters

- `<input_dir>`: Directory containing generated video files (e.g., `input/`)
- `<output_dir>`: Directory to save evaluation results (e.g., `Output/`)
- `<prompts_json>`: JSON file containing prompts (e.g., `Data/prompts.json`)

### Optional Parameters

- `[num_frames]`: Number of frames to sample for video aesthetic evaluation (default: 10)

## 🔧 Environment Setup

Each script automatically:

1. **Checks for conda installation**
2. **Creates the required conda environments** if they do not exist
3. **Installs dependencies** on first run
4. **Activates environment** before evaluation
5. **Deactivates environment** after completion

### Conda Environments

| Environment | Used By | Python Version |
|-------------|---------|----------------|
| `t2av-core` | Video Aesthetic (VA), Audio Aesthetic (AA), Speech Quality (SQ), T-V, T-A, A-V alignment | 3.10 |
| `t2av-dover` | Video Technical (VT) | 3.10 |
| `t2av-synchformer` | AV Sync (DeSync) | 3.10 |
| `t2av-latentsync` | Lip-Sync (LS) | 3.10 |

Set `T2AV_CORE_ENV` before setup if you need a different shared environment name.

### Manual Environment Management

```bash
# List all T2AV environments
conda env list | grep t2av

# Activate specific environment
conda activate t2av-core

# Remove environment (if needed)
conda remove -n t2av-core --all
```

## 📊 Output Format

All scripts generate JSON output with consistent structure:

```json
{
  "metric": "metric_name",
  "summary": {
    "key_metric": 0.85,
    "...": "..."
  },
  "results": [
    {
      "file": "input/video_001.mp4",
      "filename": "video_001.mp4",
      "score": 0.87,
      "...": "..."
    }
  ]
}
```

### Output Files

After running `eval_all_metrics.sh`, the output directory contains:

```
Output/
├── video_aesthetic.json          # VA scores
├── video_technical.json          # VT scores
├── audio_aesthetic.json          # AA scores (PQ, CU)
├── speech_quality.json           # SQ scores
├── text_video_alignment.json     # T-V alignment scores
├── text_audio_alignment.json     # T-A alignment scores
├── audio_video_alignment.json    # A-V alignment scores
├── av_sync.json                  # DeSync scores
├── lipsync.json                  # LS (Lip-sync) scores
├── evaluation_summary.json       # Summary of all metrics
└── audio_wav/                    # Extracted audio files
    ├── video_001.wav
    ├── video_002.wav
    └── ...
```

## 🐛 Troubleshooting

### Common Issues

**1. "conda not found"**
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**2. "FFmpeg not found"**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

**3. CUDA out of memory**
- Reduce batch size in evaluation scripts
- Use smaller models if available
- Process videos in smaller batches

**4. Environment conflicts**
```bash
# Remove and recreate environment
conda remove -n t2av-core --all
# Run script again to recreate
```


## 📝 Notes

- **Audio Extraction**: Audio is automatically extracted from videos when needed. Extracted audio files are saved to `<output_dir>/audio_wav/` and reused across metrics.

- **Environment Isolation**: Compatible objective metrics share `t2av-core`. DOVER, Synchformer, and LatentSync stay isolated because their pinned dependencies are more likely to conflict.

- **First Run**: The first run of each script will take longer due to environment creation and dependency installation. Subsequent runs will be faster.

- **Parallel Execution**: You can run multiple scripts in parallel if you have sufficient GPU memory. However, be cautious of memory usage.

- **Result Files**: JSON result files can be easily parsed and aggregated for analysis. The `evaluation_summary.json` provides a quick overview of all metrics.

## 🔗 Related Documentation

- [Main README](../../README.md): Project overview and complete documentation
- [Data Schema](../../README.md#-data-schema): Understanding the prompt format
- [Subjective Evaluation](../Subjective/README.md): MLLM-as-a-Judge evaluation

## 📧 Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/NJU-LINK/T2AV-Compass/issues)
- Contact: zhecao@smail.nju.edu.cn
