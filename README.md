# T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://nju-link.github.io/T2AV-Compass/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/NJU-LINK/T2AV-Compass)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2512.21094)
[![ICML 2026](https://img.shields.io/badge/ICML-2026_Accepted-green)](#citation)

> Objective evaluation on Linux is now wrapped by two top-level scripts: `setup_objective.sh` and `run_objective_batch.sh`.
> T2AV-Compass is accepted to ICML 2026.

## Objective Reproduction on Linux

This flow was validated on a Linux server with a single 4090 GPU. The public interface is repository-relative by default and can be overridden with environment variables when you need a different cache or Conda location.

Default relative layout:

- Repository root: `T2AV-Compass/`
- Input videos: `input/`
- Prompts file: `t2av-compass/Data/prompts.json`
- Output directory: `Output/`
- Cache root: `.cache/t2av-cache`
- Conda envs: `.cache/conda/envs`

The objective pipeline now uses a compact default environment layout:

- `t2av-core`: VA, AA, SQ, T-V, T-A, and A-V
- `t2av-dover`: VT
- `t2av-synchformer`: DeSync
- `t2av-latentsync`: LS

### 1. Clone the repository

Use submodules.

```bash
git clone --recurse-submodules https://github.com/NJU-LINK/T2AV-Compass.git
cd T2AV-Compass
```

If GitHub is slow in your region, you can optionally clone through a mirror instead. Keep the checked-out repository layout unchanged.

Optional environment overrides before setup:

```bash
export T2AV_CACHE_ROOT=/path/to/cache-root
export T2AV_CONDA_ROOT=/path/to/conda-root
export T2AV_CORE_ENV=t2av-core
# Default is https://hf-mirror.com for server-side reproducibility in mainland China.
# Override with the official endpoint when it is reachable in your region:
# export HF_ENDPOINT=https://huggingface.co
# optional when GitHub downloads need a mirror
export T2AV_GITHUB_MIRROR_PREFIX=https://your-mirror.example
```

### 2. Install all objective environments and checkpoints

```bash
bash setup_objective.sh
```

What this script does:

- installs system packages such as `ffmpeg`
- creates the required conda environments, using the shared `t2av-core` environment for compatible objective metrics
- downloads checkpoints for DOVER, AudioBox, ImageBind, Synchformer, and LatentSync
- pre-creates cache directories under `.cache/` by default

The script is safe to re-run.

### 3. Prepare videos and prompts

Put videos into `input/`.

Supported video naming conventions for prompt-linked metrics (`T-V`, `T-A`) include:

- `sample_0001.mp4`
- `sample_0002.mp4`
- `1.mp4`
- `0001.mp4`
- `video_0001.mp4`

The `index` field in `prompts.json` must match the video file index.

Example layout:

```text
T2AV-Compass/
├── input/
│   ├── sample_0001.mp4
│   └── sample_0002.mp4
├── Output/
├── setup_objective.sh
├── run_objective_batch.sh
└── t2av-compass/
    └── Data/
        └── prompts.json
```

Minimal `t2av-compass/Data/prompts.json` example:

```json
[
  {
    "index": 1,
    "prompt": "A person speaking directly to the camera.",
    "video_prompt": "A person speaking directly to the camera.",
    "audio_prompt": "clean speech from a person speaking indoors",
    "speech_prompt": []
  },
  {
    "index": 2,
    "prompt": "A person speaking directly to the camera.",
    "video_prompt": "A person speaking directly to the camera.",
    "audio_prompt": "clean speech from a person speaking indoors",
    "speech_prompt": []
  }
]
```

### 4. Run the full objective batch

Default paths:

```bash
bash run_objective_batch.sh
```

Custom paths:

```bash
bash run_objective_batch.sh /abs/path/to/input /abs/path/to/prompts.json /abs/path/to/output
```

The batch runs all objective metrics:

- `VT`: video technical quality
- `VA`: video aesthetic quality
- `AA`: audio aesthetic quality
- `SQ`: speech quality
- `T-V`: text-video alignment
- `T-A`: text-audio alignment
- `A-V`: audio-video alignment
- `DeSync`: audio-video synchronization error
- `LS`: lip-sync quality

### 5. Check outputs

After a successful run, `Output/` contains:

- `video_technical.json`
- `video_aesthetic.json`
- `audio_aesthetic.json`
- `speech_quality.json`
- `text_video_alignment.json`
- `text_audio_alignment.json`
- `audio_video_alignment.json`
- `av_sync.json`
- `lipsync.json`
- `evaluation_summary.json`

## Single-Metric Debug Commands

Run these from the repository root.

```bash
bash t2av-compass/scripts/eval_video_technical.sh input Output
bash t2av-compass/scripts/eval_video_aesthetic.sh input Output
bash t2av-compass/scripts/eval_audio_aesthetic.sh input Output
bash t2av-compass/scripts/eval_speech_quality.sh input Output
bash t2av-compass/scripts/eval_text_video_alignment.sh input t2av-compass/Data/prompts.json Output
bash t2av-compass/scripts/eval_text_audio_alignment.sh input t2av-compass/Data/prompts.json Output
bash t2av-compass/scripts/eval_audio_video_alignment.sh input Output
bash t2av-compass/scripts/eval_av_sync.sh input Output
bash t2av-compass/scripts/eval_lipsync.sh input Output
```

## Notes

- No manual Hugging Face login is required for the checkpoints used in the validated objective flow.
- The default Hugging Face endpoint for objective setup is `https://hf-mirror.com` to keep server-side reproduction stable in mainland China. Set `HF_ENDPOINT=https://huggingface.co` when the official endpoint is reachable and preferred.
- You can override cache and Conda locations with `T2AV_CACHE_ROOT`, `T2AV_CONDA_ROOT`, `T2AV_CONDA_EXE`, or rename the shared objective environment with `T2AV_CORE_ENV`.
- The first `DeSync` run downloads an additional large MotionFormer checkpoint.
- `LS` is intended for talking-face videos. For non-talking-face content, the score is not meaningful even if the script finishes.
- Re-running `setup_objective.sh` or `run_objective_batch.sh` is supported.

## Troubleshooting

- `视频文件未找到 (index: N)`: rename the file to match one of the supported index patterns, or fix the `index` field in `prompts.json`.
- `ffmpeg` not found: run `bash setup_objective.sh` again on a Debian/Ubuntu-like system with package manager access.
- mirror/network failures during checkpoint download: retry first; if needed, set `HF_ENDPOINT` or `T2AV_GITHUB_MIRROR_PREFIX` before running setup.
- `LS` fails on a batch with no visible speaking face: use talking-face videos for this metric.

## Citation

```bibtex
@inproceedings{cao2026t2avcompass,
  title         = {T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation},
  author        = {Cao, Zhe and Wang, Tao and Wang, Jiaming and Wang, Yanghai and Zhang, Yuanxing and Chen, Jialu and Deng, Miao and Wang, Jiahao and Guo, Yubin and Liao, Chenxi and Zhang, Yize and Zhang, Zhaoxiang and Liu, Jiaheng},
  booktitle     = {International Conference on Machine Learning (ICML)},
  year          = {2026},
  eprint        = {2512.21094},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2512.21094},
}
```

## Project Overview

T2AV-Compass is a unified benchmark for evaluating Text-to-Audio-Video generation across:

- unimodal quality
- cross-modal alignment and synchronization
- checklist-based subjective evaluation

The ICML 2026 version evaluates 15 representative T2AV systems: 7 closed-source end-to-end models, 3 open-source end-to-end models, and 5 composed generation pipelines. The benchmark includes 500 prompts and associated checklist annotations. For subjective evaluation and repository internals, see `t2av-compass/README.md`.
