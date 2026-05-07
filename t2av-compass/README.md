# t2av-compass Codebase Guide

This directory contains the objective and subjective evaluation code.

## Recommended Entry Points

If you want the validated Linux objective pipeline, do not start from `scripts/eval_all_metrics.sh` directly. Use the repository-root wrappers instead:

```bash
cd T2AV-Compass
bash setup_objective.sh
bash run_objective_batch.sh
```

Default validated paths:

- input videos: `input/`
- prompts: `t2av-compass/Data/prompts.json`
- outputs: `Output/`
- cache: `.cache/t2av-cache/`

## Objective Evaluation

The root wrapper calls into `scripts/eval_all_metrics.sh` with repository-relative paths. The objective setup uses a compact environment layout by default:

- `t2av-core`: VA, AA, SQ, T-V, T-A, and A-V
- `t2av-dover`: VT
- `t2av-synchformer`: DeSync
- `t2av-latentsync`: LS

You can still run individual metrics for debugging from the repository root:

```bash
bash t2av-compass/scripts/eval_video_aesthetic.sh input Output
bash t2av-compass/scripts/eval_video_technical.sh input Output
bash t2av-compass/scripts/eval_audio_aesthetic.sh input Output
bash t2av-compass/scripts/eval_speech_quality.sh input Output
bash t2av-compass/scripts/eval_text_video_alignment.sh input t2av-compass/Data/prompts.json Output
bash t2av-compass/scripts/eval_text_audio_alignment.sh input t2av-compass/Data/prompts.json Output
bash t2av-compass/scripts/eval_audio_video_alignment.sh input Output
bash t2av-compass/scripts/eval_av_sync.sh input Output
bash t2av-compass/scripts/eval_lipsync.sh input Output
```

Each script still manages setup automatically. Compatible metrics share `t2av-core`; DOVER, Synchformer, and LatentSync remain isolated to avoid dependency conflicts.

## Subjective Evaluation

Subjective evaluation remains separate from the objective wrapper flow. Run it manually from `Subjective/`.

## Directory Summary

- `Data/`: prompt JSON files
- `Objective/`: third-party metric implementations and wrappers
- `Subjective/`: MLLM-as-a-Judge scripts
- `scripts/`: shell entry points used by the root wrappers

## Maintenance Notes

The following directories are submodule-backed and should be cloned with `--recurse-submodules`:

- `Objective/Audio/NISQA`
- `Objective/Audio/audiobox-aesthetics`
- `Objective/Similarity/LatentSync`
- `Objective/Video/DOVER`
- `Objective/Video/aesthetic-predictor-v2-5`

Environment overrides:

- `T2AV_CACHE_ROOT`: move model/cache files outside the repository
- `T2AV_CONDA_ROOT`: move Conda env and package caches outside the repository
- `T2AV_CONDA_EXE`: point to a specific Conda installation if `conda` is not on `PATH`
- `T2AV_CORE_ENV`: rename the shared objective environment, default `t2av-core`
- `HF_ENDPOINT`: switch to a Hugging Face mirror when needed. In mainland China, prefer `https://hf-mirror.com`.
- `T2AV_GITHUB_MIRROR_PREFIX`: prefix GitHub downloads with a mirror URL when needed
