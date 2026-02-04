#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-input}"
PROMPTS_JSON="${2:-${INPUT_DIR}/prompts.json}"
OUTPUT_DIR="${3:-Output}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

OUTPUT_DIR_ABS="${PROJECT_ROOT}/${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR_ABS}"

echo "==> Video Aesthetic (Aesthetic Predictor V2.5)"
conda activate t2av-aesthetic
pushd "${PROJECT_ROOT}/Objective/Objective/Video/aesthetic-predictor-v2-5" >/dev/null
python batch_inference.py \
  --video_dir "${PROJECT_ROOT}/${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR_ABS}/video_aesthetic" \
  --num_frames 10
popd >/dev/null
if [[ -f "${OUTPUT_DIR_ABS}/video_aesthetic/results.json" ]]; then
  cp "${OUTPUT_DIR_ABS}/video_aesthetic/results.json" "${OUTPUT_DIR_ABS}/video_aesthetic.json"
fi
conda deactivate

echo "==> Video Technical (DOVER)"
conda activate t2av-dover
pushd "${PROJECT_ROOT}/Objective/Objective/Video/DOVER" >/dev/null
python batch_dover.py \
  --input "${PROJECT_ROOT}/${INPUT_DIR}" \
  --output "${OUTPUT_DIR_ABS}/video_dover.json" \
  --config "./dover.yml" \
  --device "cuda"
popd >/dev/null
conda deactivate

echo "==> Extract audio for audio metrics"
mkdir -p "${OUTPUT_DIR_ABS}/audio_wav"
bash "${PROJECT_ROOT}/Objective/Objective/Audio/audiobox-aesthetics/extract_audio.sh" \
  "${PROJECT_ROOT}/${INPUT_DIR}" \
  "${OUTPUT_DIR_ABS}/audio_wav"

echo "==> Audio Quality (AudioBox Aesthetics)"
conda activate t2av-audiobox
python "${PROJECT_ROOT}/scripts/run_audiobox_batch.py" \
  --audio_dir "${OUTPUT_DIR_ABS}/audio_wav" \
  --output "${OUTPUT_DIR_ABS}/audio_audiobox.json"
conda deactivate

echo "==> Video-Text Similarity (ImageBind)"
conda activate t2av-imagebind-vt
pushd "${PROJECT_ROOT}/Objective/Objective/Similarity/ImageBind-main" >/dev/null
python batch_inference_video_text.py \
  --json_file "${PROJECT_ROOT}/${PROMPTS_JSON}" \
  --video_dir "${PROJECT_ROOT}/${INPUT_DIR}" \
  --output_file "${OUTPUT_DIR_ABS}/video_text_imagebind.json" \
  --device "cuda:0"
popd >/dev/null
conda deactivate

echo "==> Audio-Text Similarity (ImageBind)"
conda activate t2av-imagebind-at
pushd "${PROJECT_ROOT}/Objective/Objective/Similarity/ImageBind-main" >/dev/null
python batch_inference_audio_text.py \
  --json_file "${PROJECT_ROOT}/${PROMPTS_JSON}" \
  --audio_dir "${OUTPUT_DIR_ABS}/audio_wav" \
  --output_file "${OUTPUT_DIR_ABS}/audio_text_imagebind.json" \
  --device "cuda:0"
popd >/dev/null
conda deactivate

echo "==> AV Sync (Synchformer)"
conda activate t2av-synchformer
pushd "${PROJECT_ROOT}/Objective/Objective/Similarity/Synchformer-main" >/dev/null
python batch_test_folder.py \
  --folder "${PROJECT_ROOT}/${INPUT_DIR}" \
  --exp_name "24-01-04T16-39-21" \
  --output "${OUTPUT_DIR_ABS}/av_sync.json" \
  --device "cuda:0"
popd >/dev/null
conda deactivate

echo "All metrics finished. Results in: ${OUTPUT_DIR_ABS}"
