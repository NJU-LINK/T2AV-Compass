#!/usr/bin/env bash
set -euo pipefail
SETUP_ONLY=0
if [[ ${1:-} == --setup-only ]]; then
  SETUP_ONLY=1
  shift
fi
source $(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/common.sh
ensure_cache_layout
ensure_conda
ensure_imagebind_env
if [[ ${SETUP_ONLY} -eq 1 ]]; then
  echo ${T2AV_CORE_ENV} ready
  exit 0
fi
INPUT_DIR=$(resolve_path ${1:-input})
OUTPUT_DIR=$(resolve_path ${2:-Output})
AUDIO_DIR=${OUTPUT_DIR}/audio_wav
IMAGEBIND_DIR=${CODE_ROOT}/Objective/Similarity/ImageBind-main
require_dir ${INPUT_DIR}
require_video_files ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}
bash ${CODE_ROOT}/scripts/extract_audio.sh ${INPUT_DIR} ${AUDIO_DIR}
(
  cd ${IMAGEBIND_DIR}
  conda_run_in ${T2AV_CORE_ENV} python - <<PY
import json
from pathlib import Path
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
input_dir = Path(${INPUT_DIR@Q})
audio_dir = Path(${AUDIO_DIR@Q})
output_path = Path(${OUTPUT_DIR@Q}) / 'audio_video_alignment.json'
video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
video_files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in video_exts)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)
results = []
scores = []
for video_path in video_files:
    audio_path = audio_dir / f'{video_path.stem}.wav'
    if not audio_path.exists():
        results.append({'file': str(video_path), 'filename': video_path.name, 'audio_file': str(audio_path), 'av_alignment': None, 'error': 'missing extracted audio'})
        continue
    inputs = {
        ModalityType.VISION: data.load_and_transform_video_data([str(video_path)], device),
        ModalityType.AUDIO: data.load_and_transform_audio_data([str(audio_path)], device),
    }
    with torch.no_grad():
        embeddings = model(inputs)
    v = embeddings[ModalityType.VISION]
    a = embeddings[ModalityType.AUDIO]
    v = v / v.norm(dim=-1, keepdim=True)
    a = a / a.norm(dim=-1, keepdim=True)
    score = float((v @ a.T).squeeze().item())
    scores.append(score)
    results.append({'file': str(video_path), 'filename': video_path.name, 'audio_file': str(audio_path), 'av_alignment': score, 'error': None})
payload = {
    'metric': 'audio_video_alignment',
    'summary': {'mean_av_alignment': float(sum(scores) / len(scores)) if scores else 0.0, 'total_samples': len(results)},
    'results': results,
}
output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(f'Saved results to {output_path}')
PY
)
