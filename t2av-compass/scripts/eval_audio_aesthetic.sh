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
ensure_audiobox_env
if [[ ${SETUP_ONLY} -eq 1 ]]; then
  echo ${T2AV_CORE_ENV} ready
  exit 0
fi
INPUT_DIR=$(resolve_path ${1:-input})
OUTPUT_DIR=$(resolve_path ${2:-Output})
AUDIO_DIR=${OUTPUT_DIR}/audio_wav
require_dir ${INPUT_DIR}
require_video_files ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}
bash ${CODE_ROOT}/scripts/extract_audio.sh ${INPUT_DIR} ${AUDIO_DIR}
conda_run_in ${T2AV_CORE_ENV} python ${CODE_ROOT}/scripts/run_audiobox_batch.py --audio_dir ${AUDIO_DIR} --output ${OUTPUT_DIR}/audio_aesthetic.json --ckpt ${CACHE_ROOT}/weights/audiobox-aesthetics/checkpoint.pt
