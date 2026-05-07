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
ensure_latentsync_env
if [[ ${SETUP_ONLY} -eq 1 ]]; then
  echo t2av-latentsync ready
  exit 0
fi
INPUT_DIR=$(resolve_path ${1:-input})
OUTPUT_DIR=$(resolve_path ${2:-Output})
LATENTSYNC_DIR=${CODE_ROOT}/Objective/Similarity/LatentSync
require_dir ${INPUT_DIR}
require_video_files ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}
(
  cd ${LATENTSYNC_DIR}
  conda_run_in t2av-latentsync python ${CODE_ROOT}/scripts/batch_lipsync.py --video_dir ${INPUT_DIR} --model_path ${LATENTSYNC_DIR}/checkpoints/auxiliary/syncnet_v2.model --output_file ${OUTPUT_DIR}/lipsync.json --device cuda --temp_dir ${OUTPUT_DIR}/temp_lipsync
)
