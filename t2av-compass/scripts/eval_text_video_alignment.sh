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
PROMPTS_JSON=$(resolve_path ${2:-t2av-compass/Data/prompts.json})
OUTPUT_DIR=$(resolve_path ${3:-Output})
IMAGEBIND_DIR=${CODE_ROOT}/Objective/Similarity/ImageBind-main
require_dir ${INPUT_DIR}
require_video_files ${INPUT_DIR}
require_file ${PROMPTS_JSON}
mkdir -p ${OUTPUT_DIR}
(
  cd ${IMAGEBIND_DIR}
  conda_run_in ${T2AV_CORE_ENV} python batch_inference_video_text.py --json_file ${PROMPTS_JSON} --video_dir ${INPUT_DIR} --output_file ${OUTPUT_DIR}/text_video_alignment.json --device cuda:0
)
