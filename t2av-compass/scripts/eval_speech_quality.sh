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
ensure_nisqa_env
if [[ ${SETUP_ONLY} -eq 1 ]]; then
  echo ${T2AV_CORE_ENV} ready
  exit 0
fi
INPUT_DIR=$(resolve_path ${1:-input})
OUTPUT_DIR=$(resolve_path ${2:-Output})
AUDIO_DIR=${OUTPUT_DIR}/audio_wav
require_dir ${INPUT_DIR}
require_video_files ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}/nisqa_output
bash ${CODE_ROOT}/scripts/extract_audio.sh ${INPUT_DIR} ${AUDIO_DIR}
conda_run_in ${T2AV_CORE_ENV} python ${CODE_ROOT}/Objective/Audio/NISQA/run_predict.py --mode predict_dir --pretrained_model ${CODE_ROOT}/Objective/Audio/NISQA/weights/nisqa.tar --data_dir ${AUDIO_DIR} --output_dir ${OUTPUT_DIR}/nisqa_output
${PYTHON_BIN} - <<PY
import csv
import json
from pathlib import Path
csv_path = Path(${OUTPUT_DIR@Q}) / 'nisqa_output' / 'NISQA_results.csv'
out_path = Path(${OUTPUT_DIR@Q}) / 'speech_quality.json'
with csv_path.open('r', encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
values = [float(row['mos_pred']) for row in rows if row.get('mos_pred') not in (None, '')]
payload = {
    'metric': 'speech_quality',
    'summary': {
        'SQ_mean': float(sum(values) / len(values)) if values else 0.0,
        'total_samples': len(rows),
    },
    'results': rows,
}
out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(f'Saved results to {out_path}')
PY
