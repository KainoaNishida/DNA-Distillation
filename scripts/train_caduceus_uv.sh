#!/usr/bin/env bash
set -euo pipefail

TASK=${1:-H3K4me1}
OUT=${2:-teacher_models/caduceus_model}

uv run -p 3.11 -r envs/caduceus-req.txt \
  python -m dna_distillation.cli train-teacher \
  --teacher-model-type caduceus \
  --task "$TASK" \
  --output-dir "$OUT"


