#!/usr/bin/env bash
set -euo pipefail

TASK=${1:-H3K4me1}
OUT=${2:-teacher_models/enformer_model}

uv run -p 3.11 -r envs/enformer-req.txt \
  python -m dna_distillation.cli train-teacher \
  --teacher-model-type enformer \
  --task "$TASK" \
  --output-dir "$OUT"


