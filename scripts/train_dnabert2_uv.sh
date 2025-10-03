#!/usr/bin/env bash
set -euo pipefail

# Args
TASK=${1:-H3K4me1}
OUT=${2:-teacher_models/dnabert2_model}

# Run with per-model uv environment
uv run -p 3.11 -r envs/dnabert2-req.txt \
  python -m dna_distillation.cli train-teacher \
  --teacher-model-type dna_bert2 \
  --task "$TASK" \
  --output-dir "$OUT"


