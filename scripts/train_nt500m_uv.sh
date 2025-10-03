#!/usr/bin/env bash
set -euo pipefail

TASK=${1:-H3K4me1}
OUT=${2:-teacher_models/nt500m_model}

uv run -p 3.11 -r envs/nt500m-req.txt \
  python -m dna_distillation.cli train-teacher \
  --teacher-model-type nucleotide_transformer_500m \
  --task "$TASK" \
  --output-dir "$OUT"


