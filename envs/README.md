Per-model uv environments

This folder contains minimal requirement files for running each teacher model in an isolated uv virtual environment. Fill in exact versions later.

Files:

- dnabert2-req.txt
- enformer-req.txt
- caduceus-req.txt
- nt500m-req.txt

Usage example:

```bash
# DNABERT2 teacher training (example)
scripts/train_dnabert2_uv.sh --task H3K4me1 --out teacher_models/dnabert2
```
