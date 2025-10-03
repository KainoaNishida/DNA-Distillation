# DNA Distillation

A powerful Python package for DNA sequence analysis and knowledge distillation, implementing state-of-the-art methods from recent research.

## Features

- **Advanced Student Models**: BiLSTM, XLSTM, Mamba, Hyena, Caduceus, CNN with residual blocks, MLP, RNN, BPNet
- **Knowledge Distillation Methods**: DKD, DIST, ReviewKD, Logit Standardization (CVPR 2024)
- **Teacher Model Training**: Support for Nucleotide Transformer (500M/2.5B), DNA-BERT2, Enformer, and Caduceus models
- **Real-time Monitoring**: WandB integration for experiment tracking
- **Production Ready**: Clean CLI interface and comprehensive documentation

## ⚠️ Known Issues

- **NumPy Compatibility**: There's a known compatibility issue between the `datasets` library and NumPy 2.0. The package works with NumPy 1.x versions.
- **Model Dependencies**: Some teacher models require additional packages:
  - **DNA-BERT2**: Requires `einops` (included in requirements)
  - **Enformer**: Requires `enformer-pytorch` (included in requirements)
  - **Caduceus**: Requires `mamba-ssm` (not included due to compilation issues on macOS)

## Quick Start

### Installation

```bash
pip install dna-distillation
# or use per-model uv envs (recommended for teacher models)
# e.g., DNABERT2 teacher training
bash scripts/train_dnabert2_uv.sh H3K4me1 teacher_models/dnabert2
```

### Basic Usage

```python
import dna_distillation as dna

# Create a student model
model = dna.create_student_model(
    model_type="bilstm",
    vocab_size=1000,
    embed_dim=128,
    hidden_dim=64,
    num_labels=2
)

# Run distillation
dna.run_distillation(
    student_model=model,
    teacher_model_path="teacher_models/my_teacher",
    task_name="H3K4me1",
    method="dkd"
)
```

## CLI Commands

### Quick Test (Recommended for First Use)

```bash
# Run a simple distillation test with synthetic data
python -m dna_distillation.cli simple-distill --method dkd --student-model-type bilstm --num-epochs 3

# Test different student models
python -m dna_distillation.cli simple-distill --method dkd --student-model-type cnn --num-epochs 3
python -m dna_distillation.cli simple-distill --method dist --student-model-type mamba --num-epochs 3
python -m dna_distillation.cli simple-distill --method dkd --student-model-type bpnet --num-epochs 3
```

### Create Student Models

```bash
# Create a BiLSTM model
python -m dna_distillation.cli create-model --model-type bilstm --output-path student_models/bilstm_model.pt

# Create a CNN with residual blocks
python -m dna_distillation.cli create-model --model-type cnn --num-res-blocks 16 --use-one-hot --output-path student_models/cnn_model.pt
```

### Train Teacher Models

```bash
# Train 500M teacher model
python -m dna_distillation.cli train-teacher --teacher-model-type nucleotide_transformer_500m --task H3K4me1 --output-dir teacher_models/500m_model

# Train 2.5B teacher model
python -m dna_distillation.cli train-teacher --teacher-model-type nucleotide_transformer_2b5 --task H3K4me1 --output-dir teacher_models/2b5_model

# Train DNA-BERT2 teacher model
python -m dna_distillation.cli train-teacher --teacher-model-type dna_bert2 --task H3K4me1 --output-dir teacher_models/dnabert2_model

# Train Enformer teacher model
python -m dna_distillation.cli train-teacher --teacher-model-type enformer --task H3K4me1 --output-dir teacher_models/enformer_model

# Train Caduceus teacher model
python -m dna_distillation.cli train-teacher --teacher-model-type caduceus --task H3K4me1 --output-dir teacher_models/caduceus_model

# Or run with uv-based per-model scripts (isolated envs)
bash scripts/train_nt500m_uv.sh H3K4me1 teacher_models/nt500m_model
bash scripts/train_dnabert2_uv.sh H3K4me1 teacher_models/dnabert2_model
bash scripts/train_enformer_uv.sh H3K4me1 teacher_models/enformer_model
bash scripts/train_caduceus_uv.sh H3K4me1 teacher_models/caduceus_model
```

### Run Advanced Distillation

```bash
# Advanced distillation with real data
python -m dna_distillation.cli advanced-distill --task-name H3K4me1 --method dkd --student-model-type bilstm --teacher-model-path teacher_models/my_teacher_model --output-dir ./distillation_results --num-epochs 10 --use-wandb
```

### List Available Options

```bash
# List all available tasks
python -m dna_distillation.cli list-tasks

# List all student model types
python -m dna_distillation.cli list-models

# List teacher model types
python -m dna_distillation.cli list-teacher-models
```

## WandB Integration

All experiments are automatically logged to WandB for real-time monitoring:

```bash
# View your experiments
wandb status

# View runs in terminal
wandb runs list
```

**Your WandB Account**: `kainoanishida` (kainoanishida-university-of-california-irvine)

## Examples

### Complete Distillation Pipeline

```python
import dna_distillation as dna

# 1. Create student model
student = dna.create_student_model(
    model_type="bilstm",
    vocab_size=1000,
    embed_dim=128,
    hidden_dim=64,
    num_labels=2
)

# 2. Load dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")

datasets = dna.load_nucleotide_task(
    task_name="H3K4me1",
    tokenizer=tokenizer,
    max_length=512
)

# 3. Run distillation
results = dna.run_distillation(
    student_model=student,
    teacher_model_path="teacher_models/my_teacher",
    train_dataset=datasets['train'],
    val_dataset=datasets['validation'],
    method="dkd",
    num_epochs=10,
    batch_size=32,
    learning_rate=0.0001
)

print(f"Final F1 Score: {results['final_f1']:.4f}")
print(f"Final MCC: {results['final_mcc']:.4f}")
```

## Advanced Features

### Knowledge Distillation Methods

- **DKD (Decoupled Knowledge Distillation)**: CVPR 2022 method that decouples target and non-target class knowledge
- **DIST (NeurIPS 2022)**: Pearson correlation-based distillation
- **ReviewKD (CVPR 2021)**: Cross-stage knowledge transfer using attention-based fusion
- **Logit Standardization (CVPR 2024)**: Z-score normalization of logits before KL divergence

### Student Model Architectures

- **BiLSTM**: Bidirectional LSTM with attention
- **XLSTM**: Extended LSTM with improved memory
- **Mamba**: State space model for long sequences
- **Hyena**: Long-range dependency modeling
- **Caduceus**: Bidirectional state space model
- **CNN**: Convolutional network with residual blocks
- **MLP**: Multi-layer perceptron
- **RNN**: Recurrent neural network
- **BPNet**: Dilated convolutions with attention mechanism

### Teacher Models

- **500M Nucleotide Transformer**: Human reference genome (~8GB GPU memory)
- **2.5B Nucleotide Transformer**: Multi-species genome (~24GB GPU memory)
- **DNA-BERT2**: 117M parameter DNA foundation model (~4GB GPU memory)
- **Enformer**: Regulatory element prediction model (~12GB GPU memory)
- **Caduceus**: Bidirectional state space model (~6GB GPU memory)

## Train DNABERT-2 Teacher (matches research script)

DNABERT-2 training can be run inside an isolated uv environment that mirrors the research repo pins.

Setup once:

```bash
# Create env and install requirements
uv venv .uv/dnabert2 -p 3.10
uv pip install -p .uv/dnabert2 -r envs/dnabert2-req.txt
```

Run training (Tyro CLI inside the env):

```bash
.uv/dnabert2/bin/python -m dna_distillation.cli train-teacher \
  --teacher-model-type dna_bert2 \
  --task H3K4me1 \
  --output-dir teacher_models/dnabert2_model \
  --no-use-mixed-precision \
  --batch-size 4
```

Notes:

- This path pins the DNABERT-2 tokenizer/model commit and force-downloads, loads task-specific splits, tokenizes with padding="max_length" (512), and resizes embeddings to match tokenizer vocab.
- On CPU/MPS training will be slow; for quick sanity checks use `--batch-size 1 --num-train-epochs 1` or run on a CUDA machine.

## Documentation

- [Installation Guide](INSTALLATION.md)
- [Production Guide](PRODUCTION_GUIDE.md)
- [API Reference](docs/)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{dna_distillation,
  title={DNA Distillation: A Python Package for DNA Sequence Analysis and Knowledge Distillation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/dna-distillation}
}
```
