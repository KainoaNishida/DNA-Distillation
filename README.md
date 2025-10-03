# DNA Distillation

A powerful Python package for DNA sequence analysis and knowledge distillation, implementing state-of-the-art methods from recent research.

## Features

- **Advanced Student Models**: BiLSTM, XLSTM, Mamba, Hyena, Caduceus, CNN with residual blocks, MLP, RNN, BPNet
- **Knowledge Distillation Methods**: DKD, DIST, ReviewKD, Logit Standardization (CVPR 2024)
- **Teacher Model Training**: Support for Nucleotide Transformer (500M/2.5B), DNA-BERT2, Enformer, and Caduceus models
- **Real-time Monitoring**: WandB integration for experiment tracking
- **Production Ready**: Clean CLI interface and comprehensive documentation

## Known Issues

- **NumPy Compatibility**: There's a known compatibility issue between the `datasets` library and NumPy 2.0. The package works with NumPy 1.x versions.
- **Model Dependencies**: Some teacher models require additional packages:
  - **DNA-BERT2**: Requires `einops` (included in requirements)
  - **Enformer**: Requires `enformer-pytorch` (included in requirements)
  - **Caduceus**: Requires `mamba-ssm` (not available on macOS due to compilation issues - Linux only)

### Caduceus Model Limitation

The Caduceus teacher model training is **not available on macOS** due to the `mamba_ssm` dependency which cannot be compiled on macOS. This is a known limitation of the Caduceus model itself, not our implementation.

**Workarounds:**

1. **Use Linux with CUDA**: The Caduceus training works perfectly on Linux systems with CUDA support
2. **Use other teacher models**: DNA-BERT2, Enformer, and Nucleotide Transformer work on all platforms
3. **Use pre-trained Caduceus models**: You can still use pre-trained Caduceus models for distillation on macOS

**For Linux users**, the Caduceus training includes:

- Early stopping with target MCC thresholds for all 17 downstream tasks
- AdamW optimizer with cosine annealing scheduler
- WandB logging and checkpointing
- Best model saving based on validation MCC
- CSV summary generation

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

## Complete Training Pipeline

### Step 1: Train Teacher Models

#### DNA-BERT2 Teacher (Recommended for macOS)

```bash
# Setup environment (one-time)
uv venv .uv/dnabert2 -p 3.10
uv pip install -r envs/dnabert2-req.txt -p .uv/dnabert2
uv pip install -p .uv/dnabert2 matplotlib seaborn plotly rich tyro==0.5.7

# Train DNA-BERT2 teacher
./.uv/dnabert2/bin/python -m dna_distillation.cli train-teacher \
  --teacher-model-type dna_bert2 \
  --task H3K4me1 \
  --output-dir teacher_models/dnabert2_model \
  --no-use-mixed-precision \
  --batch-size 4 \
  --num-train-epochs 10
```

#### Enformer Teacher

```bash
# Setup environment (one-time)
uv venv .uv/enformer -p 3.10
uv pip install -r envs/enformer-req.txt -p .uv/enformer
uv pip install -p .uv/enformer matplotlib seaborn plotly rich tyro==0.5.7

# Train Enformer teacher
./.uv/enformer/bin/python -m dna_distillation.cli train-teacher \
  --teacher-model-type enformer \
  --task H3K4me1 \
  --output-dir teacher_models/enformer_model \
  --num-train-epochs 10
```

#### Nucleotide Transformer 500M Teacher

```bash
# Setup environment (one-time)
uv venv .uv/nt500m -p 3.10
uv pip install -r envs/nt500m-req.txt -p .uv/nt500m
uv pip install -p .uv/nt500m matplotlib seaborn plotly rich tyro==0.5.7

# Train NT500M teacher
./.uv/nt500m/bin/python -m dna_distillation.cli train-teacher \
  --teacher-model-type nucleotide_transformer_500m \
  --task H3K4me1 \
  --output-dir teacher_models/nt500m_model \
  --no-use-mixed-precision \
  --num-train-epochs 10
```

#### Caduceus Teacher (Linux only)

```bash
# Setup environment (one-time) - Linux only
uv venv .uv/caduceus -p 3.10
uv pip install -r envs/caduceus-req.txt -p .uv/caduceus
uv pip install -p .uv/caduceus matplotlib seaborn plotly rich tyro==0.5.7

# Train Caduceus teacher (Linux only)
./.uv/caduceus/bin/python -m dna_distillation.cli train-teacher \
  --teacher-model-type caduceus \
  --task H3K4me1 \
  --output-dir teacher_models/caduceus_model \
  --num-train-epochs 10
```

### Step 2: Train Student Models with Knowledge Distillation

#### Simple Distillation (Quick Test)

```bash
# Test with synthetic data
python -m dna_distillation.cli simple-distill \
  --method dkd \
  --student-model-type bilstm \
  --num-epochs 3

# Test different student models
python -m dna_distillation.cli simple-distill \
  --method dist \
  --student-model-type cnn \
  --num-epochs 3

python -m dna_distillation.cli simple-distill \
  --method dkd \
  --student-model-type bpnet \
  --num-epochs 3
```

#### Advanced Distillation with Real Data

```bash
# Distill DNA-BERT2 teacher to BiLSTM student
python -m dna_distillation.cli advanced-distill \
  --task-name H3K4me1 \
  --method dkd \
  --student-model-type bilstm \
  --teacher-model-path teacher_models/dnabert2_model \
  --output-dir distillation_results/dnabert2_to_bilstm \
  --num-epochs 10 \
  --use-wandb

# Distill Enformer teacher to CNN student
python -m dna_distillation.cli advanced-distill \
  --task-name H3K4me1 \
  --method dist \
  --student-model-type cnn \
  --teacher-model-path teacher_models/enformer_model \
  --output-dir distillation_results/enformer_to_cnn \
  --num-epochs 15 \
  --use-wandb

# Distill NT500M teacher to BPNet student
python -m dna_distillation.cli advanced-distill \
  --task-name H3K4me1 \
  --method reviewkd \
  --student-model-type bpnet \
  --teacher-model-path teacher_models/nt500m_model \
  --output-dir distillation_results/nt500m_to_bpnet \
  --num-epochs 20 \
  --use-wandb
```

## Complete CLI Reference

### Teacher Model Training Commands

#### List Available Teacher Models

```bash
python -m dna_distillation.cli list-teacher-models
```

Output:

```
Available Teacher Model Types (5)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Model Type                  ┃ Description                                             ┃ Memory Requirement ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ nucleotide_transformer_500m │ 500M parameter Nucleotide Transformer (Human Reference) │ ~8GB GPU memory    │
│ nucleotide_transformer_2b5  │ 2.5B parameter Nucleotide Transformer (Multi-species)   │ ~24GB GPU memory   │
│ dna_bert2                   │ DNA-BERT2 117M parameter model                          │ ~4GB GPU memory    │
│ enformer                    │ Enformer model for regulatory element prediction        │ ~12GB GPU memory   │
│ caduceus                    │ Caduceus bidirectional state space model                │ ~6GB GPU memory    │
└─────────────────────────────┴─────────────────────────────────────────────────────────┴────────────────────┘
```

#### Train Teacher Models

```bash
# DNA-BERT2 (macOS compatible)
python -m dna_distillation.cli train-teacher \
  --teacher-model-type dna_bert2 \
  --task H3K4me1 \
  --output-dir teacher_models/dnabert2_model \
  --no-use-mixed-precision \
  --batch-size 4 \
  --num-train-epochs 10

# Enformer (macOS compatible)
python -m dna_distillation.cli train-teacher \
  --teacher-model-type enformer \
  --task H3K4me1 \
  --output-dir teacher_models/enformer_model \
  --num-train-epochs 10

# Nucleotide Transformer 500M (macOS compatible)
python -m dna_distillation.cli train-teacher \
  --teacher-model-type nucleotide_transformer_500m \
  --task H3K4me1 \
  --output-dir teacher_models/nt500m_model \
  --no-use-mixed-precision \
  --num-train-epochs 10

# Caduceus (Linux only)
python -m dna_distillation.cli train-teacher \
  --teacher-model-type caduceus \
  --task H3K4me1 \
  --output-dir teacher_models/caduceus_model \
  --num-train-epochs 10
```

### Student Model Training Commands

#### List Available Student Models

```bash
python -m dna_distillation.cli list-models
```

Output:

```
Available Student Model Types (9)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model Type                  ┃ Description                                             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ bilstm                      │ Bidirectional LSTM with attention                      │
│ xlstm                       │ Extended LSTM with improved memory                     │
│ mamba                       │ State space model for long sequences                   │
│ hyena                       │ Long-range dependency modeling                         │
│ caduceus                    │ Bidirectional state space model                        │
│ cnn                         │ Convolutional network with residual blocks              │
│ mlp                         │ Multi-layer perceptron                                 │
│ rnn                         │ Recurrent neural network                               │
│ bpnet                       │ BPNet with dilated convolutions and attention          │
└─────────────────────────────┴─────────────────────────────────────────────────────────┘
```

#### List Available Tasks

```bash
python -m dna_distillation.cli list-tasks
```

Output:

```
Available Tasks (17)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Task Name                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ H3K9ac                     │
│ H3K27ac                    │
│ enhancers                  │
│ enhancers_types            │
│ promoter_all               │
│ promoter_tata              │
│ promoter_no_tata           │
│ H3K4me1                    │
│ H3K4me2                    │
│ H3K4me3                    │
│ H2AFZ                      │
│ H3K27me3                   │
│ H3K36me3                   │
│ H3K9me3                    │
│ H4K20me1                   │
│ splice_sites_donors        │
│ splice_sites_all           │
│ splice_sites_acceptors     │
└─────────────────────────────┘
```

#### Create Student Models

```bash
# Create a BiLSTM model
python -m dna_distillation.cli create-model \
  --model-type bilstm \
  --output-path student_models/bilstm_model.pt

# Create a CNN with residual blocks
python -m dna_distillation.cli create-model \
  --model-type cnn \
  --num-res-blocks 16 \
  --use-one-hot \
  --output-path student_models/cnn_model.pt

# Create a BPNet model
python -m dna_distillation.cli create-model \
  --model-type bpnet \
  --output-path student_models/bpnet_model.pt
```

#### Run Knowledge Distillation

```bash
# Simple distillation with synthetic data
python -m dna_distillation.cli simple-distill \
  --method dkd \
  --student-model-type bilstm \
  --num-epochs 10

# Advanced distillation with real data
python -m dna_distillation.cli advanced-distill \
  --task-name H3K4me1 \
  --method dkd \
  --student-model-type bilstm \
  --teacher-model-path teacher_models/dnabert2_model \
  --output-dir distillation_results \
  --num-epochs 10 \
  --use-wandb
```

### Tyro CLI Usage Examples

#### Understanding Tyro Commands

Tyro provides a powerful CLI interface with automatic help and validation:

```bash
# Get help for any command
python -m dna_distillation.cli train-teacher --help

# Get help for specific parameters
python -m dna_distillation.cli train-teacher --teacher-model-type --help

# List all available options
python -m dna_distillation.cli --help
```

#### Common Tyro Patterns

```bash
# Boolean flags (use --no- prefix for False)
--use-mixed-precision          # True
--no-use-mixed-precision       # False

# Enum choices (auto-completion available)
--teacher-model-type dna_bert2
--student-model-type bilstm
--method dkd

# File paths (auto-completion available)
--output-dir teacher_models/my_model
--teacher-model-path teacher_models/dnabert2_model

# Numeric parameters
--batch-size 4
--num-train-epochs 10
--learning-rate 0.0001
```

#### Advanced Tyro Usage

```bash
# Multiple tasks in sequence
for task in H3K4me1 H3K27ac enhancers; do
  python -m dna_distillation.cli train-teacher \
    --teacher-model-type dna_bert2 \
    --task $task \
    --output-dir teacher_models/dnabert2_$task \
    --no-use-mixed-precision \
    --batch-size 4
done

# Different student models with same teacher
for student in bilstm cnn bpnet; do
  python -m dna_distillation.cli advanced-distill \
    --task-name H3K4me1 \
    --method dkd \
    --student-model-type $student \
    --teacher-model-path teacher_models/dnabert2_model \
    --output-dir distillation_results/dnabert2_to_$student \
    --num-epochs 10
done
```

## Complete Workflow Examples

### Example 1: DNA-BERT2 → BiLSTM Distillation

```bash
# 1. Train DNA-BERT2 teacher
./.uv/dnabert2/bin/python -m dna_distillation.cli train-teacher \
  --teacher-model-type dna_bert2 \
  --task H3K4me1 \
  --output-dir teacher_models/dnabert2_h3k4me1 \
  --no-use-mixed-precision \
  --batch-size 4 \
  --num-train-epochs 10

# 2. Distill to BiLSTM student
python -m dna_distillation.cli advanced-distill \
  --task-name H3K4me1 \
  --method dkd \
  --student-model-type bilstm \
  --teacher-model-path teacher_models/dnabert2_h3k4me1 \
  --output-dir distillation_results/dnabert2_to_bilstm \
  --num-epochs 15 \
  --use-wandb
```

### Example 2: Enformer → BPNet Distillation

```bash
# 1. Train Enformer teacher
./.uv/enformer/bin/python -m dna_distillation.cli train-teacher \
  --teacher-model-type enformer \
  --task H3K27ac \
  --output-dir teacher_models/enformer_h3k27ac \
  --num-train-epochs 10

# 2. Distill to BPNet student
python -m dna_distillation.cli advanced-distill \
  --task-name H3K27ac \
  --method dist \
  --student-model-type bpnet \
  --teacher-model-path teacher_models/enformer_h3k27ac \
  --output-dir distillation_results/enformer_to_bpnet \
  --num-epochs 20 \
  --use-wandb
```

### Example 3: Multi-Task Teacher Training

```bash
# Train DNA-BERT2 on multiple tasks
for task in H3K4me1 H3K27ac enhancers promoter_all; do
  ./.uv/dnabert2/bin/python -m dna_distillation.cli train-teacher \
    --teacher-model-type dna_bert2 \
    --task $task \
    --output-dir teacher_models/dnabert2_$task \
    --no-use-mixed-precision \
    --batch-size 4 \
    --num-train-epochs 10
done
```

## WandB Integration

All experiments are automatically logged to WandB for real-time monitoring and visualization.

### WandB Setup

#### 1. Create WandB Account

1. Go to [https://wandb.ai](https://wandb.ai)
2. Sign up for a free account
3. Verify your email address

#### 2. Install WandB

```bash
# Install WandB (included in requirements)
pip install wandb

# Or install in specific uv environment
uv pip install -p .uv/dnabert2 wandb
```

#### 3. Login to WandB

```bash
# Login to your WandB account
wandb login

# You'll be prompted to:
# 1. Go to https://wandb.ai/authorize
# 2. Copy your API key
# 3. Paste it in the terminal
```

#### 4. Verify Setup

```bash
# Check if you're logged in
wandb status

# Should show something like:
# wandb: Currently logged in as: your-username (your-username-organization)
# wandb: wandb version 0.19.11 is available!  To upgrade, please run:
# wandb:  $ pip install wandb --upgrade
```

### Using WandB with DNA Distillation

#### Automatic Logging

All training commands automatically log to WandB when `--use-wandb` flag is used:

```bash
# Teacher model training with WandB
python -m dna_distillation.cli train-teacher \
  --teacher-model-type dna_bert2 \
  --task H3K4me1 \
  --output-dir teacher_models/dnabert2_model \
  --use-wandb

# Student model distillation with WandB
python -m dna_distillation.cli advanced-distill \
  --task-name H3K4me1 \
  --method dkd \
  --student-model-type bilstm \
  --teacher-model-path teacher_models/dnabert2_model \
  --output-dir distillation_results \
  --use-wandb
```

#### WandB Project Organization

Experiments are automatically organized by project:

- **Teacher Training**: `huggingface` (for DNA-BERT2, NT500M)
- **Enformer Training**: `EnformerRevised-h800-epo100-maxlen1024-bz16-pretrained`
- **Caduceus Training**: `Caduceus-nt-revised-finetune`
- **Student Distillation**: `dna-distillation` (default)

#### Viewing Experiments

```bash
# View your experiments in terminal
wandb runs list

# View specific project
wandb runs list --project huggingface

# View runs for specific task
wandb runs list --project huggingface --tag H3K4me1
```

#### WandB Web Interface

1. Go to [https://wandb.ai](https://wandb.ai)
2. Navigate to your projects
3. View real-time metrics, loss curves, and model performance
4. Compare different experiments
5. Download model checkpoints and logs

#### WandB Features

- **Real-time Metrics**: Live loss curves, accuracy, F1-score, MCC
- **Model Tracking**: Automatic model checkpointing and versioning
- **Hyperparameter Logging**: All training parameters automatically logged
- **System Monitoring**: GPU/CPU usage, memory consumption
- **Artifact Storage**: Model weights, tokenizers, and configs
- **Experiment Comparison**: Side-by-side comparison of different runs
- **Collaboration**: Share experiments with team members

#### Troubleshooting WandB

```bash
# If you get login errors
wandb login --relogin

# If you want to run offline
wandb offline

# To go back online
wandb online

# Check WandB status
wandb status

# View WandB logs
wandb logs
```

#### Example WandB Dashboard

After running experiments, you'll see:

- **Loss Curves**: Training and validation loss over time
- **Metrics**: F1-score, MCC, accuracy progression
- **System Info**: GPU utilization, memory usage
- **Hyperparameters**: All training arguments and model configs
- **Model Artifacts**: Saved checkpoints and final models

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

## Per-Model Environments with uv

For maximum compatibility and reproducibility, use isolated `uv` environments:

```bash
# DNA-BERT2 environment
uv venv .uv/dnabert2 -p 3.10
uv pip install -r envs/dnabert2-req.txt -p .uv/dnabert2
uv pip install -p .uv/dnabert2 matplotlib seaborn plotly rich tyro==0.5.7

# Enformer environment
uv venv .uv/enformer -p 3.10
uv pip install -r envs/enformer-req.txt -p .uv/enformer
uv pip install -p .uv/enformer matplotlib seaborn plotly rich tyro==0.5.7

# Nucleotide Transformer 500M environment
uv venv .uv/nt500m -p 3.10
uv pip install -r envs/nt500m-req.txt -p .uv/nt500m
uv pip install -p .uv/nt500m matplotlib seaborn plotly rich tyro==0.5.7

# Caduceus environment (Linux only)
uv venv .uv/caduceus -p 3.10
uv pip install -r envs/caduceus-req.txt -p .uv/caduceus
uv pip install -p .uv/caduceus matplotlib seaborn plotly rich tyro==0.5.7
```

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
