# DNA Distillation

A comprehensive Python package for DNA sequence analysis and knowledge distillation from large nucleotide transformer models to efficient student models.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/dna-distillation.svg)](https://badge.fury.io/py/dna-distillation)

## 🚀 Features

### **One-Line Training & Distillation**

- **Training**: `dna.train_for_task("H3K27ac", "bilstm", "./outputs")`
- **Distillation**: `dna.distill_for_task("H3K27ac", "bilstm", "./teacher_checkpoints", "./outputs")`
- **SLURM Jobs**: `dna.create_training_job("H3K27ac", "bilstm", "./outputs")`

### **Comprehensive Data Loading**

- Support for 18+ downstream DNA sequence tasks
- Automatic train/validation/test splitting
- HuggingFace tokenizer integration
- Memory-efficient dataset handling

### **Advanced Knowledge Distillation**

- **Multiple Loss Functions**: Combined CE+MSE, KL divergence, attention, feature distillation
- **Precomputed Teacher Logits**: Memory-efficient offline computation
- **Automatic Teacher Loading**: Smart checkpoint finding and model loading
- **Temperature Scaling**: Configurable distillation temperature
- **Multi-Task Support**: Batch distillation across multiple tasks

### **HPC Cluster Integration**

- **SLURM Job Management**: Automatic job script generation
- **Pre-configured Setups**: GPU single/multi, CPU, debug configurations
- **Cluster-Specific Optimization**: Customizable for different HPC environments
- **Batch Job Submission**: Large-scale experiment management

### **Student Model Architectures**

- **BiLSTM**: Bidirectional LSTM with attention
- **XLSTM**: Extended LSTM with residual connections
- **Mamba**: State space model for long sequences
- **Hyena**: Efficient attention-free architecture
- **Caduceus**: DNA-specific transformer variant
- **CNN**: Convolutional neural networks
- **MLP**: Multi-layer perceptrons
- **RNN**: Recurrent neural networks

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install dna-distillation
```

### From Source

```bash
git clone https://github.com/KainoaNishida/DNA-Distillation.git
cd DNA-Distillation
pip install -e .
```

### Optional Dependencies

```bash
# For GPU support
pip install dna-distillation[gpu]

# For bioinformatics tools
pip install dna-distillation[bio]

# For visualization
pip install dna-distillation[viz]

# For nucleotide transformer models
pip install dna-distillation[nucleotide]

# For development
pip install dna-distillation[dev]
```

## 🎯 Quick Start

### Basic Training

```python
import dna_distillation as dna

# Train a BiLSTM model on H3K27ac task
results = dna.train_for_task(
    task_name="H3K27ac",
    model_type="bilstm",
    output_dir="./outputs/H3K27ac_bilstm",
    num_train_epochs=10
)

print(f"Final F1: {results['final_f1']:.4f}")
print(f"Final MCC: {results['final_mcc']:.4f}")
```

### Knowledge Distillation

```python
# Distill knowledge from teacher to student model
results = dna.distill_for_task(
    task_name="H3K27ac",
    student_model_type="bilstm",
    teacher_checkpoint_dir="./teacher_checkpoints",
    output_dir="./distilled_models/H3K27ac_bilstm",
    alpha=0.5,  # Weight for cross-entropy loss
    temperature=2.0  # Distillation temperature
)

print(f"Distilled F1: {results['final_f1']:.4f}")
```

### SLURM Job Creation

```python
# Create SLURM job for training
script_path = dna.create_training_job(
    task_name="H3K27ac",
    model_type="bilstm",
    output_dir="./outputs",
    config=dna.get_default_slurm_configs()["gpu_single"]
)

# Submit the job
job_id = dna.submit_job(script_path)
print(f"Job submitted: {job_id}")
```

## 📚 Detailed Usage

### Data Loading

#### Load a Single Task

```python
import dna_distillation as dna
from transformers import AutoTokenizer

# Load task data
datasets = dna.load_nucleotide_task("H3K27ac")
train_data = datasets["train"]
val_data = datasets["validation"]
test_data = datasets["test"]

# With tokenization
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
datasets = dna.load_nucleotide_task("H3K27ac", tokenizer=tokenizer)
```

#### Load Multiple Tasks

```python
# Load multiple tasks
all_datasets = dna.load_multiple_tasks(["H3K27ac", "H3K4me3", "promoter_all"])

# Get task information
num_labels = dna.get_num_labels("H3K27ac")
task_info = dna.get_task_info("H3K27ac")
available_tasks = dna.get_available_tasks()
```

### Training

#### Single Task Training

```python
# Basic training
results = dna.train_for_task(
    task_name="H3K27ac",
    model_type="bilstm",
    output_dir="./outputs",
    num_train_epochs=10,
    learning_rate=1e-5,
    batch_size=8
)

# Advanced training with custom configuration
results = dna.train_for_task(
    task_name="H3K27ac",
    model_type="mamba",
    output_dir="./outputs",
    num_train_epochs=15,
    learning_rate=5e-5,
    batch_size=4,
    embed_dim=1280,
    hidden_dim=512,
    wandb_project="DNA-Distillation",
    wandb_run_name="H3K27ac_mamba_experiment"
)
```

#### Multi-Task Training

```python
# Train on multiple tasks
results = dna.train_multiple_tasks(
    task_names=["H3K27ac", "H3K4me3", "promoter_all"],
    model_type="bilstm",
    output_base_dir="./multi_task_outputs",
    num_train_epochs=10
)

# Check results for each task
for task_name, task_results in results.items():
    if "error" not in task_results:
        print(f"{task_name}: F1={task_results['final_f1']:.4f}")
```

### Knowledge Distillation

#### Basic Distillation

```python
# Simple distillation
results = dna.distill_for_task(
    task_name="H3K27ac",
    student_model_type="bilstm",
    teacher_checkpoint_dir="./teacher_checkpoints",
    output_dir="./distilled_models",
    alpha=0.5,
    temperature=2.0
)
```

#### Advanced Distillation

```python
# Advanced distillation with custom settings
results = dna.distill_for_task(
    task_name="H3K27ac",
    student_model_type="mamba",
    teacher_checkpoint_dir="./teacher_checkpoints",
    output_dir="./distilled_models",
    alpha=0.3,  # More emphasis on distillation loss
    temperature=4.0,  # Higher temperature
    precompute_teacher_logits=False,  # Online computation
    num_train_epochs=15,
    learning_rate=5e-5,
    batch_size=8
)
```

#### Multi-Task Distillation

```python
# Distill on multiple tasks
results = dna.distill_multiple_tasks(
    task_names=["H3K27ac", "H3K4me3"],
    student_model_type="bilstm",
    teacher_checkpoint_dir="./teacher_checkpoints",
    output_base_dir="./multi_task_distilled"
)
```

#### High-Level Distiller Interface

```python
# Initialize distiller
distiller = dna.KnowledgeDistiller(
    teacher_checkpoint_dir="./teacher_checkpoints",
    tokenizer_name="InstaDeepAI/nucleotide-transformer-500m-human-ref",
    cache_dir="./cache",
    alpha=0.5,
    temperature=2.0
)

# Distill single task
results = distiller.distill(
    task_name="H3K27ac",
    student_model_type="bilstm",
    output_dir="./distilled_models"
)

# Distill multiple tasks
results = distiller.distill_multiple(
    task_names=["H3K27ac", "H3K4me3", "promoter_all"],
    student_model_type="bilstm",
    output_base_dir="./distilled_models"
)
```

### SLURM Job Management

#### Pre-configured Configurations

```python
# Get default configurations
configs = dna.get_default_slurm_configs()

# Single GPU
gpu_single = configs["gpu_single"]  # 1 GPU, 32G RAM, 4 CPUs

# Multi-GPU
gpu_multi = configs["gpu_multi"]    # 2 GPUs, 64G RAM, 8 CPUs

# CPU-only
cpu_config = configs["cpu"]         # 16 CPUs, 64G RAM

# Debug
debug_config = configs["debug"]     # 1 GPU, 16G RAM, 2 CPUs
```

#### Custom SLURM Configuration

```python
# Create custom configuration
custom_config = dna.SLURMConfig(
    job_name="dna_custom",
    gres="gpu:1:rtx6000",  # Specific GPU type
    mem="64G",
    cpus_per_task=8,
    partition="gpu-long",
    time="72:00:00",  # 3 days
    nodelist="gpu01,gpu02",
    mail_user="your.email@institution.edu",
    conda_env="dna-distillation",
    work_dir="/scratch/your_username/dna_project"
)
```

#### Job Creation and Submission

```python
# Create training job
training_script = dna.create_training_job(
    task_name="H3K27ac",
    model_type="bilstm",
    output_dir="./outputs",
    config=custom_config,
    num_epochs=10
)

# Create distillation job
distillation_script = dna.create_distillation_job(
    task_name="H3K27ac",
    student_model_type="bilstm",
    teacher_checkpoint_dir="./teacher_checkpoints",
    output_dir="./distilled_models",
    config=custom_config
)

# Create multi-task job
multi_task_script = dna.create_multi_task_job(
    task_names=["H3K27ac", "H3K4me3"],
    model_type="bilstm",
    output_base_dir="./multi_task_outputs",
    job_type="train",
    config=custom_config
)

# Submit jobs
job_id1 = dna.submit_job(training_script)
job_id2 = dna.submit_job(distillation_script)
job_id3 = dna.submit_job(multi_task_script)
```

#### Batch Job Submission

```python
# Create and submit multiple jobs
tasks = ["H3K27ac", "H3K4me3", "promoter_all"]
models = ["bilstm", "xlstm", "mamba"]

job_scripts = []
for task in tasks:
    for model in models:
        script = dna.create_training_job(
            task_name=task,
            model_type=model,
            output_dir=f"./outputs/{task}_{model}",
            config=dna.get_default_slurm_configs()["gpu_single"]
        )
        job_scripts.append(script)

# Submit all jobs
job_ids = []
for script in job_scripts:
    job_id = dna.submit_job(script)
    job_ids.append(job_id)
    print(f"Submitted {script}: {job_id}")
```

## 🏗️ Architecture

### Package Structure

```
dna_distillation/
├── data/                    # Data loading and preprocessing
│   ├── __init__.py
│   └── datasets.py
├── models/                  # Student model architectures
│   ├── __init__.py
│   ├── bilstm.py
│   ├── xlstm.py
│   ├── mamba.py
│   └── factory.py
├── training/                # Training utilities
│   ├── __init__.py
│   ├── core.py
│   ├── trainers.py
│   └── utils.py
├── distillation/            # Knowledge distillation
│   ├── __init__.py
│   ├── core.py
│   ├── trainers.py
│   ├── losses.py
│   └── utils.py
├── utils/                   # Utilities
│   ├── __init__.py
│   └── slurm.py
└── __init__.py
```

### Key Components

#### Data Module

- **Task Loading**: Load specific downstream DNA sequence tasks
- **Tokenization**: HuggingFace tokenizer integration
- **Splitting**: Automatic train/validation/test splits
- **Metadata**: Task information and label counts

#### Training Module

- **High-Level API**: One-line training functions
- **Custom Trainers**: Enhanced HuggingFace trainers
- **Multi-Task Support**: Batch training across tasks
- **Metrics**: F1, MCC, and custom evaluation metrics

#### Distillation Module

- **Loss Functions**: Multiple distillation loss types
- **Teacher Integration**: Automatic teacher model loading
- **Logits Precomputation**: Memory-efficient offline computation
- **Temperature Scaling**: Configurable distillation parameters

#### SLURM Module

- **Job Generation**: Automatic SLURM script creation
- **Resource Management**: Pre-configured cluster setups
- **Submission**: Direct job submission integration
- **Batch Processing**: Large-scale experiment management

## 🔬 Supported Tasks

The package supports 18+ downstream DNA sequence tasks:

### Chromatin State Prediction

- `H2AFZ` - H2A.Z histone variant
- `H3K27ac` - Active enhancer mark
- `H3K27me3` - Repressive mark
- `H3K36me3` - Transcription elongation mark
- `H3K4me1` - Enhancer mark
- `H3K4me2` - Promoter/enhancer mark
- `H3K4me3` - Active promoter mark
- `H3K9ac` - Active promoter mark
- `H3K9me3` - Heterochromatin mark
- `H4K20me1` - Repressive mark

### Regulatory Element Prediction

- `promoter_all` - All promoters
- `promoter_tata` - TATA-box promoters
- `promoter_no_tata` - TATA-less promoters
- `enhancers` - Enhancer regions
- `enhancers_types` - Enhancer subtypes

### Splicing Prediction

- `splice_sites_all` - All splice sites
- `splice_sites_acceptor` - Acceptor sites
- `splice_sites_donor` - Donor sites

## 🎛️ Configuration

### Training Configuration

```python
# Default training arguments
training_args = dna.get_default_training_args(
    output_dir="./outputs",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_score",
    greater_is_better=True
)
```

### Distillation Configuration

```python
# Distillation parameters
distillation_config = {
    "alpha": 0.5,              # Weight for cross-entropy loss
    "temperature": 2.0,        # Temperature for logit scaling
    "precompute_logits": True, # Precompute teacher logits
    "cache_dir": "./cache",    # Cache directory for logits
    "batch_size": 16,          # Batch size for logit computation
}
```

### SLURM Configuration

```python
# Cluster-specific configuration
slurm_config = dna.SLURMConfig(
    job_name="dna_distillation",
    gres="gpu:1",
    mem="32G",
    cpus_per_task=4,
    partition="gpu",
    time="24:00:00",
    conda_env="dna-distillation",
    work_dir="/path/to/project",
    mail_user="your.email@institution.edu"
)
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_data.py
pytest tests/test_training.py
pytest tests/test_distillation.py
pytest tests/test_slurm.py

# Run with coverage
pytest --cov=dna_distillation
```

### Test SLURM Functionality

```python
# Test SLURM job creation
python test_slurm_functionality.py
```

## 📊 Performance

### Model Comparison

| Model Type | Parameters | Memory (MB) | Speed (seq/s) |
| ---------- | ---------- | ----------- | ------------- |
| BiLSTM     | ~2M        | ~500        | ~1000         |
| XLSTM      | ~5M        | ~800        | ~800          |
| Mamba      | ~10M       | ~1200       | ~600          |
| CNN        | ~1M        | ~300        | ~1500         |

### Distillation Benefits

- **Size Reduction**: 10-100x smaller student models
- **Speed Improvement**: 2-10x faster inference
- **Memory Efficiency**: 5-20x less memory usage
- **Performance Retention**: 90-95% of teacher performance

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/KainoaNishida/DNA-Distillation.git
cd DNA-Distillation

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Style

We use:

- **Ruff** for linting and formatting
- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **InstaDeepAI** for the nucleotide transformer models
- **HuggingFace** for the transformers library
- **Research Community** for the downstream task datasets

## 📚 Citation

If you use this package in your research, please cite:

```bibtex
@software{dna_distillation,
  title={DNA Distillation: Knowledge Distillation for DNA Sequence Analysis},
  author={Kainoa N. Nishida},
  year={2024},
  url={https://github.com/KainoaNishida/DNA-Distillation}
}
```

## 🔗 Links

- **Documentation**: [ReadTheDocs](https://dna-distillation.readthedocs.io/)
- **PyPI**: [dna-distillation](https://pypi.org/project/dna-distillation/)
- **GitHub**: [DNA-Distillation](https://github.com/KainoaNishida/DNA-Distillation)
- **Issues**: [GitHub Issues](https://github.com/KainoaNishida/DNA-Distillation/issues)

---

**Made with ❤️ for the DNA/ML research community**
