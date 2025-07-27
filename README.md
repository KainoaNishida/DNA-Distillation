# DNA Distillation

A comprehensive Python package for DNA sequence analysis, machine learning, and bioinformatics processing.

## Features

- **DNA Sequence Analysis**: Built-in support for DNA sequence processing and manipulation
- **Machine Learning**: Integration with PyTorch, Transformers, and modern ML frameworks
- **Bioinformatics Tools**: Support for common bioinformatics file formats and operations
- **Visualization**: Rich plotting and visualization capabilities
- **GPU Acceleration**: Optional GPU support for accelerated computations

## Installation

### Basic Installation

```bash
pip install dna-distillation
```

### From Source

```bash
git clone https://github.com/kainoanishida/DNA-Distillation.git
cd DNA-Distillation
pip install -e .
```

### Optional Dependencies

Install additional functionality with optional dependency groups:

```bash
# Machine learning tools (scikit-learn, optuna, wandb, tensorboard)
pip install dna-distillation[ml]

# GPU acceleration (torch, torchvision, torchmetrics)
pip install dna-distillation[gpu]

# Bioinformatics tools (biopython, pysam, pyfaidx, spliceai)
pip install dna-distillation[bio]

# Visualization tools (matplotlib, seaborn, plotly, bokeh)
pip install dna-distillation[viz]

# All optional dependencies
pip install dna-distillation[ml,gpu,bio,viz]
```

## Development Setup

1. Create a Python 3.8+ virtual environment:

```bash
conda create -n dna-distillation python=3.9
conda activate dna-distillation
```

2. Install the package in development mode with all dependencies:

```bash
pip install -e '.[dev,ml,gpu,bio,viz]'
```

## Usage

```python
import dna_distillation

# Get the package version
print(f"DNA Distillation version: {dna_distillation.VERSION}")

# Example: DNA sequence processing
from Bio import SeqIO
import torch
from transformers import AutoTokenizer

# Your DNA processing code here
```

## Core Dependencies

### Scientific Computing
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **pandas**: Data manipulation

### DNA/Genomics
- **biopython**: Bioinformatics library
- **pysam**: SAM/BAM file processing
- **pyfaidx**: FASTA/FASTQ file indexing
- **nucleotide-transformer**: DNA sequence transformers

### Machine Learning
- **torch**: PyTorch deep learning framework
- **transformers**: Hugging Face transformers
- **datasets**: Dataset loading and processing
- **tokenizers**: Fast tokenization

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive plots
- **bokeh**: Web-based visualization

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black .
isort .
```

### Linting

```bash
ruff check .
```

### Type Checking

```bash
mypy .
```

### Building Documentation

```bash
cd docs && make html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Update the CHANGELOG.md
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Links

- [Repository](https://github.com/kainoanishida/DNA-Distillation)
- [Changelog](https://github.com/kainoanishida/DNA-Distillation/blob/main/CHANGELOG.md)
