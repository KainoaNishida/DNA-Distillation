# DNA Distillation

A Python package for DNA sequence analysis and processing.

## Installation

### From PyPI (when available)

```bash
pip install dna-distillation
```

### From source

```bash
git clone https://github.com/kainoanishida/DNA-Distillation.git
cd DNA-Distillation
pip install -e '.[dev]'
```

## Development Setup

1. Create a Python 3.8+ virtual environment:

```bash
conda create -n dna-distillation python=3.9
conda activate dna-distillation
```

2. Install the package in development mode:

```bash
pip install -e '.[dev]'
```

## Usage

```python
import dna_distillation

# Get the package version
print(f"DNA Distillation version: {dna_distillation.VERSION}")

# Your DNA processing code here
```

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
