# DNA Distillation - Production Guide

This guide covers production deployment, configuration, and best practices for the DNA Distillation package.

## Directory Structure

```
DNA-Distillation/
├── dna_distillation/          # Main package code
├── tests/                     # Test suite
├── docs/                      # Documentation
├── teacher_models/            # Teacher model storage
│   └── .gitkeep
├── student_models/            # Student model storage
│   └── .gitkeep
├── distillation_output/       # Basic distillation results
│   └── .gitkeep
├── advanced_distillation/     # Advanced distillation results
│   └── .gitkeep
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── pyproject.toml            # Package configuration
└── README.md                 # Main documentation
```

## Installation

### Production Installation

```bash
# Install from source
git clone <repository-url>
cd DNA-Distillation
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
pip install -e .[dev]
# or
pip install -r requirements-dev.txt
```

## Configuration

### Environment Variables

```bash
# Optional: Set default paths
export DNA_DISTILLATION_TEACHER_MODELS_DIR="./teacher_models"
export DNA_DISTILLATION_STUDENT_MODELS_DIR="./student_models"
export DNA_DISTILLATION_OUTPUT_DIR="./distillation_output"

# Optional: WandB configuration
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="dna-distillation"
```

### Model Storage

#### Teacher Models

Place teacher models in `teacher_models/` with the following structure:

```
teacher_models/your_teacher_model/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```

#### Student Models

Student models are automatically saved to `student_models/` when using CLI commands.

## Usage Examples

### Basic Workflow

```bash
# 1. Train teacher model
python -m dna_distillation.cli train-teacher \
    --task H3K4me1 \
    --teacher-model-type nucleotide_transformer_500m \
    --output-dir ./teacher_models/500m_model

# 2. Train student model
python -m dna_distillation.cli train \
    --task H3K4me1 \
    --model-type bilstm \
    --output-dir ./student_models

# 3. Run distillation
python -m dna_distillation.cli distill \
    --task H3K4me1 \
    --student-model-type bilstm \
    --teacher-checkpoint-dir ./teacher_models/500m_model \
    --output-dir ./distillation_output
```

### Advanced Distillation

```bash
# DKD distillation
python -m dna_distillation.cli advanced-distill \
    --task-name H3K4me1 \
    --method dkd \
    --student-model-type bilstm \
    --teacher-model-path ./teacher_models/500m_model \
    --output-dir ./advanced_distillation \
    --num-epochs 20 \
    --batch-size 32

# DIST distillation
python -m dna_distillation.cli advanced-distill \
    --task-name H3K4me1 \
    --method dist \
    --student-model-type cnn \
    --teacher-model-path ./teacher_models/500m_model \
    --output-dir ./advanced_distillation
```

## Performance Considerations

### Memory Requirements

- **500M Teacher Model**: ~8GB GPU memory
- **2.5B Teacher Model**: ~24GB GPU memory
- **Student Models**: ~1-2GB GPU memory

### Batch Size Recommendations

- **500M Teacher**: batch_size=16-32
- **2.5B Teacher**: batch_size=8-16
- **Student Training**: batch_size=32-64

### Mixed Precision Training

Enable mixed precision for faster training and lower memory usage:

```bash
python -m dna_distillation.cli train-teacher \
    --task H3K4me1 \
    --teacher-model-type nucleotide_transformer_500m \
    --use-mixed-precision
```

## Monitoring and Logging

### WandB Integration

The package automatically integrates with Weights & Biases for experiment tracking:

```bash
# Set your API key
export WANDB_API_KEY="your_api_key"

# Run experiments (automatically logged)
python -m dna_distillation.cli train-teacher --task H3K4me1
```

### Log Files

Training logs are saved in the output directories:

- `teacher_models/*/trainer_state.json`
- `student_models/*/trainer_state.json`
- `distillation_output/*/trainer_state.json`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce batch size
   - Enable mixed precision training
   - Use gradient checkpointing

2. **Model Loading Errors**

   - Ensure all required files are present in teacher model directory
   - Check file permissions
   - Verify model compatibility

3. **Dataset Loading Issues**
   - Check internet connection for HuggingFace datasets
   - Verify task name spelling
   - Check available disk space

### Debug Mode

Enable debug logging:

```bash
export DNA_DISTILLATION_DEBUG=1
python -m dna_distillation.cli train-teacher --task H3K4me1
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_models.py
pytest tests/test_data_module.py

# Run with coverage
pytest --cov=dna_distillation
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "-m", "dna_distillation.cli", "--help"]
```

### Cloud Deployment

- **AWS**: Use EC2 with GPU instances (p3.2xlarge for 2.5B models)
- **Google Cloud**: Use Compute Engine with V100 or A100 GPUs
- **Azure**: Use NC-series VMs with Tesla K80 or V100 GPUs

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the test files for usage examples
3. Open an issue on the repository
4. Check the documentation in `docs/`

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
