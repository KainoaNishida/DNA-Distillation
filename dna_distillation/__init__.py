"""
DNA Distillation: A comprehensive package for DNA sequence analysis and model distillation.

This package provides tools for:
- Training student models on DNA sequence tasks
- Knowledge distillation from large nucleotide transformer models
- Various neural network architectures optimized for DNA analysis
- Data loading and preprocessing for genomics tasks
"""

from .version import VERSION, VERSION_SHORT

# Core modules
from . import models
from . import training  
from . import data
from . import distillation
from . import utils

# Convenient imports for common use cases
from .models import (
    BiLSTMStudent,
    XLSTMStudent,
    MambaSSM,
    CNNStudent,
)
from .training import (
    DNATrainer,
    DistillationTrainer,
    compute_metrics,
    count_parameters,
    get_default_training_args,
    train_for_task,
)
from .data import (
    load_nucleotide_task,
    get_num_labels,
)
from .distillation import (
    KnowledgeDistiller,
)

__version__ = VERSION
__all__ = [
    "VERSION",
    "VERSION_SHORT", 
    "models",
    "training",
    "data", 
    "distillation",
    "utils",
    # Convenient imports
    "BiLSTMStudent",
    "XLSTMStudent", 
    "MambaSSM",
    "CNNStudent",
    "DNATrainer",
    "DistillationTrainer",
    "compute_metrics",
    "count_parameters",
    "get_default_training_args",
    "train_for_task",
    "load_nucleotide_task",
    "get_num_labels", 
    "KnowledgeDistiller",
]
