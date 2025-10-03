"""
DNA Distillation: A comprehensive package for DNA sequence analysis and model distillation.

This package provides tools for:
- Training student models on DNA sequence tasks
- Knowledge distillation from large nucleotide transformer models
- Various neural network architectures optimized for DNA analysis
- Data loading and preprocessing for genomics tasks
- SLURM job management for HPC clusters
"""

from .version import VERSION, VERSION_SHORT

# Core modules
from . import models
from . import training  
from . import data
from . import distillation
# Utils module removed (was mainly SLURM functionality)

# Convenient imports for common use cases
from .models import (
    BiLSTMStudent,
    XLSTMStudent,
    MambaSSM,
    CNNStudent,
    create_student_model,
    # Advanced models from latest research
    create_advanced_model,
    get_advanced_model_info,
    ADVANCED_MODEL_TYPES,
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
    get_available_tasks,
    get_task_info,
)
from .distillation import (
    KnowledgeDistiller,
    distill_for_task,
    DistillationLoss,
    precompute_teacher_logits,
    # Advanced distillation methods from latest research
    AdvancedDistillationLoss,
    logit_standardization_kd_loss,
    dkd_loss,
    dist_loss,
    reviewkd_loss,
    precompute_teacher_features,
    create_hierarchical_checkpoint_dir,
    hyperparameter_search,
    get_method_hyperparameters,
    setup_mixed_precision_training,
    create_advanced_distillation_config,
)
# SLURM functionality removed for simplicity

__version__ = VERSION
__all__ = [
    "VERSION",
    "VERSION_SHORT", 
    "models",
    "training",
    "data", 
    "distillation",
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
    "get_available_tasks",
    "get_task_info",
    "create_student_model",
    "KnowledgeDistiller",
    "distill_for_task",
    "DistillationLoss",
    "precompute_teacher_logits",
    # Advanced features from latest research
    "create_advanced_model",
    "get_advanced_model_info",
    "ADVANCED_MODEL_TYPES",
    "AdvancedDistillationLoss",
    "logit_standardization_kd_loss",
    "dkd_loss",
    "dist_loss",
    "reviewkd_loss",
    "precompute_teacher_features",
    "create_hierarchical_checkpoint_dir",
    "hyperparameter_search",
    "get_method_hyperparameters",
    "setup_mixed_precision_training",
    "create_advanced_distillation_config",
]
