"""Training utilities and trainers for DNA sequence models."""

from .utils import (
    compute_metrics,
    compute_metrics_f1,
    compute_metrics_mcc,
    count_parameters,
    get_model_size_mb,
    get_training_summary,
    validate_training_inputs,
    create_training_config,
    get_default_training_args,
)

from .trainers import (
    DNATrainer,
    DistillationTrainer,
    TestEvalCallback,
)

from .core import (
    train_for_task,
    train_multiple_tasks,
)

__all__ = [
    # Utilities
    "compute_metrics",
    "compute_metrics_f1", 
    "compute_metrics_mcc",
    "count_parameters",
    "get_model_size_mb",
    "get_training_summary",
    "validate_training_inputs",
    "create_training_config",
    "get_default_training_args",
    # Trainers
    "DNATrainer",
    "DistillationTrainer", 
    "TestEvalCallback",
    # Core functions
    "train_for_task",
    "train_multiple_tasks",
]
