"""Knowledge distillation utilities for DNA sequence models."""

from .losses import (
    DistillationLoss,
    SoftTargetLoss,
    AttentionDistillationLoss,
    FeatureDistillationLoss,
    compute_distillation_loss,
    get_distillation_loss,
    validate_distillation_inputs,
    # Advanced distillation methods (Latest Research)
    logit_standardization_kd_loss,
    dkd_loss,
    dist_loss,
    reviewkd_loss,
    AdvancedDistillationLoss,
)

from .utils import (
    precompute_teacher_logits,
    find_best_teacher_checkpoint,
    load_teacher_model,
    prepare_dataset_for_distillation,
    validate_distillation_setup,
    get_distillation_cache_dir,
    get_distillation_config,
    # Advanced features from latest research
    precompute_teacher_features,
    create_hierarchical_checkpoint_dir,
    hyperparameter_search,
    get_method_hyperparameters,
    setup_mixed_precision_training,
    create_advanced_distillation_config,
)

from .trainers import (
    DistillationTrainer,
    PrecomputedDistillationTrainer,
    OnlineDistillationTrainer,
    create_distillation_trainer,
)

from .core import (
    distill_for_task,
    distill_multiple_tasks,
    KnowledgeDistiller,
)

__all__ = [
    # Loss functions
    "DistillationLoss",
    "SoftTargetLoss", 
    "AttentionDistillationLoss",
    "FeatureDistillationLoss",
    "compute_distillation_loss",
    "get_distillation_loss",
    "validate_distillation_inputs",
    # Advanced distillation methods (Latest Research)
    "logit_standardization_kd_loss",
    "dkd_loss",
    "dist_loss",
    "reviewkd_loss",
    "AdvancedDistillationLoss",
    # Utilities
    "precompute_teacher_logits",
    "find_best_teacher_checkpoint",
    "load_teacher_model",
    "prepare_dataset_for_distillation",
    "validate_distillation_setup",
    "get_distillation_cache_dir",
    "get_distillation_config",
    # Advanced features from latest research
    "precompute_teacher_features",
    "create_hierarchical_checkpoint_dir",
    "hyperparameter_search",
    "get_method_hyperparameters",
    "setup_mixed_precision_training",
    "create_advanced_distillation_config",
    # Trainers
    "DistillationTrainer",
    "PrecomputedDistillationTrainer",
    "OnlineDistillationTrainer",
    "create_distillation_trainer",
    # Core functions
    "distill_for_task",
    "distill_multiple_tasks",
    "KnowledgeDistiller",
]
