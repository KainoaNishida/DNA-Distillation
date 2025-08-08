"""
Knowledge distillation utilities for DNA sequence models.

This module provides functionality for distilling knowledge from large
teacher models (like NT-2.5B) into smaller student models.
"""

from .distiller import (
    KnowledgeDistiller,
    precompute_teacher_logits,
    get_best_teacher_checkpoint,
)
from .losses import (
    DistillationLoss,
    TemperatureScaledLoss,
)

__all__ = [
    "KnowledgeDistiller",
    "precompute_teacher_logits", 
    "get_best_teacher_checkpoint",
    "DistillationLoss",
    "TemperatureScaledLoss",
]
