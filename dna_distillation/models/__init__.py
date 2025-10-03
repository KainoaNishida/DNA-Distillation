"""
Student model architectures for DNA sequence analysis.

This module contains various neural network architectures optimized for
DNA sequence classification and analysis tasks.
"""

from .student_models import (
    BiLSTMStudent,
    XLSTMStudent, 
    MambaSSM,
    HyenaSSM,
    CaduceusSSM,
    CNNStudent,
    MLPStudent,
    RNNStudent,
    create_student_model,
)

from .advanced_models import (
    BPNetClassifier,
    EnformerClassifier,
    CaduceusClassifier,
    DNABERT2Classifier,
    create_advanced_model,
    get_advanced_model_info,
    ADVANCED_MODEL_TYPES,
)

__all__ = [
    # Basic student models
    "BiLSTMStudent",
    "XLSTMStudent",
    "MambaSSM", 
    "HyenaSSM",
    "CaduceusSSM",
    "CNNStudent",
    "MLPStudent", 
    "RNNStudent",
    "create_student_model",
    # Advanced models from latest research
    "BPNetClassifier",
    "EnformerClassifier",
    "CaduceusClassifier",
    "DNABERT2Classifier",
    "create_advanced_model",
    "get_advanced_model_info",
    "ADVANCED_MODEL_TYPES",
]
