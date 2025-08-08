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

__all__ = [
    "BiLSTMStudent",
    "XLSTMStudent",
    "MambaSSM", 
    "HyenaSSM",
    "CaduceusSSM",
    "CNNStudent",
    "MLPStudent", 
    "RNNStudent",
    "create_student_model",
]
