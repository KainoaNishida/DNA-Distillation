"""
Test package import and basic functionality.
"""

import pytest
import dna_distillation as dna


def test_package_import():
    """Test that the package can be imported successfully."""
    assert hasattr(dna, '__version__')
    assert dna.__version__ == "0.1.0"


def test_core_modules_import():
    """Test that all core modules can be imported."""
    assert hasattr(dna, 'models')
    assert hasattr(dna, 'training')
    assert hasattr(dna, 'data')
    assert hasattr(dna, 'distillation')
    # Utils module removed (was mainly SLURM functionality)


def test_main_functions_available():
    """Test that main functions are available."""
    # Data functions
    assert hasattr(dna, 'load_nucleotide_task')
    assert hasattr(dna, 'get_num_labels')
    assert hasattr(dna, 'get_available_tasks')
    assert hasattr(dna, 'get_task_info')
    
    # Model functions
    assert hasattr(dna, 'create_student_model')
    assert hasattr(dna, 'BiLSTMStudent')
    assert hasattr(dna, 'CNNStudent')
    
    # Training functions
    assert hasattr(dna, 'train_for_task')
    assert hasattr(dna, 'DNATrainer')
    
    # Distillation functions
    assert hasattr(dna, 'distill_for_task')
    assert hasattr(dna, 'KnowledgeDistiller')
    
    # SLURM functions removed for simplicity


def test_version_info():
    """Test version information."""
    assert hasattr(dna, 'VERSION')
    assert hasattr(dna, 'VERSION_SHORT')
    assert dna.VERSION == "0.1.0"
    assert dna.VERSION_SHORT == "0.1"
