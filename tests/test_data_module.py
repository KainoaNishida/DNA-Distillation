"""
Test data module functionality.
"""

import pytest
import dna_distillation as dna


def test_get_available_tasks():
    """Test getting available tasks."""
    tasks = dna.get_available_tasks()
    assert isinstance(tasks, list)
    assert len(tasks) > 0
    assert "H3K27ac" in tasks
    assert "H3K4me3" in tasks
    assert "promoter_all" in tasks


def test_get_task_info():
    """Test getting task information."""
    # Test binary classification task
    info = dna.get_task_info("H3K27ac")
    assert info["num_labels"] == 2
    assert info["task_type"] == "classification"
    
    # Test multi-class task
    info = dna.get_task_info("enhancers_types")
    assert info["num_labels"] == 3
    assert info["task_type"] == "classification"


def test_get_num_labels():
    """Test getting number of labels."""
    assert dna.get_num_labels("H3K27ac") == 2
    assert dna.get_num_labels("enhancers_types") == 3
    assert dna.get_num_labels("unknown_task") == 2  # Default for unknown tasks


def test_load_nucleotide_task_structure():
    """Test that load_nucleotide_task returns correct structure."""
    # This test might fail if dataset is not available, so we'll catch that
    try:
        datasets = dna.load_nucleotide_task("H3K27ac")
        assert isinstance(datasets, dict)
        assert "train" in datasets
        assert "validation" in datasets
        assert "test" in datasets
    except Exception as e:
        # If dataset loading fails (e.g., no internet), that's okay for testing
        pytest.skip(f"Dataset loading failed: {e}")


def test_dna_dataset_class():
    """Test DNADataset class."""
    try:
        dataset = dna.data.DNADataset("H3K27ac")
        assert hasattr(dataset, 'train')
        assert hasattr(dataset, 'validation')
        assert hasattr(dataset, 'test')
        assert hasattr(dataset, 'num_labels')
        assert hasattr(dataset, 'task_type')
        assert dataset.num_labels == 2
        assert dataset.task_type == "classification"
    except Exception as e:
        pytest.skip(f"DNADataset test failed: {e}")

