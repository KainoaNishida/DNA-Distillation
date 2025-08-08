"""
Data loading utilities for DNA sequence tasks.

This module provides functions to load and preprocess datasets from the
nucleotide transformer downstream tasks.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


# Task metadata
TASK_METADATA = {
    "enhancers_types": {"num_labels": 3, "task_type": "classification"},
    "splice_sites_all": {"num_labels": 3, "task_type": "classification"},
    # All other tasks are binary classification
}

DATASET_NAME = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"


def get_task_info(task_name: str) -> Dict[str, Union[int, str]]:
    """
    Get metadata for a specific task.
    
    Args:
        task_name: Name of the downstream task
        
    Returns:
        Dictionary with task metadata (num_labels, task_type)
    """
    if task_name in TASK_METADATA:
        return TASK_METADATA[task_name].copy()
    else:
        return {"num_labels": 2, "task_type": "classification"}


def get_num_labels(task_name: str) -> int:
    """
    Get the number of labels for a task.
    
    Args:
        task_name: Name of the downstream task
        
    Returns:
        Number of labels (2 for binary, 3 for multi-class)
    """
    return get_task_info(task_name)["num_labels"]


def load_nucleotide_task(
    task_name: str,
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = 512,
    val_split_ratio: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Dataset]:
    """
    Load and preprocess a nucleotide transformer downstream task.
    
    Args:
        task_name: Name of the task to load
        tokenizer: Tokenizer for sequence encoding (optional)
        max_length: Maximum sequence length for tokenization
        val_split_ratio: Ratio for validation split from training data
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'validation', and 'test' datasets
        
    Example:
        >>> datasets = load_nucleotide_task("H3K27ac")
        >>> train_data = datasets["train"]
        >>> val_data = datasets["validation"]
        >>> test_data = datasets["test"]
    """
    try:
        # Load the full dataset
        full_dataset = load_dataset(DATASET_NAME)
        
        # Filter by task
        train_dataset = full_dataset["train"].filter(lambda x: x["task"] == task_name)
        test_dataset = full_dataset["test"].filter(lambda x: x["task"] == task_name)
        
    except Exception as e:
        raise RuntimeError(f"Error loading dataset for {task_name}: {e}")
    
    # Convert to lists to avoid numpy issues
    train_sequences = list(train_dataset["sequence"])
    train_labels = [int(label) for label in train_dataset["label"]]
    test_sequences = list(test_dataset["sequence"])
    test_labels = [int(label) for label in test_dataset["label"]]
    
    # Split training data into train/validation
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_sequences, train_labels, 
        test_size=val_split_ratio, 
        random_state=random_state
    )
    
    # Create dataset objects
    ds_train = Dataset.from_dict({"data": train_sequences, "labels": train_labels})
    ds_val = Dataset.from_dict({"data": val_sequences, "labels": val_labels})
    ds_test = Dataset.from_dict({"data": test_sequences, "labels": test_labels})
    
    # Tokenize if tokenizer is provided
    if tokenizer is not None:
        def tokenize_function(examples):
            return tokenizer(
                examples["data"], 
                truncation=True, 
                padding="max_length", 
                max_length=max_length
            )
        
        ds_train = ds_train.map(
            tokenize_function, batched=True, remove_columns=["data"]
        )
        ds_val = ds_val.map(
            tokenize_function, batched=True, remove_columns=["data"]
        )
        ds_test = ds_test.map(
            tokenize_function, batched=True, remove_columns=["data"]
        )
        
        # Set format for PyTorch
        ds_train.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        ds_val.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        ds_test.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
    
    return {
        "train": ds_train,
        "validation": ds_val, 
        "test": ds_test
    }


def load_multiple_tasks(
    task_names: List[str],
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = 512,
    val_split_ratio: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Dict[str, Dataset]]:
    """
    Load multiple tasks at once.
    
    Args:
        task_names: List of task names to load
        tokenizer: Tokenizer for sequence encoding (optional)
        max_length: Maximum sequence length for tokenization
        val_split_ratio: Ratio for validation split from training data
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping task names to their datasets
        
    Example:
        >>> all_datasets = load_multiple_tasks(["H3K27ac", "H3K4me3"])
        >>> h3k27ac_train = all_datasets["H3K27ac"]["train"]
    """
    datasets = {}
    for task_name in task_names:
        try:
            datasets[task_name] = load_nucleotide_task(
                task_name=task_name,
                tokenizer=tokenizer,
                max_length=max_length,
                val_split_ratio=val_split_ratio,
                random_state=random_state,
            )
        except Exception as e:
            print(f"Warning: Failed to load task {task_name}: {e}")
            continue
    
    return datasets


def get_available_tasks() -> List[str]:
    """
    Get list of available downstream tasks.
    
    Returns:
        List of task names that can be loaded
    """
    return [
        "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", 
        "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3", "H4K20me1",
        "promoter_all", "promoter_tata", "promoter_no_tata",
        "enhancers", "enhancers_types", "splice_sites_all",
        "splice_sites_acceptor", "splice_sites_donor",
    ]


class DNADataset:
    """
    Convenience class for working with DNA sequence datasets.
    
    This class provides a more object-oriented interface to the data loading
    functions, with additional utilities for dataset analysis.
    """
    
    def __init__(
        self,
        task_name: str,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        val_split_ratio: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize DNA dataset for a specific task.
        
        Args:
            task_name: Name of the downstream task
            tokenizer: Tokenizer for sequence encoding (optional)
            max_length: Maximum sequence length for tokenization
            val_split_ratio: Ratio for validation split from training data
            random_state: Random seed for reproducibility
        """
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_info = get_task_info(task_name)
        
        # Load datasets
        self._datasets = load_nucleotide_task(
            task_name=task_name,
            tokenizer=tokenizer,
            max_length=max_length,
            val_split_ratio=val_split_ratio,
            random_state=random_state,
        )
    
    @property
    def train(self) -> Dataset:
        """Training dataset."""
        return self._datasets["train"]
    
    @property
    def validation(self) -> Dataset:
        """Validation dataset."""
        return self._datasets["validation"]
    
    @property
    def test(self) -> Dataset:
        """Test dataset."""
        return self._datasets["test"]
    
    @property
    def num_labels(self) -> int:
        """Number of labels for this task."""
        return self.task_info["num_labels"]
    
    @property
    def task_type(self) -> str:
        """Type of task (classification, etc.)."""
        return self.task_info["task_type"]
    
    def get_dataset_info(self) -> Dict[str, Union[int, str]]:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            Dictionary with dataset statistics and metadata
        """
        info = {
            "task_name": self.task_name,
            "num_labels": self.num_labels,
            "task_type": self.task_type,
            "train_samples": len(self.train),
            "validation_samples": len(self.validation),
            "test_samples": len(self.test),
            "total_samples": len(self.train) + len(self.validation) + len(self.test),
        }
        
        if self.tokenizer:
            info["vocab_size"] = self.tokenizer.vocab_size
            info["max_length"] = self.max_length
            
        return info
    
    def print_info(self):
        """Print dataset information to console."""
        info = self.get_dataset_info()
        print(f"=== Dataset Info for {self.task_name} ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        print("=" * 40)
