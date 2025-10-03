"""
Distillation utilities for knowledge distillation.

This module provides utilities for teacher logits computation,
checkpoint management, and other distillation-related functionality.
"""

import os
import re
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset
from tqdm import tqdm
import json


def precompute_teacher_logits(
    dataset: Union[Dataset, HFDataset],
    teacher_model: PreTrainedModel,
    device: torch.device,
    batch_size: int = 16,
    cache_file: Optional[str] = None,
    num_workers: int = 4
) -> torch.Tensor:
    """
    Precompute teacher logits for a dataset.
    
    This function computes teacher logits offline to reduce GPU memory
    usage during student training. Results can be cached for reuse.
    
    Args:
        dataset: Dataset to compute logits for
        teacher_model: Teacher model to use for logit computation
        device: Device to run computation on
        batch_size: Batch size for computation
        cache_file: Optional cache file path to save/load logits
        num_workers: Number of workers for data loading
        
    Returns:
        Precomputed teacher logits
    """
    # Check if cached logits exist
    if cache_file is not None and os.path.exists(cache_file):
        print(f"Loading precomputed teacher logits from cache: {cache_file}")
        teacher_logits = torch.load(cache_file, map_location=device)
        return teacher_logits
    
    # Set teacher model to evaluation mode
    teacher_model.eval()
    teacher_model.to(device)
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    logits_list = []
    
    print(f"Precomputing teacher logits for {len(dataset)} samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Compute teacher logits
            outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu()
            logits_list.append(logits)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(loader)} batches")
    
    # Concatenate all logits
    teacher_logits = torch.cat(logits_list, dim=0)
    
    # Save to cache if requested
    if cache_file is not None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(teacher_logits, cache_file)
        print(f"Saved teacher logits to cache: {cache_file}")
    
    print(f"Precomputed teacher logits shape: {teacher_logits.shape}")
    return teacher_logits


def find_best_teacher_checkpoint(
    task_name: str,
    checkpoint_dir: str,
    metric: str = "eval_f1_score"
) -> Optional[str]:
    """
    Find the best teacher checkpoint for a given task.
    
    This function scans a directory for teacher checkpoints and
    returns the path to the best performing checkpoint.
    
    Args:
        task_name: Name of the task
        checkpoint_dir: Directory containing teacher checkpoints
        metric: Metric to use for selecting best checkpoint
        
    Returns:
        Path to best checkpoint, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    # Look for task-specific checkpoints
    task_pattern = re.compile(rf".*{re.escape(task_name)}.*")
    checkpoint_paths = []
    
    for checkpoint_path in checkpoint_dir.rglob("*"):
        if checkpoint_path.is_dir() and task_pattern.search(checkpoint_path.name):
            # Check if this is a valid checkpoint directory
            if (checkpoint_path / "pytorch_model.bin").exists() or \
               (checkpoint_path / "adapter_config.json").exists():
                checkpoint_paths.append(checkpoint_path)
    
    if not checkpoint_paths:
        print(f"No checkpoints found for task '{task_name}' in {checkpoint_dir}")
        return None
    
    # If multiple checkpoints found, try to find the best one
    if len(checkpoint_paths) > 1:
        print(f"Found {len(checkpoint_paths)} checkpoints for task '{task_name}'")
        
        # Look for checkpoint with best metric in name or config
        best_checkpoint = None
        best_score = -1
        
        for checkpoint_path in checkpoint_paths:
            # Try to extract score from checkpoint name
            checkpoint_name = checkpoint_path.name
            
            # Look for metric pattern in name (e.g., "f1_0.85", "eval_f1_0.92")
            metric_pattern = re.compile(rf"{metric.replace('eval_', '')}_(\d+\.\d+)")
            match = metric_pattern.search(checkpoint_name)
            
            if match:
                score = float(match.group(1))
                if score > best_score:
                    best_score = score
                    best_checkpoint = str(checkpoint_path)
        
        if best_checkpoint:
            print(f"Selected best checkpoint: {best_checkpoint} (score: {best_score})")
            return best_checkpoint
    
    # Return the first (and likely only) checkpoint found
    selected_checkpoint = str(checkpoint_paths[0])
    print(f"Selected checkpoint: {selected_checkpoint}")
    return selected_checkpoint


def load_teacher_model(
    checkpoint_path: str,
    num_labels: int,
    device: Optional[torch.device] = None
) -> PreTrainedModel:
    """
    Load a teacher model from checkpoint.
    
    Args:
        checkpoint_path: Path to teacher model checkpoint
        num_labels: Number of output labels
        device: Device to load model on
        
    Returns:
        Loaded teacher model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading teacher model from: {checkpoint_path}")
    
    try:
        # Try loading as a regular model first
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=num_labels,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        print("Trying alternative loading method...")
        
        # Try loading with different parameters
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=num_labels,
            trust_remote_code=False
        )
    
    teacher_model.to(device)
    teacher_model.eval()
    
    print(f"Teacher model loaded successfully on {device}")
    return teacher_model


def prepare_dataset_for_distillation(
    dataset: HFDataset,
    tokenizer,
    max_length: int = 512
) -> HFDataset:
    """
    Prepare dataset for distillation by adding teacher logits.
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        
    Returns:
        Prepared dataset with teacher logits
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Rename label column for compatibility
    if "label" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    # Set format for PyTorch
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    
    return tokenized_dataset


def validate_distillation_setup(
    teacher_model: PreTrainedModel,
    student_model: nn.Module,
    dataset: Union[Dataset, HFDataset],
    task_name: str,
    num_labels: int
) -> None:
    """
    Validate distillation setup before training.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        dataset: Training dataset
        task_name: Name of the task
        num_labels: Number of output labels
        
    Raises:
        ValueError: If setup is invalid
    """
    # Check models
    if teacher_model is None:
        raise ValueError("Teacher model is required for distillation")
    
    if student_model is None:
        raise ValueError("Student model is required for distillation")
    
    # Check dataset
    if dataset is None or len(dataset) == 0:
        raise ValueError("Dataset is required and must not be empty")
    
    # Check task name
    if not task_name or not isinstance(task_name, str):
        raise ValueError("Task name must be a non-empty string")
    
    # Check number of labels
    if num_labels < 2:
        raise ValueError("Number of labels must be at least 2")
    
    # Check model compatibility
    teacher_output_dim = teacher_model.config.num_labels
    student_output_dim = student_model.classifier.out_features if hasattr(student_model, 'classifier') else None
    
    if student_output_dim and teacher_output_dim != student_output_dim:
        raise ValueError(f"Teacher and student models must have same output dimension. "
                        f"Teacher: {teacher_output_dim}, Student: {student_output_dim}")
    
    print(" Distillation setup validation passed")


def get_distillation_cache_dir(base_dir: str, task_name: str) -> str:
    """
    Get cache directory for distillation artifacts.
    
    Args:
        base_dir: Base directory for caching
        task_name: Name of the task
        
    Returns:
        Cache directory path
    """
    cache_dir = os.path.join(base_dir, "teacher_logits_cache", task_name)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_distillation_config(
    alpha: float = 0.5,
    temperature: float = 2.0,
    cache_teacher_logits: bool = True,
    batch_size: int = 16,
    **kwargs
) -> Dict[str, Any]:
    """
    Get default distillation configuration.
    
    Args:
        alpha: Weight for cross-entropy loss
        temperature: Temperature for logit scaling
        cache_teacher_logits: Whether to cache teacher logits
        batch_size: Batch size for teacher logit computation
        **kwargs: Additional configuration parameters
        
    Returns:
        Distillation configuration dictionary
    """
    config = {
        "alpha": alpha,
        "temperature": temperature,
        "cache_teacher_logits": cache_teacher_logits,
        "batch_size": batch_size,
        "loss_type": "combined",
        "precompute_logits": True,
        "teacher_device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    # Add any additional kwargs
    config.update(kwargs)
    
    return config


# ============================================================================
# ADVANCED FEATURES FROM LATEST RESEARCH CODE
# ============================================================================

def precompute_teacher_features(
    dataset: Union[Dataset, HFDataset],
    teacher_model: PreTrainedModel,
    device: torch.device,
    batch_size: int = 16,
    cache_file: Optional[str] = None,
    feature_layers: Optional[List[int]] = None,
    use_mixed_precision: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Precompute teacher features from multiple layers for multi-level distillation.
    
    This function extracts intermediate features from teacher model layers
    for use in ReviewKD and other multi-level distillation methods.
    
    Args:
        dataset: Dataset to precompute features for
        teacher_model: Teacher model to use
        device: Device to run computation on
        batch_size: Batch size for computation
        cache_file: Optional cache file to save/load features
        feature_layers: List of layer indices to extract features from
        use_mixed_precision: Whether to use mixed precision (FP16)
        
    Returns:
        Dictionary with precomputed features from each layer
    """
    if cache_file is not None and os.path.exists(cache_file):
        print(f"Loading precomputed teacher features from {cache_file}")
        return torch.load(cache_file, map_location=device)
    
    teacher_model.eval()
    features_dict = {}
    
    if feature_layers is None:
        # Default to last 3 layers
        feature_layers = [-3, -2, -1]
    
    # Initialize feature lists for each layer
    for layer_idx in feature_layers:
        features_dict[f"layer_{layer_idx}"] = []
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"Precomputing teacher features for {len(dataset)} samples...")
    print(f"Extracting features from layers: {feature_layers}")
    print(f"Using mixed precision: {use_mixed_precision}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing teacher features"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            # Get teacher features with mixed precision if enabled
            if use_mixed_precision and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = teacher_model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
            else:
                outputs = teacher_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Extract features from specified layers
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
                for layer_idx in feature_layers:
                    layer_features = hidden_states[layer_idx].cpu()
                    features_dict[f"layer_{layer_idx}"].append(layer_features)
            else:
                # Fallback: use logits as features
                logits = outputs.logits.cpu()
                for layer_idx in feature_layers:
                    features_dict[f"layer_{layer_idx}"].append(logits)
    
    # Concatenate features for each layer
    for layer_idx in feature_layers:
        features_dict[f"layer_{layer_idx}"] = torch.cat(features_dict[f"layer_{layer_idx}"], dim=0)
    
    # Save to cache if specified
    if cache_file is not None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(features_dict, cache_file)
        print(f"Saved precomputed teacher features to {cache_file}")
    
    return features_dict


def create_hierarchical_checkpoint_dir(
    checkpoint_root: str,
    task_name: str,
    method_name: str,
    hyperparams: Dict[str, Any]
) -> str:
    """
    Create hierarchical checkpoint directory structure from latest research.
    
    Creates directory structure: checkpoint_root/task_name/method_name/hyperparams/
    
    Args:
        checkpoint_root: Root directory for checkpoints
        task_name: Name of the task
        method_name: Name of the distillation method
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        Path to the checkpoint directory
    """
    # Create hyperparameter string
    hyperparam_parts = []
    for key, value in sorted(hyperparams.items()):
        if isinstance(value, float):
            hyperparam_parts.append(f"{key}{value:.1e}")
        else:
            hyperparam_parts.append(f"{key}{value}")
    hyperparam_str = "_".join(hyperparam_parts)
    
    # Create directory path
    checkpoint_dir = os.path.join(checkpoint_root, task_name, method_name, hyperparam_str)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration file
    config_file = os.path.join(checkpoint_dir, "config.txt")
    with open(config_file, "w") as f:
        f.write(f"task_name={task_name}\n")
        f.write(f"method_name={method_name}\n")
        f.write(f"hyperparams={hyperparam_str}\n")
        import datetime
        f.write(f"timestamp={datetime.datetime.now().isoformat()}\n")
        f.write("\n# Full hyperparameters:\n")
        for key, value in hyperparams.items():
            f.write(f"{key}={value}\n")
    
    print(f"Created checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir


def hyperparameter_search(
    task_name: str,
    method_name: str,
    search_space: Dict[str, List[Any]],
    base_config: Dict[str, Any],
    max_trials: int = 20
) -> List[Dict[str, Any]]:
    """
    Perform hyperparameter search for distillation methods.
    
    Args:
        task_name: Name of the task
        method_name: Name of the distillation method
        search_space: Dictionary defining search space for each hyperparameter
        base_config: Base configuration to extend
        max_trials: Maximum number of trials to run
        
    Returns:
        List of hyperparameter configurations to try
    """
    import itertools
    import random
    
    # Generate all combinations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    
    all_combinations = list(itertools.product(*param_values))
    
    # Randomly sample if too many combinations
    if len(all_combinations) > max_trials:
        all_combinations = random.sample(all_combinations, max_trials)
    
    # Create configurations
    configurations = []
    for combination in all_combinations:
        config = base_config.copy()
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value
        config["task_name"] = task_name
        config["method_name"] = method_name
        configurations.append(config)
    
    print(f"Generated {len(configurations)} hyperparameter configurations for {method_name}")
    return configurations


def get_method_hyperparameters(method_name: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for distillation methods from latest research.
    
    Args:
        method_name: Name of the distillation method
        
    Returns:
        Dictionary of default hyperparameters
    """
    method_configs = {
        'vanilla': {
            'temperature': 4.0,
            'lambda_kl': 0.3,
            'lambda_mse': 0.2,
        },
        'logit_standard': {
            'temperature': 4.0,
            'lambda_kl': 0.3,
            'lambda_mse': 0.2,
        },
        'dkd': {
            'temperature': 4.0,
            'lambda_kl': 0.3,
            'lambda_mse': 0.0,  # DKD typically doesn't use MSE
            'alpha': 1.0,
            'beta': 8.0,
        },
        'dist': {
            'lambda_kl': 0.3,  # DIST replaces KL with correlation loss
            'lambda_mse': 0.0,  # DIST doesn't use MSE
            'beta': 2.0,
            'gamma': 2.0,
        },
        'reviewkd': {
            'temperature': 4.0,
            'lambda_kl': 0.3,
            'lambda_mse': 0.0,  # ReviewKD uses its own feature fusion
        }
    }
    
    return method_configs.get(method_name, {})


def setup_mixed_precision_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> torch.cuda.amp.GradScaler:
    """
    Setup mixed precision training for faster training.
    
    Args:
        model: Model to train
        optimizer: Optimizer to use
        scaler: Optional existing scaler
        
    Returns:
        GradScaler for mixed precision training
    """
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler()
    
    print("Mixed precision training enabled (FP16)")
    return scaler


def create_advanced_distillation_config(
    method: str,
    task_name: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Create advanced distillation configuration from latest research.
    
    Args:
        method: Distillation method name
        task_name: Name of the task
        **kwargs: Additional configuration parameters
        
    Returns:
        Complete distillation configuration
    """
    # Get method-specific hyperparameters
    method_hparams = get_method_hyperparameters(method)
    
    # Base configuration
    config = {
        "method": method,
        "task_name": task_name,
        "temperature": 4.0,
        "lambda_kl": 0.3,
        "lambda_mse": 0.2,
        "use_mixed_precision": True,
        "precompute_logits": True,
        "precompute_features": method in ["reviewkd"],
        "feature_layers": [-3, -2, -1] if method in ["reviewkd"] else None,
        "hierarchical_checkpoints": True,
        "hyperparameter_search": False,
    }
    
    # Add method-specific parameters
    config.update(method_hparams)
    
    # Add any additional kwargs
    config.update(kwargs)
    
    return config