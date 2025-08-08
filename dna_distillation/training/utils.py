"""
Training utilities for DNA sequence models.

This module provides metrics computation, parameter counting, and other
training-related utilities extracted from the research codebase.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, matthews_corrcoef
from typing import Dict, Any, Optional
from transformers import EvalPrediction


def compute_metrics_f1(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute F1 score for evaluation predictions.
    
    Args:
        eval_pred: Evaluation predictions from HuggingFace Trainer
        
    Returns:
        Dictionary containing F1 score
    """
    predictions = np.argmax(eval_pred.predictions, axis=-1).flatten()
    references = np.array(eval_pred.label_ids).flatten()
    return {"f1_score": f1_score(references, predictions, average="macro")}


def compute_metrics_mcc(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute Matthews Correlation Coefficient for evaluation predictions.
    
    Args:
        eval_pred: Evaluation predictions from HuggingFace Trainer
        
    Returns:
        Dictionary containing MCC score
    """
    predictions = np.argmax(eval_pred.predictions, axis=-1).flatten()
    references = np.array(eval_pred.label_ids).flatten()
    return {"mcc_score": matthews_corrcoef(references, predictions)}


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute both F1 and MCC metrics for evaluation predictions.
    
    Args:
        eval_pred: Evaluation predictions from HuggingFace Trainer
        
    Returns:
        Dictionary containing both F1 and MCC scores
    """
    metrics = {}
    metrics.update(compute_metrics_f1(eval_pred))
    metrics.update(compute_metrics_mcc(eval_pred))
    return metrics


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get the model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def get_training_summary(model: torch.nn.Module, task_name: str, model_type: str) -> Dict[str, Any]:
    """
    Get a comprehensive summary of model and training information.
    
    Args:
        model: PyTorch model
        task_name: Name of the task being trained on
        model_type: Type of model architecture
        
    Returns:
        Dictionary with model and training summary
    """
    return {
        "task_name": task_name,
        "model_type": model_type,
        "total_parameters": count_parameters(model),
        "model_size_mb": get_model_size_mb(model),
        "device": next(model.parameters()).device,
        "dtype": next(model.parameters()).dtype,
    }


def validate_training_inputs(
    task_name: str,
    model_type: str,
    num_labels: int,
    vocab_size: int,
    embed_dim: int,
    hidden_dim: int,
) -> None:
    """
    Validate training inputs before starting training.
    
    Args:
        task_name: Name of the task
        model_type: Type of model architecture
        num_labels: Number of output labels
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension
        
    Raises:
        ValueError: If any inputs are invalid
    """
    if not task_name or not isinstance(task_name, str):
        raise ValueError("task_name must be a non-empty string")
    
    if not model_type or not isinstance(model_type, str):
        raise ValueError("model_type must be a non-empty string")
    
    if num_labels < 2:
        raise ValueError("num_labels must be at least 2")
    
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    
    if embed_dim < 1:
        raise ValueError("embed_dim must be positive")
    
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be positive")


def create_training_config(
    output_dir: str,
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 64,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    logging_steps: int = 100,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    metric_for_best_model: str = "f1_score",
    load_best_model_at_end: bool = True,
    report_to: Optional[list] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a training configuration dictionary.
    
    Args:
        output_dir: Directory to save model outputs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        learning_rate: Learning rate
        weight_decay: Weight decay
        logging_steps: Logging frequency
        evaluation_strategy: Evaluation strategy
        save_strategy: Model saving strategy
        metric_for_best_model: Metric to use for best model selection
        load_best_model_at_end: Whether to load best model at end
        report_to: List of reporting integrations
        **kwargs: Additional arguments
        
    Returns:
        Training configuration dictionary
    """
    if report_to is None:
        report_to = ["wandb"]
    
    config = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "logging_steps": logging_steps,
        "evaluation_strategy": evaluation_strategy,
        "save_strategy": save_strategy,
        "metric_for_best_model": metric_for_best_model,
        "load_best_model_at_end": load_best_model_at_end,
        "report_to": report_to,
    }
    
    # Add any additional kwargs
    config.update(kwargs)
    
    return config


def get_default_training_args(output_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Get default training arguments for DNA sequence models.
    
    Args:
        output_dir: Directory to save model outputs
        **kwargs: Override any default arguments
        
    Returns:
        Default training configuration
    """
    defaults = {
        "num_train_epochs": 10,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 64,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "metric_for_best_model": "f1_score",
        "load_best_model_at_end": True,
        "report_to": ["wandb"],
    }
    
    # Override with any provided kwargs
    defaults.update(kwargs)
    
    return create_training_config(output_dir=output_dir, **defaults)
