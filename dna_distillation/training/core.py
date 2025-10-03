"""
Core training functions for DNA sequence models.

This module provides high-level training functions that integrate
data loading, model creation, and training in a single interface.
"""

import os
import wandb
from typing import Dict, Any, Optional, Union
from transformers import AutoTokenizer, TrainingArguments
from datasets import Dataset

from ..data import load_nucleotide_task, get_num_labels
from ..models import create_student_model
from .trainers import DNATrainer
from .utils import validate_training_inputs, get_default_training_args


def train_for_task(
    task_name: str,
    model_type: str,
    output_dir: str,
    tokenizer_name: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 64,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    max_length: int = 512,
    embed_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
    wandb_project: str = "DNA-Distillation",
    wandb_run_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a student model on a specific DNA sequence task.
    
    This function provides a complete training pipeline that:
    1. Loads and preprocesses the dataset
    2. Creates the student model
    3. Sets up training configuration
    4. Trains the model with proper logging and evaluation
    
    Args:
        task_name: Name of the downstream task (e.g., "H3K27ac")
        model_type: Type of student model ("bilstm", "xlstm", "mamba", etc.)
        output_dir: Directory to save model outputs and checkpoints
        tokenizer_name: HuggingFace model name for tokenizer
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        max_length: Maximum sequence length for tokenization
        embed_dim: Embedding dimension (auto-determined if None)
        hidden_dim: Hidden dimension (auto-determined if None)
        resume_from_checkpoint: Checkpoint to resume from
        wandb_project: WandB project name
        wandb_run_name: WandB run name (auto-generated if None)
        **kwargs: Additional arguments for training configuration
        
    Returns:
        Dictionary containing training results and model info
        
    Example:
        >>> results = train_for_task(
        ...     task_name="H3K27ac",
        ...     model_type="bilstm",
        ...     output_dir="./outputs",
        ...     num_train_epochs=5
        ... )
        >>> print(f"Final F1: {results['final_f1']}")
    """
    print(f"\n=== Training {model_type} on {task_name} ===")
    
    # 1. Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer loaded (vocab size: {vocab_size})")
    
    # 2. Load dataset
    print("Loading dataset...")
    datasets = load_nucleotide_task(
        task_name=task_name,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]
    test_dataset = datasets["test"]
    
    print(f"Dataset loaded:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # 3. Get task info
    num_labels = get_num_labels(task_name)
    print(f"Task: {task_name} (num_labels: {num_labels})")
    
    # 4. Determine model dimensions
    if embed_dim is None:
        # Try to get embedding dimension from tokenizer or use default
        try:
            embed_dim = tokenizer.model_max_length
        except:
            embed_dim = 1280  # Default for nucleotide transformer
    
    if hidden_dim is None:
        hidden_dim = embed_dim
    
    # 5. Validate inputs
    validate_training_inputs(
        task_name=task_name,
        model_type=model_type,
        num_labels=num_labels,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    
    # 6. Create model
    print(f"Creating {model_type} model...")
    model = create_student_model(
        model_type=model_type,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 7. Set up WandB
    if wandb_run_name is None:
        wandb_run_name = f"{task_name}_{model_type}"
    
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        reinit=True,
        config={
            "task_name": task_name,
            "model_type": model_type,
            "num_labels": num_labels,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "max_length": max_length,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
    )
    
    # 8. Create training arguments
    training_args = get_default_training_args(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        **kwargs
    )
    
    # 9. Create trainer
    trainer = DNATrainer(
        model=model,
        args=TrainingArguments(**training_args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        task_name=task_name,
        model_type=model_type,
        output_dir=output_dir,
    )
    
    # 10. Train the model
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 11. Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate_test()
    
    # 12. Prepare results
    results = {
        "task_name": task_name,
        "model_type": model_type,
        "train_result": train_result,
        "test_results": test_results,
        "final_f1": test_results.get("test_f1_score", 0.0),
        "final_mcc": test_results.get("test_mcc_score", 0.0),
        "model_path": output_dir,
        "config": {
            "num_labels": num_labels,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
        }
    }
    
    # 13. Log final results
    wandb.log({
        "final_test_f1": results["final_f1"],
        "final_test_mcc": results["final_mcc"],
    })
    
    wandb.finish()
    
    print(f"\n=== Training completed ===")
    print(f"Final test F1: {results['final_f1']:.4f}")
    print(f"Final test MCC: {results['final_mcc']:.4f}")
    print(f"Model saved to: {output_dir}")
    
    return results


def train_multiple_tasks(
    task_names: list,
    model_type: str,
    output_base_dir: str,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Train a model on multiple tasks.
    
    Args:
        task_names: List of task names to train on
        model_type: Type of student model
        output_base_dir: Base directory for outputs
        **kwargs: Additional arguments for train_for_task
        
    Returns:
        Dictionary mapping task names to training results
    """
    results = {}
    
    for task_name in task_names:
        print(f"\n{'='*60}")
        print(f"Training on task: {task_name}")
        print(f"{'='*60}")
        
        # Create task-specific output directory
        task_output_dir = os.path.join(output_base_dir, f"{task_name}_{model_type}")
        
        try:
            task_results = train_for_task(
                task_name=task_name,
                model_type=model_type,
                output_dir=task_output_dir,
                **kwargs
            )
            results[task_name] = task_results
            
        except Exception as e:
            print(f"Error training on {task_name}: {e}")
            results[task_name] = {"error": str(e)}
    
    return results


def train_teacher_model(
    task_name: str,
    teacher_model_type: str = "nucleotide_transformer_500m",
    datasets: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    tokenizer_name: Optional[str] = None,
    output_dir: str = "./teacher_models"
) -> Dict[str, Any]:
    """
    Train a teacher model on a DNA sequence task.
    
    Args:
        task_name: Name of the DNA sequence task
        teacher_model_type: Type of teacher model to train (nucleotide_transformer_500m, nucleotide_transformer_2b5, custom)
        datasets: Pre-loaded datasets (optional)
        training_config: Training configuration (optional)
        tokenizer_name: Name of the tokenizer to use (auto-detected if None)
        output_dir: Directory to save the trained model
        
    Returns:
        Training results dictionary
    """
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
    from datasets import load_dataset
    import torch
    import os
    
    # Define model configurations
    model_configs = {
        "nucleotide_transformer_500m": {
            "model_name": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
            "description": "500M parameter Nucleotide Transformer (Human Reference)",
            "memory_requirement": "~8GB GPU memory"
        },
        "nucleotide_transformer_2b5": {
            "model_name": "InstaDeepAI/nucleotide-transformer-2.5b-multi-species", 
            "description": "2.5B parameter Nucleotide Transformer (Multi-species)",
            "memory_requirement": "~24GB GPU memory"
        }
    }
    
    # Auto-detect tokenizer name if not provided
    if tokenizer_name is None:
        if teacher_model_type in model_configs:
            tokenizer_name = model_configs[teacher_model_type]["model_name"]
        else:
            raise ValueError(f"Unknown teacher model type: {teacher_model_type}. Available: {list(model_configs.keys())}")
    
    # Load datasets if not provided
    if datasets is None:
        datasets = load_nucleotide_task(
            task_name=task_name,
            tokenizer_name=tokenizer_name,
            max_length=training_config.get("max_length", 512) if training_config else 512
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Get number of labels
    num_labels = get_num_labels(task_name)
    
    # Load teacher model
    if teacher_model_type in model_configs:
        model_name = model_configs[teacher_model_type]["model_name"]
        print(f"Loading {model_configs[teacher_model_type]['description']}")
        print(f"Memory requirement: {model_configs[teacher_model_type]['memory_requirement']}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            trust_remote_code=True
        )
    else:
        raise ValueError(f"Unsupported teacher model type: {teacher_model_type}. Available: {list(model_configs.keys())}")
    
    # Default training configuration
    if training_config is None:
        training_config = {
            "num_train_epochs": 10,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "warmup_steps": 1000,
            "fp16": True,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1_score",
            "output_dir": output_dir,
            "logging_steps": 50,
        }
    
    # Create training arguments
    training_args = TrainingArguments(**training_config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Add early stopping if specified
    if training_config.get("early_stopping_patience", 0) > 0:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=training_config["early_stopping_patience"]
        )
        trainer.add_callback(early_stopping)
    
    # Train the model
    print(f"Training teacher model on {task_name}...")
    train_results = trainer.train()
    
    # Evaluate on test set if available
    test_results = {}
    if "test" in datasets:
        test_results = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Compile results
    results = {
        "task_name": task_name,
        "teacher_model_type": teacher_model_type,
        "train_results": train_results,
        "test_results": test_results,
        "model_path": output_dir,
        "num_labels": num_labels,
    }
    
    return results 