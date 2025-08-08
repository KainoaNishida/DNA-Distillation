"""
Core distillation functions for knowledge distillation.

This module provides high-level distillation functions that integrate
teacher model loading, logits computation, and student training.
"""

import os
import wandb
from typing import Dict, Any, Optional, Union
from transformers import AutoTokenizer, TrainingArguments, PreTrainedModel
from datasets import Dataset

from ..data import load_nucleotide_task, get_num_labels
from ..models import create_student_model
from ..training.utils import get_default_training_args, validate_training_inputs
from .trainers import create_distillation_trainer
from .utils import (
    precompute_teacher_logits,
    find_best_teacher_checkpoint,
    load_teacher_model,
    get_distillation_config,
    get_distillation_cache_dir,
)


def distill_for_task(
    task_name: str,
    student_model_type: str,
    teacher_checkpoint_dir: str,
    output_dir: str,
    tokenizer_name: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    num_train_epochs: int = 10,
    alpha: float = 0.5,
    temperature: float = 2.0,
    precompute_teacher_logits: bool = True,
    cache_dir: Optional[str] = None,
    embed_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    wandb_project: str = "DNA-Distillation",
    wandb_run_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform knowledge distillation for a specific task.
    
    This function provides a complete distillation pipeline that:
    1. Loads the teacher model from checkpoint
    2. Loads and preprocesses the dataset
    3. Creates the student model
    4. Precomputes teacher logits (optional)
    5. Trains the student model with distillation
    
    Args:
        task_name: Name of the downstream task (e.g., "H3K27ac")
        student_model_type: Type of student model ("bilstm", "xlstm", "mamba", etc.)
        teacher_checkpoint_dir: Directory containing teacher checkpoints
        output_dir: Directory to save student model outputs
        tokenizer_name: HuggingFace model name for tokenizer
        num_train_epochs: Number of training epochs
        alpha: Weight for cross-entropy loss in distillation
        temperature: Temperature for logit scaling
        precompute_teacher_logits: Whether to precompute teacher logits
        cache_dir: Directory for caching teacher logits
        embed_dim: Embedding dimension (auto-determined if None)
        hidden_dim: Hidden dimension (auto-determined if None)
        wandb_project: WandB project name
        wandb_run_name: WandB run name (auto-generated if None)
        **kwargs: Additional arguments for training configuration
        
    Returns:
        Dictionary containing distillation results and model info
        
    Example:
        >>> results = distill_for_task(
        ...     task_name="H3K27ac",
        ...     student_model_type="bilstm",
        ...     teacher_checkpoint_dir="./teacher_checkpoints",
        ...     output_dir="./distilled_models",
        ...     num_train_epochs=5
        ... )
        >>> print(f"Final F1: {results['final_f1']}")
    """
    print(f"\n=== Distilling {student_model_type} on {task_name} ===")
    
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
        max_length=512
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
    
    # 4. Find and load teacher model
    print("Finding teacher checkpoint...")
    teacher_checkpoint = find_best_teacher_checkpoint(
        task_name=task_name,
        checkpoint_dir=teacher_checkpoint_dir
    )
    
    if teacher_checkpoint is None:
        raise ValueError(f"No teacher checkpoint found for task '{task_name}'")
    
    print(f"Loading teacher model from: {teacher_checkpoint}")
    teacher_model = load_teacher_model(
        checkpoint_path=teacher_checkpoint,
        num_labels=num_labels
    )
    
    # 5. Determine model dimensions
    if embed_dim is None:
        try:
            embed_dim = tokenizer.model_max_length
        except:
            embed_dim = 1280  # Default for nucleotide transformer
    
    if hidden_dim is None:
        hidden_dim = embed_dim
    
    # 6. Validate inputs
    validate_training_inputs(
        task_name=task_name,
        model_type=student_model_type,
        num_labels=num_labels,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    
    # 7. Create student model
    print(f"Creating {student_model_type} student model...")
    student_model = create_student_model(
        model_type=student_model_type,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels
    )
    
    print(f"Student model created with {sum(p.numel() for p in student_model.parameters()):,} parameters")
    
    # 8. Precompute teacher logits (if requested)
    teacher_logits = None
    if precompute_teacher_logits:
        print("Precomputing teacher logits...")
        
        if cache_dir is None:
            cache_dir = get_distillation_cache_dir(output_dir, task_name)
        
        cache_file = os.path.join(cache_dir, f"{task_name}_teacher_logits.pt")
        
        teacher_logits = precompute_teacher_logits(
            dataset=train_dataset,
            teacher_model=teacher_model,
            device=teacher_model.device,
            batch_size=kwargs.get("batch_size", 16),
            cache_file=cache_file
        )
    
    # 9. Set up WandB
    if wandb_run_name is None:
        wandb_run_name = f"{task_name}_{student_model_type}_distillation"
    
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        reinit=True,
        config={
            "task_name": task_name,
            "student_model_type": student_model_type,
            "teacher_checkpoint": teacher_checkpoint,
            "num_labels": num_labels,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_train_epochs": num_train_epochs,
            "alpha": alpha,
            "temperature": temperature,
            "precompute_logits": precompute_teacher_logits,
        }
    )
    
    # 10. Create training arguments
    training_args = get_default_training_args(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        **kwargs
    )
    
    # 11. Create distillation trainer
    trainer = create_distillation_trainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=TrainingArguments(**training_args),
        alpha=alpha,
        temperature=temperature,
        precomputed_logits=teacher_logits,
        task_name=task_name,
        num_labels=num_labels,
        **kwargs
    )
    
    # 12. Train the student model
    print("Starting distillation training...")
    train_result = trainer.train()
    
    # 13. Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.predict(test_dataset)
    eval_results = test_results.metrics
    
    # 14. Prepare results
    results = {
        "task_name": task_name,
        "student_model_type": student_model_type,
        "teacher_checkpoint": teacher_checkpoint,
        "train_result": train_result,
        "test_results": eval_results,
        "final_f1": eval_results.get("test_f1_score", 0.0),
        "final_mcc": eval_results.get("test_mcc_score", 0.0),
        "model_path": output_dir,
        "distillation_config": {
            "alpha": alpha,
            "temperature": temperature,
            "precomputed_logits": precompute_teacher_logits,
        },
        "config": {
            "num_labels": num_labels,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
        }
    }
    
    # 15. Log final results
    wandb.log({
        "final_test_f1": results["final_f1"],
        "final_test_mcc": results["final_mcc"],
    })
    
    wandb.finish()
    
    print(f"\n=== Distillation completed ===")
    print(f"Final test F1: {results['final_f1']:.4f}")
    print(f"Final test MCC: {results['final_mcc']:.4f}")
    print(f"Student model saved to: {output_dir}")
    
    return results


def distill_multiple_tasks(
    task_names: list,
    student_model_type: str,
    teacher_checkpoint_dir: str,
    output_base_dir: str,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Perform knowledge distillation on multiple tasks.
    
    Args:
        task_names: List of task names to distill
        student_model_type: Type of student model
        teacher_checkpoint_dir: Directory containing teacher checkpoints
        output_base_dir: Base directory for outputs
        **kwargs: Additional arguments for distill_for_task
        
    Returns:
        Dictionary mapping task names to distillation results
    """
    results = {}
    
    for task_name in task_names:
        print(f"\n{'='*60}")
        print(f"Distilling on task: {task_name}")
        print(f"{'='*60}")
        
        # Create task-specific output directory
        task_output_dir = os.path.join(output_base_dir, f"{task_name}_{student_model_type}_distilled")
        
        try:
            task_results = distill_for_task(
                task_name=task_name,
                student_model_type=student_model_type,
                teacher_checkpoint_dir=teacher_checkpoint_dir,
                output_dir=task_output_dir,
                **kwargs
            )
            results[task_name] = task_results
            
        except Exception as e:
            print(f"Error distilling on {task_name}: {e}")
            results[task_name] = {"error": str(e)}
    
    return results


class KnowledgeDistiller:
    """
    High-level knowledge distillation interface.
    
    This class provides a convenient interface for knowledge distillation
    with configuration management and result tracking.
    """
    
    def __init__(
        self,
        teacher_checkpoint_dir: str,
        tokenizer_name: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        cache_dir: Optional[str] = None,
        **config
    ):
        """
        Initialize the knowledge distiller.
        
        Args:
            teacher_checkpoint_dir: Directory containing teacher checkpoints
            tokenizer_name: HuggingFace model name for tokenizer
            cache_dir: Directory for caching teacher logits
            **config: Additional configuration parameters
        """
        self.teacher_checkpoint_dir = teacher_checkpoint_dir
        self.tokenizer_name = tokenizer_name
        self.cache_dir = cache_dir
        
        # Get default configuration
        self.config = get_distillation_config(**config)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def distill(
        self,
        task_name: str,
        student_model_type: str,
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform knowledge distillation for a task.
        
        Args:
            task_name: Name of the task
            student_model_type: Type of student model
            output_dir: Output directory
            **kwargs: Additional arguments
            
        Returns:
            Distillation results
        """
        # Merge configuration
        config = self.config.copy()
        config.update(kwargs)
        
        return distill_for_task(
            task_name=task_name,
            student_model_type=student_model_type,
            teacher_checkpoint_dir=self.teacher_checkpoint_dir,
            output_dir=output_dir,
            tokenizer_name=self.tokenizer_name,
            cache_dir=self.cache_dir,
            **config
        )
    
    def distill_multiple(
        self,
        task_names: list,
        student_model_type: str,
        output_base_dir: str,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform knowledge distillation on multiple tasks.
        
        Args:
            task_names: List of task names
            student_model_type: Type of student model
            output_base_dir: Base output directory
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of results
        """
        # Merge configuration
        config = self.config.copy()
        config.update(kwargs)
        
        return distill_multiple_tasks(
            task_names=task_names,
            student_model_type=student_model_type,
            teacher_checkpoint_dir=self.teacher_checkpoint_dir,
            output_base_dir=output_base_dir,
            **config
        ) 