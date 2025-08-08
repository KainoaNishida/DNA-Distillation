"""
Distillation trainers for knowledge distillation.

This module provides specialized trainer classes for knowledge
distillation from teacher models to student models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union
from transformers import Trainer, TrainingArguments, PreTrainedModel
from datasets import Dataset

from .losses import DistillationLoss, get_distillation_loss
from .utils import precompute_teacher_logits, validate_distillation_setup


class DistillationTrainer(Trainer):
    """
    Custom trainer for knowledge distillation.
    
    This trainer extends the HuggingFace Trainer with knowledge distillation
    functionality, supporting both precomputed and on-the-fly teacher logits.
    """
    
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        alpha: float = 0.5,
        temperature: float = 2.0,
        precomputed_teacher_logits: Optional[torch.Tensor] = None,
        teacher_device: Optional[torch.device] = None,
        *args,
        **kwargs
    ):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher_model: Teacher model for distillation
            alpha: Weight for cross-entropy loss
            temperature: Temperature for logit scaling
            precomputed_teacher_logits: Precomputed teacher logits (optional)
            teacher_device: Device for teacher model
            *args: Additional arguments for Trainer
            **kwargs: Additional keyword arguments for Trainer
        """
        super().__init__(*args, **kwargs)
        
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.precomputed_teacher_logits = precomputed_teacher_logits
        
        # Set up teacher model
        if teacher_device is None:
            teacher_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.teacher_device = teacher_device
        self.teacher_model.to(teacher_device)
        self.teacher_model.eval()
        
        # Create distillation loss function
        self.distillation_loss = DistillationLoss(
            alpha=alpha,
            temperature=temperature
        )
        
        # Log distillation configuration
        if hasattr(self, 'log') and self.log:
            self.log({
                "distillation_alpha": alpha,
                "distillation_temperature": temperature,
                "teacher_model": type(teacher_model).__name__,
                "precomputed_logits": precomputed_teacher_logits is not None,
            })
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute distillation loss.
        
        Args:
            model: Student model
            inputs: Input batch
            return_outputs: Whether to return outputs
            
        Returns:
            Loss value or (loss, outputs) tuple
        """
        labels = inputs.get("labels")
        
        # Get student outputs
        outputs = model(**inputs)
        student_logits = outputs.get("logits")
        
        # Get teacher logits
        if self.precomputed_teacher_logits is not None:
            # Use precomputed logits
            batch_idx = inputs.get("batch_idx", 0)
            batch_size = student_logits.shape[0]
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            teacher_logits = self.precomputed_teacher_logits[start_idx:end_idx].to(student_logits.device)
        else:
            # Compute teacher logits on-the-fly
            with torch.no_grad():
                teacher_inputs = {
                    "input_ids": inputs["input_ids"].to(self.teacher_device),
                    "attention_mask": inputs["attention_mask"].to(self.teacher_device)
                }
                teacher_outputs = self.teacher_model(**teacher_inputs)
                teacher_logits = teacher_outputs.get("logits").to(student_logits.device)
        
        # Compute distillation loss
        loss = self.distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels
        )
        
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Get training data loader with teacher logits if precomputed.
        
        Returns:
            Training data loader
        """
        dataloader = super().get_train_dataloader()
        
        if self.precomputed_teacher_logits is not None:
            # Add batch indices to help with logit indexing
            for i, batch in enumerate(dataloader.dataset):
                batch["batch_idx"] = i
        
        return dataloader
    
    def log_distillation_metrics(self, loss_components: Dict[str, float]):
        """
        Log distillation-specific metrics.
        
        Args:
            loss_components: Dictionary of loss components
        """
        if hasattr(self, 'log') and self.log:
            self.log(loss_components)


class PrecomputedDistillationTrainer(DistillationTrainer):
    """
    Distillation trainer optimized for precomputed teacher logits.
    
    This trainer is designed for scenarios where teacher logits are
    precomputed and cached to save memory and computation time.
    """
    
    def __init__(
        self,
        teacher_logits_cache: torch.Tensor,
        *args,
        **kwargs
    ):
        """
        Initialize with precomputed teacher logits.
        
        Args:
            teacher_logits_cache: Precomputed teacher logits
            *args: Additional arguments for DistillationTrainer
            **kwargs: Additional keyword arguments for DistillationTrainer
        """
        super().__init__(
            precomputed_teacher_logits=teacher_logits_cache,
            *args,
            **kwargs
        )
        
        print(f"Using precomputed teacher logits: {teacher_logits_cache.shape}")
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute loss using precomputed teacher logits.
        
        Args:
            model: Student model
            inputs: Input batch
            return_outputs: Whether to return outputs
            
        Returns:
            Loss value or (loss, outputs) tuple
        """
        labels = inputs.get("labels")
        batch_idx = inputs.get("batch_idx", 0)
        
        # Get student outputs
        outputs = model(**inputs)
        student_logits = outputs.get("logits")
        
        # Get precomputed teacher logits for this batch
        batch_size = student_logits.shape[0]
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        teacher_logits = self.precomputed_teacher_logits[start_idx:end_idx].to(student_logits.device)
        
        # Compute distillation loss
        loss, components = self.distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            return_components=True
        )
        
        # Log distillation metrics
        self.log_distillation_metrics(components)
        
        return (loss, outputs) if return_outputs else loss


class OnlineDistillationTrainer(DistillationTrainer):
    """
    Distillation trainer that computes teacher logits on-the-fly.
    
    This trainer is suitable when memory allows for keeping both
    teacher and student models in memory simultaneously.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize online distillation trainer.
        
        Args:
            *args: Additional arguments for DistillationTrainer
            **kwargs: Additional keyword arguments for DistillationTrainer
        """
        super().__init__(precomputed_teacher_logits=None, *args, **kwargs)
        
        print("Using online teacher logits computation")
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute loss with on-the-fly teacher logits.
        
        Args:
            model: Student model
            inputs: Input batch
            return_outputs: Whether to return outputs
            
        Returns:
            Loss value or (loss, outputs) tuple
        """
        labels = inputs.get("labels")
        
        # Get student outputs
        outputs = model(**inputs)
        student_logits = outputs.get("logits")
        
        # Compute teacher logits on-the-fly
        with torch.no_grad():
            teacher_inputs = {
                "input_ids": inputs["input_ids"].to(self.teacher_device),
                "attention_mask": inputs["attention_mask"].to(self.teacher_device)
            }
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.get("logits").to(student_logits.device)
        
        # Compute distillation loss
        loss, components = self.distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            return_components=True
        )
        
        # Log distillation metrics
        self.log_distillation_metrics(components)
        
        return (loss, outputs) if return_outputs else loss


def create_distillation_trainer(
    teacher_model: PreTrainedModel,
    student_model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments,
    alpha: float = 0.5,
    temperature: float = 2.0,
    precomputed_logits: Optional[torch.Tensor] = None,
    teacher_device: Optional[torch.device] = None,
    **kwargs
) -> DistillationTrainer:
    """
    Create a distillation trainer with the specified configuration.
    
    Args:
        teacher_model: Teacher model for distillation
        student_model: Student model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Training arguments
        alpha: Weight for cross-entropy loss
        temperature: Temperature for logit scaling
        precomputed_logits: Precomputed teacher logits (optional)
        teacher_device: Device for teacher model
        **kwargs: Additional arguments for trainer
        
    Returns:
        Configured distillation trainer
    """
    # Validate setup
    validate_distillation_setup(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset=train_dataset,
        task_name=kwargs.get("task_name", "unknown"),
        num_labels=kwargs.get("num_labels", 2)
    )
    
    # Choose trainer type based on precomputed logits
    if precomputed_logits is not None:
        trainer_class = PrecomputedDistillationTrainer
        trainer_kwargs = {"teacher_logits_cache": precomputed_logits}
    else:
        trainer_class = OnlineDistillationTrainer
        trainer_kwargs = {}
    
    # Create trainer
    trainer = trainer_class(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        alpha=alpha,
        temperature=temperature,
        teacher_device=teacher_device,
        **trainer_kwargs,
        **kwargs
    )
    
    return trainer 