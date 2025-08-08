"""
Custom trainers for DNA sequence models.

This module provides specialized trainer classes and callbacks for
training DNA sequence models with proper evaluation and logging.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from typing import Dict, Any, Optional, List
from datasets import Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from .utils import compute_metrics, count_parameters, get_training_summary


class TestEvalCallback(TrainerCallback):
    """
    Custom callback to evaluate on test dataset after each epoch.
    
    This callback tracks test metrics and logs them to WandB for monitoring
    training progress on the test set.
    """
    
    def __init__(self, test_dataset: Dataset):
        """
        Initialize the test evaluation callback.
        
        Args:
            test_dataset: Test dataset for evaluation
        """
        self.test_dataset = test_dataset
        self.epochs = []
        self.test_losses = []
        self.test_f1 = []
        self.test_mcc = []
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Evaluate on test dataset at the end of each epoch.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            **kwargs: Additional arguments including trainer
        """
        trainer = kwargs.get("trainer")
        if trainer is None:
            return
        
        # Evaluate on test dataset
        test_output = trainer.predict(self.test_dataset)
        test_metrics = test_output.metrics
        
        # Extract metrics
        loss = test_metrics.get("test_loss", np.nan)
        f1 = test_metrics.get("test_f1_score", np.nan)
        mcc = test_metrics.get("test_mcc_score", np.nan)
        
        # Store metrics
        self.epochs.append(state.epoch)
        self.test_losses.append(loss)
        self.test_f1.append(f1)
        self.test_mcc.append(mcc)
        
        # Log to WandB
        wandb.log({
            "test_loss": loss,
            "test_f1": f1,
            "test_mcc": mcc,
            "epoch": state.epoch
        })


class DNATrainer(Trainer):
    """
    Custom trainer for DNA sequence models with enhanced logging and visualization.
    
    This trainer extends the HuggingFace Trainer with DNA-specific functionality
    including test evaluation, metric tracking, and visualization.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        task_name: str = "",
        model_type: str = "",
        output_dir: str = "",
        **kwargs
    ):
        """
        Initialize the DNA trainer.
        
        Args:
            model: Model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Validation dataset
            test_dataset: Test dataset (optional)
            tokenizer: Tokenizer for the model
            task_name: Name of the task being trained on
            model_type: Type of model architecture
            output_dir: Directory to save outputs
            **kwargs: Additional arguments for Trainer
        """
        self.task_name = task_name
        self.model_type = model_type
        self.output_dir = output_dir
        self.test_dataset = test_dataset
        
        # Set up callbacks
        callbacks = kwargs.get("callbacks", [])
        if test_dataset is not None:
            test_callback = TestEvalCallback(test_dataset)
            callbacks.append(test_callback)
            self.test_callback = test_callback
        else:
            self.test_callback = None
        
        # Set up compute_metrics if not provided
        if "compute_metrics" not in kwargs:
            kwargs["compute_metrics"] = compute_metrics
        
        # Initialize parent trainer
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            **kwargs
        )
        
        # Set up metrics CSV
        self._setup_metrics_csv()
        
        # Log model summary
        self._log_model_summary()
    
    def _setup_metrics_csv(self):
        """Set up CSV file for tracking metrics."""
        if not self.output_dir:
            return
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_csv_path = os.path.join(
            self.output_dir, f"{self.task_name}_{self.model_type}_metrics.csv"
        )
        
        # Create CSV file with headers
        with open(self.metrics_csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["epoch", "f1_score", "mcc_score"])
    
    def _log_model_summary(self):
        """Log model summary to WandB."""
        if self.task_name and self.model_type:
            summary = get_training_summary(self.model, self.task_name, self.model_type)
            wandb.config.update(summary)
    
    def train(self, resume_from_checkpoint: Optional[str] = None, **kwargs):
        """
        Train the model with enhanced logging.
        
        Args:
            resume_from_checkpoint: Checkpoint to resume from
            **kwargs: Additional arguments for training
            
        Returns:
            Training output
        """
        # Log training start
        if self.task_name and self.model_type:
            print(f"\n=== Training {self.model_type} on {self.task_name} ===")
            print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Train the model
        result = super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
        
        # Generate visualizations
        self._generate_training_plots()
        
        return result
    
    def _generate_training_plots(self):
        """Generate and save training plots."""
        if not self.output_dir or not self.task_name or not self.model_type:
            return
        
        # Training loss curve
        self._plot_training_loss()
        
        # Test metrics curves
        if self.test_callback and self.test_callback.epochs:
            self._plot_test_metrics()
    
    def _plot_training_loss(self):
        """Plot training loss curve."""
        log_history = self.state.log_history
        train_steps = [entry["step"] for entry in log_history if "loss" in entry]
        train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
        
        if train_steps and train_losses:
            plt.figure()
            plt.plot(train_steps, train_losses, label="Training Loss")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title(f"Training Loss Curve - {self.task_name} ({self.model_type})")
            plt.legend()
            
            loss_curve_path = os.path.join(
                self.output_dir, f"{self.task_name}_{self.model_type}_loss_curve.png"
            )
            plt.savefig(loss_curve_path)
            plt.close()
            
            wandb.log({"loss_curve": wandb.Image(loss_curve_path)})
    
    def _plot_test_metrics(self):
        """Plot test metrics curves."""
        if not self.test_callback:
            return
        
        # Test loss curve
        plt.figure()
        plt.plot(self.test_callback.epochs, self.test_callback.test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Test Loss Curve - {self.task_name} ({self.model_type})")
        plt.legend()
        
        test_loss_plot_path = os.path.join(
            self.output_dir, f"{self.task_name}_{self.model_type}_test_loss.png"
        )
        plt.savefig(test_loss_plot_path)
        plt.close()
        
        wandb.log({"test_loss_curve": wandb.Image(test_loss_plot_path)})
        
        # Test F1 curve
        plt.figure()
        plt.plot(self.test_callback.epochs, self.test_callback.test_f1, label="Test F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title(f"Test F1 Score Curve - {self.task_name} ({self.model_type})")
        plt.legend()
        
        test_f1_plot_path = os.path.join(
            self.output_dir, f"{self.task_name}_{self.model_type}_test_f1.png"
        )
        plt.savefig(test_f1_plot_path)
        plt.close()
        
        wandb.log({"test_f1_curve": wandb.Image(test_f1_plot_path)})
        
        # Test MCC curve
        plt.figure()
        plt.plot(self.test_callback.epochs, self.test_callback.test_mcc, label="Test MCC Score")
        plt.xlabel("Epoch")
        plt.ylabel("MCC Score")
        plt.title(f"Test MCC Score Curve - {self.task_name} ({self.model_type})")
        plt.legend()
        
        test_mcc_plot_path = os.path.join(
            self.output_dir, f"{self.task_name}_{self.model_type}_test_mcc.png"
        )
        plt.savefig(test_mcc_plot_path)
        plt.close()
        
        wandb.log({"test_mcc_curve": wandb.Image(test_mcc_plot_path)})
    
    def evaluate_test(self) -> Dict[str, float]:
        """
        Evaluate the model on the test dataset.
        
        Returns:
            Test evaluation results
        """
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")
        
        test_output = self.predict(self.test_dataset)
        eval_results = test_output.metrics
        
        print(f"Final test evaluation for task '{self.task_name}', model '{self.model_type}':")
        print(eval_results)
        
        return eval_results


class DistillationTrainer(DNATrainer):
    """
    Trainer for knowledge distillation tasks.
    
    This trainer extends DNATrainer with distillation-specific functionality
    for training student models with teacher guidance.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        task_name: str = "",
        model_type: str = "",
        output_dir: str = "",
        teacher_model: Optional[PreTrainedModel] = None,
        **kwargs
    ):
        """
        Initialize the distillation trainer.
        
        Args:
            model: Student model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Validation dataset
            test_dataset: Test dataset (optional)
            tokenizer: Tokenizer for the model
            task_name: Name of the task being trained on
            model_type: Type of student model architecture
            output_dir: Directory to save outputs
            teacher_model: Teacher model for distillation
            **kwargs: Additional arguments for DNATrainer
        """
        self.teacher_model = teacher_model
        
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            task_name=task_name,
            model_type=model_type,
            output_dir=output_dir,
            **kwargs
        )
        
        # Log distillation info
        if teacher_model is not None:
            wandb.config.update({
                "distillation": True,
                "teacher_model": type(teacher_model).__name__,
                "student_model": type(model).__name__,
            })
    
    def _log_model_summary(self):
        """Log model summary with distillation info."""
        super()._log_model_summary()
        
        if self.teacher_model is not None:
            teacher_summary = get_training_summary(
                self.teacher_model, 
                f"{self.task_name}_teacher", 
                "teacher"
            )
            wandb.config.update({"teacher_" + k: v for k, v in teacher_summary.items()})
