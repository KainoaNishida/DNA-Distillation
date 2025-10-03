#!/usr/bin/env python3
"""
DNA Distillation CLI Interface

A powerful command-line interface for DNA sequence analysis and knowledge distillation.
"""

import tyro
import os
import sys
from typing import Optional, List, Literal, Union, Annotated
from pathlib import Path
import dna_distillation as dna
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


def main():
    """Main CLI entry point."""
    tyro.cli(Union[Train, Distill, ListTasks, ListModels, ListTeacherModels, CreateModel, Info, AdvancedDistill, SimpleDistill, CreateAdvancedModel, TrainTeacher], 
             description="DNA Distillation CLI - A powerful tool for DNA sequence analysis and knowledge distillation")


class Train:
    """Train a student model on a DNA sequence task."""
    
    def __init__(
        self,
        task: str = "H3K27ac",
        model_type: str = "bilstm",
        output_dir: str = "./outputs",
        # Model parameters
        vocab_size: int = 1000,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        # Training parameters
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        # Data parameters
        max_length: int = 512,
        train_split: float = 0.8,
        val_split: float = 0.1,
        # WandB parameters
        use_wandb: bool = True,
        wandb_project: str = "dna-distillation",
        wandb_run_name: Optional[str] = None,
        # Other options
        seed: int = 42,
        device: str = "auto",
        num_workers: int = 4,
        save_best: bool = True,
        early_stopping_patience: int = 5,
    ):
        rprint(f" [bold blue]Training {model_type} on {task}[/bold blue]")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create student model
        model = dna.create_student_model(
            model_type=model_type,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_labels=dna.get_num_labels(task),
            num_layers=num_layers,
            dropout=dropout
        )
        
        rprint(f"Model created with {model.get_num_parameters():,} parameters")
        
        # Load dataset
        rprint(" Loading dataset...")
        datasets = dna.load_nucleotide_task(
            task_name=task,
            max_length=max_length,
            train_split=train_split,
            val_split=val_split
        )
        
        rprint(f" Dataset loaded: {len(datasets['train'])} train, {len(datasets['validation'])} val")
        
        # Create trainer
        trainer = dna.DNATrainer(
            model=model,
            train_dataset=datasets["train"],
            val_dataset=datasets["validation"],
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name or f"{task}_{model_type}",
            seed=seed,
            device=device,
            num_workers=num_workers,
            save_best=save_best,
            early_stopping_patience=early_stopping_patience,
        )
        
        # Train
        rprint(" Starting training...")
        results = trainer.train()
        
        # Display results
        _display_training_results(results)
        
        rprint(f" Training completed! Model saved to: {output_dir}")


class Distill:
    """Distill knowledge from teacher to student model."""
    
    def __init__(
        self,
        task: str = "H3K27ac",
        student_model_type: str = "bilstm",
        teacher_checkpoint_dir: str = "./teacher_checkpoints",
        output_dir: str = "./distilled_models",
        # Model parameters
        vocab_size: int = 1000,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        # Distillation parameters
        alpha: float = 0.5,
        temperature: float = 2.0,
        precompute_teacher_logits: bool = False,
        cache_dir: Optional[str] = None,
        # Training parameters
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 500,
        # Data parameters
        max_length: int = 512,
        train_split: float = 0.8,
        val_split: float = 0.1,
        # WandB parameters
        use_wandb: bool = True,
        wandb_project: str = "dna-distillation",
        wandb_run_name: Optional[str] = None,
        # Other options
        seed: int = 42,
        device: str = "auto",
        num_workers: int = 4,
        save_best: bool = True,
        early_stopping_patience: int = 5,
    ):
        rprint(f" [bold blue]Distilling {student_model_type} from teacher on {task}[/bold blue]")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run distillation
        results = dna.distill_for_task(
            task_name=task,
            student_model_type=student_model_type,
            teacher_checkpoint_dir=teacher_checkpoint_dir,
            output_dir=output_dir,
            # Model parameters
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            # Distillation parameters
            alpha=alpha,
            temperature=temperature,
            precompute_teacher_logits=precompute_teacher_logits,
            cache_dir=cache_dir,
            # Training parameters
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            # Data parameters
            max_length=max_length,
            train_split=train_split,
            val_split=val_split,
            # WandB parameters
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name or f"{task}_{student_model_type}_distilled",
            # Other options
            seed=seed,
            device=device,
            num_workers=num_workers,
            save_best=save_best,
            early_stopping_patience=early_stopping_patience,
        )
        
        # Display results
        _display_distillation_results(results)
        
        rprint(f" Distillation completed! Model saved to: {output_dir}")


class ListTasks:
    """List all available DNA sequence tasks."""
    
    def __init__(self):
        tasks = dna.get_available_tasks()
        
        rprint(f" [bold blue]Available DNA Sequence Tasks ({len(tasks)})[/bold blue]")
        
        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Task Name", style="cyan")
        table.add_column("Labels", style="green")
        table.add_column("Type", style="yellow")
        
        for task in tasks:
            info = dna.get_task_info(task)
            table.add_row(
                task,
                str(info["num_labels"]),
                info["task_type"]
            )
        
        console.print(table)


class ListModels:
    """List all available student model types."""
    
    def __init__(self):
        models = ["bilstm", "xlstm", "mamba", "hyena", "caduceus", "cnn", "mlp", "rnn", "bpnet"]
        
        rprint(f" [bold blue]Available Student Model Types ({len(models)})[/bold blue]")
        
        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model Type", style="cyan")
        table.add_column("Description", style="green")
        
        descriptions = {
            "bilstm": "Bidirectional LSTM with attention",
            "xlstm": "Extended LSTM with residual connections",
            "mamba": "State space model for long sequences",
            "hyena": "Hyena operator for efficient long-range modeling",
            "caduceus": "Caduceus bidirectional state space model",
            "cnn": "Convolutional neural network",
            "mlp": "Multi-layer perceptron",
            "rnn": "Recurrent neural network",
            "bpnet": "BPNet with dilated convolutions and attention"
        }
        
        for model in models:
            table.add_row(model, descriptions.get(model, "Unknown"))
        
        console.print(table)


class ListTeacherModels:
    """List all available teacher model types."""
    
    def __init__(self):
        teacher_models = [
            "nucleotide_transformer_500m", 
            "nucleotide_transformer_2b5",
            "dna_bert2",
            "enformer", 
            "caduceus"
        ]
        
        rprint(f"[bold blue]Available Teacher Model Types ({len(teacher_models)})[/bold blue]")
        
        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model Type", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Memory Requirement", style="yellow")
        
        model_descriptions = {
            "nucleotide_transformer_500m": "500M parameter Nucleotide Transformer (Human Reference)",
            "nucleotide_transformer_2b5": "2.5B parameter Nucleotide Transformer (Multi-species)",
            "dna_bert2": "DNA-BERT2 117M parameter model",
            "enformer": "Enformer model for regulatory element prediction",
            "caduceus": "Caduceus bidirectional state space model"
        }
        
        memory_requirements = {
            "nucleotide_transformer_500m": "~8GB GPU memory",
            "nucleotide_transformer_2b5": "~24GB GPU memory",
            "dna_bert2": "~4GB GPU memory",
            "enformer": "~12GB GPU memory",
            "caduceus": "~6GB GPU memory"
        }
        
        for model in teacher_models:
            table.add_row(
                model, 
                model_descriptions.get(model, ""),
                memory_requirements.get(model, "")
            )
        
        console.print(table)


class CreateModel:
    """Create and save a student model."""
    
    def __init__(
        self,
        model_type: str = "bilstm",
        vocab_size: int = 1000,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_labels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        # CNN-specific parameters
        num_res_blocks: int = 16,
        use_one_hot: bool = False,
        kernel_size: int = 3,
        output_path: Optional[str] = None,
    ):
        rprint(f"[bold blue]Creating {model_type} model[/bold blue]")
        
        # Create model with appropriate parameters
        if model_type == "cnn":
            model = dna.create_student_model(
                model_type=model_type,
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_labels=num_labels,
                dropout=dropout,
                # CNN-specific parameters
                num_res_blocks=num_res_blocks,
                use_one_hot=use_one_hot,
                kernel_size=kernel_size,
            )
        else:
            model = dna.create_student_model(
                model_type=model_type,
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_labels=num_labels,
                num_layers=num_layers,
                dropout=dropout,
            )
        
        rprint(f"Model created with {model.get_num_parameters():,} parameters")
        
        if output_path:
            # Save model
            import torch
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            rprint(f" Model saved to: {output_path}")


class Info:
    """Display package information."""
    
    def __init__(self):
        rprint(Panel.fit(
            f"[bold blue]DNA Distillation v{dna.__version__}[/bold blue]\n\n"
            f" A comprehensive package for DNA sequence analysis and knowledge distillation\n\n"
            f" [bold]Features:[/bold]\n"
            f"  • 18+ DNA sequence tasks\n"
            f"  • 8 student model architectures\n"
            f"  • One-line training and distillation\n"
            f"  • WandB integration\n"
            f"  • Rich CLI interface\n\n"
            f" [bold]Quick Start:[/bold]\n"
            f"  dna-distill train H3K27ac bilstm ./outputs\n"
            f"  dna-distill distill H3K27ac bilstm ./teachers ./distilled\n"
            f"  dna-distill list-tasks\n"
            f"  dna-distill list-models",
            title="DNA Distillation",
            border_style="blue"
        ))


def _display_training_results(results):
    """Display training results in a nice format."""
    
    table = Table(title="Training Results", show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in results.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)


def _display_distillation_results(results):
    """Display distillation results in a nice format."""
    
    table = Table(title="Distillation Results", show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in results.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)


class AdvancedDistill:
    """Advanced distillation with latest research methods."""
    
    def __init__(
        self,
        task_name: str,
        method: Literal["vanilla", "logit_standard", "dkd", "dist", "reviewkd"] = "dkd",
        student_model_type: str = "bilstm",
        teacher_model_path: Optional[str] = None,
        output_dir: str = "./distillation_output",
        batch_size: int = 16,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 4.0,
        alpha: float = 1.0,
        beta: float = 8.0,
        gamma: float = 2.0,
        lambda_kl: float = 0.3,
        lambda_mse: float = 0.2,
        use_mixed_precision: bool = True,
        precompute_logits: bool = True,
        precompute_features: bool = False,
        hierarchical_checkpoints: bool = True,
        hyperparameter_search: bool = False,
        max_trials: int = 20,
        use_wandb: bool = True,
        wandb_project: str = "dna-distillation",
        wandb_run_name: Optional[str] = None,
    ):
        """
        Run advanced distillation with latest research methods.
        
        Args:
            task_name: Name of the downstream task
            method: Distillation method (vanilla, logit_standard, dkd, dist, reviewkd)
            student_model_type: Type of student model
            teacher_model_path: Path to teacher model checkpoint
            output_dir: Output directory for results
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            embed_dim: Embedding dimension for student model
            hidden_dim: Hidden dimension for student model
            dropout: Dropout rate for student model
            temperature: Temperature for distillation
            alpha: Alpha parameter for DKD
            beta: Beta parameter for DKD/DIST
            gamma: Gamma parameter for DIST
            lambda_kl: Weight for KL/distillation loss
            lambda_mse: Weight for MSE/feature loss
            use_mixed_precision: Whether to use mixed precision training
            precompute_logits: Whether to precompute teacher logits
            precompute_features: Whether to precompute teacher features
            hierarchical_checkpoints: Whether to use hierarchical checkpoint organization
            hyperparameter_search: Whether to run hyperparameter search
            max_trials: Maximum number of hyperparameter trials
            use_wandb: Whether to use WandB for experiment tracking
            wandb_project: WandB project name
            wandb_run_name: WandB run name (auto-generated if None)
        """
        rprint(f"[bold blue]Advanced Distillation: {method.upper()}[/bold blue]")
        rprint(f"Task: {task_name}")
        rprint(f"Student Model: {student_model_type}")
        
        # Initialize WandB if requested
        if use_wandb:
            try:
                import wandb
                if wandb_run_name is None:
                    wandb_run_name = f"{task_name}_{method}_{student_model_type}"
                
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "task_name": task_name,
                        "method": method,
                        "student_model_type": student_model_type,
                        "teacher_model_path": teacher_model_path,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "temperature": temperature,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "lambda_kl": lambda_kl,
                        "lambda_mse": lambda_mse,
                    }
                )
                rprint(f" WandB initialized: {wandb_project}/{wandb_run_name}")
            except ImportError:
                rprint("  WandB not available, continuing without logging")
            except Exception as e:
                rprint(f"  WandB initialization failed: {e}")
        
        # Import advanced features
        from dna_distillation.distillation import (
            AdvancedDistillationLoss,
            precompute_teacher_features,
            create_hierarchical_checkpoint_dir,
            hyperparameter_search,
            get_method_hyperparameters,
            setup_mixed_precision_training,
            create_advanced_distillation_config,
        )
        
        # Create advanced configuration
        config = create_advanced_distillation_config(
            method=method,
            task_name=task_name,
            temperature=temperature,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            lambda_kl=lambda_kl,
            lambda_mse=lambda_mse,
            use_mixed_precision=use_mixed_precision,
            precompute_logits=precompute_logits,
            precompute_features=precompute_features,
            hierarchical_checkpoints=hierarchical_checkpoints,
            hyperparameter_search=hyperparameter_search,
        )
        
        rprint(f"  Configuration: {config}")
        
        if hierarchical_checkpoints:
            # Create hierarchical checkpoint directory
            hyperparams = {
                "lr": learning_rate,
                "temp": temperature,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "lambda_kl": lambda_kl,
                "lambda_mse": lambda_mse,
            }
            checkpoint_dir = create_hierarchical_checkpoint_dir(
                output_dir, task_name, method, hyperparams
            )
            rprint(f" Checkpoint directory: {checkpoint_dir}")
        
        if hyperparameter_search:
            # Run hyperparameter search
            search_space = {
                "temperature": [2.0, 4.0, 6.0],
                "alpha": [0.5, 1.0, 2.0],
                "beta": [4.0, 8.0, 16.0],
                "lambda_kl": [0.1, 0.3, 0.5],
            }
            configurations = hyperparameter_search(
                task_name, method, search_space, config, max_trials
            )
            rprint(f" Generated {len(configurations)} hyperparameter configurations")
        
        rprint("Advanced distillation setup complete!")
        
        # Actually run the distillation training
        rprint("\n Starting distillation training...")
        
        try:
            # Import training utilities
            from dna_distillation.training import train_teacher_model
            from dna_distillation.data import load_nucleotide_task
            from dna_distillation.models import create_student_model
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            from sklearn.metrics import f1_score, matthews_corrcoef
            import numpy as np
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Load dataset using research-compatible method
            rprint(" Loading dataset...")
            teacher_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
            
            # Use research-compatible data loading
            from dna_distillation.data.research_compatible import load_nucleotide_task_research_compatible
            datasets = load_nucleotide_task_research_compatible(
                task_name=task_name,
                tokenizer=teacher_tokenizer,
                max_length=512
            )
            
            rprint(f" Dataset loaded: {len(datasets['train'])} train, {len(datasets['validation'])} val")
            
            # Create student model
            rprint(" Creating student model...")
            if student_model_type == "cnn":
                student_model = create_student_model(
                    model_type=student_model_type,
                    vocab_size=1000,
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    num_labels=2,
                    dropout=dropout,
                    num_res_blocks=16,
                    use_one_hot=False,
                    kernel_size=3,
                )
            else:
                student_model = create_student_model(
                    model_type=student_model_type,
                    vocab_size=1000,
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    num_labels=2,
                    num_layers=2,
                    dropout=dropout,
                )
            
            total_params = sum(p.numel() for p in student_model.parameters())
            rprint(f" Student model created with {total_params:,} parameters")
            
            # Create teacher model
            rprint(" Loading teacher model...")
            teacher_model = AutoModelForSequenceClassification.from_pretrained(
                "InstaDeepAI/nucleotide-transformer-500m-human-ref",
                num_labels=2,
                trust_remote_code=True
            )
            
            rprint(" Teacher model loaded")
            
            # Create data loaders using research code pattern (no collate_fn)
            train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(datasets['validation'], batch_size=batch_size, shuffle=False)
            
            # Training setup
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            student_model = student_model.to(device)
            teacher_model = teacher_model.to(device)
            
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
            criterion = AdvancedDistillationLoss(
                method=method,
                temperature=temperature,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                lambda_kl=lambda_kl,
                lambda_mse=lambda_mse
            )
            
            rprint(f" Training on device: {device}")
            rprint(f" Optimizer: AdamW (lr={learning_rate})")
            rprint(f" Loss function: {method.upper()}")
            
            # Training loop
            student_model.train()
            teacher_model.eval()
            
            for epoch in range(num_epochs):
                rprint(f"\n Epoch {epoch + 1}/{num_epochs}")
                
                total_loss = 0
                num_batches = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    if batch_idx >= 50:  # Limit for CLI testing
                        break
                        
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    optimizer.zero_grad()
                    
                    # Student forward pass
                    student_outputs = student_model(input_ids)
                    student_logits = student_outputs['logits']
                    
                    # Teacher forward pass (no gradients)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
                        teacher_logits = teacher_outputs.logits
                    
                    # Compute loss
                    loss = criterion(student_logits, teacher_logits, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        rprint(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                rprint(f"   Average Loss: {avg_loss:.4f}")
                
                # Log to WandB if available
                if use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            "epoch": epoch + 1,
                            "train_loss": avg_loss,
                            "learning_rate": learning_rate
                        })
                    except:
                        pass
            
            rprint("\n Training completed!")
            
            # Save the trained model
            model_path = os.path.join(output_dir, "student_model.pt")
            torch.save(student_model.state_dict(), model_path)
            rprint(f" Model saved to: {model_path}")
            
            # Final evaluation
            rprint("\n Final evaluation...")
            student_model.eval()
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 10:  # Limit for CLI testing
                        break
                        
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = student_model(input_ids)
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            if len(all_predictions) > 0:
                f1 = f1_score(all_labels, all_predictions, average='weighted')
                mcc = matthews_corrcoef(all_labels, all_predictions)
                
                rprint(f" Final Metrics:")
                rprint(f"   F1 Score: {f1:.4f}")
                rprint(f"   MCC: {mcc:.4f}")
                
                # Log final metrics to WandB
                if use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            "final_f1": f1,
                            "final_mcc": mcc
                        })
                    except:
                        pass
            
            rprint("\n Distillation experiment completed successfully!")
            
        except Exception as e:
            rprint(f" Training failed: {e}")
            import traceback
            rprint(f"Traceback: {traceback.format_exc()}")
            
            # Log error to WandB
            if use_wandb:
                try:
                    import wandb
                    wandb.log({"error": str(e)})
                except:
                    pass


class SimpleDistill:
    """Run a simple distillation experiment with synthetic data."""
    
    def __init__(
        self,
        method: str = "dkd",
        student_model_type: str = "bilstm",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        temperature: float = 4.0,
        alpha: float = 1.0,
        beta: float = 8.0,
        gamma: float = 2.0,
        lambda_kl: float = 0.3,
        lambda_mse: float = 0.2,
        use_wandb: bool = True,
        wandb_project: str = "dna-distillation-simple",
        wandb_run_name: Optional[str] = None,
    ):
        """
        Run a simple distillation experiment with synthetic data.
        
        Args:
            method: Distillation method (dkd, dist, reviewkd, logit_std)
            student_model_type: Type of student model (bilstm, cnn, mlp, rnn, xlstm, mamba, hyena, caduceus, bpnet)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            temperature: Temperature for distillation
            alpha: Alpha parameter for DKD
            beta: Beta parameter for DKD
            gamma: Gamma parameter for DKD
            lambda_kl: Weight for KL divergence loss
            lambda_mse: Weight for MSE loss
            use_wandb: Whether to use WandB for experiment tracking
            wandb_project: WandB project name
            wandb_run_name: WandB run name (auto-generated if None)
        """
        rprint(f"[bold blue]Simple Distillation: {method.upper()}[/bold blue]")
        rprint(f"Student Model: {student_model_type}")
        
        # Initialize WandB if requested
        if use_wandb:
            try:
                import wandb
                if wandb_run_name is None:
                    wandb_run_name = f"simple_{method}_{student_model_type}"
                
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "method": method,
                        "student_model_type": student_model_type,
                        "num_epochs": num_epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "temperature": temperature,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "lambda_kl": lambda_kl,
                        "lambda_mse": lambda_mse,
                    }
                )
                rprint(f" WandB initialized: {wandb_project}/{wandb_run_name}")
            except ImportError:
                rprint("  WandB not available, continuing without logging")
            except Exception as e:
                rprint(f"  WandB initialization failed: {e}")
        
        try:
            # Import required modules
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from dna_distillation.models import create_student_model
            from dna_distillation.distillation import AdvancedDistillationLoss
            import numpy as np
            
            # Create synthetic data
            rprint("\n Creating synthetic data...")
            train_sequences, train_labels = self._create_synthetic_data(1000, 512, 4)
            val_sequences, val_labels = self._create_synthetic_data(200, 512, 4)
            
            rprint(f" Data created: {len(train_sequences)} train, {len(val_sequences)} val")
            
            # Create student model
            rprint("\n Creating student model...")
            if student_model_type == "cnn":
                student_model = create_student_model(
                    model_type=student_model_type,
                    vocab_size=4,  # A, C, G, T
                    embed_dim=64,
                    hidden_dim=32,
                    num_labels=2,
                    dropout=0.1,
                    num_res_blocks=8,
                    use_one_hot=False,
                    kernel_size=3,
                )
            else:
                student_model = create_student_model(
                    model_type=student_model_type,
                    vocab_size=4,  # A, C, G, T
                    embed_dim=64,
                    hidden_dim=32,
                    num_labels=2,
                    num_layers=2,
                    dropout=0.1,
                )
            
            total_params = sum(p.numel() for p in student_model.parameters())
            rprint(f" Student model created with {total_params:,} parameters")
            
            # Create a simple teacher model
            rprint("\n Creating teacher model...")
            teacher_model = self._create_simple_teacher(4, 64, 2)
            rprint(" Teacher model created")
            
            # Create data loaders
            train_dataset = TensorDataset(train_sequences, train_labels)
            val_dataset = TensorDataset(val_sequences, val_labels)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Training setup
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rprint(f" Training on device: {device}")
            
            student_model = student_model.to(device)
            teacher_model = teacher_model.to(device)
            
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
            criterion = AdvancedDistillationLoss(
                method=method,
                temperature=temperature,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                lambda_kl=lambda_kl,
                lambda_mse=lambda_mse
            )
            
            rprint(f" Optimizer: AdamW (lr={learning_rate})")
            rprint(f" Loss function: {method.upper()}")
            
            # Training loop
            rprint(f"\n Starting training for {num_epochs} epochs...")
            student_model.train()
            teacher_model.eval()
            
            for epoch in range(num_epochs):
                rprint(f"\n Epoch {epoch + 1}/{num_epochs}")
                
                total_loss = 0
                num_batches = 0
                
                for batch_idx, (sequences, labels) in enumerate(train_loader):
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Student forward pass
                    student_outputs = student_model(sequences)
                    student_logits = student_outputs['logits']
                    
                    # Teacher forward pass (no gradients)
                    with torch.no_grad():
                        teacher_logits = teacher_model(sequences)
                    
                    # Compute loss
                    loss = criterion(student_logits, teacher_logits, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        rprint(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                rprint(f"   Average Loss: {avg_loss:.4f}")
                
                # Log to WandB
                if use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            "epoch": epoch + 1,
                            "train_loss": avg_loss,
                            "learning_rate": learning_rate
                        })
                    except:
                        pass
            
            rprint("\n Training completed!")
            
            # Final evaluation
            rprint("\n Final evaluation...")
            student_model.eval()
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    
                    outputs = student_model(sequences)
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate accuracy
            correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
            accuracy = correct / len(all_predictions)
            
            rprint(f" Final Accuracy: {accuracy:.4f}")
            
            # Log final metrics to WandB
            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "final_accuracy": accuracy
                    })
                except:
                    pass
            
            rprint("\n Simple distillation test completed successfully!")
            if use_wandb:
                try:
                    import wandb
                    rprint(f"🔗 View results at: {wandb.run.url}")
                except:
                    pass
            
        except Exception as e:
            rprint(f" Test failed: {e}")
            import traceback
            rprint(f"Traceback: {traceback.format_exc()}")
            
            # Log error to WandB
            if use_wandb:
                try:
                    import wandb
                    wandb.log({"error": str(e)})
                except:
                    pass
        
        finally:
            if use_wandb:
                try:
                    import wandb
                    wandb.finish()
                except:
                    pass
    
    def _create_synthetic_data(self, num_samples, seq_length, vocab_size):
        """Create synthetic DNA sequence data for testing."""
        import torch
        # Generate random DNA sequences (A=0, C=1, G=2, T=3)
        sequences = torch.randint(0, vocab_size, (num_samples, seq_length))
        
        # Generate random binary labels
        labels = torch.randint(0, 2, (num_samples,))
        
        return sequences, labels
    
    def _create_simple_teacher(self, vocab_size, embed_dim, num_labels):
        """Create a simple teacher model for testing."""
        import torch.nn as nn
        class SimpleTeacher(nn.Module):
            def __init__(self, vocab_size, embed_dim, num_labels):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, 64, batch_first=True)
                self.classifier = nn.Linear(64, num_labels)
                
            def forward(self, x):
                x = self.embedding(x)
                x, _ = self.lstm(x)
                x = x.mean(dim=1)  # Global average pooling
                return self.classifier(x)
        
        return SimpleTeacher(vocab_size, embed_dim, num_labels)


class CreateAdvancedModel:
    """Create advanced model architectures from latest research."""
    
    def __init__(
        self,
        model_type: Literal["bpnet", "enformer", "caduceus", "dnabert2"] = "bpnet",
        num_labels: int = 2,
        input_length: int = 1000,
        num_filters: int = 64,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        output_path: Optional[str] = None,
    ):
        """
        Create advanced model from latest research.
        
        Args:
            model_type: Type of advanced model
            num_labels: Number of output classes
            input_length: Input sequence length
            num_filters: Number of filters (for BPNet)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_path: Path to save the model
        """
        rprint(f" [bold blue]Creating Advanced Model: {model_type.upper()}[/bold blue]")
        
        # Import advanced models
        from dna_distillation.models import create_advanced_model, get_advanced_model_info
        
        # Get model information
        model_info = get_advanced_model_info(model_type)
        rprint(f" {model_info['name']}: {model_info['description']}")
        
        # Create model
        if model_type == "bpnet":
            model = create_advanced_model(
                model_type=model_type,
                num_labels=num_labels,
                input_length=input_length,
                num_filters=num_filters,
                dropout=dropout,
            )
        elif model_type in ["enformer", "dnabert2"]:
            model = create_advanced_model(
                model_type=model_type,
                num_labels=num_labels,
                input_length=input_length,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif model_type == "caduceus":
            model = create_advanced_model(
                model_type=model_type,
                num_labels=num_labels,
                input_length=input_length,
                hidden_dim=embed_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        rprint(f" Total parameters: {total_params:,}")
        rprint(f" Trainable parameters: {trainable_params:,}")
        
        if output_path:
            # Save model
            import torch
            torch.save(model.state_dict(), output_path)
            rprint(f"Model saved to: {output_path}")
        
        rprint("Advanced model created successfully!")


class TrainTeacher:
    """Train a teacher model on a DNA sequence task."""
    
    def __init__(
        self,
        task: str,
        teacher_model_type: str = "nucleotide_transformer_500m",
        output_dir: str = "./teacher_models",
        num_train_epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_length: int = 512,
        use_mixed_precision: bool = True,
        save_strategy: str = "epoch",
        evaluation_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "f1_score",
        early_stopping_patience: int = 5,
    ):
        """
        Train a teacher model on a DNA sequence task.
        
        Args:
            task: Name of the DNA sequence task
            teacher_model_type: Type of teacher model (nucleotide_transformer_500m, nucleotide_transformer_2b5, custom)
            output_dir: Directory to save the trained model
            num_train_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
            max_length: Maximum sequence length
            use_mixed_precision: Whether to use mixed precision training
            save_strategy: Model saving strategy
            evaluation_strategy: Evaluation strategy
            load_best_model_at_end: Whether to load best model at end
            metric_for_best_model: Metric to use for best model selection
            early_stopping_patience: Early stopping patience
        """
        rprint(f"[bold blue]Training Teacher Model: {teacher_model_type.upper()}[/bold blue]")
        rprint(f"Task: {task}")
        rprint(f"Output Directory: {output_dir}")
        
        # Show model info
        model_info = {
            "nucleotide_transformer_500m": "500M parameter Nucleotide Transformer (Human Reference) - ~8GB GPU memory",
            "nucleotide_transformer_2b5": "2.5B parameter Nucleotide Transformer (Multi-species) - ~24GB GPU memory",
            "dna_bert2": "DNA-BERT2 117M parameter model - ~4GB GPU memory",
            "enformer": "Enformer model for regulatory element prediction - ~12GB GPU memory",
            "caduceus": "Caduceus bidirectional state space model - ~6GB GPU memory"
        }
        
        if teacher_model_type in model_info:
            rprint(f"Model: {model_info[teacher_model_type]}")
        else:
            rprint(f"Model: {teacher_model_type} (custom)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Import training utilities
        from dna_distillation.training import train_teacher_model
        
        # Train teacher model
        rprint("Loading dataset...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
        datasets = dna.load_nucleotide_task(
            task_name=task,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        rprint(f"Dataset loaded: {len(datasets['train'])} train, {len(datasets['validation'])} val")
        
        # Training configuration
        training_config = {
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size * 2,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "fp16": use_mixed_precision,
            "save_strategy": save_strategy,
            "evaluation_strategy": evaluation_strategy,
            "load_best_model_at_end": load_best_model_at_end,
            "metric_for_best_model": metric_for_best_model,
            "output_dir": output_dir,
            "logging_steps": 50,
            "report_to": ["wandb"] if "wandb" in sys.modules else None,
        }
        
        rprint("Starting teacher training...")
        results = train_teacher_model(
            task_name=task,
            teacher_model_type=teacher_model_type,
            datasets=datasets,
            training_config=training_config
        )
        
        # Display results
        _display_training_results(results)
        
        rprint(f"Teacher training completed! Model saved to: {output_dir}")


if __name__ == "__main__":
    main()