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
from .utils import validate_training_inputs, get_default_training_args, compute_metrics


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
        EarlyStoppingCallback,
        DataCollatorWithPadding,
    )
    from datasets import load_dataset
    import torch
    import os
    
    # Define model configurations
    model_configs = {
        "nucleotide_transformer_500m": {
            "model_name": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
            "description": "500M parameter Nucleotide Transformer (Human Reference)",
            "memory_requirement": "~8GB GPU memory",
            "trust_remote_code": True
        },
        "nucleotide_transformer_2b5": {
            "model_name": "InstaDeepAI/nucleotide-transformer-2.5b-multi-species", 
            "description": "2.5B parameter Nucleotide Transformer (Multi-species)",
            "memory_requirement": "~24GB GPU memory",
            "trust_remote_code": True
        },
        "dna_bert2": {
            "model_name": "zhihan1996/DNABERT-2-117M",
            "description": "DNA-BERT2 117M parameter model",
            "memory_requirement": "~4GB GPU memory",
            "trust_remote_code": True,
            "revision": "ec1f874253852eb3907081f57294991b4280ceb6"
        },
        "enformer": {
            "model_name": "EleutherAI/enformer-official-rough",
            "description": "Enformer model for regulatory element prediction",
            "memory_requirement": "~12GB GPU memory",
            "requires_enformer_pytorch": True,
            "target_length": 8,
            "trust_remote_code": False
        },
        "caduceus": {
            "model_name": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
            "description": "Caduceus bidirectional state space model",
            "memory_requirement": "~6GB GPU memory",
            "trust_remote_code": True
        }
    }
    
    # Auto-detect tokenizer name if not provided
    if tokenizer_name is None:
        if teacher_model_type in model_configs:
            tokenizer_name = model_configs[teacher_model_type]["model_name"]
        else:
            raise ValueError(f"Unknown teacher model type: {teacher_model_type}. Available: {list(model_configs.keys())}")
    
    # Load tokenizer first (needed by data loaders), except Enformer which uses manual encoding
    if teacher_model_type == "enformer":
        tokenizer = None
    elif teacher_model_type == "dna_bert2":
        tokenizer = AutoTokenizer.from_pretrained(
            model_configs["dna_bert2"]["model_name"],
            trust_remote_code=True,
            revision=model_configs["dna_bert2"].get("revision"),
            force_download=True,
        )
        # Ensure a fixed max length like the script
        try:
            tokenizer.model_max_length = max(512, getattr(tokenizer, "model_max_length", 512))
        except Exception:
            pass
        # Ensure pad token exists and padding behavior matches HF expectations
        try:
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.padding_side = "right"
        except Exception:
            pass
    elif teacher_model_type == "caduceus":
        # Caduceus requires trust_remote_code=True for custom tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_configs["caduceus"]["model_name"],
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load datasets if not provided
    if datasets is None and teacher_model_type != "enformer":
        max_length = training_config.get("max_length", 512) if training_config else 512
        if teacher_model_type in ("nucleotide_transformer_500m", "nucleotide_transformer_2b5", "dna_bert2"):
            # Exact per-task splits, tokenization & formatting as in the research script
            train_ds = load_dataset(
                "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
                task_name,
                split="train",
                trust_remote_code=True,
            )
            test_ds = load_dataset(
                "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
                task_name,
                split="test",
                trust_remote_code=True,
            )

            split = train_ds.train_test_split(test_size=0.1, seed=42)
            train_ds = split["train"]
            val_ds = split["test"]

            def tokenize_fn(examples):
                return tokenizer(
                    examples["sequence"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )

            train_ds = train_ds.map(tokenize_fn, batched=True)
            val_ds = val_ds.map(tokenize_fn, batched=True)
            test_ds = test_ds.map(tokenize_fn, batched=True)

            train_ds = train_ds.rename_column("label", "labels")
            val_ds = val_ds.rename_column("label", "labels")
            test_ds = test_ds.rename_column("label", "labels")

            train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
            val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
            test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) 

            datasets = {"train": train_ds, "validation": val_ds, "test": test_ds}
        else:
            datasets = load_nucleotide_task(
                task_name=task_name,
                tokenizer=tokenizer,
                max_length=max_length,
            )
    
    # Get number of labels
    num_labels = get_num_labels(task_name)
    
    # Load teacher model
    if teacher_model_type in model_configs:
        config = model_configs[teacher_model_type]
        model_name = config["model_name"]
        print(f"Loading {config['description']}")
        print(f"Memory requirement: {config['memory_requirement']}")
        
        # Handle different model types
        if teacher_model_type == "enformer":
            # Enformer teacher: mirror enformer_train_5-30-maxlen1024-bz16-epoch100-use-pretrained.py
            try:
                import torch
                import torch.nn as nn
                import torch.nn.functional as F
                import numpy as np
                from torch.utils.data import Dataset, DataLoader
                from sklearn.metrics import f1_score, matthews_corrcoef
                from sklearn.model_selection import train_test_split
                from enformer_pytorch import Enformer, from_pretrained
            except ImportError:
                raise ImportError("Enformer requires 'enformer-pytorch'. Install: pip install enformer-pytorch")

            # Char-to-index and sequence encoding
            CHAR2IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

            def encode_seq(seq: str, max_len: int) -> list:
                seq = seq.upper()
                idxs = [CHAR2IDX.get(ch, 4) for ch in seq[:max_len]]
                if len(idxs) < max_len:
                    idxs += [4] * (max_len - len(idxs))
                return idxs

            class SeqDataset(Dataset):
                def __init__(self, sequences, labels, max_len):
                    self.ids = [encode_seq(s, max_len) for s in sequences]
                    self.labels = labels
                def __len__(self):
                    return len(self.ids)
                def __getitem__(self, idx):
                    return (
                        torch.tensor(self.ids[idx], dtype=torch.long),
                        torch.tensor(self.labels[idx], dtype=torch.long),
                    )

            class EnformerClassifier(nn.Module):
                def __init__(self, dim: int, depth: int, heads: int, target_length: int, num_labels: int,
                             use_pretrained: bool = True, pretrained_name: str | None = "EleutherAI/enformer-official-rough"):
                    super().__init__()
                    if use_pretrained:
                        self.enformer = from_pretrained(pretrained_name, target_length=target_length, use_tf_gamma=False)
                    else:
                        # Fallback minimal init; exact hparams from training_config
                        self.enformer = Enformer.from_hparams(
                            dim=dim, depth=depth, heads=heads, output_heads={"dummy": 1}, target_length=target_length
                        )
                    hidden_dim = dim * 2
                    self.pool = nn.AdaptiveAvgPool1d(1)
                    self.classifier = nn.Linear(hidden_dim, num_labels)

                def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                    _, embeddings = self.enformer(input_ids, return_embeddings=True)
                    feats = embeddings.permute(0, 2, 1)
                    pooled = self.pool(feats).squeeze(-1)
                    return self.classifier(pooled)

            # Pull training config defaults matching the script
            epochs = int((training_config or {}).get("epochs", (training_config or {}).get("num_train_epochs", 100)))
            batch_size = int((training_config or {}).get("batch_size", (training_config or {}).get("per_device_train_batch_size", 16)))
            lr = float((training_config or {}).get("learning_rate", 1e-4))
            max_len = int((training_config or {}).get("max_len", (training_config or {}).get("max_length", 1024)))
            target_length = int((training_config or {}).get("target_length", config.get("target_length", 8)))
            enformer_dim = int((training_config or {}).get("enformer_dim", 1536))
            enformer_depth = int((training_config or {}).get("enformer_depth", 11))
            enformer_heads = int((training_config or {}).get("enformer_heads", 8))
            top_k = int((training_config or {}).get("top_k", 3))
            eval_every = int((training_config or {}).get("eval_every", 5))
            use_pretrained = bool((training_config or {}).get("use_pretrained", True))
            pretrained_name = (training_config or {}).get("pretrained_name", "EleutherAI/enformer-official-rough")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load per-task datasets exactly like script
            ds_train_full = load_dataset(
                "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", split="train", trust_remote_code=True
            )
            ds_test_full = load_dataset(
                "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", split="test", trust_remote_code=True
            )
            
            # Filter by task
            ds_train_full = ds_train_full.filter(lambda x: x["task"] == task_name)
            ds_test_full = ds_test_full.filter(lambda x: x["task"] == task_name)

            X_tr, X_val, y_tr, y_val = train_test_split(
                ds_train_full["sequence"], ds_train_full["label"], test_size=0.1, random_state=42, stratify=ds_train_full["label"]
            )

            train_loader = DataLoader(SeqDataset(X_tr, y_tr, max_len), batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(SeqDataset(X_val, y_val, max_len), batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(SeqDataset(ds_test_full["sequence"], ds_test_full["label"], max_len), batch_size=batch_size, shuffle=False, num_workers=0)

            # Model
            num_labels_enf = len(set(ds_train_full["label"]))
            model = EnformerClassifier(
                dim=enformer_dim,
                depth=enformer_depth,
                heads=enformer_heads,
                target_length=target_length,
                num_labels=num_labels_enf,
                use_pretrained=use_pretrained,
                pretrained_name=pretrained_name,
            ).to(device)

            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR
            optimizer = AdamW(model.parameters(), lr=lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs * max(1, len(train_loader)))

            # WandB logging
            project_name = (training_config or {}).get("wandb_project", "EnformerRevised-h800-epo100-maxlen1024-bz16")
            if use_pretrained and not project_name.endswith("-pretrained"):
                project_name += "-pretrained"
            wandb.init(
                project=project_name,
                name=f"{task_name}_e{epochs}_bs{batch_size}_lr{lr}",
                config={
                    "task": task_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "max_len": max_len,
                    "target_length": target_length,
                    "top_k": top_k,
                    "eval_every": eval_every,
                    "use_pretrained": use_pretrained,
                    "pretrained_name": pretrained_name,
                },
                reinit=True,
            )
            wandb.watch(model, log="all", log_freq=100)

            # Train loop
            def evaluate(loader):
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for ids, labels in loader:
                        ids = ids.to(device)
                        labels = labels.to(device)
                        logits = model(ids)
                        preds = logits.argmax(dim=-1).cpu().numpy()
                        all_preds.append(preds)
                        all_labels.append(labels.cpu().numpy())
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                return {"f1": f1_score(all_labels, all_preds, average="macro"), "mcc": matthews_corrcoef(all_labels, all_preds)}

            best_models: list[tuple[float, str]] = []
            ckpt_dir = os.path.join(os.getcwd(), f"{task_name}_checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            global_step = 0
            for epoch in range(1, epochs + 1):
                model.train()
                running_loss = 0.0
                batch_preds, batch_labels = [], []
                for ids, labels in train_loader:
                    ids, labels = ids.to(device), labels.to(device)
                    optimizer.zero_grad()
                    logits = model(ids)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    running_loss += float(loss.item()) * ids.size(0)
                    global_step += 1
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    batch_preds.extend(preds)
                    batch_labels.extend(labels.cpu().numpy())
                    if global_step % 50 == 0:
                        wandb.log({
                            "train/batch_loss": loss.item(),
                            "train/batch_f1": f1_score(batch_labels, batch_preds, average="macro"),
                            "step": global_step,
                        })
                        batch_preds, batch_labels = [], []

                train_loss = running_loss / max(1, len(train_loader.dataset))
                val_metrics = evaluate(val_loader)
                wandb.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/f1": val_metrics["f1"],
                    "val/mcc": val_metrics["mcc"],
                }, step=global_step)

                # keep top-k by val MCC
                current_mcc = float(val_metrics["mcc"])
                if len(best_models) < top_k or current_mcc > best_models[0][0]:
                    ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch}_mcc{current_mcc:.4f}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    best_models.append((current_mcc, ckpt_path))
                    best_models.sort(key=lambda x: x[0])
                    if len(best_models) > top_k:
                        _, worst_path = best_models.pop(0)
                        try:
                            os.remove(worst_path)
                        except OSError:
                            pass
                    test_metrics = evaluate(test_loader)
                    rank = len(best_models)
                    wandb.log({
                        f"best_model/{rank}/epoch": epoch,
                        f"best_model/{rank}/test_f1": test_metrics["f1"],
                        f"best_model/{rank}/test_mcc": test_metrics["mcc"],
                    }, step=global_step)

                if epoch % eval_every == 0:
                    periodic = evaluate(test_loader)
                    wandb.log({
                        "periodic_test/epoch": epoch,
                        "periodic_test/f1": periodic["f1"],
                        "periodic_test/mcc": periodic["mcc"],
                    }, step=global_step)

            final_metrics = evaluate(test_loader)
            wandb.log({"test/f1": final_metrics["f1"], "test/mcc": final_metrics["mcc"]}, step=global_step)
            wandb.finish()

            # Save final weights
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, f"enformer_{task_name}_final.pt"))

            return {
                "task_name": task_name,
                "teacher_model_type": teacher_model_type,
                "final_test_f1": final_metrics["f1"],
                "final_test_mcc": final_metrics["mcc"],
                "model_path": output_dir,
            }
        elif teacher_model_type == "caduceus":
            # Caduceus teacher: mirror finetune-caduceus-8-7-save-val-mcc-early-stopping-set1.py
            from transformers import AdamW, get_cosine_schedule_with_warmup, DataCollatorWithPadding
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import f1_score, matthews_corrcoef
            from torch.utils.data import Dataset, DataLoader
            import pandas as pd
            import time
            
            # Evaluation function (from research script)
            def evaluate(model, loader, device):
                model.eval()
                all_preds = []
                all_labels = []
                total_time = 0.0
                with torch.no_grad():
                    for batch in loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)

                        start_time = time.time()
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        total_time += time.time() - start_time
                        
                        logits = outputs.logits
                        preds = logits.argmax(dim=-1).cpu().numpy()
                        all_preds.append(preds)
                        all_labels.append(labels.cpu().numpy())

                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                
                return {
                    "f1": f1_score(all_labels, all_preds, average="macro"),
                    "mcc": matthews_corrcoef(all_labels, all_preds),
                    "latency": total_time / len(loader.dataset)
                }
            
            # Target MCC scores for early stopping (from research script)
            TARGET_MCCS = {
                "H3K9ac": 0.58722, "H3K27ac": 0.52796, "enhancers": 0.56375, "enhancers_types": 0.54228,
                "promoter_all": 0.81154, "promoter_tata": 0.86496, "promoter_no_tata": 0.82328,
                "H3K4me1": 0.53370, "H3K4me2": 0.60484, "H3K4me3": 0.68665, "H2AFZ": 0.55129,
                "H3K27me3": 0.64596, "H3K36me3": 0.66649, "H3K9me3": 0.50532, "H4K20me1": 0.69339,
                "splice_sites_donors": 0.79797, "splice_sites_all": 0.81413, "splice_sites_acceptors": 0.79094,
            }
            
            # Caduceus-specific training config defaults
            if training_config is None:
                training_config = {
                    "num_train_epochs": 100,
                    "per_device_train_batch_size": 8,
                    "per_device_eval_batch_size": 8,
                    "learning_rate": 3e-5,
                    "weight_decay": 0.0,
                    "max_length": 1024,
                    "use_early_stopping": True,
                    "early_stop_patience": 30,
                    "wandb_project": "Caduceus-nt-revised-finetune",
                    "log_batch_every": 50,
                    "output_dir": output_dir,
                }
            
            target_mcc = TARGET_MCCS.get(task_name) if training_config.get("use_early_stopping", False) else None
            if target_mcc:
                print(f"Early stopping enabled for '{task_name}' with target MCC: {target_mcc:.4f}")
            
            # Load datasets exactly like research script
            full_ds_trainval = load_dataset(
                "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
                split="train", trust_remote_code=True
            )
            full_ds_test = load_dataset(
                "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
                split="test", trust_remote_code=True
            )
            
            # Filter by task
            ds_trainval = full_ds_trainval.filter(lambda example: example["task"] == task_name)
            ds_test = full_ds_test.filter(lambda example: example["task"] == task_name)
            
            # Extract sequences and labels
            sequences = ds_trainval["sequence"]
            labels = ds_trainval["label"]
            
            # Split train -> train / validation (90% / 10%)
            X_train, X_val, y_train, y_val = train_test_split(
                sequences, labels, test_size=0.1, random_state=42, stratify=labels
            )
            
            # NTSeqDataset class (from research script)
            class NTSeqDataset(Dataset):
                def __init__(self, sequences, labels, tokenizer, max_len):
                    self.sequences = sequences
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.max_len = max_len

                def __len__(self):
                    return len(self.sequences)

                def __getitem__(self, idx):
                    seq = self.sequences[idx]
                    label = self.labels[idx]
                    enc = self.tokenizer(
                        seq, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
                    )
                    input_ids = enc["input_ids"].squeeze(0)
                    attention_mask = enc["attention_mask"].squeeze(0) if "attention_mask" in enc else (input_ids != self.tokenizer.pad_token_id).long()
                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": torch.tensor(label, dtype=torch.long),
                    }
            
            # Create datasets
            max_length = training_config.get("max_length", 1024)
            train_ds = NTSeqDataset(X_train, y_train, tokenizer, max_length)
            val_ds = NTSeqDataset(X_val, y_val, tokenizer, max_length)
            test_ds = NTSeqDataset(ds_test["sequence"], ds_test["label"], tokenizer, max_length)
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer, padding=True)
            
            # DataLoaders
            train_loader = DataLoader(
                train_ds, batch_size=training_config["per_device_train_batch_size"], 
                shuffle=True, num_workers=0, collate_fn=data_collator
            )
            val_loader = DataLoader(
                val_ds, batch_size=training_config["per_device_eval_batch_size"], 
                shuffle=False, num_workers=0, collate_fn=data_collator
            )
            test_loader = DataLoader(
                test_ds, batch_size=training_config["per_device_eval_batch_size"], 
                shuffle=False, num_workers=0, collate_fn=data_collator
            )
            
            # Load Caduceus model
            model = AutoModelForSequenceClassification.from_pretrained(
                config["model_name"], num_labels=num_labels, trust_remote_code=True
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            # Optimizer & Scheduler
            optimizer = AdamW(model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"])
            total_steps = training_config["num_train_epochs"] * len(train_loader)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
            )
            
            # WandB setup
            run_name = f"{task_name}_caduceus_finetune"
            wandb.init(
                project=training_config.get("wandb_project", "Caduceus-nt-revised-finetune"),
                name=run_name,
                config={"task": task_name, **training_config},
                reinit=True
            )
            wandb.watch(model, log="all", log_freq=training_config.get("log_batch_every", 50))
            
            # Checkpoint directory
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Checkpoints will be saved in: {checkpoint_dir}")
            
            # Training loop
            start_time = time.time()
            global_step = 0
            best_val_mcc = -1.0
            best_test_metrics = {}
            target_mcc_reached = False
            epochs_after_reach = 0
            
            for epoch in range(1, training_config["num_train_epochs"] + 1):
                model.train()
                epoch_loss = 0.0
                
                for step, batch in enumerate(train_loader, start=1):
                    input_ids = batch["input_ids"].to(model.device)
                    attention_mask = batch["attention_mask"].to(model.device)
                    labels_batch = batch["labels"].to(model.device)

                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item() * input_ids.size(0)
                    global_step += 1

                    if global_step % training_config.get("log_batch_every", 50) == 0:
                        wandb.log({"train/batch_loss": loss.item(), "step": global_step})

                # End of epoch evaluation
                avg_loss = epoch_loss / len(train_ds)
                val_metrics = evaluate(model, val_loader, model.device)
                elapsed = time.time() - start_time
                
                print(f"Epoch {epoch}/{training_config['num_train_epochs']} | loss {avg_loss:.4f} | val_f1 {val_metrics['f1']:.4f} | val_mcc {val_metrics['mcc']:.4f}")
                wandb.log({
                    "epoch": epoch, "train/epoch_loss": avg_loss, "val/f1": val_metrics["f1"], 
                    "val/mcc": val_metrics["mcc"], "time/elapsed_s": elapsed
                }, step=global_step)

                # Save best model checkpoint
                if val_metrics["mcc"] > best_val_mcc:
                    best_val_mcc = val_metrics["mcc"]
                    best_test_metrics = evaluate(model, test_loader, model.device)
                    print(f"🚀 New best val_mcc: {best_val_mcc:.4f}. Corresponding test mcc: {best_test_metrics['mcc']:.4f}")

                    # Save checkpoint
                    checkpoint_filename = f"epoch{epoch}_valmcc_{best_val_mcc:.4f}.pt"
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                    torch.save(model.state_dict(), checkpoint_path)
                    
                    # Save tokenizer and config
                    model.config.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    
                    print(f"Saved new best checkpoint to: {checkpoint_path}")
                    wandb.log({"best_val_mcc": best_val_mcc, "best_test_mcc": best_test_metrics['mcc']})
                
                # Early stopping logic
                if target_mcc is not None:
                    if target_mcc_reached:
                        epochs_after_reach += 1
                        print(f"Patience epoch {epochs_after_reach}/{training_config.get('early_stop_patience', 30)}...")
                    elif val_metrics["mcc"] >= target_mcc:
                        print(f"🎯 Target MCC of {target_mcc:.4f} reached! Training for {training_config.get('early_stop_patience', 30)} more epochs.")
                        target_mcc_reached = True
                        epochs_after_reach += 1
                    
                    if epochs_after_reach >= training_config.get("early_stop_patience", 30) and target_mcc_reached:
                        print(f"Finished training {training_config.get('early_stop_patience', 30)} epochs after reaching target. Stopping early.")
                        break

            # Final summary
            print(f"\n--- Training Complete ---")
            print(f"Best validation MCC achieved: {best_val_mcc:.4f}")
            print(f"Corresponding Test F1: {best_test_metrics.get('f1', 0):.4f}, Test MCC: {best_test_metrics.get('mcc', 0):.4f}")
            
            wandb.log({
                "final/best_val_mcc": best_val_mcc,
                "final/test_f1": best_test_metrics.get('f1', 0),
                "final/test_mcc": best_test_metrics.get('mcc', 0),
                "final/test_latency": best_test_metrics.get('latency', 0)
            })
            wandb.finish()

            # Save summary to CSV
            summary_file = os.path.join(output_dir, 'caduceus_summary.csv')
            summary_df = pd.DataFrame([{
                'task_name': task_name, 
                'test_mcc': best_test_metrics.get('mcc', 0), 
                'test_latency_time': best_test_metrics.get('latency', 0)
            }])
            summary_df.to_csv(summary_file, mode='a', header=not os.path.exists(summary_file), index=False)

            return {
                "task_name": task_name,
                "teacher_model_type": teacher_model_type,
                "train_results": {"metrics": {"train_loss": avg_loss}},
                "test_results": best_test_metrics,
                "model_path": output_dir,
                "num_labels": num_labels,
            }
        else:
            # Standard HuggingFace models
            load_kwargs = {
                "num_labels": num_labels,
                "trust_remote_code": config.get("trust_remote_code", True)
            }
            
            # Add revision if specified
            if "revision" in config:
                load_kwargs["revision"] = config["revision"]
            
            # Load DNABERT2 exactly like the script
            if teacher_model_type == "dna_bert2":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                                            trust_remote_code=True,
                                            revision=config.get("revision"),
                                            force_download=True,
                                        )
                # Resize to len(tokenizer vocab) to prevent index errors from fast tokenizer ids
                try:
                    vocab_len = len(getattr(tokenizer, "get_vocab")() if hasattr(tokenizer, "get_vocab") else tokenizer)
                    model.resize_token_embeddings(vocab_len)
                except Exception:
                    pass
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    **load_kwargs
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
    
    # Mirror kd-research-code Trainer args to avoid NumPy 2.0 formatting pitfalls
    # Ensure labels are preserved and no column dropping occurs
    training_args = TrainingArguments(
        **training_config,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_drop_last=True,
    )
    
    # Data collator (matches research script behavior)
    data_collator = DataCollatorWithPadding(tokenizer)

    # DNABERT2: robustly ensure embedding size >= any token id in the dataset
    if teacher_model_type == "dna_bert2":
        try:
            from torch.utils.data import DataLoader
            probe_loader = DataLoader(datasets["train"], batch_size=32)
            probe_batch = next(iter(probe_loader))
            max_token_id = int(probe_batch["input_ids"].max().item())
            emb_matrix = model.get_input_embeddings().weight
            if max_token_id >= emb_matrix.size(0):
                model.resize_token_embeddings(max_token_id + 1)
        except Exception:
            pass
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
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