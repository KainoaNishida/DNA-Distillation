"""
SLURM configuration utilities for DNA Distillation.

This module provides utilities for creating and managing SLURM job scripts
for training and distillation tasks on HPC clusters.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class SLURMConfig:
    """Configuration for SLURM job submission."""
    
    # Job identification
    job_name: str = "dna_distillation"
    mail_type: str = "END"
    mail_user: Optional[str] = None
    
    # Resource allocation
    nodes: int = 1
    nodelist: Optional[str] = None
    cpus_per_task: int = 4
    gres: Optional[str] = None  # e.g., "gpu:1", "gpu:2:rtx6000"
    mem: Optional[str] = None  # e.g., "32G", "64G"
    partition: str = "gpu"
    time: str = "24:00:00"
    
    # Python environment
    python_path: str = "/usr/bin/python3"
    conda_env: Optional[str] = None
    
    # Working directory
    work_dir: str = "."
    
    # Additional SLURM options
    additional_options: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config = asdict(self)
        if config["additional_options"] is None:
            config["additional_options"] = {}
        return config


class SLURMJobGenerator:
    """
    Generate SLURM job scripts for DNA Distillation tasks.
    
    This class provides methods to create SLURM job scripts for
    training and distillation tasks with proper resource allocation.
    """
    
    def __init__(self, config: Optional[SLURMConfig] = None):
        """
        Initialize the SLURM job generator.
        
        Args:
            config: SLURM configuration (uses defaults if None)
        """
        self.config = config or SLURMConfig()
    
    def generate_slurm_header(self) -> str:
        """
        Generate SLURM header section.
        
        Returns:
            SLURM header as string
        """
        header_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.config.job_name}",
            f"#SBATCH --nodes={self.config.nodes}",
            f"#SBATCH --cpus-per-task={self.config.cpus_per_task}",
            f"#SBATCH --partition={self.config.partition}",
            f"#SBATCH --time={self.config.time}",
        ]
        
        # Optional parameters
        if self.config.mail_user:
            header_lines.extend([
                f"#SBATCH --mail-type={self.config.mail_type}",
                f"#SBATCH --mail-user={self.config.mail_user}",
            ])
        
        if self.config.nodelist:
            header_lines.append(f"#SBATCH --nodelist={self.config.nodelist}")
        
        if self.config.gres:
            header_lines.append(f"#SBATCH --gres={self.config.gres}")
        
        if self.config.mem:
            header_lines.append(f"#SBATCH --mem={self.config.mem}")
        
        # Additional options
        if self.config.additional_options:
            for key, value in self.config.additional_options.items():
                header_lines.append(f"#SBATCH --{key}={value}")
        
        return "\n".join(header_lines)
    
    def generate_environment_setup(self) -> str:
        """
        Generate environment setup section.
        
        Returns:
            Environment setup as string
        """
        lines = [
            "# Environment setup",
            "set -e  # Exit on error",
            "",
            "# Load modules (customize for your cluster)",
            "module purge",
            "module load cuda/11.8",
            "module load python/3.9",
            "",
        ]
        
        # Conda environment activation
        if self.config.conda_env:
            lines.extend([
                "# Activate conda environment",
                f"source ~/.bashrc",
                f"conda activate {self.config.conda_env}",
                "",
            ])
        
        # Set working directory
        lines.extend([
            f"# Set working directory",
            f"cd {self.config.work_dir}",
            "",
            "# Print environment info",
            "echo '=== Environment Information ==='",
            "echo \"Job ID: $SLURM_JOB_ID\"",
            "echo \"Node: $SLURM_NODELIST\"",
            "echo \"Working directory: $(pwd)\"",
            "echo \"Python: $(which python)\"",
            "echo \"CUDA devices: $(nvidia-smi --list-gpus | wc -l)\"",
            "echo \"================================\"",
            "",
        ])
        
        return "\n".join(lines)
    
    def generate_training_job(
        self,
        task_name: str,
        model_type: str,
        output_dir: str,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        **kwargs
    ) -> str:
        """
        Generate SLURM job script for training.
        
        Args:
            task_name: Name of the task to train on
            model_type: Type of student model
            output_dir: Output directory for model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            **kwargs: Additional training arguments
            
        Returns:
            Complete SLURM job script
        """
        # Update job name
        self.config.job_name = f"train_{task_name}_{model_type}"
        
        # Generate script sections
        header = self.generate_slurm_header()
        env_setup = self.generate_environment_setup()
        
        # Training command
        training_cmd = f"""# Training command
echo "Starting training for {task_name} with {model_type} model..."

python -c "
import dna_distillation as dna

# Train the model
results = dna.train_for_task(
    task_name='{task_name}',
    model_type='{model_type}',
    output_dir='{output_dir}',
    num_train_epochs={num_epochs},
    per_device_train_batch_size={batch_size},
    learning_rate={learning_rate},
    wandb_project='DNA-Distillation-SLURM',
    wandb_run_name='{task_name}_{model_type}_slurm_$SLURM_JOB_ID'
)

print(f'Training completed!')
print(f'Final F1: {{results[\"final_f1\"]:.4f}}')
print(f'Final MCC: {{results[\"final_mcc\"]:.4f}}')
print(f'Model saved to: {output_dir}')
"

echo "Training job completed!"
"""
        
        return f"{header}\n\n{env_setup}\n{training_cmd}"
    
    def generate_distillation_job(
        self,
        task_name: str,
        student_model_type: str,
        teacher_checkpoint_dir: str,
        output_dir: str,
        num_epochs: int = 10,
        alpha: float = 0.5,
        temperature: float = 2.0,
        **kwargs
    ) -> str:
        """
        Generate SLURM job script for distillation.
        
        Args:
            task_name: Name of the task to distill
            student_model_type: Type of student model
            teacher_checkpoint_dir: Directory containing teacher checkpoints
            output_dir: Output directory for distilled model
            num_epochs: Number of training epochs
            alpha: Weight for cross-entropy loss
            temperature: Temperature for logit scaling
            **kwargs: Additional distillation arguments
            
        Returns:
            Complete SLURM job script
        """
        # Update job name
        self.config.job_name = f"distill_{task_name}_{student_model_type}"
        
        # Generate script sections
        header = self.generate_slurm_header()
        env_setup = self.generate_environment_setup()
        
        # Distillation command
        distillation_cmd = f"""# Distillation command
echo "Starting distillation for {task_name} with {student_model_type} student model..."

python -c "
import dna_distillation as dna

# Perform distillation
results = dna.distill_for_task(
    task_name='{task_name}',
    student_model_type='{student_model_type}',
    teacher_checkpoint_dir='{teacher_checkpoint_dir}',
    output_dir='{output_dir}',
    num_train_epochs={num_epochs},
    alpha={alpha},
    temperature={temperature},
    wandb_project='DNA-Distillation-SLURM',
    wandb_run_name='{task_name}_{student_model_type}_distill_slurm_$SLURM_JOB_ID'
)

print(f'Distillation completed!')
print(f'Final F1: {{results[\"final_f1\"]:.4f}}')
print(f'Final MCC: {{results[\"final_mcc\"]:.4f}}')
print(f'Student model saved to: {output_dir}')
"

echo "Distillation job completed!"
"""
        
        return f"{header}\n\n{env_setup}\n{distillation_cmd}"
    
    def generate_multi_task_job(
        self,
        task_names: List[str],
        model_type: str,
        output_base_dir: str,
        job_type: str = "train",  # "train" or "distill"
        teacher_checkpoint_dir: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate SLURM job script for multi-task training/distillation.
        
        Args:
            task_names: List of task names
            model_type: Type of model
            output_base_dir: Base output directory
            job_type: Type of job ("train" or "distill")
            teacher_checkpoint_dir: Teacher checkpoint directory (for distillation)
            **kwargs: Additional arguments
            
        Returns:
            Complete SLURM job script
        """
        # Update job name
        self.config.job_name = f"{job_type}_multi_{model_type}"
        
        # Generate script sections
        header = self.generate_slurm_header()
        env_setup = self.generate_environment_setup()
        
        # Multi-task command
        if job_type == "train":
            multi_task_cmd = f"""# Multi-task training command
echo "Starting multi-task training for {len(task_names)} tasks with {model_type} model..."

python -c "
import dna_distillation as dna

# Train on multiple tasks
results = dna.train_multiple_tasks(
    task_names={task_names},
    model_type='{model_type}',
    output_base_dir='{output_base_dir}',
    wandb_project='DNA-Distillation-SLURM-Multi',
    wandb_run_name='multi_{model_type}_slurm_$SLURM_JOB_ID'
)

print(f'Multi-task training completed!')
for task_name, task_results in results.items():
    if 'error' not in task_results:
        print(f'{{task_name}}: F1={{task_results[\"final_f1\"]:.4f}}, MCC={{task_results[\"final_mcc\"]:.4f}}')
    else:
        print(f'{{task_name}}: Error - {{task_results[\"error\"]}}')
"
"""
        else:  # distillation
            if not teacher_checkpoint_dir:
                raise ValueError("teacher_checkpoint_dir is required for distillation")
            
            multi_task_cmd = f"""# Multi-task distillation command
echo "Starting multi-task distillation for {len(task_names)} tasks with {model_type} student model..."

python -c "
import dna_distillation as dna

# Distill on multiple tasks
results = dna.distill_multiple_tasks(
    task_names={task_names},
    student_model_type='{model_type}',
    teacher_checkpoint_dir='{teacher_checkpoint_dir}',
    output_base_dir='{output_base_dir}',
    wandb_project='DNA-Distillation-SLURM-Multi',
    wandb_run_name='multi_{model_type}_distill_slurm_$SLURM_JOB_ID'
)

print(f'Multi-task distillation completed!')
for task_name, task_results in results.items():
    if 'error' not in task_results:
        print(f'{{task_name}}: F1={{task_results[\"final_f1\"]:.4f}}, MCC={{task_results[\"final_mcc\"]:.4f}}')
    else:
        print(f'{{task_name}}: Error - {{task_results[\"error\"]}}')
"
"""
        
        return f"{header}\n\n{env_setup}\n{multi_task_cmd}"
    
    def save_job_script(self, script_content: str, filename: str) -> str:
        """
        Save job script to file.
        
        Args:
            script_content: Job script content
            filename: Output filename
            
        Returns:
            Path to saved script
        """
        script_path = Path(filename)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return str(script_path)


def create_training_job(
    task_name: str,
    model_type: str,
    output_dir: str,
    config: Optional[SLURMConfig] = None,
    **kwargs
) -> str:
    """
    Create a SLURM job script for training.
    
    Args:
        task_name: Name of the task
        model_type: Type of model
        output_dir: Output directory
        config: SLURM configuration
        **kwargs: Additional training arguments
        
    Returns:
        Path to created job script
    """
    generator = SLURMJobGenerator(config)
    script_content = generator.generate_training_job(
        task_name=task_name,
        model_type=model_type,
        output_dir=output_dir,
        **kwargs
    )
    
    filename = f"slurm_train_{task_name}_{model_type}.sh"
    return generator.save_job_script(script_content, filename)


def create_distillation_job(
    task_name: str,
    student_model_type: str,
    teacher_checkpoint_dir: str,
    output_dir: str,
    config: Optional[SLURMConfig] = None,
    **kwargs
) -> str:
    """
    Create a SLURM job script for distillation.
    
    Args:
        task_name: Name of the task
        student_model_type: Type of student model
        teacher_checkpoint_dir: Teacher checkpoint directory
        output_dir: Output directory
        config: SLURM configuration
        **kwargs: Additional distillation arguments
        
    Returns:
        Path to created job script
    """
    generator = SLURMJobGenerator(config)
    script_content = generator.generate_distillation_job(
        task_name=task_name,
        student_model_type=student_model_type,
        teacher_checkpoint_dir=teacher_checkpoint_dir,
        output_dir=output_dir,
        **kwargs
    )
    
    filename = f"slurm_distill_{task_name}_{student_model_type}.sh"
    return generator.save_job_script(script_content, filename)


def create_multi_task_job(
    task_names: List[str],
    model_type: str,
    output_base_dir: str,
    job_type: str = "train",
    teacher_checkpoint_dir: Optional[str] = None,
    config: Optional[SLURMConfig] = None,
    **kwargs
) -> str:
    """
    Create a SLURM job script for multi-task training/distillation.
    
    Args:
        task_names: List of task names
        model_type: Type of model
        output_base_dir: Base output directory
        job_type: Type of job ("train" or "distill")
        teacher_checkpoint_dir: Teacher checkpoint directory (for distillation)
        config: SLURM configuration
        **kwargs: Additional arguments
        
    Returns:
        Path to created job script
    """
    generator = SLURMJobGenerator(config)
    script_content = generator.generate_multi_task_job(
        task_names=task_names,
        model_type=model_type,
        output_base_dir=output_base_dir,
        job_type=job_type,
        teacher_checkpoint_dir=teacher_checkpoint_dir,
        **kwargs
    )
    
    filename = f"slurm_{job_type}_multi_{model_type}.sh"
    return generator.save_job_script(script_content, filename)


def get_default_slurm_configs() -> Dict[str, SLURMConfig]:
    """
    Get default SLURM configurations for different scenarios.
    
    Returns:
        Dictionary of default configurations
    """
    configs = {
        "gpu_single": SLURMConfig(
            job_name="dna_distillation_gpu",
            gres="gpu:1",
            mem="32G",
            cpus_per_task=4,
            partition="gpu",
            time="24:00:00"
        ),
        "gpu_multi": SLURMConfig(
            job_name="dna_distillation_gpu_multi",
            gres="gpu:2",
            mem="64G",
            cpus_per_task=8,
            partition="gpu",
            time="48:00:00"
        ),
        "cpu": SLURMConfig(
            job_name="dna_distillation_cpu",
            mem="64G",
            cpus_per_task=16,
            partition="cpu",
            time="72:00:00"
        ),
        "debug": SLURMConfig(
            job_name="dna_distillation_debug",
            gres="gpu:1",
            mem="16G",
            cpus_per_task=2,
            partition="debug",
            time="02:00:00"
        )
    }
    
    return configs


def submit_job(script_path: str, dry_run: bool = False) -> str:
    """
    Submit a SLURM job.
    
    Args:
        script_path: Path to job script
        dry_run: If True, only print the command without submitting
        
    Returns:
        Job submission command or job ID
    """
    cmd = f"sbatch {script_path}"
    
    if dry_run:
        print(f"Would run: {cmd}")
        return cmd
    else:
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"Job submitted successfully! Job ID: {job_id}")
            return job_id
        else:
            print(f"Job submission failed: {result.stderr}")
            return "" 