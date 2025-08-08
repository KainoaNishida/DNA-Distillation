"""Utility functions for DNA Distillation."""

from .slurm import (
    SLURMConfig,
    SLURMJobGenerator,
    create_training_job,
    create_distillation_job,
    create_multi_task_job,
    get_default_slurm_configs,
    submit_job,
)

__all__ = [
    "SLURMConfig",
    "SLURMJobGenerator", 
    "create_training_job",
    "create_distillation_job",
    "create_multi_task_job",
    "get_default_slurm_configs",
    "submit_job",
]
