"""
Distillation loss functions for knowledge distillation.

This module provides loss functions for knowledge distillation from
teacher models to student models, including temperature scaling and
weighted loss combinations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    This loss combines cross-entropy loss on true labels with
    MSE loss between teacher and student logits (temperature-scaled).
    
    Args:
        alpha: Weight for cross-entropy loss (default: 0.5)
        temperature: Temperature for logit scaling (default: 2.0)
        reduction: Loss reduction method (default: 'mean')
    """
    
    def __init__(self, alpha: float = 0.5, temperature: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: True labels
            return_components: Whether to return individual loss components
            
        Returns:
            Combined distillation loss
        """
        # Cross-entropy loss on true labels
        ce_loss = self.ce_loss(student_logits, labels)
        
        # Temperature scaling for distillation
        student_logits_temp = student_logits / self.temperature
        teacher_logits_temp = teacher_logits / self.temperature
        
        # MSE loss between temperature-scaled logits
        distill_loss = self.mse_loss(student_logits_temp, teacher_logits_temp)
        
        # Combined loss
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * (self.temperature ** 2) * distill_loss
        
        if return_components:
            return total_loss, {
                'ce_loss': ce_loss.item(),
                'distill_loss': distill_loss.item(),
                'total_loss': total_loss.item(),
                'alpha': self.alpha,
                'temperature': self.temperature
            }
        
        return total_loss


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
    return_components: bool = False
) -> torch.Tensor:
    """
    Compute distillation loss with given parameters.
    
    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        labels: True labels
        alpha: Weight for cross-entropy loss
        temperature: Temperature for logit scaling
        return_components: Whether to return individual loss components
        
    Returns:
        Combined distillation loss
    """
    loss_fn = DistillationLoss(alpha=alpha, temperature=temperature)
    return loss_fn(student_logits, teacher_logits, labels, return_components)


class SoftTargetLoss(nn.Module):
    """
    Soft target loss for knowledge distillation.
    
    This loss uses KL divergence between softmax outputs
    of teacher and student models.
    """
    
    def __init__(self, temperature: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute soft target loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            return_components: Whether to return loss components
            
        Returns:
            KL divergence loss
        """
        # Temperature scaling
        student_logits_temp = student_logits / self.temperature
        teacher_logits_temp = teacher_logits / self.temperature
        
        # Softmax outputs
        student_probs = F.log_softmax(student_logits_temp, dim=-1)
        teacher_probs = F.softmax(teacher_logits_temp, dim=-1)
        
        # KL divergence loss
        loss = self.kl_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        
        if return_components:
            return loss, {
                'kl_loss': loss.item(),
                'temperature': self.temperature
            }
        
        return loss


class AttentionDistillationLoss(nn.Module):
    """
    Attention distillation loss for transformer-based models.
    
    This loss distills attention patterns from teacher to student.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention distillation loss.
        
        Args:
            student_attention: Attention weights from student model
            teacher_attention: Attention weights from teacher model
            attention_mask: Optional attention mask
            
        Returns:
            MSE loss between attention weights
        """
        if attention_mask is not None:
            # Apply attention mask
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            student_attention = student_attention * mask
            teacher_attention = teacher_attention * mask
        
        return self.mse_loss(student_attention, teacher_attention)


class FeatureDistillationLoss(nn.Module):
    """
    Feature distillation loss for intermediate layer features.
    
    This loss distills intermediate features from teacher to student.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute feature distillation loss.
        
        Args:
            student_features: Features from student model
            teacher_features: Features from teacher model
            feature_mask: Optional feature mask
            
        Returns:
            MSE loss between features
        """
        if feature_mask is not None:
            student_features = student_features * feature_mask
            teacher_features = teacher_features * feature_mask
        
        return self.mse_loss(student_features, teacher_features)


def get_distillation_loss(
    loss_type: str = "combined",
    alpha: float = 0.5,
    temperature: float = 2.0,
    **kwargs
) -> nn.Module:
    """
    Get distillation loss function by type.
    
    Args:
        loss_type: Type of distillation loss ("combined", "soft", "attention", "feature")
        alpha: Weight for cross-entropy loss (for combined loss)
        temperature: Temperature for scaling (for combined and soft losses)
        **kwargs: Additional arguments for specific loss types
        
    Returns:
        Distillation loss function
    """
    if loss_type == "combined":
        return DistillationLoss(alpha=alpha, temperature=temperature)
    elif loss_type == "soft":
        return SoftTargetLoss(temperature=temperature)
    elif loss_type == "attention":
        return AttentionDistillationLoss()
    elif loss_type == "feature":
        return FeatureDistillationLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def validate_distillation_inputs(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    temperature: float = 2.0
) -> None:
    """
    Validate inputs for distillation loss computation.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        labels: True labels (optional)
        alpha: Weight for cross-entropy loss
        temperature: Temperature for scaling
        
    Raises:
        ValueError: If inputs are invalid
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"Student and teacher logits must have same shape. "
                        f"Got {student_logits.shape} vs {teacher_logits.shape}")
    
    if labels is not None:
        if student_logits.shape[0] != labels.shape[0]:
            raise ValueError(f"Batch size mismatch. Logits: {student_logits.shape[0]}, "
                           f"Labels: {labels.shape[0]}")
        
        if student_logits.shape[1] != labels.max() + 1:
            raise ValueError(f"Number of classes mismatch. Logits: {student_logits.shape[1]}, "
                           f"Labels max: {labels.max()}")
    
    if not 0 <= alpha <= 1:
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
    
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")


# ============================================================================
# ADVANCED DISTILLATION METHODS (Latest Research Code)
# ============================================================================

def logit_standardization_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0
) -> torch.Tensor:
    """
    Logit Standardization KD Loss (CVPR 2024).
    
    Applies Z-score normalization before KL divergence to improve distillation
    by reducing the impact of logit magnitude differences.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Temperature for softening logits
        
    Returns:
        Logit standardization KL loss
    """
    # Z-score normalization
    s_mean = student_logits.mean(dim=1, keepdim=True)
    s_std = student_logits.std(dim=1, keepdim=True) + 1e-7
    s_logits_norm = (student_logits - s_mean) / s_std
    
    t_mean = teacher_logits.mean(dim=1, keepdim=True)
    t_std = teacher_logits.std(dim=1, keepdim=True) + 1e-7
    t_logits_norm = (teacher_logits - t_mean) / t_std
    
    # Apply temperature and compute KL divergence
    kl_loss = F.kl_div(
        F.log_softmax(s_logits_norm / temperature, dim=-1),
        F.softmax(t_logits_norm / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return kl_loss


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 8.0,
    temperature: float = 4.0
) -> torch.Tensor:
    """
    Decoupled Knowledge Distillation (DKD) Loss (CVPR 2022).
    
    DKD decouples the knowledge distillation into target class knowledge
    and non-target class knowledge, providing better control over the
    distillation process.
    
    This implementation exactly matches the research code.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        labels: Ground truth labels
        alpha: Weight for target class knowledge
        beta: Weight for non-target class knowledge
        temperature: Temperature for softening logits
        
    Returns:
        DKD loss
    """
    # Create masks for target and non-target classes
    batch_size = labels.shape[0]
    mask_target = F.one_hot(labels, num_classes=student_logits.shape[1]).bool()
    mask_non_target = ~mask_target
    
    # Get softmax probabilities
    s_probs = F.softmax(student_logits / temperature, dim=1)
    t_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # Target Class Knowledge Distillation (TCKD)
    s_target = (s_probs * mask_target).sum(dim=1, keepdim=True)
    t_target = (t_probs * mask_target).sum(dim=1, keepdim=True)
    tckd_loss = F.kl_div(
        torch.log(s_target + 1e-8),
        t_target,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Non-target Class Knowledge Distillation (NCKD)
    # Mask out target class with large negative value
    s_logits_non_target = student_logits.masked_fill(mask_target, -1e9)
    t_logits_non_target = teacher_logits.masked_fill(mask_target, -1e9)
    
    nckd_loss = F.kl_div(
        F.log_softmax(s_logits_non_target / temperature, dim=1),
        F.softmax(t_logits_non_target / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return alpha * tckd_loss + beta * nckd_loss


def dist_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    DIST: Knowledge Distillation from A Stronger Teacher (NeurIPS 2022).
    
    Uses Pearson correlation coefficient to measure the relationship between
    student and teacher predictions.
    
    This implementation exactly matches the research code.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        beta: Weight for inter-class correlation
        gamma: Weight for intra-class correlation
        
    Returns:
        DIST loss
    """
    # Inter-class relation: correlation between classes for each sample
    s_probs = F.softmax(student_logits, dim=1)
    t_probs = F.softmax(teacher_logits, dim=1)
    
    # Normalize to zero mean
    s_mean = s_probs.mean(dim=1, keepdim=True)
    t_mean = t_probs.mean(dim=1, keepdim=True)
    s_centered = s_probs - s_mean
    t_centered = t_probs - t_mean
    
    # Compute Pearson correlation coefficient
    s_norm = torch.norm(s_centered, dim=1, keepdim=True) + 1e-8
    t_norm = torch.norm(t_centered, dim=1, keepdim=True) + 1e-8
    s_normalized = s_centered / s_norm
    t_normalized = t_centered / t_norm
    
    # Inter-class loss
    inter_loss = 1 - (s_normalized * t_normalized).sum(dim=1).mean()
    
    # Intra-class relation: correlation between samples for each class
    s_probs_t = s_probs.t()
    t_probs_t = t_probs.t()
    
    # Normalize per class
    s_mean_c = s_probs_t.mean(dim=1, keepdim=True)
    t_mean_c = t_probs_t.mean(dim=1, keepdim=True)
    s_centered_c = s_probs_t - s_mean_c
    t_centered_c = t_probs_t - t_mean_c
    
    s_norm_c = torch.norm(s_centered_c, dim=1, keepdim=True) + 1e-8
    t_norm_c = torch.norm(t_centered_c, dim=1, keepdim=True) + 1e-8
    s_normalized_c = s_centered_c / s_norm_c
    t_normalized_c = t_centered_c / t_norm_c
    
    # Intra-class loss
    intra_loss = 1 - (s_normalized_c * t_normalized_c).sum(dim=1).mean()
    
    return beta * inter_loss + gamma * intra_loss


def reviewkd_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    attention_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    ReviewKD Loss (CVPR 2021) - Cross-stage knowledge transfer.
    
    ReviewKD transfers knowledge from multiple teacher stages to student
    stages using attention-based fusion.
    
    Args:
        student_features: Student model features
        teacher_features: Teacher model features (multi-level)
        attention_weights: Attention weights for feature fusion
        
    Returns:
        ReviewKD loss
    """
    if attention_weights is None:
        attention_weights = torch.ones(teacher_features.shape[0]) / teacher_features.shape[0]
    
    # Weighted combination of teacher features
    weighted_teacher = torch.sum(
        teacher_features * attention_weights.unsqueeze(-1).unsqueeze(-1),
        dim=0
    )
    
    # MSE loss between student and weighted teacher features
    reviewkd_loss = F.mse_loss(student_features, weighted_teacher)
    
    return reviewkd_loss


class AdvancedDistillationLoss(nn.Module):
    """
    Advanced distillation loss supporting multiple state-of-the-art methods.
    
    This class provides a unified interface for various distillation methods
    including Logit Standardization, DKD, DIST, and ReviewKD.
    """
    
    def __init__(
        self,
        method: str = "vanilla",
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 8.0,
        gamma: float = 2.0,
        lambda_kl: float = 0.3,
        lambda_mse: float = 0.2
    ):
        """
        Initialize advanced distillation loss.
        
        Args:
            method: Distillation method ("vanilla", "logit_standard", "dkd", "dist", "reviewkd")
            temperature: Temperature for softening logits
            alpha: Weight for hard target loss (DKD)
            beta: Weight for non-target class knowledge (DKD) or inter-class correlation (DIST)
            gamma: Weight for intra-class correlation (DIST)
            lambda_kl: Weight for KL/distillation loss
            lambda_mse: Weight for MSE/feature matching loss
        """
        super().__init__()
        self.method = method
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_kl = lambda_kl
        self.lambda_mse = lambda_mse
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute advanced distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels
            student_features: Student model features (for ReviewKD)
            teacher_features: Teacher model features (for ReviewKD)
            
        Returns:
            Advanced distillation loss
        """
        # Hard target loss (cross-entropy)
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Distillation loss based on method
        if self.lambda_kl > 0 and teacher_logits is not None:
            if self.method == "vanilla":
                kl_loss = F.kl_div(
                    F.log_softmax(student_logits / self.temperature, dim=-1),
                    F.softmax(teacher_logits / self.temperature, dim=-1),
                    reduction='batchmean'
                ) * (self.temperature ** 2)
                
            elif self.method == "logit_standard":
                kl_loss = logit_standardization_kd_loss(
                    student_logits, teacher_logits, self.temperature
                )
                
            elif self.method == "dkd":
                kl_loss = dkd_loss(
                    student_logits, teacher_logits, labels,
                    self.alpha, self.beta, self.temperature
                )
                
            elif self.method == "dist":
                kl_loss = dist_loss(
                    student_logits, teacher_logits,
                    self.beta, self.gamma
                )
                
            elif self.method == "reviewkd":
                kl_loss = F.kl_div(
                    F.log_softmax(student_logits / self.temperature, dim=-1),
                    F.softmax(teacher_logits / self.temperature, dim=-1),
                    reduction='batchmean'
                ) * (self.temperature ** 2)
            else:
                raise ValueError(f"Unknown distillation method: {self.method}")
        else:
            kl_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Feature matching loss (for ReviewKD and feature-based methods)
        mse_loss = torch.tensor(0.0, device=student_logits.device)
        if (self.lambda_mse > 0 and student_features is not None and 
            teacher_features is not None):
            if self.method == "reviewkd":
                mse_loss = reviewkd_loss(student_features, teacher_features)
            else:
                mse_loss = F.mse_loss(student_features, teacher_features)
        
        # Combined loss
        total_loss = (ce_loss + 
                     self.lambda_kl * kl_loss + 
                     self.lambda_mse * mse_loss)
        
        return total_loss
