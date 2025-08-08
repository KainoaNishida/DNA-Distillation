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
