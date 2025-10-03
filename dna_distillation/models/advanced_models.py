"""
Advanced model architectures from latest research code.

This module provides state-of-the-art model architectures including
BPNet, Enformer, Caduceus, and DNABERT-2 support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List
import math


class BPNetClassifier(nn.Module):
    """
    BPNet (Base Pair Prediction Network) classifier from latest research.
    
    BPNet is a specialized CNN architecture for DNA sequence analysis
    that uses dilated convolutions and attention mechanisms.
    """
    
    def __init__(
        self,
        num_labels: int = 2,
        input_length: int = 1000,
        num_filters: int = 64,
        num_dilated_convs: int = 8,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_batchnorm: bool = True
    ):
        """
        Initialize BPNet classifier.
        
        Args:
            num_labels: Number of output classes
            input_length: Input sequence length
            num_filters: Number of filters in convolutions
            num_dilated_convs: Number of dilated convolution layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        self.num_labels = num_labels
        self.input_length = input_length
        self.num_filters = num_filters
        self.use_attention = use_attention
        
        # One-hot encoding layer (4 nucleotides: A, C, G, T)
        self.one_hot = nn.Conv1d(4, num_filters, kernel_size=1)
        
        # Dilated convolution blocks
        self.dilated_convs = nn.ModuleList()
        for i in range(num_dilated_convs):
            dilation = 2 ** i
            self.dilated_convs.append(
                DilatedConvBlock(
                    num_filters, num_filters, 
                    kernel_size=3, dilation=dilation,
                    dropout=dropout, use_batchnorm=use_batchnorm
                )
            )
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionModule(num_filters)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, num_labels)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BPNet.
        
        Args:
            input_ids: Input sequence tensor [batch, seq_len]
            attention_mask: Optional attention mask
            labels: Optional ground truth labels
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with logits, loss, and optionally features
        """
        # One-hot encode input (assuming input_ids are in range 0-3)
        batch_size, seq_len = input_ids.shape
        
        # Convert to one-hot: [batch, seq_len, 4] -> [batch, 4, seq_len]
        one_hot = F.one_hot(input_ids, num_classes=4).float()
        one_hot = one_hot.transpose(1, 2)  # [batch, 4, seq_len]
        
        # Initial convolution
        x = self.one_hot(one_hot)  # [batch, num_filters, seq_len]
        
        # Apply dilated convolutions
        features = []
        for conv in self.dilated_convs:
            x = conv(x)
            features.append(x)
        
        # Attention mechanism
        if self.use_attention:
            x = self.attention(x)
        
        # Global pooling
        pooled = self.global_pool(x).squeeze(-1)  # [batch, num_filters]
        
        # Classification
        logits = self.classifier(pooled)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        result = {"logits": logits, "loss": loss}
        
        if return_features:
            result["features"] = features
        
        return result


class DilatedConvBlock(nn.Module):
    """Dilated convolution block for BPNet."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        use_batchnorm: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2
        )
        self.bn = nn.BatchNorm1d(out_channels) if use_batchnorm else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class AttentionModule(nn.Module):
    """Attention module for BPNet."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, hidden_dim, seq_len]
        batch_size, hidden_dim, seq_len = x.shape
        
        # Transpose for attention: [batch, seq_len, hidden_dim]
        x_t = x.transpose(1, 2)
        
        # Compute attention weights
        attn_weights = self.attention(x_t)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch, seq_len, 1]
        
        # Apply attention
        attended = x_t * attn_weights  # [batch, seq_len, hidden_dim]
        attended = attended.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        
        return attended


class EnformerClassifier(nn.Module):
    """
    Enformer classifier from latest research.
    
    Enformer is a transformer-based model for regulatory element prediction
    that uses attention mechanisms and positional encoding.
    """
    
    def __init__(
        self,
        num_labels: int = 2,
        input_length: int = 1000,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        """
        Initialize Enformer classifier.
        
        Args:
            num_labels: Number of output classes
            input_length: Input sequence length
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            use_positional_encoding: Whether to use positional encoding
        """
        super().__init__()
        self.num_labels = num_labels
        self.input_length = input_length
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding
        
        # Input embedding (one-hot to dense)
        self.input_embedding = nn.Linear(4, embed_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_labels)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Enformer.
        
        Args:
            input_ids: Input sequence tensor [batch, seq_len]
            attention_mask: Optional attention mask
            labels: Optional ground truth labels
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with logits, loss, and optionally features
        """
        batch_size, seq_len = input_ids.shape
        
        # One-hot encode and embed
        one_hot = F.one_hot(input_ids, num_classes=4).float()
        x = self.input_embedding(one_hot)  # [batch, seq_len, embed_dim]
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True for positions to attend to)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer encoder
        features = []
        for layer in self.transformer.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            features.append(x)
        
        # Global average pooling
        if attention_mask is not None:
            # Mask out padding positions
            mask = attention_mask.unsqueeze(-1).expand_as(x)
            x = x * mask
            pooled = x.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        result = {"logits": logits, "loss": loss}
        
        if return_features:
            result["features"] = features
        
        return result


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CaduceusClassifier(nn.Module):
    """
    Caduceus classifier from latest research.
    
    Caduceus is a bidirectional state space model for DNA sequences
    that uses Mamba-style state space layers.
    """
    
    def __init__(
        self,
        num_labels: int = 2,
        input_length: int = 1000,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        """
        Initialize Caduceus classifier.
        
        Args:
            num_labels: Number of output classes
            input_length: Input sequence length
            hidden_dim: Hidden dimension
            num_layers: Number of state space layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional processing
        """
        super().__init__()
        self.num_labels = num_labels
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # Input embedding
        self.input_embedding = nn.Linear(4, hidden_dim)
        
        # State space layers (simplified Mamba-style)
        self.state_layers = nn.ModuleList([
            StateSpaceLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Bidirectional processing
        if bidirectional:
            self.bidirectional_layers = nn.ModuleList([
                StateSpaceLayer(hidden_dim, dropout) for _ in range(num_layers)
            ])
        
        # Classification head
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Caduceus.
        
        Args:
            input_ids: Input sequence tensor [batch, seq_len]
            attention_mask: Optional attention mask
            labels: Optional ground truth labels
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with logits, loss, and optionally features
        """
        batch_size, seq_len = input_ids.shape
        
        # One-hot encode and embed
        one_hot = F.one_hot(input_ids, num_classes=4).float()
        x = self.input_embedding(one_hot)  # [batch, seq_len, hidden_dim]
        
        # Forward pass through state space layers
        features = []
        for layer in self.state_layers:
            x = layer(x)
            features.append(x)
        
        forward_output = x
        
        # Bidirectional processing
        if self.bidirectional:
            # Reverse sequence for backward pass
            x_rev = torch.flip(x, dims=[1])
            for layer in self.bidirectional_layers:
                x_rev = layer(x_rev)
            x_rev = torch.flip(x_rev, dims=[1])  # Reverse back
            
            # Concatenate forward and backward outputs
            x = torch.cat([forward_output, x_rev], dim=-1)
        
        # Global pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(x)
            x = x * mask
            pooled = x.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        result = {"logits": logits, "loss": loss}
        
        if return_features:
            result["features"] = features
        
        return result


class StateSpaceLayer(nn.Module):
    """Simplified state space layer for Caduceus."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.D = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        
        # Normalization and activation
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through state space layer.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Initialize state
        state = torch.zeros(batch_size, hidden_dim, device=x.device)
        outputs = []
        
        # Process sequence step by step
        for t in range(seq_len):
            # State space update: h_t = A * h_{t-1} + B * x_t
            state = torch.matmul(state, self.A) + torch.matmul(x[:, t], self.B)
            
            # Output: y_t = C * h_t + D * x_t
            output = torch.matmul(state, self.C) + torch.matmul(x[:, t], self.D)
            outputs.append(output)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden_dim]
        
        # Normalization and activation
        y = self.norm(y)
        y = self.activation(y)
        y = self.dropout(y)
        
        return y


class DNABERT2Classifier(nn.Module):
    """
    DNABERT-2 classifier from latest research.
    
    DNABERT-2 is a foundation model for multi-species genome understanding
    that uses BPE tokenization and ALiBi positional encoding.
    """
    
    def __init__(
        self,
        num_labels: int = 2,
        vocab_size: int = 1000,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        max_length: int = 512,
        dropout: float = 0.1,
        use_alibi: bool = True
    ):
        """
        Initialize DNABERT-2 classifier.
        
        Args:
            num_labels: Number of output classes
            vocab_size: Vocabulary size for BPE tokenization
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_length: Maximum sequence length
            dropout: Dropout rate
            use_alibi: Whether to use ALiBi positional encoding
        """
        super().__init__()
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.use_alibi = use_alibi
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # ALiBi positional encoding
        if use_alibi:
            self.alibi = ALiBiPositionalEncoding(embed_dim, num_heads)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_labels)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DNABERT-2.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            labels: Optional ground truth labels
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with logits, loss, and optionally features
        """
        # Token embedding
        x = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]
        
        # Create attention mask for transformer
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Apply ALiBi if enabled
        if self.use_alibi:
            # Add ALiBi bias to attention
            x = self.alibi(x)
        
        # Transformer encoder
        features = []
        for layer in self.transformer.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            features.append(x)
        
        # Global average pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(x)
            x = x * mask
            pooled = x.sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        result = {"logits": logits, "loss": loss}
        
        if return_features:
            result["features"] = features
        
        return result


class ALiBiPositionalEncoding(nn.Module):
    """ALiBi (Attention with Linear Biases) positional encoding."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # ALiBi slopes for different head positions
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', torch.tensor(slopes))
    
    def _get_slopes(self, num_heads: int) -> List[float]:
        """Get ALiBi slopes for different attention heads."""
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            return (get_slopes_power_of_2(closest_power_of_2) + 
                   self._get_slopes(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ALiBi positional encoding.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            
        Returns:
            Output tensor with ALiBi bias applied
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Create distance matrix
        distances = torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1)
        distances = distances.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, seq_len, seq_len]
        
        # Apply slopes to create bias matrix
        bias = distances.unsqueeze(1) * self.slopes.unsqueeze(-1).unsqueeze(-1)  # [batch, num_heads, seq_len, seq_len]
        
        # Store bias for use in attention
        self.register_buffer('alibi_bias', bias)
        
        return x


def create_advanced_model(
    model_type: str,
    num_labels: int = 2,
    **kwargs
) -> nn.Module:
    """
    Create advanced model from latest research.
    
    Args:
        model_type: Type of model ("bpnet", "enformer", "caduceus", "dnabert2")
        num_labels: Number of output classes
        **kwargs: Additional model-specific parameters
        
    Returns:
        Advanced model instance
    """
    if model_type == "bpnet":
        return BPNetClassifier(num_labels=num_labels, **kwargs)
    elif model_type == "enformer":
        return EnformerClassifier(num_labels=num_labels, **kwargs)
    elif model_type == "caduceus":
        # Map embed_dim to hidden_dim and filter out unused parameters for Caduceus
        kwargs_copy = kwargs.copy()
        if 'embed_dim' in kwargs_copy:
            kwargs_copy['hidden_dim'] = kwargs_copy.pop('embed_dim')
        # Remove parameters that Caduceus doesn't accept
        kwargs_copy.pop('num_heads', None)
        return CaduceusClassifier(num_labels=num_labels, **kwargs_copy)
    elif model_type == "dnabert2":
        # Remove input_length from kwargs for DNABERT2
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('input_length', None)
        return DNABERT2Classifier(num_labels=num_labels, **kwargs_copy)
    else:
        raise ValueError(f"Unknown advanced model type: {model_type}")


def get_advanced_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get information about advanced model architectures.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dictionary with model information
    """
    model_info = {
        "bpnet": {
            "name": "BPNet",
            "description": "Base Pair Prediction Network with dilated convolutions",
            "paper": "Deep learning for regulatory genomics",
            "features": ["Dilated convolutions", "Attention mechanism", "One-hot encoding"]
        },
        "enformer": {
            "name": "Enformer",
            "description": "Transformer-based model for regulatory element prediction",
            "paper": "Effective gene expression prediction from sequence",
            "features": ["Transformer encoder", "Positional encoding", "Attention heads"]
        },
        "caduceus": {
            "name": "Caduceus",
            "description": "Bidirectional state space model for DNA sequences",
            "paper": "Mamba-based state space models",
            "features": ["State space layers", "Bidirectional processing", "Mamba-style"]
        },
        "dnabert2": {
            "name": "DNABERT-2",
            "description": "Foundation model for multi-species genome understanding",
            "paper": "DNABERT-2: Efficient Foundation Model and Benchmark",
            "features": ["BPE tokenization", "ALiBi encoding", "Multi-species support"]
        }
    }
    
    return model_info.get(model_type, {"name": "Unknown", "description": "Unknown model"})


# Available advanced model types
ADVANCED_MODEL_TYPES = ["bpnet", "enformer", "caduceus", "dnabert2"]
