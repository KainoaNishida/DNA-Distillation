"""
Student model architectures for DNA sequence classification.

This module contains various neural network architectures that can be used
as student models in knowledge distillation or trained independently.
All models are optimized for DNA sequence analysis tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from transformers import AutoTokenizer


class DNAStudentModel(nn.Module):
    """Base class for all DNA student models."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_labels: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim  
        self.num_labels = num_labels
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass that returns loss and logits."""
        raise NotImplementedError
        
    def get_num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BiLSTMStudent(DNAStudentModel):
    """Bidirectional LSTM student model for DNA sequence classification."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int, 
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(lstm_output_dim, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.bilstm(packed_embedded)
        else:
            _, (hidden, _) = self.bilstm(embedded)
            
        if self.bilstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
            
        logits = self.classifier(hidden)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


class XLSTMStudent(DNAStudentModel):
    """LSTM with attention mechanism for DNA sequence classification."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int, 
        num_labels: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers, 
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Linear(lstm_output_dim, 1)
        self.classifier = nn.Linear(lstm_output_dim, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)
        
        # Apply attention mechanism
        attn_weights = torch.softmax(self.attention(outputs).squeeze(-1), dim=1)
        context = torch.sum(outputs * attn_weights.unsqueeze(-1), dim=1)
        
        logits = self.classifier(context)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


class MambaSSM(DNAStudentModel):
    """Mamba State Space Model (simulated with GRU) for DNA sequences."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        _, hidden = self.gru(embedded)
        hidden = hidden.squeeze(0)
        hidden = self.dropout(hidden)
        
        logits = self.classifier(hidden)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


class HyenaSSM(DNAStudentModel):
    """Hyena State Space Model (GRU + Conv1D) for DNA sequences."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int, 
        hidden_dim: int,
        num_labels: int,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        outputs, _ = self.gru(embedded)
        
        # Apply convolution
        conv_input = outputs.transpose(1, 2)  # (batch, hidden, seq_len)
        conv_out = F.relu(self.conv1(conv_input))
        pooled = torch.mean(conv_out, dim=2)  # Global average pooling
        pooled = self.dropout(pooled)
        
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


class CaduceusSSM(DNAStudentModel):
    """Caduceus State Space Model (multi-layer GRU) for DNA sequences."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        _, hidden = self.gru(embedded)
        hidden_last = hidden[-1]  # Take the last layer's hidden state
        hidden_last = self.dropout(hidden_last)
        
        logits = self.classifier(hidden_last)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


class CNNStudent(DNAStudentModel):
    """CNN-based student model for DNA sequence classification."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        num_labels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_layers: int = 3,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = embed_dim
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2)
            )
            in_channels = num_filters
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        x = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        # Apply CNN layers
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            
        # Global pooling
        pooled = self.global_pool(x).squeeze(-1)  # (batch, num_filters)
        pooled = self.dropout(pooled)
        
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


class MLPStudent(DNAStudentModel):
    """Multi-Layer Perceptron student model for DNA sequences."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        
        # MLP layers
        layers = []
        in_dim = embed_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, num_labels))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(embedded, dim=1)  # (batch, embed_dim)
        
        logits = self.mlp(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


class RNNStudent(DNAStudentModel):
    """Simple RNN student model for DNA sequences."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        _, hidden = self.rnn(embedded)
        hidden_last = hidden[-1]  # Take the last layer's hidden state
        hidden_last = self.dropout(hidden_last)
        
        logits = self.classifier(hidden_last)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


# Model factory function
def create_student_model(
    model_type: str,
    vocab_size: int,
    embed_dim: int,
    hidden_dim: int,
    num_labels: int,
    padding_idx: Optional[int] = None,
    **kwargs
) -> DNAStudentModel:
    """Factory function to create student models by type."""
    
    model_map = {
        "bilstm": BiLSTMStudent,
        "xlstm": XLSTMStudent, 
        "mamba": MambaSSM,
        "hyena": HyenaSSM,
        "caduceus": CaduceusSSM,
        "cnn": CNNStudent,
        "mlp": MLPStudent,
        "rnn": RNNStudent,
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
        
    ModelClass = model_map[model_type]
    
    # Handle different parameter names for different models
    if model_type == "cnn":
        kwargs.pop("hidden_dim", None)  # CNN uses num_filters instead
        return ModelClass(
            vocab_size=vocab_size,
            embed_dim=embed_dim, 
            num_filters=hidden_dim,  # Use hidden_dim as num_filters
            num_labels=num_labels,
            padding_idx=padding_idx,
            **kwargs
        )
    else:
        return ModelClass(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            padding_idx=padding_idx,
            **kwargs
        )
