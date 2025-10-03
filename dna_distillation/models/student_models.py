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


class ResidualBlock(nn.Module):
    """Residual block for CNN with optional pooling."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dropout: float, pool: bool = False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None
            
        # Optional pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        
        if self.pool is not None and x.size(-1) > 1:  # Only pool if sequence length > 1
            out = self.pool(out)
        return out


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
    """CNN-based student model with residual blocks for DNA sequence classification."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        num_labels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_res_blocks: int = 16,
        use_one_hot: bool = False,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_res_blocks = num_res_blocks
        self.use_one_hot = use_one_hot
        
        if not self.use_one_hot:
            # Use standard embedding layer
            self.embedding = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx
            )
            in_channels = embed_dim
        else:
            # No embedding layer needed; we'll one-hot encode inputs
            self.embedding = None
            in_channels = vocab_size  # One-hot vectors have dimension equal to vocab size

        # Initial convolution to go from input channels to num_filters
        self.initial_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Build a sequence of residual blocks (pooling every 4 blocks)
        res_blocks = []
        for i in range(num_res_blocks):
            pool = (i + 1) % 4 == 0  # Pool every 4th block
            res_blocks.append(
                ResidualBlock(num_filters, num_filters, kernel_size, dropout, pool=pool)
            )
        self.res_blocks = nn.Sequential(*res_blocks)

        # Global pooling and classification layers
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # Use MaxPool instead of AvgPool
        self.fc = nn.Linear(num_filters, num_filters)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if not self.use_one_hot:
            # Use the embedding layer to convert token IDs to vectors
            embedded = self.embedding(input_ids)  # Shape: (batch, seq_len, embed_dim)
        else:
            # One-hot encode the input tokens
            embedded = F.one_hot(input_ids, num_classes=self.initial_conv.in_channels).float()

        # Transpose to shape (batch, channels, seq_len) for Conv1d
        x = embedded.transpose(1, 2)
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        
        logits = self.classifier(x)
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
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        avg_emb = torch.mean(embedded, dim=1)  # Global average pooling
        x = self.dropout(F.relu(self.fc1(avg_emb)))
        logits = self.fc2(x)
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
            nonlinearity="tanh",  # Explicit nonlinearity specification
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        outputs, hidden = self.rnn(embedded)
        hidden_last = hidden[-1]  # Take the last layer's hidden state
        logits = self.classifier(hidden_last)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


class BPNetStudent(DNAStudentModel):
    """BPNet student model for DNA sequence classification."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_labels: int,
        input_length: int = 1000,
        num_filters: int = 64,
        num_dilated_convs: int = 8,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_batchnorm: bool = True,
        **kwargs
    ):
        super().__init__(vocab_size, embed_dim, num_labels)
        self.input_length = input_length
        self.num_filters = num_filters
        
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
        else:
            self.attention = None
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, num_labels)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # One-hot encode input (assuming input_ids are in range 0-3)
        batch_size, seq_len = input_ids.shape
        
        # Convert to one-hot: [batch, seq_len, 4] -> [batch, 4, seq_len]
        one_hot = F.one_hot(input_ids, num_classes=4).float()
        one_hot = one_hot.transpose(1, 2)  # [batch, 4, seq_len]
        
        # Initial convolution
        x = self.one_hot(one_hot)  # [batch, num_filters, seq_len]
        
        # Apply dilated convolutions
        for conv in self.dilated_convs:
            x = conv(x)
        
        # Attention mechanism
        if self.attention is not None:
            x = self.attention(x)
        
        # Global pooling
        pooled = self.global_pool(x).squeeze(-1)  # [batch, num_filters]
        
        # Classification
        logits = self.classifier(pooled)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}


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
        "bpnet": BPNetStudent,
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")
        
    ModelClass = model_map[model_type]
    
    # Handle different parameter names for different models
    if model_type == "cnn":
        # Extract CNN-specific parameters
        num_filters = kwargs.pop("num_filters", hidden_dim)  # Use hidden_dim as num_filters if not specified
        num_res_blocks = kwargs.pop("num_res_blocks", 16)
        use_one_hot = kwargs.pop("use_one_hot", False)
        kernel_size = kwargs.pop("kernel_size", 3)
        # Remove parameters that CNN doesn't use
        kwargs.pop("num_layers", None)
        
        return ModelClass(
            vocab_size=vocab_size,
            embed_dim=embed_dim, 
            num_filters=num_filters,
            num_labels=num_labels,
            kernel_size=kernel_size,
            num_res_blocks=num_res_blocks,
            use_one_hot=use_one_hot,
            padding_idx=padding_idx,
            **kwargs
        )
    elif model_type == "bpnet":
        # Extract BPNet-specific parameters
        input_length = kwargs.pop("input_length", 1000)
        num_filters = kwargs.pop("num_filters", 64)
        num_dilated_convs = kwargs.pop("num_dilated_convs", 8)
        dropout = kwargs.pop("dropout", 0.1)
        use_attention = kwargs.pop("use_attention", True)
        use_batchnorm = kwargs.pop("use_batchnorm", True)
        # Remove parameters that BPNet doesn't use
        kwargs.pop("num_layers", None)
        kwargs.pop("hidden_dim", None)
        
        return ModelClass(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_labels=num_labels,
            input_length=input_length,
            num_filters=num_filters,
            num_dilated_convs=num_dilated_convs,
            dropout=dropout,
            use_attention=use_attention,
            use_batchnorm=use_batchnorm,
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
