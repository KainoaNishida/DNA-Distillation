"""
Test model creation and functionality.
"""

import pytest
import torch
import dna_distillation as dna


def test_create_student_model():
    """Test creating student models."""
    # Test BiLSTM
    model = dna.create_student_model(
        model_type="bilstm",
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
        num_labels=2
    )
    assert isinstance(model, dna.BiLSTMStudent)
    assert model.get_num_parameters() > 0
    
    # Test CNN
    model = dna.create_student_model(
        model_type="cnn",
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
        num_labels=2
    )
    assert isinstance(model, dna.CNNStudent)
    assert model.get_num_parameters() > 0
    
    # Test MLP
    model = dna.create_student_model(
        model_type="mlp",
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
        num_labels=2
    )
    assert isinstance(model, dna.models.MLPStudent)
    assert model.get_num_parameters() > 0


def test_model_forward_pass():
    """Test model forward pass."""
    model = dna.create_student_model(
        model_type="bilstm",
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
        num_labels=2
    )
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    output = model(input_ids, labels=labels)
    
    assert "loss" in output
    assert "logits" in output
    assert output["logits"].shape == (batch_size, 2)
    assert output["loss"] is not None


def test_model_parameter_counting():
    """Test model parameter counting."""
    model = dna.create_student_model(
        model_type="bilstm",
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
        num_labels=2
    )
    
    param_count = model.get_num_parameters()
    assert param_count > 0
    assert isinstance(param_count, int)


def test_invalid_model_type():
    """Test that invalid model type raises error."""
    with pytest.raises(ValueError):
        dna.create_student_model(
            model_type="invalid_model",
            vocab_size=1000,
            embed_dim=128,
            hidden_dim=64,
            num_labels=2
        )


def test_model_architectures():
    """Test all available model architectures."""
    model_types = ["bilstm", "xlstm", "mamba", "hyena", "caduceus", "cnn", "mlp", "rnn"]
    
    for model_type in model_types:
        model = dna.create_student_model(
            model_type=model_type,
            vocab_size=1000,
            embed_dim=128,
            hidden_dim=64,
            num_labels=2
        )
        assert model.get_num_parameters() > 0


def test_cnn_residual_blocks():
    """Test CNN with residual blocks."""
    model = dna.create_student_model(
        model_type="cnn",
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
        num_labels=2,
        num_res_blocks=8,
        use_one_hot=False
    )
    assert model.get_num_parameters() > 0
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 10))
    labels = torch.randint(0, 2, (2,))
    output = model(input_ids, labels=labels)
    assert "loss" in output
    assert "logits" in output
    assert output["logits"].shape == (2, 2)


def test_cnn_one_hot_encoding():
    """Test CNN with one-hot encoding."""
    model = dna.create_student_model(
        model_type="cnn",
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=64,
        num_labels=2,
        num_res_blocks=8,
        use_one_hot=True
    )
    assert model.get_num_parameters() > 0
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 10))
    labels = torch.randint(0, 2, (2,))
    output = model(input_ids, labels=labels)
    assert "loss" in output
    assert "logits" in output
    assert output["logits"].shape == (2, 2)


def test_residual_block():
    """Test ResidualBlock component."""
    from dna_distillation.models.student_models import ResidualBlock
    
    # Test residual block without pooling
    block = ResidualBlock(64, 64, 3, 0.1, pool=False)
    x = torch.randn(2, 64, 10)
    output = block(x)
    assert output.shape == x.shape
    
    # Test residual block with pooling
    block = ResidualBlock(64, 64, 3, 0.1, pool=True)
    x = torch.randn(2, 64, 10)
    output = block(x)
    assert output.shape == (2, 64, 5)  # Should be half the length due to pooling
