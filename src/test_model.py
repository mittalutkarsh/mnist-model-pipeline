import torch
import os
import pytest
from src.models import get_model, list_available_models
from src.train import train


def get_test_models():
    """Return list of model configurations to test."""
    return list_available_models()


@pytest.mark.parametrize("model_name", get_test_models())
def test_model_parameters(model_name):
    """Test that model architecture meets parameter count requirements."""
    try:
        ModelClass = get_model(model_name)
    except ValueError as e:
        pytest.skip(f"Skipping test: {str(e)}")
        
    model = ModelClass()
    
    # Force print to stdout
    import sys
    
    sys.stdout.write(f"\nTesting model: {model_name}\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("MODEL ARCHITECTURE DETAILS\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.flush()
    
    total_params = 0
    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        total_params += param_count
        shape_str = str(tuple(parameter.shape))
        param_str = (
            f"Layer: {name:20} | Shape: {shape_str:20} | "
            f"Parameters: {param_count:,}\n"
        )
        sys.stdout.write(param_str)
        sys.stdout.flush()
    
    sys.stdout.write("-" * 60 + "\n")
    sys.stdout.write(f"Total Parameters: {total_params:,}\n")
    sys.stdout.write("=" * 60 + "\n\n")
    sys.stdout.flush()
    
    assert total_params < 25000, (
        f"Model {model_name} has {total_params:,} parameters, "
        "which exceeds limit of 25,000"
    )


@pytest.mark.parametrize("model_name", get_test_models())
def test_model_io_shapes(model_name):
    """Test input/output shapes of the model."""
    ModelClass = get_model(model_name)
    model = ModelClass()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    assert output.shape == (1, 10), (
        f"Model {model_name} output shape should be (batch_size, 10)"
    )


@pytest.mark.parametrize("model_name", get_test_models())
def test_model_training(model_name):
    """Test model training achieves required accuracy."""
    import sys
    
    sys.stdout.write("\n" + "=" * 60 + "\n")
    sys.stdout.write(f"TRAINING RESULTS: {model_name}\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.flush()
    
    accuracy = train(model_name)
    
    sys.stdout.write(f"Final Accuracy: {accuracy:.4%}\n")
    sys.stdout.write("=" * 60 + "\n\n")
    sys.stdout.flush()
    
    assert accuracy > 0.95, (
        f"Model {model_name} accuracy {accuracy:.4%} is below 95%"
    )


@pytest.mark.parametrize("model_name", get_test_models())
def test_model_save_load(model_name):
    """Test model state can be properly saved and loaded."""
    ModelClass = get_model(model_name)
    model = ModelClass()
    test_input = torch.randn(1, 1, 28, 28)
    
    # Test forward pass
    output_before = model(test_input)
    
    # Save and load
    torch.save(model.state_dict(), "test_model.pth")
    model.load_state_dict(torch.load("test_model.pth"))
    
    # Test forward pass after loading
    output_after = model(test_input)
    
    assert torch.allclose(output_before, output_after)
    os.remove("test_model.pth") 