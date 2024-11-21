from .simple_cnn import SimpleCNN
from .deeper_cnn import DeeperCNN
from .model1_cnn import Model1CNN
from .model2_cnn import Model2CNN

# Dictionary mapping model names to their classes
MODEL_REGISTRY = {
    'simple_cnn': SimpleCNN,
    'deeper_cnn': DeeperCNN,
    'model1_cnn': Model1CNN,
    'model2_cnn': Model2CNN,
}

def get_model(model_name):
    """Get model class by name."""
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available models: {available_models}"
        )
    return MODEL_REGISTRY[model_name]

def list_available_models():
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys()) 