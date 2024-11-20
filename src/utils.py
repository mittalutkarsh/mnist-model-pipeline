import torch
import os
from datetime import datetime
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def save_model(model, accuracy, base_path='models'):
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'mnist_model_{timestamp}_acc_{accuracy:.4f}.pth'
    path = os.path.join(base_path, filename)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'timestamp': timestamp
    }, path)
    
    return path

def load_model(path):
    checkpoint = torch.load(path)
    model = MNISTClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['accuracy']

def get_latest_model(base_path='models'):
    if not os.path.exists(base_path):
        return None
    
    models = [f for f in os.listdir(base_path) if f.endswith('.pth')]
    if not models:
        return None
        
    return os.path.join(base_path, sorted(models)[-1]) 