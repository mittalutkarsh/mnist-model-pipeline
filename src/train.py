import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from src.utils import setup_logging, save_model
from src.models import get_model


def train(model_name='simple_cnn'):
    """Train the specified model."""
    setup_logging()
    device = torch.device("cpu")
    
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        'data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    ModelClass = get_model(model_name)
    model = ModelClass().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    logging.info(f"Training {model_name}")
    logging.info(f"Total parameters: {model.count_parameters()}")
    
    # Training
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
    accuracy = correct / total
    logging.info(f'Training Accuracy: {accuracy:.4f}')
    
    # Save model
    save_path = save_model(model, accuracy)
    logging.info(f'Model saved to {save_path}')
    
    return accuracy


if __name__ == '__main__':
    train() 