import torch.nn as nn
import torch.nn.functional as F


class Model2CNN(nn.Module):
    """CNN with residual connections and batch normalization."""
    
    def __init__(self):
        super(Model2CNN, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second block with residual connection
        self.conv2a = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(16)
        self.conv2b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(16)
        
        # Final layers
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 7 * 7, 10)
        
    def forward(self, x):
        # First block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        
        # Residual block
        identity = x
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + identity)  # Add residual connection
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Final convolution
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 8 * 7 * 7)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 