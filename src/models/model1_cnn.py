import torch.nn as nn
import torch.nn.functional as F


class Model1CNN(nn.Module):
    def __init__(self):
        super(Model1CNN, self).__init__()
        # Reduced channels
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # Removed padding
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        # Added intermediate layer
        self.fc1 = nn.Linear(16 * 6 * 6, 32)
        # Final layer
        self.fc2 = nn.Linear(32, 10)
        self.pool = nn.MaxPool2d(2, 2)
        # Added dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))
        # 14x14 -> 6x6
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)