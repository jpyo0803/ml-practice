import torch
import torch.nn as nn
from my_modules import ConvBlock, FCBlock

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = FCBlock(64*7*7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x