import torch
import torch.nn as nn

class SimpleStockNet(nn.Module):
    """
    Tiny feed-forward network used for the POC.
    Input: stock-like feature vector.
    Output: binary or 3-class prediction (e.g., DOWN / FLAT / UP).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)          # this layer will be “monitored” for footprints
        x = self.fc2(x)
        return x
