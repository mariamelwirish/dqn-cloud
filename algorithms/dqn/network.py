import torch
import torch.nn as nn # PyTorch's base class for all neural networks.

class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # for setup
        super(DQNNetwork, self).__init__()

        # A container that runs layers one after another.
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)