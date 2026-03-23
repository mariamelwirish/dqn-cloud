import torch
import torch.nn as nn

'''
Network is defined as follows:
    1. Input layer: 12 neuros (state space = 2 jobs features <reqCom, QoS> + 10 VMs waiting times).
    2. Hidden Layer: 20 neurons.
    3. Output Layer: 10 neurons (action space = number of VMs).
'''

class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNNetwork, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.input_to_hidden(x))
        x = self.hidden_to_output(x)
        return x