import torch
import torch.nn as nn
import torch.nn.functional as F

# Define your neural networks in this class.
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(68, 50)
        self.fc2 = torch.nn.Linear(50, 4)
        # self.possibleActions = [0, 1, 2, 3]

    def forward(self, state):
        h1 = F.tanh(self.fc1(state))
        h2 = F.sigmoid(self.fc2(h1))
        return h2
