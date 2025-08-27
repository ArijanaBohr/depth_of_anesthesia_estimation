import torch
import torch.nn as nn
import torch.optim as optim

class BPNet(nn.Module):
    def __init__(self, input_size):
        super(BPNet, self).__init__()
        self.hidden = nn.Linear(input_size, 63)
        self.output = nn.Linear(63, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

