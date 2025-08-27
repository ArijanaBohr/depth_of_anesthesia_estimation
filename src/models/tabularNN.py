import torch
import torch.nn as nn
import torch.optim as optim


class TabularNN(nn.Module):
    def __init__(self, input_size=71, hidden_sizes=[128, 64, 32], dropout_rate=0.4):
        super(TabularNN, self).__init__()
        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Construct hidden layers
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.batchnorms.append(nn.BatchNorm1d(size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = size

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        for layer, batchnorm, dropout in zip(
            self.layers, self.batchnorms, self.dropouts
        ):
            x = torch.relu(batchnorm(layer(x)))
            x = dropout(x)
        x = self.output_layer(x)
        return x
