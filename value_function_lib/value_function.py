from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


class ValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.norm1 = nn.LayerNorm(hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.fc2 = nn.Linear(hidden_dims[-1], 1)

        # Apply Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.fc2(x)
