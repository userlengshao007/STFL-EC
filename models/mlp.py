import torch.nn as nn


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for baseline evaluation.
    """

    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(self.flat(x))