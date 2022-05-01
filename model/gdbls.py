import torch
from torch import nn


# gdbls base
class GrandDescentBoardLearningSystem(nn.Module):
    def __init__(self):
        super(GrandDescentBoardLearningSystem, self).__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        return x
