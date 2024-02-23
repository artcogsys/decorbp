import torch.nn as nn
from decorrelation.decorrelation import DecorrelationFC

class MLP(nn.Sequential):

    def __init__(self, input_dim):
        super().__init__(DecorrelationFC(input_dim),
                        nn.Linear(input_dim, 100),
                        nn.ReLU(),
                        DecorrelationFC(100),
                        nn.Linear(100, 10)
                        )

    def forward(self, x):
        return super().forward(x.view(len(x), -1))
    