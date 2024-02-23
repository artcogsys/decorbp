import torch.nn as nn
from decorrelation.decorrelation import DecorrelationFC, DecorrelationPatch2d

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
    
class Convnet(nn.Sequential):

    def __init__(self, input_dim):
        super().__init__(DecorrelationPatch2d(input_dim, kernel_size=(5,5)),
                        nn.Conv2d(in_channels=input_dim[0], out_channels=20, kernel_size=(5,5)),
                        nn.ReLU(),
                        DecorrelationFC(100),
                        nn.Linear(100, 10)
                        )

    def forward(self, x):
        return super().forward(x)
    