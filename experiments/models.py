import torch.nn as nn
from decorrelation.decorrelation import DecorLinear, DecorConv2d

class MLP(nn.Sequential):

    def __init__(self, input_dim, kappa=0.0):
        super().__init__(DecorLinear(input_dim, 100, kappa=kappa),
                        nn.ReLU(),
                        DecorLinear(100, 10, kappa=kappa)
                        )

    def forward(self, x):
        return super().forward(x.view(len(x), -1))
    
class Convnet(nn.Sequential):

    def __init__(self, in_channels, kappa=0.0):
        # in_channels = input_dim[0]
        # super().__init__(DecorrelationPatch2d(in_channels, kernel_size=(5,5)),
        #                 nn.Conv2d(in_channels, out_channels=10, kernel_size=(5,5)),
        #                 nn.ReLU(),
        #                 nn.Flatten(),
        #                 DecorrelationFC(5760), # expensive!
        #                 nn.Linear(5760, 10)
        #                 )

        super().__init__(DecorConv2d(in_channels, out_channels=10, kernel_size=(5,5), kappa=kappa),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(5760, 10)
                        )

    def forward(self, x):
        return super().forward(x)
    