import torch
import torch.nn as nn
from decorrelation.decorrelation import DecorLinear, DecorConv2d

class MLP(nn.Sequential):
    """Simple MLP example"""

    def __init__(self, in_features, args):
        super().__init__(DecorLinear(in_features, 100, method=args.method, decor_lr=args.decor_lr, kappa=args.kappa),
                        nn.LeakyReLU(),
                        DecorLinear(100, 10, method=args.method, decor_lr=args.decor_lr, kappa=args.kappa)
                        )

    def forward(self, x):
        return super().forward(x.view(len(x), -1))
    
    
class ConvNet(nn.Sequential):
    """Simple ConvNet example"""

    def __init__(self, in_channels, args):
        super().__init__(DecorConv2d(in_channels, out_channels=5, kernel_size=(5,5), method=args.method, decor_lr=args.decor_lr,
                                     kappa=args.kappa, downsample_perc=args.downsample_perc),
                        nn.LeakyReLU(),
                        nn.Flatten(),
                        DecorLinear(2880, 10, method=args.method, decor_lr=args.decor_lr, kappa=args.kappa)
                        )

    def forward(self, x):
        return super().forward(x)
    
