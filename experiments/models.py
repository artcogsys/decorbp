import torch
import torch.nn as nn
from decorrelation.decorrelation import DecorLinear, DecorConv2d

class MLP(nn.Sequential):
    """Simple MLP example"""

    def __init__(self, in_features, args):
        super().__init__(DecorLinear(in_features, 100, decor_lr=args.decor_lr, kappa=args.kappa),
                        nn.LeakyReLU(),
                        DecorLinear(100, 10, decor_lr=args.decor_lr, kappa=args.kappa)
                        )

    def forward(self, x):
        return super().forward(x.view(len(x), -1))
    
    
class ConvNet(nn.Sequential):
    """Simple ConvNet example"""

    def __init__(self, in_channels, args):
        super().__init__(DecorConv2d(in_channels, out_channels=5, kernel_size=(5,5), decor_lr=args.decor_lr,
                                     kappa=args.kappa, downsample_perc=args.downsample_perc),
                        nn.LeakyReLU(),
                        nn.Flatten(),
                        DecorLinear(2880, 10, decor_lr=args.decor_lr, kappa=args.kappa)
                        )

    def forward(self, x):
        return super().forward(x)
    

class LoadableNet(nn.Module):

    def __init__(self, model_name, *, decor_lr: float = 0.0, bias_lr: float = 0.0, kappa = 1e-3, full: bool = True, downsample_perc: float =1.0, device = None, dtype = None):
        super().__init__()
        factory_kwargs = {'decor_lr': decor_lr, 'bias_lr': bias_lr, 'kappa': kappa, 'full': full, 'downsample_perc': downsample_perc, 'device': device, 'dtype': dtype}
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name)
        self.replace_modules(self.model, **factory_kwargs)
       
    def forward(self, x):
        return self.model(x)

    def replace_modules(self, module, **kwargs):
        """
        replaces specific modules in model by modules specified by the replacement_fn
        """
        for name, layer in module.named_children():
            if isinstance(layer, nn.Linear):
                module.__setattr__(name, DecorLinear(layer.in_features, layer.out_features, bias=layer.bias is not None, **kwargs))
            elif isinstance(layer, nn.Conv2d):
                module.__setattr__(name, DecorConv2d(layer.in_channels, layer.out_channels, layer.kernel_size,
                                                     stride = layer.stride, padding=layer.padding, dilation=layer.dilation,
                                                     bias=layer.bias is not None, **kwargs))
            elif isinstance(layer, nn.BatchNorm2d):
                module.__setattr__(name, nn.Identity())
            elif layer.children() is not None:
                self.replace_modules(layer)


if __name__ == '__main__':   
    # testing functionality
    model = LoadableNet('resnet18')
    print(model.model)
