import torch
from torchvision import models
import numpy as np
from pylearn.experiments.models import *
from pylearn.experiments.data import *

def get_experiment(args, device):
    """
    An experiment is a combination of data, a model and a loss function
    """

    if args.experiment == 'MNIST_MLP':

        train_loader, test_loader, input_dim = get_MNIST(args)
        
        model = MLP(in_features=np.prod(input_dim), args=args).to(device)

        lossfun = torch.nn.CrossEntropyLoss().to(device)
    
    elif args.experiment == 'MNIST_CONVNET':

        train_loader, test_loader, input_dim = get_MNIST(args)
        
        model = ConvNet(in_channels=input_dim[0], args=args).to(device)

        lossfun = torch.nn.CrossEntropyLoss().to(device)
    
    elif args.experiment == 'CIFAR10_RESNET18':

        train_loader, test_loader, input_dim = get_CIFAR10(args)
        
        model = bp2decor(model=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18'),
                         decor_lr=args.decor_lr, kappa=args.kappa, downsample_perc=args.downsample_perc
                         ).to(device)

        lossfun = torch.nn.CrossEntropyLoss().to(device)

    else:
        raise ValueError(f'unknown experiment {args.experiment}')

    return model, lossfun, train_loader, test_loader


def bp2decor(model, **kwargs):
    """
    replaces specific modules in model by decorrelation modules

    Args:
    model: nn.Module
    kwargs: dict
        keyword arguments for the decorrelation modules
    """

    def _replace_modules(module, **kwargs):
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
                _replace_modules(layer, **kwargs)

    _replace_modules(model, **kwargs)

    return model


if __name__ == '__main__':   
    # test functionality
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
    model = models.segmentation.deeplabv3_resnet50(weights=False, num_classes=21)
    decor_model = bp2decor(model, decor_lr=0.1, kappa=1e-3)
    print(decor_model)

