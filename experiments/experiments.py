import torch
from torchvision import models
import numpy as np
from decorbp.experiments.models import *
from decorbp.experiments.data import *
from decorbp.decorrelation.decorrelation import DecorLinear, DecorConv2d, DecorConvTranspose2d

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


def decor2bp(model):
    def _replace_modules(module):
        for i, (name, layer) in enumerate(module.named_children()):
            if isinstance(layer, DecorLinear):
                new_layer = nn.Linear(
                    layer.linear.in_features, layer.linear.out_features, bias=layer.linear.bias is not None)
                new_layer.weight.data = layer.linear.weight.data.view(layer.linear.out_features, layer.linear.in_features)
                if layer.linear.bias is not None:
                    new_layer.bias.data = layer.linear.bias.data
                setattr(module, name, new_layer)
            elif hasattr(layer, 'children') and callable(layer.children):
                # Recursively apply the same logic to child modules
                _replace_modules(layer)
    
    _replace_modules(model)
    return model

def bp2decor(model, skip_conv=False, **kwargs):
    """
    Replaces specific modules in the model with decorrelation modules,

    Args:
        model (nn.Module): The complete model.
        kwargs (dict): Keyword arguments for the decorrelation modules.
    """

    def _replace_modules(module, skip_conv, **kwargs):
        """
        Recursively replaces specific modules in the model's backbone.
        
        Args:
            module (nn.Module): A module within the model, typically a submodule of the backbone.
            kwargs (dict): Keyword arguments for the decorrelation modules.
        """
        for i, (name, layer) in enumerate(module.named_children()):
            if isinstance(layer, nn.Linear) and name not in ('decoder_embed', 'pred'):
                # Create a new DecorLinear layer with matching parameters
                new_layer = DecorLinear(
                    layer.in_features, layer.out_features, bias=layer.bias is not None, **kwargs)
                # Transfer the weights from the original linear layer to the new layer
                new_layer.linear.weight.data = layer.weight.data.view(layer.out_features, layer.in_features)
                if layer.bias is not None:
                    new_layer.linear.bias.data = layer.bias.data
                # Replace the original layer with the new DecorLinear layer
                setattr(module, name, new_layer)
            elif (isinstance(layer, nn.Conv2d) and skip_conv == False):
                # Create a new DecorConv2d layer with matching parameters
                new_layer = DecorConv2d(
                    layer.in_channels, layer.out_channels, layer.kernel_size,
                    stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                    bias=layer.bias is not None, **kwargs)
                
                new_shape = [layer.out_channels, layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1], 1, 1]

                # Transfer the weights from the original convolutional layer to the new layer's forward_conv
                new_layer.forward_conv.weight.data = layer.weight.data.view(new_shape)

                if layer.bias is not None:
                    new_layer.forward_conv.bias.data = layer.bias.data
                
                setattr(module, name, new_layer)
            elif isinstance(layer, nn.ConvTranspose2d):
                # Create a new DecorConvTranspose2d layer with matching parameters
                if i != 0:
                # Omit the first layer, as in GANs this is sampled from a Gaussian distribution and does not need decorrelation.
                    new_layer = DecorConvTranspose2d(
                        layer.in_channels, layer.out_channels, layer.kernel_size,
                        stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                        bias=layer.bias is not None, weights=layer.weight.data, loaded_bias=layer.bias.data, **kwargs)

                    setattr(module, name, new_layer)
                    
            # elif isinstance(layer, nn.BatchNorm2d) and args.batchnorm == 'remove':
            #     # Optionally replace BatchNorm2d with nn.Identity if args.batchnorm is set to 'remove'.
            #     setattr(module, name, nn.Identity())
            elif hasattr(layer, 'children') and callable(layer.children):
                # Recursively apply the same logic to child modules
                _replace_modules(layer, skip_conv, **kwargs)

    _replace_modules(model, skip_conv, **kwargs)

    return model


if __name__ == '__main__':   
    # test functionality
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
    decor_model = bp2decor(model, decor_lr=0.1, kappa=1e-3)
    print(decor_model)

