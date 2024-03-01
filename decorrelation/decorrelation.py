import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.nn.common_types import _size_2_t
import itertools

def decor_modules(model: nn.Module):
    """Returns the list of decorrelation modules
    """
    return list(filter(lambda m: isinstance(m, Decorrelation), model.modules()))

def decor_parameters(model: nn.Module):
    """Returns all decorrelation parameters as an iterable
    """
    return itertools.chain(*map( lambda m: m.decor_parameters(), decor_modules(model)))

def decor_update(modules):
    """Updates all decorrelation modules and returns decorrelation loss
    """
    loss = 0.0
    for m in modules:
        loss += m.update()
    return loss

def lower_triangular(C: Tensor, offset: int):
    """Return lower triangular elements of a matrix as a vector
    """
    return C[torch.tril_indices(C.shape[0], C.shape[1], offset=offset).unbind()]


class Decorrelation(nn.Module):
    """A Decorrelation layer flattens the input, decorrelates, updates decorrelation parameters, and returns the reshaped decorrelated input"""

    def __init__(self, in_features: int, bias: bool = False, variance = None, device = None, dtype = None) -> None:
        """"Params:
            - in_features: input dimensionality
            - bias: whether or not to demean the data
            - variance: variance constraint. If None, it is set to the input variances. if float it is set to a constant variance
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(self.in_features, self.in_features, **factory_kwargs), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(in_features, **factory_kwargs), requires_grad=False) if bias else None
        self.uncorrelated = variance is None
        if isinstance(variance, float):
            self.variance = torch.tensor([variance] * in_features, **factory_kwargs)
        else:
            self.variance = variance
                    
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.eye_(self.weight)

    def decor_parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        else:
            return [self.weight]
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.uncorrelated:
            self.variance =  torch.mean(input.view(len(input), -1)**2, axis=0)
        self.decor_state = F.linear(input.view(len(input), -1), self.weight, self.bias)
        return self.decor_state.view(input.shape)
    
    # def __call__(self, input: Tensor) -> Tensor:
    #     return self.forward(input)

    def decorrelate(self, input: Tensor):
        """Applies the decorrelating transform. Can be overloaded in composite functions to return the decorrelating transform.
        Maps back to the original input.
        """
        return F.linear(input.view(len(input), -1), self.weight, self.bias).view(input.shape)

    def update(self): 
        """Implements whitened Gram-Schmidt decorrelation update"""

        # If using a bias, it should demean the data
        if self.bias is not None:
            self.bias.grad = self.decor_state.sum(axis=0) # NOTE: OR MEAN?? IS IT UPDATED?

        normalizer = self.variance / (torch.mean(self.decor_state**2, axis=0))
        normalizer[torch.mean(self.decor_state**2, axis=0) < 1e-8] = 1.0

        # compute the correlation matrix
        corr = (self.decor_state.T @ self.decor_state) / len(self.decor_state)

        # gradient of the error (-objective function)
        self.weight.grad = - (normalizer[:, None] * self.weight - normalizer[:, None] * corr @ self.weight)

        # return loss; NOTE: why would we count the off-diagonal elements twice?
        # return torch.mean(torch.square(torch.tril(corr - torch.diag(self.variance), diagonal=0)))
        return torch.mean(torch.square(corr - torch.diag(self.variance)))

class DecorLinear(Decorrelation):
    """Linear layer with input decorrelation"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, variance = None, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, variance=variance, **factory_kwargs)
        self.linear = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)
        
    def forward(self, input: Tensor) -> Tensor:
        return self.linear.forward(super().forward(input))      
    

class DecorConv2d(Decorrelation):
    """2d convolution with input decorrelation"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 bias: bool = False, variance = None, downsample_perc=1.0,
                 device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}

        # define decorrelation layer
        super().__init__(in_features=in_channels * np.prod(kernel_size), variance=variance, **factory_kwargs)        
        self.downsample_perc = downsample_perc

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # this applies the kernel weights
        self.forward_conv = nn.Conv2d(in_channels=self.in_features,
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=0,
                                        dilation=(1, 1),
                                        bias=bias is not None, # ?
                                        **factory_kwargs)

        self.input = None

    def forward(self, input: Tensor) -> Tensor:

        # we store a downsampled version for input decorrelation and diagonal computation
        idx = np.random.choice(np.arange(len(input)), size= int(len(input) * self.downsample_perc)) # could work better on downsampled patches instead
        self.decor_state = self.decorrelate(input[idx])
        if self.uncorrelated:
            self.variance =  torch.mean(self.patches(input[idx])**2, axis=0)

        # efficiently combines the patch-wise R update with the convolutional W update on all data
        weight = nn.functional.conv2d(self.weight.view(self.in_features, self.in_channels, *self.kernel_size).moveaxis(0, 1),
                                      self.forward_conv.weight.flip(-1, -2),
                                      padding=0).moveaxis(0, 1)
                
        # applies the combined weight to generate the desired output
        self.forward_conv.output = nn.functional.conv2d(input, weight,
                                         stride=self.stride,
                                         dilation=self.dilation,
                                         padding=self.padding)

        # needed for BP gradient propagation
        self.forward_conv.output.requires_grad_(True)
        self.forward_conv.output.retain_grad()
        
        return self.forward_conv.output
    
    def decorrelate(self, input: Tensor):
        """Applies the decorrelating transform. Can be overloaded in composite functions to return the decorrelating transform 
        """
        return nn.functional.conv2d(input, self.weight.view(self.in_features, self.in_channels, *self.kernel_size),
                                    bias=None, stride=self.stride, padding=self.padding, 
                                    dilation=self.dilation).moveaxis(1, 3).reshape(-1, self.in_features)
        
    def patches(self, input: Tensor):
        """Returns the input patches via an identity mapping"""
        identity = nn.Parameter(torch.empty(self.in_features, self.in_features, device=self.weight.device, dtype=self.weight.dtype), requires_grad=False)
        nn.init.eye_(identity)
        return nn.functional.conv2d(input, identity.view(self.in_features, self.in_channels, *self.kernel_size),
                                    bias=None, stride=self.stride, padding=self.padding, 
                                    dilation=self.dilation).moveaxis(1, 3).reshape(-1, self.in_features)

