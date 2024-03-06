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

    def __init__(self, in_features: int, bias: bool = False, kappa: float = 1e-3, device = None, dtype = None) -> None:
        """"Params:
            - in_features: input dimensionality
            - bias: whether or not to demean the data
            - whiten: whether or not to whiten the data
            - method: decorrelation method: 'copi' or 'gs' (gram-schmidt)
            - eta: decorrelation step size (eta = 0: variance constraint only; eta > 0: pushes towards normalized decorrelation)
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.register_buffer("weight", torch.empty(self.in_features, self.in_features, **factory_kwargs))
        if bias:           
            self.register_buffer("bias", torch.empty(in_features, **factory_kwargs))
        else:
            self.bias = None

        self.kappa = kappa

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
        """Implements Gram-Schmidt decorrelation update"""

        # If using a bias, it should demean the data
        if self.bias is not None:
            self.bias.grad = 1e-3 * self.decor_state.mean(axis=0)

        # strictly lower triangular part of x x' averaged over datapoints
        L = torch.tril(self.decor_state.T @ self.decor_state) / len(self.decor_state)
        # L = self.decor_state.T @ self.decor_state / len(self.decor_state) # we could also use full but then we need to scale differently

        # unit variance term averaged over datapoints; faster via kronecker?
        V = torch.diag(torch.mean(torch.square(self.decor_state), axis=0) - 1)

        # compute update; equation (3) in technical note
        self.weight.grad = (4.0/self.in_features) * ( ((1-self.kappa)/(self.in_features - 1)) * L + self.kappa * V ) @ self.weight

        # compute loss; equation (1) in technical note as separate terms
        # decorrelation_loss = (2*(1-self.kappa))/(self.size * (self.size - 1)) * torch.trace(L @ L.T) 
        # variance_loss = (self.kappa/self.size) * torch.trace(V @ V.T)
        return (1.0/self.in_features) * ( 2*(1-self.kappa)/(self.in_features - 1) * torch.trace(L @ L.T) + self.kappa * torch.trace(V @ V.T) )

class DecorLinear(Decorrelation):
    """Linear layer with input decorrelation"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, decor_bias=False, kappa = 1e-3,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, bias=decor_bias, kappa=kappa, **factory_kwargs)
        self.linear = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)
        
    def forward(self, input: Tensor) -> Tensor:
        return self.linear.forward(super().forward(input))      
    

class DecorConv2d(Decorrelation):
    """2d convolution with input decorrelation"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 bias: bool = True, decor_bias: bool = False, kappa = 1e-3, downsample_perc=1.0,
                 device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}

        # define decorrelation layer
        super().__init__(in_features=in_channels * np.prod(kernel_size), bias=decor_bias, kappa=kappa, **factory_kwargs)        
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

