import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.nn.common_types import _size_2_t
import itertools

def decor_module(model: nn.Module):
    """Returns the list of decorrelation modules
    """
    return list(filter(lambda m: isinstance(m, Decorrelation), model.modules()))

def decor_parameters(model: nn.Module):
    """Returns all decorrelation parameters as an iterable
    """
    return itertools.chain(*map( lambda m: m.parameters(), decor_module(model)))

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

    def __init__(self, in_features: int, bias: bool = False, eta: float = 1e-5, diagonal = None, device = None, dtype = None) -> None:
        """"Params:
            - in_features: input dimensionality
            - bias: whether or not to demean the data
            - eta: learning rate
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(self.in_features, self.in_features, **factory_kwargs), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(in_features, **factory_kwargs), requires_grad=False) if bias else None
        self.eta = eta
        self.X = None
        self.diagonal = diagonal
        if self.diagonal is None:
            self.uncorrelated = True # compute uncorrelated diagonal

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.eye_(self.weight)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.uncorrelated:
            self.diagonal =  torch.mean(input, axis=0)
        self.decor_state = F.linear(input, self.weight, self.bias)
        return self.decor_state

    def decorrelate(self, input: Tensor):
        """Applies the decorrelating transform. Can be overloaded in composite functions to return the decorrelating transform 
        """
        return F.linear(input.view(len(input), -1), self.weight)

    def update(self):
        """Implements whitened Gram-Schmidt decorrelation update"""

        # # strictly lower triangular part of x x' averaged over datapoints
        # L = torch.tril(self.X.T @ self.X, diagonal=-1) / len(self.X)

        # # unit variance term averaged over datapoints
        # V = torch.diag(torch.mean(torch.square(self.X), axis=0) - 1)

        # # compute update; equation (3) in technical note
        # self.weight.grad = (4.0/self.in_features) * ( ((1-self.kappa)/(self.in_features - 1)) * L + self.kappa * V ) @ self.weight

        # # compute loss; equation (1) in technical note
        # return (1.0/self.in_features) * ( 2*(1-self.kappa)/(self.in_features - 1) * torch.trace(L @ L.T) + self.kappa * torch.trace(V @ V.T) )

        # If using a bias, it should demean the data
        if self.bias is not None:
            self.bias.grad = self.decor_state.sum(axis=0)

        corr = (1/len(self.decor_state))*(
            self.decor_state.transpose(0, 1) @ self.decor_state
        )
        grads = corr @ self.weight.data

        normalizer = self.diagonal / (torch.mean(self.decor_state**2, axis=0))
        normalizer[torch.mean(self.decor_state**2, axis=0) < 1e-8] = 1.0

        grads *= normalizer[:, None]

        # make suitable for gradient descent outside of function
        grads += (1 - normalizer)[:, None] * self.weight.data

        # gradient descent step
        self.weight.data -= self.eta * grads

        # return loss
        return torch.mean(torch.square(torch.tril(corr - self.diagonal @ torch.eye(corr.shape[0]), diagonal=0)))

class DecorLinear(Decorrelation):
    """Linear layer with input decorrelation"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, kappa: float = 0.0, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, kappa=kappa, **factory_kwargs)
        self.linear = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)
        
    def forward(self, input: Tensor) -> Tensor:
        return self.linear.forward(super().forward(input))      
    

class DecorConv2d(Decorrelation):
    """2d convolution with input decorrelation"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 bias: bool = False, eta: float = 1e-5, downsample_perc=1.0,
                 device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}

        # define decorrelation layer
        super().__init__(in_features=in_channels * np.prod(kernel_size), eta=eta, **factory_kwargs)        
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

        self.input = input

        # efficiently combines the patch-wise R update with the convolutional W update
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
        
    def update(self):
        self.decor_state = self.decorrelate(self.input[np.random.choice(np.arange(len(self.input)),
                                                      size= int(len(self.input) * self.downsample_perc))])
        return super().update()
