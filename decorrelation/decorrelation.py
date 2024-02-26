import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair
import itertools

def decorrelation_modules(model: nn.Module):
    """Returns the list of decorrelation modules
    """
    return list(filter(lambda m: isinstance(m, Decorrelation), model.modules()))

def decorrelation_parameters(model: nn.Module):
    """Returns all decorrelation parameters as an iterable
    """
    return itertools.chain(*map( lambda m: m.parameters(), decorrelation_modules(model)))

def decorrelation_update(modules):
    """Updates all decorrelation modules and returns decorrelation loss
    """
    loss = 0.0
    for m in modules:
        loss += m.update()
    return loss

def lower_triangular(C, offset):
    """Return lower triangular elements of a matrix as a vector
    """
    return C[torch.tril_indices(C.shape[0], C.shape[1], offset=offset).unbind()]


class Decorrelation(nn.Module):
    r"""A Decorrelation layer flattens the input, decorrelates, updates decorrelation parameters, and returns the reshaped decorrelated input.

    NOTE:
    We combine this with a linear layer to implement the transformation. If the batch size is larger than the number of outputs it can be more 
    efficient to combine this, as we do for the convolutions. On the other hand, we always need access to z=Rx so it may not matter...
    """

    size: int

    def __init__(self, size: int, kappa=0.0, device=None, dtype=None) -> None:
        """"Params:
            - size: input size
            - kappa: weighting between decorrelation (0.0) and unit variance (1.0)
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.size = size
        self.kappa = kappa
        self.R = nn.Parameter(torch.eye(self.size, **factory_kwargs), requires_grad=False)

    def reset_parameters(self) -> None:
        nn.init.eye(self.R)

    def forward(self, input: Tensor) -> Tensor:
        """Decorrelates the input. Maps back if needed
        """
        self.X = F.linear(input.view(len(input), -1), self.R)
        return self.X.view(input.shape)

    def update(self):
        """
        Implements learning rule in Decorrelation loss technical note
        """

        # strictly lower triangular part of x x' averaged over datapoints
        L = torch.tril(self.X.T @ self.X) / len(self.X)

        # unit variance term averaged over datapoints; faster via kronecker?
        V = torch.diag(torch.mean(torch.square(self.X), axis=0) - 1)

        # compute update; equation (3) in technical note
        self.R.grad = (4.0/self.size) * ( ((1-self.kappa)/(self.size - 1)) * L + self.kappa * V ) @ self.R

        # compute loss; equation (1) in technical note as separate terms
        decorrelation_loss = (2*(1-self.kappa))/(self.size * (self.size - 1)) * torch.trace(L @ L.T) 
        variance_loss = (self.kappa/self.size) * torch.trace(V @ V.T)
        # return (1.0/self.d) * ( 2*(1-self.kappa)/(self.d - 1) * torch.trace(L @ L.T) + self.kappa * torch.trace(V @ V.T) )
      
        return decorrelation_loss, variance_loss


class DecorConv2d(Decorrelation):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = False,
                 downsample_perc=1.0, kappa=0.0, device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}

        # define decorrelation layer
        super().__init__(size=in_channels * np.prod(kernel_size), kappa=kappa, **factory_kwargs)        
        self.downsample_perc = downsample_perc

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # this applies the kernel weights
        self.forward_conv = nn.Conv2d(in_channels=self.size,
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

        # combines the patch-wise R update with the convolutional W update
        weight = nn.functional.conv2d(self.R.view(self.size, self.in_channels, *self.kernel_size).moveaxis(0, 1),
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

    def patches(self, input):
        x = nn.functional.conv2d(input, self.R.reshape(self.size, self.in_channels, *self.kernel_size),
                                    bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation)
        x = x.moveaxis(1, 3)
        x = x.reshape(-1, self.size)
        return x
        
    def update(self):

        self.X = self.patches(self.input[np.random.choice(np.arange(len(self.input)),
                                                      size= int(len(self.input) * self.downsample_perc))])
        
        return super().update()
