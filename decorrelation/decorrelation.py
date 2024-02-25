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
    return list(filter(lambda m: isinstance(m, AbstractDecorrelation), model.modules()))

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

class AbstractDecorrelation(nn.Module):
    """Abstract base class for decorrelation so we can identify decorrelation modules.
    """

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError
    
    def update(self): 
        raise NotImplementedError
    

class Decorrelation(AbstractDecorrelation):
    r"""A Decorrelation layer flattens the input, decorrelates, updates decorrelation parameters, and returns the reshaped decorrelated input.

    NOTE:
    We combine this with a linear layer to implement the transformation. If the batch size is larger than the number of outputs it can be more 
    efficient to combine this, as we do for the convolutions. On the other hand, we always need access to z=Rx so it may not matter...
    """

    d: int

    def __init__(self, d: int, kappa=0.0, device=None, dtype=None) -> None:
        """"Params:
            - d: input dimensionality
            - kappa: weighting between decorrelation (0.0) and unit variance (1.0)
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.d = d
        self.kappa = kappa
        self.R = nn.Parameter(torch.eye(self.d, **factory_kwargs), requires_grad=False)
        self.register_buffer('eye', torch.eye(self.d, **factory_kwargs))
        self.register_buffer('neg_eye', 1.0 - torch.eye(self.d, **factory_kwargs))

    def reset_parameters(self) -> None:
        nn.init.eye(self.R)

    def forward(self, input: Tensor) -> Tensor:
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
        self.R.grad = (4.0/self.d) * ( ((1-self.kappa)/(self.d - 1)) * L + self.kappa * V ) @ self.R

        # compute loss; equation (1) in technical note
        return (1.0/self.d) * ( (2*(1-self.kappa)/(self.d - 1)) * torch.trace(L @ L.T) + self.kappa * torch.trace(V @ V.T) )


class DecorConv2d(nn.Conv2d, AbstractDecorrelation):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = False,
                 downsample_perc=1.0, kappa=0.0, device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.patch_length = in_channels * np.prod(kernel_size)
        self.downsample_perc = downsample_perc
        self.kappa = kappa

        super().__init__(
            in_channels=in_channels,
            out_channels=self.patch_length,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            **factory_kwargs
        )

        self.forward_conv = nn.Conv2d(in_channels=self.patch_length,
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=0,
                                        dilation=(1, 1),
                                        bias=bias is not None,
                                        **factory_kwargs)

        self.weight.requires_grad_(False)

        self.input = None

        self.register_buffer('eye', torch.eye(self.patch_length, **factory_kwargs))
        self.register_buffer('neg_eye', 1.0 - torch.eye(self.patch_length, **factory_kwargs))

    def reset_parameters(self) -> None:
        w_square = self.weight.reshape(self.patch_length, self.patch_length)
        w_square = torch.nn.init.eye_(w_square)
        self.weight = torch.nn.Parameter(w_square.reshape(self.patch_length, self.in_channels, * self.kernel_size))

    def forward(self, x: Tensor) -> Tensor:

        self.input = x

        # combines the patch-wise R update with the convolutional W update
        weight = torch.nn.functional.conv2d(self.weight.moveaxis(0, 1),
                                            self.forward_conv.weight.flip(-1, -2),
                                            padding=0).moveaxis(0, 1)
        
        # applies the combined weight to generate the desired output
        self.forward_conv.output = torch.nn.functional.conv2d(x, weight,
                                         stride=self.stride,
                                         dilation=self.dilation,
                                         padding=self.padding)

        # needed for BP gradient propagation
        self.forward_conv.output.requires_grad_(True)
        self.forward_conv.output.retain_grad()
        
        return self.forward_conv.output


    def update(self):

        # here we compute the patch outputs explicitly. At this level we can downsample so this can 
        # be cheaper than explicitly computing in the above (since we otherwise average over patches x batches!!!)
        # from a coding perspective it would be more elegant to define a decorrelator within the conv layer which operates on the
        # patches. Then we can just plug in Decorrelator. Downside is that we can't downsample (or must downsample for W as well...) and cant use the fast combined operation (since we need .X)
        sample_size = int(len(self.input) * self.downsample_perc)
        X = super().forward(self.input[np.random.choice(np.arange(len(self.input)), sample_size)])




        # strictly lower triangular part of x x' averaged over datapoints
        L = torch.tril(self.X.T @ self.X) / len(self.X)

        # unit variance term averaged over datapoints; faster via kronecker?
        V = torch.diag(torch.mean(torch.square(self.X), axis=0) - 1)

        # compute update; equation (3) in technical note
        self.R.grad = (4.0/self.d) * ( ((1-self.kappa)/(self.d - 1)) * L + self.kappa * V ) @ self.R

        # compute loss; equation (1) in technical note
        return (1.0/self.d) * ( (2*(1-self.kappa)/(self.d - 1)) * torch.trace(L @ L.T) + self.kappa * torch.trace(V @ V.T) )



        # C = torch.einsum('nipq,njpq->ij', x, x) / (x.shape[0] * x.shape[2] * x.shape[3])
        # if self.whiten:
        #     C -= self.eye
        # else:
        #     C *= self.neg_eye

        # self.weight.grad = torch.einsum('ij,jabc->iabc', C, self.weight)

        # # decorrelation loss computed per patch
        # if self.whiten:
        #     return torch.mean(torch.square(lower_triangular(C, offset=0)))
        # else:
        #     return torch.mean(torch.square(lower_triangular(C, offset=-1)))

