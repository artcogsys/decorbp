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

# class AbstractDecorrelation(nn.Module):
#     """Abstract base class for decorrelation so we can identify decorrelation modules.
#     """

#     def forward(self, input: Tensor) -> Tensor:
#         raise NotImplementedError
    
#     def update(self): 
#         raise NotImplementedError
    

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

        # compute loss; equation (1) in technical note
        # return (1.0/self.d) * ( 2*(1-self.kappa)/(self.d - 1) * torch.trace(L @ L.T) + self.kappa * torch.trace(V @ V.T) )

        # as separate terms
        decorrelation_loss = (2*(1-self.kappa))/(self.size * (self.size - 1)) * torch.trace(L @ L.T) 
        variance_loss = (self.kappa/self.size) * torch.trace(V @ V.T)

        return decorrelation_loss, variance_loss


class DecorConv2d(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = False,
                 downsample_perc=1.0, kappa=0.0, device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.downsample_perc = downsample_perc
        self.size = in_channels * np.prod(kernel_size)
        self.kappa = kappa

        # this generates patches (we could use unfold for this)
        super().__init__(
            in_channels=in_channels,
            out_channels=self.size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            **factory_kwargs
        )

        # this applies the kernel weights W
        self.forward_conv = nn.Conv2d(in_channels=in_channels * np.prod(kernel_size),
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=0,
                                        dilation=(1, 1),
                                        bias=bias is not None, # ?
                                        **factory_kwargs)

        self.weight.requires_grad_(False)

        self.input = None

        self.register_buffer('eye', torch.eye(self.size, **factory_kwargs))
        self.register_buffer('neg_eye', 1.0 - torch.eye(self.size, **factory_kwargs))

    def reset_parameters(self) -> None:
        w_square = self.weight.reshape(self.size, self.size)
        w_square = torch.nn.init.eye_(w_square)
        self.weight = torch.nn.Parameter(w_square.reshape(self.size, self.in_channels, *self.kernel_size), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:

        self.input = x

        # combines the patch-wise R update with the convolutional W update
        # in_chan=2, out_chan=kernel = 5x5, 50 patches
        # self.weight = [50, 2, 5, 5] mapping from inputs to patches; this means each patch gets its own decorrelation matrix
        # self.forward_conv.weight = [3, 50, 1, 1] mapping from patches to outputs using 1x1 convolutions
        # weight = [3, 2, 5, 5] performing the linear mapping RW per patch with W fixed and R variable
        # moveaxis(0,1) creates [2, 50, 5, 5] input meaning that the convolutional weight is multiplied with each decorrelation kernel
        # [2, 50, 5, 5] x [3, 50, 1, 1] => moveaxis(0,1) => [3, 2, 5, 5]
        # unclear why we perform the weight flip...
        # if we handle this by Decorrelation then we need to create 50 Decorrelation objects; way less efficient
        weight = nn.functional.conv2d(self.weight.moveaxis(0, 1),
                                            self.forward_conv.weight.flip(-1, -2),
                                            padding=0).moveaxis(0, 1)
        
        # applies the combined weight to generate the desired output
        self.forward_conv.output = nn.functional.conv2d(x, weight,
                                         stride=self.stride,
                                         dilation=self.dilation,
                                         padding=self.padding)

        # needed for BP gradient propagation
        self.forward_conv.output.requires_grad_(True)
        self.forward_conv.output.retain_grad()
        
        return self.forward_conv.output

    def update(self):

        sample_size = int(len(self.input) * self.downsample_perc)
        x = super().forward(self.input[np.random.choice(np.arange(len(self.input)), sample_size)])
        x = x.moveaxis(1, 3)
        x = x.reshape(-1, self.size)

        # assuming kappa = 0.0 for now
        corr = (1 / len(x)) * torch.einsum('ki,kj->ij', x, x) * self.neg_eye

        #  if debug:
        decor_loss = torch.mean(torch.abs(corr)) # WHAT DOES THIS TELL US ABOUT OUR NEW LOSS??

        weight = self.weight
        weight = weight.reshape(-1, np.prod(weight.shape[1:]))
        decor_update = torch.einsum('ij,jk->ik', corr, weight)
        decor_update = decor_update.reshape(self.weight.shape)

        self.weight.grad = decor_update.clone()

        return decor_loss.item(), 0.0
    
        # x = self.input

        # C = torch.einsum('nipq,njpq->ij', x, x) / (x.shape[0] * x.shape[2] * x.shape[3])

        # # if self.whiten:
        # #     C -= self.eye
        # # else:
        # #     C *= self.neg_eye

        # # assuming kappa = 0.0
        # C *= self.neg_eye

        # self.weight.grad = torch.einsum('ij,jabc->iabc', C, self.weight)

        # # decorrelation loss computed per patch
        # # if self.whiten:
        # #     return torch.mean(torch.square(lower_triangular(C, offset=0)))
        # # else:
        # return torch.mean(torch.square(lower_triangular(C, offset=-1))), 0.0


        # # here we compute the patch outputs explicitly. At this level we can downsample so this can 
        # # be cheaper than explicitly computing in the above (since we otherwise average over patches x batches!!!)
        # # from a coding perspective it would be more elegant to define a decorrelator within the conv layer which operates on the
        # # patches. Then we can just plug in Decorrelator. Downside is that we can't downsample (or must downsample for W as well...) and cant use the fast combined operation (since we need .X)
        # sample_size = int(len(self.input) * self.downsample_perc)
        # X = super().forward(self.input[np.random.choice(np.arange(len(self.input)), sample_size)])
        # X = X.moveaxis(1,3).reshape(-1, X.shape[1])

        # # strictly lower triangular part of x x' averaged over datapoints
        # L = torch.tril(X.T @ X) / len(X)

        # # unit variance term averaged over datapoints; faster via kronecker?
        # V = torch.diag(torch.mean(torch.square(X), axis=0) - 1)

        # # compute update; equation (3) in technical note

        # # HERE: R = weight; shape of R / W. We should be able to simplify this... ideally by operating on decorrelation weights; but this is the same weight as used for the efficient forward mapping.
        # # Can we replace with two separate steps using unfold and still be as efficient?
        # self.R.grad = (4.0/self.d) * ( ((1-self.kappa)/(self.d - 1)) * L + self.kappa * V ) @ self.R

        # # compute loss; equation (1) in technical note
        # return (1.0/self.d) * ( 2*(1-self.kappa)/(self.d - 1) * torch.trace(L @ L.T) + self.kappa * torch.trace(V @ V.T) )



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

