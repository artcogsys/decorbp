import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair
import itertools
from typing import Optional

# def set_decor_learning_rate(model, lr): # THIS COULD BE DONE IN NP
#     for m in filter(lambda m: isinstance(m, AbstractDecorrelation), model.modules()):
#         m.lr = lr

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

# def covariance(modules):
#     """ This is the measure of interest. We return the mean off-diagonal absolute covariance and the mean variance
#     """
#     cov = 0.0
#     var = 0.0
#     for m in modules:
#         C = m.covariance(m.output)
#         cov += torch.mean(torch.abs(lower_triangular(C)))
#         var += torch.mean(torch.diag(C))
#     cov /= len(modules)
#     var /= len(modules)
#     return cov, var

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

    in_features: int

    def __init__(self, in_features: int, whiten=False, device=None, dtype=None) -> None:
        """"Params:
            - in_features: number of inputs
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.whiten=whiten
        self.R = nn.Parameter(torch.eye(self.in_features, **factory_kwargs), requires_grad=False)
        self.register_buffer('eye', torch.eye(self.in_features, **factory_kwargs))
        self.register_buffer('neg_eye', 1.0 - torch.eye(self.in_features, **factory_kwargs))

    def reset_parameters(self) -> None:
        nn.init.eye(self.R)

    def forward(self, input: Tensor) -> Tensor:
        self.z = F.linear(input.view(len(input), -1), self.R)
        return self.z.view(input.shape)

    def update(self): 
         # cant be done in forward pass due to BP (different for NP...)
        
        C = self.z.T @ self.z / len(self.z)
        if self.whiten:
            self.R.grad = (C - self.eye) @ self.R
            return torch.mean(torch.square(lower_triangular(C, offset=0)))
        else:
            self.R.grad = (C * self.neg_eye) @ self.R
            return torch.mean(torch.square(lower_triangular(C, offset=-1)))

## BE ABLE TO COMPUTE CORR LOSS FOR BP AS WELL? SEPARATE FROM ABOVE???


## DO WE WANT TO ADD THE SGD UPDATING TO THE FORWARD STEP? 

## CAN WE SEPARATE OUT PATCH COMPUTATION AND THEN RECOMBINATION TO CONVOLUTIONAL OUTPUT? THIS MEANS MAPPING CONV INPUT TO SAME SIZE WHERE CHANNELS = INCHAN x H x W PATCHES AND W IS 1 x 1 CONVOLUTION

    # def update(self):
        # """ Better at decorrelating/whitening but poor at loss minimization
        # """

    #     # compute E[X'X]
    #     Exx = self.output.T @ self.output / len(self.output)

    #     # E[X]'
    #     Ex = torch.mean(self.output, axis=0, keepdim=True)

    #     # compute autocovariance matrix
    #     K = Exx - Ex.T @ Ex
        
    #     if self.whiten:
    #         K -= self.eye
    #     else:
    #         K *= self.neg_eye
        
    #     self.R.grad = K @ self.R


    # def update(self):
    #     """
    #     Proper decorrelation requires minimizing E[X'X] - E[X']E[X]... Flipped dims... wrt standard notation
    #     """
    #     # E[X'X]
    #     Exx = self.output.T @ self.output / len(self.output)
        
    #     # E[X]'
    #     Ex = torch.mean(self.output, axis=0, keepdim=True)

    #     C = Exx - Ex.T @ Ex
    #     self.R.grad = torch.einsum('ij,jk->ik', C, self.R).clone() # NOTE: why was clone added in constence code?
    

# #### 2D CONV


### Updated Pycopi implementation
        
# class DecorConv2d(_ConvNd, Decorrelation):


#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: _size_2_t,
#         stride: _size_2_t = 1,
#         padding: str | _size_2_t = 0,
#         dilation: _size_2_t = 1,
#         bias: bool = True, # default false
#         device=None,
#         dtype=None
#     ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         kernel_size_ = _pair(kernel_size)
#         stride_ = _pair(stride)
#         padding_ = padding if isinstance(padding, str) else _pair(padding)
#         dilation_ = _pair(dilation)

#         self.patch_length = in_channels * np.prod(kernel_size)
        
#         super().__init__(
#             in_channels, self.patch_length, kernel_size_, stride_, padding_, dilation_,
#             False, _pair(0), groups=1, bias=False, padding_mode='zeros', **factory_kwargs)

#     def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
#         if self.padding_mode != 'zeros':
#             return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
#                             weight, bias, self.stride,
#                             _pair(0), self.dilation, self.groups)
#         return F.conv2d(input, weight, bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#     def _forward(self, input: Tensor) -> Tensor:
#         return self._conv_forward(input, self.weight, self.bias)


class DecorConv2d(nn.Conv2d, AbstractDecorrelation):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = False,
                 downsample_perc=1.0, whiten=False, device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.patch_length = in_channels * np.prod(kernel_size)
        self.downsample_perc = downsample_perc
        self.whiten=whiten

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
        self.output = torch.zeros(2,2) # DEBUG

        self.register_buffer('eye', torch.eye(self.patch_length, **factory_kwargs))
        self.register_buffer('neg_eye', 1.0 - torch.eye(self.patch_length, **factory_kwargs))
        # self.neg_eye = torch.nn.parameter.Parameter(1.0 - torch.eye(self.patch_length), requires_grad=False)
        # self.eye = torch.nn.parameter.Parameter(torch.eye(self.patch_length), requires_grad=False)
        # self.decor_losses = []

    def reset_parameters(self) -> None:
        w_square = self.weight.reshape(self.patch_length, self.patch_length)
        w_square = torch.nn.init.eye_(w_square)
        self.weight = torch.nn.Parameter(w_square.reshape(self.patch_length, self.in_channels, * self.kernel_size))

    def forward(self, x: Tensor) -> Tensor:

        self.input = x

        # combines the patch-wise R update with the convolutional W update
        # the idea is that for forward mapping we can just combine the R and W (also for linear)
        # think about consequences of z = Rx and y = Wz vs y = (WR)x = Ax... 
        weight = torch.nn.functional.conv2d(self.weight.moveaxis(0, 1), self.forward_conv.weight.flip(-1, -2), padding=0).moveaxis(0, 1)
        
        # applies the combined weight to generate the desired output
        self.forward_conv.output = torch.nn.functional.conv2d(x, weight,
                                         stride=self.stride,
                                         dilation=self.dilation,
                                         padding=self.padding)

        # can't we do this at a higher level?  probably not since this is created on the fly and needs to be backpropagated through
        self.forward_conv.output.requires_grad_(True)
        self.forward_conv.output.retain_grad()
        
        return self.forward_conv.output
    
        # NOTE: one approach could be to have a decorrelator that transforms into a patchwise representation and a convlayer that takes the patchwise version and maps to the output...
        
    # @staticmethod
    # def covariance(x):
    #     return torch.cov(x.T)
    #     # return torch.cov(x.moveaxis(1,2).reshape(-1,x.shape[1]).T)

    def update(self):

        # here we compute the patch outputs explicitly
        # at this level we can downsample so this might be cheaper than explicitly computing in the above (since we otherwise average over patches x batches!!!)
        sample_size = int(len(self.input) * self.downsample_perc)
        x = super().forward(self.input[np.random.choice(np.arange(len(self.input)), sample_size)])
        x = x.moveaxis(1, 3)
        x = x.reshape(-1, self.patch_length)
        self.output = x

        if self.whiten:
            C = (1 / len(x)) * torch.einsum('ki,kj->ij', x, x) - self.eye
        else:
            C = (1 / len(x)) * torch.einsum('ki,kj->ij', x, x) * self.neg_eye

        weight = self.weight
        weight = weight.reshape(-1, np.prod(weight.shape[1:]))
        decor_update = torch.einsum('ij,jk->ik', C, weight)

        self.weight.grad = decor_update.reshape(self.weight.shape)

        # decorrelation loss
        if self.whiten:
            return torch.mean(torch.square(lower_triangular(C, offset=0)))
        else:
            return torch.mean(torch.square(lower_triangular(C, offset=-1)))

        # self.weight.grad = decor_update.clone()


        

# """
# Options:
#     - patchwise decorrelation as we do now
#     - patchwise with separate R for each patch
#     - patchwise but map back to original (averaged input); advantage is separate decorrelating conv module
#     - expand the whole input and globally decorrelate and reshape back (extreme case of the naive approach); possibly combined with factorization...; advantage is separate decorrelating conv module
# """


# BELOW AN INTERESTING YET NONFUNCTIONAL IMPLEMENTATION
        
# class DecorrelationPatch2d(torch.nn.Module):

#     def __init__(self,
#                 in_channels: int,
#                 kernel_size: _size_2_t,
#                 whiten: bool = False,
#                 device=None,
#                 dtype=None) -> None:

#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()

#         self.in_channels = in_channels
#         self.kernel_size = kernel_size
#         self.whiten = whiten

#         self.patch_length = in_channels * np.prod(kernel_size) # number of elements in a patch to be represented as channels
#         self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=1, padding=0, dilation=1)

#         self.fold = None
#         self.divisor = None

#         self.register_buffer('R', torch.eye(self.patch_length, **factory_kwargs))
#         self.register_buffer('eye', torch.eye(self.patch_length, **factory_kwargs))
#         self.register_buffer('neg_eye', 1.0 - torch.eye(self.patch_length, **factory_kwargs))

#     def forward(self, input: Tensor) -> Tensor:

#         # transform from tensor to patch representation
#         z = self.unfold(input) # [N, C*H*W, P]

#         # apply decorrelating transform
#         self.output = torch.einsum('nij,ik->nij', z, self.R) # decorrelated patches [N, C*H*W, P]

#         # transform from patch to tensor representation
#         if self.fold is None: # we only compute this once
#             self.fold = torch.nn.Fold(output_size=input.shape[-2:], kernel_size=self.kernel_size, stride=1, padding=0, dilation=1)
#             self.divisor = self.fold(self.unfold(torch.ones_like(input)))

#         # we could create a path mask and apply this. But there is likely a more direct route
#         # z = self.output
#         # self.fold(z)
         
#         # BUT THIS SHOULD ALSO JUST WORK.. WHY NOT?
#         return self.fold(self.output) / self.divisor # [N, C, H, W]

#     @staticmethod
#     def covariance(x):
#         return torch.cov(x.moveaxis(1,2).reshape(-1,x.shape[1]).T)
#         # return torch.einsum('nip,njp->ij', x, x) / (x.shape[0] * x.shape[2])
    
#     def update(self):
#         C = torch.einsum('nip,njp->ij', self.output, self.output) / (self.output.shape[0] * self.output.shape[2]) # double check if this is correct
#         if self.whiten:
#             C -= self.eye
#         else:
#             C *= self.neg_eye
#         self.R.grad = torch.einsum('ij,jk->ik', C, self.R)

    # def flatten(self, x):
    #     z = self.fold(x) / self.divisor
    #     return z.view(z.shape[0],-1)

    # def mean_correlation(self):
    #     C = self.correlation(self.output)
    #     return torch.mean(C[torch.tril_indices(len(C), len(C), offset=1)])
    

#         self.weight.requires_grad_(False)  # deregister since we use a custom update

#     def forward(self, input: Tensor) -> Tensor:
        
#         weight = torch.nn.functional.conv2d(self.weight.moveaxis(0, 1),
#                                             self.weight.flip(-1, -2), # forward_conv.
#                                             padding=0).moveaxis(0, 1)
        
#         self.output = torch.nn.functional.conv2d(input, 
#                                                  weight,
#                                                  stride=self.stride,
#                                                  dilation=self.dilation,
#                                                  padding=self.padding)
        
#         self.output.requires_grad_(True) # WHY HERE; ALSO FOR LINEAR?...
#         self.output.retain_grad()

#         return self.output
    

# class Decorrelation2dNaive(nn.Conv2d):
#     """Naive approach where we locally decorrelate and then use that is input to a regular convolution. We decorrelate all elements of a kernel block and then only keep the central element
#        per channel.

#     """

#     def __init__(self, in_channels: int, kernel_size: _size_2_t, device=None, dtype=None):

#         factory_kwargs = {'device': device, 'dtype': dtype}

#         super().__init__(in_channels=in_channels,
#                          out_channels=in_channels,
#                          kernel_size=(1, 1),
#                          stride=(1, 1),
#                          padding=0,
#                          dilation=(1, 1),
#                          bias=False,
#                          **factory_kwargs)       

#         self.weight.requires_grad_(False)  # deregister since we use a custom update

#     def forward(self, input: Tensor) -> Tensor:
        
        
#         return super().forward(input)


#         # weight = torch.nn.functional.conv2d(self.weight.moveaxis(0, 1),
#         #                                     self.weight.flip(-1, -2), # forward_conv.
#         #                                     padding=0).moveaxis(0, 1)
        
#         # self.output = torch.nn.functional.conv2d(input, 
#         #                                          weight,
#         #                                          stride=self.stride,
#         #                                          dilation=self.dilation,
#         #                                          padding=self.padding)
        
#         # self.output.requires_grad_(True) # WHY HERE; ALSO FOR LINEAR?...
#         # self.output.retain_grad()

#         # return self.output
    



# # class Decorrelation2d(nn.Conv2d):

# #     def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, device=None, dtype=None):

# #         factory_kwargs = {'device': device, 'dtype': dtype}
# #         self.patch_length = in_channels * np.prod(kernel_size)

# #         super().__init__(in_channels=self.patch_length,
# #                         out_channels=out_channels,
# #                         kernel_size=(1, 1),
# #                         stride=(1, 1),
# #                         padding=0,
# #                         dilation=(1, 1),
# #                         bias=False,
# #                         **factory_kwargs)

# #         self.weight.requires_grad_(False)  # deregister since we use a custom update

# #     def forward(self, input: Tensor) -> Tensor:
        
# #         weight = torch.nn.functional.conv2d(self.weight.moveaxis(0, 1),
# #                                             self.weight.flip(-1, -2), # forward_conv.
# #                                             padding=0).moveaxis(0, 1)
        
# #         self.output = torch.nn.functional.conv2d(input, 
# #                                                  weight,
# #                                                  stride=self.stride,
# #                                                  dilation=self.dilation,
# #                                                  padding=self.padding)
        
# #         self.output.requires_grad_(True) # WHY HERE; ALSO FOR LINEAR?...
# #         self.output.retain_grad()

# #         return self.output


# class CopiConv2d(nn.Conv2d):

#     def __init__(self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: _size_2_t,
#         stride: _size_2_t = 1,
#         padding: _size_2_t = 0,
#         dilation: _size_2_t = 1,
#         alpha = 1,
#         device=None,
#         dtype=None) -> None:

#         self.patch_length = in_channels * np.prod(kernel_size)

#         super().__init__(in_channels=in_channels,
#                          out_channels=out_channels,
#                          kernel_size=kernel_size,
#                          stride=stride,
#                          padding=padding,
#                          dilation=dilation,
#                          bias=False,
#                          dtype=dtype,
#                          device=device)

#         self.weight.requires_grad_(False)  # deregister since we update using copi

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.alpha = alpha
#         self.input = None
#         self.output_shape = None

#     # def forward(self, x):
#     #     self.input = x
#     #     self.output = super().forward(x)
#     #     self.output.requires_grad_(True)
#     #     self.output.retain_grad()
#     #     return self.output

#     def get_target(self):

#         # negative gradient as error signal
#         perturbation = -self.output.grad
#         self.output.grad = None

#         # compute target activation
#         return self.output + self.alpha * perturbation

#     def copi_update(self, normalize=False, **kwargs):

#         target = self.get_target()
#         x = self.input

#         target = target.moveaxis(1, 3)
#         target = target.reshape(-1, self.out_channels)

#         corr = (1 / len(x)) * torch.einsum('ki,kj->ij', target, x)

#         weight = self.weight
#         weight = weight.reshape(-1, np.prod(weight.shape[1:]))

#         decay = torch.einsum('j,ij->ij', torch.mean(x ** 2, axis=0), weight)
#         update = corr - decay
#         update = update.reshape(self.weight.shape)

#         #if normalize:
#         #    update *= corr.shape[1] * multiplier

#         self.weight.grad = -update

#     def copi_parameters(self):
#         return [self.weight]


# class DecorConv2d(nn.Conv2d):
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
#                  stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = False,
#                  copi=False, weight=None, alpha: float = 1.0,
#                  device=None, dtype=None) -> None:

#         self.patch_length = in_channels * np.prod(kernel_size)

#         super().__init__(
#             in_channels=in_channels,
#             out_channels=self.patch_length,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             bias=False,
#             dtype=dtype,
#             device=device
#         )

#         self.copi = copi
#         if self.copi:
#             self.forward_conv = CopiConv2d(in_channels=self.patch_length,
#                                            out_channels=out_channels,
#                                            kernel_size=(1, 1),
#                                            stride=(1, 1),
#                                            padding=0,
#                                            dilation=(1, 1),
#                                            alpha=alpha,
#                                            dtype=dtype,
#                                            device=device)

#         else:
#             self.forward_conv = nn.Conv2d(in_channels=self.patch_length,
#                                           out_channels=out_channels,
#                                           kernel_size=(1, 1),
#                                           stride=(1, 1),
#                                           padding=0,
#                                           dilation=(1, 1),
#                                           bias=bias is not None,
#                                           dtype=dtype,
#                                           device=device)


#             # weight_shape = self.forward_conv.weight.shape
#             # self.forward_conv.weight.data = weight.data.reshape(weight_shape)
#             # if bias is not None:
#             #     bias_shape = self.forward_conv.bias.shape
#             #     self.forward_conv.bias.data = bias.data.reshape(bias_shape)

#         self.weight.requires_grad_(False)

#         self.input = None

#         self.neg_eye = torch.nn.parameter.Parameter(1.0 - torch.eye(self.patch_length), requires_grad=False)
#         self.eye = torch.nn.parameter.Parameter(torch.eye(self.patch_length), requires_grad=False)
#         self.decor_losses = []

#     def reset_parameters(self) -> None:
#         w_square = self.weight.reshape(self.patch_length, self.patch_length)
#         w_square = torch.nn.init.eye_(w_square)
#         self.weight = torch.nn.Parameter(w_square.reshape(self.patch_length, self.in_channels, * self.kernel_size))

#     def forward(self, x: Tensor) -> Tensor:
#         self.input = x
#         weight = torch.nn.functional.conv2d(self.weight.moveaxis(0, 1), self.forward_conv.weight.flip(-1, -2),
#                                                  padding=0).moveaxis(0, 1)
#         self.forward_conv.output = torch.nn.functional.conv2d(x, weight,
#                                          stride=self.stride,
#                                          dilation=self.dilation,
#                                          padding=self.padding)
#         self.forward_conv.output.requires_grad_(True)
#         self.forward_conv.output.retain_grad()
#         return self.forward_conv.output
        

#     def copi_update(self, bio_copi=False, whiten=False, normalize=False, debug=False, downsample_perc=.05, **kwargs):

#         # Decor update
#         sample_size = int(len(self.input)*downsample_perc)
#         x = super().forward(self.input[np.random.choice(np.arange(len(self.input)), sample_size)])
#         x = x.moveaxis(1, 3)

#         x = x.reshape(-1, self.patch_length)

#         # Double downsample. Actually doesn't seem to have much of a speedup effect
#         # sample_size = int(len(x) * .05)
#         # print(x.shape)
#         # x = x[np.random.choice(np.arange(len(x)), sample_size)]
#         # print(x.shape)

#         if whiten:
#             corr = (1 / len(x)) * torch.einsum('ki,kj->ij', x, x) - self.eye
#         else:
#             corr = (1 / len(x)) * torch.einsum('ki,kj->ij', x, x) * self.neg_eye

#         if debug:
#             decor_loss = torch.mean(torch.abs(corr))
#             self.decor_losses.append(decor_loss.item())

#         weight = self.weight
#         weight = weight.reshape(-1, np.prod(weight.shape[1:]))
#         if bio_copi:
#             decor_update = torch.einsum('ij,jk->ik', weight, corr)
#         else:
#             decor_update = torch.einsum('ij,jk->ik', corr, weight)

#         decor_update = decor_update.reshape(self.weight.shape)

#         # Normalize decor update
#         if normalize:
#             decor_update *= corr.shape[1]

#         self.weight.grad = decor_update.clone()

#         # Compute weight update if using COPI in stead of just decorrelated BP
#         if self.copi:
#             # We don't run the forward pass of this layer so we need to manually set its decorrelated input
#             self.forward_conv.input = x
#             self.forward_conv.copi_update(normalize=normalize)


#     def copi_parameters(self):
#         if self.copi:
#             return [self.weight, self.forward_conv.weight]
#         else:
#             return [self.weight]
        

