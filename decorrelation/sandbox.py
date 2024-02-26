## TRYING SOME THINGS; MAINLY OTHER CONVOLUTION FORMULATIONS

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair



class DecorConv2dOrig(nn.Conv2d):

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

        # self.conv used in 2 places:
        # - weight convolution. But weight is simply [size, in_channels, kernel_size] so we can use unfold and then reshape
        # - forward convolution. This is a 1x1 convolution so we can use unfold and then reshape

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

        # THIS IS THE CORRECT MEASURE:
        decor_loss = torch.mean(lower_triangular(torch.square(corr), offset=-1))

        weight = self.weight
        weight = weight.reshape(-1, np.prod(weight.shape[1:]))
        decor_update = torch.einsum('ij,jk->ik', corr, weight)
        decor_update = decor_update.reshape(self.weight.shape)

        self.weight.grad = decor_update.clone()

        return decor_loss.item(), 0.0
    


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

        # self.conv used in 2 places:
        # - weight convolution. But weight is simply [size, in_channels, kernel_size] so we can use unfold and then reshape
        # - forward convolution. This is a 1x1 convolution so we can use unfold and then reshape

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
    

    
class DecorConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = False,
                 downsample_perc=1.0, kappa=0.0, device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.downsample_perc = downsample_perc
        # self.d = in_channels * np.prod(kernel_size)
        self.kappa = kappa

        # # this generates patches (we could use unfold for this)
        # super().__init__(
        #     in_channels=in_channels,
        #     out_channels=self.d,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     bias=False,
        #     **factory_kwargs
        # )
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.decorrelator = Decorrelation(size=in_channels * np.prod(kernel_size), kappa=kappa, **factory_kwargs)

        # this applies the kernel weights W
        self.forward_conv = nn.Conv2d(in_channels=in_channels * np.prod(kernel_size),
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=0,
                                        dilation=(1, 1),
                                        bias=bias is not None, # ?
                                        **factory_kwargs)

        # self.weight.requires_grad_(False)

        self.input = None

        # self.register_buffer('eye', torch.eye(self.d, **factory_kwargs))
        # self.register_buffer('neg_eye', 1.0 - torch.eye(self.d, **factory_kwargs))

    # def reset_parameters(self) -> None:
    #     w_square = self.weight.reshape(self.d, self.d)
    #     w_square = torch.nn.init.eye_(w_square)
    #     self.weight = torch.nn.Parameter(w_square.reshape(self.d, self.in_channels, *self.kernel_size))
    #     # reset downstream params?

    def forward(self, x: Tensor) -> Tensor:

        self.input = x

        # combines the patch-wise R update with the convolutional W update
        # 50 x 2 x 5 x 5
        
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
        X = X.moveaxis(1,3).reshape(-1, X.shape[1])

        # strictly lower triangular part of x x' averaged over datapoints
        L = torch.tril(X.T @ X) / len(X)

        # unit variance term averaged over datapoints; faster via kronecker?
        V = torch.diag(torch.mean(torch.square(X), axis=0) - 1)

        # compute update; equation (3) in technical note

        # HERE: R = weight; shape of R / W. We should be able to simplify this... ideally by operating on decorrelation weights; but this is the same weight as used for the efficient forward mapping.
        # Can we replace with two separate steps using unfold and still be as efficient?
        self.R.grad = (4.0/self.d) * ( ((1-self.kappa)/(self.d - 1)) * L + self.kappa * V ) @ self.R

        # compute loss; equation (1) in technical note
        return (1.0/self.d) * ( 2*(1-self.kappa)/(self.d - 1) * torch.trace(L @ L.T) + self.kappa * torch.trace(V @ V.T) )



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




class DecorrelationPatch2d(torch.nn.Module):

    def __init__(self,
                in_channels: int,
                kernel_size: _size_2_t,
                whiten: bool = False,
                device=None,
                dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.whiten = whiten

        self.patch_length = in_channels * np.prod(kernel_size) # number of elements in a patch to be represented as channels
        self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=1, padding=0, dilation=1)

        self.fold = None
        self.divisor = None

        self.register_buffer('R', torch.eye(self.patch_length, **factory_kwargs))
        self.register_buffer('eye', torch.eye(self.patch_length, **factory_kwargs))
        self.register_buffer('neg_eye', 1.0 - torch.eye(self.patch_length, **factory_kwargs))

    def forward(self, input: Tensor) -> Tensor:

        # transform from tensor to patch representation
        z = self.unfold(input) # [N, C*H*W, P]

        # apply decorrelating transform
        self.output = torch.einsum('nij,ik->nij', z, self.R) # decorrelated patches [N, C*H*W, P]

        # transform from patch to tensor representation
        if self.fold is None: # we only compute this once
            self.fold = torch.nn.Fold(output_size=input.shape[-2:], kernel_size=self.kernel_size, stride=1, padding=0, dilation=1)
            self.divisor = self.fold(self.unfold(torch.ones_like(input)))

        # we could create a path mask and apply this. But there is likely a more direct route
        # z = self.output
        # self.fold(z)
         
        # BUT THIS SHOULD ALSO JUST WORK.. WHY NOT?
        return self.fold(self.output) / self.divisor # [N, C, H, W]
    
    def update(self):
        C = torch.einsum('nip,njp->ij', self.output, self.output) / (self.output.shape[0] * self.output.shape[2]) # double check if this is correct
        if self.whiten:
            C -= self.eye
        else:
            C *= self.neg_eye
        self.R.grad = torch.einsum('ij,jk->ik', C, self.R)
        
        # decorrelation loss computed per patch
        if self.whiten:
            return torch.mean(torch.square(lower_triangular(C, offset=0)))
        else:
            return torch.mean(torch.square(lower_triangular(C, offset=-1)))
# """
# Options:
#     - patchwise decorrelation as we do now
#     - patchwise with separate R for each patch
#     - patchwise but map back to original (averaged input); advantage is separate decorrelating conv module
#     - expand the whole input and globally decorrelate and reshape back (extreme case of the naive approach); possibly combined with factorization...; advantage is separate decorrelating conv module
# """






class DecorConv2d(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = False,
                 downsample_perc=1.0, kappa=0.0, device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.patch_length = in_channels * np.prod(kernel_size)
        self.downsample_perc = downsample_perc

        # this generates patches (we could use unfold for this)
        # super().__init__(
        #     in_channels=in_channels,
        #     out_channels=self.patch_length,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     bias=False,
        #     **factory_kwargs
        # )
        # however, this does not give the weight transformation used for combination
        self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=1, padding=0, dilation=1)

        self.decorrelator = Decorrelation(d=self.patch_length, kappa=kappa, **factory_kwargs)

        # this applies the kernel weights W
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
        # reset downstream params?

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