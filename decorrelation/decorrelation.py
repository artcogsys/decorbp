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

def decor_update(modules):
    """Updates all decorrelation modules and returns decorrelation loss per decorrelation module
    """
    loss = np.zeros(len(modules))
    for i, m in enumerate(modules):
        loss[i] = m.update().cpu().detach().numpy()
    return loss

def decor_loss(modules):
    """Returns decorrelation loss
    """
    loss = np.zeros(len(modules))
    for i, m in enumerate(modules):
        loss[i] = m.loss().cpu().detach().numpy()
    return loss

def lower_triangular(C: Tensor, offset: int):
    """Return lower triangular elements of a matrix as a vector
    """
    return C[torch.tril_indices(C.shape[0], C.shape[1], offset=offset).unbind()]

class Decorrelation(nn.Module):
    """A Decorrelation layer flattens the input, decorrelates, updates decorrelation parameters, and returns the reshaped decorrelated input"""

    def __init__(self, in_features: int, method: str = 'standard', decor_lr: float = 0.0, kappa: float = 0.5, full=True, downsample_perc: float = 1.0, device = None, dtype = None) -> None:
        """"Params:
            - in_features: input dimensionality
            - method: method for decorrelation: default is 'standard' with kappa=0.5 (original whitening inducing approach)
                - 'standard' (original approach weighting between decorrelation and whitening using kappa
                    (kappa=0 is pure decorrelation; kappa=1 is pure variance normalising; kappa=0.5 is original balancel 0 < kappa < 1 is whitening inducing)
                - 'normalized' (weighted approach normalized by input size)
            - decor_lr: decorrrelation learning rate
            - kappa: decorrelation strength (0-1)
            - full: learn a full (True) or lower triangular (False) decorrelation matrix
            - downsample_perc: downsampling for covariance computation
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device
        self.dtype = dtype
        super().__init__()
        
        self.decor_lr = decor_lr
        self.downsample_perc = downsample_perc
        self.method = method

        # self.optimizer = torch.optim.SGD([self.weight], lr=decor_lr, momentum=1e-2, dampening=0, weight_decay=0, nesterov=True)

        self.kappa = kappa
        self.full = full

        if in_features is not None:
            self.in_features = in_features
            self.register_buffer("weight", torch.empty(self.in_features, self.in_features, **factory_kwargs))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.weight)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """In case of node perturbation we do not need to store decor_state and can immediately update the decorrelation parameters
        """
        if self.training:
            if self.downsample_perc != 1.0:
                self.decor_state = F.linear(self.downsample(input).view(-1, np.prod(input.shape[1:])), self.weight)
                return self.decorrelate(input)
            else:
                self.decor_state = F.linear(input.view(len(input), -1), self.weight)
                return self.decor_state.view(input.shape)
        else:
            return self.decorrelate(input)
        
    def decorrelate(self, input: Tensor):
        """Applies the decorrelating transform. Can be overloaded in composite functions to return the decorrelating transform.
        Maps back to the original input.
        """
        return F.linear(input.view(len(input), -1), self.weight).view(input.shape)

    def update(self, loss_only=False):
        """Implements decorrelation update
        Args:
            - loss_only: if True, only the loss is computed and returned
        """

        # covariance; NOTE: expensive operation
        X = self.decor_state.T @ self.decor_state / len(self.decor_state)

        if self.full:
            # remove diagonal
            C = X - torch.diag(torch.diag(X))
        else:
            # strictly lower triangular part of x x' averaged over datapoints and normalized by square root of number of non-zero entries
            C = torch.sqrt(torch.arange(self.in_features)) * torch.tril(X, diagonal=-1)

        # unit variance term averaged over datapoints
        v = torch.mean(self.decor_state**2 - 1.0, axis=0)

        match self.method:

            case 'standard':

                # original decorrelation rule
                if not loss_only:
                    self.weight.data -= self.decor_lr * ((1.0 - self.kappa) * C @ self.weight + self.kappa * v * self.weight)

                # compute loss; we divide by the number of matrix elements
                return ((1-self.kappa) * torch.sum(C**2) + self.kappa * torch.sum(v**2)) / self.in_features**2

            case 'normalized':

                # compute update; NOTE: expensive operation
                if not loss_only:
                    self.weight.data -= self.decor_lr * (((1.0 - self.kappa)/(self.in_features-1)) * C @ self.weight + self.kappa * 2 * v * self.weight)

                # compute loss
                return (1/self.in_features) * (((1-self.kappa)/(self.in_features-1)) * torch.sum(C**2) + self.kappa * torch.sum(v**2))

            case _:
                raise ValueError(f"Unknown method: {self.method}")
        
        
    def loss(self):
        return self.update(loss_only=True)

    def downsample(self, input: Tensor):
        """Downsamples the input for covariance computation"""
        if self.downsample_perc is None:
            num_samples = torch.min([len(input), self.in_features+1])
            idx = np.random.choice(np.arange(len(input)), size=num_samples)
            return input[idx]
        elif self.downsample_perc < 1.0:
            num_samples = int(len(input) * self.downsample_perc)
            idx = np.random.choice(np.arange(len(input)), size=num_samples)
            return input[idx]
        else:
            return input


class DecorLinear(Decorrelation):
    """Linear layer with input decorrelation"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, method: str = 'standard', decor_lr: float = 0.0,
                 kappa = 1e-3, full: bool = True, downsample_perc:float =1.0, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, method=method, decor_lr=decor_lr, kappa=kappa, full=full, downsample_perc=downsample_perc, **factory_kwargs)
        self.linear = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, input: Tensor) -> Tensor:
        return self.linear.forward(super().forward(input))      
    

class DecorConv2d(Decorrelation):
    """2d convolution with input decorrelation"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 bias: bool = True, method: str = 'standard', decor_lr: float = 0.0, kappa = 1e-3, full: bool = True,
                 downsample_perc=1.0, device=None, dtype=None) -> None:
        """
        Args:
            - in_channels: number of input channels
            - out_channels: number of output channels
            - kernel_size: size of the convolving kernel
            - stride: stride of the convolution
            - padding: zero-padding added to both sides of the input
            - dilation: spacing between kernel elements
            - decor_dilation: dilation arg for decor operation
            - bias: whether to add a learnable bias to the output
            - method: decorrelation method
            - decor_lr: decorrelation learning rate
            - kappa: decorrelation strength (0-1)
            - full: learn a full (True) or lower triangular (False) decorrelation matrix
            - downsample_perc: downsampling for covariance computation
        """

        # define decorrelation layer
        super().__init__(in_features=in_channels * np.prod(kernel_size), method=method, decor_lr=decor_lr, kappa=kappa, full=full, downsample_perc=downsample_perc, device=device, dtype=dtype)        
        
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
                                        bias=bias,
                                        device=device,
                                        dtype=dtype)

        self.input = None

    def forward(self, input: Tensor) -> Tensor:

        if self.training:
            # we store a downsampled version for input decorrelation and diagonal computation
            self.decor_state = self.decorrelate(self.downsample(input)).reshape(-1, self.in_features)

        # efficiently combines the patch-wise R update with the convolutional W update on all data
        weight = nn.functional.conv2d(self.weight.view(self.in_features, self.in_channels, *self.kernel_size).moveaxis(0, 1),
                                      self.forward_conv.weight.flip(-1, -2),
                                      padding=0).moveaxis(0, 1)
                
        # applies the combined weight to the non-downsampled input to generate the desired output
        self.forward_conv.output = nn.functional.conv2d(input, weight,
                                         stride=self.stride,
                                         dilation=self.dilation,
                                         padding=self.padding)

        # needed for BP gradient propagation
        self.forward_conv.output.requires_grad_(True)
        self.forward_conv.output.retain_grad()
        
        return self.forward_conv.output
    
    def decorrelate(self, input: Tensor):
        """Applies the patchwise decorrelating transform and returns decorrelated feature maps
        """
        return nn.functional.conv2d(input, self.weight.view(self.in_features, self.in_channels, *self.kernel_size),
                                    bias=None, stride=self.stride, padding=self.padding, 
                                    dilation=self.dilation).moveaxis(1, 3)
    
class DecorConvTranspose2d(Decorrelation):
    """2d transposed convolution with input decorrelation"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 bias: bool = True, method: str = 'standard', decor_lr: float = 0.0, kappa = 1e-3, full: bool = True,
                 downsample_perc=1.0, device=None, dtype=None, weights=None) -> None:
        """
        Args:
            - in_channels: number of input channels
            - out_channels: number of output channels
            - kernel_size: size of the convolving kernel
            - stride: stride of the convolution
            - padding: zero-padding added to both sides of the input
            - dilation: spacing between kernel elements
            - decor_dilation: dilation arg for decor operation
            - bias: whether to add a learnable bias to the output
            - method: decorrelation method
            - decor_lr: decorrelation learning rate
            - kappa: decorrelation strength (0-1)
            - full: learn a full (True) or lower triangular (False) decorrelation matrix
            - downsample_perc: downsampling for covariance computation
        """

        # define decorrelation layer
        super().__init__(in_features=None, method=method, decor_lr=decor_lr, kappa=kappa, full=full, downsample_perc=downsample_perc, device=device, dtype=dtype)        
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.loaded_weights = weights

        self.initialized = False # Flag to check if weights are initialized

        self.input = None

    def reset_parameters(self):
        """This will be called once input size is known (in the first forward pass)."""
        if self.in_features is not None:
            # Initialize weight with the identity matrix for decorrelation
            nn.init.eye_(self.weight)

    def initialize_weights(self, input_shape):
        """Initialize weights based on the input shape"""
        _, _, h, w = input_shape

        if h == 1 and w == 1:
            # For 1x1 input, we decorrelate linearly
            self.in_features = self.in_channels
            self.register_buffer("weight", torch.empty(self.in_features, self.in_features, device=self.device, dtype=self.dtype))

            self.forward_conv_transpose = nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                device = self.device,
                dtype = self.dtype
            )
            self.forward_conv_transpose.weight.data = self.loaded_weights
        else:
            # For larger input, perform patchwise decorrelation
            self.in_features = self.in_channels * np.prod(self.kernel_size)
            self.register_buffer("weight", torch.empty(self.in_features, self.in_features, device=self.device, dtype=self.dtype))
            self.forward_conv_transpose = nn.ConvTranspose2d(
                in_channels=self.out_channels,
                out_channels=self.in_features,
                kernel_size=(1,1),  # Pointwise convolution after decorrelation
                stride=(1,1),
                padding=0,
                dilation=(1,1),
                device = self.device,
                dtype = self.dtype
            )
            self.forward_conv_transpose.weight.data = self.loaded_weights.view(self.out_channels, self.in_features, 1, 1)

        # Call reset_parameters to initialize the decorrelation matrix properly
        self.reset_parameters()

        self.initialized = True  # Mark as initialized

    def forward(self, input: Tensor) -> Tensor:
        if not self.initialized:
            self.initialize_weights(input.shape)

        if input.size(2) == 1 and input.size(3) == 1:
            if self.training:
                self.decor_state = F.linear(self.downsample(input).view(-1, np.prod(input.shape[1:])), self.weight)
            self.forward_conv_transpose.output = self.forward_conv_transpose(super().decorrelate(input))
            self.forward_conv_transpose.output.requires_grad_(True)
            self.forward_conv_transpose.output.retain_grad()
            return self.forward_conv_transpose.output
        
        else:
            if self.training:
                # we store a downsampled version for input decorrelation and diagonal computation
                self.decor_state = self.decorrelate(self.downsample(input)).reshape(-1, self.in_features)

            # efficiently combines the patch-wise R update with the convolutional W update on all data
            weight = nn.functional.conv2d(self.forward_conv_transpose.weight,
                                          self.weight.view(self.in_features, self.in_features, 1, 1),
                                          padding=0)
                    
            # applies the combined weight to the non-downsampled input to generate the desired output
            self.forward_conv_transpose.output = nn.functional.conv_transpose2d(input, weight.reshape(self.in_channels, self.out_channels, *self.kernel_size),
                                            stride=self.stride,
                                            dilation=self.dilation,
                                            padding=self.padding)

            # needed for BP gradient propagation
            self.forward_conv_transpose.output.requires_grad_(True)
            self.forward_conv_transpose.output.retain_grad()
            
            return self.forward_conv_transpose.output
    
    def decorrelate(self, input: Tensor):
        """Applies the patchwise decorrelating transform and returns decorrelated feature maps
        """
        return nn.functional.conv2d(input, self.weight.view(self.in_features, self.in_channels, *self.kernel_size),
                                    bias=None, stride=self.stride, padding=self.padding, 
                                    dilation=self.dilation).moveaxis(1, 3)