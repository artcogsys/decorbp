import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.nn.common_types import _size_2_t

def decor_modules(model: nn.Module):
    """Returns the list of decorrelation modules.
    
    Args:
        model (nn.Module): The PyTorch model containing the modules.

    Returns:
        List: A list of modules of type Decorrelation.
    """
    return list(filter(lambda m: isinstance(m, Decorrelation), model.modules()))

def decor_update(modules):
    """Updates all decorrelation modules and returns decorrelation loss per decorrelation module.
    
    Args:
        modules (List[Decorrelation]): List of decorrelation modules.

    Returns:
        np.ndarray: Array of losses for each decorrelation module.
    """
    loss = np.zeros(len(modules))
    for i, m in enumerate(modules):
        loss[i] = m.update().cpu().detach().numpy()
    return loss

def decor_loss(modules):
    """Returns decorrelation loss.
    
    Args:
        modules (List[Decorrelation]): List of decorrelation modules.

    Returns:
        np.ndarray: Array of losses for each decorrelation module.
    """
    loss = np.zeros(len(modules))
    for i, m in enumerate(modules):
        loss[i] = m.loss().cpu().detach().numpy()
    return loss

def lower_triangular(C: Tensor, offset: int):
    """Return lower triangular elements of a matrix as a vector.
    
    Args:
        C (Tensor): Input matrix.
        offset (int): Diagonal offset.

    Returns:
        Tensor: Lower triangular elements of the input matrix as a vector.
    """
    return C[torch.tril_indices(C.shape[0], C.shape[1], offset=offset).unbind()]

class Decorrelation(nn.Module):
    """A Decorrelation layer flattens the input, decorrelates, updates decorrelation parameters, 
    and returns the reshaped decorrelated input.
    """

    def __init__(self, in_features: int, method: str = 'standard', decor_lr: float = 0.0, 
                 kappa: float = 0.5, full=True, downsample_perc: float = 1.0, device=None, dtype=None) -> None:
        """Initialize the Decorrelation layer.

        Args:
            in_features (int): Input dimensionality.
            method (str): Method for decorrelation. Default is 'standard'.
            decor_lr (float): Decorrelation learning rate.
            kappa (float): Decorrelation strength (0-1).
            full (bool): Whether to learn a full or lower triangular decorrelation matrix.
            downsample_perc (float): Downsampling percentage for covariance computation.
            device (optional): Device to run the layer on.
            dtype (optional): Data type of the input.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device
        self.dtype = dtype
        super().__init__()
        
        self.decor_lr = decor_lr
        self.downsample_perc = downsample_perc
        self.method = method

        self.kappa = kappa
        self.full = full

        if in_features is not None:
            self.in_features = in_features
            self.register_buffer("weight", torch.empty(self.in_features, self.in_features, **factory_kwargs))
            self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters by initializing the weight matrix as an identity matrix."""
        nn.init.eye_(self.weight)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Decorrelation layer. Applies decorrelation transformation during training.
        
        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Decorrelated output tensor.
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
        """Applies the decorrelating transform.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Decorrelated output tensor.
        """
        return F.linear(input.view(len(input), -1), self.weight).view(input.shape)

    def update(self, loss_only=False):
        """Implements decorrelation update.
        
        Args:
            loss_only (bool): If True, only the loss is computed and returned.

        Returns:
            Tensor: Loss value.
        """
        X = self.decor_state.T @ self.decor_state / len(self.decor_state)

        if self.full:
            C = X - torch.diag(torch.diag(X))
        else:
            C = torch.sqrt(torch.arange(self.in_features).to(self.device)) * torch.tril(X.to(self.device), diagonal=-1)

        v = torch.mean(self.decor_state**2 - 1.0, axis=0)

        match self.method:
            case 'standard':
                if not loss_only:
                    self.weight.data -= self.decor_lr * ((1.0 - self.kappa) * C @ self.weight + self.kappa * v * self.weight)

                return ((1-self.kappa) * torch.sum(C**2) + self.kappa * torch.sum(v**2)) / self.in_features**2

            case 'normalized':
                if not loss_only:
                    self.weight.data -= self.decor_lr * (((1.0 - self.kappa)/(self.in_features-1)) * C @ self.weight + self.kappa * 2 * v * self.weight)

                return (1/self.in_features) * (((1-self.kappa)/(self.in_features-1)) * torch.sum(C**2) + self.kappa * torch.sum(v**2))

            case _:
                raise ValueError(f"Unknown method: {self.method}")
        
    def loss(self):
        """Returns the decorrelation loss."""
        return self.update(loss_only=True)

    def downsample(self, input: Tensor):
        """Downsamples the input for covariance computation.
        
        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Downsampled input tensor.
        """
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
    """Linear layer with input decorrelation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, method: str = 'standard', 
                 decor_lr: float = 0.0, kappa=1e-3, full: bool = True, downsample_perc: float = 1.0, device=None, dtype=None) -> None:
        """Initialize the DecorLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to add a bias term.
            method (str): Decorrelation method.
            decor_lr (float): Decorrelation learning rate.
            kappa (float): Decorrelation strength.
            full (bool): Whether to learn a full decorrelation matrix.
            downsample_perc (float): Downsampling percentage.
            device (optional): Device to run the layer on.
            dtype (optional): Data type of the input.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, method=method, decor_lr=decor_lr, kappa=kappa, full=full, downsample_perc=downsample_perc, **factory_kwargs)
        self.linear = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass combining decorrelation and linear transformation.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after decorrelation and linear transformation.
        """
        return self.linear.forward(super().forward(input))      
    

class DecorConv2d(Decorrelation):
    """2D convolution with input decorrelation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 bias: bool = True, method: str = 'standard', decor_lr: float = 0.0, kappa=1e-3, full: bool = True,
                 downsample_perc=1.0, device=None, dtype=None) -> None:
        """Initialize the DecorConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (_size_2_t): Kernel size.
            stride (_size_2_t): Stride of the convolution.
            padding (_size_2_t): Padding size.
            dilation (_size_2_t): Dilation factor.
            bias (bool): Whether to add a bias term.
            method (str): Decorrelation method.
            decor_lr (float): Decorrelation learning rate.
            kappa (float): Decorrelation strength.
            full (bool): Whether to learn a full decorrelation matrix.
            downsample_perc (float): Downsampling percentage.
            device (optional): Device to run the layer on.
            dtype (optional): Data type of the input.
        """
        super().__init__(in_features=in_channels * np.prod(kernel_size), method=method, decor_lr=decor_lr, kappa=kappa, full=full, downsample_perc=downsample_perc, device=device, dtype=dtype)        
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

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
        """Forward pass of the DecorConv2d layer.
        
        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after decorrelation and convolution.
        """
        if self.training:
            self.decor_state = self.decorrelate(self.downsample(input)).reshape(-1, self.in_features)

        weight = nn.functional.conv2d(self.weight.view(self.in_features, self.in_channels, *self.kernel_size).moveaxis(0, 1),
                                      self.forward_conv.weight.flip(-1, -2),
                                      padding=0).moveaxis(0, 1)
                
        self.forward_conv.output = nn.functional.conv2d(input, weight,
                                         stride=self.stride,
                                         dilation=self.dilation,
                                         padding=self.padding)

        self.forward_conv.output.requires_grad_(True)
        self.forward_conv.output.retain_grad()
        
        return self.forward_conv.output
    
    def decorrelate(self, input: Tensor):
        """Applies the patchwise decorrelating transform and returns decorrelated feature maps.
        
        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Decorrelated output tensor.
        """
        return nn.functional.conv2d(input, self.weight.view(self.in_features, self.in_channels, *self.kernel_size),
                                    bias=None, stride=self.stride, padding=self.padding, 
                                    dilation=self.dilation).moveaxis(1, 3)
    
class DecorConvTranspose2d(Decorrelation):
    """2D transposed convolution with input decorrelation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t,
                 stride: _size_2_t = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 bias: bool = True, method: str = 'standard', decor_lr: float = 0.0, kappa=1e-3, full: bool = True,
                 downsample_perc=1.0, device=None, dtype=None, weights=None, loaded_bias=None) -> None:
        """Initialize the DecorConvTranspose2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (_size_2_t): Kernel size.
            stride (_size_2_t): Stride of the convolution.
            padding (_size_2_t): Padding size.
            dilation (_size_2_t): Dilation factor.
            bias (bool): Whether to add a bias term.
            method (str): Decorrelation method.
            decor_lr (float): Decorrelation learning rate.
            kappa (float): Decorrelation strength.
            full (bool): Whether to learn a full decorrelation matrix.
            downsample_perc (float): Downsampling percentage.
            device (optional): Device to run the layer on.
            dtype (optional): Data type of the input.
            weights (optional): Pre-loaded weights for the layer.
            loaded_bias (optional): Pre-loaded bias for the layer.
        """
        super().__init__(in_features=None, method=method, decor_lr=decor_lr, kappa=kappa, full=full, downsample_perc=downsample_perc, device=device, dtype=dtype)        
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.loaded_weights = weights
        self.loaded_bias = loaded_bias

        self.initialized = False

    def reset_parameters(self):
        """Resets the parameters by initializing the weight matrix."""
        if self.in_features is not None:
            nn.init.eye_(self.weight)

    def initialize_weights(self, input_shape):
        """Initializes the weights based on input shape.
        
        Args:
            input_shape (Tuple): Shape of the input tensor.
        """
        _, _, h, w = input_shape

        if h == 1 and w == 1:
            self.in_features = self.in_channels
            self.register_buffer("weight", torch.empty(self.in_features, self.in_features, device=self.device, dtype=self.dtype))

            self.forward_conv_transpose = nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=self.bias,
                device=self.device,
                dtype=self.dtype
            )
            self.forward_conv_transpose.weight.data = self.loaded_weights
            if self.loaded_bias is not None:
                self.forward_conv_transpose.bias.data = self.loaded_bias
        else:
            self.in_features = self.in_channels * np.prod(self.kernel_size)
            self.register_buffer("weight", torch.empty(self.in_features, self.in_features, device=self.device, dtype=self.dtype))
            self.forward_conv_transpose = nn.ConvTranspose2d(
                in_channels=self.out_channels,
                out_channels=self.in_features,
                bias=self.bias,
                kernel_size=(1,1),
                stride=(1,1),
                padding=0,
                dilation=(1,1),
                device=self.device,
                dtype=self.dtype
            )
            self.forward_conv_transpose.weight.data = self.loaded_weights.view(self.out_channels, self.in_features, 1, 1)
            if self.loaded_bias is not None:
                self.forward_conv_transpose.bias.data = self.loaded_bias

        self.reset_parameters()
        self.initialized = True

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the DecorConvTranspose2d layer.
        
        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after decorrelation and transposed convolution.
        """
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
                self.decor_state = self.decorrelate(self.downsample(input)).reshape(-1, self.in_features)

            weight = nn.functional.conv2d(self.forward_conv_transpose.weight,
                                          self.weight.view(self.in_features, self.in_features, 1, 1),
                                          padding=0)
                    
            self.forward_conv_transpose.output = nn.functional.conv_transpose2d(input, weight.reshape(self.in_channels, self.out_channels, *self.kernel_size),
                                            stride=self.stride,
                                            dilation=self.dilation,
                                            padding=self.padding)

            self.forward_conv_transpose.output.requires_grad_(True)
            self.forward_conv_transpose.output.retain_grad()
            
            return self.forward_conv_transpose.output
    
    def decorrelate(self, input: Tensor):
        """Applies the patchwise decorrelating transform and returns decorrelated feature maps.
        
        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Decorrelated output tensor.
        """
        return nn.functional.conv2d(input, self.weight.view(self.in_features, self.in_channels, *self.kernel_size),
                                    bias=None, stride=self.stride, padding=self.padding, 
                                    dilation=self.dilation).moveaxis(1, 3)
