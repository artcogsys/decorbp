import numpy as np
import torch
from torch import nn
from torch import Tensor

def np_modules(model: nn.Module):
    """Returns the list of decorrelation modules
    """
    return list(filter(lambda m: isinstance(m, NodePerturbation), model.modules()))

def np_update(model, delta_loss):
    """Update gradients of model parameters given a perturbation in the loss.

    Args:
        model (torch.nn.Module): model
        delta_loss (torch.Tensor): perturbation in the loss
    """

    modules = np_modules(model)

    # return total norm of all activations; NOTE: non-local operation
    total_activity_norm = torch.sum(torch.vstack([m.square_norm for m in modules]), axis=0)

    # compute scaling factor # NOTE: float64; where does it come from?
    sqrt_n = torch.tensor(np.sqrt(sum([m.output_diff.numel() / m.output_diff.shape[0] for m in modules]))).to(delta_loss.device)
    scaling = delta_loss * sqrt_n / total_activity_norm

    # compute gradient updates NOTE: this assumes that all model parameters are updated via node perturbation
    for m in modules:
        m.update(scaling)

class NodePerturbation(nn.Module):
    """Node perturbation wrapper for a module"""

    def __init__(self, transform: nn.Module = None, sampler: torch.distributions.Distribution = torch.distributions.Normal(0.0, 1e-3), clean_pass: bool = True) -> None:
        super().__init__()
        self.transform = transform
        self.sampler = sampler
        self.clean_pass = clean_pass

        # stores squared norm of the activity difference
        self.square_norm = None

        # 'clean' input
        self.clean_input = None

        # stores the activity difference
        self.output_diff = None
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with node perturbation"""

        output = self.perturb(self.transform.forward(input))

        N = len(input) // 2
        self.clean_input = input[:N]
        self.output_diff = output[N:] - output[:N]
        self.square_norm = torch.sum(self.output_diff**2, axis=[a for a in range(1,self.output_diff.ndim)])

        return output

    def perturb(self, input): # NOTE: we still need to add test functionality
        """
        Adds noise to the input
        """
        if self.clean_pass:
            shape = [input.shape[0]//2, *input.shape[1:]]
            return input + torch.concat([torch.zeros(shape), self.sampler.sample(shape).to(input.device)])
        else:
            return input + self.sampler.sample(input.shape).to(input.device)                                     

class NPLinear(NodePerturbation):
    """Linear layer with node perturbation"""

    def __init__(self, in_features: int, out_features: int,
                 sampler: torch.distributions.Distribution = torch.distributions.Normal(0.0, 1e-3), **kwargs) -> None:
        super().__init__(transform=nn.Linear(in_features, out_features, **kwargs), sampler=sampler)

    def forward(self, input: Tensor) -> Tensor:
        # flattens the input
        return super().forward(input.view(input.shape[0], -1))

    def update(self, scaling):

        # scaled_out_diff = scaling.expand(len(self.output_diff))[:, None] * self.output_diff
        error_signal = scaling.view(-1, *[1]*(self.output_diff.ndim-1)) * self.output_diff

        self.transform.weight.grad = error_signal.T @ self.clean_input

        if self.transform.bias is not None:
            self.transform.bias.grad = torch.mean(error_signal, axis=0)


class NPConv2d(NodePerturbation):
    """2d convolution with node perturbation"""

    def __init__(self, in_channels: int, out_channels: int,
                 sampler: torch.distributions.Distribution = torch.distributions.Normal(0.0, 1e-3), **kwargs) -> None:
        super().__init__(transform=nn.Conv2d(in_channels, out_channels, **kwargs), sampler=sampler)

    def update(self, scaling):

        # reshape the input
        patches = torch.nn.functional.unfold(self.clean_input, kernel_size=self.transform.kernel_size, stride=self.transform.stride, padding=self.transform.padding) # NOTE: why padding=0?
        patches = patches.swapaxes(1,2)
        patches = patches.reshape(-1, patches.shape[-1])

        # Reshape the error signal
        error_signal = scaling.view(-1, *[1]*(self.output_diff.ndim-1)) * self.output_diff
        error_signal = error_signal.view(error_signal.size(0), error_signal.size(1), -1).permute(0, 2, 1)
        error_signal = error_signal.reshape(-1, error_signal.size(-1))

        delta_w = error_signal.T @ patches        
        self.transform.weight.grad = delta_w.T.reshape(self.transform.out_channels, self.transform.in_channels, *self.transform.kernel_size)

        if self.transform.bias is not None:
            self.transform.bias.grad = torch.mean(error_signal, axis=0)


        # # Scale output diff by performance diff and normalization factor (scaling_factor)
        # scaled_out_diff = self.output_diff * scaling.view(-1, 1, 1, 1)

        # update = patches.swapaxes(1,2).view(-1, patches.shape[1]).T @ scaled_out_diff        


        # x = self.clean_input
        # kernel_size = (self.forward_conv.kernel_size[0], self.forward_conv.kernel_size[1])
        # stride = (self.forward_conv.stride[0], self.forward_conv.stride[1])

        # grad = _compute_weight_updates(input=x, error_signal=scaled_output_diff, kernel_size=kernel_size[0], stride=stride[0])
        # grad = grad.reshape(self.forward_conv.weight.shape)
        # #print('conv grad')
        # #print(grad)

        # patches = torch.nn.functional.unfold(output, kernel_size=self.transform.kernel_size, stride=self.transform.stride, padding=0) # NOTE: why padding=0?

        # # 
        # patches = torch.nn.functional.unfold(output, kernel_size=self.transform.kernel_size, stride=self.transform.stride, padding=0) # NOTE: why padding=0?
        # patches.swapaxes(1,2).view(-1, patches.shape[1])


        # self.forward_conv.weight.grad = grad

        #if self.bias is not None:
        #    self.bias.grad = torch.einsum("ni->i", scaled_out_diff)


# self.out_features = self.output_diff.reshape(len(self.output_diff), -1).shape[1]
# self.square_norm = torch.sum((self.output_diff.reshape(len(self.output_diff), -1)) ** 2, axis=1)