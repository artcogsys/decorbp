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

    # compute the square root of the number of nodes in the network
    # sqrt_n = np.sqrt(sum([m.num_nodes for m in modules]))

    # compute scaling factor
    sqrt_n = torch.tensor(np.sqrt(sum([m.output_diff.numel() / m.output_diff.shape[0] for m in modules]))).to(delta_loss.device)

    scaling = delta_loss * sqrt_n / total_activity_norm

    # sum_square_norm = 0
    # n = 0
    # for l in model.layers:
    #     if hasattr(l, 'update_grads'):
    #         sum_square_norm += l.square_norm
    #         n += l.out_features
    # sqrt_n = np.sqrt(n)

    # scaling_factor = loss_diff / sum_square_norm**2
    # scaling_factor *= torch.tensor(sqrt_n).to(device)
    # scaling_factor = scaling_factor.to(torch.float32)

    # for l in model.layers:
    #     if hasattr(l, 'update_grads'):
    #         l.update_grads(scaling_factor)

    # compute gradient updates
    for m in modules:
        m.update(scaling)

    # # NOTE: this assumes that all model parameters are updated via node perturbation
    # for param in model.parameters():
    #     param.grad *= delta_loss * sqrt_n / total_activity_norm

class NodePerturbation(nn.Module):
    """Node perturbation wrapper for a module"""

    def __init__(self, num_nodes: int = None, sampler: torch.distributions.Distribution = torch.distributions.Normal(0.0, 1e-3)) -> None:
        super().__init__()
        self.sampler = sampler
        self.num_nodes = num_nodes

        # stores squared norm of the activity difference
        self.square_norm = None

        # 'clean' input
        self.input = None

        # stores the activity difference
        self.output_diff = None

    def perturb(self, input):
        return input + self.sampler.sample(input.shape).to(input.device)


class NPLinear(NodePerturbation):
    """Linear layer with node perturbation"""

    def __init__(self, in_features: int, out_features: int,
                 sampler: torch.distributions.Distribution = torch.distributions.Normal(0.0, 1e-3), **kwargs) -> None:
        super().__init__(num_nodes=out_features, sampler=sampler)
        self.transform = nn.Linear(in_features, out_features, **kwargs)
        
    def forward(self, input: Tensor) -> Tensor:

        self.input = input

        output = self.transform.forward(torch.concat([input, self.perturb(input)]))
    
        self.output_diff = output[len(input):] - output[:len(input)]
        self.square_norm = torch.sum(self.output_diff**2, axis=1)

        return output

    def update(self, scaling):

        scaled_out_diff = scaling.expand(len(self.output_diff))[:, None] * self.output_diff

        # self.linear.weight.grad = delta_a.T @ input
        # self.linear.bias.grad = torch.mean(delta_a, axis=0) # NOTE: check this
        # self.linear.weight.grad = torch.einsum("ni,nj->ij", scaled_out_diff, self.input)
        self.transform.weight.grad = scaled_out_diff.T @ self.input

        if self.transform.bias is not None:
            # self.linear.bias.grad = torch.einsum("ni->i", scaled_out_diff)
            self.transform.bias.grad = torch.mean(scaled_out_diff, axis=0)


class NPConv2d(NodePerturbation):
    """2d convolution with node perturbation"""

    def __init__(self, in_channels: int, out_channels: int,
                 sampler: torch.distributions.Distribution = torch.distributions.Normal(0.0, 1e-3), **kwargs) -> None:

        super().__init__(sampler=sampler)

        self.transform = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input: Tensor) -> Tensor:

        self.input = input

        output = self.transform(torch.concat([input, self.perturb(input)]))


        self.output_diff = output[len(input):] - output[:len(input)]
        self.square_norm = torch.sum((self.output_diff.reshape(len(self.output_diff), -1)) ** 2, axis=1).shape

        return output

    def update(self, scaling):

        # Scale output diff by performance diff and normalization factor (scaling_factor)
        scaled_output_diff = self.output_diff * scaling_factor.view(-1, 1, 1, 1)

        x = self.clean_input
        kernel_size = (self.forward_conv.kernel_size[0], self.forward_conv.kernel_size[1])
        stride = (self.forward_conv.stride[0], self.forward_conv.stride[1])

        grad = _compute_weight_updates(input=x, error_signal=scaled_output_diff, kernel_size=kernel_size[0], stride=stride[0])
        grad = grad.reshape(self.forward_conv.weight.shape)
        #print('conv grad')
        #print(grad)

        patches = torch.nn.functional.unfold(output, kernel_size=self.transform.kernel_size, stride=self.transform.stride, padding=0) # NOTE: why padding=0?

        # 
        patches = torch.nn.functional.unfold(output, kernel_size=self.transform.kernel_size, stride=self.transform.stride, padding=0) # NOTE: why padding=0?
        patches.swapaxes(1,2).view(-1, patches.shape[1])


        self.forward_conv.weight.grad = grad

        #if self.bias is not None:
        #    self.bias.grad = torch.einsum("ni->i", scaled_out_diff)


# self.out_features = self.output_diff.reshape(len(self.output_diff), -1).shape[1]
# self.square_norm = torch.sum((self.output_diff.reshape(len(self.output_diff), -1)) ** 2, axis=1)