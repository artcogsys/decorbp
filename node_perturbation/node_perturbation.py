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
    total_activity_norm = sum([m.squared_norm for m in modules])

    # compute the square root of the number of nodes in the network
    sqrt_n = np.sqrt(sum([m.num_nodes for m in modules]))

    # NOTE: this assumes that all model parameters are updated via node perturbation
    for param in model.parameters():
        param.grad *= delta_loss * sqrt_n / total_activity_norm

class NodePerturbation(nn.Module):
    """Node perturbation wrapper for a module"""

    def __init__(self, num_nodes: int = None, sampler: torch.distributions.Distribution = torch.distributions.Normal(0.0, 1e-3), device=None, dtype=None) -> None:
        super().__init__()
        self.sampler = sampler
        self.num_nodes = num_nodes

        # stores squared norm of the activity difference
        self.squared_norm = torch.empty(1, device=device, dtype=dtype)    

    def perturb(self, input):
        return input + self.sampler.sample(input.shape).to(input.device)


class NPLinear(NodePerturbation):
    """Linear layer with node perturbation"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 sampler: torch.distributions.Distribution = torch.distributions.Normal(0.0, 1e-3), device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_nodes=out_features, sampler=sampler, **factory_kwargs)
        self.linear = nn.Linear(in_features, out_features, bias=bias, **factory_kwargs)
        
    def forward(self, input: Tensor) -> Tensor:

        output = self.linear.forward(torch.concat([input, self.perturb(input)]))
    
        delta_a = output[len(input):] - output[:len(input)]
        self.squared_norm = torch.sum(delta_a**2)

        self.linear.weight.grad = delta_a.T @ input
        self.linear.bias.grad = torch.mean(delta_a, axis=0) # guess

        return output

