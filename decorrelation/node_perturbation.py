import torch
from torch import nn

class NPModule(nn.Module):
    """Node perturbation wrapper for a module"""

    def __init__(self, module: nn.Module, sampler: torch.distributions.Distribution = torch.distributions.Normal(0.0, 1.0), device=None, dtype=None) -> None:
        super().__init__()
        self.module = module.to(device)
        self.sampler = sampler

    def forward(self, input):
        noise = self.sampler.sample(input.shape).to(input.device)
        output = self.module(torch.concat([input, input + noise]))

        self.module.weight.grad

        return output
    


    