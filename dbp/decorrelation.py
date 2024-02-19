
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch import Tensor

# class Decor(nn.Module):
#     """
#     A Decorrelation layer flattens the input, decorrelates, updates decorrelation parameters, and returns the reshaped decorrelated input.
#     """

#     lr: float
#     in_features: int
    
#     def __init__(self, in_features, lr):   
#         super().__init__()
#         self.lr = lr
#         self.in_features = in_features

#         # self.R = torch.eye(in_features) #torch.nn.parameter.Parameter(torch.eye(in_features)) #, requires_grad=False)
#         # self.register_buffer('R', self.R, persistent=True)
#         self.register_buffer('R', torch.eye(in_features))

#         self.register_buffer('neg_eye', 1.0 - torch.eye(self.in_features))
#         # self.neg_eye = torch.nn.parameter.Parameter(1.0 - torch.eye(in_features), requires_grad=False)

#     def forward(self, input: Tensor) -> Tensor:
        
#         # decorrelate flattened input
#         output = torch.einsum('ni,ij->nj', input.view(len(input), -1), self.R)
       
#         # update parameters
#         # with torch.no_grad():
#         corr = (1/len(output))*torch.einsum('ni,nj->ij', output, output) * self.neg_eye
#         self.R -= self.lr * torch.einsum('ij,jk->ik', corr, self.R)

#         # debug
#         # print(f'cor: {torch.mean(corr[torch.tril_indices(len(output), len(output), offset=1)])}')

#         return output.view(input.shape)
  

class Decorrelation(nn.Module):
    r"""A Decorrelation layer flattens the input, decorrelates, updates decorrelation parameters, and returns the reshaped decorrelated input.
    """

    __constants__ = ['in_features']
    in_features: int

    def __init__(self, in_features: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        # self.R = torch.nn.parameter.Parameter(torch.eye(in_features, **factory_kwargs), requires_grad=False)
        self.register_buffer('R', torch.eye(self.in_features, **factory_kwargs))
        self.register_buffer('output', torch.zeros(in_features, **factory_kwargs))
        self.register_buffer('eye', torch.eye(self.in_features, **factory_kwargs))

    def reset_parameters(self) -> None:
        nn.init.eye(self.R)
        nn.init.zeros(self.output)
   
    def forward(self, input: Tensor) -> Tensor:
        self.output = F.linear(input.view(len(input), -1), self.R) # UPDATE ORDER? ALSO ELSEWHERE
        return self.output.view(input.shape)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}'

    def update(self):
        corr = (1/len(self.output))*torch.einsum('ni,nj->ij', self.output, self.output) - self.eye
        self.R.grad = torch.einsum('ij,jk->ik', corr, self.R)

    def correlation(self):
        x = self.output
        C = (x @ x.T) / len(x)
        return torch.mean(C[torch.tril_indices(len(C), len(C), offset=1)])


# class Sub(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.d = Decorrelation(2)

# class Test(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.d = Decorrelation(2)
#         self.s = Sub()
#         self.l = nn.Linear(2,2)

def decorrelation_modules(model: torch.nn.Module):

    def get_modules(model: torch.nn.Module):
        children = list(model.children())
        return [model] if len(children) == 0 else [ci for c in children for ci in get_modules(c)]

    return list(filter(lambda m: isinstance(m, Decorrelation), get_modules(model)))

def decorrelation_parameters(model: torch.nn.Module):
    return list(map(lambda m: m.R, decorrelation_modules(model)))

def decorrelation_update(modules):
    for m in modules:
        m.update()     

def correlation(model, input):
    model.forward(input)
    corr = 0.0
    for idx, m in enumerate(decorrelation_modules(model)):
        corr += m.correlation()
    return corr / idx