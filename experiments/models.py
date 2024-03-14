import torch
import torch.nn as nn
from decorrelation.decorrelation import DecorLinear, DecorConv2d

class MLP(nn.Sequential):
    """Simple MLP example"""

    def __init__(self, in_features, args):
        super().__init__(DecorLinear(in_features, 100, decor_lr=args.decor_lr, bias_lr=args.bias_lr, kappa=args.kappa),
                        nn.LeakyReLU(),
                        DecorLinear(100, 10, decor_lr=args.decor_lr, bias_lr=args.bias_lr, kappa=args.kappa)
                        )

    def forward(self, x):
        return super().forward(x.view(len(x), -1))
    
    
class ConvNet(nn.Sequential):
    """Simple ConvNet example"""

    def __init__(self, in_channels, args):
        super().__init__(DecorConv2d(in_channels, out_channels=5, kernel_size=(5,5), decor_lr=args.decor_lr, bias_lr=args.bias_lr,
                                     kappa=args.kappa, downsample_perc=args.downsample_perc),
                        nn.LeakyReLU(),
                        nn.Flatten(),
                        DecorLinear(2880, 10, decor_lr=args.decor_lr, bias_lr=args.bias_lr, kappa=args.kappa)
                        )

    def forward(self, x):
        return super().forward(x)
    

class LoadableNet(nn.Module):

    def __init__(self, model_name, *, decor_lr: float = 0.0, bias_lr: float = 0.0, kappa = 1e-3, full: bool = True, downsample_perc: float =1.0, device = None, dtype = None):
        super().__init__()
        factory_kwargs = {'decor_lr': decor_lr, 'bias_lr': bias_lr, 'kappa': kappa, 'full': full, 'downsample_perc': downsample_perc, 'device': device, 'dtype': dtype}
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name)
        self.replace_modules(self.model, **factory_kwargs)
       
    def forward(self, x):
        return self.model(x)

    def replace_modules(self, module, **kwargs):
        """
        replaces specific modules in model by modules specified by the replacement_fn
        """
        for name, layer in module.named_children():
            if isinstance(layer, nn.Linear):
                module.__setattr__(name, DecorLinear(layer.in_features, layer.out_features, bias=layer.bias is not None, **kwargs))
            elif isinstance(layer, nn.Conv2d):
                module.__setattr__(name, DecorConv2d(layer.in_channels, layer.out_channels, layer.kernel_size,
                                                     stride = layer.stride, padding=layer.padding, dilation=layer.dilation,
                                                     bias=layer.bias is not None, **kwargs))
            elif isinstance(layer, nn.BatchNorm2d):
                module.__setattr__(name, nn.Identity())
            elif layer.children() is not None:
                self.replace_modules(layer)

if __name__ == '__main__':   
    model = LoadableNet('resnet18')
    print(model.model)


# class LoadableNet(nn.Module, Copi):
#     def __init__(self, model_name, algorithm):
#         super().__init__()
#         self.model = self.load_net(model_name)

#         if algorithm == 'decor_bp':
#             self._replace_modules(self.model, module=nn.Linear, replacement_fn=self._decor_bp_linear)
#             self._replace_modules(self.model, module=nn.Conv2d, replacement_fn=self._decor_bp_conv)
#             self._replace_modules(self.model, module=nn.BatchNorm2d, replacement_fn=lambda x: IdentityModule())

#         elif algorithm == 'bp':
#             self._replace_modules(self.model, module=nn.Linear, replacement_fn=self._bp_linear)
#             self._replace_modules(self.model, module=nn.Conv2d, replacement_fn=self._bp_conv)
#             self._replace_modules(self.model, module=nn.BatchNorm2d, replacement_fn=lambda x: torch.nn.Identity())
#             #self._replace_modules(self.model, module=nn.BatchNorm2d, replacement_fn=self._bn_fixed)

#         elif algorithm == 'np':
#             self._replace_modules(self.model, module=nn.Linear, replacement_fn=self._np_linear)
#             #self._replace_modules(self.model, module=nn.Conv2d, replacement_fn=self._bp_conv)
#             self._replace_modules(self.model, module=nn.BatchNorm2d, replacement_fn=lambda x: torch.nn.Identity())


#         self.copi_updatables = self._get_copi_updatables(self.model)

#     def forward(self, x):
#         return self.model(x)

#     def copi_parameters(self):
#         def copi_parameters(model):
#             params = []
#             for module in model.children():
#                 if hasattr(module, 'copi_parameters'):
#                     params.extend(module.copi_parameters())
#                 elif module.children() is not None:
#                     params.extend(copi_parameters(module))
#             return params

#         return copi_parameters(self.model)
    
#     def get_updatables(self):
#         return self.copi_updatables
    
#     def _get_copi_updatables(self, model):
#         p = []
#         for module in model.children():
#             if hasattr(module, 'copi_update'):
#                 p.append(module)
#             elif module.children() is not None:
#                 p.extend(self._get_copi_updatables(module))

#         return p

#     def copi_update(self, **kwargs):
#         [m.copi_update(**kwargs) for m in self.copi_updatables]

#     @staticmethod
#     def load_net(model_name):
#         return torch.hub.load('pytorch/vision:v0.10.0', model_name, weights=None)

#     def _replace_modules(self, model, module, replacement_fn):
#         """
#         replaces specific modules in model by modules specified by the replacement_fn
#         """
#         for name, layer in model.named_children():
#             if isinstance(layer, module):
#                 new_layer = replacement_fn(layer)
#                 model.__setattr__(name, new_layer)
#             elif layer.children() is not None:
#                 self._replace_modules(layer, module, replacement_fn)

#     def _relu(selfs, x):
#         x.inplace = False
#         return x

#     @staticmethod
#     def _decor_bp_linear(x):
#         decor = Decorrelation(x.in_features)
#         return DecorLinear(x.in_features, x.out_features, decor=decor, bias=x.bias is not None, weight=x.weight)

#     @staticmethod
#     def _decor_bp_conv(x):
#         return DecorConv2d(x.in_channels, x.out_channels, x.kernel_size, bias=x.bias is not None, stride=x.stride,
#                            padding=x.padding, dilation=x.dilation, copi=False, weight=x.weight, dtype=torch.float32)

#     @staticmethod
#     def _bp_linear(x):
#         return nn.Linear(x.in_features, x.out_features, bias=x.bias is not None)

#     @staticmethod
#     def _np_linear(x):
#         return NPLinear(x.in_features, x.out_features, bias=x.bias is not None)

#     @staticmethod
#     def _bp_conv(x):
#         return nn.Conv2d(x.in_channels, x.out_channels, x.kernel_size, bias=x.bias is not None, stride=x.stride,
#                            padding=x.padding, dilation=x.dilation, dtype=torch.float32)

#     @staticmethod
#     def _bn_fixed(x):
#         return nn.BatchNorm2d(eps=1e-3, num_features=x.num_features)