import torch.nn as nn
from decorrelation.decorrelation import DecorLinear, DecorConv2d

class MLP(nn.Sequential):
    """Simple MLP example"""

    def __init__(self, in_features, kappa):
        """
        Args:
            in_features: int, number of inputs
            kappa: decorrelation strength
        """
        super().__init__(DecorLinear(in_features, 100, kappa=kappa),
                        nn.LeakyReLU(),
                        DecorLinear(100, 10, kappa=kappa)
                        )

    def forward(self, x):
        return super().forward(x.view(len(x), -1))
    
    
class ConvNet(nn.Sequential):
    """Simple ConvNet example"""

    def __init__(self, in_channels, kappa):
        """
        Args:
            in_channels: int, number of input channels
            kappa: decorrelation strength
        """

        super().__init__(DecorConv2d(in_channels, out_channels=5, kernel_size=(5,5), kappa=kappa, downsample_perc=0.1),
                        nn.LeakyReLU(),
                        nn.Flatten(),
                        DecorLinear(2880, 10, kappa=kappa)
                        )

    def forward(self, x):
        return super().forward(x)
    

    # class LoadableNet(nn.Module, Copi):
    # def __init__(self, model_name, alpha, algorithm):
    #     super().__init__()
    #     self.model = self.load_net(model_name)
    #     self.alpha = alpha
    #     if algorithm == 'copi':
    #         self._replace_modules(self.model, module=nn.Linear, replacement_fn=self._copi_linear)
    #         self._replace_modules(self.model, module=nn.Conv2d, replacement_fn=self._copi_conv)
    #         self._replace_modules(self.model, module=nn.BatchNorm2d, replacement_fn=lambda x: IdentityModule())
    #         self._replace_modules(self.model, module=nn.ReLU, replacement_fn=self._relu)

    #     elif algorithm == 'decor_bp':
    #         self._replace_modules(self.model, module=nn.Linear, replacement_fn=self._decor_bp_linear)
    #         self._replace_modules(self.model, module=nn.Conv2d, replacement_fn=self._decor_bp_conv)
    #         self._replace_modules(self.model, module=nn.BatchNorm2d, replacement_fn=lambda x: IdentityModule())

    #     elif algorithm == 'bp':
    #         self._replace_modules(self.model, module=nn.Linear, replacement_fn=self._bp_linear)
    #         self._replace_modules(self.model, module=nn.Conv2d, replacement_fn=self._bp_conv)
    #         self._replace_modules(self.model, module=nn.BatchNorm2d, replacement_fn=lambda x: torch.nn.Identity())
    #         #self._replace_modules(self.model, module=nn.BatchNorm2d, replacement_fn=self._bn_fixed)

    #     self.copi_updatables = self._get_copi_updatables(self.model)

    # def forward(self, x):
    #     return self.model(x)