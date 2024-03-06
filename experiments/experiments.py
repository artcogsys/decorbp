import torch
import numpy as np
from models import *
from data import *

def get_experiment(args, device):
    """
    An experiment is a combination of data, a model and a loss function
    """

    if args.experiment == 'MNIST_MLP':

        train_loader, test_loader, input_dim = get_MNIST(args)
        
        model = MLP(np.prod(input_dim), args).to(device)

        lossfun = torch.nn.CrossEntropyLoss().to(device)

        return model, lossfun, train_loader, test_loader
    
    elif args.experiment == 'MNIST_CONVNET':

        train_loader, test_loader, input_dim = get_MNIST(args)
        
        model = ConvNet(input_dim[0], args).to(device)

        lossfun = torch.nn.CrossEntropyLoss().to(device)

        return model, lossfun, train_loader, test_loader

    else:
        raise ValueError(f'unknown experiment {args.experiment}')



