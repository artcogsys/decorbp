import torch
import numpy as np
from decorrelation.decorrelation import decor_modules, decor_update, decor_loss #, covariance
from time import time
from node_perturbation.node_perturbation import np_update

def generate_correlated_data(d, num_samples, strength=0.3, dtype=torch.float32):
    """Generate correlated data (ugly solution; we could use vines)

    Args:
        d (int): dimensionality
        num_samples (int): number of samples
        strength (float): strength of correlation (0-1); 0.0 implies uncorrelated data (in infinite data limit)
        dtype (torch.dtype): data type
    """
    dist = torch.distributions.MultivariateNormal(torch.zeros(d), (1-strength) * torch.eye(d) + strength * torch.ones((d, d)))
    data = dist.sample((num_samples,)).to(dtype)    
    return (data - torch.mean(data, axis=0)) / torch.std(data, axis=0)

def decor_train(args, model, lossfun, train_loader, device, decorrelate=True, interval=1):
    """Train using (decorrelated) backpropagation.

    Args:
        args (argparse.Namespace): command line arguments
        model (torch.nn.Module): model
        lossfun (callable): loss function
        train_loader (torch.utils.data.DataLoader): training data loader
        device (torch.device): device
        decorrelate (bool): whether or not to train decorrelation layers
        interval (int): interval for decorrelation updates
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.lr > 0.0 else None

    if decorrelate:
        decorrelators = decor_modules(model)
    
    L = np.zeros(args.epochs+1) # loss
    D = np.zeros(args.epochs+1) # decorrelation loss
    T = np.zeros(args.epochs+1) # time per train epoch
    for epoch in range(args.epochs+1):

        tic = time()
        for batchnum, batch in enumerate(train_loader):
        
            optimizer.zero_grad() if args.lr > 0.0 else None
          
            input = batch[0].to(device)
            target = batch[1].to(device)

            loss = lossfun(model(input), target)

            if epoch > 0 and args.lr > 0.0:
                loss.backward()
                optimizer.step()

            if decorrelate and epoch % interval == 0:
                if epoch > 0:
                    D[epoch] += decor_update(decorrelators)
                else:
                    D[epoch] += decor_loss(decorrelators)
    
            L[epoch] += loss
                     
        L[epoch] /= batchnum

        if epoch > 0:
            T[epoch] = time() - tic

        print(f'epoch {epoch:<3}\ttime:{T[epoch]:.3f} s\tbp loss: {L[epoch]:3f}\tdecorrelation loss: {D[epoch]:3f}')

    return model, L, D, T
