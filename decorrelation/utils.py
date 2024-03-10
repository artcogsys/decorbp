import torch
import numpy as np
from decorrelation.decorrelation import decor_modules, decor_update, decor_loss #, covariance
from time import time
from decorrelation.node_perturbation import np_update

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

def train(args, model, lossfun, train_loader, device, decorrelate=True):
    """Train using (decorrelated) backpropagation.

    Args:
        args (argparse.Namespace): command line arguments
        model (torch.nn.Module): model
        lossfun (callable): loss function
        train_loader (torch.utils.data.DataLoader): training data loader
        device (torch.device): device
        decorrelate (bool): whether or not to train decorrelation layers
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

            if decorrelate:
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


def train_np(args, model, lossfun, train_loader, device, decorrelate=True):
    """Train using (decorrelated) node perturbation.

    Args:
        args (argparse.Namespace): command line arguments
        model (torch.nn.Module): model
        lossfun (callable): loss function
        train_loader (torch.utils.data.DataLoader): training data loader
        device (torch.device): device
        decorrelate (bool): whether or not to train decorrelation layers
    """
  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.lr > 0.0 else None

    if decorrelate:
        decorrelators = decor_modules(model)
    
    L = np.zeros(args.epochs+1) # loss
    D = np.zeros(args.epochs+1) # decorrelation loss
    T = np.zeros(args.epochs+1) # time per train epoch

    with torch.no_grad():
        
        for epoch in range(args.epochs+1):

            tic = time()
            for batchnum, batch in enumerate(train_loader):
            
                optimizer.zero_grad() if args.lr > 0.0 else None
                            
                input = batch[0].to(device)
                target = batch[1].to(device)

                output = model(input)

                loss = lossfun(output[:len(input)], target)
                L[epoch] += loss

                delta_loss = lossfun(output[len(input):], target) - loss
                np_update(model, delta_loss)

                if epoch > 0:
                    optimizer.step()

                if decorrelate:
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

