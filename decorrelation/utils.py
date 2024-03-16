import torch
import numpy as np
from decorrelation.decorrelation import decor_modules, decor_update, decor_loss #, covariance
from time import time
from tqdm.auto import tqdm

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

def decor_train(args, model, lossfun, train_loader, test_loader=None, device=None, decorrelate=True):
    """Train using (decorrelated) backpropagation.

    Args:
        args (argparse.Namespace): command line arguments
        model (torch.nn.Module): model
        lossfun (callable): loss function
        train_loader (torch.utils.data.DataLoader): training data loader
        test_loader (torch.utils.data.DataLoader): test data loader
        device (torch.device): device
        decorrelate (bool): whether or not to train decorrelation layers
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.lr > 0.0 else None

    if decorrelate:
        decorrelators = decor_modules(model)
        D = np.zeros((args.epochs+1, len(decorrelators)))
    else:
        D = np.zeros(args.epochs+1)
    
    train_loss = np.zeros(args.epochs+1)
    test_loss = np.zeros(args.epochs+1)
    
    T = np.zeros(args.epochs+1)
    for epoch in (pbar := tqdm(range(args.epochs+1), leave=True)):

        model.train(True)

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
                    D[epoch,:] += decor_update(decorrelators)
                else:
                    D[epoch, :] += decor_loss(decorrelators)
    
            train_loss[epoch] += loss
                   
        train_loss[epoch] /= batchnum

        if epoch > 0:
            T[epoch] = time() - tic

        model.train(False)
        
        desc = f'epoch {epoch:<3} train: {train_loss[epoch]:5.3} decor: {np.mean(D[epoch]):5.3}'

        if test_loader is not None:
            for batchnum, batch in enumerate(test_loader):
                input = batch[0].to(device)
                target = batch[1].to(device)
                test_loss[epoch] += lossfun(model(input), target)
            test_loss[epoch] /= batchnum
            desc += f' test: {test_loss[epoch]:5.3}'

        pbar.set_description(desc)

        # pbar.set_description(f'epoch {epoch:<3}\ttime:{T[epoch]:.3f} s\ttrain loss: {train_loss[epoch]:<3}\tdecorrelation loss: {np.mean(D[epoch]):3f}\ttest loss: {test_loss:3f}')

    return model, train_loss, test_loss, D, T
