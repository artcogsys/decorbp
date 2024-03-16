import torch
import numpy as np
from decorrelation.decorrelation import decor_modules, decor_update, decor_loss #, covariance
from time import time
from node_perturbation.node_perturbation import np_update

def np_train(args, model, lossfun, train_loader, test_loader=None, device=None, decorrelate=True):
    """Train using (decorrelated) node perturbation.

    NOTE: we want to integrate this with decorrelation.utils one level higher up and allow mixed BP, NP, and decorrelation training.

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
        D = np.zeros((args.epochs+1, len(decorrelators))) # decorrelation loss
    else:
        D = np.zeros(args.epochs+1) # decorrelation loss
    
    train_loss = np.zeros(args.epochs+1) # loss
    T = np.zeros(args.epochs+1) # time per train epoch
    with torch.no_grad():
        
        for epoch in range(args.epochs+1):

            tic = time()
            for batchnum, batch in enumerate(train_loader):
            
                optimizer.zero_grad() if args.lr > 0.0 else None
                            
                input = batch[0].to(device)
                target = batch[1].to(device)

                output = model(torch.concat([input, input]))

                loss = lossfun(output[:len(input)], target)
                train_loss[epoch] += loss

                delta_loss = lossfun(output[len(input):], target) - loss
                np_update(model, delta_loss)
                if epoch > 0:
                    optimizer.step()

                if decorrelate:
                    if epoch > 0:
                        D[epoch,:] += decor_update(decorrelators)
                    else:
                        D[epoch,:] += decor_loss(decorrelators)
        
                train_loss[epoch] += loss
                        
            train_loss[epoch] /= batchnum

            if epoch > 0:
                T[epoch] = time() - tic

            test_loss = 0.0
            if test_loader is not None:
                with torch.no_grad():
                    for batchnum, batch in enumerate(test_loader):
                        input = batch[0].to(device)
                        target = batch[1].to(device)
                        test_loss += lossfun(model(input), target)
                    test_loss /= batchnum

            print(f'epoch {epoch:<3}\ttime:{T[epoch]:.3f} s\tnp loss: {train_loss[epoch]:3f}\tdecorrelation loss: {np.sum(D[epoch]):3f}\ttest loss: {test_loss:3f}')

    return model, train_loss, D, T

