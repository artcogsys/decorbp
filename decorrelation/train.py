import torch
import numpy as np
from decorrelation.decorrelation import decor_parameters, decor_modules, decor_update #, covariance
from time import time

def decor_train(args, model, lossfun, train_loader, device):
    """Train using decorrelated backpropagation. Can also be used to run regular bp with args.decor_lr = 0.0. But for fair comparison see bp_train.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    decorrelators = decor_modules(model)
    decor_optimizer = torch.optim.SGD(decor_parameters(model), lr=args.decor_lr)
    
    L = np.zeros(args.epochs+1) # loss
    D = np.zeros(args.epochs+1) # decorrelation loss
    T = np.zeros(args.epochs+1) # time per train epoch
    for epoch in range(args.epochs+1):

        tic = time()
        for batchnum, batch in enumerate(train_loader):
        
            optimizer.zero_grad()
            decor_optimizer.zero_grad()

            input = batch[0].to(device)
            target = batch[1].to(device)

            loss = lossfun(model(input), target)

            if epoch > 0:
                loss.backward()
                optimizer.step()

            decor_loss = decor_update(decorrelators)
            if epoch > 0:
                decor_optimizer.step()

            D[epoch] += decor_loss
            L[epoch] += loss
                     
        L[epoch] /= batchnum

        if epoch > 0:
            T[epoch] = time() - tic

        print(f'epoch {epoch:<3}\ttime:{T[epoch]:.3f} s\tbp loss: {L[epoch]:3f}\tdecorrelation loss: {D[epoch]:3f}')

    return model, L, D, T



def bp_train(args, model, lossfun, train_loader, device):
    """Train using backpropagation only. A fair comparison would require optimal settings for learning rate and batch size as well as 
    running on models that don't incorporate the decorrelation layers.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    L = np.zeros(args.epochs+1) # loss
    T = np.zeros(args.epochs+1) # time per train epoch
    for epoch in range(args.epochs+1):

        tic = time()
        for batchnum, batch in enumerate(train_loader):
        
            optimizer.zero_grad()

            input = batch[0].to(device)
            target = batch[1].to(device)

            loss = lossfun(model(input), target)

            if epoch > 0:
                loss.backward()
                optimizer.step()

            L[epoch] += loss.item()
                        
        L[epoch] /= batchnum
        
        if epoch > 0:
            T[epoch] = time() - T[epoch-1]

        if epoch > 0:
            T[epoch] = time() - tic

        print(f'epoch {epoch:<3}\ttime:{T[epoch]:.3f} s\tbp loss: {L[epoch]:3f}')
    
    return model, L, T
