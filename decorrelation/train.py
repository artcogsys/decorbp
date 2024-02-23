import torch
import numpy as np
from decorrelation.decorrelation import decorrelation_parameters, decorrelation_modules, decorrelation_update #, covariance

def train_loop(args, model, lossfun, train_loader, device):

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        decorrelators = decorrelation_modules(model)
        decor_optimizer = torch.optim.SGD(decorrelation_parameters(model), lr=args.lr_decor)

        L = np.zeros(args.epochs+1) # loss
        D = np.zeros(args.epochs+1) # decorrelation loss
        for epoch in range(args.epochs+1):

            for step, batch in enumerate(train_loader):
            
                optimizer.zero_grad()

                input = batch[0].to(device)
                target = batch[1].to(device)

                loss = lossfun(model(input), target)

                if epoch > 0:

                    loss.backward()
                    optimizer.step()

                    if args.lr_decor > 0.0:
                        decor_loss = decorrelation_update(decorrelators)
                        decor_optimizer.step()
                        D[epoch] += decor_loss

                L[epoch] += loss.item()
                            
            L[epoch] /= step

            print(f'epoch {epoch:<3}\tloss: {L[epoch]:3f}\tdecorrelation loss: {D[epoch]:3f}')

        return model, L
