import torch
import numpy as np
from decorrelation import decorrelation_parameters, decorrelation_modules, decorrelation_update, mean_correlation

def train_loop(args, model, lossfun, train_loader, device):

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        decorrelators = decorrelation_modules(model)
        decor_optimizer = torch.optim.SGD(decorrelation_parameters(model), lr=args.lr_decor)

        L = np.zeros(args.epochs+1)
        C = np.zeros(args.epochs+1)
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
                        decorrelation_update(decorrelators)
                        decor_optimizer.step()

                L[epoch] += loss.item()
                C[epoch] += mean_correlation(decorrelators)
            
            L[epoch] /= step
            C[epoch] /= step

            print(f'epoch {epoch:<3}\tloss: {L[epoch]:3f}\tcorrelation: {C[epoch]:3f}')

        return model, L, C
