import torch
import torch.nn as nn
import torch
import numpy as np
import argparse
from experiment import get_experiment
from decorrelation import decorrelation_parameters, decorrelation_modules, decorrelation_update, mean_correlation, lower_triangular_correlation, DecorrelationPatch2d, DecorrelationFC
from utils import *

def parse_arguments():

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decor', default=1e-3, type=float, help="learning rate for decorrelation update")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--train_samples', default=-1, type=int, help="number of train samples (-1 = all)")
    parser.add_argument('--test_samples', default=-1, type=int, help="number of test samples (-1 = all)")
    parser.add_argument('--experiment', default='MNIST', type=str)
    parser.add_argument('--data_path', default='~/Data', type=str)
    return parser.parse_args()

def train_loop(args, model, lossfun, train_loader):

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

if __name__ == '__main__':

    args = parse_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for i in range(2):

        if i == 0:
            print('Decorrelated BP:')
        else:
            print('Regular BP:')
            args.lr_decor = 0.0

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        model, lossfun, train_loader, test_loader = get_experiment(args, device)

        model, L, C = train_loop(args, model, lossfun, train_loader)

    
