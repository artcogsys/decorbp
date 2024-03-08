import torch
import torch.nn as nn
import torch
import numpy as np
import argparse
from experiments import get_experiment
from decorrelation.utils import decor_train, bp_train
import time

def parse_arguments():

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--train_samples', default=-1, type=int, help="number of train samples (-1 = all)")
    parser.add_argument('--test_samples', default=-1, type=int, help="number of test samples (-1 = all)")
    parser.add_argument('--experiment', default='MNIST_MLP', type=str)
    parser.add_argument('--data_path', default='~/Data', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--decor_lr', default=1e-4, type=float)
    parser.add_argument('--kappa', default=1e-3, type=float, help="strenght of unit variance constraint [0-1]") 
    parser.add_argument('--downsample_perc', default=1.0, type=float, help="downsampling for covariance computation") 
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for i in range(2):

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        model, lossfun, train_loader, test_loader = get_experiment(args, device)

        if i == 0:
            print('Decorrelated BP:')
            tic=time.time()
            res = decor_train(args, model, lossfun, train_loader, device)
            print(f'time: {time.time() - tic:.2} s')
        else:
            print('Regular BP:')
            tic=time.time()
            res = bp_train(args, model, lossfun, train_loader, device)
            print(f'time: {time.time() - tic:.2} s')

    