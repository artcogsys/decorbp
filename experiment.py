import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from models import *
import numpy as np

def get_experiment(args, device):

    if args.experiment == 'MNIST':

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
            ])

        train_data = MNIST(root=args.data_path, train=True, download=True, transform=transform)
        test_data = MNIST(root=args.data_path, train=True, download=True, transform=transform)

        in_features = np.prod(train_data.data.shape[1:])

        if args.train_samples != -1:
            train_data = Subset(train_data, np.random.permutation(len(train_data.data))[:args.train_samples])

        if args.test_samples != -1:
            test_data = Subset(test_data, np.random.permutation(len(test_data.data))[:args.test_samples])

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

        model = MLP(in_features).to(device)

        lossfun = torch.nn.CrossEntropyLoss().to(device)

        return model, lossfun, train_loader, test_loader



