import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
import numpy as np

def get_MNIST(args):

    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(1),
            transforms.Normalize((0.5), (0.5))
            ])

    if args.train_samples == 0:
        train_loader = None
    else:
        
        train_data = MNIST(root=args.data_path, train=True, download=True, transform=transform)
        # train_data.data = torch.unsqueeze(train_data.data, 1)
    
        if args.train_samples != -1:
            train_data = Subset(train_data, np.random.permutation(len(train_data.data))[:args.train_samples])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    if args.test_samples == 0:
        test_loader = None
    else:
        test_data = MNIST(root=args.data_path, train=True, download=True, transform=transform)
        # test_data.data = torch.unsqueeze(test_data.data, 1)

        if args.test_samples != -1:
            test_data = Subset(test_data, np.random.permutation(len(test_data.data))[:args.test_samples])
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    input_dim = (1, 28, 28)

    return train_loader, test_loader, input_dim
