import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, VOCSegmentation
from typing import Any, Callable, List, Optional, Tuple

class Autoencoder():

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index: int) -> Any:
        img, target = self.dataset[index]
        return img, img
    
    def __len__(self) -> int:
        return len(self.dataset)

def get_task(task, batch_size, num_workers):
    """
    A task defines a dataloader and a loss function
    """

    if task == "CIFAR10_AUTOENCODER":
        # we use CIFAR10 as our default since it is complex enough yet not too complex

        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        dataset = Autoencoder(CIFAR10(root='~/Data', train=True, download=True, transform=transform))
        
        lossfun = torch.nn.MSELoss()
    
    else:
        raise ValueError(f'Unrecognized task {task}')
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    
    return train_loader, lossfun



# PASCAL DATASET FOR SEGMENTATION
# def replace_tensor_value_(tensor, a, b):
#         tensor[tensor == a] = b
#         return tensor

#     imagenet_mean = [0.485, 0.456, 0.406]  # mean of the imagenet dataset for normalizing
#     imagenet_std = [0.229, 0.224, 0.225]  # std of the imagenet dataset for normalizing

#     input_resize = transforms.Resize((FLAGS.image_size, FLAGS.image_size))
#     input_transform = transforms.Compose(
#         [
#             input_resize,
#             transforms.ToTensor(),
#             transforms.Normalize(imagenet_mean, imagenet_std),
#         ]
#     )

#     target_resize = transforms.Resize((FLAGS.image_size, FLAGS.image_size), interpolation=InterpolationMode.NEAREST)
#     target_transform = transforms.Compose(
#         [
#             target_resize,
#             transforms.PILToTensor(),
#             transforms.Lambda(lambda x: replace_tensor_value_(x.squeeze(0).long(), 255, 21)),
#         ]
#     )

#     # Creating the dataset
#     train_dataset = VOCSegmentation(
#         root='~/Data', 
#         year='2007',
#         download=False, # no download
#         image_set='train',
#         transform=input_transform,
#         target_transform=target_transform,
#     )
#     valid_dataset = VOCSegmentation(
#         root='~/Data', 
#         year='2007',
#         download=False,
#         image_set='val',
#         transform=input_transform,
#         target_transform=target_transform,
#     )
#     test_dataset = VOCSegmentation(
#         root='~/Data', 
#         year='2007',
#         download=False,
#         image_set='test',
#         transform=input_transform,
#         target_transform=target_transform,
#     )

#     # Creating the dataloader
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)
#     valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers)
