import torch
# from absl import app, flags
import tqdm
# from tasks import get_task
# from dbp.models import get_model
from dbp.decorrelation import Decorrelation, decorrelation_parameters, decorrelation_modules, decorrelation_update, input_correlation
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

# FLAGS = flags.FLAGS
# # Training
# flags.DEFINE_float('lr', 2e-4, help='target learning rate')
# flags.DEFINE_integer('epochs', 6, help='number of epochs')
# flags.DEFINE_integer('batch_size', 128, help='batch size')
# # Data
# flags.DEFINE_string('task', 'MNIST', help='task to train on')
# flags.DEFINE_integer('input_channels', 1, help='number of input channels')
# flags.DEFINE_integer('image_size', 28, help='image size assuming <size x size> image')
# flags.DEFINE_integer('num_workers', 0, help='number of workers for data loading (default 0)')
# # Model
# flags.DEFINE_string('model', 'MLP_AUTOENCODER', help='model to train')

# def train_loop(epochs, model, lossfun, train_loader, optimizer,device):

#     L = torch.zeros(epochs+1)
#     with tqdm.trange(0, epochs+1, desc="epoch", unit="epoch") as pbar:
#         for epoch in pbar:

#             for step, batch in enumerate(train_loader):
            
#                 optimizer.zero_grad()

#                 batch_input = batch[0].to(device)
#                 batch_target = batch[1].to(device)

#                 loss = lossfun(model(batch_input), batch_target)

#                 L[epoch] += loss.item()

#                 if epoch >= 0: # DEBUGGING >=
#                     loss.backward()
#                     optimizer.step()

#             pbar.set_postfix(loss='%.3f' % L[epoch])

#             # C = 0
#             # for step, batch in enumerate(train_loader):
#             #     C += model.input_correlation(batch_input)
#             # C /= step
#             # print(f'C: {C}')

#     return L

# def train():
#     """
#     Train using BP and DBP
#     """

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     train_loader, lossfun = get_task(FLAGS.task, FLAGS.batch_size, FLAGS.num_workers)
#     lossfun = lossfun.to(device)    

#     # with decorrelation
#     model = get_model(FLAGS.model, FLAGS.input_channels, FLAGS.image_size).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
#     loss1 = train_loop(FLAGS.epochs, model, lossfun, train_loader, optimizer, device)

#     # without decorrelation
#     model = get_model(FLAGS.model, FLAGS.input_channels, FLAGS.image_size).to(device)
#     model.set_decorrelation_learning_rate(0.0)
#     optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
#     loss2 = train_loop(FLAGS.epochs, model, lossfun, train_loader, optimizer, device)

#     plt.plot(loss1)
#     plt.plot(loss2)
#     plt.legend('DBP', 'BP')
#     plt.show()

# def main(argv):
#     train()

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decor', default=1e-3, type=float, help="learning rate for decorrelation update")
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--save_steps', default=100, type=int, help="print loss every save_steps mini-batches")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## DATA

    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])

    dataset = MNIST(root='~/Data', train=True, download=True, transform=transform)
    
    input_channels=1
    image_size=28

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True)

    ## MODEL DEFINITION

    class MLP(nn.Sequential):

        def __init__(self, input_dim):
            super().__init__(Decorrelation(input_dim),
                            nn.Linear(input_dim, 100),
                            nn.ReLU(),
                            Decorrelation(100),
                            nn.Linear(100, 10)
                            )

        def forward(self, x):
            return super().forward(x.view(len(x), -1))

    lossfun = torch.nn.CrossEntropyLoss().to(device)
    
    ## TRAIN USING REGULAR BP

    print('regular BP:')

    # torch.manual_seed(args.seed) # NOT YET THE SAME!
    # model = MLP(input_channels*image_size**2).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # running_loss = 0.0
    # for epoch in range(args.epochs+1):

    #     for step, batch in enumerate(train_loader):
        
    #         optimizer.zero_grad()

    #         batch_input = batch[0].to(device)
    #         batch_target = batch[1].to(device)

    #         loss = lossfun(model(batch_input), batch_target)

    #         running_loss += loss.item()
    #         if (step+1) % args.save_steps == 0:
    #             print(f'[{epoch + 1}, {step + 1:5d}] loss: {running_loss / args.save_steps:.3f}')
    #             running_loss = 0.0

    #         loss.backward()
    #         optimizer.step()
    
    ## TRAIN USING DECORRELATED BP

    print('decorrelated BP:')

    torch.manual_seed(args.seed)
    model = MLP(input_channels*image_size**2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    decorrelators = decorrelation_modules(model)
    decor_optimizer = torch.optim.SGD(decorrelation_parameters(model), lr=args.lr_decor)

    running_loss = 0.0
    for epoch in range(args.epochs+1):

        for step, batch in enumerate(train_loader):
        
            optimizer.zero_grad()

            batch_input = batch[0].to(device)
            batch_target = batch[1].to(device)

            loss = lossfun(model(batch_input), batch_target)

            loss.backward()
            optimizer.step()

            decorrelation_update(decorrelators)
            decor_optimizer.step()

            running_loss += loss.item()
            if (step+1) % args.save_steps == 0:
                print(f'[{epoch + 1}, {step + 1:5d}] loss: {running_loss / args.save_steps:.3f}')
                running_loss = 0.0

        print(f'average input correlation for one batch: {input_correlation(model, batch_input)}')



