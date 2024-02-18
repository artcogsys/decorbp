import torch
from absl import app, flags
import tqdm
from tasks import get_task
from model import get_model
# from torchvision.utils import save_image
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from torchvision.transforms.functional import InterpolationMode

FLAGS = flags.FLAGS
# flags.DEFINE_bool('train', False, help='train from scratch')
# flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_integer('epochs', 6, help='number of epochs')
flags.DEFINE_integer('batch_size', 128, help='batch size')
# Data
flags.DEFINE_string('task', 'MNIST', help='task to train on')
flags.DEFINE_integer('input_channels', 1, help='number of input channels')
flags.DEFINE_integer('image_size', 28, help='image size assuming <size x size> image')
flags.DEFINE_integer('num_workers', 0, help='number of workers for data loading (default 0)')
# Model
flags.DEFINE_string('model', 'MLP', help='model to train')

def train_loop(epochs, model, lossfun, train_loader, optimizer,device):

    L = torch.zeros(epochs+1)
    with tqdm.trange(0, epochs+1, desc="epoch", unit="epoch") as pbar:
        for epoch in pbar:

            for step, batch in enumerate(train_loader):
            
                optimizer.zero_grad()

                batch_input = batch[0].to(device)
                batch_target = batch[1].to(device)

                loss = lossfun(model(batch_input), batch_target)

                L[epoch] += loss.item()

                if epoch > 0:
                    loss.backward()
                    optimizer.step()

            pbar.set_postfix(loss='%.3f' % L[epoch])

            C = 0
            for step, batch in enumerate(train_loader):
                C += model.input_correlation(batch_input)
            C /= step
            print(f'C: {C}')

    return L

def train():
    """
    Train using BP and DBP
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(FLAGS.model, FLAGS.input_channels, FLAGS.image_size).to(device)

    train_loader, lossfun = get_task(FLAGS.task, FLAGS.batch_size, FLAGS.num_workers)
    lossfun = lossfun.to(device)    

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
        
    loss = train_loop(FLAGS.epochs, model, lossfun, train_loader, optimizer, device)

def main(argv):
    train()
    
if __name__ == '__main__':
    app.run(main)
