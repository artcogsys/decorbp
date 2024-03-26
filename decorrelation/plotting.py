import matplotlib.pyplot as plt
import matplotlib
from pylearn.decorrelation.decorrelation import lower_triangular, decor_modules
import numpy as np
import torch

def plot_loss(ax, *losses):
    for L in losses:
        ax.plot(L)
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.set_title('loss')

def plot_decorrelation_matrix(ax, R):
    min = np.max([np.min(np.abs(R.numpy().flatten())), 1e-10])
    max = np.max(np.abs(R.numpy().flatten()))
    im = ax.imshow(np.abs(R), cmap=plt.get_cmap('hot'), interpolation='nearest', norm=matplotlib.colors.LogNorm(vmin=min, vmax=max))
    ax.set_title('|decorrelation weights|')
    plt.colorbar(im, fraction=0.046, pad=0.04)

def covariance_histogram(ax, *Cs, labels=None):
    """Plot covariance of one or more covariance matrices
    """

    covs = [lower_triangular(C, offset=-1) for C in Cs]
    ax.hist(covs, bins=30, density=True)
    ax.set_xlabel('$x_i x_j$')
    ax.set_title('covariance')
    if labels is not None:
        ax.legend(labels)

def variance_histogram(ax, *Cs, labels = None):
    """Plot variance of one or more covariance matrices
    """

    covs = [torch.diagonal(C) for C in Cs]
    ax.hist(covs, bins=30, density=True)
    ax.set_xlabel('$x_i^2$')
    ax.set_title('variance')
    if labels is not None:
        ax.legend(labels)

def plot_covariance_matrix(ax, C):
    im = ax.imshow(C, cmap=plt.get_cmap('hot'), interpolation='nearest')
    ax.set_title('$<x_i x_j>$')
    ax.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)

def plot_decorrelation_results(model, L, A1, A2):

    fig, ax = plt.subplots(2,3, figsize=(14, 9))
    plot_loss(ax[0,0], L)
    covariance_histogram(ax[0,1], A1, A2)
    variance_histogram(ax[0,2], A1, A2, labels=['correlated', 'decorrelated'])
    plot_decorrelation_matrix(ax[1,0], model.weight)
    plot_covariance_matrix(ax[1,1], A1)
    plot_covariance_matrix(ax[1,2], A2)


def plot_correlations(init_model, model, dataloader, device):
    """Plot correlations for an untrained and trained model using the first batch of a dataloader (ensuring the same transformations)
    """
    init_modules = decor_modules(init_model)
    modules = decor_modules(model)
    num_mods = len(list(modules))

    fig, ax = plt.subplots(2, num_mods, figsize=(4*num_mods, 9))
    for batch in dataloader:
        init_model.forward(batch[0].to(device))
        model.forward(batch[0].to(device))
        for i, (imod, mod) in enumerate(zip(init_modules, modules)):
        
            state = imod.decor_state            
            Ci = (state.T @ state) / len(state)
            Ci = Ci.detach()

            state = mod.decor_state            
            C = (state.T @ state) / len(state)
            C = C.detach()

            covariance_histogram(ax[0,i] if num_mods > 1 else ax[0], Ci, C)
            variance_histogram(ax[1,i]  if num_mods > 1 else ax[1], Ci, C)

            print(f'layer {i+1}:\n')
            print(f'mean covariance before decorrelation: {torch.mean(lower_triangular(Ci, offset=-1)):.2f}')
            print(f'mean variance before decorrelation: {torch.mean(torch.diagonal(Ci)):.2f}\n')
            print(f'mean covariance after decorrelation: {torch.mean(lower_triangular(C, offset=-1)):.2f}')
            print(f'mean variance after decorrelation: {torch.mean(torch.diagonal(C)):.2f}\n')

        break
    
    ax[0,-1].legend(['correlated', 'decorrelated']) if num_mods > 1 else ax[-1].legend(['correlated', 'decorrelated'])
