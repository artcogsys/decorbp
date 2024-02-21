"""
Random Correlation matrix using the algorithm in LKJ 2009 (vine method based on a C-vine)
https://www.sciencedirect.com/science/article/pii/S0047259X09000876
Created on Wed Aug  2 09:09:02 2017
@author: junpenglao
"""
import torch
import numpy as np
from scipy import stats

def lkj_random(n, eta):
    """Create an n x n correlation matrix with correlation controlled by eta
    """
    beta0 = eta - 1 + n/2
    shape = n * (n-1) // 2
    triu_ind = np.triu_indices(n, 1)
    beta = np.array([beta0 - k/2 for k in triu_ind[0]])
    # partial correlations sampled from beta dist.
    P = np.ones((n, n))
    P[triu_ind] = stats.beta.rvs(a=beta, b=beta, size=(shape,)).T
    # scale partial correlation matrix to [-1, 1]
    P = (P-.5)*2
    
    for k, i in zip(triu_ind[0], triu_ind[1]):
        p = P[k, i]
        for l in range(k-1, -1, -1):  # convert partial correlation to raw correlation
            p = p * np.sqrt((1 - P[l, i]**2) *
                            (1 - P[l, k]**2)) + P[l, i] * P[l, k]
        P[k, i] = p
        P[i, k] = p

    return torch.Tensor(P)