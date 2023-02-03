import torch
from torch.autograd import Function
import numpy as np
import math
import torch.linalg as linalg
from torch.nn.functional import adaptive_avg_pool2d
from inception import InceptionV3
from sqrtm import sqrtm

def calculate_frechet_distance(X, mu_Y, sigma_Y):
    # the linear algebra ops will need some extra precision -> convert to double
    X = X.transpose(0, 1).double()  # [n, b]
    mu_X = torch.mean(X, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = sqrtm(sigma_Y)
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = sigma_Y

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()

def get_activations(x, model, batch_size=10, dims=2048, device='cuda', num_workers=1):
    model.eval()

    pred = model(x)[0]

    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    return pred.squeeze()

def fid(x, m2, s2, batch_size=10, device='cuda', dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)
    x = get_activations(x, model, batch_size=batch_size, device=device, dims=dims)
    return calculate_frechet_distance(x, m2, s2)