import argparse
import torch
from torchvision import datasets, transforms
import pytorch_fid_wrapper as pfw

import torch
from torch.autograd import Function
import numpy as np
import math
import torch.linalg as linalg
from torch.nn.functional import adaptive_avg_pool2d
from inception import InceptionV3

def calculate_frechet_distance(X):
    X = X.transpose(0, 1).double()  # [n, b]
    mu_X = torch.mean(X, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    return mu_X, E_X, cov_X

def get_activations(x, model, batch_size=10, dims=2048, device='cuda', num_workers=1):
    model.eval()

    pred = model(x)[0]

    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    return pred.squeeze()

def fid(x, model, batch_size=10, device='cuda', dims=2048):
    x = get_activations(x, model, batch_size=batch_size, device=device, dims=dims)
    print('Got Act', flush=True)
    return calculate_frechet_distance(x, m2, s2)

train_kwargs = {'batch_size': 10000, 'shuffle':True}

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
    ])


dataset1 = datasets.CIFAR10('../data/', train=True, download=True,
                   transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

classes = [[] for _ in range(10)]

for _, (x,y) in enumerate(train_loader):
    for i, xi in enumerate(x):
        classes[y[i]].append(xi.reshape(1,3,32,32))

final_cl = []

for cl in classes:
    final_cl.append(torch.cat(cl))


final_cl = torch.stack(final_cl)

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx]).eval().to(device)

with torch.no_grad():
    for i in range(10):
        print(f'Class {i}' flush=True)
        c = final_cl[i]
        c = (c+1)/2
        m, e, c = fid(c.to('cuda'), model, batch_size=10)
        torch.save(m, f'm_{i}.pt')
        torch.save(e, f'e_{i}.pt')
        torch.save(c, f'c_{i}.pt')

    print('Full', flush=True)
    final_cl = final_cl.reshape(-1,3,32,32)
    m, e, c = fid(final_cl.to('cuda'), model, batch_size=10)
    torch.save(m, f'm_full.pt')
    torch.save(e, f'e_full.pt')
    torch.save(c, f'c_full.pt')
