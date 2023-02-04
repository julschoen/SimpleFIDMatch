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
import torch.utils.data as data_utils

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

    dataset = data_utils.TensorDataset(x)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((x.shape[0], dims))

    start_idx = 0

    for i, batch in enumerate(dataloader):
        batch = batch[0].to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return torch.from_numpy(pred_arr).squeeze()

def fid(x, model, batch_size=10, device='cuda', dims=2048):
    x = get_activations(x, model, batch_size=batch_size, device=device, dims=dims)
    print('Got Act', flush=True)
    return calculate_frechet_distance(x)


device='cuda'

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
        print(f'Class {i}', flush=True)
        c = final_cl[i]
        c = (c+1)/2
        m, e, c = fid(c, model, batch_size=500, device=device)
        torch.save(m.cpu(), f'm_{i}.pt')
        torch.save(e.cpu(), f'e_{i}.pt')
        torch.save(c.cpu(), f'c_{i}.pt')

    print('Full', flush=True)
    final_cl = final_cl.reshape(-1,3,32,32)
    m, e, c = fid(final_cl, model, batch_size=500, device=device)
    torch.save(m.cpu(), f'm_full.pt')
    torch.save(e.cpu(), f'e_full.pt')
    torch.save(c.cpu(), f'c_full.pt')
