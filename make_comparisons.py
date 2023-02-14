import argparse
import torch
from torchvision import datasets, transforms
import os


comp_dir = '../comparison_synth'

def save(file_name, data):
        file_name = os.path.join(comp_dir, file_name)
        torch.save(data.cpu(), file_name)

def make_random():
    train_kwargs = {'batch_size': 1000, 'shuffle':True}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    dataset1 = datasets.CIFAR10('../data/', train=True, download=True,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

    if not os.path.isdir(comp_dir):
        os.mkdir(comp_dir)

    data_all = []
    label_all = []

    for i, (x,y) in enumerate(train_loader):
        for c in range(10):
            data = x[y == c]
            perm = torch.randperm(data.shape[0])[:100]
            data, label = data[perm], torch.ones(100)*c

            data_all.append(data)
            label_all.append(label)

        data = torch.concat(data_all)
        label = torch.concat(label_all)
        save(f'rand_x_{i}.pt', data)
        save(f'rand_y_{i}.pt', label)

        if i == 3:
            break

if __name__ == '__main__':
    make_random()
