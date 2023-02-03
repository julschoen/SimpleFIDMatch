import argparse
import torch
from torchvision import datasets, transforms
import pytorch_fid_wrapper as pfw

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
full = final_cl.reshape(-1,3,32,32)
print(full.shape, final_cl.shape)

pfw.set_config(batch_size=500)
for i in range(10):
    print(final_cl[i].shape)
    c = final_cl[i]
    c = (c+1)/2
    real_m, real_s = pfw.get_stats(c)
    torch.save(real_m, f'real_m_{i}.pt')
    torch.save(real_s, f'real_s_{i}.pt')


real_m, real_s = pfw.get_stats(full)
torch.save(real_m, f'real_m_full.pt')
torch.save(real_s, f'real_s_full.pt')
