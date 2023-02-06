import argparse
import torch
from torchvision import datasets, transforms
from train import Trainer

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # General Training
    parser.add_argument('--niter', type=int, default=20000, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--num_ims', type=int, default=10)
    parser.add_argument('--cifar', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--init_ims', type=bool, default=False)
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--mmd', type=bool, default=False)
    args = parser.parse_args()

    train_kwargs = {'batch_size': 500, 'shuffle':True}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
        ])
    
    dataset1 = datasets.CIFAR10('../data/', train=True, download=True,
                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

    trainer = Trainer(args, train_loader)
    trainer.train()
    

if __name__ == '__main__':
    main()