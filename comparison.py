import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import numpy as np

class ConvNet(nn.Module):
    def __init__(self, params, num_classes=10, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling'):
        super(ConvNet, self).__init__()

        channel = 3
        im_size = (32,32)

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 100. * correct / len(test_loader.dataset)

def resize_comp(x,y, num_ims, num_classes):
    data = []
    labels = []
    for c in range(num_classes):
        xc = x[y == c]
        perm = torch.randperm(xc.shape[0])[:num_ims]
        xc, yc = xc[perm], torch.ones(num_ims)*c

        data.append(xc)
        labels.append(yc)

    data = torch.concat(data)
    labels = torch.concat(labels).long()
    return data, labels


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_ims', type=int, default=10)
    args = parser.parse_args()

    device = args.device

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5))
    ])

    test_kwargs = {'batch_size': args.test_batch_size}
    
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    res = []
    for i in range(10):
        comp_dir = '../comparison_synth'
        
        ## random
        targets = torch.load(os.path.join(comp_dir,f'rand_y_{i}.pt'))
        features = torch.load(os.path.join(comp_dir, f'rand_x_{i}.pt'))

        features, targets = resize_comp(features, targets, args.num_ims, 10)

        synth = data_utils.TensorDataset(features, targets)
        train_loader = torch.utils.data.DataLoader(synth, batch_size=args.batch_size, shuffle=True)
        for _ in range(10):
            model = ConvNet(args).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(1, 200):
                train(args, model, device, train_loader, optimizer, epoch)
            acc = test(model, device, test_loader)
            res.append(acc)

    res = np.array(res)
    print(res.mean(), res.std())



if __name__ == '__main__':
    main()