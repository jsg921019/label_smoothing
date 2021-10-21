import os
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import AlexNet
from torchvision.datasets import CIFAR10

parser = argparse.ArgumentParser(description='Plot distribution')
parser.add_argument('data_path', type=str)
parser.add_argument('weight_path', type=str)
parser.add_argument('--img_name', type=str, default='output')
parser.add_argument('--classes', type=int, nargs=3, default=[0, 1, 2])
args = parser.parse_args()
print(args)

class Projector:

    def __init__(self, model, classes):

        self.model = copy.deepcopy(model)
        self.last_layer = list(self.model.modules())[-1]
        self.model.classifier = torch.nn.Sequential(*list(self.model.classifier)[:-1])
        self.classes = classes

        weight = self.last_layer.weight.detach().cpu().numpy()
        bias = self.last_layer.bias.detach().cpu().numpy()
        self.template = np.concatenate([weight, bias.reshape(-1, 1)], axis=-1)

        self.r0 = self.template[classes].mean(axis=0)

        basis = self.template[classes[1:]] - self.r0
        self.orthonormal_basis, _ = np.linalg.qr(basis.T)
    
    def plot(self, dataloader, n_data = 300):
        
        device = next(self.model.parameters()).device
        
        cnt = {class_:0 for class_ in self.classes}
        projection = {class_:[] for class_ in self.classes}
        
        self.model.eval()
        for imgs, label in dataloader:
            imgs, label = imgs.to(device), label.to(device)
            with torch.no_grad():
                outputs = self.model(imgs)
            for class_ in self.classes:
                if cnt[class_] < n_data:
                    class_embeddings = outputs[label == class_].cpu().numpy()
                    class_embeddings = np.concatenate([class_embeddings, np.ones((len(class_embeddings), 1))], axis=-1)
                    if len(class_embeddings):
                        cnt[class_] += len(class_embeddings)
                        projection[class_].append((class_embeddings - self.r0) @ self.orthonormal_basis)
            if all(cnt[class_] >= n_data for class_ in self.classes):
                #self.model.classifier = torch.nn.Sequential(*list(self.model.classifier), self.last_layer)
                return {class_:np.concatenate(projection[class_], axis=0) for class_ in self.classes}

        raise ValueError(f'Not enough datas (must include at least {n_data} datas)')                
            

transform = transforms.Compose([transforms.Resize(227),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

batch_size = 128

trainset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
testset = CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = AlexNet().cuda()
model.classifier[6] = torch.nn.Linear(4096, 10).cuda()
model.load_state_dict(torch.load(args.weight_path))

p = Projector(model, args.classes)

for loader, title in zip([trainloader, testloader], ['Training', 'Validation']):
    ret = p.plot(loader)

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw={'aspect':1})

    for c in [0,1,2]:
        ax.scatter(ret[c][:,1], ret[c][:,0], s=5, alpha=0.8)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    lim = max(abs(xmin), abs(ymin), abs(xmax), abs(ymax))
    ax.set_xlim(xmin=-lim, xmax=lim)
    ax.set_ylim(ymin=-lim, ymax=lim)
    ax.set_title(title + (' w/ LS' if 'smooth' in args.weight_path else ' w/o LS'))
    plt.savefig(args.img_name + '_' + title + '.png')