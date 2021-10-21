import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import alexnet
from label_smoothing_loss import LabelSmoothingCrossEntropy

parser = argparse.ArgumentParser(description='Train Alexnet on CIFAR10')
parser.add_argument('data_path', type=str)
parser.add_argument('weight_name', type=str)
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--label_smooth', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--step', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.1)
args = parser.parse_args()
print(args)

os.makedirs(args.save_path, exist_ok=True)

transform = transforms.Compose([transforms.Resize(227),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
testset = CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

model = alexnet(pretrained=False)
model.classifier[6] = nn.Linear(4096, 10)
model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
criterion = LabelSmoothingCrossEntropy(args.label_smooth)
scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

patience_cnt = 0
best_score = 0

for epoch in range(1, args.epochs+1):
    
    # Training Phase

    running_loss = 0.0
    running_correct = 0

    model.train()
    
    for inputs, labels in tqdm(trainloader):
        
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(outputs, dim=-1)
            running_correct += torch.sum(pred == labels).item()
            running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss/len(trainset)
    epoch_acc = running_correct/len(trainset)

    if scheduler is not None:
        scheduler.step()
    
    # Validation Phase
    
    test_running_loss = 0
    test_running_correct = 0

    model.eval()
    
    for inputs, labels in tqdm(testloader):
        
        inputs, labels = inputs.cuda(), labels.cuda()
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            pred = torch.argmax(outputs, dim=-1)
            test_running_correct += torch.sum(pred == labels).item()
            test_running_loss += loss.item() * inputs.size(0)
    
    test_epoch_loss = test_running_loss/len(testset)
    test_epoch_acc = test_running_correct/len(testset)

    print(f'[Epoch {epoch}]')
    print(f'  Train : loss = {epoch_loss:.4f}, acc = {epoch_acc:.4f}')
    print(f'  Valid : loss = {test_epoch_loss:.4f}, acc = {test_epoch_acc:.4f}')
    
    if test_epoch_acc > best_score:
        torch.save(model.state_dict(), os.path.join(args.save_path, args.weight_name))
        best_score = test_epoch_acc
        patience_cnt = 0
    else:
        if patience_cnt == args.patience:
            break
        patience_cnt += 1

print('Finished Training')