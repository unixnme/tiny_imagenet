import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=9e-1)
parser.add_argument('--train_dir', type=str, default='dataset/tiny-imagenet-200/train')
parser.add_argument('--val_dir', type=str, default='dataset/tiny-imagenet-200/val_out')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--print_freq', type=int, default=1000)

args = parser.parse_args()

model = models.resnet18().to(args.device)
optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_set = datasets.ImageFolder(args.train_dir,
                                transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))

train_loader = DataLoader(train_set,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.workers,
                          pin_memory=True)

criterion = nn.CrossEntropyLoss().to(args.device)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_epoch(epoch:int, model:nn.Module, optim:torch.optim.Optimizer, criterion:nn.Module, loader:DataLoader):
    model.train()
    for i, (x,y) in enumerate(tqdm(loader)):
        x,y = x.to(args.device), y.to(args.device)
        pred = model(x)
        loss = criterion(pred, y)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(pred, y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))

        # compute gradient and do SGD step
        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))

for epoch in range(args.epochs):
    train_epoch(epoch, model, optim, criterion, train_loader)
