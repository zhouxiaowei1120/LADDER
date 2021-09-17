'''
@Author: kuangliu
@Date: missing
@LastEditors: Dave Zhou
@LastEditTime: 2019-05-28 20:44:40
@Description: download from https://github.com/kuangliu/pytorch-cifar
Train CIFAR10 with PyTorch.
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from collections import OrderedDict
import time
import sys
from PIL import Image
#from generate_latent.wideresnet import WideResNet

# import multiprocessing
# multiprocessing.set_start_method('spawn', True)

import random
import numpy as np
# Reference: https://blog.csdn.net/hyk_1996/article/details/84307108 

class defense_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, eps, img_num, fgsm=False, transform=None, dataAug=True):
        self.filenames = []
        self.ori_filenames = []
        self.fgsm = fgsm
        img_filter = ['png','jpg']
        label_list = [0,1,2,3,4,5,6,7,8,9]
        label_arr = np.zeros(len(label_list))
        
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                self.filenames.append(os.path.join(root, filename))
                                            
        self.transform = transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename)
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(np.array(img))
        if self.fgsm:
            label = int(filename.split('/')[-2][0])
        else:
            label = int(filename.split('/')[-3][0])

        return img, label
    
    def __len__(self):
        return len(self.filenames)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'VGG8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512],
}

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        return input_tensor.view(*self.shape)


class svhnVGG(nn.Module):
    def __init__(self, vgg_name):
        super(svhnVGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def classify(self, feature):
        out = feature.view(feature.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [View(-1, 512 * 2 * 2),
                    nn.Linear(512 * 2 * 2, 4096),
                    nn.ReLU(True),
                    nn.Dropout()]
        return nn.Sequential(*layers)

def evalu(x, gpu_ids, filename_model, g, device, wholemodel=True):
    if len(gpu_ids) == 0:
        # load weights to cpu
        state_dict = torch.load(filename_model, map_location=lambda storage, loc: storage)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.','') # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict
    elif len(gpu_ids) == 1:
        state_dict = torch.load(filename_model)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.','') # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict
    else:
        g = torch.nn.DataParallel(g,device_ids=gpu_ids)
        state_dict = torch.load(filename_model)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                name = ''.join(['module.',k]) # add `module.`
                new_state_dict[name] = v
        if new_state_dict:
            state_dict = new_state_dict
    g.load_state_dict(state_dict)
    if len(gpu_ids) > 1:
        g = g.module
    g.to(device)
    g.eval()
    if wholemodel:
        output = g(x)
    else:
        output = g.classify(x)
    return output

if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    
#mylogger('./cifar_train.log')
    parser = argparse.ArgumentParser(description='PyTorch SVHN Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--percent', default=0.3, type=float, help='percent of dataset for training classifier')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--dataAug', action='store_true', default=False, help='data augmenration training')
    parser.add_argument('--aug_dir',type=str, default= '.experiments/gsn_hf/SVHN/svhn_4096_norm1_GAN/eps_diff/', help='dir of adversarail samples')
    parser.add_argument('--exp_dir',type=str, default= './generate_latent/svhn/', help='dir of exp')
    parser.add_argument('--model',type=str, default= 'VGG', help='model name')
    parser.add_argument('--batch',type=int, default= 1280, help='batch size')
    args = parser.parse_args()

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if not args.dataAug:
        trainset = torchvision.datasets.SVHN(root='../datasets/SVHN/', split='train', download=True, transform=transform_train)
        indices = torch.randperm(len(trainset)).tolist()[0:int(args.percent*len(trainset))]
        trainset = torch.utils.data.Subset(trainset, indices)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=0)
    else:
        trainset = defense_dataset(args.aug_dir,
                                   -1,
                                   0,
                                   False,
                                   transform=transform_train, dataAug=args.dataAug)
        print('The number of training images are {}.'.format(trainset.__len__()))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    testset = torchvision.datasets.SVHN(root='../datasets/SVHN/', split='test', download=True, transform=transform_test)
#   if args.dataAug:
#       indices = range(len(testset))[-7325:]
#       testset = torch.utils.data.Subset(testset, indices)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=8)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # Model
    print('==> Building model..')
    print(args.model)
    if args.model == 'VGG':
       net = svhnVGG('VGG8')
    elif args.model == 'WideRes':
       net = WideResNet(28,10,10,0.4)
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet() 
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx%80 == 0:
                print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 100 == 0:
                    print(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './ckpt.t7')
            torch.save(state['net'], os.path.join(args.exp_dir,'./svhn_cnn_'+str(args.percent)+'.pt'))
            best_acc = acc
        print('Accuray of current epoch {}: {}; best accuracy till now:{}.'.format(epoch, acc, best_acc))

    for epoch in range(start_epoch, start_epoch+150):
        train(epoch)
        print('-------------------------------------------------')
        test(epoch)
    print('Finished')
