#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Dave Zhou
@LastEditors: Dave Zhou
@Description: This is used for classifying attribute(smiling) of celebA using a binary classifier based on latent features from VGG11 with batch normalization.
@Date: 2019-03-08 16:34:48
@LastEditTime: 2019-04-04 15:15:00
'''
import torch.nn as nn
from torchvision import models, datasets, transforms
import torch.utils.data as data
import torch
import argparse
import os
from PIL import Image
import numpy as np
import torch.optim as optim
import logging
import sys
sys.path.append('../')
from myutils import mylogger, loadweights

def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    """ VGG11_bn 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    """
    if model_name == 'vgg11_bn':
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
    else:
        print('Unknown type of network: {}'.format(model_name))
        exit(0)
    # set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )
    input_size = 128

    return model_ft, input_size

class celebANet(nn.Module):
    def __init__(self, model_name, num_classes, feature_extract=False, use_pretrained=True):
        super(celebANet, self).__init__()
        self.model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)
    
    def forward(self, x):
        output = self.model(x)
        return output

    def classify(self, features):
        x = features
        for i in range(6):
            x = self.model.classifier[i+1](x)
        output = x
        return output


class celebA_dataset(data.Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.filenames = []
        self.train = train
        if train == True:
            data_dir = os.path.join(data_dir, 'train')
        else:
            data_dir = os.path.join(data_dir, 'test')
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename[-4:] not in ['.png','.jpg']:
                    continue
                self.filenames.append(os.path.join(root, filename))
        self.transform = transform
        with open('./datasets/celebA/labels/list_attr_celeba.txt') as labels:
            self.labelsList = labels.readlines()
        self.labelsList = self.labelsList[2:]

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename)
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(np.array(img))

        idx = int(filename.split('/')[-1].split('.')[0])
        labelLine = self.labelsList[idx-1]
        labelLine = labelLine.rstrip('\n')
        labelLine = labelLine.split()
        label = int(labelLine[32]) # 32 is used for smile and non-smile; 16 is used for eyeglasses and non-eyeglasses
        if label == -1:
            label = 0

        return img, label
    
    def __len__(self):
        return len(self.filenames)

def train(args, model, device, train_loader, optimizer, epoch):
    logger = logging.getLogger('mylogger')
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = torch.unsqueeze(target, 1).type(torch.float32)        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    logger = logging.getLogger('mylogger')
    model.eval()
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = torch.unsqueeze(target, 1).type(torch.float32)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item() # sum up batch loss
            pred = ((output >= 0.5).type(torch.float32)).to(device)  # get the label
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct/len(test_loader.dataset)

def evalu(x, gpu_ids, filename_model, g, device, wholemodel=True):
    g, state_dict = loadweights(g, filename_model, gpu_ids)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CelebA Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 200)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model_name', type=str, default='vgg11_bn',
                        help='Name of model for classifying celebA')
    parser.add_argument('--num_classes', type=int, default=1, metavar='N',
                        help='Number of classes. 1 for smiling or not smile.') 
    parser.add_argument('--res_dir', type=str, default='generate_latent/', help='path to save trained model')

    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    mylogger(args.res_dir+'/celebA_classify.log')
    logger = logging.getLogger('mylogger')

    logger.info(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        celebA_dataset('./datasets/celebA_post/', train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        celebA_dataset('./datasets/celebA_post/', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = celebANet(args.model_name, args.num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        curr_acc = test(args, model, device, test_loader)
        torch.save(model.state_dict(),args.res_dir+"/celebA_cnn_latest.pt")
        if curr_acc > best_acc:
           best_acc = curr_acc
        if (args.save_model):
           torch.save(model.state_dict(),args.res_dir+"/celebA_cnn.pt")
