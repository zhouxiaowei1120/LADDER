from __future__ import print_function
import argparse
import os
import PIL.Image as Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms as trans
from collections import OrderedDict
from myutils import mylogger, loadweights
import logging
import torchvision
import sys
sys.path.append('./generate_latent/')
from generate_latent.pytorch_cifar.models import ResNet18


class defense_dataset(data.Dataset):
    def __init__(self, data_dir, eps, img_num, fgsm=False, transform=None, GA=False):
        self.filenames = []
        self.ori_filenames = []
        self.fgsm = fgsm
        img_filter = ['png','jpg']
        label_list = [0,1,2,3,4,5,6,7,8,9]
        label_arr = np.zeros(len(label_list))
        
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename[-3:] in img_filter and filename.split('_')[-1] != 'ori.png':
                    if img_num == 0:
                        tmp_filename = filename.split('_')
                        if eps == -1 or float(tmp_filename[-1][:-4]) == eps:
                            self.filenames.append(os.path.join(root, filename))
                            if not GA:
                                ori_filename = '{0}_{1:06d}_ori.png'.format(tmp_filename[0], int(tmp_filename[1]))
                            else:
                                ori_filename = filename
                            self.ori_filenames.append(os.path.join(root, ori_filename))
                    else:
                        img_path = os.path.join(root,filename)
                        if self.fgsm: # True for that adversarial samples are from fgsm or jsma
                            label = int(img_path.split('/')[-2][0])
                        else:
                            label = int(img_path.split('/')[-3][0])
                        if label_arr[label] >= img_num/len(label_list): # if the number of images for one class is enough, just continue
                            continue
                        else:
                            tmp_filename = filename.split('_')
                            if eps == -1 or float(tmp_filename[-1][:-4]) == eps:
                                label_arr[label] += 1
                                self.filenames.append(os.path.join(root, filename))
                                ori_filename = '{0}_{1:06d}_ori.png'.format(tmp_filename[0], int(tmp_filename[1]))
                                self.ori_filenames.append(os.path.join(root, ori_filename))
        assert img_num == label_arr.sum()
                        
        self.transform = transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename)
        ori_filename = self.ori_filenames[index]
        ori_img = Image.open(ori_filename)
        if self.transform is not None:
            img = self.transform(img)
            ori_img = self.transform(ori_img)
        img = torch.from_numpy(np.array(img))
        ori_img = torch.from_numpy(np.array(ori_img))
        if self.fgsm:
            label = int(filename.split('/')[-2][0])
        else:
            label = int(filename.split('/')[-3][0])

        return img, label, ori_img
    
    def __len__(self):
        return len(self.filenames)

class defense_test_dataset(data.Dataset):
    def __init__(self, data_dir, eps, img_num=0, ohter_attack=False, transform=None):
        self.filenames = []
        self.attack = ohter_attack # True for data from other attack
        img_filter = ['png','jpg']
        label_list = [0,1,2,3,4,5,6,7,8,9]
        label_arr = np.zeros(len(label_list))

        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename[-3:] in img_filter and filename.split('_')[-1] != 'ori.png':
                    if img_num == 0:
                        tmp_filename = filename.split('_')
                        if eps == -1 or float(tmp_filename[-1][:-4]) == eps:
                            self.filenames.append(os.path.join(root, filename))
                    else:
                        img_path = os.path.join(root,filename)
                        if self.attack: # True for that adversarial samples are from fgsm or jsma
                            label = int(img_path.split('/')[-2][0])
                        else:
                            label = int(img_path.split('/')[-3][0])
                        if label_arr[label] >= img_num/len(label_list): # if the number of images for one class is enough, just continue
                            continue
                        else:
                            tmp_filename = filename.split('_')
                            if eps == -1 or float(tmp_filename[-1][:-4]) == eps:
                                label_arr[label] += 1
                                self.filenames.append(os.path.join(root, filename))
        assert img_num == label_arr.sum()
        self.transform = transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename)
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(np.array(img))
        if self.attack:
          label = int(filename.split('/')[-2][0])
        else:
          label = int(filename.split('/')[-3][0])          
        return img, label
    
    def __len__(self):
        return len(self.filenames)

def train(args, model, device, train_loader, optimizer, epoch):
    logger = logging.getLogger('mylogger')
    criterion = nn.CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target, ori_data) in enumerate(train_loader):
        data, target, ori_data = data.to(device), target.to(device), ori_data.to(device)
        optimizer.zero_grad()
        output = model(data)
        ori_output = model(ori_data)
        loss = 0.5 * criterion(output, target) + 0.5 * criterion(ori_output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    logger = logging.getLogger('mylogger')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def evalu(x, gpu_ids, filename_model, g, device, wholemodel=True):
    state_dict = loadweights(g, filename_model, gpu_ids)
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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--img_num', type=int, default=0, metavar='N',
                        help='number of samples to train. 0 means loading all samples (default: 1800)')
    parser.add_argument('--epoch_to_restore', type=int, default=1000, metavar='N',
                        help='number of epochs to restore to continue to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--gpu_ids',type=str, default= '', help='the ids of GPUs')
    parser.add_argument('--i',type=str, default= '', help='Saved information')
    parser.add_argument('--adversarial_samples',type=str, default= './experiments/gsn_hf/cifar_train_test_cifarnet_3_1norm_ncfl512_NormL1/eps_diff/att_svm/', help='dir of adversarail samples')
    parser.add_argument('--eps', type=float, default=0.1, help='learning rate (default: 0.01)')
    parser.add_argument('--exp_dir',type=str, default= './', help='the dir for saving the results')
    parser.add_argument('--exp_name',type=str, default= 'defence', help='the name of the experiments')
    parser.add_argument('--model4defence',type=str, default= '', help='dir of defence model weithgs')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--defence_module', action='store_true', default=False,
                        help='train or test model:False for train and test, True for test.')
    parser.add_argument('--attack_dataset', action='store_true', default=False,
                        help='when defend, dataset from other attack methods or our attack:False for our, True for other.')
    parser.add_argument('--GA', action='store_true', default=False,
                        help='when adversarial training with GA generated adversarial examples, set as true')
                        
    args = parser.parse_args()
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    if args.defence_module:
        defence_str = '_defence_'
    else:
        defence_str = '_advTrain_'

    mylogger(args.exp_dir+'cifar10'+defence_str+'-'+str(args.img_num)+'-'+str(args.eps)+args.exp_name+'.log')
    # mylogger(args.exp_dir+'cifar10'+defence_str+'.log')
    
    logger = logging.getLogger('mylogger')
    logger.info(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform_train = trans.Compose([
        trans.ToTensor(),
    ])

    transform_test = trans.Compose([
        trans.ToTensor(),
    ])
    if args.defence_module:
        test_attack_dataset = defense_test_dataset(args.adversarial_samples, args.eps, args.img_num, args.attack_dataset, transform=transform_test)
        test_attack_dataloader = torch.utils.data.DataLoader(test_attack_dataset, batch_size=args.batch_size, shuffle=True,**kwargs)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root='./datasets/cifar/', train=False, download=True, transform=transform_test),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        # attack our original model
        model = ResNet18().to(device)
        filename_model = os.path.join('./generate_latent/pytorch_cifar/checkpoint/cifar_cnn.pt')
        model, state_dict = loadweights(model, filename_model, args.gpu_ids)
        model.load_state_dict(state_dict)

        logger.info('Accuracy on adversarial samples using original model.')
        acc1=test(args, model, device, test_attack_dataloader)

        # attack our finetuned model
        filename_model = args.model4defence
        model, state_dict = loadweights(model, filename_model, args.gpu_ids)
        model.load_state_dict(state_dict)

        logger.info('Accuracy on adversarial samples using finetuned model.')
        acc2=test(args, model, device, test_attack_dataloader)
        logger.info('Accuracy on Cifar Test dataset using finetuned model.')
        acc3=test(args, model, device, test_loader)
        fw = open(args.exp_dir+'logacc.txt', 'a')
        fw.writelines('original model accuracy on adv examples: '+str(acc1)+' accuracy of '+args.model4defence+' on '+args.adversarial_samples+' is: '+str(acc2)+' accuracy of defence model on clean test set is: '+str(acc3)+'\n')
        fw.close()
    else:
        train_dataset = defense_dataset(args.adversarial_samples,
                            args.eps,
                            args.img_num,
                            args.attack_dataset,
                        transform=transform_train,GA=args.GA)
        logger.info('The number of training images are {}.'.format(train_dataset.__len__()))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root='./datasets/cifar/', train=False, download=True, transform=transform_test),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        model = ResNet18().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

        if args.epoch_to_restore > 0:
            filename_model = os.path.join('./generate_latent/pytorch_cifar/checkpoint/cifar_cnn.pt')
            model, state_dict = loadweights(model, filename_model, args.gpu_ids)
            model.load_state_dict(state_dict)
        else:
            logger.info('Initialize the cifarnet model with random weights.')

        logger.info('This is the test result before retraining cifarnet.')
        test(args, model, device, test_loader)
        
        bestacc = 0
        for epoch in range(args.epoch_to_restore+1, args.epoch_to_restore+args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            testacc = test(args, model, device, test_loader)
            if (args.save_model) and testacc >= bestacc:
                torch.save(model.state_dict(), args.exp_dir+"cifar_cnn_defence_"+args.exp_name+'_'+str(args.eps)+'_best.pt')
                bestacc = testacc

        if (args.save_model):
            torch.save(model.state_dict(), args.exp_dir+"cifar_cnn_defence_"+args.exp_name+'_'+str(args.eps)+".pt")
    

        
if __name__ == '__main__':
    if not os.path.exists('./experiments/cifar10_defence/'):
        os.makedirs('./experiments/cifar10_defence/') 
    main()
    
