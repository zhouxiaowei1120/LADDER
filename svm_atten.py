#!/usr/bin/env python
# coding: utf-8
# reference URL:      https://github.com/kazuto1011 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.softmax  = nn.Softmax(dim=-1) #
        self.activate = nn.Tanh()
        if torch.cuda.is_available():
            self.softmax.cuda()
            self.conv.cuda()
            self.activate.cuda()

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X channel)
            returns :
                out : self attention value 
                attention: B X channel(channel is dim of latenr features)
        """
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
            if torch.cuda.is_available():
                x = x.cuda()
        if len(x.shape) != 4:
            x = torch.unsqueeze(torch.unsqueeze(x,dim=-1), dim=-1) # B X channel X 1 X 1
        proj_query  = self.conv(x) # B X channel X 1 X 1
        # proj_key = torch.squeeze(self.conv(x), dim=-1) # channel
        # energy =  torch.bmm(torch.squeeze(proj_query, dim=-1), proj_key.permute(0,2,1)) # transpose check
        attention = self.softmax(self.activate(torch.squeeze(torch.squeeze(proj_query, dim=-1), dim=-1))) # BX (channel) 
        # proj_value = torch.unsqueeze(self.value_conv(x), dim=2) # B X channel X 1
        attention = torch.exp(attention)
        out = torch.squeeze(torch.squeeze(x, dim=-1), dim=-1) * attention
        # out = self.gamma*out + x
        # out = torch.squeeze(out, dim = -1)
        return out,attention

class LinearSVM(nn.Module):
    """Support Vector Machine"""

    def __init__(self, input_dim):
        super(LinearSVM, self).__init__()
        self.attention_layer = Self_Attn(input_dim)
        if torch.cuda.is_available():
            self.attention_layer.cuda()
        self.fc = nn.Linear(input_dim, 1) # input_dim is 500 for mnist.

    def forward(self, x):
        x, attetion = self.attention_layer(x)
        # attetion = torch.ones([1,500])
        h = self.fc(x)
        return h, attetion
    
    def predict(self, x):
        # predict labels of input x
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
            if torch.cuda.is_available():
                x = x.cuda()
        x, attetion = self.attention_layer(x)
        h = self.fc(x)
        return h

class svm_attention():
    '''
    @Description: attention SVM for learning attention or feature selector in SVM classifier
    @param {type}:
                   
    @return: 
    '''
    def __init__(self, input_dim, save_path='att_svm.pkl', batch_size = 1, c = 0.01, learning_rate=0.01, epoch =10):
        #super(svm_attention, self).__init__()
        self.model = LinearSVM(input_dim) 
        if torch.cuda.is_available():
           self.model.cuda()
        self.lr = learning_rate
        self.epoch = epoch
        self.batchsize = batch_size
        self.c = c
        self.coef_ = []
        self.save_path = save_path

    def fit(self, X, Y):
        Y[Y == 0] = -1
        if not torch.is_tensor(X):
           X = torch.FloatTensor(X)
        else:
           X = X.float()
        if not torch.is_tensor(Y):
           Y = torch.FloatTensor(Y)
        else:
           Y = Y.float()
        N = len(Y)

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epoch):
            perm = torch.randperm(N)
            sum_loss = 0

            for i in range(0, N, self.batchsize):
                x = X[perm[i : i + self.batchsize]]
                y = Y[perm[i : i + self.batchsize]]

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                optimizer.zero_grad()
                output, self.attention = self.model(x)

                loss = self.c * torch.mean(torch.clamp(1 - output.t() * y, min=0))  # hinge loss
                loss += torch.mean(self.model.fc.weight ** 2)  # l2 penalty
                loss.backward()
                optimizer.step()

                sum_loss += loss.data.cpu().numpy()

            print("Epoch:{:4d}\tloss:{}".format(epoch, sum_loss / N))
        # self.coef_.append(torch.matmul(self.attention, self.model.fc.weight))
        self.coef_.append(self.model.fc.weight)
        torch.save(self.model.state_dict(),self.save_path)

    def predict(self, x):
        self.model.eval()
        pred = torch.squeeze(self.model.predict(x)).data.cpu()
        label_zero = torch.zeros(pred.shape)
        label_ones = torch.ones(pred.shape)
        pred = torch.where(pred > 0, label_ones, label_zero)
        pred = pred.data.cpu().numpy()
        return pred
    
    def get_attention(self, x):
        self.model.load_state_dict(torch.load(self.save_path))
        output, attention = self.model(x)
        return attention
