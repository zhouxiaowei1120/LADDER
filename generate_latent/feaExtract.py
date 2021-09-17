'''
@Author: Dave Zhou
@Date: 2018-12-03 14:26:34
@LastEditors: Dave Zhou
@LastEditTime: 2019-05-29 16:17:44
@Description: This is used for extracting features
''' 

import sys
import torch.nn as nn
import torch
import torchvision.models as models
import logging
from torch.autograd import Variable
from mnist_main import Net
# sys.path.append('./pytorch_cifar/')
from pytorch_cifar.models import *
from utils import loadweights
from celebA_classify import celebANet
from svhn_clssifier import svhnVGG
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, modelname, gpu_ids):
        super(FeatureExtractor,self).__init__()
        
        self.modelname = modelname
        self.gpu_ids = gpu_ids
        logger = logging.getLogger('mylogger')
        try:
            if modelname == 'vgg19_bn':
                self.mymodel = models.vgg19_bn(pretrained=True)
            elif modelname == 'vgg19':
                self.mymodel = models.vgg19(pretrained=True)
            elif modelname == 'lenet':
                self.mymodel = Net()
                state_dic = torch.load('./generate_latent/mnist_cnn.pt', map_location=lambda storage, loc: storage)
                self.mymodel.load_state_dict(state_dic)
            elif modelname == 'cifarnet':
                self.mymodel = ResNet18() 
                self.mymodel, state_dic = loadweights(self.mymodel,'./generate_latent/pytorch_cifar/checkpoint/cifar_cnn.pt', self.gpu_ids)
                self.mymodel.load_state_dict(state_dic)
                # self.mymodel = self.mymodel.module
            elif modelname == 'celebANet':
                self.mymodel = celebANet('vgg11_bn', 1)
                self.mymodel, state_dic = loadweights(self.mymodel, './generate_latent/celebA_cnn.pt', self.gpu_ids)
                self.mymodel.load_state_dict(state_dic)
            elif modelname == 'svhnNet':
                self.mymodel = svhnVGG('VGG8')
                self.mymodel, state_dic = loadweights(self.mymodel, './generate_latent/svhn_cnn.pt', self.gpu_ids)
                self.mymodel.load_state_dict(state_dic)
        except RuntimeError as e:
            logger.error("Error in feaExtract!{}".format(e))
            sys.exit(0)
        if len(self.gpu_ids) >= 1:
           self.mymodel.cuda()
           logger.info("Use gpu {}.".format(self.gpu_ids))
        self.mymodel.eval()     
        logger.info('Initialize the model:{}'.format(modelname))
        logger.info(self.mymodel.modules)
        logger.info('#################################################')

    def get_activation(self, layer, x):
        activation=[]
        def hook(model,input,output):
            if self.modelname == 'cifarnet':
#               print(output.shape)
#               output = output.view(output.shape[0], -1)
#print(output.shape)
                output = F.avg_pool2d(output, 4).squeeze().detach()
            activation.append(output)

        if len(self.gpu_ids) > 1:
            if self.modelname == 'lenet':
                fea_handle = self.mymodel.module.fc1.register_forward_hook(hook)
            elif self.modelname == 'cifarnet':
                fea_handle = self.mymodel.module.layer4.register_forward_hook(hook)
            elif self.modelname == 'svhnNet':
                fea_handle = self.mymodel.module.features.register_forward_hook(hook)
            elif self.modelname == 'celebANet':
                fea_handle = self.mymodel.module.model.classifier[layer].register_forward_hook(hook) # layer=0, when it is celebANet
            else:
                fea_handle = self.mymodel.module.classifier[layer].register_forward_hook(hook) # layer=0, when it is celebANet
        else:
            if self.modelname == 'lenet':
                fea_handle = self.mymodel.fc1.register_forward_hook(hook)
            elif self.modelname == 'cifarnet':
                fea_handle = self.mymodel.layer4.register_forward_hook(hook)
            elif self.modelname == 'svhnNet':
                fea_handle = self.mymodel.features.register_forward_hook(hook)
            elif self.modelname == 'celebANet':
                fea_handle = self.mymodel.model.classifier[layer].register_forward_hook(hook) # layer=0, when it is celebANet; mymodel.model, please refer to generator_architecture for the reason
            else:
                fea_handle = self.mymodel.classifier[layer].register_forward_hook(hook) 
            
        if len(self.gpu_ids) >= 1:
            outputs = self.mymodel(x.cuda())
        else:
            outputs = self.mymodel(x)
        fea_handle.remove()
        return activation

