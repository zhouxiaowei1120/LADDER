# coding=utf-8

import os
import sys
sys.path.append('./')

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
from feaExtract import FeatureExtractor


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        return input_tensor.view(*self.shape)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # 'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'VGG8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512],
}

class discriminator(nn.Module):
    def __init__(self, vgg_name='VGG8'):
        super(discriminator, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128,1)
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
  
class mnist_generator(nn.Module):
    def __init__(self, z_dim, nb_channels_first_layer=50, size_first_layer=4):
        super(mnist_generator, self).__init__()

        nb_channels_input = nb_channels_first_layer * 1
        self.l1 = nn.Sequential(
            nn.Linear(in_features=z_dim,
                      out_features=size_first_layer*size_first_layer*nb_channels_input,
                      bias=False),
            View(-1, nb_channels_input, size_first_layer, size_first_layer),
            nn.ReLU(inplace=True), # [50, 4,4]
            nn.ConvTranspose2d(nb_channels_input,nb_channels_input,2,4), # [50,14,14]
            # nn.ReLU(inplace=True)
            nn.Conv2d(50,20,3,1), # [20,12,12]
            nn.ReLU(inplace=True)
            )
        # self.atnn = Self_Attn(20, 'relu') # attention layer
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(20,20,2,3), # [20, 35,35]
            nn.Conv2d(20,20,4,1), # [50,32,32]
            nn.Conv2d(20,1,5,1), # [50,28,28]
            nn.Sigmoid()   
        )

    def forward(self,latent_fea):
        x = self.l1(latent_fea)
        # x, att1 = self.atnn(x)
        x = self.l2(x)
        return x        


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,     bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.d1= nn.Dropout(0.5)
        self.lrl1= nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lrl2= nn.LeakyReLU(0.2, inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes),
            )

    def forward(self, x):
        out = self.lrl1(self.d1(self.bn1(self.conv1(x))))
        out = (self.d1(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.lrl2(out)
        return out


class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class cifarGenerator(nn.Module):
    def __init__(self, ngpu, nc=3, nz=512, ngf=64):
        super(cifarGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(in_features=nz,
                      out_features=512*1*1,
                      bias=True),
            View(-1, 512, 1, 1), #[512,2,2] 
           # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            ResGenBlock(ngf*8, ngf*4),
            #BasicBlock(ngf*4, ngf*4),
            # state size. (ngf*4) x 8 x 8
            ResGenBlock(ngf*4, ngf*2),
            #BasicBlock(ngf*2, ngf*2),
            # state size. (ngf*2) x 16 x 16
            ResGenBlock(ngf*2, ngf*1),
            nn.BatchNorm2d(ngf*1),
            nn.ReLU(),
            nn.Conv2d(ngf*1, 3, 3, stride=1, padding=1),
#            nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class cifar_Discriminator(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=64):
        super(cifar_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            BasicBlock(ndf, ndf*2, 2),
            #BasicBlock(ndf*2, ndf*2),
            # state size. (ndf*2) x 16 x 16
            BasicBlock(ndf*2, ndf*4, 2),
            #BasicBlock(ndf*4, ndf*4),
            # state size. (ndf*4) x 8 x 8
            BasicBlock(ndf*4, ndf*8, 2),
            #BasicBlock(ndf*8, ndf*8),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 2, 0, bias=False),
#            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


#https://github.com/kvfrans/generative-adversial/blob/master/main.py
#https://github.com/YPDLsegmentation/VGG16-UNet/blob/master/model.py
class cifar_generator(nn.Module):
    def __init__(self, z_dim, nb_channels_first_layer=512, size_first_layer=4, archi=[512, 'M', 512, 512, 512, 512, 'M', 256, 256, 256, 256, 'M', 128, 128, 128, 128, 64, 64, 32, 32, 8, 8]):
        super(cifar_generator, self).__init__()
        
        self.archi = archi
        self.nb_channels_first_layer = nb_channels_first_layer
        self.l1 = nn.Sequential(
            nn.Linear(in_features=z_dim,
                      out_features=nb_channels_first_layer*size_first_layer*size_first_layer,
                      bias=True),
            View(-1, nb_channels_first_layer, size_first_layer, size_first_layer)#[512,2,2] 
            )     
        self.l2 = self._make_layers(self.archi)
        self.l3 = nn.Sequential(
            # nn.ConvTranspose2d(nb_channels_input//4,3,2,2), #[3,32,32]
            # nn.Conv2d(nb_channels_input, 3, 1, 1),
              nn.Tanh()
        #    nn.Sigmoid()
        )

    def forward(self,latent_fea):
#        x = nn.functional.interpolate(latent_fea.view(-1, self.nb_channels_first_layer, 1, 1), scale_factor=4)
        x = self.l1(latent_fea)
        # x, att1 = self.atnn(x)
        x = self.l2(x)
        x = self.l3(x)
        return x  
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.nb_channels_first_layer
        for x in cfg:
            if x == 'M':
                layers += [nn.ConvTranspose2d(in_channels,in_channels,2,2),
                           nn.BatchNorm2d(in_channels)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True) 
                           ]
                in_channels = x
        layers += [nn.Conv2d(in_channels, 3, 1, 1)]
        return nn.Sequential(*layers)


class svhn_generator(nn.Module):
    def __init__(self, z_dim, nb_channels_first_layer=512, size_first_layer=2, archi=[512, 'M', 512, 'M', 256, 'M', 128, 'M', 64]):
        super(svhn_generator, self).__init__()
        
        self.archi = archi
        self.nb_channels_first_layer = nb_channels_first_layer
        self.l1 = nn.Sequential(
            nn.Linear(in_features=z_dim,
                      out_features=nb_channels_first_layer*size_first_layer*size_first_layer,
                      bias=False),
            View(-1, nb_channels_first_layer, size_first_layer, size_first_layer)#[512,2,2] 
            )     
        self.l2 = self._make_layers(self.archi)
        self.l3 = nn.Sequential(
            # nn.ConvTranspose2d(nb_channels_input//4,3,2,2), #[3,32,32]
            # nn.Conv2d(nb_channels_input, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self,latent_fea):
        x = self.l1(latent_fea)
        # x, att1 = self.atnn(x)
        x = self.l2(x)
        x = self.l3(x)
        return x  
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.nb_channels_first_layer
        for x in cfg:
            if x == 'M':
                layers += [nn.ConvTranspose2d(in_channels,in_channels,2,2),
                           nn.BatchNorm2d(in_channels)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.Conv2d(in_channels, 3, 1, 1)]
        return nn.Sequential(*layers)

class celebA_generator(nn.Module):
    def __init__(self, z_dim, nb_channels_first_layer=512, size_first_layer=4, archi=['M', 512, 512, 512, 'M', 512, 512, 512, 'M', 256, 256, 256, 'M', 128, 128, 'M', 64, 64]):
        super(celebA_generator, self).__init__()
        
        self.archi = archi
        self.nb_channels_first_layer = nb_channels_first_layer
        self.l1 = nn.Sequential(
            nn.Linear(in_features=z_dim,
                      out_features=nb_channels_first_layer*size_first_layer*size_first_layer,
                      bias=False),
            View(-1, nb_channels_first_layer, size_first_layer, size_first_layer)#[512,4,4] 
            )     
        self.l2 = self._make_layers(self.archi)
        self.l3 = nn.Sequential(
            # nn.ConvTranspose2d(nb_channels_input//4,3,2,2), #[3,32,32]
            # nn.Conv2d(nb_channels_input, 3, 1, 1),
            nn.Tanh()   
        )

    def forward(self,latent_fea):
        x = self.l1(latent_fea)
        # x, att1 = self.atnn(x)
        x = self.l2(x)
        x = self.l3(x)
        return x  
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.nb_channels_first_layer
        for x in cfg:
            if x == 'M':
                layers += [nn.ConvTranspose2d(in_channels,in_channels,2,2),
                           nn.BatchNorm2d(in_channels)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.Conv2d(in_channels, 3, 1, 1)]
        return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, nb_channels_first_layer, z_dim, size_first_layer=7):
        super(Generator, self).__init__()

        nb_channels_input = nb_channels_first_layer * 32
        # self.att = att

        self.l1 = nn.Sequential(
            nn.Linear(in_features=z_dim,
                      out_features=size_first_layer * size_first_layer * nb_channels_input,
                      bias=False),
            View(-1, nb_channels_input, size_first_layer, size_first_layer),
            nn.BatchNorm2d(nb_channels_input, eps=0.001, momentum=0.9),
            nn.ReLU(inplace=True),

            ConvBlock(nb_channels_input, nb_channels_first_layer * 16, upsampling=True),
            ConvBlock(nb_channels_first_layer * 16, nb_channels_first_layer * 8, upsampling=True),
            ConvBlock(nb_channels_first_layer * 8, nb_channels_first_layer * 4, upsampling=True))
        self.attn1 = Self_Attn(nb_channels_first_layer * 4, 'relu') # attention layer
        self.l2 = nn.Sequential(
            ConvBlock(nb_channels_first_layer * 4, nb_channels_first_layer * 2, upsampling=True),
            # ConvBlock(nb_channels_first_layer * 2, nb_channels_first_layer * 2, upsampling=True),
            ConvBlock(nb_channels_first_layer * 2, nb_channels_first_layer, upsampling=True))
        # self.attn2 = Self_Attn(nb_channels_first_layer, 'relu') # attention layer
        self.l3 = ConvBlock(nb_channels_first_layer, nb_channels_output=3, relu=False)

    def forward(self, input_tensor):
        out = self.l1(input_tensor)
        out, att1 = self.attn1(out)
        out = self.l2(out)
        #out, att2 = self.attn2(out)
        out = self.l3(out)

        return out, att1#, att2

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//4 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//4 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) transpose the tensor
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class ConvBlock(nn.Module):
    def __init__(self, nb_channels_input, nb_channels_output, upsampling=False, relu=True):
        super(ConvBlock, self).__init__()

        self.relu = relu
        self.upsampling = upsampling

        filter_size = 3
        padding = (filter_size - 1) // 2

        if self.upsampling:
            self.up = nn.ConvTranspose2d(nb_channels_input,nb_channels_input,2,2)
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(nb_channels_input, nb_channels_output, filter_size, bias=False)
        self.bn_layer = nn.BatchNorm2d(nb_channels_output, eps=0.001, momentum=0.9)

    def forward(self, input_tensor):
        if self.upsampling:
            output = self.up(input_tensor)
        else:
            output = input_tensor

        output = self.pad(output)
        output = self.conv(output)
        output = self.bn_layer(output)

        if self.relu:
            output = torch.relu(output)
        else:
            #output = F.relu(output) 
            output = torch.sigmoid(output)

        return output


def weights_init(layer):
    if isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)
    elif isinstance(layer, nn.ConvTranspose2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


