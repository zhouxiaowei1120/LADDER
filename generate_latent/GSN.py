# coding=utf-8

import os
import numpy as np
from PIL import Image
# from tqdm import tqdm
import logging
from time import time
from collections import OrderedDict

import torch
from torch import nn
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
#from torchsummary import summary

from generator_architecture import Generator, weights_init, mnist_generator, cifar_generator, celebA_generator, svhn_generator, discriminator, cifarGenerator, cifar_Discriminator
from utils import create_folder, normalize, make_single_grid
from feaExtract import FeatureExtractor
from myutils import loadweights, time_stamp
from celebA_classify import celebA_dataset, celebANet

FIXBATCH = 16

class GSN:
    def __init__(self, parameters, args):
        dir_datasets = os.path.expanduser('./datasets')
        dir_experiments = os.path.expanduser(parameters['res_dir'])

        dataset_name = parameters['dataset']
        self.dataset_name = dataset_name
        train_attribute = parameters['train_attribute']
        test_attribute = parameters['test_attribute']
        embedding_attribute = parameters['embedding_attribute']
        self.modelname = parameters['DNN']
        self.layer = parameters['layer']
        self.attention = parameters['attention']
        self.device = parameters['device']
        self.gpuNum = parameters['GpuNum']
        self.ori_lr = parameters['ori_lr']
        self.dim = parameters['dim']
        self.nb_channels_first_layer = parameters['nb_channels_first_layer']
        name_experiment = parameters['name_experiment']

        if self.modelname == 'other':
            self.dir_x_train = os.path.join(dir_datasets, dataset_name, '{0}'.format(train_attribute))
            self.dir_x_test = os.path.join(dir_datasets, dataset_name,'{0}'.format(test_attribute))
        else: # lenet cifarnet celebANet svhnNet
            self.dir_x_train = os.path.join(dir_datasets, dataset_name) # not useful for lenet and cifarnet
            self.dir_x_test = os.path.join(dir_datasets, dataset_name)
        self.dir_z_train = os.path.join(dir_datasets, dataset_name, '{0}_{1}'.format(train_attribute, embedding_attribute))
        self.dir_z_test = os.path.join(dir_datasets, dataset_name, '{0}_{1}'.format(test_attribute, embedding_attribute))

        self.dir_experiment = os.path.join(dir_experiments, 'gsn_hf', name_experiment)
        timestamp = parameters['time_stamp']
        self.dir_models = os.path.join(self.dir_experiment, 'models', timestamp)
        self.dir_logs = os.path.join(self.dir_experiment, 'logs', timestamp)

        self.batch_size = parameters['batch_size']
        self.nb_epochs_to_save = 1
        self.iter = parameters['iteration']
        self.save_freq = parameters['save_freq']
        self.gpu_ids = parameters['gpu_ids']
        self.feaExtractor = FeatureExtractor(self.modelname, self.gpu_ids)
        self.restore_file = parameters['restore_file']
        self.d_filename = args.d_filename
        self.logger = args.logger
        self.decayLr = args.decay
        self.generator = args.generator
        self.percent = args.percent
        self.num_workers = args.num_workers
        self.lam = args.lam
        self.norm = args.norm
        self.phase = args.phase
        if self.logger.level == 10:
            if self.modelname == 'celebANet':
                input_size = (3,128,128)
            elif self.modelname == 'cifarnet' or self.modelname== 'svhnNet':
                input_size = (3,32,32)
#summary(self.feaExtractor.mymodel,input_size)        
#        if dataset == 'cifar':
#            if not os.path.exists(os.path.join(self.dir_x_train, dataset+'_x_y_z_train.pt')):
#                pre_save_embeddings(os.path.join(self.dir_x_train, dataset+'_x_y_z_'), self.feaExtractor, dataset, self.device, self.layer)
        if dataset_name == 'mnist':
            self.dataset = datasets.MNIST('./datasets/mnist/', train=True, download=True, transform=transforms.Compose([
                                           transforms.ToTensor(),
                                        ]))
        elif dataset_name == 'cifar':
            self.dataset = datasets.CIFAR10(root='./datasets/cifar/', train=True, download=True,
                             transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))
        elif dataset_name == 'celebA_post':
            self.dataset  = celebA_dataset('./datasets/celebA_post/', train=True,
                             transform=transforms.Compose([
                                     transforms.ToTensor(),
                                 ]))
        elif dataset_name == 'SVHN':
            self.dataset = datasets.SVHN('./datasets/SVHN/', split='train', download=True,
                                       transform=transforms.ToTensor())
        else:
            raise Exception('Unknow dataset.')
    
    @staticmethod
    def img_untransform(img):
        #mean=[0.485, 0.456, 0.406]
        #std=[0.229, 0.224, 0.225]
        mean=[0.5, 0.5, 0.5]
        std=[0.5, 0.5, 0.5]
        for i in [0,1,2]:
            img[i,:,:] = img[i,:,:] * std[i] + mean[i]
        Img_untrans = transforms.Compose([transforms.Lambda(lambda img: img * 255)])
        img = Img_untrans(img)
        return img
    
    @staticmethod
    def mnist_untransform(img):
        mean=[]
        std=[]
        #for i in [0,1,2]:
         #   img[i,:,:] = img[i,:,:] * std[i] + mean[i]
        Img_untrans = transforms.Compose([transforms.Lambda(lambda img: img * 255)])
        img = Img_untrans(img)
        return img

    def train(self, epoch_to_restore=0):
        create_folder(self.dir_models)
        create_folder(self.dir_logs)
        if self.modelname == 'lenet':
            g = mnist_generator(self.dim, self.nb_channels_first_layer)
        elif self.modelname == 'cifarnet':
            if self.generator == 'VAEGAN':
                #g = cifarGenerator(0)
                g = cifar_generator(self.dim, self.nb_channels_first_layer)
            else:
                g = cifar_generator(self.dim, self.nb_channels_first_layer)
        elif self.modelname == 'celebANet':
            g = celebA_generator(self.dim, self.nb_channels_first_layer)
        elif self.modelname == 'svhnNet':
            g = svhn_generator(self.dim, self.nb_channels_first_layer)
        else:
            g = Generator(self.nb_channels_first_layer, self.dim)

        if epoch_to_restore > 0:
            filename_model = os.path.join(self.restore_file)
            g, state_dict = loadweights(g, filename_model, self.gpu_ids)
            g.load_state_dict(state_dict)
        else:
            if len(self.gpu_ids) > 1:
                g = nn.DataParallel(g, device_ids=self.gpu_ids)
                g.module.apply(weights_init)
            else:
                g.apply(weights_init)

        g.to(self.device)
        self.logger.info('Use device:{}'.format(self.device))
        
        if self.generator == 'VAEGAN':
            self.logger.info('Use GAN as the generator. This is the sturcture of discriminator')
            d = cifar_Discriminator(0)
            #d = discriminator()
            d.to(self.device)
            d.apply(weights_init)
            if epoch_to_restore > 0:
                filename_model = os.path.join(self.d_filename)
                d, state_dict = loadweights(d, filename_model, self.gpu_ids)
                d.load_state_dict(state_dict)
            d.train()
            self.logger.info(d.modules)
            d_criterion = nn.BCEWithLogitsLoss()
            d_optimizer = optim.Adam(d.parameters(), lr=self.ori_lr*0.1, betas=(0.5, 0.999))
        
        self.logger.info('This is the structure of generator')
        self.logger.info(g.modules)
        if self.logger.level == 10:
            summary(g, (self.dim,))
        self.logger.info('#####################################')
        self.logger.debug('Before GetEmbedding is ok.')
        
        indices = torch.randperm(len(self.dataset)).tolist()[0:int(self.percent*len(self.dataset))]
        dataset = torch.utils.data.Subset(self.dataset, indices)
        self.logger.info('The number of images for training is {}.'.format(dataset.__len__()))
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False)
        fixed_batch_size = FIXBATCH
        fixed_dataloader = DataLoader(dataset, fixed_batch_size)
        fixed_batch = next(iter(fixed_dataloader))

        if self.norm == 'l2': #or self.modelname=='cifarnet':
            criterion = torch.nn.MSELoss()
        elif self.norm == 'l1':
            criterion = torch.nn.L1Loss()
        if self.modelname == 'cifarnet':
            g_optimizer = optim.Adam(g.parameters(), lr=self.ori_lr, betas=(0.5, 0.999))
        else:
            g_optimizer = optim.Adam(g.parameters(), lr=self.ori_lr)
        writer = SummaryWriter(self.dir_logs)

        if self.decayLr == True:
#g_optimizer = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10*287, 80*287, 200*287]) # decrease lr
             g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, factor=0.8, patience=30, verbose=True) # decrease lr
             if self.generator == 'VAEGAN':
                 d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, factor=0.8, patience=30, verbose=True) # decrease lr
#            g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=200) # decrease lr

        total_time = 0
        try:
            epoch = epoch_to_restore
            while epoch < self.iter:
                g.train()
                g_train_loss = 0
                d_train_loss = 0
                g_ae_loss = 0
                total_num = 0
                for _ in range(self.nb_epochs_to_save):
                    epoch += 1
                    self.logger.info('This is the iteration {}.'.format(epoch))
                    startT = time()
                    for current_batch in dataloader:
                        batch_time = time()
                        g_optimizer.zero_grad()
                        if self.generator == 'VAEGAN':
                           d_optimizer.zero_grad()
                        if self.dataset_name == 'cifar':
                            x = (current_batch[0].to(self.device)-0.5)*2
                        else:
                            x = (current_batch[0].to(self.device))
                        y = current_batch[1].to(self.device)
                        z = self.feaExtractor.get_activation(self.layer,x)[0] 
                        g_z  = g.forward(z.detach())
                        #print(x.min(), x.max(), g_z.min(), g_z.max())
                        #self.logger.debug(type(g_z),g_z.shape)
                        if self.generator == 'VAEGAN':
                            real_pre = d(x)
                            d_real_loss = d_criterion(real_pre, torch.ones_like(y).float())
                            fake_pre = d(g_z.detach()) #.squeeze()
                            d_fake_loss = d_criterion(fake_pre, torch.zeros_like(y).float())
                            d_loss = d_real_loss + d_fake_loss
                            d_train_loss += d_loss.item() * y.size(0)
                            
                            D_x = torch.sigmoid(real_pre).mean().item()
                            D_fake = torch.sigmoid(fake_pre).mean().item()
                            d_loss.backward()
                            d_optimizer.step()

                        if self.generator == 'VAEGAN':
                           g_fake_pre = d(g_z)
                           g_loss1 = d_criterion(g_fake_pre, torch.ones_like(y).float())
                           g_D_fake = torch.sigmoid(g_fake_pre).mean().item()
                        else:
                           g_loss1 = 0
                        if self.modelname == 'other':
                            # g_z = g_z[:,:,16:240,16:240]
                            g_z = g_z[0]
                        self.logger.debug('type g_z:{}, shape g_z:{}, max and min:{}, {}'.format(type(g_z),g_z.shape, torch.max(g_z), torch.min(g_z)))
                        self.logger.debug('shape x:{}, max and min:{}, {}'.format(x.shape, torch.max(x), torch.min(x)))

                        g_loss2 = criterion(g_z, x)
                        g_loss = self.lam * g_loss1 + g_loss2
                        g_train_loss += g_loss.item() * y.size(0)
                        g_ae_loss += g_loss2.item() * y.size(0)

                        g_loss.backward()
                        g_optimizer.step()
                        
                        batch_time_end = time() - batch_time
                        total_num += y.size(0)
                    endT = time() - startT
                    total_time += endT
                    if self.decayLr == True:
                        g_scheduler.step(g_train_loss/total_num)
                        if self.generator == 'VAEGAN':
                            d_scheduler.step(g_train_loss/total_num)
                    writer.add_scalar('generator train_loss', g_train_loss/total_num, epoch)
                    if self.generator == 'VAEGAN':
                        writer.add_scalar('discriminator train_loss', d_train_loss/total_num, epoch)
                        writer.add_scalar('generator train_loss/AE loss', g_ae_loss/total_num, epoch)
                        self.logger.info('epoch: {}, D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}'.format(epoch,D_x, D_fake, g_D_fake))
                    writer.add_scalar('Learning rate',g_optimizer.param_groups[0]['lr'], epoch)
                    if self.generator == 'VAEGAN':
                        self.logger.info('epoch: {}, generator train_loss: {}, g_aeLoss: {}, discriminator loss: {},  running time: {}s, lr: {}'.format(epoch,g_train_loss/total_num, g_ae_loss/total_num, d_train_loss/total_num, endT, g_optimizer.param_groups[0]['lr']))
                    else:
                        self.logger.info('epoch: {}, generator train_loss: {}, running time: {}s, lr: {}'.format(epoch,g_train_loss/total_num, endT, g_optimizer.param_groups[0]['lr']))
                if self.dataset_name == 'cifar':
                    x = (fixed_batch[0].to(self.device)-0.5)*2
                else:
                    x = fixed_batch[0].to(self.device)
                y = fixed_batch[1].to(self.device)
                z = self.feaExtractor.get_activation(self.layer,x)[0] 

                g.eval()
                g_z = g.forward(z)
                if self.modelname == 'lenet':
                    input_images = make_single_grid(x.data[:fixed_batch_size], nrow=4)
                    images = make_single_grid(g_z.data[:fixed_batch_size], nrow=4)
                elif self.modelname == 'celebANet' or self.modelname == 'svhnNet':
                    input_images = make_single_grid(x.data[:fixed_batch_size], nrow=4, range=(0.0,1.0))
                    images = make_single_grid(g_z.data[:fixed_batch_size], nrow=4, range=(0.0,1.0))
                elif self.modelname == 'cifarnet':
                    input_images = make_single_grid(x.data[:fixed_batch_size], nrow=4,normalize=True, range=(-1.0,1.0))
                    images = make_single_grid(g_z.data[:fixed_batch_size], nrow=4,normalize=True, range=(-1.0,1.0))
                else:
                    g_z = g_z[0]
                    input_images = make_grid(x.data[:fixed_batch_size], nrow=4, normalize=True)
                    images = make_grid(g_z.data[:fixed_batch_size], nrow=4, normalize=True)
                # att_images = make_grid(att1.data[:fixed_batch_size], nrow=4, normalize=True)
                writer.add_image('input images', input_images, epoch)
                writer.add_image('generations', images, epoch)
                writer.add_histogram('generated images histgram',g_z.data[0], epoch)
                writer.add_histogram('input images histgram',x.data[0], epoch)
                # writer.add_image('attention images', att_images, epoch)
                filename = os.path.join(self.dir_models, 'epoch_latest.pth')
                torch.save(g.state_dict(), filename)
                if self.generator == 'VAEGAN':
                    filename = os.path.join(self.dir_models, 'epoch_latest_d.pth')
                    torch.save(d.state_dict(), filename)
                if epoch%self.save_freq == 0:
                    filename = os.path.join(self.dir_models, 'epoch_{}.pth'.format(epoch))
                    torch.save(g.state_dict(), filename)
                    if self.generator == 'VAEGAN':
                        filename = os.path.join(self.dir_models, 'epoch_{}_d.pth'.format(epoch))
                        torch.save(d.state_dict(), filename)
                del z, x # This is used to relase the gpu memory
                del g_z, images, input_images
        finally:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                self.logger.info("Empty the cuda cache")
            self.logger.info("Total training time: {}s".format(total_time))
            self.logger.info('[*] Closing Writer.')
            writer.close()
        return g

    def save_originals(self):
        def _save_originals(dir_z, dir_x, train_test):
            fixed_dataloader = DataLoader(self.dataset, 16)
            fixed_batch = next(iter(fixed_dataloader))

            filename_images = os.path.join(self.dir_experiment, 'originals_{}.png'.format(train_test))
            if self.modelname == 'lenet':
                x = fixed_batch[0]
                temp = make_single_grid(x, nrow=4)
                temp = GSN.mnist_untransform(temp)
            elif self.modelname == 'celebANet' or self.modelname == 'svhnNet':
                x = fixed_batch[0]
                temp = make_single_grid(x, nrow=4)
                temp = GSN.mnist_untransform(temp)
            elif self.modelname == 'cifarnet':
                x = fixed_batch[0]
                temp = make_grid(fixed_batch[0], nrow=4)
                temp = GSN.img_untransform(temp)
                #temp = make_grid(fixed_batch['x'], nrow=4).numpy().transpose((1, 2, 0))
            temp = np.uint8(temp.cpu().numpy().transpose((1,2,0)))
            Image.fromarray(temp).save(filename_images)
            
        self.logger.info('Running in save_originals')
        _save_originals(self.dir_z_train, self.dir_x_train, 'train')

    def compute_errors(self, epoch):
        filename_model = os.path.join(self.restore_file)
        if self.modelname == 'lenet':
            g = mnist_generator(self.dim, self.nb_channels_first_layer)
        elif self.modelname == 'cifarnet':
            g = cifar_generator(self.dim, self.nb_channels_first_layer)
        elif self.modelname == 'celebANet':
            g = celebA_generator(self.dim, self.nb_channels_first_layer)
        elif self.modelname == 'svhnNet':
            g = svhn_generator(self.dim, self.nb_channels_first_layer)
        else:
            g = Generator(self.nb_channels_first_layer, self.dim)
        
        g, state_dict = loadweights(g, filename_model, self.gpu_ids)
        g.load_state_dict(state_dict)
        g.to(self.device)
        g.eval()

        if self.modelname == 'lenet':
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.L1Loss()

        def _compute_error(dir_z, dir_x, train_test):
            dataloader = DataLoader(self.dataset, batch_size=512, num_workers=0, pin_memory=False)

            error = 0

            for current_batch in dataloader:
                x = fixed_batch[0].to(self.device)
                y = fixed_batch[1].to(self.device)
                z = self.feaExtractor.get_activation(self.layer,x)[0] 
                g_z = g.forward(z)

                if self.modelname == 'other':
                    # g_z = g_z[:,:,16:240,16:240]
                    g_z = g_z[0]
                g_z = g_z.squeeze()
                error += criterion(g_z, x).data.cpu().numpy()

            error /= len(dataloader)

            self.logger.info('Error for {}: {}'.format(train_test, error))

        self.logger.info('Running in compute_erro')
        _compute_error(self.dir_z_train, self.dir_x_train, 'train')

    def restore_model(self, epoch):
        filename_model = os.path.join(self.restore_file)
        if self.modelname == 'lenet':
            g = mnist_generator(self.dim, self.nb_channels_first_layer)
        elif self.modelname == 'cifarnet':
            g = cifar_generator(self.dim, self.nb_channels_first_layer)
        elif self.modelname == 'celebANet':
            g = celebA_generator(self.dim, self.nb_channels_first_layer)
        elif self.modelname == 'svhnNet':
            g = svhn_generator(self.dim, self.nb_channels_first_layer)
        else:
            g = Generator(self.nb_channels_first_layer, self.dim)

        g, state_dict = loadweights(g, filename_model, self.gpu_ids)
        
        g.load_state_dict(state_dict)
        g.to(self.device)
        g.eval()
        return g

    def generate_from_feature(self, g, feature):
        start_t = time()
        g_z = g.forward(feature)
        end_t = time() - start_t
        self.logger.debug('Generating time: {}s'.format(end_t))
        self.logger.debug('shape of g_z:{}'.format(g_z.shape))
        if self.modelname in ['lenet', 'celebANet', 'svhnNet']:
            g_z = GSN.mnist_untransform(g_z)
            g_z = torch.squeeze(g_z)
        elif self.modelname in ['cifarnet']:
            g_z = GSN.img_untransform(g_z)
            g_z = torch.squeeze(g_z)
        else:
            g_z = GSN.img_untransform(g_z)
            #temp = make_grid(fixed_batch['x'], nrow=4).numpy().transpose((1, 2, 0))
        return g_z

    def generate_from_model(self, epoch, g):
        if self.phase == 'test':
           g = self.restore_model(epoch)
          
        def _generate_from_model(dir_z, dir_x, train_test):
            fixed_dataloader = DataLoader(self.dataset, 16)
            fixed_batch = next(iter(fixed_dataloader))

            x = fixed_batch[0].to(self.device)
            y = fixed_batch[1].to(self.device)
            z = self.feaExtractor.get_activation(self.layer,x)[0] 
            start_t = time()
            g_z = g.forward(z)
            if self.modelname == 'other':
                g_z = g_z[0]
            end_t = time() - start_t
            self.logger.info('Total test(model) time: {}s'.format(end_t))
            self.logger.debug('shape of g_z:{}'.format(g_z.shape))
            filename_images = os.path.join(self.dir_experiment, 'epoch_{}_{}.png'.format(epoch, train_test))

            if self.modelname in ['lenet', 'celebANet', 'svhnNet']:
                temp = make_single_grid(g_z.data[:16], nrow=4).cpu()
                temp = GSN.mnist_untransform(temp)
            elif self.modelname in ['cifarnet']:
                temp = make_grid(g_z.data[:16], nrow=4).cpu()
                temp = GSN.img_untransform(temp)
            else:
                temp = make_grid(g_z.data[:16], nrow=4).cpu()
                temp = GSN.img_untransform(temp)

            self.logger.debug('shape of temp:{}'.format(temp.shape))
            
            Image.fromarray(np.uint8(temp.cpu().numpy().transpose((1, 2, 0)))).save(filename_images)
            #temp = make_grid(g_z.data[:16], nrow=4).cpu().numpy().transpose((1, 2, 0))
            #Image.fromarray(np.uint8((temp + 1) * 127.5)).save(filename_images)
            del g_z, z, temp, fixed_batch
        
        self.logger.info('Running in generate_from_model')
        _generate_from_model(self.dir_z_train, self.dir_x_train, 'train')

        self.logger.info('Generation finished.')

    def analyze_model(self, epoch):
        filename_model = os.path.join(self.restore_file)
        g = Generator(self.nb_channels_first_layer, self.dim)
        if self.gpuNum > 1:
            g = torch.nn.DataParallel(g,device_ids=self.gpu_ids)
        g.to(self.device)
        g.load_state_dict(torch.load(filename_model))
        g.eval()

        nb_samples = 50
        batch_z = np.zeros((nb_samples, 32 * self.nb_channels_first_layer, 4, 4))
        # batch_z = np.maximum(5*np.random.randn(nb_samples, 32 * self.nb_channels_first_layer, 4, 4), 0)
        # batch_z = 5 * np.random.randn(nb_samples, 32 * self.nb_channels_first_layer, 4, 4)

        for i in range(4):
            for j in range(4):
                batch_z[:, :, i, j] = create_path(nb_samples)
        # batch_z[:, :, 0, 0] = create_path(nb_samples)
        # batch_z[:, :, 0, 1] = create_path(nb_samples)
        # batch_z[:, :, 1, 0] = create_path(nb_samples)
        # batch_z[:, :, 1, 1] = create_path(nb_samples)
        batch_z = np.maximum(batch_z, 0)

        z = Variable(torch.from_numpy(batch_z)).type(torch.FloatTensor).to(self.device)
        temp = g.main._modules['4'].forward(z)
        for i in range(5, 10):
            temp = g.main._modules['{}'.format(i)].forward(temp)

        g_z = temp.data.cpu().numpy().transpose((0, 2, 3, 1))

        folder_to_save = os.path.join(self.dir_experiment, 'epoch_{}_path_after_linear_only00_path'.format(epoch))
        create_folder(folder_to_save)

        for idx in range(nb_samples):
            filename_image = os.path.join(folder_to_save, '{}.png'.format(idx))
            Image.fromarray(np.uint8((g_z[idx] + 1) * 127.5)).save(filename_image)


def create_path(nb_samples):
    z0 = 5 * np.random.randn(1, 32 * 32)
    z1 = 5 * np.random.randn(1, 32 * 32)

    # z0 = np.zeros((1, 32 * 32))
    # z1 = np.zeros((1, 32 * 32))

    # z0[0, 0] = -20
    # z1[0, 0] = 20

    batch_z = np.copy(z0)

    interval = np.linspace(0, 1, nb_samples)
    for t in interval:
        if t > 0:
            zt = (1 - t) * z0 + t * z1
            batch_z = np.vstack((batch_z, zt))

    return batch_z
