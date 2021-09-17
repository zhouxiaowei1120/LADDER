# coding=utf-8


import os
import sys
sys.path.append('./generate_latent/')
from myutils import *
import logging
from time import time
import torch
import numpy as np
import random
from random import choice
import matplotlib as mpl
mpl.use('Agg') # make plot figure work fine without 
import matplotlib.pyplot as plt
import multiprocessing
multiprocessing.set_start_method('spawn', True)
#multiprocessing.set_start_method('fork', force=True)
from img_trans import get_direction, compute_eps_generate_img
import generate_latent.mnist_main as mnist 
from generate_latent.GSN import GSN
from torchvision import datasets, transforms
from generate_latent.feaExtract import FeatureExtractor
from pytorch_cifar.models import ResNet18
from generate_latent.celebA_classify import celebA_dataset, celebANet
from generate_latent.svhn_clssifier import svhnVGG


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parseArg()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            #
    torch.cuda.manual_seed(seed)       #
    torch.cuda.manual_seed_all(seed)   #
    # Reference: https://blog.csdn.net/hyk_1996/article/details/84307108 

    parameters = dict()
    parameters['dataset'] = args.dataset
    parameters['train_attribute'] = args.train_attribute
    parameters['test_attribute'] = args.test_attribute
    parameters['dim'] = args.dim
    parameters['DNN'] = args.DNN
    parameters['layer'] = args.layer # modules of classifier[3]
    parameters['embedding_attribute'] = '{}_{}_1norm'.format(parameters['DNN'],parameters['layer'])
    parameters['nb_channels_first_layer'] = args.nb_channels_first_layer
    parameters['name_experiment'] = create_name_experiment(parameters, args.exp_att)
    parameters['iteration'] = args.iteration
    parameters['save_freq'] = args.save_freq
    parameters['batch_size'] = args.batch
    parameters['attention'] = args.att
    parameters['gpu_ids'] = args.gpu_ids
    parameters['res_dir'] = args.res_dir
    parameters['time_stamp'] =time_stamp() #'2021-09-07T16-32-20' #
    parameters['ori_lr'] = args.ori_lr
    parameters['restore_file'] = args.restore_file

    logpath = os.path.join(parameters['res_dir'], 'gsn_hf', parameters['name_experiment'], 'logs', parameters['time_stamp'])
    create_folder(logpath)
    mylogger(os.path.join(logpath, args.logfile))
    logger = logging.getLogger('mylogger')
    args.logger = logger
    if args.v == False:
        logger.setLevel('INFO')
    logger.info("Generate data on the boundary. {}".format(parameters['time_stamp']))

    device = torch.device("cuda" if args.gpu_ids != '' else "cpu")
    logger.info('Use device:{}'.format(device.type))
    parameters['device'] = device
    args.device = device

    parameters['GpuNum'] = 0
    if device.type == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = parameters['gpu_ids']
        parameters['GpuNum'] = torch.cuda.device_count()
        # When run code on single gpu, restore weights trained on multi gpus without bugs
        logger.info('There are {} GPUs, use GPU {}'.format(parameters['GpuNum'],parameters['gpu_ids']))

    args.gpu_ids = list(map(int,args.gpu_ids.replace(',','')))
    parameters['gpu_ids'] = args.gpu_ids

    logger.info(args)
    logger.info("Staring generation")
    gsn = GSN(parameters, args)
    if args.phase == 'train':
        g=gsn.train(args.epoch)
    else:
        g=''
    gsn.save_originals()
    gsn.generate_from_model(args.iteration, g)
    if args.phase == 'train':
        logger.info('Training Finished.')
        sys.exit(0) 

#########################################################
    # --Get some images from train dataset batch
    if parameters['DNN'] == 'lenet':
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./datasets/mnist/', train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=1, shuffle=True)
        if args.testdata:
            testset = datasets.MNIST('./datasets/mnist/', train=False, transform=transforms.Compose([transforms.ToTensor(),]))
            test_loader = torch.utils.data.DataLoader(testset,
            batch_size=1, shuffle=True)
        modelfile = './generate_latent/mnist_cnn.pt'
        classifier = mnist.Net()
        feaExtractor = FeatureExtractor('lenet', args.gpu_ids)
    elif parameters['DNN'] == 'cifarnet':
        trainset = datasets.CIFAR10(root='./datasets/cifar/', train=True, download=True, 
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
        if args.testdata:
            testset = datasets.CIFAR10(root='./datasets/cifar/', train=False, download=True, 
                            transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
            test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
        modelfile = './generate_latent/pytorch_cifar/checkpoint/cifar_cnn.pt'
        classifier = ResNet18()
        feaExtractor = FeatureExtractor('cifarnet', args.gpu_ids)
    elif parameters['DNN'] == 'celebANet':
        trainset = celebA_dataset('./datasets/celebA_post/', train=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
        if args.testdata:
            testset = celebA_dataset('./datasets/celebA_post/', train=False, 
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]))
            test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
        modelfile = './generate_latent/celebA_cnn.pt'
        classifier = celebANet('vgg11_bn', args.num_classes)
        feaExtractor = FeatureExtractor('celebANet', args.gpu_ids)
    elif parameters['DNN'] == 'svhnNet':
        trainset = datasets.SVHN('./datasets/SVHN/', split='train', download=True,
                                transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
        if args.testdata:
            testset = datasets.SVHN('./datasets/SVHN/', split='test', download=True,
                                    transform=transforms.ToTensor())
            test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
        modelfile = './generate_latent/svhn_cnn.pt'
        classifier = svhnVGG('VGG8')
        feaExtractor = FeatureExtractor('svhnNet', args.gpu_ids)
    # --
    classifier, state_dict = loadweights(classifier, modelfile, gpu_ids=args.gpu_ids)
    classifier.load_state_dict(state_dict)
    classifier.to(args.device)

    img_num = args.img_num # The maximum of images for one class to generate adv. Examples
    cav_imgnum = args.cav_imgnum # default is 200
    classIndex = 0
    ind_mat = np.array([], dtype=np.int32)
    labels_mat = np.array([])
    sample_mat = np.array([])
    cav_dir = os.path.join('./experiments/gsn_hf', parameters['name_experiment'], 'cavs')
    args.cav_dir = cav_dir
    if args.testdata:
        res_dir = os.path.join(args.res_dir, 'gsn_hf', parameters['name_experiment'], 'testdata_eps_diff', parameters['time_stamp'])
    else:
        res_dir = os.path.join(args.res_dir, 'gsn_hf', parameters['name_experiment'], 'eps_diff', parameters['time_stamp'])
    if not os.path.exists(cav_dir):
        os.makedirs(cav_dir)

    if parameters['DNN'] in ['lenet', 'cifarnet', 'svhnNet']:
        label_list = [0,1,2,3,4,5,6,7,8,9]
        label_max = 9
    #    if args.dataset == 'mnist':
    #        args.eps_list = [0.1, 1.0, 7.5, 8.0, 8.2, 8.5, 9.0, 15.0]
    #    elif args.dataset == 'SVHN':
    #        args.eps_list = [0.4, 0.6, 0.8, 1.0, 1.5]
    #    elif args.dataset == 'cifar':
    #        args.eps_list = [0.4, 1.5]
    elif parameters['DNN'] == 'celebANet':
        label_list = [0,1]
        label_max = 1
        #args.eps_list = [0.1, 0.3, 1.0, 10.0, 15.0, 17.0, 20.0]
    
        
    if args.testdata:
        eps_list = args.eps_list
        img_idx = 0
        t1 = time()
        logger.info('number of images for generating adversarial examples: {}'.format(len(testset)))
        for data, target in test_loader: 
            x = data.to(device)
            source_label = int(target.numpy())
            target_list = label_list[:]
            target_list.remove(source_label)
            target_label = choice(target_list)
            target_label_name = target_label
            source_target='source'
            dir_num = img_idx
            label_dir = str(source_label)+str(target_label)
    # find the closest data point in test dataset with different label than input x
            if args.dataset == 'mnist' or args.dataset=='mnist_part':
                train_set = './datasets/mnist/processed/training.pt'
            elif args.dataset == 'cifar':
                train_set = './datasets/cifar/'
            elif args.dataset == 'celebA_post':
                train_set = './datasets/celebA_post/'
            elif args.dataset == 'SVHN':
                train_set = './datasets/SVHN/train_32x32.mat'
            else:
                logger.error('Error! Unknow dataset: {}'.format(args.dataset))
                sys.exit(0)
            direction = get_direction(args, x, cav_imgnum, source_label, target_label, train_set, device, classifier, feaExtractor, cav_dir, source_target)
                                
            # --compute the epsilon and generate images from one side of decision boundary to another side--
            saveImgDir = os.path.join(res_dir, args.direction_model, label_dir, str(dir_num))
            if not os.path.exists(saveImgDir):
                os.makedirs(saveImgDir)
    #logger.info('Source label {} to Target label {}'.format(source_label, target_label))
            min_eps, success, label_arr = compute_eps_generate_img(args, gsn, x, direction, target_label_name, saveImgDir, dir_num, device, classifier, feaExtractor, eps_list, g)
            img_idx += 1
    
        t2 = time()-t1
        logger.info('Time cost for generation: {:.2f}'.format(t2))
        logger.info('Finished.')
        sys.exit(0)

    eps_list = args.eps_list
    img_idx = 0
    count_mtx = np.zeros([label_max+1, label_max+1])

    for data, target in train_loader: # Every time, randomly select data point from datasets; If img_num=20, for each class, we will select 20*9=180 data points.
        x = data.to(device)
        source_label = int(target.numpy())
        target_list = label_list[:]
        target_list.remove(source_label)
        if img_num != -1: # -1 means to generate one adv example for each training image
            if count_mtx.sum() == img_num*(label_max+1):
                break
            if np.sum(count_mtx[source_label]) >= img_num:
                continue
            for (_, tmp_label) in enumerate(target_list):
                if count_mtx[source_label, tmp_label] < img_num/label_max:
                    target_label = tmp_label
                    break
            count_mtx[source_label,target_label] += 1
        target_label = choice(target_list)
        target_label_name = target_label
        source_target='source'
        dir_num = img_idx
        label_dir = str(source_label)+str(target_label)
# find the closest data point in test dataset with different label than input x
        if args.dataset == 'mnist' or args.dataset=='mnist_part':
            train_set = './datasets/mnist/processed/training.pt'
        elif args.dataset == 'cifar':
            train_set = './datasets/cifar/'
        elif args.dataset == 'celebA_post':
            train_set = './datasets/celebA_post/'
        elif args.dataset == 'SVHN':
            train_set = './datasets/SVHN/train_32x32.mat'
        else:
            logger.error('Error! Unknow dataset: {}'.format(args.dataset))
            sys.exit(0)
        direction = get_direction(args, x, cav_imgnum, source_label, target_label, train_set, device, classifier, feaExtractor, cav_dir, source_target)
                            
        # --compute the epsilon and generate images from one side of decision boundary to another side--
        saveImgDir = os.path.join(res_dir, args.direction_model, label_dir, str(dir_num))
        if not os.path.exists(saveImgDir):
            os.makedirs(saveImgDir, exist_ok=True)
#logger.info('Source label {} to Target label {}'.format(source_label, target_label))
        min_eps, success, label_arr = compute_eps_generate_img(args, gsn, x, direction, target_label_name, saveImgDir, dir_num, device, classifier, feaExtractor, eps_list, g)
        img_idx += 1
    logger.info('Finished.')
