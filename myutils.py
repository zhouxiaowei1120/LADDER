# Reference from https://blog.csdn.net/u010895119/article/details/79470443

import logging
import argparse
import ast
from collections import OrderedDict
import torch
import os
from datetime import datetime

seed = 1

import random
import numpy as np
# random.seed(seed)
# np.random.seed(seed)
# # Reference: https://blog.csdn.net/hyk_1996/article/details/84307108 

def mylogger(logpath='./param.log'):
    logger = logging.getLogger('mylogger')
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    fhlr = logging.FileHandler(logpath) # 
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)
    str_sharp = '#####################################################################'
    logger.info('Record Experiment Information and Conditions\n'+str_sharp+'\n\n\n'+str_sharp)
    # logger.info('  Experiment Setting and Running Logs\n\n')

    chlr = logging.StreamHandler() # 
    chlr.setFormatter(formatter)
    chlr.setLevel('DEBUG')  # 
    logger.addHandler(chlr)

def parseArg ():
    parseArgs = argparse.ArgumentParser(description='Arguments for project.')
    parseArgs.add_argument('--att',type=ast.literal_eval, default= 'False', help='add attention layer or not') 
    parseArgs.add_argument('--batch',type=int, default= 200, help='Number of batch size') 
    parseArgs.add_argument('--dataset',type=str,default = 'cifar', help='specify the training dataset')
    parseArgs.add_argument('--dataAug',type=str,default = 'True', help='True for data autmentation, false for defence')
    parseArgs.add_argument('--dim',type=int, default= 512, help='the dimension of z') 
    parseArgs.add_argument('--DNN',type=str, default= 'cifarnet', help='the name of DNN for extracting features') 
    parseArgs.add_argument('--epoch',type=int, default= 0, help='the num of training epoch for restore,0 means training from scrach')
    parseArgs.add_argument('--restore_file',type=str, default='', help='the path/file for restore models')
    parseArgs.add_argument('--eps_step',type=float, default= 0.01, help='the step size for updating epsilon')
    parseArgs.add_argument('--eps_init',type=float, default= 0.0, help='the step size for updating epsilon')
    parseArgs.add_argument('--eps_max',type=int, default= 100.0, help='the maximum of epsilon')
    parseArgs.add_argument('--gpu_ids',type=str, default= '', help='the ids of GPUs')
    parseArgs.add_argument('--generator',type=str, default= 'autoencoder', help='name of generator: autoencoder or VAEGAN')
    parseArgs.add_argument('--iteration',type=int, default= 1000, help='the num of max iteration')
    parseArgs.add_argument('--layer',type=int, default= 3, help='the layer of extracting features in classifier module')
    parseArgs.add_argument('--ori_lr',type=float, default= 0.001, help='the original learing rate')
    parseArgs.add_argument('--lam',type=float, default= 0, help='the weight importance for mse loss and gan loss')
    parseArgs.add_argument('--nb_channels_first_layer',type=int, default= 512, help='the num of channels in first layer') 
    parseArgs.add_argument('--num_classes',type=int, default= 1, help='the num of classes in celebA dataset')    
    parseArgs.add_argument('--num_workers',type=int, default= 8, help='the num of process when load data')    
    parseArgs.add_argument('--norm',type=str, default= 'l2', help='type of loss function: l1 or l2') 
    parseArgs.add_argument('--percent',type=float, default= 1.0, help='the percentage of dataset for training classifier and generator')    
    parseArgs.add_argument('--phase',type=str, default= 'test', help='train or test') 
    parseArgs.add_argument('--save_freq',type=int, default= 100, help='the num of training epoch for restore')
    parseArgs.add_argument('--seed', type=int, default= 1, help='the seed for random selection')
    parseArgs.add_argument('--train_attribute',type=str, default= 'train', help='the name of training path')
    parseArgs.add_argument('--test_attribute',type=str, default= 'test', help='the name of test path')
    parseArgs.add_argument('--ite_max_eps',type=int, default= 300, help='the num of iteration for finding epsilon')
    parseArgs.add_argument('--img_num',type=int, default= 20, help='the maximum of images to generate adversatial images')
    parseArgs.add_argument('--direction_type',type=str, default= 'cav', help='the name of direction type')
    parseArgs.add_argument('--direction_model',type=str, default= 'att_svm', choices=['linear','att_svm','logistic', 'max_dis_svm', 'att_neighbor_svm'], help='the name of model for training direction')
    parseArgs.add_argument('--cav_imgnum',type=int, default= 200, help='the num of examples for training svm')
    parseArgs.add_argument('--v',type=ast.literal_eval, default = True, help='display the debug info or not')
    parseArgs.add_argument('--info', '-I', type=str, default='Info for running program',
                        help='This info is used to record the running conditions for the current program, which is stored in param.log')
    parseArgs.add_argument('--res_dir',type=str, default='./experiments', help='the path for saving results')
    parseArgs.add_argument('--exp_att',type=str, default='mnist_512_Norm2', help='the name of current experiment')
    parseArgs.add_argument('--logfile',type=str, default= './param.log', help='the name of log file')
    parseArgs.add_argument('--d_filename',type=str, default='', help='the path for discriminator weight file')
    parseArgs.add_argument('--decay',action='store_true', default=False, help='True for decay of optimizer')
    parseArgs.add_argument('--testdata',action='store_true', default=False, help='True for generating adv example from testdata, false for generating adv example from traindata')
    parseArgs.add_argument('--eps_list', type=float, nargs='+', help='the eps list for generating several adversarial examples')

    return parseArgs.parse_args()

def time_stamp():
  TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
  return TIMESTAMP

def create_name_experiment(parameters, attribute_experiment):
    name_experiment = '{}/{}'.format(parameters['dataset'], attribute_experiment)

    print('Name experiment: {}'.format(name_experiment))

    return name_experiment

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def write2file(filename, textList, options=[1,1]):
    fw = open(filename,'a')
    if options[0] == 1:
        fw.writelines(time_stamp()+'\n')
    for text in textList:
        fw.writelines(text+'\t')
    if options[1] == 1:
        fw.writelines('\n')
    fw.close()

def loadweights(model, filename_model, gpu_ids=''):
    '''
    @Description: Load weights for pytorch model in different hardware environments
    @param {type} : {model: pytorch model, model that waits for loading weights
                     filename_model: str, name of pretrained weights
                     gpu_ids: list, available gpu list}
    @return: 
    '''
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
        state_dict = torch.load(filename_model)
        model = torch.nn.DataParallel(model,device_ids=gpu_ids)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                name = ''.join(['module.',k]) # add `module.`
                new_state_dict[name] = v
        if new_state_dict:
            state_dict = new_state_dict
    return model, state_dict

def cover_pytorch_tf(pytorch_weights, tf_model_var, sess, match_dict):
    '''
    @Description: This function is used to copy trained weights from pytorch to tensorflow.
    @param {type} : {pytorch_weights: OrderDict, save the weights of one model
                     tf_model_var: tf variable list, save the variable list in tf model
                     sess: tf.Session()
                     match_dict: dic, the match relationship between pytorch weights and tf weiths}
    @return: copied weights file name for tf
    '''
    import tensorflow as tf
    # py_weights_name = ['num_batches_tracked']
    tf_py_weights_name = {'kernel':'weight', 'bias':'bias', 'gamma':'weight', 'beta':'bias', 'moving_mean':'running_mean', 'moving_variance':'running_var'}
    for tf_v in tf_model_var:
        tf_names = tf_v.name.split('/')
        tf_layer_name = '/'.join(tf_names[1:3]) # used for confirm the layer relationship
        
        py_weight_name = tf_py_weights_name.get(tf_names[3].split(':')[0]) # used for confirming the weight or bias relationship
        py_layer_name = match_dict.get(tf_layer_name)
        if py_layer_name == None:
            continue
        py_name = '.'.join([py_layer_name, py_weight_name])
        py_w = pytorch_weights.get(py_name)
        if len(py_w.shape) == 4:
            py_w = py_w.permute(3,2,1,0) # [64, 3, 3, 3] => [3, 3, 3, 64]
        elif py_w.dim() == 2:
            py_w = py_w.permute(1,0)
        assign_op = tf.assign(tf_v, py_w.cpu().detach().numpy())
        sess.run(assign_op)
    return tf_model_var

