'''
@Author: Dave Zhou
@Date: 2018-12-03 11:24:03
@LastEditors: Dave Zhou
@LastEditTime: 2018-12-03 11:52:10
@Description: 
'''
# coding=utf-8


import os
import sys
from myutils import *
import logging
import time
import torch

from utils import create_name_experiment
from GSN import GSN


if __name__ == '__main__':
    mylogger()
    args = parseArg()

    parameters = dict()
    parameters['dataset'] = args.dataset
    parameters['train_attribute'] = args.train_attribute
    parameters['test_attribute'] = args.test_attribute
    parameters['dim'] = args.dim
    parameters['DNN'] = args.DNN
    parameters['layer'] = args.layer # modules of classifier[3]
    #parameters['embedding_attribute'] = 'ScatJ4_projected{}_1norm'.format(parameters['dim'])
    parameters['embedding_attribute'] = '{}_{}_1norm'.format(parameters['DNN'],parameters['layer'])
    parameters['nb_channels_first_layer'] = args.nb_channels_first_layer
    parameters['name_experiment'] = create_name_experiment(parameters, 'NormL1')
    parameters['iteration'] = args.iteration
    parameters['save_freq'] = args.save_freq
    parameters['attention'] = args.att

    #parameters['']
    logger = logging.getLogger('mylogger')
    if args.v == False:
        logger.setLevel('INFO')
    logger.info("Generate data on the boundary. {}".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
    logger.info(args)
    logger.info("Staring generation")
    gsn = GSN(parameters)
    if args.phase == 'train':
        gsn.train(args.epoch)
    gsn.save_originals()
    gsn.generate_from_model(args.iteration)
    # gsn.compute_errors(536)
    # gsn.analyze_model(404)
