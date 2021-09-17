'''
@Author: Dave Zhou
@Date: 2018-12-09 20:48:13
LastEditors: Dave Zhou
LastEditTime: 2021-09-06 21:22:50
@Description: This is used to implement the code for paper: The space of transferable adversarial examples
'''

import torch
import scipy.spatial.distance as distance
import generate_latent.mnist_main as mnist 
import generate_latent.pytorch_cifar.cifar_main as cifar
import generate_latent.celebA_classify as celebA
from generate_latent.celebA_classify import celebA_dataset
import generate_latent.svhn_clssifier as svhn
import logging
import os.path
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as trans
import torchvision.datasets as datasets
from generate_latent.GSN import GSN
# import matplotlib.pyplot as plt
import sys
from cav import get_or_train_cav, CAV
from svm_atten import svm_attention
import pickle
import scipy.io as scio

def find_closest_img(x, modelname, target_class, test_set, device, gpu_ids, classifier):
    '''
    @Description: Find the closest data point in test dataset with a different class label than input x
    @param {type}: {x: input image, type: tensor}
                   {modelname: the name of classifier, type: model object}
                   {target_class: the target class label, type: int}
                   {test_set: the dir of test dataset, type: string} 
    @return: the closest data point, type: tensor 
    '''
    mylogger = logging.getLogger('mylogger')    
    if modelname == 'lenet':
        modelfile = './generate_latent/mnist_cnn.pt'
        output = mnist.evalu(x, gpu_ids, modelfile, classifier, device)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        mylogger.info('Finding the closest image in the test set with target class label.')
        test_img = torch.load(test_set, map_location=lambda storage, loc: storage) # load all test data in mnist test dataset
        target_index = []
        for i in range(0, len(test_img[1])):
            if test_img[1][i] == target_class:
                target_index.append(i)
        target_index = torch.from_numpy(np.array(target_index))
        test_labels = torch.index_select(test_img[1], 0, target_index)
        test_imgs = torch.index_select(test_img[0], 0, target_index)
        min_dist = float('inf')
        target_img = []
        for img in test_imgs:
            img = Image.fromarray(img.numpy(), mode='L')
            transform=trans.Compose([trans.ToTensor()])
            img = transform(img)
            img.to(device)
            dist = torch.dist(img, x.cpu())
            output = mnist.evalu(torch.unsqueeze(img,dim=0), gpu_ids, modelfile, classifier, device)
            target_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if dist < min_dist:
                min_dist = dist
                target_img = img
        mylogger.info('The predicted label of input image is {}; the targetsedly predicted label is {}'.format(pred,target_pred))        
        return target_img

# def get_activations():
    

def get_direction(args, input_img, cav_imgnum, source_class, target_class, train_set, device, classifier, feaExtractor, cav_dir, source_target):
    '''
    @Description: This function is used for getting feature transform directions
    @param {type} 
        direction_type: 
        input_img: img for getting diretions
        train_set: training dataset for getting cav
    @return: 
    '''
    mylogger = logging.getLogger('mylogger')
    if args.direction_type == 'closest':
        target_img = find_closest_img(input_img, args.DNN, target_class, train_set, device, args.gpu_ids, classifier)
        if input_img.shape.__len__() != 4:
            input_img = torch.unsqueeze(input_img,dim=0)
        if target_img.shape.__len__() != 4:
            target_img = torch.unsqueeze(target_img,dim=0)
        act_input = feaExtractor.get_activation(layer=0, x=input_img)[0] # get activation of input image    
        act_target = feaExtractor.get_activation(layer=0, x=target_img)[0]
        direction = (act_input - act_target) / torch.dist(act_input, act_target) # the direction of image transform
    elif args.direction_type == 'random':
        act_input = feaExtractor.get_activation(layer=0, x=input_img)[0] # get activation of input image
        direction = torch.randn(act_input.squeeze().shape[0])/10.0
#print(direction)
        direction = [direction]
    elif args.direction_type == 'cav' or args.direction_type == 'cavRandom':
        alpha = 0.1
        cav_hapram = dict()
        cav_hapram['alpha'] = alpha
        cav_hapram['model_type'] = args.direction_model
        cav_hapram['input_dim'] = args.dim
        cav_path = os.path.join(
        cav_dir,CAV.cav_key([source_class, target_class], args.layer, args.direction_model,
                    alpha).replace('/', '.') + '.pkl')
        if not os.path.exists(cav_path):
            mylogger.info('CAV {} does not exist. Start training.'.format(cav_path))
            train_img = []
            if args.dataset == 'mnist' or args.dataset == 'mnist_part':
                train_img = torch.load(train_set, map_location=lambda storage, loc: storage) # load all train data in mnist train dataset
            elif args.dataset == 'cifar':
                train_data = datasets.CIFAR10(root='./datasets/cifar/', train=True, download=True, transform=trans.Compose([trans.ToTensor(),]))

                indices = torch.randperm(len(train_data)).tolist()[0:int(args.percent*len(train_data))]
                train_img.append(torch.from_numpy((train_data.data[indices].transpose(0, 3, 1,2))/255.0))
                train_img.append(torch.from_numpy(np.array(train_data.targets)[indices]))
            elif args.dataset == 'celebA_post':
                train_imgTmp = np.empty(shape=[0,128,128,3])
                train_labelTmp = np.empty(shape=[0,1])
                imgset = celebA_dataset(train_set, train=False)
                imgTrans = trans.ToTensor()
                img_names = imgset.filenames
                labels = imgset.labelsList
                for filename in img_names:
                    img = Image.open(filename)
                    train_imgTmp = np.append(train_imgTmp, np.array(img)[np.newaxis,:,:,:], axis=0)
                    idx = int(filename.split('/')[-1].split('.')[0])
                    labelLine = labels[idx-1]
                    labelLine = labelLine.rstrip('\n')
                    labelLine = labelLine.split()
                    label = int(labelLine[32])
                    if label == -1:
                        label = 0
                    train_labelTmp = np.append(train_labelTmp, np.array([[label]]), axis=0)
                train_imgTmp = torch.from_numpy((train_imgTmp/255).transpose((0,3,1,2))) # change to [-1, 3,128,128]
                train_labelTmp = torch.from_numpy(train_labelTmp)
                train_img.append(train_imgTmp)
                train_img.append(train_labelTmp)
            elif args.dataset == 'SVHN':
                if args.dataAug == 'True':
                   train_data =  datasets.SVHN('./datasets/SVHN/', split='train', download=True, transform=trans.ToTensor())
                   indices = torch.randperm(len(train_data)).tolist()[0:int(args.percent*len(train_data))]
                   train_img.append(torch.from_numpy(train_data.data[indices]))
                   train_img.append(torch.from_numpy(train_data.labels[indices]))
                else:
                   train_data = datasets.SVHN('./datasets/SVHN/', split='train', download=True,
                                transform=trans.ToTensor())
                   train_img.append(torch.from_numpy(train_data.data))
                   train_img.append(torch.from_numpy(train_data.labels))
            
            get_act_batch = args.cav_imgnum
            target_index = []
            source_index = []
            for i in range(0, len(train_img[1])):
                if train_img[1][i] == target_class:
                    target_index.append(i)  # get index of source image
                elif train_img[1][i] == source_class:
                    source_index.append(i)  # get index of source image
            target_index = torch.from_numpy(np.array(target_index[:cav_imgnum])) # get specific number of target images
            train_target_labels = torch.index_select(train_img[1], 0, target_index)
            train_target_imgs = torch.index_select(train_img[0], 0, target_index) # shape: [num_img, channels, weight, height]
            source_index = torch.from_numpy(np.array(source_index[:cav_imgnum])) # get specific number of source images
            train_source_labels = torch.index_select(train_img[1], 0, source_index)
            train_source_imgs = torch.index_select(train_img[0], 0, source_index)

            if len(args.gpu_ids) == 0:
                train_target_imgs = train_target_imgs.type('torch.FloatTensor')
                train_source_imgs = train_source_imgs.type('torch.FloatTensor')
            else:
                train_target_imgs = train_target_imgs.type('torch.cuda.FloatTensor')
                train_source_imgs = train_source_imgs.type('torch.cuda.FloatTensor')
            if train_source_imgs.shape.__len__() != 4:
                train_source_imgs = torch.unsqueeze(train_source_imgs,dim=1)
            if train_target_imgs.shape.__len__() != 4:
                train_target_imgs = torch.unsqueeze(train_target_imgs,dim=1)
            src_acts = []
            for i in range(int(np.ceil(args.cav_imgnum//get_act_batch))): # choose the upper number of batch
                src_act = feaExtractor.get_activation(layer=0, x=train_source_imgs[i*get_act_batch:(i+1)*get_act_batch,:]) # get activation of source images within the batch; type: list
                #print(src_act.__len__())
                src_acts.extend(src_act)
                #print(i, cav_imgnum, args.cav_imgnum, train_source_imgs[i*get_act_batch:(i+1)*get_act_batch,:].shape, src_acts.__len__())
            
            source_acts = np.empty(shape=[0,args.dim])
            for src_act in src_acts:
               source_acts = np.append(source_acts, src_act.detach().cpu().numpy(), axis=0)
            mylogger.info('Shape of source images activations:{}'.format(source_acts.shape))
            del src_acts, src_act

            tgt_acts = []
            for i in range(int(np.ceil(args.cav_imgnum//get_act_batch))): # choose the upper number of batch
                tgt_act = feaExtractor.get_activation(layer=0, x=train_target_imgs[i*get_act_batch:(i+1)*get_act_batch,:]) # get activation of target images within the batch: type list
                tgt_acts.extend(tgt_act)
            target_acts = np.empty(shape=[0, args.dim])
            for tgt_act in tgt_acts:
              target_acts = np.append(target_acts, tgt_act.detach().cpu().numpy(), axis=0)
            mylogger.info('Shape of target images activations:{}'.format(target_acts.shape))
            del tgt_act, tgt_acts

            source_acts = source_acts.astype(np.float32)
            target_acts = target_acts.astype(np.float32)

            source_acts_dit = dict()
            target_acts_dit = dict()
            source_acts_dit[args.layer] = source_acts # save source activations (array type) to dictionary
            target_acts_dit[args.layer] = target_acts # save target activations (array type) to dictionary
            # print(source_acts.dtype, target_acts.dtype)
            '''
            save activations into a dit: acts[class][bottleneck]  acts takes for of
                {'source_class':{'bottleneck name1':[...act array...],
                            'bottleneck name2':[...act array...],...
                'target_class':{'bottleneck name1':[...act array...],
            '''
            acts = {source_class:source_acts_dit,target_class:target_acts_dit} # Save in this structure for using code in TCAV directly
            cav_instance = get_or_train_cav([source_class, target_class], args.layer, acts, args.gpu_ids, cav_dir, cav_hapram)
        else:
# mylogger.info('CAV already exists: {}'.format(cav_path))
            cav_instance = CAV.load_cav(cav_path)
        
        if source_target == 'source':
            direction_class = target_class
        elif source_target == 'target':
            direction_class = source_class

        if cav_hapram['model_type'] == 'att_svm':
            direction, attention = cav_instance.get_direction(direction_class) # If the parameter is source_class, the direction is target class to source class. If it is target_class, the direction is source class to target class.
        else:
            direction = cav_instance.get_direction(direction_class) # If the parameter is source_class, the direction is target class to source class. If it is target_class, the direction is source class to target class.
            if not torch.is_tensor(direction):
                direction = torch.from_numpy(direction)
        direction = torch.unsqueeze(direction, dim=0)
        if len(args.gpu_ids) == 0:
            direction = direction.type('torch.FloatTensor')
        else:
            direction = direction.type('torch.cuda.FloatTensor')
            if cav_hapram['model_type'] == 'att_svm':
              attention = attention.type('torch.cuda.FloatTensor')
        if args.direction_type == 'cavRandom':
            direction = direction.squeeze() + torch.randn(direction.squeeze().shape[0], device=args.device)/100.0
#print(direction)
        if cav_hapram['model_type'] == 'att_svm':
            direction = [direction, attention]
        else:
            direction = [direction]            
    else:
        mylogger.error('Unknown direction type. Please check the parameter direction_type.')
        raise ValueError('Unknown direction type: {}'.format(args.direction_type))
        # sys.exit()
    return direction


def compute_eps_generate_img(args, generator, input_img, direction, targetlabel, img_dir, img_idx, device, classifier, feaExtractor, eps_list,g):
    mylogger = args.logger
    if input_img.shape.__len__() != 4:
        input_img = torch.unsqueeze(input_img,dim=0)
    act_input = feaExtractor.get_activation(layer=0, x=input_img)[0] # get activation of input image;shape:1 X input_dim
    if len(act_input.shape) != 2:
        act_input = torch.unsqueeze(torch.squeeze(act_input), dim=0)
    
    # --initialize the model, to check whether the target class label is equal to original label--
    if args.DNN == 'lenet':
        modelfile = './generate_latent/mnist_cnn.pt'
        output = mnist.evalu(act_input, args.gpu_ids, modelfile, classifier, device, wholemodel=False)
        input_label = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        filename = 'mnist_{0:06d}_ori.png'.format(img_idx)
    elif args.DNN == 'cifarnet':
        modelfile = './generate_latent/pytorch_cifar/checkpoint/cifar_cnn.pt'
        output = cifar.evalu(act_input, args.gpu_ids, modelfile, classifier, device, wholemodel=False)
        _, input_label = output.max(1)
        filename = 'cifar_{0:06d}_ori.png'.format(img_idx)
    elif args.DNN == 'celebANet':
        modelfile = './generate_latent/celebA_cnn.pt'
        output = celebA.evalu(act_input, args.gpu_ids, modelfile, classifier, device, wholemodel=False)
        input_label = output >= 0.5
        filename = 'celebA_{0:06d}_ori.png'.format(img_idx)
    elif args.DNN == 'svhnNet':
        modelfile = './generate_latent/svhn_cnn.pt'
        output = svhn.evalu(act_input, args.gpu_ids, modelfile, classifier, device, wholemodel=False)
        _, input_label = output.max(1)
        filename = 'svhn_{0:06d}_ori.png'.format(img_idx)
    
    filename_images = os.path.join(img_dir, filename)
    if os.path.exists(filename_images):
        return 0,0,0
    input_img = GSN.mnist_untransform(input_img)
    input_img = torch.squeeze(input_img)
    if args.dataset == 'mnist':
        Image.fromarray(np.uint8(input_img.cpu().numpy()), mode='L').save(filename_images)
    elif args.dataset in ['cifar', 'celebA_post', 'SVHN']:
        Image.fromarray(np.uint8(input_img.cpu().numpy().transpose(1,2,0))).save(filename_images)
    
    if args.phase == 'test':
        g = generator.restore_model(args.iteration)
    eps = args.eps_init
    # if len(args.gpu_ids) != 0:
    #   eps = eps.cuda()
    success = False
    min_eps = float('inf')
    label_arr = np.array([])

    if args.direction_model == 'att_svm' and -1 in args.eps_list:
        #print((torch.squeeze(torch.matmul(eps * direction[1], direction[0].permute(0,2,1)), dim = -1)).device)
        concept = img_dir.split('/')[-2]
        if int(concept[0]) < int(concept[1]):
            concept1 = concept[0]
            concept2 = concept[1]
        else:
            concept1 = concept[1]
            concept2 = concept[0]
        model_path = os.path.join(args.cav_dir, '-{}-{}-{}'.format(str(concept1)+str(concept2), args.layer, 'svm_model_weight.pkl'))
#model_path = 'att_svm.pkl'
        att_svm = svm_attention(args.dim, model_path)
        eps = args.eps_init
        eps_step = args.eps_step
        tmp_ite_max_eps = args.ite_max_eps
        eps_list = [0]
        while tmp_ite_max_eps >= 0:
            fea_change = eps * (att_svm.get_attention(act_input) * torch.squeeze(direction[0], dim = 0).to(device))
        # print(fea_change.max())
            act_trans_update = act_input + fea_change

            g_z = generator.generate_from_feature(g, act_trans_update)

            if args.DNN == 'lenet':
                output = mnist.evalu(torch.unsqueeze(torch.unsqueeze(g_z,dim=0),dim=0), args.gpu_ids, modelfile, classifier, device)
                pred = output.max(1, keepdim=True)[1] # get the label of generated images                
            elif args.DNN == 'cifarnet':
                output = cifar.evalu(torch.unsqueeze(g_z/255.,dim=0), args.gpu_ids, modelfile, classifier, device)
                _, pred = output.max(1)
            elif args.DNN == 'celebANet':
                output = celebA.evalu(torch.unsqueeze(g_z,dim=0), args.gpu_ids, modelfile, classifier, device)
                pred = output >= 0.5
            elif args.DNN == 'svhnNet':
                output = svhn.evalu(torch.unsqueeze(g_z,dim=0), args.gpu_ids, modelfile, classifier, device)
                _, pred = output.max(1)
            pred = pred.cpu().numpy()[0]
            if pred != input_label or eps >= args.eps_max or tmp_ite_max_eps ==0:
              #or pred != img_dir.split('/')[-2][0]:
                print(eps, pred, targetlabel, input_label.cpu().numpy()[0], img_dir.split('/')[-2][0])
                eps_list = [eps]
                break
            else:
                tmp_ite_max_eps -= 1
                if eps >= 1.0:
                    eps = eps + eps_step*10
                else:
                    eps = eps + eps_step

#   for i in range(args.ite_max_eps):
    for i, eps in enumerate(eps_list):
        if eps not in eps_list:
            eps += args.eps_step # the default update step size is 0.05
            eps = round(eps, 6)
            continue
        if args.direction_model == 'att_svm':
            #print((torch.squeeze(torch.matmul(eps * direction[1], direction[0].permute(0,2,1)), dim = -1)).device)
            concept = img_dir.split('/')[-2]
            if int(concept[0]) < int(concept[1]):
                concept1 = concept[0]
                concept2 = concept[1]
            else:
                concept1 = concept[1]
                concept2 = concept[0]
            model_path = os.path.join(args.cav_dir, '-{}-{}-{}'.format(str(concept1)+str(concept2), args.layer, 'svm_model_weight.pkl'))
#model_path = 'att_svm.pkl'
            att_svm = svm_attention(args.dim, model_path)
            fea_change = eps * (att_svm.get_attention(act_input) * torch.squeeze(direction[0], dim = 0).to(device))
            # print(fea_change.max())
            #act_trans_update = act_input + fea_change
            act_trans_update = act_input + fea_change
        elif args.direction_model == 'max_dis_svm':
            act_trans_update = act_input + eps * torch.squeeze(direction[0], dim = 0)
        else:
            act_trans_update = act_input + eps * direction[0]
        # check whether the features have became another class (in other words, whether we have crossed the decision boundary)
        if args.DNN == 'lenet':
            output = mnist.evalu(act_trans_update, args.gpu_ids, modelfile, classifier, device, wholemodel=False)
            trans_label = output.max(1, keepdim=True)[1] # get the label of changed features
        elif args.DNN == 'cifarnet':
            output = cifar.evalu(act_trans_update, args.gpu_ids, modelfile, classifier, device, wholemodel=False)
            _, trans_label = output.max(1)
        elif args.DNN == 'celebANet':
            output = celebA.evalu(act_trans_update, args.gpu_ids, modelfile, classifier, device, wholemodel=False)
            trans_label = output >= 0.5
        elif args.DNN == 'svhnNet':
            output = svhn.evalu(act_trans_update, args.gpu_ids, modelfile, classifier, device, wholemodel=False)
            _, trans_label = output.max(1)

        g_z = generator.generate_from_feature(g, act_trans_update)

        if args.DNN == 'lenet':
            output = mnist.evalu(torch.unsqueeze(torch.unsqueeze(g_z,dim=0),dim=0), args.gpu_ids, modelfile, classifier, device)
            pred = output.max(1, keepdim=True)[1] # get the label of generated images
            filename = 'mnist_{0:02d}_{1}_img{2}_fea{3}_{4:02f}.png'.format(img_idx, i, int(pred.cpu().numpy()), int(trans_label.cpu().numpy()), eps)
        elif args.DNN == 'cifarnet':
            output = cifar.evalu(torch.unsqueeze(g_z/255.,dim=0), args.gpu_ids, modelfile, classifier, device)
            _, pred = output.max(1)
            # print(pred, targetlabel)
            filename = 'cifar_{0:02d}_{1}_img{2}_fea{3}_{4:02f}.png'.format(img_idx, i, int(pred.cpu().numpy()), int(trans_label.cpu().numpy()), eps)
        elif args.DNN == 'celebANet':
            output = celebA.evalu(torch.unsqueeze(g_z,dim=0), args.gpu_ids, modelfile, classifier, device)
            pred = output >= 0.5
            filename = 'celebA_{0:02d}_{1}_img{2}_fea{3}_{4:02f}.png'.format(img_idx, i, int(pred.cpu().numpy()), int(trans_label.cpu().numpy()), eps)
        elif args.DNN == 'svhnNet':
            output = svhn.evalu(torch.unsqueeze(g_z,dim=0), args.gpu_ids, modelfile, classifier, device)
            _, pred = output.max(1)
            filename = 'svhn_{0:02d}_{1}_img{2}_fea{3}_{4:02f}.png'.format(img_idx, i, int(pred.cpu().numpy()), int(trans_label.cpu().numpy()), eps)
        
        filename = os.path.join(img_dir, filename)
        if eps in eps_list and not os.path.exists(filename):
            if label_arr.shape[0] == 0:
                label_arr = np.array([img_idx, i, int(pred.cpu().numpy()), int(trans_label.cpu().numpy()), eps, targetlabel])
            else:
                label_arr = np.r_['0,2,1', label_arr, np.array([img_idx, i, int(pred.cpu().numpy()), int(trans_label.cpu().numpy()), eps, targetlabel])] # record the labels of changed features
            if args.DNN == 'lenet':
                Image.fromarray(np.uint8(g_z.detach().cpu().numpy()), mode='L').save(filename)
            elif args.DNN in ['cifarnet', 'celebANet', 'svhnNet']:
                Image.fromarray(np.uint8(g_z.detach().cpu().numpy().transpose(1,2,0))).save(filename)
                
        if eps >= args.eps_max:
            break
        if eps > 0 and pred == targetlabel: # the output label is equal to targetlabel
            success = True
            if min_eps == float('inf'):
                min_eps = eps
            mylogger.debug("After {} iteration, the minimum of epsilon for image {} is {}".format(i,img_idx,eps))
        eps += args.eps_step # the default update step size is 0.05
        eps = round(eps, 6)
            
    if eps >= args.eps_max:
        mylogger.debug("After {} iteration, the value of epsilon ({}) for image {} is bigger than the maximum of epsilon ({})".format(i,eps,img_idx, args.eps_max))
    if i == args.ite_max_eps:
        mylogger.debug("After {} iteration, the minimum of epsilon for image {} is {}".format(i,img_idx,eps))

    return min_eps, success, label_arr
    

if __name__ == "__main__":
    test_img = torch.load('./datasets/mnist/processed/test.pt')
    trans_index = []
    for i in range(0, len(test_img[1])):
        if test_img[1][i] == 3:
            trans_index.append(i)
    input()
