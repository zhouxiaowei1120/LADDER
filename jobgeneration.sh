#!/bin/bash

python main.py --DNN lenet --gpu_ids 0 --batch 256 --iteration 1000 --epoch 1000 --phase test --v False --dim 500 --logfile './mnist_gene.log' --nb_channels_first_layer 50 --dataset mnist --layer 3  --generator 'autoencoder' --img_num 450 --direction_type 'cav' --exp_att 'mnist_500_norm2_ae' --num_classes 10 --restore_file='experiments/pre-trained/mnist/epoch_1000.pth' --num_workers 1 --percent 1.0 --eps_max 26 --eps_step 0.1 --ite_max_eps 10 --eps_init 0.5 --eps_list 0.5 2.0 5.0 10.0 12.0 15.0 18.0 20.0 25.0
