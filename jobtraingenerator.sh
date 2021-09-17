#!/bin/bash

python main.py --DNN lenet --gpu_ids 0 --batch 256 --iteration 2 --epoch 0 --phase train --v False --dim 500 --logfile './mnist_gene.log' --nb_channels_first_layer 50 --dataset mnist --layer 3 --info 'training generator' --generator 'autoencoder' --exp_att 'mnist_500_norm2_ae' --num_classes 10 --restore_file='' --num_workers 1 --percent 1.0 
