#!/bin/bash

# before running this, please set the --eps to the specific perturbation strength and set the path to the location of your generated adversarial examples, i.e. experiments/gsn_hf/mnist/mnist_500_norm2_ae/eps_diff/2021-09-17T19-18-30/att_svm/

echo 'ADV train on LAD'
python mnist_defence.py --gpu_ids 0 --eps 0.5  --adversarial_samples 'path/to/your/adversarial/examples'  --i 'ADv. train leNet using LAD data generated from training dataset' --exp_dir experiments/mnist_defence/our_model_LAD/ --exp_name 'LADTrainData' --img_num -1 
echo 'Finished!'
