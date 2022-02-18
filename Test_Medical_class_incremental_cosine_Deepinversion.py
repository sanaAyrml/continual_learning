#!/usr/bin/env python
# coding=utf-8



from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import sys
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
import math

import modified_resnet_cifar
import modified_linear
import utils_pytorch
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.compute_confusion_matrix import compute_confusion_matrix
from utils_incremental.incremental_train_and_eval import incremental_train_and_eval
from utils_incremental.incremental_train_and_eval_MS import incremental_train_and_eval_MS
from utils_incremental.incremental_train_and_eval_LF import incremental_train_and_eval_LF
from utils_incremental.incremental_train_and_eval_MR_LF import incremental_train_and_eval_MR_LF
from utils_incremental.incremental_train_and_eval_AMR_LF import incremental_train_and_eval_AMR_LF
from run_handler import train_model
from run_handler import test_model

import matplotlib.pyplot as plt
import argparse
import torch
from torch import distributed, nn
import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import torch.cuda.amp as amp
import os
from utils.utils import load_model_pytorch, distributed_is_initialized
from medical_deepinversion import DeepInversionClass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from my_code.data_loader import get_heart_dataset

import wandb

def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        model.eval()
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 1))

    print("Verifier accuracy: ", prec1.item())
    return prec1.item()


######### Modifiable Settings ##########
parser = argparse.ArgumentParser()

parser.add_argument('--directory', default='./medical_checkpoint/', type=str, \
                    help='Checkpoint directory')

parser.add_argument('--ckp_prefix', default='', type=str, \
                    help='Checkpoint prefix')
parser.add_argument('--ckp_prefix_verifier', default='', type=str, \
                    help='Checkpoint prefix verifier')
parser.add_argument('--num_classes', default=14, type=int)
parser.add_argument('--nb_cl_fg', default=2, type=int, \
                    help='the number of classes in first group')
parser.add_argument('--nb_cl', default=1, type=int, \
                    help='Classes per group')
parser.add_argument('--nb_phases', default=2, type=int, \
                    help='the number of phases')
parser.add_argument('--nb_protos', default=20, type=int, \
                    help='Number of prototypes per class at the end')
parser.add_argument('--nb_runs', default=1, type=int, \
                    help='Number of runs (random ordering of classes at each run)')

parser.add_argument('--epochs', default=100, type=int, \
                    help='Epochs')

parser.add_argument('--T', default=2, type=float, \
                    help='Temporature for distialltion')
parser.add_argument('--beta', default=0.25, type=float, \
                    help='Beta for distialltion')
parser.add_argument('--resume', default=False , type=bool, \
                    help='resume from checkpoint')
parser.add_argument('--fix_budget', default=False , type=bool, \
                    help='fix budget')
########################################
parser.add_argument('--mimic_score', default=False , type=bool, \
                    help='To mimic scores for cosine embedding')
parser.add_argument('--lw_ms', default=1, type=float, \
                    help='loss weight for mimicking score')
########################################
#improved class incremental learning
parser.add_argument('--rs_ratio', default=0, type=float, \
                    help='The ratio for resample')
parser.add_argument('--imprint_weights', default=False , type=bool, \
                    help='Imprint the weights for novel classes')
parser.add_argument('--less_forget', default=False , type=bool, \
                    help='Less forgetful')
parser.add_argument('--lamda', default=5, type=float, \
                    help='Lamda for LF')
parser.add_argument('--adapt_lamda', default=False , type=bool, \
                    help='Adaptively change lamda')
parser.add_argument('--mr_loss', default=False , type=bool, \
                    help='Margin ranking loss v1')
parser.add_argument('--amr_loss', default=False , type=bool, \
                    help='Margin ranking loss v2')
parser.add_argument('--dist', default=0.5, type=float, \
                    help='Dist for MarginRankingLoss')
parser.add_argument('--K', default=1, type=int, \
                    help='K for MarginRankingLoss')
parser.add_argument('--lw_mr', default=1, type=float, \
                    help='loss weight for margin ranking loss')
parser.add_argument('--lr', type=float, default=0.2,
                    help='learning rate for optimization')
parser.add_argument('--lr_verifier', type=float, default=0.2,
                    help='learning rate for optimization verifier')
parser.add_argument('--batch_size_1', type=int, default=64,
                    help='batch size for training model')

########################################


parser.add_argument('--epochs_generat', default=4000, type=int,
                    help='number of epochs')
parser.add_argument('--setting_id', default=1, type=int,
                    help='settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations')
parser.add_argument('--bs', default=64, type=int,
                    help='batch size for generation')
parser.add_argument('--jitter', default=30, type=int,
                    help='batch size')
parser.add_argument('--adi_scale', type=float, default=0.0,
                    help='Coefficient for Adaptive Deep Inversion')

parser.add_argument('--fp16', default=False , type=bool,
                    help='use FP16 for optimization')

parser.add_argument('--do_flip', default=False , type=bool,
                    help='apply flip during model inversion')
parser.add_argument('--random_label', default=False , type=bool,
                    help='generate random label for optimization')
parser.add_argument('--r_feature', type=float, default=0.05,
                    help='coefficient for feature distribution regularization')
parser.add_argument('--first_bn_multiplier', type=float, default=10.,
                    help='additional multiplier on first bn layer of R_feature')
parser.add_argument('--tv_l1', type=float, default=0.0,
                    help='coefficient for total variation L1 loss')
parser.add_argument('--tv_l2', type=float, default=0.0001,
                    help='coefficient for total variation L2 loss')
parser.add_argument('--l2', type=float, default=0.00001,
                    help='l2 loss on the image')
parser.add_argument('--main_loss_multiplier', type=float, default=1.0,
                    help='coefficient for the main loss in optimization')
parser.add_argument('--store_best_images', default=False , type=bool,
                    help='save best images as separate files')
parser.add_argument('--generation_lr', type=float, default=0.2,
                    help='learning rate for optimization')



parser.add_argument('--verifier', default=False , type=bool,
                    help='evaluate batch with another model')
#####################################################################################################
parser.add_argument('--add_sampler',  default=False , type=bool,
                    help='enable deep inversion part')

parser.add_argument('--add_data',  default=False , type=bool,
                    help='enable deep to add generated data')

parser.add_argument('--input_dim', type=int, default=1, help='input dimension')

parser.add_argument('--cosine_normalization', default=False , type=bool,
                    help='change laste layer of networks')
parser.add_argument('--save_samples', default=False , type=bool,
                    help='save some samples')
parser.add_argument('--save_samples_dir', type=str, default='./saved_Sample',
                    help='place to save samples')
parser.add_argument('--small_model', default=False , type=bool,
                    help='use model with 3 layers')
parser.add_argument('--big_model', default=False , type=bool,
                    help='use model with resnet 50 layers')

parser.add_argument('--validate', default=False , type=bool,
                    help='run the validate part of network')

parser.add_argument('--sampler_type', default='paper1' , type=str,
                    help='use which sampler strategy')

parser.add_argument('--mode', default='paper1' , type=str,
                    help='use which sampler strategy')

parser.add_argument('--use_mean_initialization', default=False , type=bool,
                    help='use mean of classes to initialize vectors')

parser.add_argument('--type_of_verifier', default='Resnet' , type=str,
                    help='model type of verifier')

parser.add_argument('--beta_2', default=0.9 , type=float,
                    help='beta 2 for adam optimizer in generating')

parser.add_argument('--cuda_number', default=0.9 , type=str,
                    help='which cuda use to train')

parser.add_argument('--generate_more', default=False , type=bool,
                    help='generate more batches of data')

parser.add_argument('--Plax', default=False , type=bool,
                    help='new classes to train')

parser.add_argument('--nb_generation', default=0 , type=int,
                    help='number of batch to train')

parser.add_argument('--alpha_3', default=1 , type=int,
                    help='number of batch to train')


parser.add_argument('--CL', default=0 , type=int,
                    help='if none zero enable contrastive learning')

parser.add_argument('--ro', default=0.9 , type=float,
                    help='ro for updating centroids')









parser.add_argument('--random_seed', default=1993, type=int, \
                    help='random seed')

args = parser.parse_args()
if args.cosine_normalization:
    if not args.small_model:
        from Medical_predictor_model_modified import ResNet,ResidualBlock
    else:
        from Medical_predictor_model_3_layers_modified import ResNet,ResidualBlock
else:
    if args.small_model:
        from Medical_predictor_model_3_layers import ResNet,ResidualBlock
    else:
        from Medical_predictor_model import ResNet,ResidualBlock

os.environ["WANDB_API_KEY"] = 'f87c7a64e4a4c89c4f1afc42620ac211ceb0f926'

wandb.init(project="continual_learning", entity="sanaayr",config=args)

########################################
assert(args.nb_cl_fg % args.nb_cl == 0)
assert(args.nb_cl_fg >= args.nb_cl)
train_batch_size       = args.batch_size_1            # Batch size for train
test_batch_size        = args.batch_size_1            # Batch size for test
eval_batch_size        = args.batch_size_1            # Batch size for eval
base_lr                = args.lr            # Initial learning rate
lr_factor              = 0.3            # Learning rate decrease factor
lr_patience            = 5
lr_threshold           = 0.0001
custom_weight_decay    = 5e-4           # Weight Decay
custom_momentum        = 0.9            # Momentum
args.epochs = 50

if not os.path.exists('sweep_checkpoint/'+args.directory):
    os.makedirs('sweep_checkpoint/'+args.directory)

main_ckp_prefix       = '{}/{}_nb_cl_fg_{}_nb_cl_{}_lr_{}_bs_{}'.format(args.directory,args.ckp_prefix,
                                                                        args.nb_cl_fg,
                                                                        args.nb_cl,
                                                                        args.lr,
                                                                        args.batch_size_1)

np.random.seed(args.random_seed)        # Fix the random seed
print(args)
if args.Plax:
    order_mine = [2,4,0,1,3,5,6,7,8,9,10,11,12,13,14,15]
else:
    order_mine = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

########################################
print("cuda:"+args.cuda_number)
device = torch.device("cuda:"+args.cuda_number if torch.cuda.is_available() else "cpu")

testset = get_heart_dataset(mode='test', used_labels = None,order = order_mine)


# Initialization
X_test_total = np.array(testset.data)
Y_test_total = np.array(testset.targets)

# Launch the different runs
for iteration_total in range(args.nb_runs):
    # Select the order for the class learning
    order_name = "./sweep_checkpoint/seed_{}_rder_run_{}.pkl".format(args.random_seed, iteration_total)
    print("Order name:{}".format(order_name))
    if os.path.exists(order_name):
        print("Loading orders")
        order = utils_pytorch.unpickle(order_name)
    else:
        print("Generating orders")
        order = np.arange(args.num_classes)
        # np.random.shuffle(order)
        utils_pytorch.savepickle(order, order_name)

    order_list = list(order)
    print("order_list: ",order_list)

    #     # Initialization of the variables for this run
    dictionary_size = 2000
    
    X_test_cumuls    = []
    Y_test_cumuls    = []

    alpha_dr_herding  = np.zeros((int(args.num_classes/args.nb_cl),dictionary_size,args.nb_cl),np.float32)


    # The following contains all the training samples of the different classes
    # because we want to compare our method with the theoretical case where all the training samples are stored
    prototypes = [[] for i in range(args.num_classes)]
    for orde in range(args.num_classes):
        prototypes[orde] = X_test_total[np.where(Y_test_total==order[orde])]
        print(len(prototypes[orde]))

    print("class sample sizes for train:")
    for orde in range(args.num_classes):
        print(orde,len(prototypes[orde]))
    prototypes = np.array(prototypes,dtype=object,)
    print(prototypes.shape)


    start_iter = int(args.nb_cl_fg/args.nb_cl)-1
    print(start_iter)
    for iteration in range(start_iter, min(args.nb_phases+start_iter,int(args.num_classes/args.nb_cl))):
        print(iteration)
        if iteration == start_iter:
            if args.mode == 'vanilla':
                main_ckp_prefix        = '{}/{}/{}_nb_cl_fg_{}_nb_cl_{}_lr_{}_bs_{}'.format(args.directory,args.mode,args.ckp_prefix,
                                                                                            args.nb_cl_fg,
                                                                                            args.nb_cl,
                                                                                            args.lr,
                                                                                            args.batch_size_1)
                verifier_ckp_prefix        = '{}/{}/{}_nb_cl_fg_{}_nb_cl_{}_lr_{}_bs_{}'.format(args.directory,args.mode,args.ckp_prefix_verifier,
                                                                                                args.nb_cl_fg,
                                                                                                args.nb_cl,
                                                                                                args.lr_verifier,
                                                                                                args.batch_size_1)
            else:
                main_ckp_prefix        = '{}/{}_nb_cl_fg_{}_nb_cl_{}_lr_{}_bs_{}'.format(args.directory,args.ckp_prefix,
                                                                                         args.nb_cl_fg,
                                                                                         args.nb_cl,
                                                                                         args.lr,
                                                                                         args.batch_size_1)
                verifier_ckp_prefix        = '{}/{}_nb_cl_fg_{}_nb_cl_{}_lr_{}_bs_{}'.format(args.directory,
                                                                                             args.ckp_prefix_verifier,
                                                                                             args.nb_cl_fg,
                                                                                             args.nb_cl,
                                                                                             args.lr_verifier,
                                                                                             args.batch_size_1)
        else:
            if args.CL == 0:
                main_ckp_prefix        = '{}/{}/{}_nb_cl_fg_{}_nb_cl_{}_lr_{}_bs_{}'.format(args.directory,args.mode,args.ckp_prefix,
                                                                                            args.nb_cl_fg,
                                                                                            args.nb_cl,
                                                                                            args.lr,
                                                                                            args.batch_size_1)
            else:
                main_ckp_prefix        = '{}/{}/{}_nb_cl_fg_{}_nb_cl_{}_lr_{}_bs_{}'.format(args.directory,args.mode+"_CL",args.ckp_prefix,
                                                                                            args.nb_cl_fg,
                                                                                            args.nb_cl,
                                                                                            args.lr,
                                                                                            args.batch_size_1)
            verifier_ckp_prefix        = '{}/{}/{}_nb_cl_fg_{}_nb_cl_{}_lr_{}_bs_{}'.format(args.directory,args.mode,args.ckp_prefix_verifier,
                                                                                            args.nb_cl_fg,
                                                                                            args.nb_cl,
                                                                                            args.lr_verifier,
                                                                                            args.batch_size_1)
        

        #TODO
        wandb.run.name = '{}_run_{}_iteration_{}_model.pth'.format(main_ckp_prefix, iteration_total, iteration)
        wandb.run.save()


        #init model
        if iteration == start_iter:
            ############################################################
            last_iter = 0
            ############################################################
            print("making original model")
            if args.small_model:
                print("small resnet original model with layers 2 2 2 ")
                tg_model = ResNet(ResidualBlock, [2, 2, 2],input_dim=args.input_dim,num_classes=iteration*args.nb_cl+1).to(device)
            else:
                print("resnet original model with layers 2 2 2 2 ")
                tg_model = ResNet(ResidualBlock, [2, 2, 2, 2],input_dim=args.input_dim,num_classes=iteration*args.nb_cl+1).to(device)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            ref_model = None
        elif iteration == start_iter+1:
            ############################################################
            last_iter = iteration
            ############################################################
            #increment classes
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, args.nb_cl)
            new_fc.fc1.weight.data = tg_model.fc.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            lamda_mult = out_features*1.0 / args.nb_cl
        else:
            ############################################################
            last_iter = iteration
            ############################################################
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features1 = tg_model.fc.fc1.out_features
            out_features2 = tg_model.fc.fc2.out_features
            print("in_features:", in_features, "out_features1:", \
                  out_features1, "out_features2:", out_features2)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, args.nb_cl)
            new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            lamda_mult = (out_features1+out_features2)*1.0 / (args.nb_cl)
        tg_model.to(device)

        # Prepare the training data for the current batch of classes
        actual_cl        = order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)]
        print("classes to be trained:",last_iter*args.nb_cl,"-",(iteration+1)*args.nb_cl)
        indices_test_10 = np.array([i in order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_test_total])

        X_test          = X_test_total[indices_test_10]
     
        X_test_cumuls.append(X_test)
        
        X_test_cumul    = np.concatenate(X_test_cumuls)
        print("len data to be validate ==> test:", len(X_test))

        Y_test          = Y_test_total[indices_test_10]
        Y_test_cumuls.append(Y_test)
        
        Y_test_cumul    = np.concatenate(Y_test_cumuls)

        # Add the stored exemplars to the training data
        if iteration == start_iter:
            X_test_ori = X_test
            Y_test_ori = Y_test

        # Launch the training loop
        print('Batch of classes number {0} arrives ...'.format(iteration+1))
        map_Y_test = np.array([order_list.index(i) for i in Y_test])
        
        map_Y_test_cumul = np.array([order_list.index(i) for i in Y_test_cumul])


        ############################################################



        print("making original dataloader")
        testset.data = X_test_cumul
        testset.targets = map_Y_test_cumul

        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, \
                                                  shuffle=False, num_workers=2)
        

        print('Max and Min of test labels: {}, {}'.format(min(map_Y_test_cumul), max(map_Y_test_cumul)))

        ##############################################################


        #TODO
        if iteration > start_iter and args.mode == "paper_2" :
            main_ckp_prefix = main_ckp_prefix + '_bsg_' + str(args.bs) + '_lrg_' + str(args.generation_lr) + '_rfg_' + str(args.r_feature) + '_tv_l2g_' + str(args.tv_l2) + '_l2g_' + str(args.l2) + '_beta2_' +str(args.beta_2)  + '_alpha3_'+str(args.alpha_3) + '_dist_' + str(args.dist)


            wandb.run.name = '{}_run_{}_iteration_{}_model.pth'.format(main_ckp_prefix, iteration_total, iteration)
            wandb.run.save()
        ckp_name = './sweep_checkpoint/{}_run_{}_iteration_{}_model.pth'.format(main_ckp_prefix, iteration_total, iteration)
        print('check point address of original model', ckp_name)

        if args.resume and os.path.exists(ckp_name):
            print("###############################")
            print("Loading original models weights from checkpoint")
            tg_model.load_state_dict(torch.load(ckp_name)['model_state_dict'])
            model_loaded = True
            print("###############################")

    

            #############################

            test_model(testloader, tg_model,ckp_name,main_ckp_prefix,
                                   args,iteration_total,iteration,start_iter,device,mode = "original",train_mode = args.mode)



    # Final save of the data
    # torch.save(top1_acc_list_ori, \
    #     './sweep_checkpoint/{}_run_{}_top1_acc_list_ori.pth'.format(main_ckp_prefix, iteration_total))
    # torch.save(top1_acc_list_cumul, \
    #     './sweep_checkpoint/{}_run_{}_top1_acc_list_cumul.pth'.format(main_ckp_prefix, iteration_total))










