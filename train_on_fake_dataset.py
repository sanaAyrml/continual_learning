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
from Medical_train import train_model

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

from my_code.fake_dataset import fake_Heart

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
parser.add_argument('--fix_budget', default=False , type=bool,  \
    help='fix budget')
########################################
parser.add_argument('--mimic_score', default=False , type=bool,  \
    help='To mimic scores for cosine embedding')
parser.add_argument('--lw_ms', default=1, type=float, \
    help='loss weight for mimicking score')
########################################
#improved class incremental learning
parser.add_argument('--rs_ratio', default=0, type=float, \
    help='The ratio for resample')
parser.add_argument('--imprint_weights', default=False , type=bool, \
    help='Imprint the weights for novel classes')
parser.add_argument('--less_forget', default=False , type=bool,  \
    help='Less forgetful')
parser.add_argument('--lamda', default=5, type=float, \
    help='Lamda for LF')
parser.add_argument('--adapt_lamda', default=False , type=bool,  \
    help='Adaptively change lamda')
parser.add_argument('--mr_loss', default=False , type=bool,  \
    help='Margin ranking loss v1')
parser.add_argument('--amr_loss', default=False , type=bool,  \
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
trainset = get_heart_dataset(mode='train', used_labels = None,order = order_mine)
trainset_verifier = get_heart_dataset(mode='train', used_labels = None,order = order_mine)
evalset = get_heart_dataset(mode='valid', used_labels = None,order = order_mine)
testset = get_heart_dataset(mode='valid', used_labels = None,order = order_mine)


# Initialization
top1_acc_list_cumul = np.zeros((int(args.num_classes/args.nb_cl),3,args.nb_runs))
top1_acc_list_ori   = np.zeros((int(args.num_classes/args.nb_cl),3,args.nb_runs))

X_train_total = np.array(trainset.data)
Y_train_total = np.array(trainset.targets)
X_valid_total = np.array(testset.data)
Y_valid_total = np.array(testset.targets)

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
    X_valid_cumuls    = []
    X_protoset_cumuls = []
    X_train_cumuls    = []
    Y_valid_cumuls    = []
    Y_protoset_cumuls = []
    Y_train_cumuls    = []
    alpha_dr_herding  = np.zeros((int(args.num_classes/args.nb_cl),dictionary_size,args.nb_cl),np.float32)


    # The following contains all the training samples of the different classes
    # because we want to compare our method with the theoretical case where all the training samples are stored
    prototypes = [[] for i in range(args.num_classes)]
    for orde in range(args.num_classes):
        prototypes[orde] = X_train_total[np.where(Y_train_total==order[orde])]
        print(len(prototypes[orde]))
    for orde in range(args.num_classes):
        print(orde,len(X_valid_total[np.where(Y_valid_total==order[orde])]))
    print("class sample sizes for train:")
    for orde in range(args.num_classes):
        print(orde,len(prototypes[orde]))
    prototypes = np.array(prototypes,dtype=object,)
    print(prototypes.shape)

    if args.save_samples:
        print('save samples')
        view_classes = trainset.view_classes
        print(view_classes)
        w  = [[] for i in range(len(view_classes))]
        for i,label in enumerate(trainset.data_dict['view_label']):
            w[label].append(i)
        for i in w:
            print(len(i))
        if not os.path.exists(args.save_samples_dir):
            os.makedirs(args.save_samples_dir)
        for i in w[0:14]:
            integrated_image = torch.zeros(trainset.__getitem__(i[0])[0].view(1,1,224,224).shape)
            for j in range(len(i)):
                integrated_image += trainset.__getitem__(i[j])[0].view(1,1,224,224)
                # print(trainset.__getitem__(i[j])[0].view(1,1,224,224).max(),
                #       trainset.__getitem__(i[j])[0].view(1,1,224,224).min(),
                #       torch.std(trainset.__getitem__(i[j])[0].view(1,1,224,224), unbiased=False),
                #       torch.mean(trainset.__getitem__(i[j])[0].view(1,1,224,224)))
                
            vutils.save_image(integrated_image/len(i),
                              args.save_samples_dir+'/label_'+str(trainset.__getitem__(i[j])[1])+"_"+"integrated"+"_"+view_classes[trainset.__getitem__(i[j])[1]]+'.png',
                              normalize=False, scale_each=True, nrow=int(1))

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
            
        
        wandb.run.name = '{}_run_{}_iteration_{}_model.pth'.format(main_ckp_prefix, iteration_total, iteration)
        wandb.run.save()

        if args.verifier:
            print("making verifier model")
            if args.small_model:
                if args.type_of_verifier == 'Resnet' :
                    print("small resnet verifier model with layers 3 4 6 ")
                    verifier_model = ResNet(ResidualBlock, [3, 4, 6],input_dim=args.input_dim,num_classes=iteration*args.nb_cl+1).to(device)
                elif args.type_of_verifier == 'VGG' :
                    print("VGG11 verifier model")
                    verifier_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).to(device)
                    print(verifier_model)
                    verifier_model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    verifier_model.classifier[6] = nn.Linear(in_features=4096, out_features=iteration*args.nb_cl+1, bias=True) 
                    print(verifier_model)
                    verifier_model = verifier_model.to(device)
            else:
                if args.type_of_verifier == 'Resnet' :
                    print("resnet verifier model with layers 3 4 6 3")
                    verifier_model = ResNet(ResidualBlock, [3, 4, 6, 3],input_dim=args.input_dim,num_classes=iteration*args.nb_cl+1).to(device)
                elif args.type_of_verifier == 'VGG' :
                    print("VGG11 verifier model")
                    verifier_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).to(device)
                    print(verifier_model)
                    verifier_model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    verifier_model.classifier[6] = nn.Linear(in_features=4096, out_features=iteration*args.nb_cl+1, bias=True) 
                    print(verifier_model)
                    verifier_model = verifier_model.to(device)
        
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



        cur_lamda = args.lamda
        if iteration > start_iter and args.less_forget:
            print("###############################")
            print("Lamda for less forget is set to ", cur_lamda)
            print("###############################")

        # Prepare the training data for the current batch of classes
        actual_cl        = order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)]
        print("classes to be trained:",last_iter*args.nb_cl,"-",(iteration+1)*args.nb_cl)
        indices_train_10 = np.array([i in order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_train_total])
        indices_test_10  = np.array([i in order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_valid_total])

        X_train          = X_train_total[indices_train_10]
        X_valid          = X_valid_total[indices_test_10]
        print("len data to be trained ==> train:", len(X_train),"  validation:",len(X_valid))
        if iteration == start_iter:
            X_valid_cumuls.append(X_valid)
            X_train_cumuls.append(X_train)
        X_valid_cumul    = np.concatenate(X_valid_cumuls)
        X_train_cumul    = np.concatenate(X_train_cumuls)
        print("len total data seen till this phase ==> train:", len(X_train_cumul),"  validation:",len(X_valid_cumul))

        Y_train          = Y_train_total[indices_train_10]
        Y_valid          = Y_valid_total[indices_test_10]
        if iteration == start_iter:
            Y_valid_cumuls.append(Y_valid)
            Y_train_cumuls.append(Y_train)
        Y_valid_cumul    = np.concatenate(Y_valid_cumuls)
        Y_train_cumul    = np.concatenate(Y_train_cumuls)

        # Add the stored exemplars to the training data
        if iteration == start_iter:
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            pass
            ##TODO
            

        # Launch the training loop
        print('Batch of classes number {0} arrives ...'.format(iteration+1))
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_train_cumul = np.array([order_list.index(i) for i in Y_train_cumul])
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])

        
        
        
        print("making original dataloader")
        if iteration == start_iter:
            trainset.data = X_train
            trainset.targets = map_Y_train
        if iteration > start_iter:
            
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                shuffle=True, num_workers=2)           
        else:
            if args.Plax:
                sampler_list = []
                for yi in trainset.targets:
                    sampler_list.append(trainset.view_label_weights[yi])
                train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sampler_list, len(sampler_list))
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                shuffle=False, num_workers=2,sampler=train_sampler)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                shuffle=True, num_workers=2)
            
        testset.data = X_valid_cumul
        testset.targets = map_Y_valid_cumul
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
            shuffle=False, num_workers=2)
        
        print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))
                
        ##############################################################
        
                    
        ckp_name = './sweep_checkpoint/{}_run_{}_iteration_{}_model.pth'.format(main_ckp_prefix, iteration_total, iteration)
        print('check point address of original model', ckp_name)

        if args.resume and os.path.exists(ckp_name):
            print("###############################")
            print("Loading original models weights from checkpoint")
            tg_model.load_state_dict(torch.load(ckp_name)['model_state_dict'])
            model_loaded = True
            print("###############################")
            
        else:
            ###############################

            tg_params = tg_model.parameters()
                
            ###############################
            tg_model = tg_model.to(device)
            tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
            tg_lr_scheduler =  ReduceLROnPlateau(tg_optimizer, factor=lr_factor, patience=lr_patience, threshold=lr_threshold)
            #############################
            
            tg_model = train_model(trainloader, testloader, tg_model,ref_model,ckp_name,main_ckp_prefix,
                        tg_optimizer,tg_lr_scheduler,
                        args,iteration_total,iteration,start_iter,cur_lamda,device,mode = "original",train_mode = args.mode)

                        
        ### training verifier               
        if args.verifier:
            print("making verifier dataloader")
            trainset_verifier.data = X_train_cumul
            trainset_verifier.targets = map_Y_train_cumul
            if args.Plax:
                sampler_list = []
                for yi in trainset_verifier.targets:
                    sampler_list.append(trainset_verifier.view_label_weights[yi])
                train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sampler_list, len(sampler_list))
                trainloader = torch.utils.data.DataLoader(trainset_verifier, batch_size=train_batch_size,
                                shuffle=False, num_workers=2,sampler=train_sampler)
            trainloader_verifier = torch.utils.data.DataLoader(trainset_verifier, batch_size=train_batch_size,
                shuffle=True, num_workers=2)
            verifier_ckp_name = './sweep_checkpoint/{}_{}_run_{}_iteration_{}_model.pth'.format(verifier_ckp_prefix,'verifier', iteration_total, iteration)
            print("chechpoint verifier address", verifier_ckp_name)

            if args.resume and os.path.exists(verifier_ckp_name):
                print("###############################")
                print("Loading verifier models weights from checkpoint")
                verifier_model.load_state_dict(torch.load(verifier_ckp_name)['model_state_dict'])
                print("###############################")

            else:
                verifier_params = verifier_model.parameters()
                ###############################
                verifier_model = verifier_model.to(device)
                verifier_optimizer = optim.SGD(verifier_params, lr=args.lr_verifier, momentum=custom_momentum, weight_decay=custom_weight_decay)
                verifier_lr_scheduler =  ReduceLROnPlateau(verifier_optimizer, factor=lr_factor, patience=lr_patience, threshold=lr_threshold)
                #############################
                train_model(trainloader_verifier, testloader, verifier_model,None,verifier_ckp_name,verifier_ckp_prefix,
                            verifier_optimizer,verifier_lr_scheduler,
                            args,iteration_total,iteration,start_iter,cur_lamda,device,mode = "verifier",train_mode = args.mode)
                    
                    
        ### Exemplars
        if args.fix_budget:
            print("fixing budget")
            nb_protos_cl = int(np.ceil(args.nb_protos*100./args.nb_cl/(iteration+1)))
        else:
            nb_protos_cl = args.nb_protos
            
        nn.Sequential(*list(tg_model.children())[:-1])
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        num_features = tg_model.fc.in_features
        
        # Herding
        if args.sampler_type == 'paper1' and args.add_sampler:
            print('Updating exemplar set...')
            for iter_dico in range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl):
                # Possible exemplars in the feature space and projected on the L2 sphere
                evalset.data = prototypes[iter_dico]
                evalset.targets = np.zeros(len(evalset.data)) #zero labels
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=2)
                num_samples = len(evalset.data)          
                mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features,device)
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)

                # Herding procedure : ranking of the potential exemplars
                mu  = np.mean(D,axis=1)
                index1 = int(iter_dico/args.nb_cl)
                index2 = iter_dico % args.nb_cl
                alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
                w_t = mu
                iter_herding     = 0
                iter_herding_eff = 0
                while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                    tmp_t   = np.dot(w_t,D)
                    ind_max = np.argmax(tmp_t)
                    iter_herding_eff += 1
                    if alpha_dr_herding[index1,ind_max,index2] == 0:
                        alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                        iter_herding += 1
                    w_t = w_t+mu-D[:,ind_max]
########################################################################################################################################################################################################################################################################################################

        # Prepare the protoset
        X_protoset_cumuls = []
        Y_protoset_cumuls = []
        if args.sampler_type == 'paper2' and args.add_data:
            if iteration >= start_iter:
                print("generation")
                main_ckp_prefix = main_ckp_prefix + '_bsg_' + str(args.bs) + '_lrg_' + str(args.generation_lr) + '_rfg_' + str(args.r_feature) + '_tv_l2g_' + str(args.tv_l2) + '_l2g_' + str(args.l2) + '_beta2_' +str(args.beta_2)

                wandb.run.name = '{}_run_{}_iteration_{}_model.pth'.format(main_ckp_prefix, iteration_total, iteration)
                wandb.run.save()
                #trained tg_model
                if args.generate_more:
                    generation_path = "generation_more_optimized"
                else:
                    generation_path = "generations"
                # final images will be stored here:
                adi_data_path = './sweep_checkpoint/final_images/{}_run_{}_iteration_{}_model.pth'.format(main_ckp_prefix, iteration_total, iteration)
                
                # temporal data and generations will be stored here
                exp_name = './sweep_checkpoint/{}/{}_run_{}_iteration_{}_model.pth'.format(generation_path,main_ckp_prefix, iteration_total, iteration)


                generated_batch_add = './sweep_checkpoint/{}/1_{}_gerated_data_run_{}_iteration_{}.pkl'.format(generation_path,main_ckp_prefix, iteration_total, iteration)
                generated_target_add = './sweep_checkpoint/{}/1_{}_gerated_label_run_{}_iteration_{}.pkl'.format(generation_path,main_ckp_prefix, iteration_total, iteration)

                if args.add_sampler: 
                    args.iterations = 2000
                    args.start_noise = True
                    # args.detach_student = False

                    args.resolution = 224
                    bs = args.bs
                    jitter = 30

                    parameters = dict()
                    parameters["resolution"] = args.resolution
                    parameters["random_label"] = False
                    parameters["start_noise"] = True
                    parameters["detach_student"] = False
                    parameters["do_flip"] = True

                    parameters["do_flip"] = args.do_flip
                    parameters["random_label"] = args.random_label
                    parameters["store_best_images"] = args.store_best_images

                    criterion = nn.CrossEntropyLoss()

                    coefficients = dict()
                    coefficients["r_feature"] = args.r_feature
                    coefficients["first_bn_multiplier"] = args.first_bn_multiplier
                    coefficients["tv_l1"] = args.tv_l1
                    coefficients["tv_l2"] = args.tv_l2
                    coefficients["l2"] = args.l2
                    coefficients["lr"] = args.generation_lr
                    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
                    coefficients["adi_scale"] = args.adi_scale

                    network_output_function = lambda x: x

                    # check accuracy of verifier
                    if args.verifier:
                        hook_for_display = lambda x,y: validate_one(x, y, verifier_model)
                        hook_for_self_eval = lambda x,y: validate_one(x, y, tg_model)
                    else:
                        hook_for_display = None
                    print("labels",min(map_Y_train),max(map_Y_train))
                    DeepInversionEngine = DeepInversionClass(net_teacher=tg_model,
                                                             final_data_path=adi_data_path,
                                                             path=exp_name,
                                                             parameters=parameters,
                                                             setting_id=args.setting_id,
                                                             bs = bs,
                                                             use_fp16 = args.fp16,
                                                             jitter = jitter,
                                                             criterion=criterion,
                                                             coefficients = coefficients,
                                                             network_output_function = network_output_function,
                                                             hook_for_display = hook_for_display,
                                                             hook_for_self_eval = hook_for_self_eval,
                                                             device = device,
                                                             target_classes_min = 0,
                                                             target_classes_max = max(map_Y_train),
                                                             mean_image_dir =args.save_samples_dir,
                                                            order_mine = order_mine)




                    print("number of generated batch loops",int((len(trainset.targets)/10)/bs))
                    
                    if args.generate_more:
                        if args.nb_generation == 0:
                            number_of_batches = int((len(trainset.targets)/10)/bs)
                        else:
                            number_of_batches = args.nb_generation
                    else:
                        number_of_batches = 1
                    for j in range(number_of_batches):
                        generated , targets = DeepInversionEngine.generate_batch(net_student=None,use_mean_initialization = args.use_mean_initialization, beta_2 = args.beta_2)

                        X_protoset_cumuls.append(generated)
                        Y_protoset_cumuls.append(targets)
                        with open(generated_batch_add,'wb') as f:
                            pickle.dump(X_protoset_cumuls, f)
                        with open(generated_target_add,'wb') as f:
                            pickle.dump(Y_protoset_cumuls, f)
                else:
                    if os.path.exists(generated_batch_add):
                        print("read previouse generated data")
                        with open(generated_batch_add,'rb') as f:
                            X_protoset_cumuls = pickle.load(f)
                        with open(generated_target_add,'rb') as f:
                            Y_protoset_cumuls = pickle.load(f)
                        trainset = fake_Heart(generated_batch_add, generated_target_add , transform=None, used_labels = None,order= None)
                    else:
                        print("no new generated data for this phase")
        
        
        
# ########################################################################################################################################################################################################################################################################################################        

        # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
        if args.sampler_type == 'paper1' and args.add_sampler:
            print('Computing mean-of_exemplars and theoretical mean...')
            class_means = np.zeros((num_features, args.num_classes, 2))
            for iteration2 in range(iteration+1):
                for iter_dico in range(args.nb_cl):
                    current_cl = order[range(iteration2*args.nb_cl,(iteration2+1)*args.nb_cl)]

                    # Collect data in the feature space for each class
                    evalset.data = prototypes[iteration2*args.nb_cl+iter_dico]
                    evalset.targets = np.zeros(len(evalset.data)) #zero labels
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                        shuffle=False, num_workers=2)
                    num_samples = len(evalset.data)
                    mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features,device)
                    D = mapped_prototypes.T
                    D = D/np.linalg.norm(D,axis=0)
                    # Flipped version also
                    evalset.data = prototypes[iteration2*args.nb_cl+iter_dico]
                    evalset.flipped_version = True
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                        shuffle=False, num_workers=2)
                    mapped_prototypes2 = compute_features(tg_feature_model, evalloader, num_samples, num_features,device)
                    D2 = mapped_prototypes2.T
                    D2 = D2/np.linalg.norm(D2,axis=0)
                    evalset.flipped_version = False

                    # iCaRL
                    alph = alpha_dr_herding[iteration2,:,iter_dico]
                    alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                    # print(alph,np.where(alph==1),len(prototypes[iteration2*args.nb_cl+iter_dico]))
                    X_protoset_cumuls.append(prototypes[iteration2*args.nb_cl+iter_dico][np.where(alph==1)[0]])
                    Y_protoset_cumuls.append(order[iteration2*args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                    alph = alph/np.sum(alph)
                    board = D.shape[1]
                    class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph[0:board])+np.dot(D2,alph[0:board]))/2
                    class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])

                    # Normal NCM
                    alph = np.ones(dictionary_size)/dictionary_size
                    class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph[0:board])+np.dot(D2,alph[0:board]))/2
                    class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])

            torch.save(class_means, \
                './sweep_checkpoint/{}_run_{}_iteration_{}_class_means.pth'.format(main_ckp_prefix,iteration_total, iteration))

            current_means = class_means[:, order[range(0,(iteration+1)*args.nb_cl)]]
        ##############################################################
        # Calculate validation error of model on the first nb_cl classes:
        if args.validate:
            if iteration > start_iter:
                map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
                print('Computing accuracy on the original batch of classes...')
                evalset.data = X_valid_ori
                evalset.targets = map_Y_valid_ori
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                        shuffle=False, num_workers=2)
                ori_acc = compute_accuracy(tg_model, tg_feature_model, None, evalloader,device=device)
                top1_acc_list_ori[iteration, :, iteration_total] = np.array(ori_acc).T
                ##############################################################
                # Calculate validation error of model on the cumul of classes:
                map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
                print('Computing cumulative accuracy...')
                evalset.data = X_valid_cumul
                evalset.targets = map_Y_valid_cumul
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                        shuffle=False, num_workers=2)        
                cumul_acc = compute_accuracy(tg_model, tg_feature_model, None, evalloader,device=device)
                top1_acc_list_cumul[iteration, :, iteration_total] = np.array(cumul_acc).T
                ##############################################################
                # Calculate confusion matrix
                print('Computing confusion matrix...')
                cm = compute_confusion_matrix(tg_model, tg_feature_model, None, evalloader,device=device)
                cm_name = './sweep_checkpoint/{}_run_{}_iteration_{}_confusion_matrix.pth'.format(main_ckp_prefix,iteration_total, iteration)
                with open(cm_name, 'wb') as f:
                    pickle.dump(cm, f) #for reading with Python 2
            ##############################################################   

    # Final save of the data
    # torch.save(top1_acc_list_ori, \
    #     './sweep_checkpoint/{}_run_{}_top1_acc_list_ori.pth'.format(main_ckp_prefix, iteration_total))
    # torch.save(top1_acc_list_cumul, \
    #     './sweep_checkpoint/{}_run_{}_top1_acc_list_cumul.pth'.format(main_ckp_prefix, iteration_total))







