# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

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
from my_code.Medical_predictor_model import ResNet,ResidualBlock
from utils.utils import load_model_pytorch, distributed_is_initialized

random.seed(0)


def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
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


def run(args):
    torch.manual_seed(args.local_rank)
    device = torch.device('cuda:1' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    # print(torch.cuda.current_device(),device)

    if args.arch_name == "resnet18":
        net = ResNet(ResidualBlock, [2, 2, 2],input_dim = 1,num_classes=2)
        net.load_state_dict(torch.load(args.arch_weights_dir)['model_state_dict'])
    

    net = net.to(device)

    use_fp16 = args.fp16
    if use_fp16:
        net, _ = amp.initialize(net, [], opt_level="O2")

    print('==> Resuming from checkpoint..')

    net.eval()

    # reserved to compute test accuracy on generated images by different networks
    net_verifier = None
    if args.verifier and args.adi_scale == 0:
        # if multiple GPUs are used then we can change code to load different verifiers to different GPUs
        if args.local_rank == 0:
            if args.verifier_arch == "resnet34":
                net_verifier = ResNet(ResidualBlock, [3, 4, 6],input_dim = 1,num_classes=2)
                net_verifier.load_state_dict(torch.load(args.verifier_arch_weights_dir)['model_state_dict'])
                net_verifier.to(device)
            if use_fp16:
                net_verifier = net_verifier.half()

    if args.adi_scale != 0.0:
        student_arch = "resnet18"
        net = ResNet(ResidualBlock, [2, 2, 2],input_dim = 1,num_classes=2)
        net.load_state_dict(torch.load(args.arch_weights_dir)['model_state_dict'])

        if use_fp16:
            net_verifier, _ = amp.initialize(net_verifier, [], opt_level="O2")

        net_verifier = net_verifier.to(device)
        net_verifier.train()

        if use_fp16:
            for module in net_verifier.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval().half()
    else:
        print("No adaptive version")

    from medical_deepinversion import DeepInversionClass

    exp_name = args.exp_name
    # final images will be stored here:
    adi_data_path = "./final_images/%s"%exp_name
    # temporal data and generations will be stored here
    exp_name = "generations/%s"%exp_name

    args.iterations = 2000
    args.start_noise = True
    # args.detach_student = False

    args.resolution = 224
    bs = args.bs
    jitter = 30

    parameters = dict()
    parameters["resolution"] = 224
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
    coefficients["lr"] = args.lr
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    coefficients["adi_scale"] = args.adi_scale

    network_output_function = lambda x: x

    # check accuracy of verifier
    if args.verifier:
        hook_for_display = lambda x,y: validate_one(x, y, net_verifier)
    else:
        hook_for_display = None

    DeepInversionEngine = DeepInversionClass(net_teacher=net,
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
                                             device = device)
    net_student=None
    if args.adi_scale != 0:
        net_student = net_verifier
    DeepInversionEngine.generate_batch(net_student=net_student)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--worldsize', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--adi_scale', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion')
    parser.add_argument('--no-cuda', action='store_true')

    parser.add_argument('--epochs', default=20000, type=int, help='batch size')
    parser.add_argument('--setting_id', default=1, type=int, help='settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--jitter', default=30, type=int, help='batch size')
    parser.add_argument('--comment', default='', type=str, help='batch size')
    parser.add_argument('--arch_name', default='resnet18', type=str, help='model name')
    parser.add_argument('--arch_weights_dir', default='', type=str, help='model weights dir')

    parser.add_argument('--fp16', action='store_true', help='use FP16 for optimization')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')

    parser.add_argument('--verifier', action='store_true', help='evaluate batch with another model')
    parser.add_argument('--verifier_arch', type=str, default='resnet34', help = "arch name from torchvision models to act as a verifier")
    parser.add_argument('--verifier_arch_weights_dir', default='', type=str, help='model weights dir')

    parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--random_label', action='store_true', help='generate random label for optimization')
    parser.add_argument('--r_feature', type=float, default=0.05, help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10., help='additional multiplier on first bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--store_best_images', action='store_true', help='save best images as separate files')

    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    run(args)


    #  CUDA_VISIBLE_DEVICES=1 python3 medical_inversion.py 
    
if __name__ == '__main__':
    main()
