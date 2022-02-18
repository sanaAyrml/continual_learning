import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
import sys
from extras import update_lr,disable_dropout
from Medical_predictor_model import ResNet,ResidualBlock
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
# from tensorboard_maker import make_tensorboard
import random

import numpy as np
import time
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from data_loader import get_iid_loader, get_ood_loader

def run(args):
    config = {}
    config['outlier_exposure'] = args.outlier_exposure
    config['dataset_name'] = args.dataset_name

    config['sub_iid'] = -1
    config['sub_test']= -1

    seed = args.seed
    random.seed(args.seed)

    device = torch.device("cuda:1")

    num_epochs = args.num_epochs
    learning_rate = args.lr 
    save_interval =  args.save_interval 
    start_training = args.train


    """ grab dataloader for heart dataset """
    mean = [0.122, 0.122, 0.122] # mean and std is pre-computed using training set
    std = [0.184, 0.184, 0.184]
    train_set, valid_set, train_loader, valid_loader, view_c, view_w = get_iid_loader(
        config['dataset_name'], config['sub_iid'], config['sub_test'], config['outlier_exposure'], seed= seed)
    test_set, test_loader = get_ood_loader( (config['dataset_name']+'_test'), config['sub_test'], mean, std, seed= seed)

    dataset_sizes = {'train': len(train_set), 'val': len(valid_set), 'test': len(test_set)}
    print(dataset_sizes)
    dataloaders = {'train':train_loader, 'val':valid_loader, 'test':test_loader}



    # model = ResNet(ResidualBlock, [2, 2, 2],input_dim=args.input_dim,num_classes=2).to(device)
    model = ResNet(ResidualBlock, [3, 4, 6],input_dim=args.input_dim,num_classes=2).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.3, patience=5, threshold=0.0001)
    
    if args.load_ckpt:
        model.load_state_dict(torch.load(args.checkpoint_dir))

    if args.save_samples: 
        view_classes = train_set.view_classes
        print(view_classes)
        w  = [[] for i in range(len(view_classes))]
        for i,label in enumerate(train_set.data_dict['view_label']):
            w[label].append(i)
        print(len(w),len(w[0]),len(w[1]))
        if not os.path.exists(args.save_samples_dir):
            os.makedirs(args.save_samples_dir)
        for i in w:
            for j in range(10):
                plt.imshow(np.transpose(train_set.__getitem__(i[j])[0].numpy(), (1, 2, 0)))
                plt.savefig(args.save_samples_dir+'/label_'+str(train_set.__getitem__(i[j])[1])+
                            "_"+str(j)+"_"+view_classes[train_set.__getitem__(i[j])[1]])
    if start_training:
        print("start_training")


        #### log files for multiple runs are NOT overwritten


        log_dir = args.log_dir 
        print(log_dir)
        if not os.path.exists(log_dir):
            print("here2")
            os.makedirs(log_dir)
        # self.model = w #TODO

        #### get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run 
        log_f_name_train = log_dir + '/Train_normal_' + "_log_" + str(run_num) + ".csv"
        print(log_f_name_train)

        log_f_name_val = log_dir + '/Valid_normal_' +  "_log_" + str(run_num) + ".csv"
        print(log_f_name_val)


        # logging file
        log_f_train = open(log_f_name_train,"w+")
        log_f_train.write('num_trained_data,epoch,loss,acc\n')

        log_f_valid = open(log_f_name_val,"w+")
        log_f_valid.write('num_trained_data,epoch,loss,acc\n')

        # Train the model
        total_step = len(train_loader)
        curr_lr = learning_rate
        best_performance = 0

        number_of_total_images = 0
        
        ckpt_dir = log_dir+"/"+str(run_num)
        if not os.path.exists(ckpt_dir):
            print("here2")
            os.makedirs(ckpt_dir)
        print("here")
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            total_loss= 0
            iter_num = 0
            for i, (images, labels, if_noisy) in enumerate(dataloaders['train']):
                optimizer.zero_grad()
                model.train()
                disable_dropout(model)
                if args.input_dim == 1:
                    images = images[:,0,:,:].unsqueeze(1).to(device)
                else:
                    images = images[:,0,:,:].to(device)
                labels = labels.to(device)
                number_of_total_images += len(labels)
                # Forward pass
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                total_loss += loss.item()
                iter_num += 1
                # loss.requres_grad = True

                # Backward and optimizea
                loss.backward()
                optimizer.step()

            if (epoch)  % save_interval == 0:
                print('Accuracy of the model on the train images: {} %'.format(100 * correct / total))
                print ("Epoch [{}/{}], with number of seen data {} Loss: {:.4f}"
                       .format(epoch+1, num_epochs, number_of_total_images, total_loss/iter_num))

            if (epoch) % 1 == 0:
                log_f_train.write('{},{},{},{}\n'.format(number_of_total_images,epoch,loss.item(), correct/total))
                log_f_train.flush() 


            if (epoch) % save_interval ==  0:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    total_val_loss = 0
                    iter_num = 0
                    for images, labels,if_noisy in dataloaders['val']:
                        if args.input_dim == 1:
                            images = images[:,0,:,:].unsqueeze(1).to(device)
                        else:
                            images = images[:,0,:,:].to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        val_loss = criterion(outputs, labels)
                        total_val_loss += val_loss.item()
                        iter_num += 1
                    print("learning rate is",optimizer.param_groups[0]['lr'])

                    print('Accuracy of the model on the test images: {} %, Loss: {:.4f}'.format(100 * correct / total,total_val_loss/iter_num))

                if correct / total > best_performance:
                    best_performance = correct / total
                    # Save the model checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'Acc': best_performance}, 
                        ckpt_dir +"/"+args.checkpoint_dir+"epoch_"+str(epoch)+".ckpt")
                scheduler.step(total_val_loss/iter_num)
                    
                log_f_valid.write('{},{},{}\n'.format(number_of_total_images,epoch, total_val_loss/iter_num, correct / total))
                log_f_valid.flush()



    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--outlier_exposure', action='store_true', help='Use all the classes')
    parser.add_argument('--train', action='store_true', help='train')
    parser.add_argument('--save_samples', action='store_true', help='save samples from training set')
    parser.add_argument('--save_samples_dir', default='', type=str , help='save samples from training set direction')
    parser.add_argument('--load_ckpt', action='store_true', help='load from checkpoint')
    parser.add_argument('--dataset_name', default='heart', type=str, help='name of  dataset')
    parser.add_argument('--seed', default='2020', type=int, help='seed')
    
    
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--input_dim', type=int, default=1, help='input dimention')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for optimization')
    parser.add_argument('--save_interval', default=5, type=int, help='saving interval')
    
    parser.add_argument('--log_dir', default='', type=str, help='loging direction')
    parser.add_argument('--checkpoint_dir', default='', type=str, help='checkpoint direction')
    
    #python3 train_on_heart_dataset.py  --log_dir "./../trained_models_on_medical_data" --checkpoint_dir "resnet34_3_4_6_trained_on_2_classes" --train
    args = parser.parse_args()
    print(args)

    run(args)


if __name__ == '__main__':
    main()