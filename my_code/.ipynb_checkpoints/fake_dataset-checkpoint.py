import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import scipy.io as sio
import random

import pickle

class fake_Heart(data.Dataset):

    def __init__(self, data_address,label_address , transform=None, used_labels = None,order= None): #label_option='iid', dataset_option='single',
        # if outlier_exposure is true, then the dataset will include outlier images

        # mean = [0, 0, 0]
        # std = [1, 1, 1]
        print("newww")
        with open(data_address,'rb') as f:
            self.proto_sets_x = np.concatenate(pickle.load(f))
        with open(label_address,'rb') as f:
            self.proto_sets_y = np.concatenate(pickle.load(f))
        print(len(self.proto_sets_x))
        print(len(self.proto_sets_y))
        self.mean = [-0.0118]
        self.std = [0.9243]
        print(self.mean,self.std)
        mode = 'train'
        self.transform = _get_transform(self.mean, self.std, mode, to_pil_image=True)
        
    def __getitem__(self, index):
            # print("using protoset",self.o)
            # self.o += 1
            
            img_3ch_tensor = self.transform(self.proto_sets_x[index].reshape((224,224,1)))
            view = self.proto_sets_y[index]
            return img_3ch_tensor[0,:,:].unsqueeze(0), view 
    def __len__(self):
        return len(self.proto_sets_x)
    
    
def _get_transform(mean, std, mode='valid', to_pil_image=False):
    if mode=='train':
        aug_transform = transforms.Compose(
            [transforms.RandomRotation(5),
             transforms.RandomCrop((224,224)) ])
    else: # mode=='valid'
        aug_transform = transforms.Resize((224,224))
    t_transform = transforms.Compose(
        [aug_transform,
         transforms.ToTensor(),
         transforms.Normalize(mean, std)
        ])
    
    # t_transform = transforms.Compose(
    #     [
    #      transforms.ToTensor()
    #     ])
    
    if to_pil_image:
        t_transform = transforms.Compose([transforms.ToPILImage(), 
                                            t_transform])
    return t_transform