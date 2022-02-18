import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import scipy.io as sio
import random
# -------------------------- DATASET DESCRIPTION -------------------------------------------------------------------#
# There were three trials of labels for quality. The first includes the quality+view classification (16000+ cases).
# The second (2400+) and third (2800+) were relabelling the quality and with no overlapping (in total 5151).
# In short, we say the second and third belongs to the second trial.
# The labels (see list_train/valid/test_with_3rd_trail) were arranged as:
# file_address, label_quality_first_trial, label_view_classification, label_quality_second_trial, i.e.,
# xxxx.mat, 0, 12, -1
# -------------------------------------------------------------------------------------------------------------------#
# for quality labels, the range is (-1, 0, 1, 2, 3, 4),
# -1 indicates not labelled in the second trial (only happens in label_quality_second_trial),
# 0 indicates garbage, and 1-4 are the four classes, from Poor (0~25%), to Fair (25%~50%), Good (50%~75%), and Excellent
# (75%~100%).
# In the original regression model (before the observer variability thing), the regression labels were defined as:
# Garbage -> 0, Poor -> 0.25, Fair -> 0.5, Good -> 0.75, Excellent -> 1.0
# with the observer variability, the regression labels are defined as the expected value of each class:
# Garbage -> 0, Poor -> 0.125, Fair -> 0.375, Good -> 0.625, Excellent -> 0.875.
# with the observer variability + CDF (Cumulative distribution function): the intervals defined to reflect the classes
# are aligned with the original definition:
# Garbage -> [0, 0.25], Poor -> [0, 0.25], Fair -> [0.25, 0.5], Good -> [0.5, 0.75], Excellent -> [0.75, 1.0]
# *** for Garbage, I really did not find a good way to represent this class, the problem with it is that the formula
# needs an interval
# *** the alternative is for Garbage class alone, using a regression loss.
# -------------------------------------------------------------------------------------------------------------------#
# for view label, the range is 0-16, where the first 0:13 are the 14 cardiac classes, and 14 and 15 are unknown and
# other classes, which are not used. 16 is the NON_CARDIAC class, which is the garbage class.
# For 0 to 13, the classes are:
# {'AP2', 'AP3', 'AP4', 'AP5', 'PLAX', 'RVIF', 'SUBC4', ...
#  'SUBC5', 'SUBCIVC', 'PSAX(A)', 'PSAX(M)', 'PSAX(PM)', ...
#  'PSAX(APIX)', 'SUPRA'}
# After loading the dataset and 'dataset_usage_option' is 'include_noncardiac_class', NON_CARDIAC class will appear as
# value 14, so 15 classes in total.
# -------------------------------------------------------------------------------------------------------------------#


class Heart(data.Dataset):

    def __init__(self, data_list, transform=None, used_labels = None): #label_option='iid', dataset_option='single',
        # if outlier_exposure is true, then the dataset will include outlier images
        self.transform = transform
        self.data_list = [data_list]
        
        self.labels = ['quality_label','view_label','quality_label_t2']
        self.option = 'single' #dataset_option
        
        self.data_dict = self.read_from_list(self.data_list, '')
        self.quality_classes = ['garb', 'poor', 'fair', 'good', 'xlnt']
        
        labels = np.unique(self.data_dict['view_label'])
        ignore_list = np.delete(labels,used_labels)
        self.data_dict = self.ignore_class(self.data_dict, ignore_list, tag='view_label')

        self.view_classes = ['AP2', 'AP3','AP4', 'AP5', 'PLAX', 'RVIF', 'SUBC4', 'SUBC5',
             'SUBIVC', 'PSAX(A)', 'PSAX(M)', 'PSAX(PM)',  'PSAX(APIX)', 'SUPRA' , 'unknown', 'garbage']
        self.view_classes = [self.view_classes[index] for index in self.view_classes]
        
        # compute class weights using inverse frequency
        self.quality_label_weights = self.compute_class_weights('quality_label')
        
        # this view weight has a size of 15, not like the 13 output classes. We use this to weigh the random sampler.
        self.view_label_weights = self.compute_class_weights('view_label')
        print("Heart dataset loaded, N = " + str(len(self.data_dict['file_address'])) )
        self.flipped_version = False

    
    def __getitem__(self, index):

        img_path = self.data_dict['file_address'][index]
        
        # augment the view label so "other" and "garbage" mean something when it comes to uncertainty
        # the view label being one-hot, when it is "other" or "garbage", it is uniform distribution
        view = self.data_dict['view_label'][index]
        view_oh = np.zeros(len(self.view_classes))
        if view < len(self.view_classes):
            view_oh[view] = 1
        else:
            view_oh = view_oh + 1/len(self.view_classes)
        
        # in the training loop, augment the quality label for interobserver variability
        target = self.data_dict['quality_label'][index]
        # check for opinion from 2nd doctor
        target2 = self.data_dict['quality_label_t2'][index]
        if target2 == -1:
            target2 = self.data_dict['quality_label'][index]
        if self.option == 'single': # only use 1st doctor
            target2 = self.data_dict['quality_label'][index]
        
        # load the image
        img = self.read_image(img_path) # returns a 2D DxD uint8 array
        self.flipped_version:
            print(img.shape)
        img_3ch_tensor = self.transform(img)
        return img_3ch_tensor, view 


                
    def __len__(self):
        return len(self.data_dict['quality_label'])

    def get_label_loss_weights(self):
        return self.view_label_weights[:len(self.view_classes)], self.quality_label_weights[:len(self.quality_classes)]
        
    def get_label_sampling_weights(self):
        return self.view_label_weights, self.quality_label_weights
        
    def get_label_sampling_weights_per_item(self):
        # directly supports pytorch's WeightedRandomSampler, associates the sampling weight of each sample with its view class
        weights = [0] * len(self.data_dict['view_label'])
        for i in range(len(self.data_dict['view_label']) ):
            class_of_i = self.data_dict['view_label'][i]
            weights[i] = self.view_label_weights[class_of_i]
        return weights
    
    def get_classes(self):
        return self.view_classes, self.quality_classes
    
    # read file addresses and labels from Jorden's list
    # added some options for taking different subsets of data
    def read_from_list(self, list_of_files, data_root):

        def read_text_file(fname, data_root=''):
            f = open(fname, 'r')
            lines = f.readlines()
            file_address = []
            quality_label = []
            view_label = []
            quality_label_t2 = []

            for line in lines:
                parts = line.split('\n')
                line = parts[0]
                parts = line.split(',')
                fa = '../..'+data_root + parts[0]
                if not os.path.isfile(fa):
                    print(fa + ' does not exist')
                    continue
                if self.option == 'inter' and int(parts[3]) == -1:
                    continue
                file_address.append(fa)
                quality_label.append(int(parts[1]))
                view_label.append(int(parts[2]))

                if len(parts) == 4:
                    quality_label_t2.append(int(parts[3]))
                else:
                    quality_label_t2.append(int(-1))

            return file_address, quality_label, view_label, quality_label_t2

        file_address = []
        quality_label = []
        view_label = []
        quality_label_t2 = []

        for fname in list_of_files:
            fa, ql, vl, ql2 = read_text_file(fname, data_root)
            file_address += fa
            quality_label += ql
            view_label += vl
            quality_label_t2 += ql2

        quality_label = np.asarray(quality_label)
        view_label = np.asarray(view_label)
        quality_label_t2 = np.asarray(quality_label_t2)

        return {'file_address': file_address, 'quality_label': quality_label, 'view_label': view_label,
                'quality_label_t2': quality_label_t2, 'num_files': len(file_address)}
                
    # remove a subset of classes, however, it does not rebase the label, all classes continue using the original index
    # input is a data_dict as defined by read_from_list
    def ignore_class(self, data_dict, class_no=None, tag=None):
        # checker
        if type(class_no) == int:
            class_no = [class_no]
        elif type(class_no) == list:
            for c in class_no:
                if type(c) != int:
                    raise ValueError('A class index must be a int value or a list of int value')
        else:
            raise ValueError('A class index must be a int value or a list of int value')

        file_address = []
        quality_label = []
        quality_label_t2 = []
        view_label = []

        for i, c in enumerate(data_dict[tag]):
            if c in class_no:
                continue
            file_address.append(data_dict['file_address'][i])
            quality_label.append(data_dict['quality_label'][i])
            quality_label_t2.append(data_dict['quality_label_t2'][i])
            view_label.append(data_dict['view_label'][i])

        return {'file_address': file_address, 'quality_label': quality_label, 'view_label': view_label,
                'quality_label_t2': quality_label_t2, 'num_files': len(file_address)}
                
    # slide the classification labels "leftward", ie. (0, 2, 3, 5) -> (0, 1, 2, 3)
    def rebase_label(self, data_dict, tag=None, convert_dict=None):

        if convert_dict is None or type(convert_dict) is not dict:
            convert_dict = dict()
            for new_index, old_index in enumerate(np.unique(data_dict[tag])):
                convert_dict[old_index] = new_index

        for i, c in enumerate(data_dict[tag]):
            data_dict[tag][i] = convert_dict[data_dict[tag][i]]
            
        return data_dict
        
    # compute class weights using inverse frequency, should only be used if there are no gaps in class labels
    def compute_class_weights(self, tag):
        arr = self.data_dict[tag]
        freq = np.bincount(arr)
        total_samples = len(arr)
        inv_freq = 1. / freq
        #print(total_samples * inv_freq / len(np.unique(arr)) )
        return (total_samples * inv_freq / len(np.unique(arr)) )
        
    # read image as HxW uint8 from the cine mat file at img_path
    def read_image(self, img_path, frame_option='random'):
        try:
            matfile = sio.loadmat(img_path, verify_compressed_data_integrity=False)
        # except ValueError:
        #     print(fa)
        #     raise ValueError()
        except TypeError:
            print(img_path)
            raise TypeError()

        d = matfile['Patient']['DicomImage'][0][0]

        # if 'ImageCroppingMask' in matfile['Patient'].dtype.names:
            # mask = matfile['Patient']['ImageCroppingMask'][0][0]
            # # d = d*np.expand_dims(mask, axis=2)

        if frame_option == 'start':
            d = d[:, :, 0]
        elif frame_option == 'end':
            d = d[:, :, -1]
        elif frame_option == 'random':
            r = np.random.randint(0, d.shape[2])
            d = d[:, :, r]

        return d