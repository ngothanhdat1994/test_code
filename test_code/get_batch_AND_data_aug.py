#---- general packages
import numpy as np
import os
import argparse
import math
import scipy.io
import re
import time
import datetime
import sys
import tensorflow as tf
from sklearn import datasets, svm, metrics



def data_aug(self, i_data, i_label, beta=0.4):
        half_batch_size = round(np.shape(i_data)[0]/2)
        #print(half_batch_size, np.shape(i_data))

        #x1  = i_data[:half_batch_size,:] #for input vector
        #x2  = i_data[half_batch_size:,:]
        x1  = i_data[:half_batch_size, :, :, :] #for input image
        x2  = i_data[half_batch_size:, :, :, :]

        # frequency/time masking is only for image
        #for j in range(x1.shape[0]):
        #   # spectrum augment
        #   for c in range(x1.shape[3]):
        #       x1[j, :, :, c] = self.frequency_masking(x1[j, :, :, c])
        #       x1[j, :, :, c] = self.time_masking(x1[j, :, :, c])
        #       x2[j, :, :, c] = self.frequency_masking(x2[j, :, :, c])
        #       x2[j, :, :, c] = self.time_masking(x2[j, :, :, c])

        y1  = i_label[:half_batch_size,:]
        y2  = i_label[half_batch_size:,:]

        # Beta dis
        b   = np.random.beta(beta, beta, half_batch_size)
        #X_b = b.reshape(half_batch_size, 1) #for vector input
        X_b = b.reshape(half_batch_size, 1, 1, 1) #for image input
        y_b = b.reshape(half_batch_size, 1)

        xb_mix   = x1*X_b     + x2*(1-X_b)
        xb_mix_2 = x1*(1-X_b) + x2*X_b

        yb_mix   = y1*y_b     + y2*(1-y_b)
        yb_mix_2 = y1*(1-y_b) + y2*y_b

        # Uniform dis
        l   = np.random.random(half_batch_size)
        #X_l = l.reshape(half_batch_size, 1) #for vector input
        X_l = l.reshape(half_batch_size, 1, 1, 1) #for image input
        y_l = l.reshape(half_batch_size, 1)

        xl_mix   = x1*X_l     + x2*(1-X_l)
        xl_mix_2 = x1*(1-X_l) + x2*X_l

        yl_mix   = y1* y_l    + y2 * (1-y_l)
        yl_mix_2 = y1*(1-y_l) + y2*y_l

        #o_data     = np.concatenate((xb_mix,    x1,    xl_mix,    xb_mix_2,    x2,    xl_mix_2),    0)
        #o_label    = np.concatenate((yb_mix,    y1,    yl_mix,    yb_mix_2,    y2,    yl_mix_2),    0)
        o_data     = np.concatenate((xb_mix,    x1,    xb_mix_2,    x2),    0)
        o_label    = np.concatenate((yb_mix,    y1,    yb_mix_2,    y2),    0)
        #o_data     = np.concatenate((xb_mix,    xl_mix,    xb_mix_2,    xl_mix_2),    0)
        #o_label    = np.concatenate((yb_mix,    yl_mix,    yb_mix_2,    yl_mix_2),    0)

        return o_data, o_label


def get_batch (batch_num = 0, batch_size = 40, is_mixup = False):
        label_dict     = dict(p=1, n=0)
        class_num     = len(label_dict)

        org_class_list = os.listdir("/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/01_fea_ext_for_dat/01_spec_gen_cough/11_MFCC/data_dev")
        class_list = []  #remove .file
        for nClass in range(0,len(org_class_list)):
            isHidden=re.match("\.",org_class_list[nClass])
            if (isHidden is None):
                class_list.append(org_class_list[nClass])
        class_num  = len(class_list)
        class_list = sorted(class_list)

        nImage = 0
        for class_mem in class_list:
            file_dir = os.path.join("/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/01_fea_ext_for_dat/01_spec_gen_cough/11_MFCC/data_dev", class_mem)

            org_file_list = os.listdir(file_dir)
            file_list = []  #remove .file
            for nFile in range(0,len(org_file_list)):
                isHidden=re.match("\.",org_file_list[nFile])
                if (isHidden is None):
                    file_list.append(org_file_list[nFile])
            file_num  = len(file_list)
            file_list = sorted(file_list)
            train_file_id  = np.random.RandomState(seed=42).permutation(file_num)
            print(train_file_id)
            

            for ind in range(batch_num*batch_size, (batch_num+1)*batch_size):
                if ind >= file_num:
                    mul = int(ind/file_num)
                    ind = ind - mul*file_num
                # open file
                file_name = file_list[train_file_id[ind]]
                file_open = os.path.join("/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/01_fea_ext_for_dat/01_spec_gen_cough/11_MFCC/data_dev", class_mem, file_name)
   # batch is 40 but file_num have 30 --> 1-30-1-10
                #create label
                class_name = file_name.split('.')[0].split('_')[-1]  #patch
                nClass = label_dict[class_name]                                        
                expectedClass = np.zeros([1,class_num])
                expectedClass[0,nClass] = 1

                ##create data with vector input
                #one_vector = np.reshape(np.load(file_open),(1,-1))  
                #if (nImage == 0):
                #   seq_x = one_vector
                #   seq_y = expectedClass
                #else:            
                #   seq_x = np.concatenate((seq_x, one_vector), axis=0)  
                #   seq_y = np.concatenate((seq_y, expectedClass), axis=0)  

                #create data with image input
                one_image = np.load(file_open)  
                #file_str = scipy.io.loadmat(file_open)  
                #one_image = file_str['final_data']
                #nF, nT    = np.shape(one_image)
                #one_image = np.reshape(one_image, (1,nF,nT,1))
                if (nImage == 0):
                   seq_x = one_image
                   seq_y = expectedClass
                else:            
                   seq_x = np.concatenate((seq_x, one_image), axis=0)  
                   seq_y = np.concatenate((seq_y, expectedClass), axis=0)  

                nImage += 1
        print(np.shape(seq_x))
        print(np.shape(seq_y))
        print(nImage)
        
        half_batch_size = round(np.shape(seq_x)[0]/2)
        #print(half_batch_size, np.shape(i_data))

        #x1  = i_data[:half_batch_size,:] #for input vector
        #x2  = i_data[half_batch_size:,:]
        x1  = seq_x[:half_batch_size, :, :, :] #for input image
        x2  = seq_x[half_batch_size:, :, :, :]

        # frequency/time masking is only for image
        #for j in range(x1.shape[0]):
        #   # spectrum augment
        #   for c in range(x1.shape[3]):
        #       x1[j, :, :, c] = self.frequency_masking(x1[j, :, :, c])
        #       x1[j, :, :, c] = self.time_masking(x1[j, :, :, c])
        #       x2[j, :, :, c] = self.frequency_masking(x2[j, :, :, c])
        #       x2[j, :, :, c] = self.time_masking(x2[j, :, :, c])

        y1  = seq_y[:half_batch_size,:]
        y2  = seq_y[half_batch_size:,:]

        # Beta dis
        beta=0.4
        b   = np.random.beta(beta, beta, half_batch_size)
        #X_b = b.reshape(half_batch_size, 1) #for vector input
        X_b = b.reshape(half_batch_size, 1, 1, 1) #for image input
        y_b = b.reshape(half_batch_size, 1)

        xb_mix   = x1*X_b     + x2*(1-X_b)
        xb_mix_2 = x1*(1-X_b) + x2*X_b
        print(np.shape(xb_mix))
        print(np.shape(xb_mix_2))

        yb_mix   = y1*y_b     + y2*(1-y_b)
        yb_mix_2 = y1*(1-y_b) + y2*y_b
        print(np.shape(yb_mix))
        print(np.shape(yb_mix_2))

        # Uniform dis
        l   = np.random.random(half_batch_size)
        #X_l = l.reshape(half_batch_size, 1) #for vector input
        X_l = l.reshape(half_batch_size, 1, 1, 1) #for image input
        y_l = l.reshape(half_batch_size, 1)

        xl_mix   = x1*X_l     + x2*(1-X_l)
        xl_mix_2 = x1*(1-X_l) + x2*X_l

        yl_mix   = y1* y_l    + y2 * (1-y_l)
        yl_mix_2 = y1*(1-y_l) + y2*y_l

        #o_data     = np.concatenate((xb_mix,    x1,    xl_mix,    xb_mix_2,    x2,    xl_mix_2),    0)
        #o_label    = np.concatenate((yb_mix,    y1,    yl_mix,    yb_mix_2,    y2,    yl_mix_2),    0)
        o_data     = np.concatenate((xb_mix,    x1,    xb_mix_2,    x2),    0)
        o_label    = np.concatenate((yb_mix,    y1,    yb_mix_2,    y2),    0)
        print(np.shape(o_data))
        print(np.shape(o_label))

        # if is_mixup:
        #     o_data, o_label = self.data_aug(seq_x, seq_y, 0.4)
        # else:
        #     o_data  = seq_x
        #     o_label = seq_y

        return o_data, o_label, nImage
    
def main():

    get_batch()


if __name__ == "__main__":
    main()