import numpy as np
import os
from itertools import islice
import sys
import re
from natsort import natsorted, ns
#from hypara import *
import random
import scipy


def get_batch_num(batch_size = 40):

        org_class_list = os.listdir("/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/01_fea_ext_for_dat/01_spec_gen_cough/11_MFCC/data_dev")
        class_list = []  #remove .file
        for nClass in range(0,len(org_class_list)):
            isHidden=re.match("\.",org_class_list[nClass])
            if (isHidden is None):
                class_list.append(org_class_list[nClass])
        class_num  = len(class_list)
        class_list = sorted(class_list)

        max_file_num = 0
        for class_mem in class_list:
            file_dir = os.path.join("/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/07_Dat_cough/01_fea_ext_for_dat/01_spec_gen_cough/11_MFCC/data_dev", class_mem)

            org_file_list = os.listdir(file_dir)
            file_list = []  #remove .file
            for nFile in range(0,len(org_file_list)):
                isHidden=re.match("\.",org_file_list[nFile])
                if (isHidden is None):
                    file_list.append(org_file_list[nFile])
            file_num  = len(file_list)
            if file_num > max_file_num:
                max_file_num = file_num
                
                
        return int(max_file_num/batch_size) + 1

def main():
    print(get_batch_num())
    get_batch_num()


if __name__ == "__main__":
    main()