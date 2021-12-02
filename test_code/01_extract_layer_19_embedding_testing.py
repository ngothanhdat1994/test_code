import os
import sys
import re
import numpy as np
import librosa
import soundfile as sf

# Import TF 2.X and make sure we're running eager.
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()

import tensorflow_hub as hub

import csv

metadata_csv_dir = "./metadata_testing.csv"

# Input directory
input_directory = "./../../07_Dat_cough/for_dat/Second_DiCOVA_Challenge_Test_Data_Release/AUDIO/cough"

# # Output directory
# output_directory = "/Users/thanhdat/EMBC_2022/Genrerate_MFCC_26_11_21/02_dl_for_dat/test_code"

# CSV title making
embed_list = []
#for i in range (512):     # 512 is for last layer of TRILL 
for i in range (12288):     # 12288 is for layer 19 of TRILL
    i +=1
    name = ",embed" + str(i)
    embed_list.append(name)
with open(os.path.join("trill_layer_19_embeddings_testing"+".csv"), "a") as text_file:
        text_file.write("file_ID" + ''.join(embed_list) + ",covid_status" + "\n")


# CSV content making
with open(metadata_csv_dir) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        org_file_ID = row[0]
        org_file_class = row[1]
        
        file_name = org_file_ID + ".flac"
        file_class = org_file_class
    
        file_name_dir = os.path.join(input_directory, file_name)
        wav, sr = librosa.load(file_name_dir)
        
        # Load the module and run inference.
        module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/2')
        # `wav_as_float_or_int16` can be a numpy array or tf.Tensor of float type or
        # int16. The sample rate must be 16kHz. Resample to this sample rate, if
        # necessary.
        emb_dict = module(samples = wav, sample_rate = 16000)
        # For a description of the difference between the two endpoints, please see our
        # paper (https://arxiv.org/abs/2002.12764), section "Neural Network Layer".
        
        # Embeddings are a [time, feature_dim] Tensors.
        
        #emb = emb_dict['embedding']
        emb_layer19 = emb_dict['layer19']
        
        #print(emb)
        #print(emb.shape)
        # Calculate mean on Time dimension
        
        #time_avg_emb = tf.reduce_mean(emb, axis = 0)
        time_avg_emb_layer19 = tf.reduce_mean(emb_layer19, axis = 0)
        
        #print(time_avg_emb)
        #print(time_avg_emb.shape)
        #array_form_tve = time_avg_emb.numpy()
        array_form_tve = time_avg_emb_layer19.numpy()
        
        list_form_tve = array_form_tve.tolist()
        with open(os.path.join("trill_layer_19_embeddings_testing"+".csv"), "a") as text_file:
            #text_file.write("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f} \n".format(str(round(str_ind_dur,2))+"_to_"+str(round(end_ind_dur,2)), sum_valid_end_output[0], sum_valid_end_output[1], sum_valid_end_output[2], sum_valid_end_output[3], sum_valid_end_output[4]))
            text_file.write("{}{},{} \n".format(file_name.split('.')[0], ''.join("," + str(e) for e in list_form_tve), str(file_class)))

    
    
    

    



















