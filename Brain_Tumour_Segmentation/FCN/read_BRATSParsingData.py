__author__ = 'charlie'
import numpy as np
import os, sys
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
from glob import glob

import TensorflowUtils as utils

def read_dataset(data_dir):
    pickle_filename = "BRATS2015_parsing.pkl"
    pickle_filepath = os.path.join("/home/seonwhee/Deep_Learning/MRImage_Pipeline/", pickle_filename)
    if not os.path.exists(pickle_filepath):        
        result = create_image_lists(data_dir)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)        
        training_records = result[:int(len(result)*2/3)]
        validation_records = result[int(len(result)*2/3)+1:]
        
        del result

    return training_records, validation_records

def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    image_list_training = []
    patients = glob(image_dir+"/*/")    
    
    for patient in patients:
        file_glob = os.path.join(patient, '*/*.'+'mha')
        file_list = glob(file_glob)
        #print("file_list", file_list)
        if not file_list:
            print("No files found")
        else:
            record = {}
            for f in file_list:
                #print('f', f)
                if 'T1.' in f:
                    record["T1"] = f
                elif 'T1c.' in f:
                    record["T1c"] = f
                elif 'T2' in f:
                    record["T2"] = f
                elif 'Flair' in f:
                    record["FLAIR"] = f
                elif '.OT.' in f:
                    record["GT"] = f               
            
            for i in list(record.values()):                
                print(i)
                if "GT" in list(record.keys()):
                    if i != record["GT"]:                        
                        annot = record["GT"]
                        new_record = {}
                        new_record["image"] = i
                        new_record["annotation"] = annot
                        del annot
                        image_list_training.append(new_record)
                        del new_record
                else:
                    new_record = {}
                    new_record["image"] = i
                    image_list_training.append(new_record)
                    del new_record
                    

    return image_list_training


def read_test_dataset(data_dir):
    pickle_filename = "BRATS2015_testing.pkl"
    pickle_filepath = os.path.join("/home/seonwhee/Deep_Learning/MRImage_Pipeline/", pickle_filename)
    if not os.path.exists(pickle_filepath):        
        result = create_image_lists(data_dir)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)        
        test_records = result
        
        del result

    return test_records

