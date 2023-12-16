# Author: ag6 (adapted from diffusion_policy)
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import pickle
import numpy as np
import sys
import os
import cv2
import time
import sys
sys.path.append("/home/ros_ws/dataset")
sys.path.append("/home/ros_ws/")
from data_utils import *
from preprocess_audio import process_audio
from collections import OrderedDict
from torchvision import transforms
import shutil

#@markdown ### **Dataset**
#@markdown
#@markdown Defines `ToyDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data ((image, agent_pos), action) from a pkl storage
#@markdown - Normalizes each dimension of agent_pos and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `image`: shape (obs_hoirzon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_hoirzon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)

# dataset

def read_audio_files(dataset_path:str, num_classes=3):
    """ Read audio files from the dataset path"""

    label2name = {0: "coins", 1: "box", 2: "other"}
    name2label = {"coins": 0, "box": 1, "other": 2}

    coins_path = os.path.join(dataset_path, "0")
    box_path = os.path.join(dataset_path, "1")
    other_path = os.path.join(dataset_path, "2")

    # read all data files one by one
    coins_files = [[os.path.join(coins_path, f),int(name2label["coins"])] for f in sorted(os.listdir(coins_path)) if f.endswith('.npy')]
    box_files = [[os.path.join(box_path, f),int(name2label["box"])] for f in sorted(os.listdir(box_path)) if f.endswith('.npy')]
    other_files = [[os.path.join(other_path, f),int(name2label['other'])] for f in sorted(os.listdir(other_path)) if f.endswith('.npy')]

    audio_data_info = []
    max_idx = max(len(coins_files), len(box_files), len(other_files))

    if(num_classes==3):
        for idx in range(max_idx):
            if idx < len(coins_files):
                audio_data_info.extend(np.array(coins_files[idx]).reshape(1,2))
            if idx < len(box_files):
                audio_data_info.extend(np.array(box_files[idx]).reshape(1,2))
            if idx < len(other_files):
                audio_data_info.extend(np.array(other_files[idx]).reshape(1,2))
    elif(num_classes==2):
        for idx in range(max_idx):
            if idx < len(coins_files):
                audio_data_info.extend(np.array(coins_files[idx]).reshape(1,2))
            if idx < len(box_files):
                audio_data_info.extend(np.array(box_files[idx]).reshape(1,2))

    print("*************************************")
    print("Loaded audio data info")
    print(" Total files=", len(audio_data_info))
    print(" Coins files=", len(coins_files))
    print(" Box files=", len(box_files))
    if(num_classes==3):
        print(" Other files=", len(other_files))
    print("*************************************")

    return audio_data_info

def clean_directory(directory_path:str):

    assert os.path.exists(directory_path), "Directory path does not exist"
    shutil.rmtree(directory_path)
    print("Removed directory: ", directory_path)

    return

def create_directory(directory_path:str):

    assert not os.path.exists(directory_path), "Directory path already exists"
    os.mkdir(directory_path)
    os.mkdir(os.path.join(directory_path, "0"))
    os.mkdir(os.path.join(directory_path, "1"))
    os.mkdir(os.path.join(directory_path, "2"))
    print("Created directory: ", directory_path)

    return

def split_data(dataset_path:str, split_ratio=0.8):
    """ Split data into train, val set"""

    label2name = {0: "coins", 1: "box", 2: "other"}
    name2label = {"coins": 0, "box": 1, "other": 2}

    train_folder_path = os.path.join(dataset_path, "train")
    val_folder_path = os.path.join(dataset_path, "eval")

    if(os.path.exists(train_folder_path)):
        print("Train folder already exists!!!")
        clean_directory(train_folder_path)
    
    if(os.path.exists(val_folder_path)):
        print("Val folder already exists!!!")
        clean_directory(val_folder_path)
    
    # Create directories
    create_directory(train_folder_path)
    create_directory(val_folder_path)

    coins_path = os.path.join(dataset_path, "0")
    box_path = os.path.join(dataset_path, "1")
    other_path = os.path.join(dataset_path, "2")

    # read all data files one by one
    coins_files = [[os.path.join(coins_path, f),int(name2label["coins"])] for f in sorted(os.listdir(coins_path)) if f.endswith('.npy')]
    box_files = [[os.path.join(box_path, f),int(name2label["box"])] for f in sorted(os.listdir(box_path)) if f.endswith('.npy')]
    other_files = [[os.path.join(other_path, f),int(name2label['other'])] for f in sorted(os.listdir(other_path)) if f.endswith('.npy')]

    train_idx_coins = int(len(coins_files)*split_ratio)
    train_idx_box = int(len(box_files)*split_ratio)
    train_idx_other = int(len(other_files)*split_ratio)
    ## Stats
    total_train_files = train_idx_box + train_idx_coins + train_idx_other
    total_val_files = len(coins_files)+len(box_files)+len(other_files)-total_train_files 

    max_idx = max(len(coins_files), len(box_files), len(other_files))

    print("train idx coins: ", train_idx_coins)
    print("train idx box: ", train_idx_box)
    print("train idx other: ", train_idx_other)

    for idx in range(max_idx):
        if idx < train_idx_coins:
            shutil.copy(coins_files[idx][0], os.path.join(train_folder_path,str(name2label["coins"])))
        elif idx >= train_idx_coins and idx<len(coins_files):
            print("idx coins",idx)
            shutil.copy(coins_files[idx][0], os.path.join(val_folder_path,str(name2label["coins"])))

        if idx < train_idx_box:
            shutil.copy(box_files[idx][0], os.path.join(train_folder_path,str(name2label["box"])))
        elif idx >= train_idx_box and idx<len(box_files):
            print("idx box",idx)
            shutil.copy(box_files[idx][0], os.path.join(val_folder_path,str(name2label["box"])))

        if idx < train_idx_other:
            shutil.copy(other_files[idx][0], os.path.join(train_folder_path,str(name2label["other"])))
        elif idx >= train_idx_other and idx<len(other_files):
            print("idx other",idx)
            shutil.copy(other_files[idx][0], os.path.join(val_folder_path,str(name2label["other"])))

    print("*************************************")
    print("Splitted data into train, val set")
    print(" Total train files=", total_train_files)
    print(" Total val files=", total_val_files)
    print("*************************************") 

    return


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 num_classes:int  =3
                 ): 
        
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        print("dataset_path: ", dataset_path)
        assert os.path.exists(dataset_path), "Dataset path does not exist"

        # Read audio files
        self.audio_data_info = read_audio_files(dataset_path,num_classes=num_classes)
        # Convert to numpy array
        self.audio_data_info = np.array(self.audio_data_info)

        # transforms for image data
        # Use some transforms for audio data
        # if self.is_state_based==False:
        #     self.image_transforms = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.Resize((256,256)),
        #         transforms.CenterCrop(224),
        #         # transforms.Resize((96,96)),
        #         transforms.ToTensor(), # converts to [0,1] and (C,H,W)
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ])

    def print_size(self, name):
        print("Size of {} dataset: {}".format(name, len(self)))
        

    def __len__(self):
        # return len(self.indices)
        return self.audio_data_info.shape[0]-1
    
    def get_audio(self, audio_data_info):
        """
        Input: audio_data_info: shape (2,) np array
        Output: audio_data: shape (57,100)
        """
        file_path = audio_data_info[0]
        assert os.path.exists(file_path), "Audio file path does not exist"
        audio_label = np.array(audio_data_info[1]).astype(float)

        # Process each audio individually
        audio_data = process_audio(file_path, sample_rate=16000, num_freq_bins=100, num_time_bins=57) # shape (57, 100)

        return audio_data, audio_label  
    
    def __getitem__(self, idx):

        nsample = {}
        # Processing for Audio
        audio_data_info = self.audio_data_info[idx]       
        audio_data, audio_label = self.get_audio(audio_data_info)
        nsample['audio'] = torch.from_numpy(np.array(audio_data))

        ## Make one hot encoding of audio label
        one_hot = torch.zeros(self.num_classes)
        one_hot[audio_label] = 1.0
        nsample['audio_label'] = one_hot
            
        return nsample

if __name__=="__main__":
    # Just for testing
    dataset_path = '/home/ros_ws/dataset/audio_classes'
    assert os.path.exists(dataset_path), "Dataset path does not exist"

    ## Split data into train, val set
    split_data(dataset_path, split_ratio=0.8)

    plot_raw_audio_data(dataset_path)

    # dataset = AudioDataset(dataset_path,num_classes=3)

    # print(dataset.audio_data_info.shape)
    
    # # iterate over the dataset and print shapes of each element
    # ### Create dataloader
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=128,
    #     num_workers=1,
    #     shuffle=False,
    #     # accelerate cpu-gpu transfer
    #     pin_memory=True,
    #     # don't kill worker process afte each epoch
    #     persistent_workers=False
    # )  
    
    # print("Len of dataloader: ", len(dataloader))
    
    # # iterate over the dataset and print shapes of each element
    # for i, data in enumerate(dataloader):
    #     print("Batch: ", i)
    #     print("size of audio: ", data['audio'].shape)
    #     # print min and max of audio
    #     print("Min of audio: ", torch.min(data['audio']))
    #     print("Max of audio: ", torch.max(data['audio']))

        # plot_audio_data(data['audio'][0].numpy(), data['audio_label'][0].numpy(),save_dir='/home/ros_ws/dataset/audio_classes/results',save_prefix=str(i))