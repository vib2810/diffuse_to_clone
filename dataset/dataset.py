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
from data_utils import *
from preprocess_audio import process_audio
from collections import OrderedDict
from torchvision import transforms

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
class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 is_state_based: bool = True,       # if False, then use image 
                 is_audio_based: bool = False):     # if True, then use audio
        
        self.dataset_path = dataset_path
        self.is_state_based = is_state_based
        self.is_audio_based = is_audio_based
        
        # Assert that there is an Images folder if is_state_based==False
        if self.is_state_based==False:
            assert os.path.exists(os.path.join(dataset_path, "Images")), "Images folder does not exist"   

        if self.is_audio_based==True:
            assert os.path.exists(os.path.join(dataset_path, "Audio")), "Audio folder does not exist" 

        # read all pkl files one by one
        self.files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.pkl')]
        # sort files in ascending order
        self.files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        train_data = initialize_data(is_state_based=is_state_based, is_audio_based=is_audio_based)
        
        for idx, file in enumerate(self.files):
            dataset_root = pickle.load(open(file, 'rb'))
            state_data = parse_states(dataset_root['observations'])
            actions_data = parse_actions(dataset_root['actions'], mode='xyz')
            
            # store the idx of the file and the idx of the datapoint within the file
            if self.is_state_based==False:
                trajectory_idx = int(file.split("/")[-1].split("_")[-1].split(".")[0])
                image_file_idx = np.array([trajectory_idx]*len(state_data))
                image_data_idx = np.arange(len(state_data))
                image_data_info = np.stack((image_file_idx, image_data_idx), axis=1) # shape (N,2)

            if self.is_audio_based==True:
                trajectory_idx = int(file.split("/")[-1].split("_")[-1].split(".")[0])
                audio_file_idx = np.array([trajectory_idx]*len(state_data))
                audio_data_idx = np.arange(len(state_data))
                audio_data_info = np.stack((audio_file_idx, audio_data_idx), axis=1) # shape (N,2)

            episode_length = len(state_data)
                
            terminals = np.concatenate((np.zeros(episode_length-1), np.ones(1))) # 1 if episode ends, 0 otherwise
            
            # Store in global dictionary for all data
            train_data['nagent_pos'].extend(state_data)
            train_data['actions'].extend(actions_data)
            train_data['terminals'].extend(terminals)
            
            if self.is_state_based==False:
                train_data['image_data_info'].extend(image_data_info)

            if self.is_audio_based==True:
                train_data['audio_data_info'].extend(audio_data_info)

        # print train_data dict stats
        train_data['nagent_pos'] = np.array(train_data['nagent_pos']) # shape (N,obs_dim)
        train_data['actions'] = np.array(train_data['actions']) # shape (N,action_dim)
        train_data['terminals'] = np.array(train_data['terminals']) # shape (N,)
        
        if self.is_state_based==False:
            train_data['image_data_info'] = np.array(train_data['image_data_info']) # shape (N,2)

        if self.is_audio_based==True:
            train_data['audio_data_info'] = np.array(train_data['audio_data_info']) # shape (N,2)

        self.state_dim = train_data['nagent_pos'].shape[1]
        self.action_dim = train_data['actions'].shape[1]
        
        # compute statistics and normalize joint states and actions to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        data_to_normalize = ['nagent_pos', 'actions']

        for key, data in train_data.items():
            if key in data_to_normalize:
                stats[key] = get_data_stats(data)
                normalized_train_data[key] = normalize_data(data, stats[key]).astype(np.float32)
            else:
                normalized_train_data[key] = np.array(data).astype(np.float32)
        
        # transforms for image data
        if self.is_state_based==False:
            self.image_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256)),
                transforms.CenterCrop(224),
                # transforms.Resize((96,96)),
                transforms.ToTensor(), # converts to [0,1] and (C,H,W)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # return len(self.indices)
        return self.normalized_train_data['nagent_pos'].shape[0] - self.obs_horizon - self.pred_horizon + 1

    def print_size(self, string):
        print("Dataset {}".format(string))
        ### Store some stats about training data.
        print_data_dict_shapes(self.normalized_train_data)
    
    def get_images(self, image_data_info):
        """
        Input: image_data_info: shape (N,2) np array
        Output: image_data: shape (N,3,224,224)
        """
        # Prepare an array to hold the images in the original order
        image_data = [None] * len(image_data_info)

        for idx in range(len(image_data_info)):
            file_idx = int(image_data_info[idx, 0])
            data_idx = int(image_data_info[idx, 1])
            
            file_path = os.path.join(self.dataset_path, "Images", str(file_idx), str(data_idx)+".png")
            image = cv2.imread(file_path) # shape (480, 640, 3)
            if image is None:
                print("Error reading image: ", file_path)
                continue
            # Transform each image individually
            image = self.image_transforms(image) # shape (3, 224, 224)
            image_data[idx] = image

        # Stack the transformed images into a batch
        image_data = torch.stack(image_data, dim=0) # shape (N,3,224,224)

        return image_data
    
    def get_audio(self, audio_data_info):
        """
        Input: audio_data_info: shape (N,2) np array
        Output: audio_data: shape (N,57,100)
        """
        # Prepare an array to hold the audio in the original order
        audio_data = [None] * len(audio_data_info)

        for idx in range(len(audio_data_info)):
            file_idx = int(audio_data_info[idx, 0])
            data_idx = int(audio_data_info[idx, 1])
            
            file_path = os.path.join(self.dataset_path, "Audio", str(file_idx), str(data_idx)+".npy")
            # audio = np.load(file_path) # shape (32000, 1)
            # print("Shape of audio: ", audio.shape)

            # Process each audio individually
            audio = process_audio(file_path, sample_rate=16000, num_freq_bins=100, num_time_bins=57) # shape (57, 100)
            audio_data[idx] = torch.tensor(audio)

        # Stack the processed audio into a batch
        audio_data = torch.stack(audio_data, dim=0) # shape (N,57,100)
        return audio_data
    
    def __getitem__(self, idx):
        nsample = {}
        stacked_obs, stacked_action, stacked_image_data_info, stacked_audio_data_info = get_stacked_samples(
            observations=self.normalized_train_data['nagent_pos'],
            actions=self.normalized_train_data['actions'],
            image_data_info=None if self.is_state_based else self.normalized_train_data['image_data_info'],
            audio_data_info=None if self.is_audio_based==False else self.normalized_train_data['audio_data_info'],
            terminals=self.normalized_train_data['terminals'],
            ob_seq_len=self.obs_horizon,
            ac_seq_len=self.pred_horizon,
            batch_size=1,
            start_idxs=[idx]
        ) # stacked_image_data_info is None if is_state_based==True

        # convert to tensors
        nsample['nagent_pos'] = torch.from_numpy(stacked_obs[0].astype(np.float32)) # shape (obs_horizon, state_dim)
        nsample['actions'] = torch.from_numpy(stacked_action[0].astype(np.float32)) # shape (pred_horizon, action_dim)
        
        # Processing for Image
        # stacked_image_data_info shape (1, seq_len, 2)
        if not self.is_state_based:            
            # get data for all images required
            stacked_image_data_info = stacked_image_data_info[0] # shape (seq_len, 2)
            image_data = self.get_images(stacked_image_data_info) # shape (1, seq_len, 3, 224, 224)

            # add to nsample
            nsample['image'] = image_data
        
        # Processing for Audio
        # stacked_audio_data_info shape (1, seq_len, 2)
        if self.is_audio_based==True:
            # get data for all audio required
            stacked_audio_data_info = stacked_audio_data_info[0] # shape (seq_len, 2)
            audio_data = self.get_audio(stacked_audio_data_info) # shape (1, seq_len, 57, 100)
            
            # add to nsample
            nsample['audio'] = audio_data            
            
        return nsample


if __name__=="__main__":
    # Just for testing
    dataset = "vision_audio_coins"
    dataset_path = '/home/ros_ws/dataset/data/'+dataset+'/train'
    assert os.path.exists(dataset_path), "Dataset path does not exist"
    dataset = DiffusionDataset(dataset_path=dataset_path,
                                 pred_horizon=1,
                                 obs_horizon=2,
                                 action_horizon=1, 
                                 is_state_based=False,
                                 is_audio_based=True)
    
    # iterate over the dataset and print shapes of each element
    ### Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=8,
        shuffle=False,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=False
    )  
    
    print("Len of dataloader: ", len(dataloader))
    
    # iterate over the dataset and print shapes of each element
    for i, data in enumerate(dataloader):
        print("Batch: ", i)
        print("size of nagent_pos: ", data['nagent_pos'].shape)
        print("size of actions: ", data['actions'].shape)
        if not dataset.is_state_based:
            print("size of image: ", data['image'].shape)
        if dataset.is_audio_based==True:
            print("size of audio: ", data['audio'].shape)
            # print min and max of audio
            print("Min of audio: ", torch.min(data['audio']))
            print("Max of audio: ", torch.max(data['audio']))