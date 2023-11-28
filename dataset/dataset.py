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
import time
import sys
from data_utils import *
from collections import OrderedDict


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
                 is_state_based: bool = True):

        # read all pkl files one by one
        files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.pkl')]
        train_data = initialize_data()
        self.is_img_available = not is_state_based

        episode_ends = []
        for file in files:
            dataset_root = pickle.load(open(file, 'rb'))
            ### Image data not present
            # image_data = parse_images(dataset_root['images'])   
            state_data = parse_states(dataset_root['observations'])
            actions_data = parse_actions(dataset_root['actions'])
            tool_poses_data = parse_poses(dataset_root['tool_poses'], mode='xyz_quat')
            self.is_img_available = 'images' in dataset_root.keys()
            if self.is_img_available:
                image_data = parse_images(dataset_root['images'])
            else:
                image_data = np.zeros((len(state_data), 3, 96, 96)) # dummy data

            episode_length = len(state_data)
            episode_ends.append(episode_length-1) # index at which ends
            
            # Store in global dictionary for all data
            train_data['nagent_pos'].extend(state_data)
            train_data['actions'].extend(actions_data)
            train_data['images'].extend(image_data)

        # print train_data dict stats
        train_data['nagent_pos'] = np.array(train_data['nagent_pos'])
        train_data['actions'] = np.array(train_data['actions'])
        train_data['images'] = np.array(train_data['images'])
        train_data['episode_ends'] = np.array(episode_ends)

        ### Store some stats about training data
        print_data_dict_shapes(train_data)
        
        self.state_dim = train_data['nagent_pos'].shape[1]
        self.action_dim = train_data['actions'].shape[1]

        if(self.is_img_available):
            self.image_feat_dim = train_data['images'].shape[1]
        else:
            self.image_feat_dim = 0

        self.obs_dim = self.state_dim + self.image_feat_dim

        # # float32, [0,1], (N,96,96,3)
        # train_image_data = dataset_root['data']['img'][:]
        # train_image_data = np.moveaxis(train_image_data, -1,1)
        # # (N,3,96,96)

        # # (N, D)
        # train_data = {
        #     # first two dims of state vector are agent (i.e. gripper) locations
        #     'agent_pos': dataset_root['data']['state'][:,:2],
        #     'action': dataset_root['data']['action'][:]
        # }
        # episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=train_data['episode_ends'],
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)
        
        print("Indices shape: ", indices.shape)

        # # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            print("Key: ", key)
            if key not in ['images','episode_ends']: # save time since we don't have images rn. TODO: ag6: Fix this later
                stats[key] = get_data_stats(data)
                normalized_train_data[key] = normalize_data(data, stats[key])

        # # images are normalized for authors but not for us.
        # normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon


        print('*****************************************************************************')
        print(' **********************Data Loaded Successfully!!!*******************************')
        print('*****************************************************************************')


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        do_not_sample = ['images','episode_ends'] # Not present rn
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
            do_not_sample_labels=do_not_sample
        )

        # discard unused observations
        if(self.is_img_available):
            nsample['image'] = nsample['image'][:self.obs_horizon,:]

        nsample['nagent_pos'] = nsample['nagent_pos'][:self.obs_horizon,:]
        nsample['nagent_pos'] = nsample['nagent_pos'].astype(np.float64)

        ### print dtype of all keys
        # for key, data in nsample.items():
        #     print("Key: ", key, ", dtype: ", data.dtype)
        #     # print("states", nsample['nagent_pos'])

        return nsample
    

if __name__=="__main__":
    dataset_path = '/home/ros_ws/dataset/data/toy_expt_first_try'
    assert os.path.exists(dataset_path), "Dataset path does not exist"
    data_obj = DiffusionDataset(dataset_path=dataset_path,
                                 pred_horizon=16,
                                 obs_horizon=2,
                                 action_horizon=8)
