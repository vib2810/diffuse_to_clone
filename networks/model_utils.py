# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3

import pickle
import numpy as np
import torch
import sys
sys.path.append("/home/ros_ws/")
from src.il_packages.manipulation.src.moveit_class import get_mat_norm, get_posestamped, get_pose_norm

def extract_obs_acts(expert_observations, expert_actions):
    observations = []
    actions = []
    terminals = []
    previous_observations = []
    for i in range(expert_observations):
        for j in range(len(expert_observations[i])):
            # observations 
            observations.append(torch.tensor(expert_observations[i][j]))
            actions.append(torch.tensor(expert_actions[i][j]))
            
            # append terminals
            if(j == len(expert_observations[i]) - 1):
                terminals.append(torch.tensor([1.0]))
            else:
                terminals.append(torch.tensor([0.0]))
            
            # append previous observations
            if j == 0:
                previous_observations.append(torch.tensor(expert_observations[i][j]))
            else:
                previous_observations.append(torch.tensor(expert_observations[i][j-1]))
    observations = torch.stack(observations).float()
    actions = torch.stack(actions).float()
    terminals = torch.stack(terminals).float()
    previous_observations = torch.stack(previous_observations).float()
    return observations, actions, previous_observations, terminals

def read_pkl_trajectory_data(train_trajectory_num, print_data=False, eval_split=None):
    trajectory_path = '/home/ros_ws/logs/recorded_trajectories/recorded_trajectoriestrajectories_' + str(train_trajectory_num) + '.pkl'

    # Load data from pickle file
    expert_data = pickle.load(open('/home/ros_ws/logs/recorded_trajectories/trajectories_' + str(train_trajectory_num) + '.pkl', 'rb'))
    expert_observations = expert_data["observations"] # N x [(ep_len_i x ob_dim)]
    expert_actions = expert_data["actions"] # N x [(ep_len_i x ac_dim)]
    expert_target_pose = expert_data["target_poses"][0] # Assumes all have same target pose

    # Set ob_dim and ac_dim same as joint_dim
    ob_dim = np.array(expert_observations[0]).shape[-1]
    ac_dim = np.array(expert_actions[0]).shape[-1]

    if print_data:
        print("Loading data from pickle file: ", trajectory_path)
        print("Observations length: ", len(expert_observations))
        print("Actions length: ", len(expert_actions))
        
    n_train_samples = int(len(expert_observations)*(1 - eval_split)) if eval_split is not None else len(expert_observations)

    # Make train observations of shape N x (joint_dim)
    # (expert_observations is a list)
    observations, actions, previous_observations, terminals = extract_obs_acts(expert_observations[:n_train_samples], expert_actions[:n_train_samples])
    ret_dict = {
        "observations": observations,
        "actions": actions,
        "previous_observations": previous_observations,
        "terminals": terminals,
        "ob_dim": ob_dim,
        "ac_dim": ac_dim,
        "expert_target_pose": expert_target_pose
    }
    if eval_split is not None:
        eval_observations, eval_actions, eval_previous_observations, eval_terminals = extract_obs_acts(expert_observations[n_train_samples:], expert_actions[n_train_samples:])
        ret_dict["eval_observations"] = eval_observations
        ret_dict["eval_actions"] = eval_actions
        ret_dict["eval_previous_observations"] = eval_previous_observations
        ret_dict["eval_terminals"] = eval_terminals

    return ret_dict


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # set torch to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Adapted from https://diffusion-policy.cs.columbia.edu/ 
# normalize data
def get_data_stats(data):
    # data shape is (N, ep_len, dim)
    data = data.reshape(-1,data.shape[-1])
    # data shape is (N*ep_len, dim)
    stats = {
        'min': torch.min(data, axis=0).values,
        'max': torch.max(data, axis=0).values,
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # TODO: add checks to not get nans
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# Adapted from https://sites.google.com/view/16-831-cmu/home HW3
def get_stacked_sample(observations, terminals, seq_len, start_idx):
    """
    Input Shapes
    - Observation: (N, ob_dim)
    - Terminals: (N, 1)
    - Previous Observations: (N, ob_dim)
    """
    end_idx = start_idx + seq_len
    # check if there is a terminal state between start and end
    for idx in range(start_idx, end_idx - 1):
        if terminals[idx]:
            start_idx = idx + 1
    missing_context = seq_len - (end_idx - start_idx)
    # if zero padding is needed for missing context
    if start_idx < 0 or missing_context > 0:
        frames = [np.zeros_like(observations[0])] * missing_context
        for idx in range(start_idx, end_idx):
            frames.append(observations[idx])
        frames = np.stack(frames)
        return torch.tensor(frames) # shape (seq_len, ob_dim)
    else:
        return observations[start_idx:end_idx] # shape (seq_len, ob_dim)

def get_stacked_samples(observations, actions, terminals, seq_len, batch_size, start_idxs=None):
    """
    Returns a batch of stacked samples
    """
    if start_idxs is None:
        start_idxs = np.random.randint(1, len(observations) - seq_len, batch_size) # start from 1 so we always have a previous observation
    
    stacked_observations = []
    for start_idx in start_idxs:
        obs = get_stacked_sample(observations, terminals, seq_len, start_idx)
        stacked_observations.append(obs)
    stacked_actions = actions[start_idxs + seq_len - 1] # (batch_size, ac_dim)
    stacked_previous_observations = [get_stacked_sample(observations, terminals, seq_len, start_idx - 1) for start_idx in start_idxs]
    return torch.stack(stacked_observations), stacked_actions, torch.stack(stacked_previous_observations)