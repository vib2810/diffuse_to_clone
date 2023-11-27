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
from geometry_msgs.msg import Pose
import scipy
import tf
from collections import OrderedDict
### Utility functions for Dataset class

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx,do_not_sample_labels=[]):
    result = dict()
    for key, input_arr in train_data.items():
        if (key in do_not_sample_labels):
            continue
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    print("Data shape: ", data.shape)
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    print("Stats: ", stats)
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def parse_poses(tool_poses_msg:list, mode='xyz_quat'):
    """ Parse the tool poses into a numpy array (N,7) or (N,6)
        mode: 'xyz_quat' or 'xyz_euler'
        tool_poses_msg: list of geometry_msgs.msg.Pose
    """
    tool_poses = []
    for pose in tool_poses_msg:

        x,y,z = pose.position.x, pose.position.y, pose.position.z
        quat_x, quat_y, quat_z, quat_w = pose.orientation.x, pose.orientation.y, \
                                        pose.orientation.z, pose.orientation.w
        if(mode=='xyz_quat'):
            tool_poses.append([x,y,z,quat_x, quat_y, quat_z, quat_w])
        elif(mode=='xyz_euler'):
            # convert to euler
            roll, pitch, yaw = tf.transformations.euler_from_quaternion([quat_x, quat_y, quat_z, quat_w])
            tool_poses.append([x,y,z,roll, pitch, yaw])

    tool_poses = np.array(tool_poses)

    ## Sanity check if mode is xyz_quat
    if(mode=='xyz_quat'):
        assert tool_poses.shape[1] == 7,"tool_poses should be of shape (N,7)"

    if(mode=='xyz_euler'):
        assert tool_poses.shape[1] == 6,"tool_poses should be of shape (N,6)"

    return tool_poses


def parse_states(observations: list, mode='concat'):
    """ Parse the observations into a numpy array (N,8)
        mode: 'concat' or 'separate'
        observations: list with shape (N,2)
    
    """

    observations = np.array(observations)
    robot_joint_values = np.array([np.array(x[0]) for x in observations])
    gripper_width = observations[:,1]
    gripper_width = np.expand_dims(gripper_width,axis=1)

    # Concat
    if(mode=='concat'):
        state_data = np.concatenate([robot_joint_values, gripper_width], axis=1)
        # Sanity check
        assert state_data.shape[1] == 8, "state_data should be of shape (N,8)"
        return state_data
    
    elif(mode=='separate'):
        state_data = {
            'robot_joint_values': robot_joint_values,
            'gripper_width': gripper_width
        }
        return state_data
    
def parse_actions(actions_msg:list, mode='xyz_quat'):
    """ Parse the actions into a numpy array (N,2)
        actions: list with shape (N,2)
    """
    actions = []
    for ee_state in actions_msg:

        pose, gripper_width = ee_state

        x,y,z = pose.position.x, pose.position.y, pose.position.z
        quat_x, quat_y, quat_z, quat_w = pose.orientation.x, pose.orientation.y, \
                                        pose.orientation.z, pose.orientation.w
        if(mode=='xyz_quat'):
            actions.append([x,y,z,quat_x, quat_y, quat_z, quat_w,gripper_width])
        elif(mode=='xyz_euler'):
            # convert to euler
            roll, pitch, yaw = tf.transformations.euler_from_quaternion([quat_x, quat_y, quat_z, quat_w])
            actions.append([x,y,z,roll, pitch, yaw])

    actions = np.array(actions)

    ## Sanity check if mode is xyz_quat
    if(mode=='xyz_quat'):
        assert actions.shape[1] == 8,"actions should be of shape (N,7) but is of shape {}".format(actions.shape)

    if(mode=='xyz_euler'):
        assert actions.shape[1] == 7,"actions should be of shape (N,6) but is of shape {}".format(actions.shape)

    return actions

def initialize_data():
        
    data = OrderedDict()
    data['images'] = []
    data['states'] = []
    data['actions'] = []
    data['tool_poses'] = []
    data['episode_ends'] = []

    return data

def print_data_dict(train_data: OrderedDict):


    # print separator
    print("--------------------------------------------------")
    print("Train_data dict:")
    for key, data in train_data.items():
        print(key, data.shape)
    print("--------------------------------------------------")
