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
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import plotly.figure_factory as ff
### Utility functions for Dataset class

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    # TODO ag6 put some comments to explain this
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
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
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

    observations = np.array(observations, dtype=object)
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
            actions.append([x,y,z,quat_x, quat_y, quat_z, quat_w, gripper_width])
        elif(mode=='xyz_euler'):
            # convert to euler
            roll, pitch, yaw = tf.transformations.euler_from_quaternion([quat_x, quat_y, quat_z, quat_w])
            actions.append([x,y,z,roll, pitch, yaw, gripper_width])
        elif(mode=='xyz'):
            actions.append([x,y,z,gripper_width])

    actions = np.array(actions)

    ## Sanity check if mode is xyz_quat
    if(mode=='xyz_quat'):
        assert actions.shape[1] == 8,"actions should be of shape (N,7) but is of shape {}".format(actions.shape)

    if(mode=='xyz_euler'):
        assert actions.shape[1] == 7,"actions should be of shape (N,6) but is of shape {}".format(actions.shape)

    return actions

def initialize_data(is_state_based=True, is_audio_based=False):
        
    data = OrderedDict()
    if is_state_based is False:
        data["image_data_info"] = []
        
    if is_audio_based is True:
        data["audio_data_info"] = []
        
    data['nagent_pos'] = []
    data['actions'] = []
    data['terminals'] = []

    return data

def print_data_dict_shapes(train_data: OrderedDict):
    # print separator
    print("--------------------------------------------------")
    print("Train_data dict:")
    for key, data in train_data.items():
        if isinstance(data, list):
            print(key, len(data))
        if isinstance(data, np.ndarray):
            print(key, data.shape)
    print("--------------------------------------------------")


def get_stacked_sample(observations, terminals, seq_len, start_idx):
    """
    Input Shapes
    - Observation: (N, ob_dim)
    - Terminals: (N, 1)
    - Previous Observations: (N, ob_dim)
    This functions puts zero padding in the start if there is a terminal state in between!
    """
    end_idx = start_idx + seq_len
    # check if there is a terminal state between start and end, if yes then shift the start_idx
    # dataloader repeats the first observation for missing_context times in such cases
    for idx in range(start_idx, end_idx):
        if terminals[idx]:
            start_idx = idx + 1
    missing_context = seq_len - (end_idx - start_idx)
    
    if missing_context > 0:
        # frames = [np.zeros_like(observations[0])] * missing_context
        frames = []
        # repeat the first observation for missing_context times
        for idx in range(missing_context):
            frames.append(observations[start_idx])
        for idx in range(start_idx, end_idx):
            frames.append(observations[idx])
        frames = np.stack(frames)
        return frames
    else:
        return observations[start_idx:end_idx] # shape (seq_len, ob_dim)

def get_stacked_action(actions, terminals, seq_len, start_idx):
    """
    get_stacked_samples cuts from the start_idx
    This function pads the end with the last action
    """
    end_idx = start_idx + seq_len
    # check if there is a terminal state between start and end
    for idx in range(start_idx, end_idx):
        if terminals[idx]:
            end_idx = idx + 1
            break
    missing_context = seq_len - (end_idx - start_idx)
    
    # pad the end with the last action
    if missing_context > 0:
        frames = []
        for idx in range(start_idx, end_idx):
            frames.append(actions[idx])
        frames += [actions[end_idx - 1]] * missing_context
        frames = np.stack(frames)
        return frames
    else:
        return actions[start_idx:end_idx] # shape (seq_len, ob_dim)

def get_stacked_samples(observations, actions, 
                        image_data_info, audio_data_info,
                        terminals, ob_seq_len, ac_seq_len,
                        batch_size, start_idxs=None):
    """
    Observations: (N, ob_dim)
    Actions: (N, ac_dim)
    Image_data_info: (N, 2)
    Audio_data_info: (N, 2)
    Terminals: (N, 1)
    Returns a batch of stacked samples
        - Observations: (batch_size, ob_seq_len, ob_dim)
        - Actions: (batch_size, ac_seq_len, ac_dim)
        - Image_data_info: (batch_size, ob_seq_len, 2)
        - Audio_data_info: (batch_size, 1, 2)
    Padding:
        - Observations: zero padding at the start
        - Actions: last action padding at the end
    """
    if start_idxs is None:
        start_idxs = np.random.randint(0, len(observations) - ob_seq_len - ac_seq_len, batch_size)
        
    stacked_observations = []
    stacked_actions = []
    stacked_image_data_info = []
    stacked_audio_data_info = []
        
    ### TODO: For loop is not needed here!
    for start_idx in start_idxs:
        obs = get_stacked_sample(observations, terminals, ob_seq_len, start_idx)
        ac = get_stacked_action(actions, terminals, ac_seq_len, start_idx + ob_seq_len - 1)
        stacked_observations.append(obs)
        stacked_actions.append(ac)
        
        if image_data_info is not None:
            im = get_stacked_sample(image_data_info, terminals, ob_seq_len, start_idx)
            stacked_image_data_info.append(im)

        if audio_data_info is not None:
            au = get_stacked_sample(audio_data_info, terminals, 1, start_idx)
            stacked_audio_data_info.append(au)
        
    if image_data_info is None and audio_data_info is None:
        return np.stack(stacked_observations), np.stack(stacked_actions), None, None
    elif image_data_info is None:
        return np.stack(stacked_observations), np.stack(stacked_actions), None, np.stack(stacked_audio_data_info)    
    elif audio_data_info is None:
        return np.stack(stacked_observations), np.stack(stacked_actions), np.stack(stacked_image_data_info), None
    
    # (batch_size, seq_len, ob_dim), (batch_size, ac_seq_len, ac_dim), (batch_size, seq_len, 2)
    return np.stack(stacked_observations), np.stack(stacked_actions), np.stack(stacked_image_data_info), np.stack(stacked_audio_data_info)


def plot_audio_data(audio_data, audio_label, save_dir:str,save_prefix:str):
    """ Plot audio data. Incomplete function!! """
    raise NotImplementedError

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    assert os.path.exists(save_dir), "save_dir does not exist"
    # from matplotlib import pyplot as plt
    # # plt.figure(figsize=(20,10))
    # plt.plot(audio_data)
    # plt.title(audio_label)
    # plt.savefig(os.path.join(save_dir, audio_label+".png"))

    # use plotly to plot data and save 
    import plotly.graph_objects as go
    import plotly.io as pio
    x_data = audio_data[0,:]
    y_data = audio_data[:]
    trace = go.Scatter(x=x_data, y=y_data, mode='lines', name='Line Plot')
    fig = go.Figure(data=go.Scatter(y=audio_data))
    layout = go.Layout(title='Raw audio data', xaxis=dict(title='X-axis'), yaxis=dict(title='Y-axis'))
    fig = go.Figure(data=[trace], layout=layout)
    pio.write_image(fig, os.path.join(save_dir,save_prefix+".png"), format='png')

    # plot 3d spectogram of audio frequencies over time
    # Use plotly
    import plotly.graph_objects as go
    import plotly.io as pio
    import plotly.express as px
    import plotly.figure_factory as ff
    import numpy as np

    go.Figure(data=[go.Histogram3d(x=x_data, y=y_data, z=z_data, opacity=0.7)])
    fig = go.Figure(data=[trace])
    pio.write_image(fig, os.path.join(save_dir,save_prefix+"_spectogram.png"), format='png')


def plot_raw_audio_data_mean(dataset_path:str):
    """ Plot the distibution of mean values for all classes, 'coins','no coins', 'other'"""

    label2name = {0: "coins", 1: "box", 2: "other"}
    name2label = {"coins": 0, "box": 1, "other": 2}

    # assert path is valid
    assert os.path.exists(dataset_path), "dataset_path does not exist"

    save_dir = os.path.join(dataset_path,"results")

    coins_path = os.path.join(dataset_path, "0")
    box_path = os.path.join(dataset_path, "1")
    other_path = os.path.join(dataset_path, "2")

    # read all data files one by one
    coins_files = [[os.path.join(coins_path, f),int(name2label["coins"])] for f in sorted(os.listdir(coins_path)) if f.endswith('.npy')]
    box_files = [[os.path.join(box_path, f),int(name2label["box"])] for f in sorted(os.listdir(box_path)) if f.endswith('.npy')]
    other_files = [[os.path.join(other_path, f),int(name2label['other'])] for f in sorted(os.listdir(other_path)) if f.endswith('.npy')]


    coins_mean = []
    box_mean = []
    other_mean = []

    for f in coins_files:
        audio_data = np.load(f[0]).astype(np.uint8)

        print(" Raw audio coins", audio_data)

        coins_mean.append(np.mean(audio_data))

    for f in box_files:
        audio_data = np.load(f[0]).astype(np.uint8)
        box_mean.append(np.mean(audio_data))
    
    for f in other_files:
        audio_data = np.load(f[0]).astype(np.uint8)
        other_mean.append(np.mean(audio_data))

    # print("coins_mean: ",coins_mean)
    # print("box_mean: ",box_mean)
    # print("other_mean: ",other_mean)
    
    # Fit a gussian distribution to the data points in a single plot
    
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=coins_mean, name='coins'))
    fig.add_trace(go.Histogram(x=box_mean, name='box'))
    fig.add_trace(go.Histogram(x=other_mean, name='other'))

    # Overlay histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.5)
    pio.write_image(fig, os.path.join(save_dir,"raw_audio_data_combined.png"), format='png')

    # Make separate histograms for each class
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=coins_mean, name='coins'))
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.5)
    pio.write_image(fig, os.path.join(save_dir,"raw_audio_data_coins.png"), format='png')

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=box_mean, name='box'))
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.5)
    pio.write_image(fig, os.path.join(save_dir,"raw_audio_data_box.png"), format='png')


    fig = go.Figure()
    fig.add_trace(go.Histogram(x=other_mean, name='other'))
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.5)
    pio.write_image(fig, os.path.join(save_dir,"raw_audio_data_other.png"), format='png')

def convert_uint8_to_s16le(uint8_array):
    # Check if the length of the array is even
    if len(uint8_array) % 2 != 0:
        raise ValueError("The length of the array should be even")

    # Reshape the array to have 2 columns, each row will represent a 16-bit sample
    reshaped_array = uint8_array.reshape(-1, 2)

    # Convert the two uint8 values to one 16-bit value (S16LE format)
    # The first byte is the least significant byte, and the second byte is the most significant
    s16le_array = reshaped_array[:, 0].astype(np.int16) + (reshaped_array[:, 1].astype(np.int16) << 8)

    return s16le_array

def plot_raw_audio_data(dataset_path:str):
    """ Plot the tsne of audio features for all classes, 'coins','no coins', 'other'"""

    label2name = {0: "coins", 1: "box", 2: "other"}
    name2label = {"coins": 0, "box": 1, "other": 2}

    # assert path is valid
    assert os.path.exists(dataset_path), "dataset_path does not exist"

    save_dir = os.path.join(dataset_path,"results")

    coins_path = os.path.join(dataset_path, "0")
    box_path = os.path.join(dataset_path, "1")
    other_path = os.path.join(dataset_path, "2")

    # read all data files one by one
    coins_files = [[os.path.join(coins_path, f),int(name2label["coins"])] for f in sorted(os.listdir(coins_path)) if f.endswith('.npy')]
    box_files = [[os.path.join(box_path, f),int(name2label["box"])] for f in sorted(os.listdir(box_path)) if f.endswith('.npy')]
    other_files = [[os.path.join(other_path, f),int(name2label['other'])] for f in sorted(os.listdir(other_path)) if f.endswith('.npy')]

    idx=100
    coin_file = coins_files[idx]
    box_file = box_files[idx]
    other_file = other_files[idx]
    scale=255.0

    coin_data = convert_uint8_to_s16le(np.load(coin_file[0]).astype(np.uint8))
    box_data = convert_uint8_to_s16le(np.load(box_file[0]).astype(np.uint8))
    other_data = convert_uint8_to_s16le(np.load(other_file[0]).astype(np.uint8))
    print(np.count_nonzero(coin_data>100))

    # Plot the raw audio data using plotly
    plotly_data = [go.Scatter(y=coin_data, name='coins'),
                go.Scatter(y=box_data, name='box'),
                go.Scatter(y=other_data, name='other')]

    fig = go.Figure(data=plotly_data)
    pio.write_image(fig, os.path.join(save_dir,"raw_audio_data.png"), format='png')

    # make seprate plots
    fig = go.Figure(data=go.Scatter(y=coin_data))
    pio.write_image(fig, os.path.join(save_dir,"raw_audio_data_coins.png"), format='png')

    fig = go.Figure(data=go.Scatter(y=box_data))
    pio.write_image(fig, os.path.join(save_dir,"raw_audio_data_box.png"), format='png')

    fig = go.Figure(data=go.Scatter(y=other_data))
    pio.write_image(fig, os.path.join(save_dir,"raw_audio_data_other.png"), format='png')




