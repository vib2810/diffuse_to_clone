# Author: Abhinav Gupta ag6@andrew.cmu.edu
#!/usr/bin/env python3

import sys
sys.path.append("/home/ros_ws/")
sys.path.append("/home/ros_ws/dataset")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import DiffusionDataset
from dataset.data_utils import *
from tqdm.auto import tqdm
import collections
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import hydra

class LSTMTrainer(nn.Module):

    def __init__(self,
                train_params,
                cfg,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
            ):

        super().__init__()

        # init vars
        self.action_dim = train_params["ac_dim"]
        self.obs_dim = train_params["obs_dim"]
        self.obs_horizon = train_params["obs_horizon"]
        self.pred_horizon = train_params["pred_horizon"]
        self.action_horizon = train_params["action_horizon"]
        self.lstm_len = train_params["lstm_len"]
        self.device = train_params["device"]
        self.num_epochs = train_params["num_epochs"]
        self.lr = train_params["learning_rate"]
        self.num_traj = train_params["num_traj"]
        self.stats = train_params["stats"]
        self.is_state_based = train_params["is_state_based"]
        self.device = device

        ## Sanity checks
        assert self.pred_horizon == 1, "LSTM Model only works for pred_horizon = 1"
        assert self.action_horizon == 1, "LSTM Model only works for action_horizon = 1"
        assert self.lstm_len > 1, "LSTM Model needs lstm_len > 1"
        assert not self.is_state_based, "LSTM Model only works for image based models"

        # create network object. Automatically puts on device
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.nets = self.agent.actor
        self.obs_shape = self.agent.obs_shape
        
        # convert stats to tensors and put on device
        for key in self.stats.keys():
            for subkey in self.stats[key].keys():
                if type(self.stats[key][subkey]) != torch.Tensor:
                    self.stats[key][subkey] = torch.tensor(self.stats[key][subkey].astype(np.float32)).to(self.device)

        # loss fn
        self.loss_fn = train_params["loss"]

        # Standard ADAM optimizer
        # Question(Abhinav): Why optimizing vision encoder???

        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), 
                                                lr=self.lr)
    def eval(self):
        self.nets.eval()
        
    def get_all_actions_normalized(self, nimage: torch.Tensor, nagent_pos: torch.Tensor):
        """
        Sampler: either ddpm or ddim
        Returns the actions for the entire horizon (self.pred_horizon)
        Assumes that the data is normalized
        Returns normalized actions of shape (B, action_dim)
        """
        with torch.no_grad():

            dist = self.nets(nagent_pos, nimage)
            agent_action = dist.mean
            assert agent_action.shape[0] == nimage.shape[0]   

            return agent_action
    
    def initialize_mpc_action(self):
        self.mpc_actions = []

    def get_mpc_action(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, sampler = "ddim"):
        """
        Assumes data is not normalized
        Meant to be called for live control of the robot
        Assumes that the batch size is 1
        """
        # Compute next pred_horizon actions and store the next action_horizon actions in a list
        if len(self.mpc_actions) == 0:          
            nagent_pos = normalize_data(nagent_pos, self.stats['nagent_pos'],mode='minmax')
            naction = self.get_all_actions_normalized(nimage, nagent_pos, sampler=sampler)
            naction_unnormalized = naction
            naction_unnormalized = unnormalize_data(naction, stats=self.stats['actions'],mode='minmax') # (B, pred_horizon, action_dim)
            
            # append the next action_horizon actions to the list
            for i in range(self.action_horizon):
                self.mpc_actions.append(naction_unnormalized[0][i])
                
        print("MPC Actions: ", len(self.mpc_actions))
        
        # get the first action in the list
        action = self.mpc_actions[0]
        self.mpc_actions.pop(0)
        return action.squeeze(0).cpu().numpy()
            
        
    def train_model_step(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naction: torch.Tensor):
        # nimage dims
        assert nimage.shape[0] == nagent_pos.shape[0] == naction.shape[0]
        assert nimage.shape[1] == nagent_pos.shape[1] == naction.shape[1]
        assert nimage.shape[2] == 3*self.obs_horizon # 3 channels for rgb image
        assert len(nagent_pos.shape) == 4 # (B, lstm_len, obs_horizon, agentpos_dim)
        assert len(nimage.shape) == 5 # (B, lstm_len, obs_horizon*3, H, W)

        dist = self.nets(nagent_pos, nimage)

        B = naction.shape[0]
        agent_action = dist.mean
        naction = naction[:, -1, :] #pick last action to compare to
        print("Desired action shape", naction.shape)
        loss = self.loss_fn(agent_action, naction)

        # optimize
        loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        return loss.item()
    
    def eval_model(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naction: torch.Tensor, return_actions=False):
        """
        Input: nimage, nagent_pos, naction in the dataset [normalized inputs]
        Returns the MSE loss between the normalized model actions and the normalized actions in the dataset
        """
        model_actions = self.get_all_actions_normalized(nimage, nagent_pos)
        loss = self.loss_fn(model_actions, naction)
        if return_actions:
            return loss.item(), model_actions
        return loss.item()

    def put_network_on_device(self):
        self.nets.to(self.device)
        # put everything in stats on device
        for key in self.stats.keys():
            for subkey in self.stats[key].keys():
                if type(self.stats[key][subkey]) == torch.Tensor:
                    self.stats[key][subkey] = self.stats[key][subkey].to(self.device)
                    
    def load_model_weights(self, model_weights):
        """
        Load the model weights
        """
        self.put_network_on_device()
        self.load_state_dict(model_weights)

