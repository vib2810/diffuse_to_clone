# Author: Abhinav Gupta ag6@andrew.cmu.edu
#!/usr/bin/env python3

import sys
sys.path.append("/home/ros_ws/")
sys.path.append("/home/ros_ws/dataset")
sys.path.append("/home/ros_ws/networks")
import torch
import torch.nn as nn
import numpy as np
from networks.model_utils import normalize_data, unnormalize_data
from dataset.data_utils import *
from networks.diffusion_model_utils import *

class FCTrainer(nn.Module):
    def __init__(self,
                train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
            ):

        super().__init__()

        # init vars
        self.action_dim = train_params["ac_dim"]
        self.obs_dim = train_params["obs_dim"]
        self.obs_horizon = train_params["obs_horizon"]
        self.pred_horizon = train_params["pred_horizon"]
        self.action_horizon = train_params["action_horizon"]
        self.device = train_params["device"]
        self.num_epochs = train_params["num_epochs"]
        self.lr = train_params["learning_rate"]
        self.num_batches = train_params["num_batches"]
        self.stats = train_params["stats"]
        self.is_state_based = train_params["is_state_based"]
        self.is_audio_based = train_params["is_audio_based"]
        self.device = device
        
        assert self.pred_horizon == 1, "FC Model only works for pred_horizon = 1"
        assert self.action_horizon == 1, "FC Model only works for action_horizon = 1"
        
        if(not self.is_state_based):
            # Get vision encoder (obs dim already includes vision features added in the dataloader)
            self.vision_encoder = get_vision_encoder('resnet18', weights='IMAGENET1K_V2')
        
        if(self.is_audio_based):
            # Get audio encoder
            self.audio_encoder = get_audio_encoder(audio_steps=57, audio_bins=100)
            
        # create network object
        self.mean_net = self.make_network(
            input_size=self.obs_dim,
            output_size=self.action_dim*self.pred_horizon,
            n_layers=train_params["n_layers"],
            size=train_params["hidden_size"],
            activation=train_params["activation"],
            output_activation=train_params["output_activation"]
        )
        
        # convert stats to tensors and put on device
        for key in self.stats.keys():
            for subkey in self.stats[key].keys():
                if type(self.stats[key][subkey]) != torch.Tensor:
                    self.stats[key][subkey] = torch.tensor(self.stats[key][subkey].astype(np.float32)).to(self.device)
        
        # put network on device
        self.put_network_on_device()
        
        # loss fn
        self.loss_fn = train_params["loss"]
        
        if self.is_state_based:
            self.optimizer = torch.optim.Adam(
                params=self.mean_net.parameters(),
                lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(
                params=itertools.chain(
                    self.mean_net.parameters(),
                    self.vision_encoder.parameters()
                ),
                lr=self.lr)
    
    def put_network_on_device(self):
        self.mean_net.to(self.device)
        
        if not self.is_state_based:
            self.vision_encoder.to(self.device)
            
        if self.is_audio_based:
            self.audio_encoder.to(self.device)
        
        for key in self.stats.keys():
            for subkey in self.stats[key].keys():
                if type(self.stats[key][subkey]) == torch.Tensor:
                    self.stats[key][subkey] = self.stats[key][subkey].to(self.device)
    
    def make_network(self, input_size, output_size, n_layers, size, activation, output_activation):
        layers = []
        # layers.append(nn.BatchNorm1d(input_size)) # batch normalization
        layers.append(nn.Linear(input_size, size))
        layers.append(activation)
        for i in range(n_layers):
            layers.append(nn.Linear(size, size))
            layers.append(activation)
        layers.append(nn.Linear(size, output_size))
        layers.append(output_activation)
        return nn.Sequential(*layers) # make an nn sequential object with the layers list

    def eval(self):
        # self.nets.eval()
        self.mean_net.eval()
        self.vision_encoder.eval()
        self.audio_encoder.eval()
        if not self.is_state_based:
            self.vision_encoder.eval()
    
    def get_obs_cond(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naudio: torch.Tensor):
        if self.is_state_based:
            return torch.cat([nagent_pos], dim=-1).flatten(start_dim=1)
        else:
            # encoder vision features
            image_features = self.vision_encoder(
                nimage.flatten(end_dim=1)) # shape (B*obs_horizon, D)
            image_features = image_features.reshape(
                *nimage.shape[:2],-1) # (B,obs_horizon,D)

            # concatenate vision feature and low-dim obs
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1) # (B, obs_horizon * obs_dim)
            # Add audio features if audio based
            if self.is_audio_based:
                # encoder vision features
                input_audio = naudio.flatten(end_dim=1) # shape (B, audio_steps, audio_bins)
                audio_features = self.audio_encoder(
                    input_audio) # shape (B, audio_dim)
                obs_cond = torch.cat([obs_cond, audio_features], dim=-1)
            return obs_cond
        
    def get_all_actions_normalized(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naudio: torch.Tensor):
        """
        Returns the actions for the entire horizon (self.pred_horizon)
        Assumes that the data is normalized
        Returns normalized actions of shape (B, pred_horizon, action_dim)
        """
        with torch.no_grad():
            obs_cond = self.get_obs_cond(nimage, nagent_pos, naudio)
            naction = self.mean_net(obs_cond)
            return naction
    
    def initialize_mpc_action(self):
        "Dummy function"
        pass

    def get_mpc_action(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naudio: torch.Tensor):
        """
        Assumes nagent_pos is not normalized
        Assumes image is normalized with imagenet stats
        Dummy function with same name, just return the next action
        """
        nagent_pos = normalize_data(nagent_pos, self.stats['nagent_pos'])
        naction = self.get_all_actions_normalized(nimage, nagent_pos, naudio)
        naction_unnormalized = unnormalize_data(naction, stats=self.stats['actions']) # (B, action_horizon * action_dim) where action_horizon = 1
        assert naction_unnormalized.shape[0] == 1
        return naction_unnormalized.squeeze(0).cpu().numpy()            
        
    def train_model_step(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naudio: torch.Tensor, naction: torch.Tensor):
        self.optimizer.zero_grad()
        obs_cond = self.get_obs_cond(nimage, nagent_pos, naudio)
        model_actions = self.mean_net(obs_cond)

        loss = self.loss_fn(model_actions, naction.flatten(start_dim=1))
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def run_after_epoch(self):
        pass
    
    def eval_model(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naudio: torch.Tensor, naction: torch.Tensor, return_actions=False):
        """
        Input: nimage, nagent_pos, naction in the dataset [normalized inputs]
        Returns the MSE loss between the normalized model actions and the normalized actions in the dataset
        """
        model_actions = self.get_all_actions_normalized(nimage, nagent_pos, naudio)
        loss = self.loss_fn(model_actions, naction.flatten(start_dim=1))
        if return_actions:
            return loss.item(), model_actions
        return loss.item()
 
    def load_model_weights(self, model_weights):
        """
        Load the model weights
        """
        self.put_network_on_device()
        self.load_state_dict(model_weights)