# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append("/home/ros_ws/")
from networks.model_utils import normalize_data, unnormalize_data

class BCTrainer(nn.Module):
    """
    Trainer for Behavior Cloning
    Make a model with given input, output and hidden layers
    """
    def __init__(self,
                train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
            ):
        super().__init__()

        # init vars
        self.ac_dim = train_params["ac_dim"]
        self.ob_dim = train_params["ob_dim"]
        self.n_layers = train_params["n_layers"]
        self.hidden_size = train_params["hidden_size"]
        self.device = device
        self.activation = train_params["activation"]
        self.output_activation = train_params["output_activation"]
        self.stats = train_params["stats"]

        self.mean_net = self.make_network(
            self.ob_dim,
            self.ac_dim,
            self.n_layers,
            self.hidden_size,
            self.activation,
            self.output_activation
        )

        # optimizer
        self.optimizer = optim.Adam(
                self.mean_net.parameters(),
                lr=train_params["learning_rate"]
            )
        print("Setup continuous policy, Network is:")
        print(self.mean_net)
    
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

    def put_network_on_device(self):
        self.mean_net.to(self.device)
        if self.stats is not None:
            for key in self.stats:
                self.stats[key]['min'] = self.stats[key]['min'].to(self.device)
                self.stats[key]['max'] = self.stats[key]['max'].to(self.device)
    
    def eval(self):
        self.mean_net.eval()
        # self.logstd.eval()

    def get_action(self, obs, requires_grad=False):
        """
        Forward pass of the network
        """
        obs = obs.to(self.device)

        if self.stats:
            obs = normalize_data(obs, self.stats['observations'])

        with torch.set_grad_enabled(requires_grad):
            sample = self.mean_net(obs)
        
        if self.stats:
            sample = unnormalize_data(sample, self.stats['actions'])
        
        return sample
    
    def train_model_step(self, batch_observations, batch_actions, batch_prev_observations, loss_fn):
        batch_observations = batch_observations.to(self.device)
        batch_actions = batch_actions.to(self.device)

        self.optimizer.zero_grad()

        model_actions = self.get_action(batch_observations, requires_grad=True)

        loss = loss_fn(model_actions, batch_actions)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_model(self, batch_observations, batch_actions, batch_prev_observations, loss_fn):
        batch_observations = batch_observations.to(self.device)
        batch_actions = batch_actions.to(self.device)

        model_actions = self.get_action(batch_observations, requires_grad=False)

        loss = loss_fn(model_actions, batch_actions)

        return loss.item()

    def load_model_weights(self, model_weights):
        """
        Load the model weights
        """
        self.put_network_on_device()
        self.load_state_dict(model_weights)