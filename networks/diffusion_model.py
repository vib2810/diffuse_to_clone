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
from diffusion_model_vision import *
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import collections
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video

class DiffusionTrainer(nn.Module):

    def __init__(self,
                train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
            ):

        super().__init__()

        # init vars
        self.action_dim = train_params["ac_dim"]
        self.obs_dim = train_params["obs_dim"]
        self.obs_horizon = train_params["obs_horizon"]
        self.device = train_params["device"]
        self.num_diffusion_iters = train_params["num_diffusion_iters"]
        self.num_ddim_iters = train_params["num_ddim_iters"]
        self.num_epochs = train_params["num_epochs"]
        self.lr = train_params["learning_rate"]
        self.num_traj = train_params["num_traj"]
        self.stats = train_params["stats"]
        self.is_state_based = train_params["is_state_based"]
        self.device = device
        

        # create network object
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon
        )
        
        if(not self.is_state_based):
            # Get vision encoder
            self.vision_encoder = get_vision_encoder('resnet18')

            # the final arch has 2 parts
            self.nets = nn.ModuleDict({
                'vision_encoder': self.vision_encoder,
                'noise_pred_net': self.noise_pred_net
            })
        
        else:
            self.nets = nn.ModuleDict({
                'noise_pred_net': self.noise_pred_net
            })

        # for this demo, we use DDPMScheduler with 100 diffusion iterations
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )
        
        self.ddim_sampler = DDIMScheduler(
            num_train_timesteps=self.num_ddim_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        _ = self.nets.to(self.device)

        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        # Comment (Abhinav): EMA Model in diffusers has been updated. CHeck this file for reference.
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py

        self.ema = EMAModel(
            self.nets,
            power=0.75)
        
        # loss fn
        self.loss_fn = train_params["loss"]

        # Standard ADAM optimizer
        # Question(Abhinav): Why optimizing vision encoder???

        self.optimizer = torch.optim.AdamW(
            params=self.nets.parameters(),
            lr=self.lr, weight_decay=1e-6)

        # Cosine LR schedule with linear warmup
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps= self.num_traj* self.num_epochs
        )

    def eval(self):
        self.nets.eval()

    def get_action_eval(self, nimage: torch.Tensor, states: torch.Tensor):
        """
        Forward pass of the network
        """

        ##### This is dummy. We need to change this to our own code
        ##### There's no self.stats['observations'] in our code
        ##### Batches are already normalized in our case so no use of normalization here
        # TODO: vib2810- Verify this!!
        if self.stats:
            obs = normalize_data(obs, self.stats['nagent_pos'])

        with torch.no_grad():
            if(not self.is_state_based):
                # encoder vision features
                image_features = self.nets['vision_encoder'](
                    nimage.flatten(end_dim=1))
                image_features = image_features.reshape(
                    *nimage.shape[:2],-1)
                # (B,obs_horizon,D)

                # concatenate vision feature and low-dim obs
                obs_features = torch.cat([image_features, states], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)
                # (B, obs_horizon * obs_dim)

            else:
                obs_features = torch.cat([states], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)
                # (B, obs_horizon * obs_dim)        
                
            B = states.shape[0]
            
            sample = self.inference_model(obs) # dummy
            
            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            # init scheduler
            self.ddim_sampler.set_timesteps(self.num_ddim_iters)

            for k in self.ddim_sampler.timesteps:
                # predict noise
                noise_pred = self.ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.ddim_sampler.step(
                    model_output=noise_pred,
                    timestep=loss_fn
loss_fnk,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=self.stats['actions'])
        
        return action_pred
        
    def train_model_step(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naction: torch.Tensor):

        if(not self.is_state_based):
            # encoder vision features
            image_features = self.nets['vision_encoder'](
                nimage.flatten(end_dim=1))
            image_features = image_features.reshape(
                *nimage.shape[:2],-1)
            # (B,obs_horizon,D)

            # concatenate vision feature and low-dim obs
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
            # (B, obs_horizon * obs_dim)

        else:
            obs_features = torch.cat([nagent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
            # (B, obs_horizon * obs_dim)

        B = naction.shape[0]
        
        # sample noise to add to actions
        noise = torch.randn(naction.shape, device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            naction, noise, timesteps)

        # predict the noise residual
        noise_pred = self.nets['noise_pred_net'](
            noisy_actions, timesteps, global_cond=obs_cond)

        # L2 loss
        loss = self.loss_fn(noise_pred, noise)

        # optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        self.lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        self.ema.step(self.nets)

        return loss.item()
    
    def eval_model(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naction: torch.Tensor):
        model_actions = self.get_action_eval(nimage, nagent_pos)
        loss = self.loss_fn(model_actions, naction)
        return loss.item()