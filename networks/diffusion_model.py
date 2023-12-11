# Author: Abhinav Gupta ag6@andrew.cmu.edu
#!/usr/bin/env python3

import sys
sys.path.append("/home/ros_ws/")
sys.path.append("/home/ros_ws/dataset")
sys.path.append("/home/ros_ws/networks")
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
from networks.diffusion_model_utils import *
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
        self.pred_horizon = train_params["pred_horizon"]
        self.action_horizon = train_params["action_horizon"]
        self.device = train_params["device"]
        self.num_diffusion_iters = train_params["num_diffusion_iters"]
        self.num_ddim_iters = train_params["num_ddim_iters"]
        self.num_epochs = train_params["num_epochs"]
        self.lr = train_params["learning_rate"]
        self.num_batches = train_params["num_batches"]
        self.stats = train_params["stats"]
        self.is_state_based = train_params["is_state_based"]
        self.is_audio_based = train_params["is_audio_based"]
        self.device = device
        
        # create network object
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim
        )
        self.noise_pred_net_eval = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim
        )
        
        if(not self.is_state_based):
            # Get vision encoder
            self.vision_encoder = get_vision_encoder('resnet18', weights='IMAGENET1K_V2')
            self.vision_encoder_eval = get_vision_encoder('resnet18', weights='IMAGENET1K_V2')

            # the final arch has 2 parts
            self.nets = nn.ModuleDict({
                'vision_encoder': self.vision_encoder,
                'noise_pred_net': self.noise_pred_net
            })
            self.inference_nets = nn.ModuleDict({
                'vision_encoder': self.vision_encoder_eval,
                'noise_pred_net': self.noise_pred_net_eval
            })
            
            if self.is_audio_based:
                self.audio_encoder = get_audio_encoder(audio_steps=57, audio_bins=100)
                self.audio_encoder_eval = get_audio_encoder(audio_steps=57, audio_bins=100)
                # add to module dict
                self.nets['audio_encoder'] = self.audio_encoder
                self.inference_nets['audio_encoder'] = self.audio_encoder_eval
                
        
        else:
            self.nets = nn.ModuleDict({
                'noise_pred_net': self.noise_pred_net
            })
            self.inference_nets = nn.ModuleDict({
                'noise_pred_net': self.noise_pred_net_eval
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
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        # put network on device
        _ = self.nets.to(self.device)
        _ = self.inference_nets.to(self.device)
        
        # convert stats to tensors and put on device
        for key in self.stats.keys():
            for subkey in self.stats[key].keys():
                if type(self.stats[key][subkey]) != torch.Tensor:
                    self.stats[key][subkey] = torch.tensor(self.stats[key][subkey].astype(np.float32)).to(self.device)

        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        # Comment (Abhinav): EMA Model in diffusers has been updated. CHeck this file for reference.
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py

        self.ema = EMAModel(
            parameters=self.nets.parameters(),
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
            num_training_steps= self.num_batches* self.num_epochs
        )

    def eval(self):
        # self.nets.eval()
        self.inference_nets.eval()
        
    def get_obs_cond(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naudio: torch.Tensor):
        if self.is_state_based:
            return torch.cat([nagent_pos], dim=-1).flatten(start_dim=1)
        else:
            # encoder vision features
            image_features = self.inference_nets['vision_encoder'](
                nimage.flatten(end_dim=1)) # shape (B*obs_horizon, D)
            image_features = image_features.reshape(
                *nimage.shape[:2],-1) # (B,obs_horizon,D)

            # concatenate vision feature and low-dim obs
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1) # (B, obs_horizon * obs_dim)
            print("Obs cond shape: ", obs_cond.shape)
            # Add audio features if audio based
            if self.is_audio_based:
                # encoder vision features
                audio_features = self.inference_nets['audio_encoder'](
                    naudio.flatten(end_dim=1)) # shape (B, audio_dim)
                print("Audio features shape: ", audio_features.shape)
                
                obs_cond = torch.cat([obs_cond, audio_features], dim=-1)
            return obs_cond
                
    def get_all_actions_normalized(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naudio: torch.Tensor, sampler = "ddim"):
        """
        Sampler: either ddpm or ddim
        Returns the actions for the entire horizon (self.pred_horizon)
        Assumes that the data is normalized
        Returns normalized actions of shape (B, pred_horizon, action_dim)
        """
        if sampler not in ["ddpm", "ddim"]:
            print("Sampler must be either ddpm or ddim")
            return None
        
        with torch.no_grad():
            obs_cond = self.get_obs_cond(nimage, nagent_pos, naudio)    
                
            B = nagent_pos.shape[0]   
            # initialize action from Guassian noise
            noisy_action = torch.randn( (B, self.pred_horizon, self.action_dim) , device=self.device)
            naction = noisy_action

            if sampler == "ddpm":
                eval_sampler = self.noise_scheduler
                eval_itrs = self.num_diffusion_iters
            if sampler == "ddim":
                eval_sampler = self.ddim_sampler
                eval_itrs = self.num_ddim_iters
            
            # init scheduler
            eval_sampler.set_timesteps(eval_itrs)

            for k in eval_sampler.timesteps:
                # predict noise
                noise_pred = self.inference_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = eval_sampler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            
            return naction
    
    def initialize_mpc_action(self):
        self.mpc_actions = []

    def get_mpc_action(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naudio: torch.Tensor, sampler = "ddim"):
        """
        Assumes data is not normalized
        Meant to be called for live control of the robot
        Assumes that the batch size is 1
        """
        # Compute next pred_horizon actions and store the next action_horizon actions in a list
        if len(self.mpc_actions) == 0:          
            nagent_pos = normalize_data(nagent_pos, self.stats['nagent_pos'])
            naction = self.get_all_actions_normalized(nimage, nagent_pos, sampler=sampler)
            naction_unnormalized = naction
            naction_unnormalized = unnormalize_data(naction, stats=self.stats['actions']) # (B, pred_horizon, action_dim)
            
            # append the next action_horizon actions to the list
            for i in range(self.action_horizon):
                self.mpc_actions.append(naction_unnormalized[0][i])
                
        print("MPC Actions: ", len(self.mpc_actions))
        
        # get the first action in the list
        action = self.mpc_actions[0]
        self.mpc_actions.pop(0)
        return action.squeeze(0).cpu().numpy()
            
        
    def train_model_step(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naudio: torch.Tensor, naction: torch.Tensor):
        """
        Input: nimage, nagent_pos, naction in the dataset [normalized inputs]
        Train the model for one step
        Returns the loss        
        """
        obs_cond = self.get_obs_cond(nimage, nagent_pos, naudio)
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
        # PASS all model parameters
        self.ema.step(self.nets.parameters())

        return loss.item()
    
    def run_after_epoch(self):
        self.ema.copy_to(self.inference_nets.parameters())
    
    def eval_model(self, nimage: torch.Tensor, nagent_pos: torch.Tensor, naudio: torch.Tensor, naction: torch.Tensor, return_actions=False, sampler="ddim"):
        """
        Input: nimage, nagent_pos, naction in the dataset [normalized inputs]
        Returns the MSE loss between the normalized model actions and the normalized actions in the dataset
        """
        model_actions_ddim = self.get_all_actions_normalized(nimage, nagent_pos, naudio, sampler=sampler) # (B, pred_horizon, action_dim)
        loss_ddim = self.loss_fn(model_actions_ddim, naction)
        loss = loss_ddim
        if return_actions:
            return loss.item(), model_actions_ddim
        return loss.item()

    def put_network_on_device(self):
        self.nets.to(self.device)
        self.inference_nets.to(self.device)
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