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
        if(train_params["noise_scheduler"] == 'DDPM'):
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
        # self.logstd.eval()

    def get_action(self, obs, requires_grad=False):
        """
        Forward pass of the network
        """

        ##### This is dummy. We need to change this to our own code
        ##### There's no self.stats['observations'] in our code
        ##### Batches are already normalized in our case so no use of normalization here
        # TODO: vib2810- Verify this!!

        obs = obs.to(self.device)

        # if self.stats:
        #     obs = normalize_data(obs, self.stats['observations'])

        with torch.set_grad_enabled(requires_grad):
            sample = self.inference_model(obs) # dummy
        
        # if self.stats:
        #     sample = unnormalize_data(sample, self.stats['actions'])
        
        return sample
        
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
    
    def run_inference_demo(self,):
        
        #@markdown ### **Inference**

        # limit enviornment interaction to 200 steps before termination
        max_steps = 200
        env = PushTImageEnv()
        # use a seed >200 to avoid initial states seen in the training dataset
        env.seed(100000)

        # get first observation
        obs, info = env.reset()

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * self.obs_horizon, maxlen=self.obs_horizon)
        # save visualization and rewards
        imgs = [env.render(mode='rgb_array')]
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon number of observations
                images = np.stack([x['image'] for x in obs_deque])
                agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

                # normalize observation
                nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
                # images are already normalized to [0,1]
                nimages = images

                # device transfer
                nimages = torch.from_numpy(nimages).to(self.device, dtype=torch.float32)
                # (2,3,96,96)
                nagent_poses = torch.from_numpy(nagent_poses).to(self.device, dtype=torch.float32)
                # (2,2)

                # infer action
                with torch.no_grad():
                    # get image features
                    image_features = self.ema_nets['vision_encoder'](nimages)
                    # (2,512)

                    # concat with low-dim observations
                    obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, self.pred_horizon, self.action_dim), device=self.device)
                    naction = noisy_action

                    # init scheduler
                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                    for k in self.noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = self.ema_nets['noise_pred_net'](
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action'])

                # only take action_horizon number of actions
                start = self.obs_horizon - 1
                end = start + self.action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning

                ### TODO: ag6- Use our own code here
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        # print out the maximum target coverage
        print('Score: ', max(rewards))

        # visualize
        from IPython.display import Video
        vwrite('vis.mp4', imgs)
        Video('vis.mp4', embed=True, width=256, height=256)
    