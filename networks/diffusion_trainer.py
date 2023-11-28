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

    def __init__(self,):

        super().__init__()

        dataset_path = '/home/ros_ws/dataset/data/toy_expt_first_try'
        self.is_state_based = True

        # parameters
        self.pred_horizon = 16
        self.obs_horizon = 2
        self.action_horizon = 8

        # create dataset from file
        dataset = DiffusionDataset(
            dataset_path=dataset_path,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon
        )
 
        # save training data statistics (min, max) for each dim
        self.stats = dataset.stats

        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            num_workers=4,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )

        self.vision_encoder = get_vision_encoder('resnet18')

        # ResNet18 has output dim of 512
        vision_feature_dim = 512
        # agent_pos is 2 dimensional
        lowdim_obs_dim = 8
        # observation feature has 514 dims in total per step
        self.obs_dim = vision_feature_dim + lowdim_obs_dim
        self.action_dim = 8

        # if(self.is_state_based):
        #     self.obs_dim = 8 # no images
        
        # create network object
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon
        )

        # the final arch has 2 parts
        self.nets = nn.ModuleDict({
            'vision_encoder': self.vision_encoder,
            'noise_pred_net': self.noise_pred_net
        })

               # for this demo, we use DDPMScheduler with 100 diffusion iterations
        self.num_diffusion_iters = 100
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

        # device transfer
        self.device = torch.device('cuda')
        _ = self.nets.to(self.device)

        # setup training params
        self.num_epochs = 100

        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        # Comment (Abhinav): EMA Model in diffusers has been updated. CHeck this file for reference.
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py    
        self.ema = EMAModel(
            self.nets,
            power=0.75)

        # Standard ADAM optimizer
        # Note that EMA parametesr are not \

        # Question(Abhinav): Why optimizing vision encoder???

        self.optimizer = torch.optim.AdamW(
            params=self.nets.parameters(),
            lr=1e-4, weight_decay=1e-6)

        # Cosine LR schedule with linear warmup
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.dataloader) * self.num_epochs
        )

    
    def run_network_demo(self,):

        # demo
        with torch.no_grad():
            # example inputs
            image = torch.zeros((1, self.obs_horizon,3,96,96))
            agent_pos = torch.zeros((1, self.obs_horizon, 2))
            # vision encoder
            image_features = self.nets['vision_encoder'](
                image.flatten(end_dim=1))
            # (2,512)
            image_features = image_features.reshape(*image.shape[:2],-1)
            # (1,2,512)
            obs = torch.cat([image_features, agent_pos],dim=-1)
            # (1,2,514)

            noised_action = torch.randn((1, self.pred_horizon, self.action_dim))
            diffusion_iter = torch.zeros((1,))

            # the noise prediction network
            # takes noisy action, diffusion iteration and observation as input
            # predicts the noise added to action
            noise = self.nets['noise_pred_net'](
                sample=noised_action,
                timestep=diffusion_iter,
                global_cond=obs.flatten(start_dim=1))

            # illustration of removing noise
            # the actual noise removal is performed by NoiseScheduler
            # and is dependent on the diffusion noise schedule
            denoised_action = noised_action - noise

    def run_traning_demo(self,):

        with tqdm(range(self.num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(self.dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # device transfer

                        # convert ti float 32
                        nbatch = {k: v.float() for k, v in nbatch.items()}
                      
                        if(self.is_state_based):
                            nimage = torch.zeros((nbatch['nagent_pos'].shape[0], self.obs_horizon,3,96,96)).to(self.device)
                        else:
                            nimage = nbatch['image'][:,:self.obs_horizon].to(self.device)
                  
                        nagent_pos = nbatch['nagent_pos'][:,:self.obs_horizon].to(self.device)
                        naction = nbatch['actions'].to(self.device)
                        B = nagent_pos.shape[0]

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
                        loss = nn.functional.mse_loss(noise_pred, noise)

                        # optimize
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        self.lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        self.ema.step(self.nets)

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))

        # Weights of the EMA model
        # is used for inference
        ema_nets = self.ema.averaged_model

    
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
    