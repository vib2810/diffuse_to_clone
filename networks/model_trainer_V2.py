# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3

import time
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import sys
sys.path.append("/home/ros_ws/")
from tqdm.auto import tqdm
from networks.bc_model import BCTrainer
import torch
import torch.nn as nn
from dataset.dataset import DiffusionDataset
from diffusion_model import DiffusionTrainer
from lstm_trainer import LSTMTrainer

class ModelTrainer:
    def __init__(self, train_params, data_params, eval_every=100):
        self.eval_every = eval_every
        self.train_params = train_params
        self.data_params = data_params
        self.is_state_based = data_params["is_state_based"]
        self.device = torch.device(train_params["device"]) if torch.cuda.is_available() else torch.device("cpu")

        # Initialize the writer
        curr_time= time.strftime("%d-%m-%Y_%H-%M-%S")
        self.experiment_name_timed = train_params["experiment_name"] + "_" + curr_time
        logdir = '/home/ros_ws/logs/train_logs/'+ self.experiment_name_timed
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)
        print("Logging to: ", logdir)

        self.writer = SummaryWriter(logdir)

        ### Load dataset for training
        dataset = DiffusionDataset(
            dataset_path=data_params['dataset_path'],
            pred_horizon=data_params['pred_horizon'],
            obs_horizon=data_params['obs_horizon'],
            action_horizon=data_params['action_horizon'],
        )
        
        eval_dataset = DiffusionDataset(
            dataset_path=data_params['eval_dataset_path'],
            pred_horizon=data_params['pred_horizon'],
            obs_horizon=data_params['obs_horizon'],
            action_horizon=data_params['action_horizon'],
        )

        ## Store stats
        self.stats = dataset.stats

        ### Create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_params['batch_size'],
            num_workers=train_params['num_workers'],
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )
        
        self.eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=train_params['eval_batch_size'],
            num_workers=train_params['num_workers'],
            shuffle=False,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )

        # Add info to train_params
        self.train_params["obs_dim"] = dataset.obs_dim
        self.train_params["ac_dim"] = dataset.action_dim
        self.train_params["num_traj"] = len(self.dataloader)
        self.train_params["obs_horizon"] = data_params["obs_horizon"]
        self.train_params["pred_horizon"] = data_params["pred_horizon"]
        self.train_params["action_horizon"] = data_params["action_horizon"]
        self.train_params["stats"] = self.stats
        self.train_params["is_state_based"] = self.is_state_based

        # Initialize model
        if str(self.train_params["model_class"]).find("DiffusionTrainer") != -1:
            self.model = DiffusionTrainer(
                train_params=train_params,
                device = self.device if torch.cuda.is_available() else "cpu"
            )
        elif str(self.train_params["model_class"]).find("LSTMTrainer") != -1:
            self.model = LSTMTrainer(
                train_params=train_params,
                device = self.device if torch.cuda.is_available() else "cpu"
            )
        self.best_eval_loss_noisy = 1e10

    def train_model(self, test_eval_split_ratio=0.1):

        with tqdm(range(self.train_params["num_epochs"]), desc='Epoch') as tglobal:
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
                            nimage = None
                        else:
                            nimage = nbatch['image'][:,:self.train_params['obs_horizon']].to(self.device)
                  
                        nagent_pos = nbatch['nagent_pos'][:,:self.train_params['obs_horizon']].to(self.device)
                        naction = nbatch['actions'].to(self.device)
                        B = nagent_pos.shape[0]

                        loss = self.model.train_model_step(nimage, nagent_pos, naction)

                        # logging
                        loss_cpu = loss
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))

        # Weights of the EMA model
        # is used for inference
        ema_nets = self.ema.averaged_model
    
    def save_model(self, step=None):
        save_dict = {'model_weights': self.model.state_dict()}
        
        # add train params to save_dict
        save_dict.update(self.train_params)

        # Save the model (mean net and logstd [nn.Parameter] in on dict)
        if not os.path.exists('/home/ros_ws/logs/models'):
            os.makedirs('/home/ros_ws/logs/models')
        torch.save(save_dict, '/home/ros_ws/logs/models/' + self.experiment_name_timed + '.pt')

    def evaluate_model(self, observations, actions, previous_observations, terminals):
        """
        Evaluates a given model on a given dataset
        Saves the model if the test loss is the best so far
        """
        # Evaluate the model by computing the MSE on test data
        total_loss = 0
        # iterate over all the test data
        with tqdm(self.eval_dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nbatch = {k: v.float() for k, v in nbatch.items()}
                if(self.is_state_based):
                    nimage = None
                else:
                    nimage = nbatch['image'][:,:self.train_params['obs_horizon']].to(self.device)
                    
                nagent_pos = nbatch['nagent_pos'][:,:self.train_params['obs_horizon']].to(self.device)
                naction = nbatch['actions'].to(self.device)
                B = nagent_pos.shape[0]

                loss = self.model.eval_model(nimage, nagent_pos, naction)
                total_loss += loss*B
        
        return total_loss/len(self.eval_dataloader.dataset)