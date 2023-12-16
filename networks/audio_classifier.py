# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3

import pickle
import numpy as np
import torch
import sys
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as T_f
import time
import os
from torch.utils.tensorboard import SummaryWriter
sys.path.append("/home/ros_ws/")
sys.path.append("/home/ros_ws/dataset")
from src.il_packages.manipulation.src.moveit_class import get_mat_norm, get_posestamped, get_pose_norm
from dataset.audio_classifier_dataset import AudioDataset

# Adapted from https://sites.google.com/view/playitbyear/
class AudioEncoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, audio_steps, audio_bins):
        super().__init__()
        self.audio_steps = audio_steps #how many timesteps (~57)
        self.audio_bins = audio_bins #how many bins (~100)

        self.audioConvs = nn.Sequential(
            nn.Conv1d(self.audio_bins, 64, kernel_size = 7),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size = 7),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size = 7),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size = 7),
            nn.ReLU(),
        )
        self.out_dim = 8 * (self.audio_steps - 24)

    # 1D convolution acts on the audio_bins dimension, which correlates with frequency
    def audio_forward_conv(self, audio):
        # Audio shape: (batch_size, audio_steps, audio_bins)
        audio = audio.transpose(1, 2) # (batch_size, audio_bins, audio_steps)
        h = self.audioConvs(audio) # (batch_size, 8, audio_steps - 24)
        h = h.view(h.size(0), -1) # (batch_size, 8 * (audio_steps - 24))
        return h

    def forward(self, audio):
        #forward prop
        h_aud = self.audio_forward_conv(audio)
        return h_aud


class AudioCNN(nn.Module):
    """ Convolutional network for audio classification."""
    def __init__(self, train_params: dict):
        super().__init__()
        self.audio_steps = train_params['audio_steps']
        self.audio_bins = train_params['audio_bins']
        self.num_classes = train_params['num_classes']
        self.lossfn = train_params['loss']()
        self.lr = train_params['learning_rate']

        self.audio_encoder = AudioEncoder(self.audio_steps, self.audio_bins)
        self.encoder_out_dim = self.audio_encoder.out_dim
        self.fc = nn.Sequential(
            nn.Linear(self.encoder_out_dim,self.encoder_out_dim//2),
            nn.ReLU(),
            nn.Linear(self.encoder_out_dim//2, self.num_classes),
        )

        self.nets = nn.ModuleDict({
            'audio_encoder': self.audio_encoder,
            'fc': self.fc,
        })

        self.optimizer = torch.optim.AdamW(
            params=self.nets.parameters(),
            lr=self.lr, weight_decay=1e-6)

    def forward(self, audio):
        #forward prop
        h_aud = self.audio_encoder(audio)
        output = self.fc(h_aud)
        return output

    def train_model_step(self, audio, targets):
        """
        Performs a single training step
        """
        # Forward pass
        output = self.forward(audio)
        # print("Model output shape=", output.shape)
        # Compute loss
        loss = self.lossfn(output, targets)
        self.optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update
        self.optimizer.step()
        
        return loss.item()

    def eval_model(self, audio, targets):
        """
        Evaluates the model on the given data
        """
        # Forward pass
        output = self.forward(audio)
        # Compute loss
        loss = self.lossfn(output, targets)
        return loss.item()

    
class AudioTrainer():

    def __init__(self, train_params, data_params, eval_every=100):
        self.eval_every = eval_every
        self.train_params = train_params
        self.data_params = data_params
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
        dataset = AudioDataset(
            dataset_path=data_params['dataset_path'],
            num_classes=train_params['num_classes'],
        )
        print("################ Train Dataset loaded #################")

        
        eval_dataset = AudioDataset(
            dataset_path=data_params['eval_dataset_path'],
            num_classes=train_params['num_classes'],
        )
        print("################ Eval Dataset loaded #################")
        
        ## Store stats
        # self.stats = dataset.stats

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
        dataset.print_size("train")
        
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
        eval_dataset.print_size("eval")

        # Construct input dims to model
        obs_dim = 8*(57-24)
        
        self.train_params["obs_dim"] = obs_dim
        self.train_params["ac_dim"] = train_params["num_classes"]
        self.train_params["num_batches"] = len(self.dataloader)
        # self.train_params["stats"] = self.stats
      
        self.lossfn = train_params["loss"]()
        self.model = AudioCNN(self.train_params).to(self.device)
        
        self.best_eval_loss = 1e10

    def train_model(self):
        global_step = 0
        for epoch_idx in range(self.train_params["num_epochs"]):
            epoch_loss = list()
            print("-----Epoch {}-----".format(epoch_idx))
            
            # evaluate model
            eval_loss = self.evaluate_model()
            print("Eval loss: {}".format(eval_loss))
            self.writer.add_scalar('Loss/eval', eval_loss, global_step)
            if eval_loss < self.best_eval_loss:
                self.best_model_epoch = epoch_idx
                self.best_eval_loss = eval_loss
                self.save_model()
                
            # batch loop
            for nbatch in self.dataloader:  
                # data normalized in dataset

                # device transfer
                # convert ti float 32
                # print data shape
                # print(" Audio data shape", nbatch['audio'].shape)
                # print(" Audio label shape", nbatch['audio_label'].shape)
                nbatch = {k: v.float() for k, v in nbatch.items()}                      
                naudio = nbatch['audio'].to(self.device)
                targets = nbatch['audio_label'].to(self.device)
                B = naudio.shape[0]

                loss = self.model.train_model_step(naudio, targets)

                # logging
                loss_cpu = loss
                epoch_loss.append(loss_cpu)
                
                # log to tensorboard
                self.writer.add_scalar('Loss/train', loss_cpu, global_step)
                global_step += 1
                
                if(not global_step%20):
                    print("Epoch: {}, Step: {}, Loss: {}".format(epoch_idx, global_step, loss_cpu))
                
    def save_model(self, step=None):
        save_dict = {}
        save_dict["model_weights"] = self.model.state_dict()
        save_dict["best_model_epoch"] = self.best_model_epoch
        
        # add train params to save_dict
        save_dict.update(self.train_params)

        # Save the model (mean net and logstd [nn.Parameter] in on dict)
        if not os.path.exists('/home/ros_ws/logs/models'):
            os.makedirs('/home/ros_ws/logs/models')
        torch.save(save_dict, '/home/ros_ws/logs/models/' + self.experiment_name_timed + '.pt')

    def evaluate_model(self):
        """
        Evaluates a given model on a given dataset
        Saves the model if the test loss is the best so far
        """
        # Evaluate the model by computing the MSE on test data
        total_loss = 0
        # Put all params in eval mode
        self.put_on_eval()
        # iterate over all the test data
        for nbatch in self.eval_dataloader:
            # data normalized in dataset
            # device transfer
            nbatch = {k: v.float() for k, v in nbatch.items()}
            naudio = nbatch['audio'].to(self.device)
            targets = nbatch['audio_label'].to(self.device)
            B = naudio.shape[0]

            loss = self.model.eval_model(naudio, targets)
            total_loss += loss*B
        
        ## Put on training after eval
        self.put_on_train()

        return total_loss/len(self.eval_dataloader.dataset)

    def put_on_eval(self):
        self.model.eval()

    def put_on_train(self):
        self.model.train()