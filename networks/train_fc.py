# Author: Vibhakar Mohta vib2810@gmail.com
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
from fc_model import FCTrainer
from model_trainer import ModelTrainer
sys.path.append("/home/ros_ws/")
from model_utils import seed_everything

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train_diffusion.py dataset_name")
        exit()
    dataset_name = sys.argv[1]
    seed_everything(0)
    
    data_params = {
        "dataset_path": f'/home/ros_ws/dataset/data/{dataset_name}/train',
        'eval_dataset_path': f'/home/ros_ws/dataset/data/{dataset_name}/eval',
        "is_state_based": False,
        "is_audio_based": True,
        "pred_horizon": 1,  # for FC model
        "obs_horizon": 2,
        "action_horizon": 1, # for FC model
    }
    
    # Make the train params
    train_params = {
        "batch_size": 256,
        "eval_batch_size": 256,
        "learning_rate": 1e-4,
        "n_layers": 4,
        "hidden_size": 1024,
        "num_workers": 8,
        "num_epochs": 400,
        "loss": nn.functional.mse_loss,
        "activation": nn.ReLU(),
        "output_activation": nn.Identity(),
        'model_class': FCTrainer,
        'device': 'cuda:0',
        'audio_cnn_pretrained_ckpt': 'AudioTrainer_dataset_audio_classes_lr_0.0001_bs_256_epochs_300_loss_CrossEntropyLoss_16-12-2023_21-39-02'
    }

    train_params["experiment_name"] = train_params['model_class'].__name__ + \
                                    '_dataset_' + dataset_name + \
                                    '_hidden_size_' + str(train_params['hidden_size']) + \
                                    '_state_based_' + str(data_params['is_state_based']) + \
                                    '_lr_' + str(train_params['learning_rate']) + \
                                    '_bs_' + str(train_params['batch_size']) + \
                                    '_epochs_' + str(train_params['num_epochs']) + \
                                    '_loss_' + train_params['loss'].__name__

    model_trainer = ModelTrainer(train_params, data_params)
    model_trainer.train_model()
