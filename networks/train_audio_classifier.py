# Author: Abhinav Gupta ag6@andrew.cmu.edu
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
sys.path.append("/home/ros_ws/")
from audio_classifier import AudioTrainer
from model_utils import seed_everything

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 train_diffusion.py dataset_name")
        exit()
    dataset_name = sys.argv[1]
    seed_everything(0)
    
    data_params = {
        "dataset_path": f'/home/ros_ws/dataset/{dataset_name}/train',
        'eval_dataset_path': f'/home/ros_ws/dataset/{dataset_name}/eval',
        'audio_steps': 57,
        'audio_bins': 100,
        'num_classes': 3,
    }
    
    # Make the train params
    train_params = {
        "batch_size": 256,
        "eval_batch_size": 256,
        "learning_rate": 1e-4,
        "n_layers": 4,
        "hidden_size": 1024,
        "num_workers": 8,
        "num_epochs": 50,
        "loss": nn.CrossEntropyLoss,
        "activation": nn.ReLU(),
        "output_activation": nn.Identity(),
        'model_class': AudioTrainer,
        'audio_steps': 57,
        'audio_bins': 100,
        'num_classes': 3,
        'device': 'cuda:0',
    }

    train_params["experiment_name"] = train_params['model_class'].__name__ + \
                                    '_dataset_' + dataset_name + \
                                    '_hidden_size_' + str(train_params['hidden_size']) + \
                                    '_lr_' + str(train_params['learning_rate']) + \
                                    '_bs_' + str(train_params['batch_size']) + \
                                    '_epochs_' + str(train_params['num_epochs']) + \
                                    '_loss_' + train_params['loss'].__name__

    model_trainer = AudioTrainer(train_params, data_params)
    model_trainer.train_model()
