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
from diffusion_model import DiffusionTrainer
from model_trainer import ModelTrainer   
import sys
sys.path.append("/home/ros_ws/")
from model_utils import seed_everything

if __name__ == "__main__":

    seed_everything(0)

    data_params = {
        "dataset_path": '/home/ros_ws/dataset/data/toy_expt_first_try/train',
        'eval_dataset_path': '/home/ros_ws/dataset/data/toy_expt_first_try/eval',
        "is_state_based": True,
        "pred_horizon": 16,  # must be a multiple of 2
        "obs_horizon": 2,
        "action_horizon": 8, # MPC number of actions to take
    }

    # Make the train params
    train_params = {
        "batch_size": 64,
        "eval_batch_size": 64,
        "num_workers": 4,
        "num_epochs": 200,
        "learning_rate": 1e-4,
        "loss": nn.functional.mse_loss,
        'use_stats': False,
        'model_class': DiffusionTrainer,
        'num_diffusion_iters': 100,
        'num_ddim_iters': 10, # for DDIM sampling
        'device': 'cuda:0',
    }

    train_params["experiment_name"] = train_params['model_class'].__name__ + \
                                    '_lr_' + str(train_params['learning_rate']) + \
                                    '_bs_' + str(train_params['batch_size']) + \
                                    '_epochs_' + str(train_params['num_epochs']) + \
                                    '_loss_' + train_params['loss'].__name__

    model_trainer = ModelTrainer(train_params, data_params)
    model_trainer.train_model()
