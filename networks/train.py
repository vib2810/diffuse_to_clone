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
from diffusion_trainer import DiffusionTrainer
from model_trainer_V2 import ModelTrainer   
import sys
sys.path.append("/home/ros_ws/")
from model_utils import seed_everything

if __name__ == "__main__":

    seed_everything(0)

    data_params = {
        "dataset_path": '/home/ros_ws/dataset/data/toy_expt_first_try/train',
        'eval_dataset_path': '/home/ros_ws/dataset/data/toy_expt_first_try/eval',
        "is_state_based": True,
        "pred_horizon": 16,
        "obs_horizon": 2,
        "action_horizon": 8,
    }

    # Make the train params
    train_params = {
        "batch_size": 64,
        "eval_batch_size": 64,
        "num_workers": 4,
        "num_epochs": 100,
        "num_steps": 500000,
        "learning_rate": 1e-5,
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
                                    '_steps_' + str(train_params['num_steps']) + \
                                    str(train_params['loss'].__class__.__name__)

    model_trainer = ModelTrainer(train_params, data_params)
    model_trainer.train_model()
