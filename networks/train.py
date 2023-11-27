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
import sys
sys.path.append("/home/ros_ws/")
from network_utils import seed_everything

if __name__ == "__main__":

    seed_everything(0)

    # Make the train params
    # train_params = {
    #     "batch_size": 256,
    #     "num_steps": 500000,
    #     "learning_rate": 1e-5,
    #     "n_layers": 4,
    #     "hidden_size": 128,
    #     "eval_every": 100,
    #     "loss": nn.MSELoss(),
    #     "activation": nn.ReLU(),
    #     "output_activation": nn.Identity(),
    #     'use_stats': False,
    #     'model_class': BCTrainer,
    #     'experiment_name': 'bc_model_' + str(train_trajectory_num),
    #     'trajectory_num': train_trajectory_num,
    # }

    model_trainer = DiffusionTrainer()
    model_trainer.run_traning_demo()
