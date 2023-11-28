# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3

import time
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import sys
sys.path.append("/home/ros_ws/")
from networks.bc_model import BCTrainer
from networks.lstm_model import LSTMTrainer
from networks.residuil_model import ResiduILTrainer
from networks.model_utils import read_pkl_trajectory_data, get_data_stats, normalize_data, get_stacked_samples

class ModelTrainer:
    def __init__(self, train_params, eval_every=100):
        self.eval_every = eval_every
        self.train_params = train_params
       
        # Initialize the writer
        curr_time= time.strftime("%d-%m-%Y_%H-%M-%S")
        self.experiment_name_timed = train_params["experiment_name"] + "_" + curr_time
        logdir = '/home/ros_ws/logs/train_logs/'+ self.experiment_name_timed
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)
        print("Logging to: ", logdir)

        self.writer = SummaryWriter(logdir)

        # Load the data [Train and Noisy Eval]
        pkl_traj_data = read_pkl_trajectory_data(train_params["trajectory_num"], eval_split=0.1)
        self.observations = pkl_traj_data["observations"]
        self.actions = pkl_traj_data["actions"]
        self.prev_observations = pkl_traj_data["previous_observations"]
        self.terminals = pkl_traj_data["terminals"]
        self.noisy_eval_obs = pkl_traj_data["eval_observations"]
        self.noisy_eval_acts = pkl_traj_data["eval_actions"]
        self.noisy_eval_prev_obs = pkl_traj_data["eval_previous_observations"]
        self.noisy_eval_terminals = pkl_traj_data["eval_terminals"]

        # Load clean data for metric logging
        clean_obs, clean_act, clean_prev_obs, clean_terminals, _, _, _= read_pkl_trajectory_data("eval")
        self.noiseless_obs = clean_obs
        self.noiseless_acts = clean_act
        self.noiseless_prev_obs = clean_prev_obs
        self.noiseless_terminals = clean_terminals

        # Add info to train_params
        self.train_params["ob_dim"] = pkl_traj_data["ob_dim"]
        self.train_params["ac_dim"] = pkl_traj_data["ac_dim"]

        if train_params['use_stats']:
            stats = {}
            stats["observations"] = get_data_stats(self.observations)
            stats["actions"] = get_data_stats(self.actions)
            train_params["stats"] = stats
        else:
            train_params["stats"] = None

        # Initialize model
        if str(self.train_params["model_class"]).find("BCTrainer") != -1:
            self.model = BCTrainer(
                train_params=train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
        elif str(self.train_params["model_class"]).find("LSTMTrainer") != -1:
            self.model = LSTMTrainer(
                train_params=train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
        elif str(self.train_params["model_class"]).find("ResiduILTrainer") != -1:
            self.model = ResiduILTrainer(
                train_params=train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
        self.best_eval_loss_noisy = 1e10
            
    def train_model(self, test_eval_split_ratio=0.1):
        """
        Trains a given model on a given dataset
        Assumes the update function of the model takes in a single batch of data and trains the model for 1 step
        """
        obs_train = self.observations; acs_train = self.actions
        prev_obs_train = self.prev_observations; terminals_train = self.terminals
        obs_test = self.noisy_eval_obs; acs_test = self.noisy_eval_acts
        prev_obs_test = self.noisy_eval_prev_obs; terminals_test = self.noisy_eval_terminals
      
        # print size of train and test data
        print("Train data size: ", len(obs_train))
        print("Eval Noisy data size: ", len(obs_test))
        print("Eval Noiseless data size: ", len(self.noiseless_obs))

        self.model.put_network_on_device()

        # Train the model   
        for step in range(self.train_params["num_steps"]):
            # Sample a batch of observations and actions
            if 'seq_len' in self.train_params:
                obs_batch, acs_batch, prev_obs_batch = get_stacked_samples(obs_train, acs_train, terminals_train, 
                                                        self.train_params["seq_len"], self.train_params["batch_size"])
            else:
                # randomly permute batch size samples
                indices = np.random.randint(0, len(obs_train), self.train_params["batch_size"])
                obs_batch = obs_train[indices]
                acs_batch = acs_train[indices]
                prev_obs_batch = prev_obs_train[indices]
            
            # Train the model for 1 step
            loss = self.model.train_model_step(obs_batch, acs_batch, prev_obs_batch, self.train_params["loss"])

            # Log the loss
            if type(loss) == dict:
                for key in loss:
                    self.writer.add_scalar(key, loss[key], step)
            else:
                self.writer.add_scalar("Loss", loss, step)

            # Evaluate the model every eval_every steps
            if step % self.train_params["eval_every"] == 0:
                # Evaluate the model by computing the MSE on test data
                eval_loss_noiseless = self.evaluate_model(self.noiseless_obs, self.noiseless_acts, self.noiseless_prev_obs, self.noiseless_terminals)
                eval_loss_noisy = self.evaluate_model(obs_test, acs_test, prev_obs_test, terminals_test)
                if eval_loss_noisy < self.best_eval_loss_noisy:
                    self.best_eval_loss_noisy = eval_loss_noisy
                    self.save_model(step)
                print("Step: {}, Train Loss: {}, Eval Loss Noiseless: {}, Eval Loss Noisy: {}".format(step, loss, eval_loss_noiseless, eval_loss_noisy))
                self.writer.add_scalar("Eval_Loss", eval_loss_noiseless, step)
                self.writer.add_scalar("Eval_Loss_Noisy", eval_loss_noisy, step)
        
        self.writer.close()
    
    def save_model(self, step=None):
        save_dict = {'model_weights': self.model.state_dict()}
        if step is not None:
            save_dict['step'] = step
        
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
        num_eval_batches = int(len(observations)/self.train_params["batch_size"])
        for eval_step in range(num_eval_batches):
            # Sample the next batch of data
            eval_idxs = np.arange(eval_step*self.train_params["batch_size"], (eval_step+1)*self.train_params["batch_size"])
            if 'seq_len' in self.train_params:
                # get stacked samples
                obs_batch, acs_batch, prev_obs_batch = get_stacked_samples(observations, actions, terminals, 
                                                        self.train_params["seq_len"], self.train_params["batch_size"], start_idxs=eval_idxs)
            else:
                obs_batch = observations[eval_idxs]
                acs_batch = actions[eval_idxs]
                prev_obs_batch = previous_observations[eval_idxs]
            
            loss = self.model.eval_model(obs_batch, acs_batch, prev_obs_batch, self.train_params["loss"])
            total_loss += loss*self.train_params["batch_size"]
        
        test_loss = total_loss/len(observations)
        return test_loss