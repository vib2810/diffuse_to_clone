# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3
import sys
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
import torch

sys.path.append("/home/ros_ws/src/il_packages/manipulation/src")
sys.path.append("/home/ros_ws/")
from networks.diffusion_model import DiffusionTrainer
from networks.fc_model import FCTrainer
from dataset.dataset import DiffusionDataset
from diffusion_model import DiffusionTrainer
from networks.model_utils import normalize_data, unnormalize_data

sys.path.append("/home/ros_ws/networks") # for torch.load to work

import copy
class ModelEvaluator:
    ACTION_HORIOZON = 1
    DDIM_STEPS = 20
    DIFFUSION_SAMPLER = 'ddpm'
    def __init__(self,
            model_name
        ):
        # Initialize the model
        stored_pt_file = torch.load("/home/ros_ws/logs/models/" + model_name + ".pt", map_location=torch.device('cpu'))
        
        # Uncomment to fix weights and store fixed model
        # stored_pt_file_orig = torch.load("/home/ros_ws/logs/models/" + model_name + ".pt", map_location=torch.device('cpu'))
        # stored_pt_file = copy.deepcopy(stored_pt_file_orig)
        # # correct obs_dim to 1040
        # stored_pt_file['obs_dim'] = 1040        
        # # copy of stored_pt_file
        # for key in stored_pt_file_orig['model_weights'].keys():
        #     # for all keys of stored_pt_file['model_weights'] that start with 'noise_pred_net', make a copy with the name 'noise_pred_net_eval'
        #     if key.startswith('noise_pred_net'):
        #         eval_key = key.replace('noise_pred_net', 'noise_pred_net_eval')
        #         if eval_key not in stored_pt_file_orig['model_weights'].keys():
        #             stored_pt_file['model_weights'][eval_key] = copy.deepcopy(stored_pt_file_orig['model_weights'][key])
        #             print("Copying weights from", key, "to", eval_key)
            
        #     if key.startswith('vision_encoder'):
        #         if key.replace('vision_encoder', 'vision_encoder_eval') not in stored_pt_file_orig['model_weights'].keys():
        #             eval_key = key.replace('vision_encoder', 'vision_encoder_eval')
        #             stored_pt_file['model_weights'][eval_key] = copy.deepcopy(stored_pt_file_orig['model_weights'][key])
        #             print("Copying weights from", key, "to", eval_key)
                    
        # # save the fixed model as same_name_fixed.pt
        # torch.save(stored_pt_file, "/home/ros_ws/logs/models/" + model_name + "_fixed.pt")

        self.train_params = {key: stored_pt_file[key] for key in stored_pt_file if key != "model_weights"}
        self.train_params["action_horizon"] = self.ACTION_HORIOZON
        self.train_params["num_ddim_iters"] = self.DDIM_STEPS
        if 'is_audio_based' not in self.train_params:
            self.train_params['is_audio_based'] = False
        if 'num_batches' not in self.train_params:
            self.train_params['num_batches'] = self.train_params['num_traj']
    

        if str(stored_pt_file["model_class"]).find("DiffusionTrainer") != -1:
            print("Loading Diffusion Model")
            self.model = DiffusionTrainer(
                train_params=self.train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            self.model.initialize_mpc_action()
        if str(stored_pt_file["model_class"]).find("FCTrainer") != -1:
            print("Loading FC Model")
            self.model = FCTrainer(
                train_params=self.train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            self.model.initialize_mpc_action()
            
        self.model.load_model_weights(stored_pt_file["model_weights"])

        # print model hparams (except model_weights)
        for key in self.train_params:
            print(key, ":", self.train_params[key])

        # Put model in eval mode
        self.model.eval()
        
    def eval_model_dataset(self, dataset_name):
        data_params = self.train_params
        data_params["eval_dataset_path"] = f'/home/ros_ws/dataset/data/{dataset_name}/eval'
        
        eval_dataset = DiffusionDataset(
            dataset_path=data_params['eval_dataset_path'],
            pred_horizon=data_params['pred_horizon'],
            obs_horizon=data_params['obs_horizon'],
            action_horizon=data_params['action_horizon'],
            is_state_based=self.model.is_state_based
        )

        ### Create dataloader
        self.eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=256,
            num_workers=4,
            shuffle=False,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=False
        )  
        eval_dataset.print_size("eval")
        
        # Evaluate the model by computing the MSE on test data
        loss = self.evaluate_model()
        return loss
        
    def evaluate_model(self):
        """
        Evaluates a given model on a given dataset
        Saves the model if the test loss is the best so far
        """
        # Evaluate the model by computing the MSE on test data
        total_loss = 0
        max_loss = 0
        # iterate over all the test data
        counter = 0

        for nbatch in self.eval_dataloader:
            # data normalized in dataset
            # device transfer
            nbatch = {k: v.float() for k, v in nbatch.items()}
            if(self.model.is_state_based):
                nimage = None
            else:
                nimage = nbatch['image'][:,:self.train_params['obs_horizon']].to(self.model.device)
                
            nagent_pos = nbatch['nagent_pos'][:,:self.train_params['obs_horizon']].to(self.model.device)
            naction = nbatch['actions'].to(self.model.device)
            B = nagent_pos.shape[0]

            loss, model_actions = self.model.eval_model(nimage, nagent_pos, None, naction, return_actions=True, sampler=self.DIFFUSION_SAMPLER)
            
            # print("Input to eval: nagent_pos", nagent_pos)
            # print("Input to eval: naction", naction)
            # print("Output of eval: model_actions", model_actions)
            
            # unnormalized printing
            # model_actions_unnorm = unnormalize_data(model_actions, self.model.stats['actions']).squeeze()
            # print("Input to eval unnorm: nagent_pos", unnormalize_data(nagent_pos, self.model.stats['nagent_pos']))
            # print("Input to eval unnorm: naction", unnormalize_data(naction, self.model.stats['actions']))
            # print("Output of eval unnorm: model_actions", unnormalize_data(model_actions, self.model.stats['actions']))
            # print()
            total_loss += loss*B    
            counter += B
            max_loss = max(max_loss, loss)
            print(f"Loss: {loss}, Max loss: {max_loss}, Counter: {counter}, Running Mean Loss: {total_loss/counter}")
            
        
        print(f"Max loss: {max_loss}")
        
        return total_loss/len(self.eval_dataloader.dataset)
    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: please provide model_name as node argument")
        print("Example: rosrun manipulation network_evaluator.py <model_name> <dataset_name>")
        sys.exit()

    model_name = sys.argv[1]
    dataset_name = sys.argv[2]

    N_EVALS = 2
    print(f"Testing model {model_name} for {N_EVALS} iterations")
    
    model_tester = ModelEvaluator(model_name)
    loss = model_tester.eval_model_dataset(dataset_name)
    print(f"Loss: {loss}")
    
    # exit
    exit()
