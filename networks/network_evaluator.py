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


class ModelEvaluator:
    ACTION_HORIOZON = 8
    DDIM_STEPS = 10
    def __init__(self,
            model_name
        ):
        # Initialize the model
        stored_pt_file = torch.load("/home/ros_ws/logs/models/" + model_name + ".pt", map_location=torch.device('cpu'))
        self.train_params = {key: stored_pt_file[key] for key in stored_pt_file if key != "model_weights"}
        self.train_params["action_horizon"] = self.ACTION_HORIOZON
        self.train_params["num_ddim_iters"] = self.DDIM_STEPS
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
        )

        ### Create dataloader
        self.eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.train_params['batch_size'],
            num_workers=self.train_params['num_workers'],
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

            loss, model_actions = self.model.eval_model(nimage, nagent_pos, naction, return_actions=True)
            # if the first nagent_pos stacked input is all zeros, then the output is also all zeros
            for bidx in range (B):
                if torch.sum(nagent_pos[bidx, 0]) == 0:
                    counter += 1
            print("Num data points with first seq all zeros: ", counter)
            # print("Input to eval: nagent_pos", nagent_pos)
            # print("Input to eval: naction", naction)
            # print("Output of eval: model_actions", model_actions)
            
            # unnormalized printing
            # print("Input to eval unnorm: nagent_pos", unnormalize_data(nagent_pos, self.model.stats['nagent_pos']))
            # print("Input to eval unnorm: naction", unnormalize_data(naction, self.model.stats['actions']))
            # print("Output of eval unnorm: model_actions", unnormalize_data(model_actions, self.model.stats['actions']))
            total_loss += loss*B    
            max_loss = max(max_loss, loss)
        
        print(f"Max loss: {max_loss}")
        
        return total_loss/len(self.eval_dataloader.dataset)
    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: please provide model_name as node argument")
        print("Example: rosrun manipulation test_network.py <model_name> <dataset_name>")
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
