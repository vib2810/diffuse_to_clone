#!/usr/bin/env python3

import sys
sys.path.append("/home/ros_ws/")
sys.path.append("/home/ros_ws/src/il_packages/")
sys.path.append("/home/ros_ws/src/il_packages/manipulation/src")
sys.path.append("/home/ros_ws/src/git_packages/frankapy")

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import numpy as np
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive, Mesh
import scipy.spatial.transform as spt
import copy
from geometry_msgs.msg import PoseStamped, Pose
import pickle
import os
import actionlib
import time
import json
# from il_msgs.msg import RecordJointsAction, RecordJointsResult, RecordJointsGoal
from src.git_packages.frankapy.frankapy import FrankaArm, SensorDataMessageType
from src.git_packages.frankapy.frankapy import FrankaConstants as FC
from src.git_packages.frankapy.frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from src.git_packages.frankapy.frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from moveit_class import MoveitPlanner

#### What should it do?
#### 1) Keep running run guide mode and collect multiple home poses for n bags.
#### 2) Save the poses (both bag and circle) in a json file in the configs folder.
#### 3) Run pick primitive for any pose in the json file.
#### 4) Randomly sample poses to act as a placing position.
#### 5) Place the objects in the new starting position. Using moveit plans.
#### 6) Make plans for starting position to placing position.
#### 7) Execute the plan once, record and replan routine.
#### 8) Save state/action pairs in a pickle file.
#### 9) Repeat 1-8 for n bags.


class Manipulation():
    #### What class variables to define
    ## Make moveit class objects.
    
    def __init__(self,):
        
        rospy.init_node('manipulation')
        
        self.franka_moveit = MoveitPlanner()
        
        ### Reset robot
        print("Resetting robot")
        self.franka_moveit.fa.reset_joints()
        
        ### Run guide mode params
        self.rgm_dict = {}
        self.rgm_dict["time"] = 100
        self.rgm_dict["num_bags"] = 2
        
        ### Run config
        self.class_labels = ["home","goal"]
        self.data_dict = initialize_data_dict(self.class_labels,self.rgm_dict["num_bags"])
        
        ### Paths
        self.config_dir = '/home/ros_ws/src/il_packages/manipulation/config/'
        all_dirs = [self.config_dir]
        create_dir(all_dirs)
        
    def pick_and_place(self,start_pose,goal_pose):
        """ Runs the pick and place primitive for a given start and goal pose. """
        
       ### Go to the home pose 
       ### Pick the bag by continuously reducing the gripper width
       ### Go to the sampled pose 
       ### Drop the bag and record the sampled pose and gripper width
       ### Use dynamic trajectory
       
    def record_single_trajectory(self,):
        """ Records the trajectory for a given start and goal pose. """
        
        ### For each bag i 
        ### Go to the recorded start pose 
        ### Pick the bag by continuously reducing the gripper width
        ### Go to the goal pose 
        ### Drop the bag
        ### Continuously record state/action pairs
        ### Compute plan using moveit and implement only one step of the plan
        
    def record_multiple_trajectories(self,):
        """ Records multiple trajectories for a given start and goal pose. """
        
        ### For each bag i 
        ### Run pick and place primitive 
        ### record single trajectory
        ### Move to default location
        ### Repeat for all bags
        
    def store_config_in_run_guide_mode(self,):
        """ Runs the run guide mode and collects various user-defined home/start/goal positions 
        and stores them in a json file in the configs folder."""
        
        idx=0
        last_time = time.time()
        current_time = time.time()
        tf_mats = []

        self.franka_moveit.fa.run_guide_mode(self.rgm_dict["time"],block=False)  
        rospy.loginfo("Running guide mode for {} seconds".format(self.rgm_dict["time"]))
        
        # sleep for 1 second to ensure robot is in guide mode
        rospy.sleep(1)
         
        # read user input from terminal. Exit when user presses q
        while current_time - last_time < self.rgm_dict["time"] and idx < 2*self.rgm_dict["num_bags"]:
            print("Press 's' to store pose, 'p' to print current pose")
            key = input("Press q to quit: ")
            if key == 'q':
                break
            elif key =='s':
                curr_pose = self.franka_moveit.fa.get_pose()
                tvec = np.array(curr_pose.translation)
                rot_mat = np.array(curr_pose.rotation)
                tf_mat = np.eye(4)
                tf_mat[:3,:3] = rot_mat
                tf_mat[:3,3] = tvec
                tf_mats.append(tf_mat)
                idx += 1
                print(" Stored pose: ", tf_mat)
            elif key == 'p':
                print("Current pose: ", self.franka_moveit.fa.get_pose())
             
        # stop skill
        self.franka_moveit.fa.stop_skill()
        
        if(not idx < 2*self.rgm_dict["num_bags"]):
            print("Reached max number of poses allowed for num bags = {}. So exiting!!".format(self.rgm_dict["num_bags"]))
        
        # save in dictionary
        labels = self.class_labels
        
        for label in labels:
            for idx in range(self.rgm_dict["num_bags"]):
                self.data_dict[label]["bag"+str(idx)]["pose"] = tf_mats[idx].tolist()
            
        # dump dictionary to json file
        print(" data dict = ", self.data_dict)
        dump_dict_to_json(self.data_dict,self.config_dir)
        
### Helper functions

def parse_json_to_dict(config_dir):
    """ Parses the json file in the config folder to a dictionary."""
    
    config_path = config_dir + "config.json"
    assert os.path.exists(config_path), "Config file does not exist"
    with open(config_path, 'r') as f:
        data_dict = json.load(f)
        
    return data_dict
        
def dump_dict_to_json(data_dict,config_dir):
    """ Dumps the data dictionary to a json file in the config folder."""
    
    config_path = config_dir + "config.json"
    assert os.path.exists(config_dir), "Config directory does not exist"
    with open(config_path, 'w') as f:
        json.dump(data_dict, f, indent=4)

def create_dir(all_dirs):
    for dir in all_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print("Created directory: ", dir)
        else:
            print("Directory already exists: ", dir)
                
def initialize_data_dict(labels: list,num_bags: int=2):
    """ Initializes the data dictionary with the labels provided."""
        
    data_dict = {}
    for label in labels:
        data_dict[label] = {}
        for idx in range(num_bags):
            data_dict[label]["bag"+str(idx)] = {}
            data_dict[label]["bag"+str(idx)]["pose"] = None
                
    return data_dict
        

if __name__ == "__main__":
    manipulation = Manipulation()
    manipulation.store_config_in_run_guide_mode()
    print("Done")
        
        
        
        
    