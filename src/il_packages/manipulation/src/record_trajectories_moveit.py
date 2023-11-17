# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3
import sys
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
from geometry_msgs.msg import PoseStamped
import pickle
import actionlib
import yaml
from il_msgs.msg import RecordJointsAction, RecordJointsResult, RecordJointsGoal
import os

sys.path.append("/home/ros_ws/src/il_packages/manipulation/src")
from moveit_class import MoveitPlanner, get_posestamped

sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

global action_client
if __name__ == "__main__":
    # get noisy or not from command line
    if len(sys.argv) < 3:
        print("Please provide [n_trajs, noise std, noise type, n_steps_cnfnd] or [n_trajs, eval]") 
        sys.exit(0)
    
    n_trajs = int(sys.argv[1])
    if (sys.argv[2] == "eval"):
        noise_std = 0
        noise_type = "gaussian" #doesn't matter
        n_steps_cnfnd = 1 #doesn't matter
        eval_mode = True
        trajectory_num = "eval"
    else:
        noise_std = float(sys.argv[2])
        noise_type = sys.argv[3]
        if noise_type != "gaussian" and noise_type != "uniform":
            print("Noise type must be gaussian or uniform")
            sys.exit(0)
        n_steps_cnfnd = int(sys.argv[4])
        eval_mode = False

        # use os to get all file names in /home/ros_ws/bags
        all_files = os.listdir('/home/ros_ws/bags/recorded_trajectories/')
        all_trajectory_numbers = []
        for file_name in all_files:
            if file_name.startswith('trajectory_') and file_name.find('eval') == -1:
                trajectory_number = int(file_name.split('.')[0].split('_')[1])
                all_trajectory_numbers.append(trajectory_number)
        if len(all_trajectory_numbers) == 0:
            trajectory_num = 1
        else:
            trajectory_num = max(all_trajectory_numbers) + 1
            
    rospy.init_node('record_trajectories')
    
    print("Recording Trajectories in trajectory_num: ", trajectory_num)

    franka_moveit = MoveitPlanner()

    # Reset Joints
    franka_moveit.reset_joints()

    # Collect trajectories
    trajectory_limits_dict = {}
    trajectory_limits_dict["xrange"] = [0.45, .55]
    trajectory_limits_dict["yrange"] = [-.1, .1]
    trajectory_limits_dict["zrange"] = [.15, .45]
    trajectory_limits_dict["xang_range"] = [-185, -175]
    trajectory_limits_dict["yang_range"] = [-5, 5]
    trajectory_limits_dict["zang_range"] = [-5, 5]
    trajectory_limits_dict["min_norm_diff"] = 0.1
    trajectory_limits_dict["n_trajectories"] = n_trajs
    trajectory_limits_dict["num_steps_cnfnd"] = n_steps_cnfnd
    trajectory_limits_dict["eval_mode"] = eval_mode
    trajectory_limits_dict["noise_std"] = noise_std
    trajectory_limits_dict["noise_type"] = noise_type

    # # store above 7 variables in /home/ros_ws/bags/recorded_trajectories/trajectory_{num}.yaml
    trajectory_path = '/home/ros_ws/bags/recorded_trajectories/trajectory_' + str(trajectory_num) + '.yaml'
    with open(trajectory_path, 'w') as f:
        data_dict = copy.deepcopy(trajectory_limits_dict)
        data_dict["Collection_Mode"] = "Noisy"
        yaml.dump(data_dict, f)
    
    # Add this after storing in yaml file    
    trajectory_limits_dict["target_pose"] = get_posestamped(np.array([0.5, 0, 0.3]), np.array([1,0,0,0]))
        
    print("Collecting Trajectories with Config:\n ", trajectory_limits_dict)
    
    franka_moveit.collect_trajectories_noisy(trajectory_limits_dict, trajectory_num)

    print("Trajectories Collected")
    franka_moveit.fa.reset_joints()

