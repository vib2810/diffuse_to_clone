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
        print("Please provide experiment_name, num_trajs_to_collect, eval_mode[optional] as command line arguments")
        sys.exit(0)
        
    experiment_name = sys.argv[1]
    num_trajs_to_collect = int(sys.argv[2])
    eval_mode = False
    if(len(sys.argv) > 3):
        eval_mode = bool(sys.argv[3])
    
            
    rospy.init_node('record_experiment')
    
    print("Recording Experiment: ", experiment_name , " with ", num_trajs_to_collect, " trajectories")

    franka_moveit = MoveitPlanner()

    # Reset Joints
    franka_moveit.reset_joints()

    # Collect trajectories
    expt_data_dict = {}
    expt_data_dict["experiment_name"] = "toy_expt_"+ experiment_name
    expt_data_dict["n_trajectories"] = num_trajs_to_collect
    expt_data_dict["eval_mode"] = eval_mode
    expt_data_dict["pick_pose"] = get_posestamped(np.array([0.5, 0, 0.3]), np.array([1,0,0,0]))
    expt_data_dict["place_pose"] = get_posestamped(np.array([0.5, 0.2, 0.3]), np.array([1,0,0,0]))

        
    print("Collecting Experiment with Config:\n ", expt_data_dict)
    
    franka_moveit.collect_toy_trajectories(expt_data_dict)

    print("Trajectories Collected")
    franka_moveit.fa.reset_joints()

