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
from moveit_class import moveit_planner

sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

if __name__ == "__main__":
    # get all trajectory numbers from folder /home/ros_ws/bags by reading file names traejctory_{num}.yaml
    # use os to get all file names in /home/ros_ws/bags
    all_files = os.listdir('/home/ros_ws/bags')
    all_trajectory_numbers = []
    for file_name in all_files:
        if file_name.startswith('trajectory_'):
            trajectory_number = int(file_name.split('.')[0].split('_')[1])
            all_trajectory_numbers.append(trajectory_number)
    if len(all_trajectory_numbers) == 0:
        trajectory_num = 1
    else:
        trajectory_num = max(all_trajectory_numbers) + 1
    rospy.init_node('record_trajectories')
    
    print("Recording Trajectories in trajectory_num: ", trajectory_num)

    franka_moveit = moveit_planner()

    # Reset Joints
    franka_moveit.reset_joints()

    # Collect trajectories
    trajectory_limits_dict = {}
    trajectory_limits_dict["xrange"] = [0.45, .55]
    trajectory_limits_dict["yrange"] = [-.05, .05]
    trajectory_limits_dict["zrange"] = [.15, .25]
    trajectory_limits_dict["xang_range"] = [-185, -175]
    trajectory_limits_dict["yang_range"] = [-5, 5]
    trajectory_limits_dict["zang_range"] = [-5, 5]
    trajectory_limits_dict["min_norm_diff"] = 0.3
    trajectory_limits_dict["n_trajectories"] = 50

    # Save trajectory limits dict
    trajectory_path = '/home/ros_ws/bags/recorded_trajectories/trajectory_' + str(trajectory_num) + '.yaml'
    with open(trajectory_path, 'w') as f:
        data_dict = trajectory_limits_dict
        data_dict["Collection_Mode"] = "Manual"
        yaml.dump(data_dict, f)
        
    print("Collecting Trajectories")
    
    franka_moveit.collect_trajectories_manual(trajectory_limits_dict, trajectory_num, speed = 3)

    print("Trajectories Collected")

