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
from src.git_packages.frankapy.frankapy import FrankaArm, SensorDataMessageType
from src.git_packages.frankapy.frankapy import FrankaConstants as FC
from src.git_packages.frankapy.frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from src.git_packages.frankapy.frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from src.il_packages.manipulation.src.moveit_class import get_pose_norm, get_posestamped, EXPERT_RECORD_FREQUENCY
from src.il_packages.manipulation.src.data_class import getRigidTransform


sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage

class ResetArm:

    def __init__(self,):
    
        # Initialize frankapy
        self.fa = FrankaArm(init_node = False)
        
        # test gripper
        self.fa.reset_joints()
    
if __name__ == "__main__":
    rospy.init_node('reset_arm')
    model_tester = ResetArm()
