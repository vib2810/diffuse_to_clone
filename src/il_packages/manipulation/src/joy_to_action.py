#!/usr/bin/env python3
import time
import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Pose, PoseStamped
import sys
sys.path.append("/home/ros_ws/src/il_packages/manipulation/src")
sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.utils import min_jerk, min_jerk_weight
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage


class DataCollection:

    def __init__(self) -> None:
        
        rospy.init_node('joy_subscribe')
        rospy.loginfo('Initializing Node')
        
        self.state_val = None
        self.fa = FrankaArm(init_node = False)
        self.fa.reset_joints()
        
        # Define maximum displacement (in meters)
        d = 0.03
        self.dx, self.dy, self.dz = d, d, d
        # Define maximum rotation (in radians)
        self.da = np.deg2rad(5)
        # hz to run the joy subsriber node
        self.fps = 10
        self.T =100
        self.curr_gripper_width = self.fa.get_gripper_width()
        
        print("Current gripper width", self.curr_gripper_width)
        
        self.current_time = time.time()
        self.last_time = time.time()

        # Define file name for data collection
        self.filename = f'task1_{int(time.time())}'

        # Define publishers
        self.action_pub = rospy.Publisher('action', PoseStamped, queue_size=1)
        
        # Initialize dynamic pose
        self.initialize_dynamic_pose()
        
        # Define subscribers
        self.joy_sub = rospy.Subscriber('/joy', Joy,self.joy_callback)

    def joy_callback(self, joy_msg):
        self.current_time = time.time()
        if(self.current_time - self.last_time > 1/self.fps):
            print("Reached the callback")
            curr_pose = self.fa.get_pose()
            self.curr_gripper_width = self.fa.get_gripper_width()
            print("Got current pose")
            # print("Curr pose Translation", curr_pose.translation)
            
            # assert joy_msg.header.stamp == state_msg.header.stamp
            delta_normalized = self.transform_to_base(joy_msg)
            delta_unnormalized = self.get_delta(delta_normalized)
            print(" Deltas", delta_unnormalized)
            next_pose, gripper_width = self.get_next_pose(curr_pose,delta_unnormalized)
            # gripper_width  = -gripper_width
            print(" Current gripper width", self.fa.get_gripper_width())
            print(" New gripper width", gripper_width)
            # self.collect_data(state_msg, action)
            # self.action_pub.publish(action_pose)
            
            # rospy.sleep(1)
            # self.fa.goto_pose(next_pose, duration=0.5, dynamic=False)
            
            self.go_to_pose_dynamic(next_pose,gripper_width)
            print("Got next pose")
            # print(" Next pose Translation", next_pose.translation)
            print("***************"*3)
            # print("Rotation", next_pose.rotation)
           
            self.current_time = time.time()
            self.last_time = self.current_time
        
        else:
            pass
        
    def initialize_dynamic_pose(self, ):
        
        self.idx=0
        rospy.loginfo('Initializing dynamic pose')
        self.fa.goto_pose(self.fa.get_pose(), duration=self.T, dynamic=True, buffer_time=10,
            cartesian_impedances=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:3] + FC.DEFAULT_ROTATIONAL_STIFFNESSES)
        
        self.init_time = rospy.Time.now().to_time()
        
        rospy.loginfo('Initializing Sensor Publisher')
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        
    
    def go_to_pose_dynamic(self, pose, gripper_width):
        timestamp = rospy.Time.now().to_time() - self.init_time
        self.idx+=1
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=self.idx, timestamp=timestamp, 
            position=pose.translation, quaternion=pose.quaternion
        )
        fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            id=self.idx, timestamp=timestamp,
            translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:3],
            rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            feedback_controller_sensor_msg=sensor_proto2ros_msg(
                fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            )

        self.fa.goto_gripper(gripper_width, block=False)

        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        self.pub.publish(ros_msg)

        # Stop the skill
        # Alternatively can call fa.stop_skill()
        if(time.time() - self.init_time > self.T):
            term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - self.init_time, should_terminate=True)
            ros_msg = make_sensor_group_msg(
                termination_handler_sensor_msg=sensor_proto2ros_msg(
                    term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
                )
            self.pub.publish(ros_msg)

        
    def get_next_pose(self, curr_pose, delta_unnormalized):
        """ Adds deltas to the current state and finds future pose"""
        
        next_pose = curr_pose.copy()
        # print("Rotation matrix", curr_pose.rotation)
        rot_matrix_state = np.array(curr_pose.rotation).reshape(3,3)
        # Find rot matrix for rotation about z axis
        rot_matrix_delta = RigidTransform.z_axis_rotation(delta_unnormalized[3])
        rot_matrix_next_state = np.matmul(rot_matrix_state, rot_matrix_delta)
        
        next_pose.translation += delta_unnormalized[:3]
        next_pose.rotation = rot_matrix_next_state

        gripper_width = self.curr_gripper_width + delta_unnormalized[4]
        
        return next_pose, gripper_width
        
    def transform_to_base(self, msg): 
        axes_msg = msg.axes
        buttons_msg = msg.buttons                      
        x = -axes_msg[1]
        y = -axes_msg[0]
        z = axes_msg[4]
        theta_positive = axes_msg[5]
        theta_negative = axes_msg[2]
        gripper_open = buttons_msg[4]
        gripper_close = buttons_msg[5]
        return (x, y, z, theta_positive, theta_negative, gripper_open, gripper_close)
    
    
    def get_delta(self, delta_normalized):
        delta_x = delta_normalized[0]*self.dx
        delta_y = delta_normalized[1]*self.dy
        delta_z = delta_normalized[2]*self.dz
        if delta_normalized[3] < 0.7:                          # TODO should i make this class member?
            delta_a = self.da
        elif delta_normalized[4] < 0.7:
            delta_a = -self.da
        else:
            delta_a = 0    
            
        if delta_normalized[5] == 1:
            delta_width = -0.01
        elif delta_normalized[6] == 1:
            delta_width = 0.01
        else:
            delta_width = 0
            
        return (delta_x, delta_y, delta_z, delta_a, delta_width)
    

if __name__ == '__main__': 
    try:
        print("hello")
        data_collector = DataCollection()
        rate = rospy.Rate(10)
        rospy.spin()
    except rospy.ROSInitException:
        pass