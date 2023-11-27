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
from il_msgs.msg import RecordJointsAction, RecordJointsResult, RecordJointsGoal
import torch

sys.path.append("/home/ros_ws/src/il_packages/manipulation/src")
from moveit_class import moveit_planner
sys.path.append("/home/ros_ws/")
from networks.bc_model import BCTrainer
from networks.residuil_model import ResiduILTrainer
from networks.lstm_model import LSTMTrainer
from networks.model_utils import read_pkl_trajectory_data
import yaml
import os
sys.path.append("/home/ros_ws/networks") # for torch.load to work

from src.git_packages.frankapy.frankapy import FrankaArm, SensorDataMessageType
from src.git_packages.frankapy.frankapy import FrankaConstants as FC
from src.git_packages.frankapy.frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from src.git_packages.frankapy.frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from src.il_packages.manipulation.src.moveit_class import get_pose_norm, get_posestamped, EXPERT_RECORD_FREQUENCY
from geometry_msgs.msg import Pose

class ModelTester:
    # Constants
    GOAL_THRESHOLD = 0.05
    ACTION_LIMIT_THRESHOLD = 0.2
    SMALL_NORM_CHANGE_BREAK_THRESHOLD = 1e-5
    NUM_STEPS = 250

    def __init__(self,
            model_name,
            traj_num,
        ):
        # Initialize the model
        train_params = torch.load("/home/ros_ws/bags/models/" + model_name + ".pt", map_location=torch.device('cpu'))
        if str(train_params["model_class"]).find("BCTrainer") != -1:
            print("Loading BC Model")
            self.model = BCTrainer(
                train_params=train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
        if str(train_params["model_class"]).find("LSTMTrainer") != -1:
            print("Loading LSTM Model")
            self.model = LSTMTrainer(
                train_params=train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
        if str(train_params["model_class"]).find("ResiduILTrainer") != -1:
            print("Loading ResiduIL Model")
            self.model = ResiduILTrainer(
                train_params=train_params,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
        self.model.load_model_weights(train_params["model_weights"])
        self.train_params = train_params

        # print model hparams (except model_weights)
        for key in self.train_params:
            if key != "model_weights":
                print(key, ":", self.train_params[key])

        # load point and joint ranges from yaml file
        trajectory_path = '/home/ros_ws/bags/recorded_trajectories/trajectory_' + str(traj_num) + '.yaml'
        print("Loading limits from: ", trajectory_path)
        with open(trajectory_path, 'r') as f:
            self.limits = yaml.load(f, Loader=yaml.FullLoader)
            self.noise_std = self.limits["noise_std"]
            self.num_steps_cnfnd = self.limits["num_steps_cnfnd"]
            self.noise_type = self.limits["noise_type"]

        # Put model in eval mode
        self.model.eval()

        # Make folder for saving trajectories
        self.trajectory_folder = '/home/ros_ws/bags/test_trajectories/'
        if not(os.path.exists(self.trajectory_folder)):
            os.makedirs(self.trajectory_folder)
        self.pickle_file_name = f'{traj_num}__{model_name}.pkl'

        # move robot to a valid start position
        self.franka_moveit = moveit_planner()

        # Load goal joints as average of all goal joints in the trajectory
        _,_,_,_,_,_, self.target_pose = read_pkl_trajectory_data(traj_num, print_data=False)
        print("Goal Joints loaded form traj_num ", traj_num, "with target pose:\n", self.target_pose)
        
        # Reset joints
        self.franka_moveit.fa.reset_joints()

        if "seq_len" not in self.train_params:
            self.train_params["seq_len"] = 1
        # Initialize sequence buffer with zeoros of size seq_len
        self.seq_buffer = [np.zeros(self.train_params["ac_dim"])]*self.train_params["seq_len"]

        # Initialize prev norm diff for small norm change break
        self.prev_norm_diff = [None]*5
    


    def move_robot_to_a_start_position(self):
        """
        Move robot to start position
        """
        start_pose = self.franka_moveit.sample_pose(self.limits, self.target_pose)

        # move robot to start pose
        self.franka_moveit.goto_pose_moveit(start_pose, speed = 3)

    def test_model(self, N_EVALS=20):
        eval_counter = 0
        all_robot_trajectories = {"trajectories": [], "timestamps": [], "tool_poses": [], "target_poses": [], "terminal_reasons": []}
        while not rospy.is_shutdown():
            if(eval_counter>=N_EVALS):
                break
            eval_counter += 1
            print(f"-------------------Evaluating Model at iteration: {eval_counter}-------------------")
            
            # Set random seed for numpy
            np.random.seed(eval_counter*100)

            # Move robot to start position
            self.move_robot_to_a_start_position()
            
            # Run the control loop
            test_robot_joints, test_robot_poses, test_robot_timestamps, terminal_reason = self.test_model_once()
            all_robot_trajectories["trajectories"].append(test_robot_joints)
            all_robot_trajectories["tool_poses"].append(test_robot_poses)
            all_robot_trajectories["timestamps"].append(test_robot_timestamps)
            all_robot_trajectories["target_poses"].append(self.target_pose)
            all_robot_trajectories["terminal_reasons"].append(terminal_reason)

            # Save test trajectories to pickle file 
            with open(self.trajectory_folder + "/" + self.pickle_file_name, 'wb') as f:
                pickle.dump(all_robot_trajectories, f)
                
        print("Done")
        self.franka_moveit.fa.reset_joints()
    
    def test_model_once(self):
        """
        Returns test_robot_joints, test_robot_poses, test_robot_timestamps, terminal_reason
        """
        rate = rospy.Rate(EXPERT_RECORD_FREQUENCY)

        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        curr_pose, curr_joints, next_action, terminal_reason = self.get_next_joints()
        if next_action is None:
            return [], [], [], None
        self.franka_moveit.fa.goto_joints(next_action, duration=5, dynamic=True, buffer_time=30, ignore_virtual_walls=True)
        
        # Initialize variables to store data
        init_time = rospy.Time.now().to_time()
        test_robot_joints = []
        test_robot_poses = []
        test_robot_timestamps = []
        
        counter = 0
        terminal_reason = None
        u_past = np.zeros((self.num_steps_cnfnd,1))
        while not rospy.is_shutdown():
            # Get curr robot joints, pose and next action
            curr_pose_rigid, curr_joints, next_action, terminal_reason = self.get_next_joints()
            if next_action is None:
                break
            if self.noise_type == "uniform":
                u = np.random.uniform(-self.noise_std, self.noise_std, 1)
            elif self.noise_type == "gaussian":
                u = np.random.normal(0, self.noise_std, 1)
            else:
                u = 0
            noise = (1 * u + 1 * np.sum(u_past, axis=0))/1
            next_action += noise
            u_past = np.roll(u_past, 1, axis=0)
            u_past[0,:] = u

            # Log the data
            test_robot_joints.append(curr_joints)
            test_robot_poses.append(curr_pose_rigid)
            test_robot_timestamps.append(rospy.Time.now().to_nsec())
            
            # Publish the data
            traj_gen_proto_msg = JointPositionSensorMessage(
                id=counter, timestamp=rospy.Time.now().to_time() - init_time, 
                joints=next_action
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
            )
            self.franka_moveit.pub.publish(ros_msg)

            # Break if counter exceeds num_steps
            counter += 1
            if counter > self.NUM_STEPS:
                print(f"Break due to counter {counter} > {self.NUM_STEPS}")
                terminal_reason = f"Counter Exceeded | {counter} > {self.NUM_STEPS}"
                break
            rate.sleep()

        # Stop the skill
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        
        self.franka_moveit.pub.publish(ros_msg)
        self.franka_moveit.fa.stop_skill()
        return test_robot_joints, test_robot_poses, test_robot_timestamps, terminal_reason

    def terminate_run(self, curr_pose):
        norm_diff = get_pose_norm(curr_pose, self.target_pose)
        print(f"Norm diff: {norm_diff}")
        # If norm diff is less than threshold, break
        if(norm_diff<self.GOAL_THRESHOLD):
            print("Break due to goal joints")
            return True, f"Goal Reached | Norm diff {norm_diff} < {self.GOAL_THRESHOLD}"

        # If change in norm_diff is small, break
        if self.prev_norm_diff[0] is not None:
            norm_change = abs(norm_diff-self.prev_norm_diff[0])
            print(f"Norm change: {norm_change}")
            if norm_change<self.SMALL_NORM_CHANGE_BREAK_THRESHOLD:
                print(f"Break due to small norm change: {norm_change} < {self.SMALL_NORM_CHANGE_BREAK_THRESHOLD}")
                return True, f"Small Norm Change | {norm_change} < {self.SMALL_NORM_CHANGE_BREAK_THRESHOLD}"      
        print()
        self.prev_norm_diff.pop(0)
        self.prev_norm_diff.append(norm_diff)
        return False, None

    def get_next_joints(self):
        """
        Return curr_pose_rigid, curr_joints, next_action, terminal_reason
        """
        # Read current pose and joints
        curr_pose_rigid = self.franka_moveit.fa.get_pose()
        curr_pose = get_posestamped(curr_pose_rigid.translation, [curr_pose_rigid.quaternion[1], curr_pose_rigid.quaternion[2], curr_pose_rigid.quaternion[3], curr_pose_rigid.quaternion[0]]).pose
        curr_joints = self.franka_moveit.fa.get_joints()

        # Append current joints to sequence buffer
        self.seq_buffer.pop(0)
        self.seq_buffer.append(curr_joints)

        # Check if run should be terminated
        terminate, terminal_reason = self.terminate_run(curr_pose)
        if terminate:
            return curr_pose_rigid, curr_joints, None, terminal_reason
        
        # Get next action
        if len(self.seq_buffer) == 1:
            joints_tensor = torch.tensor(self.seq_buffer[0]).float().to(self.model.device)
        else:
            joints_tensor = torch.tensor(np.stack(self.seq_buffer)).float().to(self.model.device)

        # forward pass model to get action
        action = self.model.get_action(joints_tensor, requires_grad=False).squeeze(0).cpu().detach().numpy()

        # Check if action is safe
        norm_diff_joints = np.linalg.norm(action-curr_joints)
        if(norm_diff_joints>self.ACTION_LIMIT_THRESHOLD):
            print(f"Break due to action limits {norm_diff_joints} > {self.ACTION_LIMIT_THRESHOLD}")
            return curr_pose_rigid, curr_joints, None, f"Action Limits | {norm_diff_joints} > {self.ACTION_LIMIT_THRESHOLD}"
        
        # Return next action
        return curr_pose_rigid, curr_joints, action, None
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: please provide model_name as node argument")
        print("Example: rosrun manipulation test_network.py bags/models/bc_model_3_03-10-2023_05-56-06.pt")
        sys.exit()

    rospy.init_node('record_trajectories')
    # sys.argv[1] will be of format bags/models/bc_model_3_03-10-2023_05-56-06.pt
    traj_num = sys.argv[1].split("_")[2].split(".")[0]
    model_name = sys.argv[1].split("/")[2].split(".")[0]

    N_EVALS = 20
    print(f"Testing model {model_name} on trajectory {traj_num} for {N_EVALS} times")
    
    model_tester = ModelTester(model_name, traj_num)
    model_tester.test_model(N_EVALS=N_EVALS)
