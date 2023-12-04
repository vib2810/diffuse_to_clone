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
from networks.bc_model import BCTrainer
from src.git_packages.frankapy.frankapy import FrankaArm, SensorDataMessageType
from src.git_packages.frankapy.frankapy import FrankaConstants as FC
from src.git_packages.frankapy.frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from src.git_packages.frankapy.frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from src.il_packages.manipulation.src.moveit_class import get_pose_norm, get_posestamped, EXPERT_RECORD_FREQUENCY
from src.il_packages.manipulation.src.data_class import getRigidTransform


sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage


sys.path.append("/home/ros_ws/networks") # for torch.load to work


class ModelTester:
    # Constants
    GOAL_THRESHOLD = 0.05 # TODO: need to define a goal state or condition to end the model testing
    ACTION_LIMIT_THRESHOLD = 0.6
    SMALL_NORM_CHANGE_BREAK_THRESHOLD = 1e-5
    NUM_STEPS = 500
    ACTION_HORIOZON = 1
    ACTION_SAMPLER = "ddim"
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
        if str(stored_pt_file["model_class"]).find("BCTrainer") != -1:
            print("Loading BC Model")
            self.model = BCTrainer(
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

        # Initialize frankapy
        self.fa = FrankaArm(init_node = False)
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

        if "obs_horizon" not in self.train_params:
            self.train_params["obs_horizon"] = 1
        
        # Initialize sequence buffer with zeoros of size obs_horizon
        # self.seq_buffer = [np.zeros(self.train_params["obs_dim"])]*self.train_params["obs_horizon"]
        self.seq_buffer = None
        self.prev_joints = self.fa.get_joints()
        
        # test gripper
        self.fa.close_gripper()
        self.fa.open_gripper()


    def test_model(self, N_EVALS=20):
        eval_counter = 0
        while not rospy.is_shutdown():
            if(eval_counter>=N_EVALS):
                break
            eval_counter += 1
            print(f"-------------------Evaluating Model at iteration: {eval_counter}-------------------")
            
            # Set random seed for numpy
            np.random.seed(eval_counter*100)

            # Move robot to start position
            self.fa.reset_joints()
            self.fa.open_gripper()
            
            # Run the control loop
            self.test_model_once()
                
        self.fa.reset_joints()        
        print("Done")
    
    def test_model_once(self):
        """
        Run the control loop for one episode
        """
        #TODO: vib2810 - Not a good way to get EXPERT_RECORD_FREQUENCY like this. Need to fix this.
        rate = rospy.Rate(EXPERT_RECORD_FREQUENCY)

        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        self.fa.goto_pose(self.fa.get_pose(), duration=120, dynamic=True, buffer_time=10,
                cartesian_impedances=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:3] + FC.DEFAULT_ROTATIONAL_STIFFNESSES)
        
        # Initialize variables to store data
        init_time = rospy.Time.now().to_time()
        counter = 0
        while not rospy.is_shutdown():
            # Get curr robot joints, pose and next action
            next_pose, next_gripper = self.get_next_action()
            if next_pose is None:
                break
            
            # Publish the data
            timestamp = rospy.Time.now().to_time() - init_time
            traj_gen_proto_msg = PosePositionSensorMessage(
                id=counter, timestamp=timestamp, 
                position=[next_pose.position.x, next_pose.position.y, next_pose.position.z],
                quaternion=[next_pose.orientation.w, next_pose.orientation.x, next_pose.orientation.y, next_pose.orientation.z]
            )
            fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                id=counter, timestamp=timestamp,
                translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:3],
                rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                feedback_controller_sensor_msg=sensor_proto2ros_msg(
                    fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            )
            
            self.pub.publish(ros_msg)
            
            # next_pose.position.z = max(next_pose.position.z, 0.0)
            # next_pose_rigid = getRigidTransform(next_pose)
            # self.fa.goto_pose(next_pose_rigid, duration=1)
            
            next_gripper = np.clip(next_gripper, FC.GRIPPER_WIDTH_MIN, FC.GRIPPER_WIDTH_MAX)
            self.fa.goto_gripper(next_gripper, block=False, speed = 0.2)
            
            # Break if counter exceeds num_steps
            counter += 1
            if counter > self.NUM_STEPS:
                print(f"Break due to counter {counter} > {self.NUM_STEPS}")
                break
            rate.sleep()

        # Stop the skill
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        
        self.pub.publish(ros_msg)
        self.fa.stop_skill()
        return

    def get_next_action(self):
        """
        Return next action of type (Pose, gripper_width)
        """
        # Read current pose and joints
        curr_pose_rigid = self.fa.get_pose()
        curr_pose = get_posestamped(curr_pose_rigid.translation, [curr_pose_rigid.quaternion[1], curr_pose_rigid.quaternion[2], curr_pose_rigid.quaternion[3], curr_pose_rigid.quaternion[0]]).pose
        curr_joints = self.fa.get_joints() #list of 7 elements
        curr_gripper = self.fa.get_gripper_width() #scalar

        # Append current joints to sequence buffer
        # if seq buffer is not initialized, initialize it and tile next_state
        next_state = np.concatenate((curr_joints, [curr_gripper]))
        if self.seq_buffer is None:
            self.seq_buffer = [next_state]*self.train_params["obs_horizon"]

        self.seq_buffer.pop(0)
        self.seq_buffer.append(next_state)
        
        # Append norm diff to prev_joint_norm_diffs 
        # norm_diff = np.linalg.norm(self.prev_joints - curr_joints)
        # if norm_diff < self.SMALL_NORM_CHANGE_BREAK_THRESHOLD:
        #     print(f"Break due to small norm change: {norm_diff} < {self.SMALL_NORM_CHANGE_BREAK_THRESHOLD}")
        #     return None, None
        # self.prev_joints = curr_joints
        
        # Get next action
        stacked_input = np.stack(self.seq_buffer, axis=0)
        print(f"stacked input: {stacked_input}")
        nagent_pos = torch.from_numpy(stacked_input).float().unsqueeze(0).to(self.model.device)
        # print obs_tensor.shape
        print(f"nagent_pos: {nagent_pos.shape}")

        # forward pass model to get action
        action = self.model.get_mpc_action(nimage=None, nagent_pos=nagent_pos)
        
        print("Curr State: ", curr_pose.position.x, curr_pose.position.y, curr_pose.position.z, curr_gripper)
        print(f"Model Action: {action}")
        
        next_pose = Pose()
        next_pose.position.x = action[0]
        next_pose.position.y = action[1]
        next_pose.position.z = action[2]
        next_pose.orientation.x = 1; next_pose.orientation.y = 0; next_pose.orientation.z = 0; next_pose.orientation.w = 0
        next_gripper = action[-1]
        
        # Check if action is safe
        euclid_dist_xyz = np.sqrt((next_pose.position.x - curr_pose.position.x)**2 + (next_pose.position.y - curr_pose.position.y)**2 + (next_pose.position.z - curr_pose.position.z)**2)
        print(f"euclid_dist_xyz: {euclid_dist_xyz}")
        if(euclid_dist_xyz>self.ACTION_LIMIT_THRESHOLD):
            print(f"Break due to action limits {euclid_dist_xyz} > {self.ACTION_LIMIT_THRESHOLD}")
            return None, None
        
        # Return next action
        return next_pose, next_gripper
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: please provide model_name as node argument")
        print("Example: rosrun manipulation test_network.py logs/models/<model_name>.pt")
        sys.exit()

    rospy.init_node('record_trajectories')
    model_name = sys.argv[1].split('/')[-1]
    # remove the .pt extension
    model_name = model_name[:-3]

    N_EVALS = 2
    print(f"Testing model {model_name} for {N_EVALS} iterations")
    
    model_tester = ModelTester(model_name)
    model_tester.test_model(N_EVALS=N_EVALS)
