# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3
import sys
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
import torch
import cv2
from sensor_msgs.msg import Image
from audio_common_msgs.msg import AudioData

sys.path.append("/home/ros_ws/src/il_packages/manipulation/src")
sys.path.append("/home/ros_ws/")
from networks.diffusion_model import DiffusionTrainer
from networks.fc_model import FCTrainer
from src.git_packages.frankapy.frankapy import FrankaArm, SensorDataMessageType
from src.git_packages.frankapy.frankapy import FrankaConstants as FC
from src.git_packages.frankapy.frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from src.git_packages.frankapy.frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from src.il_packages.manipulation.src.moveit_class import get_pose_norm, get_posestamped, EXPERT_RECORD_FREQUENCY
from src.il_packages.manipulation.src.data_class import getRigidTransform
from torchvision import transforms
from dataset.preprocess_audio import process_audio

sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage


sys.path.append("/home/ros_ws/networks") # for torch.load to work

class GripperActuator():
    gripped = False
    GRASP_WIDTH = 0.041 # slightly lower than the real grasp width'
    thresh_for_opening = {True: 0.0095, False: 0.001} # keys are gripped, values are threshold for opening
    # Gripper with call close_gripper when it is within 0.01m of the GRASP_WIDTH
    def __init__(self) -> None:
        pass
    
    ### Change this logic!!
    def actuate_gripper(self, fa, next_gripper, curr_gripper_width):
        
        gripper_val = next_gripper
        curr_gripper_val = curr_gripper_width
        if (gripper_val > (curr_gripper_val + self.thresh_for_opening[self.gripped])):
            print("Opening gripper, gripper_val: ", gripper_val, "curr_gripper_val: ", curr_gripper_val)
            self.gripped = False
            fa.goto_gripper(gripper_val, block=False, speed=0.2)
        elif (gripper_val - self.GRASP_WIDTH < 0.01) and (not self.gripped): # if not opening and close to GRASP_WIDTH
            print("Closing gripper, gripper_val: ", gripper_val, "curr_gripper_val: ", curr_gripper_val)
            fa.stop_gripper()
            fa.close_gripper()
            self.gripped = True
        elif not self.gripped: # General case
            fa.goto_gripper(gripper_val, block=False, speed=0.2)

class ModelTester:
    # Constants
    USE_GOTO_POSE = False # If true, will use goto_pose instead of publishing to sensor topic
    TERMINATE_POSE = get_posestamped(np.array([0.5, 0.2, 0.3]),
                                       np.array([1,0,0,0]))
    
    ACTION_LIMIT_THRESHOLD = 0.05 # Safety threshold for actions
    GOAL_THRESH = 0.05
    NUM_STEPS = 600
    
    # Overwrite DiffusionTrainer params
    ACTION_HORIZON_DIFFUSION = 8
    ACTION_SAMPLER = "ddim"
    DDIM_STEPS = 10

    def __init__(self,
            model_name
        ):
        # Initialize the model
        stored_pt_file = torch.load("/home/ros_ws/logs/models/" + model_name + ".pt", map_location=torch.device('cpu'))
        self.train_params = {key: stored_pt_file[key] for key in stored_pt_file if key != "model_weights"}

        # For models trained prior to audio based training implementation
        if 'is_audio_based' not in self.train_params:
            self.train_params['is_audio_based'] = False
        
        if "obs_horizon" not in self.train_params:
            self.train_params["obs_horizon"] = 1

        if str(stored_pt_file["model_class"]).find("DiffusionTrainer") != -1:
            self.train_params["action_horizon"] = self.ACTION_HORIZON_DIFFUSION
            self.train_params["num_ddim_iters"] = self.DDIM_STEPS
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
        
        # Config for evaluation environment
        self.is_state_based = self.train_params["is_state_based"]
        self.is_audio_based = self.train_params["is_audio_based"]
        
        self.gripper_actuator = GripperActuator()

        # print model hparams (except model_weights)
        for key in self.train_params:
            print(key, ":", self.train_params[key])

        # Put model in eval mode
        self.model.eval()

        # Initialize frankapy
        self.fa = FrankaArm(init_node = False)
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

        # Initialize image subscriber
        self.img_sub = rospy.Subscriber('camera/color/image_raw', Image, self.image_callback, queue_size=1)
        self.img_buffer = None
        self.curr_image = None
        self.image_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(), # converts to [0,1] and (C,H,W)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        if self.is_audio_based:
            # Initialize audio params
            self.audio_buffer_size = 32000
            self.audio_data = [0]*self.audio_buffer_size
            self.got_image = False
            self.got_audio = False
            rospy.Subscriber('/audio/audio', AudioData, self.audio_callback)

        # Initialize sequence buffer with zeoros of size obs_horizon
        self.seq_buffer = None
        self.prev_joints = self.fa.get_joints()
        
        # test gripper
        self.fa.close_gripper()
        self.fa.open_gripper()
        
    def audio_callback(self, data):
        """
        Maintain an audio buffer of size 32000 (last ~2 seconds of audio)
        """
        if not self.got_audio:
            self.got_audio = True
        
        audio = np.frombuffer(data.data, dtype=np.int8)
        self.audio_data = np.concatenate((self.audio_data, audio))
        
        if len(self.audio_data) > self.audio_buffer_size:
            self.audio_data = self.audio_data[-self.audio_buffer_size:]
            assert len(self.audio_data) == self.audio_buffer_size

    def image_callback(self, img_msg):
        """ Callback function from realsense camera """

        # Convert image to numpy array
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        
        img = self.image_transforms(img)
        self.curr_image = img


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
        if(not self.USE_GOTO_POSE):
            self.fa.goto_pose(self.fa.get_pose(), duration=120, dynamic=True, buffer_time=10,
                    cartesian_impedances=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:3] + FC.DEFAULT_ROTATIONAL_STIFFNESSES)
            
        # Initialize variables to store data
        init_time = rospy.Time.now().to_time()
        counter = 0
        while not rospy.is_shutdown():
            # Get curr robot joints, pose and next action
            next_pose, next_gripper, curr_gripper_width = self.get_next_action()
            if next_pose is None:
                break
            
            # Publish the data
            if(not self.USE_GOTO_POSE):
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
            
            else:
            
                next_pose.position.z = max(next_pose.position.z, 0.0)
                next_pose_rigid = getRigidTransform(next_pose)
                self.fa.goto_pose(next_pose_rigid, duration=1)
            
            next_gripper = np.clip(next_gripper, FC.GRIPPER_WIDTH_MIN, FC.GRIPPER_WIDTH_MAX)
            self.gripper_actuator.actuate_gripper(self.fa, next_gripper, curr_gripper_width)
            
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
        Return next action of type (Pose, gripper_width, curr_gripper_width)
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
        
        # Break Condition Check
        norm_diff = get_pose_norm(curr_pose, self.TERMINATE_POSE)
        if norm_diff < self.GOAL_THRESH:
            print(f"Break due to reaching terminate pose {norm_diff} < {self.GOAL_THRESH}")
            return None, None, None
        self.prev_joints = curr_joints
        
        # Get stacked joints
        stacked_input = np.stack(self.seq_buffer, axis=0)
        nagent_pos = torch.from_numpy(stacked_input).float().unsqueeze(0).to(self.model.device)

        ### Getting images
        if(self.is_state_based):
            nimage = None
        else:
            if self.curr_image == None:
                rospy.logerr("No image received. Exiting")
                sys.exit()
            if self.img_buffer is None:
                self.img_buffer = [self.curr_image]*self.train_params["obs_horizon"]
                
            self.img_buffer.pop(0)
            self.img_buffer.append(self.curr_image)

            stacked_images = np.stack(self.img_buffer, axis=0)
            nimage = torch.from_numpy(stacked_images).float().unsqueeze(0).to(self.model.device)
            
        ### Getting audio
        if self.is_audio_based:
            if not self.got_audio:
                rospy.logerr("No audio received. Exiting")
                sys.exit()
            
            # Save audio as npz file
            save_path = "/home/ros_ws/logs/test_audiofile.npy"
            np.save(save_path, self.audio_data)
            
            # preprocess audio
            processed_audio = process_audio(save_path, sample_rate=16000, num_freq_bins=100, num_time_bins=57) # shape (57, 100)
            naudio = torch.from_numpy(processed_audio).float().unsqueeze(0).unsqueeze(0).to(self.model.device) # shape (1, 1, 57, 100)
        else:
            naudio = None
        
        # forward pass model to get action
        action = self.model.get_mpc_action(nimage=nimage, nagent_pos=nagent_pos, naudio=naudio)
        
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
        print()
        
        # if euclid dist is greater than 0.3, return vector in the direction of the action
        if(euclid_dist_xyz>self.ACTION_LIMIT_THRESHOLD):
            unit_vect = np.array([next_pose.position.x - curr_pose.position.x, next_pose.position.y - curr_pose.position.y, next_pose.position.z - curr_pose.position.z])/euclid_dist_xyz
            next_pose_xyz = np.array([curr_pose.position.x, curr_pose.position.y, curr_pose.position.z]) + self.ACTION_LIMIT_THRESHOLD*unit_vect
            next_pose.position.x = next_pose_xyz[0]
            next_pose.position.y = next_pose_xyz[1]
            next_pose.position.z = next_pose_xyz[2]
            
            euclid_dist_xyz = np.sqrt((next_pose.position.x - curr_pose.position.x)**2 + (next_pose.position.y - curr_pose.position.y)**2 + (next_pose.position.z - curr_pose.position.z)**2)
            print(f"euclid_dist_xyz clipped: {euclid_dist_xyz}")
            print()

        # Return next action
        return next_pose, next_gripper, curr_gripper
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: please provide model_name as node argument")
        print("Example: rosrun manipulation test_network.py logs/models/<model_name>.pt")
        sys.exit()

    rospy.init_node('record_trajectories')
    model_name = sys.argv[1].split('/')[-1]
    # remove the .pt extension
    model_name = model_name[:-3]

    N_EVALS = 4
    print(f"Testing model {model_name} for {N_EVALS} iterations")
    
    model_tester = ModelTester(model_name)
    model_tester.test_model(N_EVALS=N_EVALS)
