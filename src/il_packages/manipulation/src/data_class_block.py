#!/usr/bin/env python3
import os
import sys
import copy
import rospy
import pickle
import numpy as np
from sensor_msgs.msg import Image
from audio_common_msgs.msg import AudioData
from geometry_msgs.msg import Pose, PoseStamped
from autolab_core import RigidTransform
from cv_bridge import CvBridge, CvBridgeError
import scipy.spatial.transform as spt
from scipy.spatial.transform import Rotation as R

sys.path.append("/home/ros_ws/")
from src.git_packages.frankapy.frankapy import FrankaArm, SensorDataMessageType
from src.git_packages.frankapy.frankapy import FrankaConstants as FC
from src.git_packages.frankapy.frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from src.git_packages.frankapy.frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

sys.path.append("/home/ros_ws/src/git_packages/frankapy")
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage

EXPERT_RECORD_FREQUENCY = 10
RESET_SPEED = 3
Z_PICK = 0.018
class Data():
    def __init__(self) -> None:
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.fa = FrankaArm(init_node = False)

        self.audio_buffer_size = 32000
        self.audio_data = [0]*self.audio_buffer_size
        self.bridge = CvBridge()
        self.got_image = False

        rospy.Subscriber('/audio/audio', AudioData, self.audio_callback)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

    def audio_callback(self, data):
        self.audio_data.extend(data.data)
        if len(self.audio_data) > self.audio_buffer_size:
            self.audio_data = self.audio_data[-self.audio_buffer_size:]

    def image_callback(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8") #(480, 640, 3)
            self.got_image = True
        except CvBridgeError as e:
            self.got_image = False
            print(e)

    def reset_joints(self):
        self.fa.reset_joints()

    def goto_joint(self, joint_goal):
        self.fa.goto_joints(joint_goal, duration=5, dynamic=True, buffer_time=10)

    def get_next_pose_interp(self, target_pose: Pose, curr_pose: Pose, speed=0.02):
        """
        Returns the next tool pose to goto from the current pose of the robot given a target pose 
        """
        current_position = np.array([curr_pose.position.x, curr_pose.position.y, curr_pose.position.z])
        r = R.from_quat([curr_pose.orientation.x, curr_pose.orientation.y, curr_pose.orientation.z, curr_pose.orientation.w])
        current_orientation = r.as_euler('zyx', degrees=True)

        target_position = np.array([target_pose.position.x, target_pose.position.y, target_pose.position.z])
        r = R.from_quat([target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w])
        target_orientation = r.as_euler('zyx', degrees=True)
        
        unit_diff_in_position = (target_position - current_position) / np.linalg.norm(target_position - current_position)
        # if np.linalg.norm(target_position - current_position) < self.POSE_DIFF_TOL:
        #     next_position = target_position
        # else:
        next_position = current_position + unit_diff_in_position*speed

        diff_in_yaw_deg = target_orientation[0] - current_orientation[0]
        if abs(diff_in_yaw_deg) < 0.5:
            next_yaw_in_deg = target_orientation[0]
        else:
            next_yaw_in_deg = current_orientation[0] + np.sign(diff_in_yaw_deg)*0.5

        r = R.from_euler('zyx', [next_yaw_in_deg, 0, 180], degrees=True)
        next_orientation = r.as_quat()

        next_pose = Pose()
        next_pose.position.x = next_position[0]
        next_pose.position.y = next_position[1]
        next_pose.position.z = next_position[2]
        # next_pose.orientation.x = next_orientation[0]
        # next_pose.orientation.y = next_orientation[1]
        # next_pose.orientation.z = next_orientation[2]
        # next_pose.orientation.w = next_orientation[3]
        next_pose.orientation.x = 1
        next_pose.orientation.y = 0
        next_pose.orientation.z = 0
        next_pose.orientation.w = 0
        return next_pose
    
    def get_current_pose_norm(self, target_pose: PoseStamped, fa_pose: Pose):
        """
        Returns the norm of the difference between the fa_rigid and target_pose
        """
        norm_diff = get_pose_norm(target_pose.pose, fa_pose)
        return norm_diff

    def get_target_pose(self, target_center_pose: PoseStamped):
        """
        Returns a random gaussian-sampled pose around the target_center_pose
        """
        target_pose = copy.deepcopy(target_center_pose)
        target_pose.pose.position.x += np.random.normal(0, 0.02)
        target_pose.pose.position.y += np.random.normal(0, 0.02)
        return target_pose
    
    current_toy_state = 0  
    FINAL_GRIPPER_WIDTH = 0.045 # slightly highter than grasp width
    GRASP_WIDTH = 0.040 # slightly lower than grasp width
    NORM_DIFF_TOL = 0.03
    Z_ABS_TOL = 0.003
    
    def get_next_joint_planner_toy_joints(self, curr_pose: Pose, curr_gripper_width = float):
        """
        Takes as input the current robot joints, pose and gripper width
        Returns the next joint position for the toy task
        Returns: next_action(7x1), gripper_width

        STATE MACHINE states
        0: Go to pick hover 
        1: Go to pick
        2: Grasp
        2.5: Go to post pick hover
        3: Go to place hover
        4: Go to place
        5: Ungrasp
        6: Go to reset

        STATE MACHINE transitions
        0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 3 -> 6
        """
        self.FINAL_POSE_RESET = get_posestamped(np.array([0.5, 0.2, 0.3]),
                                       np.array([1,0,0,0]))
        if self.current_toy_state == 0: # pick hover
            hover_pose = copy.deepcopy(self.pick_pose)
            hover_pose.pose.position.z += 0.2
            # check if arrived at hover pick pose
            norm_diff = self.get_current_pose_norm(hover_pose, curr_pose)
            if norm_diff < self.NORM_DIFF_TOL:
                self.previous_toy_state = 0
                self.current_toy_state = 1
                print("Arrived at hover pick pose")
                return self.get_next_joint_planner_toy_joints(curr_pose, curr_gripper_width)
                
            # plan to hover pick pose
            next_action = self.get_next_pose_interp(hover_pose.pose, curr_pose)
            return next_action, curr_gripper_width
        
        elif self.current_toy_state == 1: # pick
            # check if arrived at pick pose
            norm_diff = self.get_current_pose_norm(self.pick_pose, curr_pose)
            if norm_diff < self.NORM_DIFF_TOL and abs(curr_pose.position.z - Z_PICK) < self.Z_ABS_TOL:
                self.previous_toy_state = 1
                self.current_toy_state = 2
                print("Arrived at pick pose")
                return self.get_next_joint_planner_toy_joints(curr_pose, curr_gripper_width)
            
            # plan to pick pose
            next_action = self.get_next_pose_interp(self.pick_pose.pose, curr_pose, speed=0.015)
            return next_action, curr_gripper_width
            
        elif self.current_toy_state == 2: # grasp
            # check if current gripper width is less than FINAL_GRIPPER_WIDTH
            if curr_gripper_width < self.FINAL_GRIPPER_WIDTH:
                self.previous_toy_state = 2
                self.current_toy_state = 2.5
                print("Gripper Closed")
                return self.get_next_joint_planner_toy_joints(curr_pose, curr_gripper_width)
        
            # close gripper for 0.01 m
            next_gripper_width = curr_gripper_width - 0.01
            return curr_pose, next_gripper_width

        elif self.current_toy_state == 2.5: # pick hover
            hover_pose = copy.deepcopy(self.pick_pose)
            hover_pose.pose.position.z += 0.1
            # check if arrived at hover pick pose
            norm_diff = self.get_current_pose_norm(hover_pose, curr_pose)
            if norm_diff < self.NORM_DIFF_TOL:
                self.previous_toy_state = 2.5
                self.current_toy_state = 3
                print("Arrived at post pick hover pose")
                return self.get_next_joint_planner_toy_joints(curr_pose, curr_gripper_width)

            # plan to hover pick pose
            next_action = self.get_next_pose_interp(hover_pose.pose, curr_pose)
            return next_action, curr_gripper_width
        
        elif self.current_toy_state == 3: # place hover
            hover_pose = copy.deepcopy(self.place_pose)
            hover_pose.pose.position.z += 0.1
            # check if arrived at hover place pose
            norm_diff = self.get_current_pose_norm(hover_pose, curr_pose)
            if norm_diff < self.NORM_DIFF_TOL:
                if self.previous_toy_state == 5:
                    self.previous_toy_state = 3
                    self.current_toy_state = 6
                else:
                    self.previous_toy_state = 3
                    self.current_toy_state = 4
                print("Arrived at hover place pose")
                return self.get_next_joint_planner_toy_joints(curr_pose, curr_gripper_width)

            # plan to hover place pose
            next_action = self.get_next_pose_interp(hover_pose.pose, curr_pose)
            return next_action, curr_gripper_width
        
        elif self.current_toy_state == 4: # place
            # check if arrived at place pose
            norm_diff = self.get_current_pose_norm(self.place_pose, curr_pose)
            # print place pose and curr pose
            if norm_diff < self.NORM_DIFF_TOL and abs(curr_pose.position.z - Z_PICK) < self.Z_ABS_TOL:
                self.previous_toy_state = 4
                self.current_toy_state = 5
                print("Arrived at place pose")
                return self.get_next_joint_planner_toy_joints(curr_pose, curr_gripper_width)
            
            # plan to place pose
            next_action = self.get_next_pose_interp(self.place_pose.pose, curr_pose, speed=0.015)
            return next_action, curr_gripper_width
        
        elif self.current_toy_state == 5: # ungrasp
            # check if current gripper width is more than FINAL_GRIPPER_WIDTH
            if curr_gripper_width > 0.07:
                self.previous_toy_state = 5
                self.current_toy_state = 3
                print("Gripper Opened, action complete")
                # record accurate place pose
                self.place_pose = get_posestamped(np.array([curr_pose.position.x, curr_pose.position.y, Z_PICK]),
                                                  np.array([1,0,0,0]))
                return self.get_next_joint_planner_toy_joints(curr_pose, curr_gripper_width)
            
            # open gripper for 0.01 m 
            next_gripper_width = curr_gripper_width + 0.01
            return curr_pose, next_gripper_width
        
        elif self.current_toy_state == 6: # reset
            # check if arrived at reset pose
            norm_diff = self.get_current_pose_norm(self.FINAL_POSE_RESET, curr_pose)
            if norm_diff < self.NORM_DIFF_TOL:
                self.previous_toy_state = 6
                self.current_toy_state = 0
                print("Arrived at reset pose")
                return None, None
            
            next_action = self.get_next_pose_interp(self.FINAL_POSE_RESET.pose, curr_pose)
            return next_action, curr_gripper_width
        
        else:
            print("Invalid State")
            return None, None
        
    gripped = False
    def collect_trajectories_joints(self, expt_data_dict):
        self.pick_pose = expt_data_dict["pick_pose"]
        self.place_pose = expt_data_dict["place_pose"]
        expt_folder = '/home/ros_ws/logs/recorded_trajectories/'+ expt_data_dict["experiment_name"]
        if not os.path.exists(expt_folder):
            os.makedirs(expt_folder)
        
        if not self.got_image:
            # wait for 3 seconds to get image
            print("Waiting for image")
            rospy.sleep(3)
        if self.got_image == False:
            print("No image received")
            sys.exit(0)

        if not os.path.exists(expt_folder):
            os.makedirs(expt_folder)

        for i in range (expt_data_dict["n_trajectories"]):
            print(f"--------------Recording Trajectory {i}--------------")

            # set different seeds when recording train and eval trajectories
            if expt_data_dict["eval_mode"]:
                np.random.seed(1234*i)
            else:
                # set seed based on time
                np.random.seed(int(rospy.Time.now().to_time()))
                
            # reset to home position
            self.previous_toy_state = -1
            self.current_toy_state = 0
            self.fa.reset_joints()
            self.fa.open_gripper()
            
            rate = rospy.Rate(EXPERT_RECORD_FREQUENCY)
            self.fa.goto_pose(self.fa.get_pose(), duration=120, dynamic=True, buffer_time=10,
                cartesian_impedances=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:3] + FC.DEFAULT_ROTATIONAL_STIFFNESSES)
        
            # Variables for recording data
            obs, acts, timesteps, tool_poses_i = [], [], [], []
            images = []

            # Start recording data
            counter = 0
            init_time = rospy.Time.now().to_time()
            print("Start Recording Trajectory: ", i)
            while not rospy.is_shutdown():
                curr_joints = self.fa.get_joints()
                curr_pose_rigid = self.fa.get_pose()
                curr_pose = get_posestamped(curr_pose_rigid.translation, [curr_pose_rigid.quaternion[1], curr_pose_rigid.quaternion[2], curr_pose_rigid.quaternion[3], curr_pose_rigid.quaternion[0]]).pose
                curr_gripper_width = self.fa.get_gripper_width()
                next_pose, next_gripper = self.get_next_joint_planner_toy_joints(curr_pose, curr_gripper_width)
                if next_pose is None and next_gripper is None:
                    break
                    
                # print("Counter: ", counter)
                # print("Curr Pose: ", curr_pose, "\n Curr Gripper Width: ", curr_gripper_width)
                # print("Next Pose: ", next_pose, "\n Next Gripper Width: ", next_gripper, "gripped: ", self.gripped)
                # print("***************"*3)
                # print()

                # Record data
                obs.append([curr_joints, curr_gripper_width])
                tool_poses_i.append(curr_pose)
                acts.append([next_pose, next_gripper])
                images.append(self.image)
                timesteps.append(rospy.Time.now().to_nsec())
                
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

                # gripper actuate
                if (next_gripper > (curr_gripper_width + 0.008)):
                    self.gripped = False
                if (next_gripper - self.GRASP_WIDTH < 0.01) and (not self.gripped):
                    # self.fa.goto_gripper(width = FC.GRIPPER_WIDTH_MIN, 
                    #                      grasp=True, block=False, force = FC.GRIPPER_MAX_FORCE)
                    # print("calling with grasp")
                    self.fa.stop_gripper()
                    self.fa.close_gripper()
                    self.gripped = True
                elif not self.gripped:
                    self.fa.goto_gripper(next_gripper, block=False, speed=0.2)
                counter += 1
                rate.sleep()
            
            # Stop the skill
            # Alternatively can call fa.stop_skill()
            term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
            ros_msg = make_sensor_group_msg(
                termination_handler_sensor_msg=sensor_proto2ros_msg(
                    term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
                )
            
            self.pub.publish(ros_msg)
            self.fa.stop_skill()

            # Stop recording data
            data = {
                "observations": obs,
                "actions": acts,
                "timestamps": timesteps,
                "tool_poses": tool_poses_i,
                # "audio_data": self.audio_data,
                "image_data": images
            }
            print("Recorded Trajectory of length: ", len(obs))
            
            # Get the next trajectory number to store the data to
            traj_nums = []
            for file in os.listdir(expt_folder):
                if file.endswith(".pkl"):
                    traj_nums.append(int(file.split("_")[-1].split(".")[0])) # get the number after the last underscore and before the .pkl
            if len(traj_nums) == 0:
                traj_num = 0
            else:
                traj_num = max(traj_nums)+1
            
            print("Saving Trajectory: ", traj_num)
                
            # Save data
            with open('/home/ros_ws/logs/recorded_trajectories/'+ expt_data_dict["experiment_name"] + '/'+ expt_data_dict["experiment_name"] + '_' + str(traj_num) + '.pkl', 'wb') as f:
                pickle.dump(data, f)

            self.reset()
            
    def goto_hover_pose(self, given_pose: RigidTransform):
        hover_pose = copy.deepcopy(given_pose)
        hover_pose.translation[2] += 0.2
        self.fa.goto_pose(hover_pose, duration=3)
    
    def composed_pick_drop(self, given_pose: RigidTransform, pick=True):
        # goto hover pose above given pose
        self.goto_hover_pose(given_pose)
        
        # goto given pose
        self.fa.goto_pose(given_pose, duration=3)
        
        # either pick or drop
        if pick:
            self.fa.close_gripper()
        else:
            self.fa.open_gripper()
        
        # reset to hover pose
        self.goto_hover_pose(given_pose)

    # Boundaries for randomly placing the block to after reset
    LOWER_X, LOWER_Y = 0.4, -0.33
    UPPER_X, UPPER_Y = 0.6, 0.06
    def reset(self):
        """
        Resets the object
        """
        print("Resetting")
        # Compute new place pose
        new_x = np.random.uniform(self.LOWER_X, self.UPPER_X); new_y = np.random.uniform(self.LOWER_Y, self.UPPER_Y)
        new_place_pose = get_posestamped(np.array([new_x, new_y, Z_PICK]), np.array([1,0,0,0]))
        
        reset_from_pose = getRigidTransform(self.place_pose.pose)
        reset_to_pose = getRigidTransform(new_place_pose.pose)

        # pickup from reset_from_pose
        self.composed_pick_drop(reset_from_pose, pick=True)
        
        # drop at reset_to_pose
        self.composed_pick_drop(reset_to_pose, pick=False)
        
        # record new pick pose
        curr_pose = self.fa.get_pose()
        self.pick_pose = get_posestamped(np.array([curr_pose.translation[0], curr_pose.translation[1], Z_PICK]), np.array([1,0,0,0]))
        
        print("Reset Done")
        print("--------------Trajectory Done--------------")


# Utility Functions
def get_mat_norm(transform_mat1, transform_mat2):
    # pose1 and pose2 are of type PoseStamped
    position_1 = np.array([transform_mat1[0,3], transform_mat1[1,3], transform_mat1[2,3]])
    position_2 = np.array([transform_mat2[0,3], transform_mat2[1,3], transform_mat2[2,3]])
    dist_norm = np.linalg.norm(position_1 - position_2)
    r_relative = transform_mat1[0:3,0:3]@transform_mat2[0:3,0:3].T
    relative_angle = np.arccos((np.trace(r_relative)-1)/2)
    return dist_norm + 0.5*relative_angle

def get_pose_norm(pose1, pose2):
    mat1 = pose_to_transformation_matrix(pose1)
    mat2 = pose_to_transformation_matrix(pose2)
    return get_mat_norm(mat1, mat2)

def get_posestamped(translation, orientation):
    """
    translation: x,y,z
    orientation: x,y,z,w
    """
    ret = PoseStamped()
    ret.header.frame_id = "panda_link0"
    ret.pose.position.x = translation[0]
    ret.pose.position.y = translation[1]
    ret.pose.position.z = translation[2]
    ret.pose.orientation.x = orientation[0]
    ret.pose.orientation.y = orientation[1]
    ret.pose.orientation.z = orientation[2]
    ret.pose.orientation.w = orientation[3]
    return ret

def pose_to_transformation_matrix(pose):
    """
    Converts geometry_msgs/Pose to a 4x4 transformation matrix
    """
    if type(pose) == PoseStamped:
        pose = pose.pose
    T = np.eye(4)
    T[0,3] = pose.position.x
    T[1,3] = pose.position.y
    T[2,3] = pose.position.z
    r = spt.Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    T[0:3, 0:3] = r.as_matrix()
    return T

def getRigidTransform(pose):
    translation = np.array([pose.position.x, pose.position.y, pose.position.z])
    r = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    rotation = r.as_matrix()
    from_frame = "franka_tool" 
    to_frame = "world"
    rt = RigidTransform(rotation, translation, from_frame, to_frame)
    return rt

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide experiment_name, num_trajs_to_collect, eval_mode[optional] as command line arguments")
        sys.exit(0)
        
    experiment_name = sys.argv[1]
    num_trajs_to_collect = int(sys.argv[2])
    eval_mode = False
    if(len(sys.argv) > 3):
        eval_mode = bool(sys.argv[3])
    
    rospy.init_node('data_collection')
    
    print("Recording Experiment: ", experiment_name , " with ", num_trajs_to_collect, " trajectories")

    data = Data()

    # Reset Joints
    data.reset_joints()

    # Collect trajectories
    expt_data_dict = {}
    expt_data_dict["experiment_name"] = experiment_name
    expt_data_dict["n_trajectories"] = num_trajs_to_collect
    expt_data_dict["eval_mode"] = eval_mode
    expt_data_dict["pick_pose"] = get_posestamped(np.array([0.47739821, -0.2, Z_PICK]),
                                                  np.array([1,0,0,0]))
    expt_data_dict["place_pose"] = get_posestamped(np.array([0.4758666, 0.23351082, Z_PICK]),
                                                  np.array([1,0,0,0]))
        
    print("Collecting Experiment with Config:\n ", expt_data_dict)
    
    data.collect_trajectories_joints(expt_data_dict)

    print("Trajectories Collected")
    data.fa.reset_joints()