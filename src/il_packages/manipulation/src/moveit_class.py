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
from scipy.spatial.transform import Rotation as R
import copy
from geometry_msgs.msg import PoseStamped, Pose
import pickle
import actionlib
import os
# from il_msgs.msg import RecordJointsAction, RecordJointsResult, RecordJointsGoal

sys.path.append("/home/ros_ws/")
from src.git_packages.frankapy.frankapy import FrankaArm, SensorDataMessageType
from src.git_packages.frankapy.frankapy import FrankaConstants as FC
from src.git_packages.frankapy.frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from src.git_packages.frankapy.frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

EXPERT_RECORD_FREQUENCY = 5
RESET_SPEED = 3
class MoveitPlanner():
    def __init__(self) -> None: #None means no return value
        moveit_commander.roscpp_initialize(sys.argv)
        # rospy.init_node('move_group_python_interface_tutorial',anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.group.set_planner_id("RRTstarkConfigDefault")
        self.group.set_end_effector_link("panda_hand")
        self.obs_pub = rospy.Publisher('/planning_scene', PlanningScene, queue_size=10)

        # used to visualize the planned path
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',moveit_msgs.msg.DisplayTrajectory,queue_size=20)

        planning_frame = self.group.get_planning_frame()
        eef_link = self.group.get_end_effector_link()

        print("---------Moveit Planner Class Initialized---------")

        # frankapy 
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.fa = FrankaArm(init_node = False)
    
    # Utility Functions
    def print_robot_state(self):
        print("Joint Values:\n", self.group.get_current_joint_values())
        print("Pose Values (panda_hand):\n", self.group.get_current_pose())
        print("Pose Values (panda_end_effector):\nNOTE: Quaternion is (w,x,y,z)\n", self.fa.get_pose())
     
    def reset_joints(self):
        self.fa.reset_joints()

    def goto_joint(self, joint_goal):
        self.fa.goto_joints(joint_goal, duration=5, dynamic=True, buffer_time=10)
    
    def get_plan_given_pose(self, pose_goal: geometry_msgs.msg.Pose):
        """
        Plans a trajectory given a tool pose goal
        Returns joint_values 
        joint_values: numpy array of shape (N x 7)
        """
        output = self.group.plan(pose_goal)
        plan = output[1]
        joint_values = []
        for i in range(len(plan.joint_trajectory.points)):
            joint_values.append(plan.joint_trajectory.points[i].positions)
        joint_values = np.array(joint_values)
        return joint_values
    
    def get_plan_given_joint(self, joint_goal_list):
        """
        Plans a trajectory given a joint goal
        Returns joint_values and moveit plan
        joint_values: numpy array of shape (N x 7)
        plan: moveit_msgs.msg.RobotTrajectory object
        """
        joint_goal = sensor_msgs.msg.JointState()
        joint_goal.name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
        joint_goal.position = joint_goal_list

        output = self.group.plan(joint_goal)
        plan = output[1]
        joint_values = []
        for i in range(len(plan.joint_trajectory.points)):
            joint_values.append(plan.joint_trajectory.points[i].positions)
        joint_values = np.array(joint_values)
        return joint_values, plan

    def interpolate_traj(self, joints_traj, speed=1):
        num_interp_slow = int(50/speed) # number of points to interpolate for the start and end of the trajectory
        num_interp = int(20/speed) # number of points to interpolate for the middle part of the trajectory
        interpolated_traj = []
        t_linear = np.linspace(1/num_interp, 1, num_interp)
        t_slow = np.linspace(1/num_interp_slow, 1, num_interp_slow)
        t_ramp_up = t_slow**2
        t_ramp_down = 1 - (1-t_slow)**2

        interpolated_traj.append(joints_traj[0,:])
        for t_i in range(len(t_ramp_up)):
            dt = t_ramp_up[t_i]
            interp_traj_i = joints_traj[1,:]*dt + joints_traj[0,:]*(1-dt)
            interpolated_traj.append(interp_traj_i)
            
        for i in range(2, joints_traj.shape[0]-1):
            for t_i in range(len(t_linear)):
                dt = t_linear[t_i]
                interp_traj_i = joints_traj[i,:]*dt + joints_traj[i-1,:]*(1-dt)
                interpolated_traj.append(interp_traj_i)

        for t_i in range(len(t_ramp_down)):
            dt = t_ramp_down[t_i]
            interp_traj_i = joints_traj[-1,:]*dt + joints_traj[-2,:]*(1-dt)
            interpolated_traj.append(interp_traj_i)

        interpolated_traj = np.array(interpolated_traj)
        return interpolated_traj

    def execute_plan(self, joints_traj, speed = 1):
        """
        joints_traj shape: (N x 7)
        """
        # interpolate the trajectory
        interpolated_traj = self.interpolate_traj(joints_traj, speed=speed)

        print('Executing joints trajectory of shape: ', interpolated_traj.shape)
        rate = rospy.Rate(50)

        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        self.fa.goto_joints(interpolated_traj[1], duration=5, dynamic=True, buffer_time=30, ignore_virtual_walls=True)
        init_time = rospy.Time.now().to_time()

        for i in range(2, interpolated_traj.shape[0]):
            traj_gen_proto_msg = JointPositionSensorMessage(
                id=i, timestamp=rospy.Time.now().to_time() - init_time, 
                joints=interpolated_traj[i]
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
            )
            self.pub.publish(ros_msg)
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
    
    def get_straight_plan_given_pose(self, pose_goal: geometry_msgs.msg.Pose):
        """
        pose_goal: geometry_msgs.msg.Pose
        Plans a trajectory given a tool pose goal
        Returns joint_values
        joint_values: numpy array of shape (N x 7)
        """
        waypoints = []
        waypoints.append(copy.deepcopy(pose_goal))

        (plan, fraction) = self.group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )  # jump_threshold

        joint_values = []
        for i in range(len(plan.joint_trajectory.points)):
            joint_values.append(plan.joint_trajectory.points[i].positions)
        joint_values = np.array(joint_values)
        return joint_values

    def get_moveit_pose_given_frankapy_pose(self, pose):
        """
        Converts a frankapy pose (in panda_end_effector frame) to a moveit pose (in panda_hand frame) 
        by adding a 10 cm offset to z direction
        """
        transform_mat =  np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,-0.1034],
                                   [0,0,0,1]])
        pose_mat = pose_to_transformation_matrix(pose)
        transformed = pose_mat @ transform_mat
        pose_goal = transformation_matrix_to_pose(transformed)
        return pose_goal
    
    def goto_pose_moveit(self, pose_goal: PoseStamped, speed = 1):
        """
        Plans and executes a trajectory to a pose goal using moveit
        pose_goal in frame franka_end_effector (frankapy)
        """
        pose_goal = self.get_moveit_pose_given_frankapy_pose(pose_goal.pose)
        plan = self.get_straight_plan_given_pose(pose_goal)
        trajectory_dict = self.execute_plan(plan, speed=speed)
        return trajectory_dict

    def sample_pose(self, trajectory_limits_dict, goal_pose)-> PoseStamped:
        """
        Samples a random pose within the range specified
        xang_range, yang_range, zang_range are in degrees
        """
        x = np.random.uniform(trajectory_limits_dict["xrange"][0], trajectory_limits_dict["xrange"][1])
        y = np.random.uniform(trajectory_limits_dict["yrange"][0], trajectory_limits_dict["yrange"][1])
        z = np.random.uniform(trajectory_limits_dict["zrange"][0], trajectory_limits_dict["zrange"][1])
        xang = np.random.uniform(trajectory_limits_dict["xang_range"][0], trajectory_limits_dict["xang_range"][1])
        yang = np.random.uniform(trajectory_limits_dict["yang_range"][0], trajectory_limits_dict["yang_range"][1])
        zang = np.random.uniform(trajectory_limits_dict["zang_range"][0], trajectory_limits_dict["zang_range"][1])
        orientation = spt.Rotation.from_euler('xyz', [xang, yang, zang], degrees=True).as_quat()
        sample_pose = get_posestamped(np.array([x,y,z]), orientation)
        if(type(goal_pose) == PoseStamped):
            goal_pose = goal_pose.pose
        if get_pose_norm(sample_pose.pose, goal_pose) < trajectory_limits_dict["min_norm_diff"]:
            sample_pose = self.sample_pose(trajectory_limits_dict, goal_pose)
        else:
            print("Start Norm Diff: ", get_pose_norm(sample_pose.pose, goal_pose))
        return sample_pose

    def get_next_joint_planner(self, target_pose: Pose):
        """
        Returns the next joint position to goto from the current position of the robot given a target pose 
        """
        pose_goal = self.get_moveit_pose_given_frankapy_pose(target_pose.pose)
        plan = self.get_straight_plan_given_pose(pose_goal)
        if len(plan) == 1:
            interpolated_traj= plan
        else:
            interpolated_traj = plan[1:]
        
        return interpolated_traj[0]
    
    def get_next_joint_planner_interpolate(self, target_pose: Pose):
        """
        Returns the next tool pose to goto from the current pose of the robot given a target pose 
        """
        current_pose = self.fa.get_pose()
        current_position = np.array([current_pose.translation.x, current_pose.translation.y, current_pose.translation.z])
        r = R.from_quat([current_pose.quaternion[1], current_pose.quaternion[2], current_pose.quaternion[3], current_pose.quaternion[0]])
        current_orientation = r.as_euler('zyx', degrees=True)
        print("Current Pose: ", current_position, current_orientation)

        target_position = np.array([target_pose.position.x, target_pose.position.y, target_pose.position.z])
        r = R.from_quat([target_pose.orientation[0], target_pose.orientation[1], target_pose.orientation[2], target_pose.orientation[3]])
        target_orientation = r.as_euler('zyx', degrees=True)
        print("Target Pose: ", target_position, target_orientation)
        
        unit_diff_in_position = (target_position - current_position) / np.linalg.norm(target_position - current_position)
        if np.linalg.norm(target_position - current_position) < 0.005:
            next_position = target_position
        else:
            next_position = current_position + 0.005*unit_diff_in_position

        diff_in_yaw_deg = target_orientation[0] - current_orientation[0]
        if diff_in_yaw_deg < 0.5:
            next_yaw_in_deg = target_orientation[0]
        else:
            next_yaw_in_deg = current_orientation[0] + np.sign(diff_in_yaw_deg)*0.5

        r = R.from_euler('z', next_yaw_in_deg, degrees=True)
        next_orientation = r.as_quat()

        next_pose = Pose(position=next_position, orientation=next_orientation)
        return next_pose

    def get_current_pose_norm(self, target_pose: Pose, fa_rigid):
        """
        Returns the norm of the difference between the fa_rigid and target_pose
        """
        fa_pose = get_posestamped(fa_rigid.translation, [fa_rigid.quaternion[1], fa_rigid.quaternion[2], fa_rigid.quaternion[3], fa_rigid.quaternion[0]])
        norm_diff = get_pose_norm(target_pose.pose, fa_pose.pose)
        return norm_diff

    current_toy_state = 0 #0: pick, 1: grasp, 2: goto hover place, 3: goto_place, 4: ungrasp
    FINAL_GRIPPER_WIDTH=0.01 # gripper width ranges from 0 to 0.08
    def get_next_joint_planner_toy(self, pick_pose: PoseStamped, place_pose: PoseStamped, curr_joints, curr_pose, curr_gripper_width):
        """
        Takes as input the current robot joints, pose and gripper width
        Returns the next joint position for the toy task
        Returns: next_action(7x1), gripper_width
        """
        if self.current_toy_state == 0: # pick
            # check if arrived at pick pose
            if self.get_pose_norm(pick_pose, curr_pose) < 0.05:
                self.current_toy_state = 1
                print("Arrived at pick pose")
                return self.get_next_joint_planner_toy(pick_pose, place_pose, curr_joints, curr_pose, curr_gripper_width)
            
            # plan to pick pose
            next_action = self.get_next_joint_planner(pick_pose)
            return next_action, curr_gripper_width
        
        elif self.current_toy_state == 1: # grasp
            # check if current gripper width is less than FINAL_GRIPPER_WIDTH
            if curr_gripper_width < self.FINAL_GRIPPER_WIDTH:
                self.current_toy_state = 2
                print("Gripper Closed")
                return self.get_next_joint_planner_toy(pick_pose, place_pose, curr_joints, curr_pose, curr_gripper_width)
        
            # close gripper for 0.01 m
            next_gripper_width = curr_gripper_width - 0.01
            return curr_joints, next_gripper_width
        
        elif self.current_toy_state == 2: # goto hover place
            hover_pose = copy.deepcopy(place_pose)
            hover_pose.pose.position.z += 0.1
            # check if arrived at hover place pose
            if self.get_pose_norm(hover_pose) < 0.05:
                self.current_toy_state = 3
                print("Arrived at hover place pose")
                return self.get_next_joint_planner_toy(pick_pose, place_pose, curr_joints, curr_pose, curr_gripper_width)

            # plan to hover place pose
            next_action = self.get_next_joint_planner(hover_pose)
            return next_action, curr_gripper_width
        
        elif self.current_toy_state == 3: # goto place
            # check if arrived at place pose
            if self.get_pose_norm(place_pose) < 0.05:
                self.current_toy_state = 4
                print("Arrived at place pose")
                return self.get_next_joint_planner_toy(pick_pose, place_pose)
            
            # plan to place pose
            next_action = self.get_next_joint_planner(place_pose)
            return next_action, curr_gripper_width
        
        elif self.current_toy_state == 4: # ungrasp
            # check if current gripper width is more than FINAL_GRIPPER_WIDTH
            if curr_gripper_width > 0.07:
                self.current_toy_state = 0
                print("Gripper Opened, action complete")
                return None, None
            
            # open gripper for 0.01 m
            next_gripper_width = curr_gripper_width + 0.01
            return curr_joints, next_gripper_width
        
        else:
            print("Invalid State")
            return None, None
                            
    def collect_toy_trajectories(self, expt_data_dict):
        """
        Plans a path from a randomly joint pose to another randomly sampled joint pose
        """
        pick_pose = expt_data_dict["pick_pose"]
        place_pose = expt_data_dict["place_pose"]
        expt_folder = '/home/ros_ws/bags/recorded_trajectories/'+ expt_data_dict["experiment_name"]
        if not os.path.exists(expt_folder):
            os.makedirs(expt_folder)

        for i in range (expt_data_dict["n_trajectories"]):
            print(f"--------------Recording Trajectory {i}--------------")

            # set different seeds when recording train and eval trajectories
            if expt_data_dict["eval_mode"]:
                np.random.seed(1234*i)
            else:
                np.random.seed(i)
                
            # reset to home position
            self.current_toy_state = 0
            self.fa.reset_joints()
            
            rate = rospy.Rate(EXPERT_RECORD_FREQUENCY)
            next_action = None
            while next_action is None:
                next_action, _ = self.get_next_joint_planner_toy(pick_pose, place_pose)
            self.fa.goto_joints(next_action, duration=5, dynamic=True, buffer_time=30, ignore_virtual_walls=True)
            init_time = rospy.Time.now().to_time()
            
            # Variables for recording data
            obs, acts, timesteps, tool_poses_i = [], [], [], []

            # Start recording data
            counter = 0
            while not rospy.is_shutdown():
                curr_joints = self.fa.get_joints()
                curr_pose = self.fa.get_pose()
                curr_gripper_width = self.fa.get_gripper_width()
                next_action, next_gripper = self.get_next_joint_planner_toy(pick_pose, place_pose, curr_joints, curr_pose, curr_gripper_width)
                if next_action is None and next_gripper is None:
                    break

                # Record data
                obs.append(np.array([curr_joints, curr_gripper_width]))
                tool_poses_i.append(curr_pose)
                acts.append([next_action, next_gripper])
                timesteps.append(rospy.Time.now().to_nsec())

                traj_gen_proto_msg = JointPositionSensorMessage(
                    id=counter, timestamp=rospy.Time.now().to_time() - init_time, 
                    joints=next_action
                )
                ros_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
                )
                self.pub.publish(ros_msg)

                # gripper actuate
                self.fa.goto_gripper(next_gripper, duration=1/EXPERT_RECORD_FREQUENCY, block=False, speed=0.1)
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
                "tool_poses": tool_poses_i
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
            with open('/home/ros_ws/bags/recorded_trajectories/'+ expt_data_dict["experiment_name"] + '/'+ expt_data_dict["experiment_name"] + '_' + str(traj_num) + '.pkl', 'wb') as f:
                pickle.dump(data, f)


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

def transformation_matrix_to_pose(trans_mat):   
    """
    Converts a 4x4 transformation matrix to geometry_msgs/Pose
    """
    out_pose = geometry_msgs.msg.Pose()
    out_pose.position.x = trans_mat[0,3]
    out_pose.position.y = trans_mat[1,3]
    out_pose.position.z = trans_mat[2,3]

    #convert rotation matrix to quaternion
    r = spt.Rotation.from_matrix(trans_mat[0:3, 0:3])
    quat = r.as_quat() 
    out_pose.orientation.x = quat[0]
    out_pose.orientation.y = quat[1]
    out_pose.orientation.z = quat[2]
    out_pose.orientation.w = quat[3] 
    return out_pose