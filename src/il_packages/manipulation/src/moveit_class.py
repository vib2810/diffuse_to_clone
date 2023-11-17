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
from geometry_msgs.msg import PoseStamped, Pose
import pickle
import actionlib
from il_msgs.msg import RecordJointsAction, RecordJointsResult, RecordJointsGoal

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

    def collect_trajectories_manual(self, trajectory_limits_dict, execute_func_record_data: dict, reset_speed=RESET_SPEED):
        """
        Function to collect manual data
        """
        target_pose_orig = get_posestamped(np.array([0.5, 0, 0.3]), np.array([1,0,0,0]))
        for i in range (trajectory_limits_dict["n_trajectories"]):
            print(f"--------------Recording Trajectory {i}--------------")
            np.random.seed(i)
            # sample a start pose and end pose for the trajectory
            start_pose = self.sample_pose(trajectory_limits_dict, target_pose_orig)

            # move robot to start pose
            self.goto_pose_moveit(start_pose, speed = reset_speed)
            
            execute_func_record_data["record_joints"] = True
            
            print("Robot Put in Guide Mode")
            # Put robot in run guide mode
            self.fa.run_guide_mode(10000,block=False)

            # sleep for 1 second to ensure robot is in guide mode
            rospy.sleep(1)

            # Record data till robot moves
            last_joint_pose = self.fa.get_joints()
            rate = rospy.Rate(2)
            started = False
            while not rospy.is_shutdown():
                rate.sleep()
                curr_joint_pose = self.fa.get_joints()
                if np.linalg.norm(curr_joint_pose-last_joint_pose)>0.005 and not started:
                    # Start recording data
                    print("Starting Recording")
                    action_goal = RecordJointsGoal()
                    action_goal.start_msg.trajectory_num.data = execute_func_record_data["trajectory_num"]
                    action_goal.start_msg.pose = target_pose_orig.pose
                    execute_func_record_data["action_client"].send_goal(action_goal)
                    started = True

                if np.linalg.norm(curr_joint_pose-last_joint_pose)<1e-4 and started:
                    # Stop recording data
                    break
                last_joint_pose = curr_joint_pose
            
            print("Finishing Recording")
            execute_func_record_data["action_client"].cancel_goal()
            execute_func_record_data["action_client"].wait_for_result()

            # Stop guide mode
            self.fa.stop_skill()

        self.fa.reset_joints()

    # Variables and Functions for NextJointUsingPlanner computation
    next_joint_itr = 0
    def interpolate_traj_dynamic(self, joints_traj, speed=1):
        """
        Deprecated. The next planner joint is directly used instead in get_next_joint_planner
        Interpolates joints_traj by adding fixed number of points between each point
        If itr<10, then the number of points added is increased to make the trajectory smoother
        """
        interpolated_traj = []
        NUM_INTERP = 5 #fixed as planner returns points with norm ~0.035    
        for i in range(1, joints_traj.shape[0]):
            if i==1 and self.next_joint_itr<20 and False:
                NUM_INTERP_SLOW = int((NUM_INTERP-0.15*self.next_joint_itr) *2.5)//speed
                t_interp = np.linspace(1/NUM_INTERP_SLOW, 1, NUM_INTERP_SLOW)
                t_interp = t_interp**2
            else:
                NUM_INTERP_NORMAL = NUM_INTERP//speed #fixed as planner returns points with norm ~0.035 
                t_interp = np.linspace(1/NUM_INTERP_NORMAL, 1, NUM_INTERP_NORMAL) # linear

            for t_i in range(len(t_interp)):
                dt = t_interp[t_i]
                interp_traj_i = joints_traj[i,:]*dt + joints_traj[i-1,:]*(1-dt)
                interpolated_traj.append(interp_traj_i)
        
        interpolated_traj = np.array(interpolated_traj)
        return interpolated_traj

    def get_next_joint_planner(self, target_pose: Pose):
        self.next_joint_itr += 1

        pose_goal = self.get_moveit_pose_given_frankapy_pose(target_pose.pose)
        plan = self.get_straight_plan_given_pose(pose_goal)
        if len(plan) == 1:
            interpolated_traj= plan
        else:
            interpolated_traj = plan[1:]
        
        # check if goal is reached:
        fa_pose_rigid = self.fa.get_pose()
        fa_pose = get_posestamped(fa_pose_rigid.translation, [fa_pose_rigid.quaternion[1], fa_pose_rigid.quaternion[2], fa_pose_rigid.quaternion[3], fa_pose_rigid.quaternion[0]])
        norm_diff = get_pose_norm(target_pose.pose, fa_pose.pose)
        # print("Norm Diff: ", norm_diff)
        if norm_diff < 0.05:
            return None
        
        return interpolated_traj[0]
        
    def collect_trajectories_noisy(self, trajectory_limits_data, trajectory_num, reset_speed=RESET_SPEED):
        """
        Plans a path from a randomly joint pose to another randomly sampled joint pose
        """
        data = {"observations": [], "actions": [], "timestamps": [], "tool_poses": [], "target_poses": []}
        target_pose = trajectory_limits_data["target_pose"]
        
        num_steps_cnfnd = trajectory_limits_data["num_steps_cnfnd"]
        for i in range (trajectory_limits_data["n_trajectories"]):
            print(f"--------------Recording Trajectory {i}--------------")

            # set different seeds when recording train and eval trajectories
            if trajectory_limits_data["eval_mode"]:
                np.random.seed(1234*i)
            else:
                np.random.seed(i)
                
            # set variables for next_joint_planner
            self.next_joint_itr = 0
            
            # sample a start pose and end pose for the trajectory
            start_pose = self.sample_pose(trajectory_limits_data, target_pose)

            # Move robot to start pose
            self.goto_pose_moveit(start_pose, speed = reset_speed)
            
            rate = rospy.Rate(EXPERT_RECORD_FREQUENCY)
            next_action = None
            while next_action is None:
                next_action = self.get_next_joint_planner(target_pose)
            self.fa.goto_joints(next_action, duration=5, dynamic=True, buffer_time=30, ignore_virtual_walls=True)
            init_time = rospy.Time.now().to_time()
            
            counter = 0
            u_past = np.zeros((num_steps_cnfnd,1))

            # Variables for recording data
            obs, acts, timesteps, tool_poses_i = [], [], [], []
            data["target_poses"].append(target_pose.pose)
            while not rospy.is_shutdown():
                if trajectory_limits_data["noise_type"] == "uniform":
                    u = np.random.uniform(-trajectory_limits_data["noise_std"], trajectory_limits_data["noise_std"], 1)
                elif trajectory_limits_data["noise_type"] == "gaussian":
                    u = np.random.normal(0, trajectory_limits_data["noise_std"], 1)
                else:
                    u = 0
                next_action = self.get_next_joint_planner(target_pose)
                if next_action is None:
                    break
                noise = (1 * u + 1 * np.sum(u_past, axis=0))/1
                next_action += noise
                u_past = np.roll(u_past, 1, axis=0)
                u_past[0,:] = u

                # Record data
                obs.append(self.fa.get_joints())
                tool_poses_i.append(self.fa.get_pose())
                acts.append(next_action)
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
            data["observations"].append(obs)
            data["actions"].append(acts)
            data["timestamps"].append(timesteps)
            data["tool_poses"].append(tool_poses_i)
            print("Recorded Trajectory of length: ", len(obs))

            # Save data
            with open('/home/ros_ws/bags/recorded_trajectories/trajectories_' + str(trajectory_num) + '.pkl', 'wb') as f:
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