import numpy as np
import pickle
import rospy
import tf
from sensor_msgs.msg import JointState
from tf.transformations import quaternion_from_matrix
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

# =================== DMPMotionGenerator ===================
class DMPMotionGenerator:
    def __init__(self, urdf_path, mesh_path=None, joint_names=None, base_link="world", end_effector_link="end_effector_link"):
        from movement_primitives.kinematics import Kinematics
        self.urdf_path = urdf_path
        self.mesh_path = mesh_path
        self.kin = Kinematics(open(urdf_path).read(), mesh_path=mesh_path)
        self.joint_names = joint_names or ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.base_link = base_link
        self.end_effector_link = end_effector_link
        self.chain = self.kin.create_chain(self.joint_names, base_link, end_effector_link)
        self.dmp = None

    def learn_from_rosbag(self, bag_path, joint_topic, dt=None, n_weights=10):
        import rosbag, pytransform3d.trajectories as ptr
        transforms, joint_trajectory, gripper_trajectory, time_stamp = self._process_rosbag(bag_path, joint_topic)
        Y = ptr.pqs_from_transforms(transforms)
        if dt is None:
            dt = 1/self.frequency
        from movement_primitives.dmp import CartesianDMP
        self.dmp = CartesianDMP(execution_time=max(time_stamp), dt=dt, n_weights_per_dim=n_weights)
        self.dmp.imitate(time_stamp, Y)
        return Y, transforms, joint_trajectory, gripper_trajectory

    def _process_rosbag(self, bag_path, joint_topic):
        import rosbag
        transforms, joint_trajectory, gripper_trajectory, time_stamp = [], [], [], []
        bag = rosbag.Bag(bag_path)
        for topic, msg, t in bag.read_messages(topics=[joint_topic]):
            joint_pos = msg.position[:6]
            gripper_pos = msg.position[6]
            joint_trajectory.append(joint_pos)
            gripper_trajectory.append(gripper_pos)
            transforms.append(self.chain.forward(joint_pos))
            time_stamp.append(msg.header.stamp.to_sec())    
        bag.close()
        transforms = np.array(transforms)
        joint_trajectory = np.array(joint_trajectory)
        gripper_trajectory = np.array(gripper_trajectory)
        time_stamp = np.array(time_stamp)
        dt = np.diff(time_stamp)
        self.frequency = 1 / np.average(dt)
        positions = np.array([T[:3, 3] for T in transforms])
        mask = self.remove_outliers_mad(positions, threshold=12.0)[0]
        filtered_time = time_stamp[mask]
        normalized_time = filtered_time - filtered_time[0]
        return transforms[mask], joint_trajectory[mask], gripper_trajectory[mask], normalized_time

    def remove_outliers_mad(self, data, threshold=3.5):
        median = np.median(data, axis=0)
        diff = np.abs(data - median)
        mad = np.median(diff, axis=0)
        mod_z = 0.6745 * diff / (mad + 1e-6)
        mask = np.all(mod_z < threshold, axis=1)
        return mask, data[mask]

    def generate_trajectory(self, start_y=None, goal_y=None):
        import pytransform3d.trajectories as ptr
        if self.dmp is None:
            raise ValueError("No DMP model available. Learn or load a model first.")
        if start_y is not None:
            self.dmp.start_y = start_y
        if goal_y is not None:
            self.dmp.goal_y = goal_y
        T, Y = self.dmp.open_loop()
        trajectory = ptr.transforms_from_pqs(Y)
        return T, trajectory

    def save_dmp(self, filepath):
        if self.dmp is None:
            raise ValueError("No DMP model available to save")
        with open(filepath, 'wb') as f:
            pickle.dump(self.dmp, f)

    def load_dmp(self, filepath):
        with open(filepath, 'rb') as f:
            self.dmp = pickle.load(f)

    def compute_IK_trajectory(self, trajectory, time_stamp, q0=None, subsample_factor=1):
        if q0 is None:
            q0 = np.array([0.0, -0.78, 1.5, 0., 0.8, 0.])
        sub_traj = trajectory[::subsample_factor] if subsample_factor > 1 else trajectory
        sub_time = time_stamp[::subsample_factor] if subsample_factor > 1 else time_stamp
        random_state = np.random.RandomState(0)
        joint_trajectory = self.chain.inverse_trajectory(sub_traj, random_state=random_state)
        return sub_traj, joint_trajectory, sub_time

# =================== ROS_OM_Node ===================
class ROS_OM_Node:
    def __init__(self, joint_names, topic_name='/gravity_compensation_controller/traj_joint_states', rate_hz=20):
        if not rospy.core.is_initialized():
            rospy.init_node("om_pick_and_place", anonymous=True)
        self.publisher = rospy.Publisher(topic_name, JointState, queue_size=10)
        self.listener = tf.TransformListener()
        self.joint_names = joint_names + ["gripper"]
        self.rate = rospy.Rate(rate_hz)
        self.gripper = 0.01
        self.position = []
        self.current_joint_positions = None
        self.arm_joint_names = joint_names.copy()
        self.start_position = [-0.40719096, -0.36561228, 1.09260597, -0.02829443, 1.02020369, 0.07299084]
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)

    def publish_trajectory(self, joint_trajectory, timestamps):
        start_time = rospy.Time.now()
        for i in range(len(joint_trajectory)):
            if rospy.is_shutdown():
                break
            msg = JointState()
            msg.header.stamp = start_time + rospy.Duration.from_sec(timestamps[i] - timestamps[0])
            msg.name = self.joint_names
            self.position = joint_trajectory[i].tolist() + [self.gripper]
            vel_eff = [0.0] * len(self.joint_names)
            msg.velocity = vel_eff   
            msg.effort = vel_eff
            msg.position = self.position
            self.publisher.publish(msg)
            self.rate.sleep()

    def set_gripper(self, gripper_position=0.01):
        self.gripper = gripper_position
        msg = JointState()
        msg.name = self.joint_names
        vel_eff = [0.0] * len(self.joint_names)
        msg.velocity = vel_eff   
        msg.effort = vel_eff 
        msg.position = self.position
        msg.position[-1] = self.gripper
        self.publisher.publish(msg)
        rospy.sleep(5)
        self.rate.sleep()

    def publish_home_position(self, home_position=None, execution_time=5.0, steps=100):
        if home_position is None:
            home_position = [-0.03834952, -1, 1.26093221, 0.00613592, 1.825, -0.00460194]
        while (self.current_joint_positions is None or np.allclose(self.current_joint_positions[:6], np.zeros(6))) and not rospy.is_shutdown():
            rospy.sleep(0.1)
        start_joints = np.array(self.current_joint_positions[:6])
        goal_joints = np.array(home_position)
        rate = rospy.Rate(steps / execution_time)
        for i in range(1, steps + 1):
            alpha = i / float(steps)
            interp_joints = (1 - alpha) * start_joints + alpha * goal_joints
            self.position = interp_joints.tolist() + [self.gripper]
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = self.joint_names
            msg.position = self.position
            msg.velocity = [0.0] * len(self.joint_names)
            msg.effort = [0.0] * len(self.joint_names)
            self.publisher.publish(msg)
            rate.sleep()

    def joint_state_callback(self, msg):
        if set(self.arm_joint_names).issubset(set(msg.name)):
            name_to_pos = dict(zip(msg.name, msg.position))
            self.current_joint_positions = [name_to_pos[name] for name in self.arm_joint_names]

    def publish_joint2_only(self, delta_joint2=-0.8, execution_time=5.0, steps=100):
        while (self.current_joint_positions is None or np.allclose(self.current_joint_positions[:6], np.zeros(6))) and not rospy.is_shutdown():
            rospy.sleep(0.1)
        start_joints = np.array(self.current_joint_positions[:6])
        goal_joints = start_joints.copy()
        current_joint2 = start_joints[1]
        target_joint2 = max(current_joint2 + delta_joint2, -2.042035225)
        goal_joints[1] = target_joint2
        rate = rospy.Rate(steps / execution_time)
        for i in range(1, steps + 1):
            alpha = i / float(steps)
            interp_joints = (1 - alpha) * start_joints + alpha * goal_joints
            self.position = interp_joints.tolist() + [self.gripper]
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = self.joint_names
            msg.position = self.position
            msg.velocity = [0.0] * len(self.joint_names)
            msg.effort = [0.0] * len(self.joint_names)
            self.publisher.publish(msg)
            rate.sleep()

    def execute_pick(self, pick, pick_angle=0.0, cube_name=""):
        joint_traj_1, time_stamps_1 = gen_pick(
            '/root/catkin_ws/src/shaky_new/DMP_ToH/execution_scripts/recordings/dmp/home2pick.pkl',
            pick_position=pick,
            pick_angle=pick_angle)
        try:
            self.gripper = -0.005
            self.publish_home_position(home_position=self.start_position, execution_time=5.0)
            rospy.sleep(0.1)
            self.publish_trajectory(joint_traj_1[0], time_stamps_1[0])
            rospy.sleep(1)
            self.publish_trajectory(joint_traj_1[1], time_stamps_1[1])
            rospy.sleep(1)
            self.set_gripper(0.006)
            self.publish_joint2_only()
            rospy.sleep(4.0)
            self.publish_home_position(execution_time=5.0)
            rospy.sleep(7.0)
        except rospy.ROSInterruptException:
            pass

    def execute_place(self, place, place_angle=0.0, cube_name=""):
        joint_traj_1, time_stamps_1 = gen_place(
            '/root/catkin_ws/src/shaky_new/DMP_ToH/execution_scripts/recordings/dmp/home2pick.pkl',
            place_position=place,
            place_angle=place_angle)
        try:
            self.gripper = 0.006
            self.publish_home_position(home_position=self.start_position, execution_time=5.0)
            rospy.sleep(0.1)
            self.publish_trajectory(joint_traj_1[0], time_stamps_1[0])
            rospy.sleep(3)
            self.publish_trajectory(joint_traj_1[1], time_stamps_1[1])
            rospy.sleep(3)
            self.set_gripper(-0.005)
            self.publish_joint2_only()
            rospy.sleep(6.0)
            self.publish_home_position(execution_time=5.0)
            rospy.sleep(7.0)
        except rospy.ROSInterruptException:
            pass

# =================== Helper Functions ===================
def save_trajectory_data(joint_trajectory, timestamps, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump({'trajectory': joint_trajectory, 'timestamps': timestamps}, f)

def load_trajectory_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['trajectory'], data['timestamps']

def interpolate_joint_trajectory(joint_traj, time_stamps, target_freq=20.0):
    num_joints = joint_traj.shape[1]
    duration = time_stamps[-1] - time_stamps[0]
    num_samples = int(duration * target_freq)
    new_timestamps = np.linspace(time_stamps[0], time_stamps[-1], num_samples)
    interp_traj = np.zeros((num_samples, num_joints))
    for i in range(num_joints):
        interpolator = interp1d(time_stamps, joint_traj[:, i], kind='linear', fill_value="extrapolate")
        interp_traj[:, i] = interpolator(new_timestamps)
    return interp_traj, new_timestamps

def gen_trajectory(dmp_path, start=None, goal=None, visualize=False, store_cart_traj=False, name=''):
    if start is None:
        start = np.zeros(7)
    if goal is None:
        goal = np.zeros(7)
    urdf_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/urdf/open_manipulator_6dof.urdf'
    mesh_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/meshes'
    dmp_gen = DMPMotionGenerator(urdf_path, mesh_path, base_link="world")
    dmp_gen.load_dmp(dmp_path)
    new_start = dmp_gen.dmp.start_y.copy() if np.all(start == 0) else start
    new_goal = dmp_gen.dmp.goal_y.copy() if np.all(goal == 0) else goal
    T, trajectory = dmp_gen.generate_trajectory(start_y=new_start, goal_y=new_goal)
    if store_cart_traj:
        store_cart_traj_path = dmp_path.replace('/dmp/', '/cart_traj/').replace('.pkl', f'_{name}.pkl')
        save_trajectory_data(trajectory, T, store_cart_traj_path)
    trajectory, IK_joint_trajectory, T = dmp_gen.compute_IK_trajectory(trajectory, T, subsample_factor=10)
    if visualize:
        dmp_gen.visualize_trajectory(trajectory, IK_joint_trajectory)
    IK_joint_trajectory = IK_joint_trajectory[:IK_joint_trajectory.shape[0], :]
    interpolated_traj, interpolated_time = interpolate_joint_trajectory(IK_joint_trajectory, T, target_freq=20.0)
    return interpolated_traj, interpolated_time, new_goal

def gripper_orientation_pick(position):
    x_, y_, z_ = position
    gripper_x = np.array([-1.0, 0.0, 0.0])
    to_origin_zy = np.array([0.0, -y_, -z_])
    norm = np.linalg.norm(to_origin_zy)
    if norm < 1e-6:
        raise ValueError("Position too close to origin in ZY plane; Y-axis undefined.")
    gripper_y = to_origin_zy / norm
    gripper_z = np.cross(gripper_x, gripper_y)
    gripper_y = np.cross(gripper_z, gripper_x)
    gripper_y /= np.linalg.norm(gripper_y)
    gripper_z /= np.linalg.norm(gripper_z)
    rot_matrix = np.eye(4)
    rot_matrix[:3, 0] = gripper_x
    rot_matrix[:3, 1] = gripper_y
    rot_matrix[:3, 2] = gripper_z
    quat = quaternion_from_matrix(rot_matrix)
    pose = np.concatenate([position, quat])
    return pose

def rotate_pose_around_y(pose, phi=0.0):
    position = pose[:3]
    orientation_quat = pose[3:]
    r_orig = R.from_quat(orientation_quat)
    r_y = R.from_euler('y', phi)
    r_new = r_y * r_orig
    return np.concatenate([position, r_new.as_quat()])

def gen_pick(path, pick_position, position_home=np.array([0.07, 0, 0.275]), pick_angle=0.0):
    joint_traj = np.empty(7, dtype=object)
    time_stamps = np.empty(7, dtype=object)
    dmp_path = path
    position = pick_position.copy()
    position[2] = 0.2
    pose = gripper_orientation_pick(position)
    pose = rotate_pose_around_y(pose, np.deg2rad(75))
    joint_traj[0], time_stamps[0], goal = gen_trajectory(dmp_path, goal=pose, visualize=True, store_cart_traj=False, name='pick1')
    new_start = goal
    position = pick_position
    pose = gripper_orientation_pick(position)
    pose = rotate_pose_around_y(pose, pick_angle)
    joint_traj[1], time_stamps[1], goal = gen_trajectory(dmp_path, start=new_start, goal=pose, visualize=True, store_cart_traj=False, name='pick2')
    return joint_traj, time_stamps

def gen_place(path, place_position, position_home=np.array([0.00, 0, 0.21]), place_angle=0.0):
    joint_traj = np.empty(3, dtype=object)
    time_stamps = np.empty(3, dtype=object)
    dmp_path = path
    position = place_position.copy()
    position[2] = 0.2
    pose = gripper_orientation_pick(position)
    pose = rotate_pose_around_y(pose, np.deg2rad(75))
    joint_traj[0], time_stamps[0], goal = gen_trajectory(dmp_path, goal=pose, visualize=True, store_cart_traj=False, name='pick1')
    new_start = goal
    position = place_position
    pose = gripper_orientation_pick(position)
    pose = rotate_pose_around_y(pose, place_angle)
    joint_traj[1], time_stamps[1], goal = gen_trajectory(dmp_path, start=new_start, goal=pose, visualize=True, store_cart_traj=False, name='place2')
    return joint_traj, time_stamps

def main():
    try:
        om_node = ROS_OM_Node(['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
    except rospy.ROSInterruptException:
        print("ROS node interrupted.")
    position_pick = np.array([0.2, 0.2, 0.0])
    pick_angle = np.deg2rad(45)
    place_angle = np.deg2rad(45)
    position_place = np.array([0.2, -0.2, 0.0])
    om_node.execute_pick(position_pick, pick_angle)
    om_node.execute_place(position_place, place_angle)

if __name__ == "__main__":
    main()
