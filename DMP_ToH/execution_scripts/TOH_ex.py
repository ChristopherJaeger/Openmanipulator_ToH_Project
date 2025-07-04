import rospy
from std_msgs.msg import String
import numpy as np
import tf
from pick_from_vision_V4 import ROS_OM_Node

class TOH:
    def __init__(self, joint_names, request_topic='/object_request', rate_hz=1, tower_a=None, tower_b=None, tower_c=None):
        rospy.init_node('continuous_pick_and_place', anonymous=True)
        self.request_pub = rospy.Publisher(request_topic, String, queue_size=1)
        self.listener = tf.TransformListener()
        self.om_node = ROS_OM_Node(joint_names)
        self.rate = rospy.Rate(rate_hz)

        self.tower_origins = {
            'A': np.array([0.28, -0.14, 0.05]),
            'B': np.array([0.28, 0.0, 0.05]),
            'C': np.array([0.28, 0.14, 0.05])
        }
        self.towers = {
            'A': tower_a or [],
            'B': tower_b or [],
            'C': tower_c or []
        }
        self.block_height = 0.07

    def get_object_pose_world(self, target_frame="world", object_frame="detected_object"):
        while not rospy.is_shutdown():
            try:
                self.listener.waitForTransform(target_frame, object_frame, rospy.Time(0), rospy.Duration(5.0))
                trans, rot = self.listener.lookupTransform(target_frame, object_frame, rospy.Time(0))
                return np.array(trans), np.array(rot)
            except Exception as e:
                rospy.logwarn(f"Waiting for transform from '{object_frame}' to '{target_frame}': {str(e)}")
                self.rate.sleep()

    def get_object_trans(self, obj_name, supervisor=True):
        while not rospy.is_shutdown():
            trans, _ = self.get_object_pose_world(object_frame=obj_name)
            if trans is not None:
                if supervisor:
                    user_input = input(f"Is the received position for '{obj_name}' accurate? (Y/n): ").strip().lower()
                    if user_input in ["y", ""]:
                        return trans
                    rospy.sleep(0.5)
                else:
                    return trans
            rospy.sleep(0.5)

    def execute_pick_and_place(self, pick_cube, placetower, supervisor=False):
        trans = self.get_object_trans(pick_cube, supervisor=supervisor)
        if trans is None:
            rospy.logerr(f"Could not get position for {pick_cube}")
            return None
        pick_position = np.array(trans) + np.array([-0.025, 0.015, 0.005])
        self.om_node.execute_pick(pick_position, np.deg2rad(40), cube_name=pick_cube)
        self.rate.sleep()

        tower_stack = self.towers.get(placetower)
        if tower_stack is None:
            rospy.logerr(f"Unknown tower '{placetower}'")
            return None

        if not tower_stack:
            place_pos = self.tower_origins[placetower].copy() + np.array([0.0, 0.0, 0.01])
            below_block = None
        else:
            top_block = tower_stack[-1]
            top_block_pos = self.get_object_trans(top_block, supervisor=supervisor)
            if top_block_pos is None:
                rospy.logerr(f"Could not get position for top block '{top_block}' on tower '{placetower}'")
                return None
            place_pos = np.array(top_block_pos) + np.array([-0.03, 0.015, 0.02])
            below_block = top_block

        self.om_node.execute_place(place_pos, np.deg2rad(40), cube_name=pick_cube)
        self.rate.sleep()

        # Update tower configuration
        for blocks in self.towers.values():
            if pick_cube in blocks:
                blocks.remove(pick_cube)
                break
        tower_stack.append(pick_cube)
        rospy.loginfo(f"Moved {pick_cube} to tower {placetower} on top of {below_block}")
        return below_block

    def get_tower_configuration(self):
        """Returns the current configuration of the towers as a dict."""
        return {tower_name: list(blocks) for tower_name, blocks in self.towers.items()}

def process_hanoi_commands(commands, toh):
    block_map = {'1': 'two', '2': 'three', '3': 'four'}
    for cmd in commands:
        if not cmd.startswith("MD") or len(cmd) != 5:
            rospy.logwarn(f"Invalid command skipped: {cmd}")
            continue
        block_id, from_tower, to_tower = cmd[2], cmd[3], cmd[4]
        cube_name = block_map.get(block_id)
        if not cube_name:
            rospy.logwarn(f"Unknown block {block_id} in command {cmd}")
            continue
        towers = toh.get_tower_configuration()
        if towers[from_tower] and towers[from_tower][-1] == cube_name:
            rospy.loginfo(f"Executing command: {cmd} ({cube_name} from {from_tower} to {to_tower})")
            toh.execute_pick_and_place(cube_name, to_tower, supervisor=True)
        else:
            rospy.logwarn(f"Block {cube_name} is not on top of tower {from_tower}, command {cmd} skipped.")

if __name__ == '__main__':
    try:
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        toh = TOH(joint_names=joint_names, tower_a=[], tower_b=['three', 'two'], tower_c=['four'])
        command_list = ["MD1BC"]
        process_hanoi_commands(command_list, toh)
        print(toh.get_tower_configuration())
    except rospy.ROSInterruptException:
        pass
