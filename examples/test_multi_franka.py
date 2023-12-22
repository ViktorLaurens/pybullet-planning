#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pybullet as p
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add the parent directory (pybullet-planning) to the Python path
from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, wait_if_gui, \
    get_movable_joints, get_sample_fn, get_joint_positions, set_joint_positions, plan_joint_motion, plan_multi_robot_motion, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF, get_links, get_link_name, wait_for_duration
from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, pybullet_inverse_kinematics, closest_inverse_kinematics, either_inverse_kinematics, check_ik_solver
from pybullet_tools.panda_never_collisions import NEVER_COLLISIONS

# Classes
class Robot:
    def __init__(self, info, urdf_model, tool_link_name, weights=None, base_position=[0, 0, 0], base_orientation=[0, 0, 1, 0]):
        self.body = load_pybullet(urdf_model, fixed_base=True, base_position=base_position, base_orientation=base_orientation)
        self.info = info
        self.tool_link = link_from_name(self.body, tool_link_name)
        self.movable_joints = get_movable_joints(self.body)
        self.ik_joints = get_ik_joints(self.body, self.info, self.tool_link)
        self.weights = weights

    def set_pose(self, joints, pose):
        set_joint_positions(self.body, joints, pose)

    def plan_motion(self, goal_conf, start_conf=None, obstacles=[]):
        # Plan motion logic
        pass

    def execute_motion(self, arm_path):
        # Execute motion logic
        pass

    # Additional methods as needed


######################################################
# Functions
def get_disabled_collisions(panda):
    disabled_names = NEVER_COLLISIONS
    link_mapping = {get_link_name(panda, link): link for link in get_links(panda)}
    return {(link_mapping[name1], link_mapping[name2])
            for name1, name2 in disabled_names if (name1 in link_mapping) and (name2 in link_mapping)}

def plan_multi_franka_motion(robots, goal_confs, start_confs=None, obstacles=[]):
    """
    Plan motion for multiple Franka Panda robots from start to goal configurations.
    :param robots: List of robot objects.
    :param goal_confs: List of goal configurations, one for each robot.
    :param start_confs: List of start configurations, one for each robot (optional).
    :param obstacles: List of obstacles in the environment.
    :return: List of paths for each robot or None if unable to find a solution.
    """

    if start_confs is None:
        start_confs = [get_joint_positions(robot.body, robot.ik_joints) for robot in robots]

    disabled_collisions = [get_disabled_collisions(robot.body) for robot in robots]

    with LockRenderer(lock=False):
        arm_paths = plan_multi_robot_motion(robots, goal_confs, start_confs=start_confs, 
                                            obstacles=obstacles, disabled_collisions=disabled_collisions)

    if arm_paths is None:
        print('Unable to find arm paths for all robots')
        return None

    return arm_paths

def execute_multi_robot_motion(robots, arm_paths):
    """
    Execute the planned motion on multiple Franka Panda robots.
    :param robots: List of robot objects.
    :param arm_paths: List of paths for each robot.
    """
    # Ensure the number of paths matches the number of robots
    assert len(robots) == len(arm_paths), "Number of paths must match number of robots"

    # Find the longest path to determine the overall duration of execution
    max_length = max(len(path) for path in arm_paths)

    for step in range(max_length):
        for robot, path in zip(robots, arm_paths):
            # Check if the current robot has a step at this index
            if step < len(path):
                q = path[step]
                set_joint_positions(robot.body, robot.ik_joints, q)

        # Wait for a short duration before moving to the next step
        wait_for_duration(0.05)


#####################################
# Main
def main():
    connect(use_gui=True)
    add_data_path()
    draw_pose(Pose(), length=1.)
    set_camera_pose(camera_point=[0, -1.2, 1.2])

    plane = p.loadURDF("plane.urdf")
    obstacles = [] # TODO: collisions with the ground
    info = PANDA_INFO

    # Load the first robot facing along the positive x-axis
    with LockRenderer(), HideOutput(True):
        robot1 = Robot(PANDA_INFO, FRANKA_URDF, 'panda_hand', base_position=[0.5, 0, 0], base_orientation=[0, 0, 1, 0])
        # robot1 = load_pybullet(FRANKA_URDF, fixed_base=True, base_position=[0.5, 0, 0], base_orientation=[0, 0, 1, 0])  # Specify base position and orientation for robot1
        assign_link_colors(robot1.body, max_colors=3, s=0.5, v=1.)

    dump_body(robot1.body)

    # Load the second robot, rotated to face the first
    with LockRenderer(), HideOutput(True):
        robot2 = Robot(PANDA_INFO, FRANKA_URDF, 'panda_hand', base_position=[-0.5, 0, 0], base_orientation=[0, 0, 0, 1])
        # robot2 = load_pybullet(FRANKA_URDF, fixed_base=True, base_position=[-0.5, 0, 0], base_orientation=[0, 0, 0, 1])  # Specify base position and orientation for robot2
        assign_link_colors(robot2.body, max_colors=3, s=0.5, v=1.)

    dump_body(robot2.body)
    
    wait_if_gui('Set robot 1 in so-called neutral pose?')
    arm_standby_pose = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    arm_neutral_pose = (0, 0, 0, -1.51, 0, 1.877, 0, 0.04, 0.04)
    arm1_start = (1, 0.785, -1, -1.51, 0, 1.877, 0, 0.04, 0.04)
    arm1_goal = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    arm2_start = (0, 0, 0, -1.51, 0, 1.877, 0, 0.04, 0.04)
    arm2_goal = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    # info = PANDA_INFO
    # tool_link1 = link_from_name(robot1, 'panda_hand')
    # tool_link2 = link_from_name(robot2, 'panda_hand')
    draw_pose(Pose(), parent=robot1.body, parent_link=robot1.tool_link)
    draw_pose(Pose(), parent=robot2.body, parent_link=robot2.tool_link)
    # joints1 = get_movable_joints(robot1.body)
    # joints2 = get_movable_joints(robot2.body)
    print('Joints1', [get_joint_name(robot1.body, joint) for joint in robot1.movable_joints])
    print('IK_joints1', robot1.ik_joints)
    print('Joints2', [get_joint_name(robot2.body, joint) for joint in robot2.movable_joints])
    print('IK_joints2', robot2.ik_joints)
    check_ik_solver(info)

    # robot2 = Robot(body=robot2_body, info=panda_info, tool_link=tool_link2)

    wait_if_gui('Set robot 1 in so-called neutral pose?')
    robot1.set_pose(robot1.movable_joints, arm_neutral_pose)
    wait_if_gui('Set robot 2 in so-called standby pose?')
    robot2.set_pose(robot2.movable_joints, arm_standby_pose)

    # --- ARM1 --- #
    print('Proceeding with respectively goal and start pose for arm1...')
    # ik_joints1 = get_ik_joints(robot1, info, tool_link1)

    # Calculating IK for the goal pose
    print('Calculating IK for the goal pose...')
    tool_position_goal1 = (-0.1, 0.4, 0.5)
    tool_orientation_goal1 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    tool_pose_goal1 = (tool_position_goal1, tool_orientation_goal1)
    arm1_goal = next(either_inverse_kinematics(robot1.body, robot1.info, robot1.tool_link, tool_pose_goal1), None)

    if arm1_goal is not None:
        wait_if_gui('Show goal pose for arm1?')
        print("IK Configuration for panda 1 goal pose:", arm1_goal)
        robot1.set_pose(robot1.ik_joints, arm1_goal)
    else:
        print("No IK solution found for panda 1 goal pose.")

    # Calculating IK for the start pose
    print('Calculating IK for the start pose...')
    tool_position_start1 = (-0.1, -0.4, 0.5)
    tool_orientation_start1 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    tool_pose_start1 = (tool_position_start1, tool_orientation_start1)
    arm1_start = next(closest_inverse_kinematics(robot1.body, robot1.info, robot1.tool_link, tool_pose_start1), None)

    if arm1_start is not None:
        wait_if_gui('Show start pose for arm1?')
        print("IK Configuration for panda 1 start pose:", arm1_start)
        robot1.set_pose(robot1.ik_joints, arm1_start)
    else:
        print("No IK solution found for panda 1 start pose.")

    wait_if_gui('Set robot 1 in so-called neutral pose?')
    robot1.set_pose(robot1.movable_joints, arm_neutral_pose)


    # --- ARM2 --- #
    print('Proceeding with respectively goal and start pose for arm2...')
    # ik_joints2 = get_ik_joints(robot2., info, tool_link2)

    # Calculating IK for the goal pose
    print('Calculating IK for the goal pose...')
    tool_position_goal2 = (0.1, -0.4, 0.5)
    tool_orientation_goal2 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    tool_pose_goal2 = (tool_position_goal2, tool_orientation_goal2)
    arm2_goal = next(either_inverse_kinematics(robot2.body, robot2.info, robot2.tool_link, tool_pose_goal2), None)

    if arm2_goal is not None:
        wait_if_gui('Show goal pose for arm2?')
        print("IK Configuration for panda 2 goal pose:", arm2_goal)
        robot2.set_pose(robot2.ik_joints, arm2_goal)
    else:
        print("No IK solution found for panda 2 goal pose.")

    # Calculating IK for the start pose
    print('Calculating IK for the start pose...')
    tool_position_start2 = (0.1, 0.4, 0.5)
    tool_orientation_start2 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    tool_pose_start2 = (tool_position_start2, tool_orientation_start2)
    arm2_start = next(closest_inverse_kinematics(robot2.body, robot2.info, robot2.tool_link, tool_pose_start2), None)

    if arm2_start is not None:
        wait_if_gui('Show start pose for arm2?')
        print("IK Configuration for panda 2 start pose:", arm2_start)
        robot2.set_pose(robot2.ik_joints, arm2_start)
    else:
        print("No IK solution found for panda 2 start pose.")

    # Plan motion
    wait_if_gui('Plan both arms together?')
    # arm_path1 = plan_franka_motion(robot1, ik_joints1, arm1_goal, arm1_start, obstacles=[robot2])
    robots = [robot1, robot2]
    goal_confs = [arm1_goal, arm2_goal]
    start_confs = [arm1_start, arm2_start]
    arm_paths = plan_multi_franka_motion(robots, goal_confs, start_confs, obstacles=[])
    # if arm_path1 is not None:
    #     execute_motion(robot1, ik_joints1, arm_path1)
    if arm_paths is not None:
        execute_multi_robot_motion(robots, arm_paths)
    
    # # determine joint angles using IK for panda 2
    # print('set panda 2?')
    # wait_for_user()
    # ik_joints2 = get_ik_joints(robot2, info, tool_link2)
    # tool_position = (0.1, 0.4, 0.5) 
    # tool_orientation = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    # tool_pose = (tool_position, tool_orientation)
    # arm2_start = next(pybullet_inverse_kinematics(robot2, info, tool_link2, tool_pose), None)
    # # Check if a solution was found
    # if arm2_start is not None:
    #     print("IK Configuration for panda 2:", arm2_start)
    #     set_joint_positions(robot2, ik_joints2, arm2_start)
    #     wait_for_user()
    # else:
    #     print("No IK solution found for panda 2.")
    
    wait_if_gui('Disconnect?')
    disconnect()

if __name__ == '__main__':
    main()