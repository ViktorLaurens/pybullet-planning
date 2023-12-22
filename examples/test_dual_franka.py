#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import pybullet as p

import sys
import os

# Add the parent directory (pybullet-planning) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, wait_if_gui, \
    get_movable_joints, get_sample_fn, get_joint_positions, set_joint_positions, plan_joint_motion, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF, get_links, get_link_name, wait_for_duration

from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, pybullet_inverse_kinematics, closest_inverse_kinematics, either_inverse_kinematics, check_ik_solver
from pybullet_tools.panda_never_collisions import NEVER_COLLISIONS

class Robot:
    def __init__(self, body, info, tool_link):
        self.body = body
        self.info = info
        self.tool_link = tool_link

# def test_retraction(robot, info, tool_link, distance=0.1, **kwargs):
#     ik_joints = get_ik_joints(robot, info, tool_link)
#     start_pose = get_link_pose(robot, tool_link)
#     end_pose = multiply(start_pose, Pose(Point(z=-distance)))
#     handles = [add_line(point_from_pose(start_pose), point_from_pose(end_pose), color=BLUE)]
#     #handles.extend(draw_pose(start_pose))
#     #handles.extend(draw_pose(end_pose))
#     path = []
#     pose_path = list(interpolate_poses(start_pose, end_pose, pos_step_size=0.01))
#     for i, pose in enumerate(pose_path):
#         print('Waypoint: {}/{}'.format(i+1, len(pose_path)))
#         handles.extend(draw_pose(pose))
#         conf = next(either_inverse_kinematics(robot, info, tool_link, pose, **kwargs), None)
#         if conf is None:
#             print('Failure!')
#             path = None
#             wait_for_user()
#             break
#         set_joint_positions(robot, ik_joints, conf)
#         path.append(conf)
#         wait_for_user()
#         # for conf in islice(ikfast_inverse_kinematics(robot, info, tool_link, pose, max_attempts=INF, max_distance=0.5), 1):
#         #    set_joint_positions(robot, joints[:len(conf)], conf)
#         #    wait_for_user()
#     remove_handles(handles)
#     return path

# def test_ik(robot, info, tool_link, tool_pose):
#     draw_pose(tool_pose)
#     # TODO: sort by one joint angle
#     # TODO: prune based on proximity
#     ik_joints = get_ik_joints(robot, info, tool_link)
#     for conf in closest_inverse_kinematics(robot, info, tool_link, tool_pose, use_pybullet=False,
#                                           max_distance=INF, max_time=10, max_candidates=INF):
#         # TODO: profile
#         set_joint_positions(robot, ik_joints, conf)
#         wait_for_user()

# def test_arm_motion(robot, joints, goal_conf):
#     disabled_collisions = get_disabled_collisions(robot)
#     wait_if_gui('Plan Arm?')
#     with LockRenderer(lock=False):
#         arm_path = plan_joint_motion(robot, joints, goal_conf, disabled_collisions=disabled_collisions)
#     if arm_path is None:
#         print('Unable to find an arm path')
#         return
#     print(len(arm_path))
#     for q in arm_path:
#         set_joint_positions(robot, joints, q)
#         #wait_if_gui('Continue?')
#         wait_for_duration(0.01)

def get_disabled_collisions(panda):
    disabled_names = NEVER_COLLISIONS
    link_mapping = {get_link_name(panda, link): link for link in get_links(panda)}
    return {(link_mapping[name1], link_mapping[name2])
            for name1, name2 in disabled_names if (name1 in link_mapping) and (name2 in link_mapping)}

def plan_franka_motion(robot, arm_joints, goal_conf, start_conf=None, obstacles=[]):
    """
    Plan motion for Franka Panda robot from start to goal configuration
    """
    disabled_collisions = get_disabled_collisions(robot)
    if start_conf is None:
        get_joint_positions(robot)
    with LockRenderer(lock=False):
        arm_path = plan_joint_motion(robot, arm_joints, goal_conf, obstacles=obstacles, disabled_collisions=disabled_collisions)
    if arm_path is None:
        print('Unable to find an arm path')
        return None
    return arm_path

def execute_motion(robot, arm_joints, arm_path):
    """
    Execute the planned motion on Franka Panda robot
    """
    for q in arm_path:
        set_joint_positions(robot, arm_joints, q)
        wait_for_duration(0.05)

#####################################

def main():
    connect(use_gui=True)
    add_data_path()
    draw_pose(Pose(), length=1.)
    set_camera_pose(camera_point=[0, -1.2, 1.2])

    plane = p.loadURDF("plane.urdf")
    obstacles = [] # TODO: collisions with the ground

    # Load the first robot facing along the positive x-axis
    with LockRenderer(), HideOutput(True):
        robot1 = load_pybullet(FRANKA_URDF, fixed_base=True, base_position=[0.5, 0, 0], base_orientation=[0, 0, 1, 0])  # Specify base position and orientation for robot1
        assign_link_colors(robot1, max_colors=3, s=0.5, v=1.)

    dump_body(robot1)

    # Load the second robot, rotated to face the first
    with LockRenderer(), HideOutput(True):
        robot2 = load_pybullet(FRANKA_URDF, fixed_base=True, base_position=[-0.5, 0, 0], base_orientation=[0, 0, 0, 1])  # Specify base position and orientation for robot2
        assign_link_colors(robot2, max_colors=3, s=0.5, v=1.)

    dump_body(robot2)
    
    wait_if_gui('Set robot 1 in so-called neutral pose?')
    arm_standby_pose = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    arm_neutral_pose = (0, 0, 0, -1.51, 0, 1.877, 0, 0.04, 0.04)
    arm1_start = (1, 0.785, -1, -1.51, 0, 1.877, 0, 0.04, 0.04)
    arm1_goal = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    arm2_start = (0, 0, 0, -1.51, 0, 1.877, 0, 0.04, 0.04)
    arm2_goal = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    info = PANDA_INFO
    tool_link1 = link_from_name(robot1, 'panda_hand')
    tool_link2 = link_from_name(robot2, 'panda_hand')
    draw_pose(Pose(), parent=robot1, parent_link=tool_link1)
    draw_pose(Pose(), parent=robot2, parent_link=tool_link2)
    joints1 = get_movable_joints(robot1)
    joints2 = get_movable_joints(robot2)
    print('Joints1', [get_joint_name(robot1, joint) for joint in joints1])
    print('Joints2', [get_joint_name(robot2, joint) for joint in joints2])
    check_ik_solver(info)

    # robot2 = Robot(body=robot2_body, info=panda_info, tool_link=tool_link2)


    wait_if_gui('Set robot 1 in so-called neutral pose?')
    set_joint_positions(robot1, joints1, arm_neutral_pose)
    wait_if_gui('Set robot 2 in so-called standby pose?')
    set_joint_positions(robot2, joints2, arm_standby_pose)

    # # determine joint angles using IK
    # print('IK?')
    # wait_for_user()
    # ik_joints = get_ik_joints(robot1, info, tool_link1)
    # tool_position = (-0.1, -0.4, 0.5) 
    # tool_orientation = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    # tool_pose = (tool_position, tool_orientation)
    # for conf in either_inverse_kinematics(robot1, info, tool_link1, tool_pose, use_pybullet=False,
    #                                       max_distance=INF, max_time=10, max_candidates=INF):
    #     # TODO: profile
    #     print("IK Configuration:", conf)
    #     set_joint_positions(robot1, ik_joints, conf)
    #     wait_for_user()

    # --- ARM1 --- #
    print('Proceeding with respectively goal and start pose for arm1...')
    ik_joints1 = get_ik_joints(robot1, info, tool_link1)

    # Calculating IK for the goal pose
    print('Calculating IK for the goal pose...')
    tool_position_goal1 = (-0.1, 0.4, 0.5)
    tool_orientation_goal1 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    tool_pose_goal1 = (tool_position_goal1, tool_orientation_goal1)
    arm1_goal = next(either_inverse_kinematics(robot1, info, tool_link1, tool_pose_goal1), None)

    if arm1_goal is not None:
        wait_if_gui('Show goal pose for arm1?')
        print("IK Configuration for panda 1 goal pose:", arm1_goal)
        set_joint_positions(robot1, ik_joints1, arm1_goal)
    else:
        print("No IK solution found for panda 1 goal pose.")

    # Calculating IK for the start pose
    print('Calculating IK for the start pose...')
    tool_position_start1 = (-0.1, -0.4, 0.5)
    tool_orientation_start1 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    tool_pose_start1 = (tool_position_start1, tool_orientation_start1)
    arm1_start = next(closest_inverse_kinematics(robot1, info, tool_link1, tool_pose_start1), None)

    if arm1_start is not None:
        wait_if_gui('Show start pose for arm1?')
        print("IK Configuration for panda 1 start pose:", arm1_start)
        set_joint_positions(robot1, ik_joints1, arm1_start)
    else:
        print("No IK solution found for panda 1 start pose.")


    # --- ARM2 --- #
    print('Proceeding with respectively goal and start pose for arm2...')
    ik_joints2 = get_ik_joints(robot2, info, tool_link2)

    # Calculating IK for the goal pose
    print('Calculating IK for the goal pose...')
    tool_position_goal2 = (0.1, -0.4, 0.5)
    tool_orientation_goal2 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    tool_pose_goal2 = (tool_position_goal2, tool_orientation_goal2)
    arm2_goal = next(either_inverse_kinematics(robot2, info, tool_link2, tool_pose_goal2), None)

    if arm2_goal is not None:
        wait_if_gui('Show goal pose for arm2?')
        print("IK Configuration for panda 2 goal pose:", arm2_goal)
        set_joint_positions(robot2, ik_joints2, arm2_goal)
    else:
        print("No IK solution found for panda 2 goal pose.")

    # Calculating IK for the start pose
    print('Calculating IK for the start pose...')
    tool_position_start2 = (0.1, 0.4, 0.5)
    tool_orientation_start2 = p.getQuaternionFromEuler([np.radians(180), 0, 0])
    tool_pose_start2 = (tool_position_start2, tool_orientation_start2)
    arm2_start = next(closest_inverse_kinematics(robot2, info, tool_link2, tool_pose_start2), None)

    if arm2_start is not None:
        wait_if_gui('Show start pose for arm2?')
        print("IK Configuration for panda 2 start pose:", arm2_start)
        set_joint_positions(robot2, ik_joints2, arm2_start)
    else:
        print("No IK solution found for panda 2 start pose.")

    # Plan motion
    wait_if_gui('Plan both arms together?')
    # arm_path1 = plan_franka_motion(robot1, ik_joints1, arm1_goal, arm1_start, obstacles=[robot2])
    arm_path2 = plan_franka_motion(robot2, ik_joints2, arm2_goal, arm2_start, obstacles=[robot1])
    # if arm_path1 is not None:
    #     execute_motion(robot1, ik_joints1, arm_path1)
    if arm_path2 is not None:
        execute_motion(robot2, ik_joints2, arm_path2)
    
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