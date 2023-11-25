#!/usr/bin/env python

from __future__ import print_function

import pybullet as p

from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF

from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver


def test_retraction(robot, info, tool_link, distance=0.1, **kwargs):
    ik_joints = get_ik_joints(robot, info, tool_link)
    start_pose = get_link_pose(robot, tool_link)
    end_pose = multiply(start_pose, Pose(Point(z=-distance)))
    handles = [add_line(point_from_pose(start_pose), point_from_pose(end_pose), color=BLUE)]
    #handles.extend(draw_pose(start_pose))
    #handles.extend(draw_pose(end_pose))
    path = []
    pose_path = list(interpolate_poses(start_pose, end_pose, pos_step_size=0.01))
    for i, pose in enumerate(pose_path):
        print('Waypoint: {}/{}'.format(i+1, len(pose_path)))
        handles.extend(draw_pose(pose))
        conf = next(either_inverse_kinematics(robot, info, tool_link, pose, **kwargs), None)
        if conf is None:
            print('Failure!')
            path = None
            wait_for_user()
            break
        set_joint_positions(robot, ik_joints, conf)
        path.append(conf)
        wait_for_user()
        # for conf in islice(ikfast_inverse_kinematics(robot, info, tool_link, pose, max_attempts=INF, max_distance=0.5), 1):
        #    set_joint_positions(robot, joints[:len(conf)], conf)
        #    wait_for_user()
    remove_handles(handles)
    return path

def test_ik(robot, info, tool_link, tool_pose):
    draw_pose(tool_pose)
    # TODO: sort by one joint angle
    # TODO: prune based on proximity
    ik_joints = get_ik_joints(robot, info, tool_link)
    for conf in either_inverse_kinematics(robot, info, tool_link, tool_pose, use_pybullet=False,
                                          max_distance=INF, max_time=10, max_candidates=INF):
        # TODO: profile
        set_joint_positions(robot, ik_joints, conf)
        wait_for_user()

#####################################

def main():
    connect(use_gui=True)
    add_data_path()
    draw_pose(Pose(), length=1.)
    set_camera_pose(camera_point=[1, -1, 1])

    plane = p.loadURDF("plane.urdf")
    obstacles = [plane] # TODO: collisions with the ground

    # Load the first robot facing along the positive x-axis
    with LockRenderer(), HideOutput(True):
        robot1 = load_pybullet(FRANKA_URDF, fixed_base=True, base_position=[0.5, 0, 0], base_orientation=[0, 0, 0, 1])  # Specify base position and orientation for robot1
        assign_link_colors(robot1, max_colors=3, s=0.5, v=1.)

    dump_body(robot1)

    # Load the second robot, rotated to face the first
    with LockRenderer(), HideOutput(True):
        robot2 = load_pybullet(FRANKA_URDF, fixed_base=True, base_position=[-0.5, 0, 0], base_orientation=[0, 0, 1, 0])  # Specify base position and orientation for robot2
        assign_link_colors(robot2, max_colors=3, s=0.5, v=1.)

    dump_body(robot2)
    
    print('Start?')
    wait_for_user()

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

    sample_fn1 = get_sample_fn(robot1, joints1)
    sample_fn2 = get_sample_fn(robot2, joints2)
    for i in range(10):
        print('Iteration:', i)
        conf1 = sample_fn1()
        conf2 = sample_fn2()
        set_joint_positions(robot1, joints1, conf1)
        set_joint_positions(robot2, joints2, conf2)
        wait_for_user()
        #test_ik(robot, info, tool_link, get_link_pose(robot, tool_link))
        test_retraction(robot1, info, tool_link1, use_pybullet=False,
                        max_distance=0.1, max_time=0.05, max_candidates=100)
        test_retraction(robot2, info, tool_link2, use_pybullet=False,
                        max_distance=0.1, max_time=0.05, max_candidates=100)
    disconnect()

if __name__ == '__main__':
    main()
