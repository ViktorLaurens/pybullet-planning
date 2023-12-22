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

# For RRT
from motion.motion_planners.primitives import extend_towards
from motion.motion_planners.rrt import TreeNode, configs
from motion.motion_planners.utils import irange, RRT_ITERATIONS, INF, elapsed_time
import time
import random

# For planning (utils)
from pybullet_tools.utils import MAX_DISTANCE, get_default_weights, is_circular, circular_difference, get_custom_limits, interval_generator, get_default_resolutions, get_refine_fn, \
    CIRCULAR_LIMITS, get_self_link_pairs, get_moving_links, can_collide, get_limits_fn, CollisionPair, cached_fn, get_buffered_aabb, aabb_overlap, pairwise_link_collision, pairwise_collision, \
    get_extend_fn, get_distance_fn, get_collision_fn
from motion.motion_planners.meta import solve


# Classes
class Robot:
    def __init__(self, info, urdf_model, tool_link_name, weights=None, base_position=[0, 0, 0], base_orientation=[0, 0, 1, 0]):
        self.body = load_pybullet(urdf_model, fixed_base=True, base_position=base_position, base_orientation=base_orientation)
        self.info = info
        self.tool_link = link_from_name(self.body, tool_link_name)
        self.movable_joints = get_movable_joints(self.body)
        self.ik_joints = get_ik_joints(self.body, self.info, self.tool_link)
        self.weights = weights
        self.path = None

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
### Functions for RRT
def random_swap(nodes1, nodes2):
    p = float(len(nodes1)) / (len(nodes1) + len(nodes2))
    swap = (random.random() < p)
    return swap

def rrt_connect_multi_robots(start_qs, goal_qs, distance_fn, extend_fn, composite_sample_fn, composite_collision_fn,
                max_iterations=RRT_ITERATIONS, max_time=INF, **kwargs):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :param max_iterations: Maximum number of iterations - int
    :param max_time: Maximum runtime - float
    :param kwargs: Keyword arguments
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: goal sampling function connected to a None node
    start_time = time.time()
    if composite_collision_fn(start_qs) or composite_collision_fn(goal_qs):
        return None
    # TODO: support continuous collision_fn with two arguments
    #collision_fn = wrap_collision_fn(collision_fn)
    start_q = np.concatenate(start_qs)
    goal_q = np.concatenate(goal_qs)
    nodes1, nodes2 = [TreeNode(start_q)], [TreeNode(goal_q)] # TODO: allow a tree to be pre-specified (possibly as start)
    for iteration in irange(max_iterations):
        if elapsed_time(start_time) >= max_time:
            break
        #swap = alternating_swap(nodes1, nodes2)
        swap = random_swap(nodes1, nodes2)
        tree1, tree2 = nodes1, nodes2
        if swap:
            tree1, tree2 = nodes2, nodes1

        target = composite_sample_fn()
        target = np.concatenate(target)
        last1, _ = extend_towards(tree1, target, distance_fn, extend_fn, composite_collision_fn, swap, **kwargs) # Will extend tree 1 in direction of target as long as no collisions occur or target reached 
        last2, success = extend_towards(tree2, last1.config, distance_fn, extend_fn, composite_collision_fn, not swap, **kwargs) #tries to connect tree 2 with last node added to tree 1

        if success:
            path1, path2 = last1.retrace(), last2.retrace()
            if swap:
                path1, path2 = path2, path1
            #print('{} max_iterations, {} nodes'.format(iteration, len(nodes1) + len(nodes2)))
            path = configs(path1[:-1] + path2[::-1])
            return path
    return None

def concatenate_qs(configs):
    """
    Concatenate a list of configurations (NumPy arrays) into a single array.
    :param configs: List of NumPy arrays representing the configurations.
    :return: A single concatenated NumPy array.
    """
    return np.concatenate(configs)

def split_combined_q(combined_q, dimensions):
    """
    Split a combined configuration array into individual configurations.
    :param combined_config: The combined NumPy array of configurations.
    :param dimensions: List of integers representing the number of joints (dimensions) for each robot.
    :return: List of NumPy arrays, each representing a robot's configuration.
    """
    configs = []
    index = 0
    for dim in dimensions:
        configs.append(combined_q[index:index + dim])
        index += dim
    return configs

### Functions for planning
def get_composite_difference_fn(robots):
    # Create a list of circular joint flags for all robots
    all_circular_joints = [is_circular(robot.body, joint) for robot in robots for joint in robot.ik_joints]

    def fn(q2, q1):
        assert len(q2) == len(q1) == len(all_circular_joints), "Configuration lengths must match"
        # Compute the difference for each joint
        return tuple(circular_difference(value2, value1) if circular else (value2 - value1)
                     for circular, value2, value1 in zip(all_circular_joints, q2, q1))
    return fn

def get_composite_distance_fn(robots, weights=None, norm=2):
    # Concatenate weights for all robots
    all_weights = []
    for robot in robots:
        robot_weights = get_default_weights(robot.body, robot.ik_joints, weights)
        all_weights.extend(robot_weights)

    difference_fn = get_composite_difference_fn(robots)

    def fn(q1, q2):

        assert len(q1) == len(q2) == len(all_weights), "Configuration and weight lengths must match"
        diff = np.array(difference_fn(q2, q1))

        if norm == 2:
            return np.sqrt(np.dot(all_weights, diff * diff))
        else:
            weighted_diff = np.multiply(all_weights, diff)
            return np.linalg.norm(weighted_diff, ord=norm)

    return fn

def get_composite_sample_fn(robots, custom_limits={}, **kwargs):
    generators = []
    for robot in robots:
        lower_limits, upper_limits = get_custom_limits(robot.body, robot.ik_joints, custom_limits, circular_limits=CIRCULAR_LIMITS)
        generator = interval_generator(lower_limits, upper_limits, **kwargs)
        generators.append(generator)

    def fn():
        return tuple(next(gen) for gen in generators)

    return fn

def get_composite_extend_fn(robots, resolutions=None, norm=2):
    # norm = 1, 2, INF
    resolutions = []
    for robot in robots:
        resolutions.append(get_default_resolutions(robot.body, robot.ik_joints, resolutions))
    difference_fn = get_composite_difference_fn(robots)
    def fn(q1, q2):
        #steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions))))
        steps = int(np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=norm))
        refine_fn = get_refine_fn(robot.body, robot.ik_joints, num_steps=steps)
        return refine_fn(q1, q2)
    return fn

def get_composite_collision_fn(robots, obstacles=[], attachments=[], self_collisions=True, disabled_collisions=set(),
                               custom_limits={}, use_aabb=False, cache=False, max_distance=MAX_DISTANCE, **kwargs):
    check_link_pairs = []
    # attached_bodies = []
    moving_bodies = []
    limits_fn = []
    for robot in robots: 
        # attached_bodies.append([attachment.child for attachment in attachments])
        # moving_bodies.append([CollisionPair(robot.body, moving_links)] + list(map(parse_body, attached_bodies)))
        robot_check_link_pairs = get_self_link_pairs(robot.body, robot.ik_joints, disabled_collisions) if self_collisions else []
        robot_moving_links = frozenset(link for link in get_moving_links(robot.body, robot.ik_joints)
                                       if can_collide(robot.body, link))
        robot_limits_fn = get_limits_fn(robot.body, robot.ik_joints, custom_limits=custom_limits)

        check_link_pairs.append(robot_check_link_pairs)
        moving_bodies.append(CollisionPair(robot.body, robot_moving_links))
        limits_fn.append(robot_limits_fn)
    
    get_obstacle_aabb = cached_fn(get_buffered_aabb, cache=cache, max_distance=max_distance/2., **kwargs)
    robot_collision_pairs = [(robot1, robot2) for robot1 in robots for robot2 in robots if robot1 != robot2]
    # attached_bodies = [attachment.child for attachment in attachments]
    # obstacle_bodies = list(map(parse_body, obstacles + attached_bodies))

    def collision_fn(qs, verbose=False):
        for q, robot, robot_check_link_pairs, robot_moving_body, robot_limits_fn in zip(qs, robots, check_link_pairs, moving_bodies, limits_fn):
            assert len(q) == len(robot.ik_joints), "Configuration length must match number of joints in robot"
            if robot_limits_fn(q):
                return True
            set_joint_positions(robot.body, robot.ik_joints, q)
            get_moving_aabb = cached_fn(get_buffered_aabb, cache=True, max_distance=max_distance/2., **kwargs)

            if self_collisions:
                for link1, link2 in robot_check_link_pairs:
                    if (not use_aabb or aabb_overlap(get_moving_aabb(robot.body), get_moving_aabb(robot.body))) and \
                        pairwise_link_collision(robot.body, link1, robot.body, link2):
                        if verbose: print(robot.body, link1, robot.body, link2)
                        return True

            for obstacle in obstacles:
                if (not use_aabb or aabb_overlap(get_moving_aabb(robot_moving_body), get_obstacle_aabb(obstacle))) and \
                    pairwise_collision(robot_moving_body, obstacle):
                    if verbose: print(robot.body, obstacle)
                    return True

        for robot1, robot2 in robot_collision_pairs:
            if (not use_aabb or aabb_overlap(get_moving_aabb(robot1.body), get_moving_aabb(robot2.body))) and \
                    pairwise_collision(robot1.body, robot2.body):
                if verbose: print(robot1.body, robot2.body)
                return True

        return False

    return collision_fn

def check_multi_initial_end(start_confs, end_confs, collision_fn, verbose=True):
    # TODO: collision_fn might not accept kwargs
    if collision_fn(start_confs, verbose=verbose):
        print('Warning: initial configuration is in collision')
        return False
    if collision_fn(end_confs, verbose=verbose):
        print('Warning: end configuration is in collision')
        return False
    return True



def plan_multi_robot_motion(robots, end_confs, start_confs, obstacles=[], attachments=[],
                        self_collisions=True, disabled_collisions=set(),
                        weights=None, resolutions=None, max_distance=MAX_DISTANCE,
                        use_aabb=False, cache=True, custom_limits={}, algorithm=None, **kwargs):
    """
    Plan motion for multiple robots.
    :param robots: List of robot bodies.
    :param end_confs: List of end configurations for each robot.
    :param obstacles: Shared environmental obstacles.
    :param attachments: Attachments for each robot.
    :param ...: Other parameters.
    :return: Paths for each robot or None if unable to find a solution.
    """
    assert len(robots) == len(end_confs) == len(start_confs) # Number of robots must match number of end configurations
    start_q = concatenate_qs(start_confs)
    start_confs = split_combined_q(start_q, [7,7])
    for robot, end_conf in zip(robots, end_confs):
        len(robot.ik_joints) == len(end_conf) # Mismatch in number of IK joints and end configuration length for a robot?
        if (robot.weights is None) and (resolutions is not None):
            robot.weights = np.reciprocal(resolutions)
    
    composite_distance_fn = get_composite_distance_fn(robots, weights=weights)
    composite_sample_fn = get_composite_sample_fn(robots, custom_limits=custom_limits)
    composite_extend_fn = get_composite_extend_fn(robots, resolutions=resolutions)
    composite_collision_fn = get_composite_collision_fn(robots, obstacles, attachments, self_collisions, disabled_collisions,
                                                        custom_limits, max_distance, use_aabb, cache) # This function should check for collisions between robots and between each robot and the environment
    
    # check that starting and end configurations are feasible
    if not check_multi_initial_end(start_confs, end_confs, composite_collision_fn):
        return None

    # If algorithm is not specified, use bi-directional RRT by default
    if algorithm is None:
        # from motion_planners.rrt_connect import birrt
        # from motion.motion_planners.rrt_connect import rrt_connect_multi_robots
        paths = rrt_connect_multi_robots(start_confs, end_confs, composite_distance_fn, composite_extend_fn, composite_sample_fn, composite_collision_fn, **kwargs)
    else:
        paths = solve(start_confs, end_confs, composite_distance_fn, composite_sample_fn, composite_extend_fn, composite_collision_fn,
                    algorithm=algorithm, weights=weights, **kwargs)
    if paths is None:
        return None
    return paths

def parametrize_paths_in_time(arm_paths, time_step=0.05):
    """
    Parametrize the robot paths in time.
    :param arm_paths: List of paths for each robot.
    :param time_step: Duration of each step in the paths.
    :return: A list of time-parameterized paths.
    """
    time_parametrized_paths = []

    # Calculate the total time for the longest path
    max_length = max(len(path) for path in arm_paths)
    total_time = max_length * time_step

    for path in arm_paths:
        time_parametrized_path = []
        current_time = 0.0

        for q in path:
            time_parametrized_path.append((current_time, q))
            current_time += time_step

        # If the path is shorter than the longest path, extend it with the last configuration
        while current_time < total_time:
            time_parametrized_path.append((current_time, path[-1]))
            current_time += time_step

        time_parametrized_paths.append(time_parametrized_path)

    return time_parametrized_paths

def get_priority_collision_fn(robot, static_obstacles, higher_priority_robots, attachments, self_collisions, disabled_collisions,
                              custom_limits, max_distance, use_aabb, cache, time_step=0.05):
    """
    Create a collision function considering both static and dynamic obstacles.
    :param dynamic_robot_paths: A list of time-parameterized paths for each robot.
    :param time_step: Duration of each step in the paths.
    """
    # Static collision function
    static_collision_fn = get_collision_fn(robot.body, robot.ik_joints, static_obstacles, attachments, self_collisions, disabled_collisions,
                                           custom_limits, use_aabb, cache, max_distance)
    dynamic_robot_paths = []
    for robot in higher_priority_robots:
        dynamic_robot_paths.append(higher_priority_robots.path)
    # Time-parameterized paths for the dynamic obstacles
    time_parametrized_paths = parametrize_paths_in_time(dynamic_robot_paths, time_step)

    def collision_fn(q, time, verbose=False):
        # Check against static obstacles
        if static_collision_fn(q):
            return True

        # Set the robot's joints to the query configuration
        set_joint_positions(robot.body, robot.ik_joints, q)

        # Check collision with each robot path at the current time
        for path in time_parametrized_paths:
            # Find the configuration of the dynamic robot at the given time
            other_robot_q = next((q for t, q in path if t >= time), path[-1][1])

            # TODO: Implement the logic to check collision with the other robot at this position
            # This could involve AABB overlap checks, distance computations, etc.
            if pairwise_collision((robot.body, frozenset(get_moving_links(robot.body, robot.ik_joints))), 
                                  (other_robot_q, frozenset(get_moving_links(other_robot_q, robot.ik_joints))), **kwargs):
                if verbose:
                    print(f"Collision at time {time} between {robot.body} and dynamic robot")
                return True

        return False

    return collision_fn

def plan_priority_motion(robots, end_confs, start_confs, obstacles=[], attachments=[],
                         self_collisions=True, disabled_collisions=set(),
                         weights=None, resolutions=None, max_distance=MAX_DISTANCE,
                         use_aabb=False, cache=True, custom_limits={}, **kwargs):
    """
    Plan motion for multiple robots with priority.
    Each robot's path is planned considering higher priority robots as moving obstacles and lower priority as static.
    """
    assert len(robots) == len(end_confs) == len(start_confs), "Mismatch in numbers of robots, start, and end configurations"

    paths = []
    static_obstacles = obstacles.copy()  # Starting with the provided environmental obstacles
    static_obstacles.extend([robot.body for robot in robots])  # All robots initially considered as static obstacles

    for i, (robot, start_conf, end_conf) in enumerate(zip(robots, start_confs, end_confs)):
        assert len(robot.ik_joints) == len(end_conf) == len(start_conf), "Mismatch in number of IK joints and end configuration length for a robot"

        # Update weights if necessary
        if weights is None and resolutions is not None:
            weights = np.reciprocal(resolutions)

        # Update obstacle list: higher priority robots are moving, lower priority are static
        dynamic_obstacles = paths[:i]  # Paths of higher priority robots
        current_obstacles = static_obstacles[i+1:]  # Exclude the current and higher priority robots from static obstacles

        # Define collision function considering dynamic obstacles
        collision_fn = get_priority_collision_fn(robot.body, robot.ik_joints, current_obstacles, paths,
                                                 attachments, self_collisions, disabled_collisions,
                                                 custom_limits, max_distance, use_aabb, cache)

        # Define other functions for the current robot
        distance_fn = get_distance_fn(robot.body, robot.ik_joints, weights=weights)
        sample_fn = get_sample_fn(robot.body, robot.ik_joints, custom_limits=custom_limits)
        extend_fn = get_extend_fn(robot.body, robot.ik_joints, resolutions=resolutions)

        # Plan path for the current robot
        path = plan_robot_motion(robot, start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)
        if path is None:
            return None  # Planning failed for the current robot

        paths.append(path)  # Store the successfully planned path

    return paths

def plan_robot_motion(robot, start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs):
    """
    Plan motion for a single robot using Bi-directional RRT or other specified algorithm.
    """
    # Example using Bi-directional RRT
    from motion.motion_planners.rrt_connect import birrt
    return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)


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
        start_confs = [np.array(get_joint_positions(robot.body, robot.ik_joints)) for robot in robots]

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

    # info
    plane = p.loadURDF("plane.urdf")
    obstacles = [] # TODO: collisions with the ground
    info = PANDA_INFO
    arm_standby_pose = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    arm_neutral_pose = (0, 0, 0, -1.51, 0, 1.877, 0, 0.04, 0.04)
    arm1_start = (1, 0.785, -1, -1.51, 0, 1.877, 0, 0.04, 0.04)
    arm1_goal = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    arm2_start = (0, 0, 0, -1.51, 0, 1.877, 0, 0.04, 0.04)
    arm2_goal = (0, 0, 0, 0, 0, 0, 0, 0, 0)


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

    draw_pose(Pose(), parent=robot1.body, parent_link=robot1.tool_link)
    draw_pose(Pose(), parent=robot2.body, parent_link=robot2.tool_link)
    # joints1 = get_movable_joints(robot1.body)
    # joints2 = get_movable_joints(robot2.body)
    print('Joints1', robot1.movable_joints)
    print('IK_joints1', robot1.ik_joints)
    print('Joints2', robot2.movable_joints)
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

    # wait_if_gui('Set robot 1 in so-called neutral pose?')
    # robot1.set_pose(robot1.movable_joints, arm_neutral_pose)


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
    print("Robots:", robots)
    goal_confs = [arm1_goal, arm2_goal]
    # goal_confs is initially a list of lists
    goal_confs = [np.array(conf) for conf in goal_confs]
    print("goal_confs:", goal_confs)
    start_confs = [arm1_start, arm2_start]
    # # start_confs is initially a list of lists
    start_confs = [np.array(conf) for conf in start_confs]
    # start_confs = None
    print("start_confs:", start_confs)
    arm_paths = plan_priority_motion(robots, goal_confs, start_confs, obstacles=[])
    # if arm_path1 is not None:
    #     execute_motion(robot1, ik_joints1, arm_path1)
    if arm_paths is not None:
        execute_multi_robot_motion(robots, arm_paths)
    
    wait_if_gui('Disconnect?')
    disconnect()

if __name__ == '__main__':
    main()