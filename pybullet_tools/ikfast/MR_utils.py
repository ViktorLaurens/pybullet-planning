import numpy as np


def get_sample_fn(body, joints, custom_limits={}, **kwargs):
    lower_limits, upper_limits = get_custom_limits(body, joints, custom_limits, circular_limits=CIRCULAR_LIMITS)
    generator = interval_generator(lower_limits, upper_limits, **kwargs)
    def fn():
        return tuple(next(generator))
    return fn

def get_distance_fn(body, joints, weights=None, norm=2):
    weights = get_default_weights(body, joints, weights)
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        if norm == 2:
            return np.sqrt(np.dot(weights, diff * diff))
        return np.linalg.norm(np.multiply(weights, diff), ord=norm)
    return fn

def get_extend_fn(body, joints, resolutions=None, norm=2):
    # norm = 1, 2, INF
    resolutions = get_default_resolutions(body, joints, resolutions)
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        #steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions))))
        steps = int(np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=norm))
        refine_fn = get_refine_fn(body, joints, num_steps=steps)
        return refine_fn(q1, q2)
    return fn

def get_collision_fn(body, joints, obstacles=[], attachments=[], self_collisions=True, disabled_collisions=set(),
                     custom_limits={}, use_aabb=False, cache=False, max_distance=MAX_DISTANCE, **kwargs):
    # TODO: convert most of these to keyword arguments
    check_link_pairs = get_self_link_pairs(body, joints, disabled_collisions) if self_collisions else []
    moving_links = frozenset(link for link in get_moving_links(body, joints)
                             if can_collide(body, link)) # TODO: propagate elsewhere
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [CollisionPair(body, moving_links)] + list(map(parse_body, attached_bodies))
    #moving_bodies = list(flatten(flatten_links(*pair) for pair in moving_bodies)) # Introduces overhead
    #moving_bodies = [body] + [attachment.child for attachment in attachments]
    get_obstacle_aabb = cached_fn(get_buffered_aabb, cache=cache, max_distance=max_distance/2., **kwargs)
    limits_fn = get_limits_fn(body, joints, custom_limits=custom_limits)
    # TODO: sort bodies by bounding box size
    # TODO: cluster together links that remain rigidly attached to reduce the number of checks

    def collision_fn(q, verbose=False):
        if limits_fn(q):
            return True
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()
        #wait_for_duration(1e-2)
        get_moving_aabb = cached_fn(get_buffered_aabb, cache=True, max_distance=max_distance/2., **kwargs)

        for link1, link2 in check_link_pairs:
            # Self-collisions should not have the max_distance parameter
            # TODO: self-collisions between body and attached_bodies (except for the link adjacent to the robot)
            if (not use_aabb or aabb_overlap(get_moving_aabb(body), get_moving_aabb(body))) and \
                    pairwise_link_collision(body, link1, body, link2): #, **kwargs):
                #print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                if verbose: print(body, link1, body, link2)
                return True

        # #step_simulation()
        # #update_scene()
        # for body1 in moving_bodies:
        #     overlapping_pairs = [(body2, link2) for body2, link2 in get_bodies_in_region(get_moving_aabb(body1))
        #                          if body2 in obstacles]
        #     overlapping_bodies = {body2 for body2, _ in overlapping_pairs}
        #     for body2 in overlapping_bodies:
        #         if pairwise_collision(body1, body2, **kwargs):
        #             #print(get_body_name(body1), get_body_name(body2))
        #             if verbose: print(body1, body2)
        #             return True
        # return False

        for body1, body2 in product(moving_bodies, obstacles):
            if (not use_aabb or aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))) \
                    and pairwise_collision(body1, body2, **kwargs):
                #print(get_body_name(body1), get_body_name(body2))
                if verbose: print(body1, body2)
                return True
        return False
    return collision_fn


def plan_joint_motion(body, joints, end_conf, obstacles=[], attachments=[],
                      self_collisions=True, disabled_collisions=set(),
                      weights=None, resolutions=None, max_distance=MAX_DISTANCE,
                      use_aabb=False, cache=True, custom_limits={}, algorithm=None, **kwargs):

    assert len(joints) == len(end_conf)
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance,
                                    use_aabb=use_aabb, cache=cache)
    
    start_conf = get_joint_positions(body, joints)
    if not check_initial_end(start_conf, end_conf, collision_fn):
        return None
    if algorithm is None:
        from motion_planners.rrt_connect import birrt
        return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)
    return solve(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn,
                 algorithm=algorithm, weights=weights, **kwargs)
    #return plan_lazy_prm(start_conf, end_conf, sample_fn, extend_fn, collision_fn)