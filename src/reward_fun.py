import math
from hfo import *
from functools import partial


def baseline(status, checkLst, newState):
    """ Baseline reward function, only assign 1 on GOAL
    """
    info = {}
    reward = 0

    kickable = newState[0][12]
    if 'kickable' not in info and kickable == 1:
        info['kickable'] = True

    if status == GOAL:
        reward += 1

    return reward, info


def penalty_oob(status, checkLst, newState):
    """ Baseline + penalty on out of bound
    """
    info = {}
    reward = 0

    kickable = newState[0][12]
    if 'kickable' not in info and kickable == 1:
        info['kickable'] = True

    if status == GOAL:
        reward += 1

    elif status == OUT_OF_BOUNDS:
        reward -= 0.5

    return reward, info


def closer2ball_template_v1(status, checkLst, newState, r):
    """ baseline (1 for GOAL) + r for closer to ball -- 1 step
    """
    info = {}
    reward = 0

    kickable = newState[0][12]
    if 'kickable' not in info and kickable == 1:
        info['kickable'] = True

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if len(checkLst) > 0:
            item = checkLst[-1]
            if not item['kickable'] and item['closer2ball']:
                reward += r

    return reward, info


def closer2ball_template_v2(status, checkLst, newState, r):
    """ baseline (1 for GOAL) + r for closer to ball -- 4 steps
    """
    info = {}
    reward = 0

    kickable = newState[0][12]
    if 'kickable' not in info and kickable == 1:
        info['kickable'] = True

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if len(checkLst) == 4:
            reward_close_to_ball = True
            for item in checkLst:
                if not item['kickable'] and item['closer2ball']:
                    pass
                else:
                    reward_close_to_ball = False
                    break

            if reward_close_to_ball:
                reward += r

    return reward, info

def closer2ball_n_closer2goal_template_v1(status, checkLst, newState, r_ball, r_goal, goal_dist_lim):
    """ baseline (1 for GOAL)
    + r_ball for closer to ball -- 1 - step
    + r_goal for closer to goal && dist_to_goal < 0.75 -- 4 - step
    """
    info = {}
    reward = 0

    kickable = newState[0][12]
    if 'kickable' not in info and kickable == 1:
        info['kickable'] = True

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if len(checkLst) > 0:
            item = checkLst[-1]
            assert 'kickable' in item and 'closer2ball' in item
            if not item['kickable'] and item['closer2ball']:
                reward += r_ball

        if len(checkLst) == 4:
            reward_close_to_goal = True

            for item in checkLst:
                if item['kickable'] and item['closer2goal']:
                    pass
                else:
                    reward_close_to_goal = False
                    break

            if reward_close_to_goal and checkLst[-1]['dist2goal'] > goal_dist_lim:
                reward += r_goal

    return reward, info

def closer2ball_n_closer2goal_template_v2(status, checkLst, newState, r_ball, r_goal, goal_dist_lim):
    """ baseline (1 for GOAL) + r for closer to ball
    """
    info = {}
    reward = 0

    kickable = newState[0][12]
    if 'kickable' not in info and kickable == 1:
        info['kickable'] = True

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if len(checkLst) == 4:
            reward_close_to_goal = True
            reward_close_to_ball = True

            for item in checkLst:
                if not item['kickable'] and item['closer2ball']:
                    pass
                else:
                    reward_close_to_ball = False
                    break

            for item in checkLst:
                if item['kickable'] and item['closer2goal']:
                    pass
                else:
                    reward_close_to_goal = False
                    break

            if reward_close_to_ball:
                reward += r_ball

            if reward_close_to_goal and checkLst[-1]['dist2goal'] > goal_dist_lim:
                reward += r_goal

    return reward, info




REWARD_OPTS = {'baseline': baseline,
               'closer2ball-0.25-v1' : partial(closer2ball_template_v1, r=0.25),
               'closer2ball-0.25-v2': partial(closer2ball_template_v2, r=0.25),
               'closer2ball-0.25-closer2-goal-0.25-goal-dist-lim-0.75-v1': partial(closer2ball_n_closer2goal_template_v1, r_ball=0.25, r_goal=0.25, goal_dist_lim=0.75),
               'closer2ball-0.25-closer2-goal-0.25-goal-dist-lim-0.75-v2': partial(closer2ball_n_closer2goal_template_v2, r_ball=0.25, r_goal=0.25, goal_dist_lim=0.75),
               }