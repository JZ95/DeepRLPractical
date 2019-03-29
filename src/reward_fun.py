import math
from hfo import *
from functools import partial


def baseline(status, oldState, newState):
    """ Baseline reward function, only assign 1 on GOAL
    """
    info = {}
    reward = 0

    kickable = newState[0][12]
    if 'kickable' not in info and kickable == 1:
        info['kickable'] = True

    if status == GOAL:
        print('GOAL')
        reward = 1

    return reward, info


def baseline_penalty(status, oldState, newState):
    """ Baseline reward function, only assign 1 on GOAL
    """
    info = {}
    reward = 0

    kickable = newState[0][12]
    if 'kickable' not in info and kickable == 1:
        info['kickable'] = True

    if status == GOAL:
        reward = 1

    elif status in (OUT_OF_BOUNDS, OUT_OF_TIME):
        reward = -1

    return reward, info


def closer2ball(status, oldState, newState, r):
    """ baseline (1 for GOAL) + r for closer to ball -- 1 step
    """
    info = {}
    reward = 0

    ball_dist_old = oldState[0][53]
    ball_dist = newState[0][53]  # higher the value is, closer to the ball

    closer_to_ball = (ball_dist_old - ball_dist) < 0

    kickable = newState[0][12]

    if 'kickable' not in info and kickable == 1:
        info['kickable'] = True

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if kickable != 1 and closer_to_ball:
            reward += r

    return reward, info


def closer2ball_n_closer2goal(status, oldState, newState, r_ball, r_goal, goal_dist_lim):
    """ baseline (1 for GOAL)
    + r_ball for closer to ball -- 1 - step
    + r_goal for closer to goal && dist_to_goal < 0.75 -- 4 - step
    """
    info = {}
    reward = 0

    goal_dist_old = oldState[0][15]
    ball_dist_old = oldState[0][53]

    goal_dist = newState[0][15]
    ball_dist = newState[0][53]  # higher the value is, closer to the ball

    closer_to_goal = (goal_dist_old - goal_dist) < 0
    closer_to_ball = (ball_dist_old - ball_dist) < 0

    kickable = newState[0][12]

    if 'kickable' not in info and kickable == 1:
        info['kickable'] = True

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if kickable != 1 and closer_to_ball:
            reward += r_ball

        if kickable == 1 and closer_to_goal and goal_dist < goal_dist_lim:
            reward += r_goal

    return reward, info


REWARD_OPTS = {'baseline': baseline,
               'baseline-penalty': baseline_penalty,
               'closer2ball-0.1': partial(closer2ball, r=0.1),
               'closer2ball-0.1-closer2goal-0.1-goal-dist-lim-0.75': partial(closer2ball_n_closer2goal, r_ball=0.1, r_goal=0.1, goal_dist_lim=0.75),
               }