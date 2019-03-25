import math
from hfo import *
from functools import partial


def baseline(status, oldState, newState):
    """ Baseline reward function, only assign 1 on GOAL
    """
    info = {}

    kickable = newState[0][12]
    reward = 0

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if kickable == 1.0:
            if 'kickable' not in info:
                info['kickable'] = True

        elif kickable == -1.0:
            pass

    return reward, info


def closer2ball_template(status, oldState, newState, r):
    """ baseline (1 for GOAL) + r for closer to ball
    """
    info = {}

    ball_dist_old = oldState[0][53]
    ball_dist = newState[0][53]  # higher the value is, closer to the ball
    closer_to_ball = (ball_dist_old - ball_dist) < 0

    kickable = newState[0][12]

    reward = 0

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if kickable == -1.0 and closer_to_ball:
            reward += r

        if kickable == 1.0:
            if 'kickable' not in info:
                info['kickable'] = True

        elif kickable == -1.0:
            pass

    return reward, info

def closer2ball_n_closer2goal_template(status, oldState, newState, r_ball, r_goal):
    """ baseline (1 for GOAL) + 0.5 for closer to ball +
    0.5 for closer to goal && dist_to_goal < 0.75
    """
    info = {}

    goal_dist_old = oldState[0][15]
    ball_dist_old = oldState[0][53]

    goal_dist = newState[0][15]
    ball_dist = newState[0][53]  # higher the value is, closer to the ball

    closer_to_goal = (goal_dist_old - goal_dist) < 0
    closer_to_ball = (ball_dist_old - ball_dist) < 0

    kickable = newState[0][12]

    reward = 0

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if kickable == -1.0 and closer_to_ball:
            reward += r_ball

        if kickable == 1.0:
            if 'kickable' not in info:
                info['kickable'] = True

            if closer_to_goal and goal_dist < 0.75:
                reward += r_goal

    return reward, info


def get_reward(status, oldState, newState):
    info = {}

    goal_dist_old = oldState[0][15]
    ball_dist_old = oldState[0][53]

    goal_dist = newState[0][15]
    ball_dist = newState[0][53]  # higher the value is, closer to the ball

    a, b = newState[0][5: 7]
    a_top, b_top = newState[0][16:18]
    a_bot, b_bot = newState[0][19:21]

    theta = math.atan2(*newState[0][5: 7])
    ball_vel_angle = math.atan2(*newState[0][56: 58])
    theta_top = math.atan2(*newState[0][16:18])
    theta_bot = math.atan2(*newState[0][19:21])

    closer_to_goal = (goal_dist_old - goal_dist) < 0
    closer_to_ball = (ball_dist_old - ball_dist) < 0

    kickable = newState[0][12]

    reward = 0

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if closer_to_ball:
            reward += 0.5

        if kickable == 1.0:
            if 'kickable' not in info:
                info['kickable'] = True

            if closer_to_goal and goal_dist < 0.75:
                reward += 0.25

            if ball_vel_angle > theta_top * 0.95 and ball_vel_angle < theta_bot * 0.95:
                reward += 0.25

        elif kickable == -1.0:
            pass

    return reward, info

REWARD_OPTS = {'baseline': baseline,
               'baseline-closer2ball-0.5': partial(closer2ball_template, r=0.5),
               'baseline-closer2ball-0.25': partial(closer2ball_template, r=0.25),
               'baseline-closer2ball-0.1': partial(closer2ball_template, r=0.1)}