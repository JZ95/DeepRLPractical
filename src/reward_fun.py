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


def closer2ball_template_v2(status, checkLst, newState, r):
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


def closer2ball_template(status, oldStateLst, newState, r):
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


def closer2ball_n_closer2goal_template(status, oldStateList, newState, r_ball, r_goal, goal_dist_lim):
    """ baseline (1 for GOAL)
    + r_ball for closer to ball
    + r_goal for closer to goal && dist_to_goal < 0.75
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

            if closer_to_goal and goal_dist < goal_dist_lim:
                reward += r_goal

    return reward, info


def closer2ball_n_angle_template(status, oldState, newState, r_ball, r_angle):
    """ baseline (1 for GOAL)
    + r_ball for closer to ball
    + r_goal for closer to goal && dist_to_goal < 0.75
    """
    info = {}

    goal_dist_old = oldState[0][15]
    ball_dist_old = oldState[0][53]

    goal_dist = newState[0][15]
    ball_dist = newState[0][53]  # higher the value is, closer to the ball

    # a, b = newState[0][5: 7]  # sin(theta), cos(theta), angle of agent
    # a_top, b_top = newState[0][16: 18]   # angle 
    # a_bot, b_bot = newState[0][19: 21]

    agent_angle = math.atan2(*newState[0][5: 7])  # angle of agent
    ball_vel_angle = math.atan2(*newState[0][56: 58]) # angle of ball velocity


    theta_top = math.atan2(*newState[0][16: 18])
    theta_bot = math.atan2(*newState[0][19: 21])

    # agent_angle - theta_top
    # agent_angle - theta_bot

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

            if ball_vel_angle < agent_angle - theta_top and ball_vel_angle > agent_angle - theta_bot:
                reward += r_angle

    return reward, info


def get_reward(status, oldState, newState):
    info = {}

    goal_dist_old = oldState[0][15]
    ball_dist_old = oldState[0][53]

    goal_dist = newState[0][15]
    ball_dist = newState[0][53]  # higher the value is, closer to the ball

    a, b = newState[0][5: 7]
    a_top, b_top = newState[0][16: 18]
    a_bot, b_bot = newState[0][19: 21]

    theta = math.atan2(*newState[0][5: 7])
    ball_vel_angle = math.atan2(*newState[0][56: 58])
    theta_top = math.atan2(*newState[0][16: 18])
    theta_bot = math.atan2(*newState[0][19: 21])

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

            # if closer_to_goal and goal_dist < 0.75:
            #     reward += 0.25

            if ball_vel_angle > theta_top * 0.95 and ball_vel_angle < theta_bot * 0.95:
                reward += 0.25

        elif kickable == -1.0:
            pass

    return reward, info

REWARD_OPTS = {'baseline': baseline,
               'closer2ball-0.25': partial(closer2ball_template_v2, r=0.25),
               'closer2ball-0.25-closer2-goal-0.25-goal-dist-lim-0.75': partial(closer2ball_n_closer2goal_template_v2, r_ball=0.25, r_goal=0.25, goal_dist_lim=0.75),
               }