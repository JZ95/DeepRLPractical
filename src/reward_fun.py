from hfo import *
from functools import partial


def baseline(status, oldState, newState):
    """ Baseline reward function, only assign 1 on GOAL
    """
    info = {}
    reward = 0

    kickable = newState[0][12]
    info['kickable'] = True if kickable == 1 else False

    if status == GOAL:
        reward = 1

    return reward, info


def getball(status, oldState, newState, r):
    """ baseline (1 for GOAL) + r for get the ball
    """
    info = {}
    reward = 0

    kickable_old = oldState[0][12]
    kickable = newState[0][12]

    info['kickable'] = True if kickable == 1 else False

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if kickable == 1 and kickable_old == -1:
            reward += r

    return reward, info


def goal_dist(status, oldState, newState, r):
    """ baseline (1 for GOAL)
    + r  reward on the distance to goal
    """
    info = {}
    reward = 0

    goal_dist = newState[0][15]
    kickable = newState[0][12]

    info['kickable'] = True if kickable == 1 else False

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        # goal dist only affect when agent has the ball
        if kickable == 1:
            reward += goal_dist * r_d

    return reward, info


def ball_goal_dist(status, oldState, newState, r_b, r_d):
    """ baseline (1 for GOAL)
    + r_b  reward on getting the ball
    + r_d  reward on the distance to goal
    """
    info = {}
    reward = 0

    goal_dist = newState[0][15]

    kickable_old = oldState[0][12]
    kickable = newState[0][12]

    info['kickable'] = True if kickable == 1 else False

    if status == GOAL:
        reward += 1

    elif status == IN_GAME:
        if kickable == 1 and kickable_old == -1:
            reward += r_b

        # goal dist only affect when agent has the ball
        if kickable == 1:
            reward += goal_dist * r_d

    return reward, info


REWARD_OPTS = {'baseline': baseline,
               'ball-0.1': partial(getball, r=0.1),
               'goal-dist-0.1': partial(goal_dist, r=0.1),
               'ball-0.1-goal-dist-0.1': partial(ball_goal_dist, r_b=0.1, r_d=0.1),
               }