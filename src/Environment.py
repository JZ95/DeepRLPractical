#!/usr/bin/env python3
# encoding utf-8

from hfo import *
import os
import time
from collections import deque
from reward_fun import REWARD_OPTS

HFO_PATH = os.environ['HFO_PATH']
if HFO_PATH is None:
    raise FileNotFoundError('please set environment variable HFO_PATH.')


class HFOEnv(object):
    def __init__(self, reward_opt,
                 config_dir=os.path.join(HFO_PATH, 'bin/teams/base/config/formations-dt'),
                 port=6000, server_addr='localhost', team_name='base_left', play_goalie=False,
                 numOpponents=0, numTeammates=0, seed=123):

        self.config_dir = config_dir
        self.port = port
        self.server_addr = server_addr
        self.team_name = team_name
        self.play_goalie = play_goalie

        self.curState = None
        self.possibleActions = ['MOVE', 'SHOOT', 'DRIBBLE', 'GO_TO_BALL']
        self.numOpponents = numOpponents
        self.numTeammates = numTeammates
        self.seed = seed
        self.startEnv()
        self.hfo = HFOEnvironment()
        self.reward_opt = reward_opt
        self.checkLst = deque([], 4)
        self.lastState = None

    # Method to initialize the server for HFO environment
    def startEnv(self):
        hfo_cmd = os.path.join(HFO_PATH, 'bin/HFO')
        if self.numTeammates == 0:
            os.system(hfo_cmd + " --port {} --seed {} --defense-npcs=0 --defense-agents={} --offense-agents=1 --trials 8000 --untouched-time 500 --frames-per-trial 500 --fullstate &".format(str(self.port), str(self.seed),
                                                                                                                                                                  str(self.numOpponents)))
        else:
            os.system(hfo_cmd + " --port {} --seed {} --defense-agents={} --defense-npcs=0 --offense-npcs={} --offense-agents=1 --trials 8000 --untouched-time 500 --frames-per-trial 500 --fullstate &".format(
                str(self.port), str(self.seed), str(self.numOpponents), str(self.numTeammates)))
        time.sleep(5)

    # Reset the episode and returns a new initial state for the next episode
    # You might also reset important values for reward calculations
    # in this function
    def reset(self):
        processedStatus = self.preprocessState(self.hfo.getState())
        self.curState = processedStatus

        return self.curState

    # Connect the custom weaker goalkeeper to the server and
    # establish agent's connection with HFO server
    def connectToServer(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        launch_gk_cmd = os.path.join(src_path, 'Goalkeeper.py')
        os.system(launch_gk_cmd + " --numEpisodes=8000 --port={} &".format(str(self.port)))
        time.sleep(2)
        self.hfo.connectToServer(LOW_LEVEL_FEATURE_SET, self.config_dir,
                                 self.port, self.server_addr, self.team_name, self.play_goalie)

    # This method computes the resulting status and states
    # after an agent decides to take an action
    def act(self, actionString):

        if actionString == 'MOVE':
            self.hfo.act(MOVE)
        elif actionString == 'SHOOT':
            self.hfo.act(SHOOT)
        elif actionString == 'DRIBBLE':
            self.hfo.act(DRIBBLE)
        elif actionString == 'GO_TO_BALL':
            self.hfo.act(GO_TO_BALL)
        else:
            raise Exception('INVALID ACTION!')

        status = self.hfo.step()
        currentState = self.hfo.getState()
        processedStatus = self.preprocessState(currentState)
        self.lastState = self.curState
        self.curState = processedStatus

        return status, self.curState

    # Define the rewards you use in this function
    # You might also give extra information on the name of each rewards
    # for monitoring purposes.

    def get_reward(self, status, nextState):
        return REWARD_OPTS[self.reward_opt](status, self.checkLst, nextState)

    # Method that serves as an interface between a script controlling the agent
    # and the environment. Method returns the nextState, reward, flag indicating
    # end of episode, and current status of the episode

    def step(self, action_params):
        status, nextState = self.act(action_params)
        done = (status != IN_GAME)
        reward, info = self.get_reward(status, nextState)
        return nextState, reward, done, status, info

    # This method enables agents to quit the game and the connection with the server
    # will be lost as a result
    def quitGame(self):
        self.hfo.act(QUIT)

    # Preprocess the state representation in this function
    def preprocessState(self, state):
        """as a baseline , we dont do any preprocess
        just return a 68-d vector when using low-level features
        """
        newState = np.reshape(state, (1, -1))
        oldState = self.lastState

        if oldState is not None:
            goal_dist_old = oldState[0][15]
            ball_dist_old = oldState[0][53]

            goal_dist = newState[0][15]
            ball_dist = newState[0][53]  # higher the value is, closer to the ball

            closer_to_goal = (goal_dist_old - goal_dist) < 0
            closer_to_ball = (ball_dist_old - ball_dist) < 0

            kickable = newState[0][12]

            item = {}
            item['closer2ball'] = closer_to_ball
            item['closer2goal'] = closer_to_goal
            item['dist2goal'] = goal_dist
            item['kickable'] = (kickable == 1)

            self.checkLst.append(item)

        return newState
