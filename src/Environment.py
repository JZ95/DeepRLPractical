#!/usr/bin/env python3
# encoding utf-8

from hfo import *
from copy import copy, deepcopy
import math
import random
import os
import time
HFO_PATH = '/Users/j.zhou/coursework_rl/HFO'

class HFOEnv(object):
    def __init__(self, config_dir=HFO_PATH + '/bin/teams/base/config/formations-dt',
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

    # Method to initialize the server for HFO environment
    def startEnv(self):
        if self.numTeammates == 0:
            os.system(HFO_PATH + "/bin/HFO --headless --seed {} --defense-npcs=0 --defense-agents={} --offense-agents=1 --trials 8000 --untouched-time 500 --frames-per-trial 500 --port {} --fullstate &".format(str(self.seed),
                                                                                                                                                                                                      str(self.numOpponents), str(self.port)))
        else:
            os.system(HFO_PATH + "/bin/HFO --headless --seed {} --defense-agents={} --defense-npcs=0 --offense-npcs={} --offense-agents=1 --trials 8000 --untouched-time 500 --frames-per-trial 500 --port {} --fullstate &".format(
                str(self.seed), str(self.numOpponents), str(self.numTeammates), str(self.port)))
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
        os.system(
            "./Goalkeeper.py --numEpisodes=8000 --port={} &".format(str(self.port)))
        time.sleep(2)
        self.hfo.connectToServer(LOW_LEVEL_FEATURE_SET, self.config_dir,
                                 self.port, self.server_addr, self.team_name, self.play_goalie)

    # This method computes the resulting status and states after an agent decides to take an action
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
        ########################
        ### define your reward #
        ########################
        info = {}

        ball_dist_old = self.lastState[0][53]
        ball_dist = nextState[0][53]  # higher the value is, closer to the ball
        closer_to_ball = (ball_dist_old - ball_dist) < 0

        kickable = nextState[0][12]

        reward = 0

        # if status == GOAL:
        #     reward += 1
        # else:
        if closer_to_ball:
            reward += 1
            pass

        elif kickable == 1.0:
            if 'kickable' not in info:
                info['kickable'] = True

        elif kickable == -1.0:
            pass

        return reward, info

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
        ###############
        ### insert your code
        ###############
        """as a baseline , we dont do any preprocess
        just return a 68-d vector when using low-level features
        """
        return np.reshape(state, (1, -1))
