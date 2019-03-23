import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import math
import numpy as np
import os


def log(msg, logfile):
    with open(logfile, 'w') as f:
        f.write(msg)


def train(idx, args, valueNetwork, targetNetwork, optimizer, lock, counter):
    port = args.port
    # port = 8000 + idx * 5
    seed = 2019 + idx * 46

    discountFactor = args.discountFactor
    hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()

    episodeNumber = 0
    numTakenActions = 0
    numTakenActionCKPT = 0
    newEpisode = True

    state = torch.tensor(hfoEnv.reset())  # get initial state

    logs = {}
    steps_to_ball = []
    steps_in_episode = []
    status_lst = []
    cnt = None

    while True:
        # take action based on eps-greedy policy
        action = select_action(state, valueNetwork, episodeNumber, numTakenActions, args)  # action -> int
        act = hfoEnv.possibleActions[action]
        newObservation, reward, done, status, info = hfoEnv.step(act)

        if 'kickable' in info and newEpisode:
            cnt = numTakenActions - numTakenActionCKPT
            newEpisode = False

        next_state = torch.tensor(hfoEnv.preprocessState(newObservation))
        target = computeTargets(reward, next_state, discountFactor, done, targetNetwork)  # target -> tensor
        pred = computePrediction(state, action, valueNetwork)  # pred -> tensor

        loss = F.mse_loss(pred, target)  # compute loss
        loss.backward()  # accumulate loss

        # all updates on the parameters shall acqure lock
        lock.acquire()

        counter.value += 1
        numTakenActions += 1

        if done:
            status_lst.append(status)
            steps_in_episode.append(numTakenActions - numTakenActionCKPT)

            newEpisode = True
            if cnt is None:
                steps_to_ball.append(500)
            else:
                steps_to_ball.append(cnt)
                cnt = None

            if episodeNumber >= 500:
                s1 = 'HOW MANY STEPS TO BALL: % s' % steps_to_ball
                s2 = 'AVG STEPS TO BALL: % d' % (sum(steps_to_ball) / len(steps_to_ball))
                s3 = 'HOW MANY STEPS TO GOAL: % s' % steps_in_episode
                s4 = 'AVG STEPS TO GOAL: % d' % (sum(steps_in_episode) / len(steps_in_episode))
                s5 = 'STATUS: %s' % status_lst

                msg = '\n'.join([s1, s2, s3, s4, s5])
                print('============' * 2)
                print(msg)
                print('============' * 2)

                log(msg, args.logfile)
                exit()

            episodeNumber += 1
            numTakenActionCKPT = numTakenActions
            # hfoEnv alreay call 'preprocessState' in 'reset'
            state = torch.tensor(hfoEnv.reset())

        if done or numTakenActions % args.i_async_update == 0:
            optimizer.step()  # apply grads
            optimizer.zero_grad()  # clear all cached grad

        if counter.value % args.i_target == 0:
            targetNetwork.load_state_dict(valueNetwork.state_dict())  # update target network

        if counter.value % args.ckpt_interval == 0:
            filename = os.path.join(args.model_param_path, 'params_%d' % (counter.value / args.ckpt_interval))
            saveModelNetwork(valueNetwork, filename)

        # if numTakenActions > myWordLoad:
        #     lock.release()
        #     return

        lock.release()


def select_action(state, valueNetwork, episodeNumber, numTakenActions, args):
    """ Take action a with ε-greedy policy based on Q(s, a; θ)
    return: int
    """
    assert isinstance(state, torch.Tensor)

    sample = random.random()
    eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
        math.exp(-1. * numTakenActions / args.eps_decay)

    if sample < eps_threshold:
        return random.randrange(4)
    else:
        action_values = valueNetwork(state)
        return torch.max(valueNetwork(state), 1)[-1].item()


def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
    """
    return y = r + \\gamma max_{a'} Q(s', a'; θ-) for non-terminal s'
    else y = r for terminal s'

    return: tensor
    """
    # nextObservation is a 2-D tensor
    assert len(nextObservation.shape) == 2

    if done:
        ret = torch.tensor(reward, dtype=torch.float32)
    else:
        action_values = targetNetwork(nextObservation)
        ret = torch.tensor(reward, dtype=torch.float32) + discountFactor * torch.max(action_values, 1)[0]
    return ret


def computePrediction(state, action, valueNetwork):
    """ a wrapper for the forward method of ValueNetwork
    return: tensor
    """
    assert len(state.shape) == 2  # state is a 2-D tensor
    assert action in (0, 1, 2, 3)  # action is an int in [0, 1, 2, 3]

    action_values = valueNetwork(state)
    return action_values[:, action]


def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)
