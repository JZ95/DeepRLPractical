import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import math
import os
import pickle


def train(idx, args, valueNetwork, targetNetwork, optimizer, lock, counter):
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    port = 6006 + idx * 9
    seed = 2019 + idx * 46

    discountFactor = args.discountFactor
    hfoEnv = HFOEnv(args.reward_opt, numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()

    episodeNumber = 0
    numTakenActions = 0
    numTakenActionCKPT = 0

    totalWorkLoad = args.t_max // args.n_jobs
    state = torch.tensor(hfoEnv.reset()).to(device)  # get initial state

    steps_to_ball = []
    steps_in_episode = []
    status_lst = []

    log = {'steps_to_ball': steps_to_ball,
           'steps_in_episode': steps_in_episode,
           'status_lst': status_lst}

    cnt = None
    firstRecord = True
    optimizer.zero_grad()

    while True:
        # take action based on eps-greedy policy
        action = select_action(
            state, valueNetwork, episodeNumber, numTakenActions, args)  # action -> int

        act = hfoEnv.possibleActions[action]
        newObservation, reward, done, status, info = hfoEnv.step(act)

        if 'kickable' in info and firstRecord:
            cnt = numTakenActions - numTakenActionCKPT
            firstRecord = False

        next_state = torch.tensor(hfoEnv.preprocessState(newObservation)).to(device)

        target = computeTargets(
            reward, next_state, discountFactor, done, targetNetwork, device)  # target -> tensor
        pred = computePrediction(state, action, valueNetwork, device)  # pred -> tensor

        loss = F.mse_loss(pred, target)  # compute loss
        loss.backward()  # accumulate loss

        lock.acquire()
        counter.value += 1
        counterValue = counter.value
        lock.release()

        numTakenActions += 1

        if done:
            firstRecord = True
            status_lst.append(status)
            steps_in_episode.append(numTakenActions - numTakenActionCKPT)

            if cnt is None:
                steps_to_ball.append(numTakenActions - numTakenActionCKPT)
            else:
                steps_to_ball.append(cnt)
                cnt = None

            episodeNumber += 1
            numTakenActionCKPT = numTakenActions

            # hfoEnv alreay call 'preprocessState' in 'reset'
            state = torch.tensor(hfoEnv.reset()).to(device)

        if done or numTakenActions % args.i_async_update == 0:
            optimizer.step()  # apply grads
            optimizer.zero_grad()  # clear all cached grad

        if counterValue % args.i_target == 0:
            targetNetwork.load_state_dict(
                valueNetwork.state_dict())  # update target network

        if counterValue % args.ckpt_interval == 0:
            ckpt_path = os.path.join(args.log_dir, 'ckpt')
            if not os.path.exists(ckpt_path):
                os.mkdir(ckpt_path)

            filename = os.path.join(ckpt_path, 'params_%d' % (
                counterValue / args.ckpt_interval))

            lock.acquire()
            saveModelNetwork(valueNetwork, filename)
            lock.release()

        if numTakenActions > totalWorkLoad:
            filename = os.path.join(args.log_dir, 'log_worker_%d.pkl' % idx)
            saveLog(log, filename)
            return


def select_action(state, valueNetwork, episodeNumber, numTakenActions, args):
    """ Take action a with ε-greedy policy based on Q(s, a; θ)
    return: int
    """
    assert isinstance(state, torch.Tensor)

    sample = random.random()
    if args.mode == 'train':
        eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
            math.exp(-1. * numTakenActions / args.eps_decay)
    else:
        eps_threshold = args.eps_end

    if sample < eps_threshold:
        return random.randrange(4)
    else:
        action_values = valueNetwork(state)
        return torch.max(valueNetwork(state), 1)[-1].item()


def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork, device):
    """
    return y = r + \\gamma max_{a'} Q(s', a'; θ-) for non-terminal s'
    else y = r for terminal s'

    return: tensor
    """
    # nextObservation is a 2-D tensor
    assert len(nextObservation.shape) == 2

    if done:
        ret = torch.tensor(reward, dtype=torch.float32).to(device)
    else:
        action_values = targetNetwork(nextObservation)
        ret = torch.tensor(reward, dtype=torch.float32).to(device) + \
            discountFactor * torch.max(action_values, 1)[0].to(device)
    return ret


def computePrediction(state, action, valueNetwork, device):
    """ a wrapper for the forward method of ValueNetwork
    return: tensor
    """
    assert len(state.shape) == 2  # state is a 2-D tensor
    assert action in (0, 1, 2, 3)  # action is an int in [0, 1, 2, 3]

    action_values = valueNetwork(state)
    return action_values[:, action]


def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)


def saveLog(log_data, filename):
    """ save the log Data to the given filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(log_data, f)


def evaluation(args, valueNetwork):
    port = 6050
    seed = 2019

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hfoEnv = HFOEnv(args.reward_opt, numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfoEnv.connectToServer()

    episodeNumber = 0
    numTakenActions = 0
    numTakenActionCKPT = 0

    state = torch.tensor(hfoEnv.reset()).to(device)  # get initial state

    steps_to_ball = []
    steps_in_episode = []
    status_lst = []

    log = {'steps_to_ball': steps_to_ball,
           'steps_in_episode': steps_in_episode,
           'status_lst': status_lst}
    cnt = None
    firstRecord = True

    while numTakenActions < args.t_max:
        # take action based on eps-greedy policy
        action = select_action(
            state, valueNetwork, episodeNumber, numTakenActions, args)  # action -> int

        act = hfoEnv.possibleActions[action]
        newObservation, reward, done, status, info = hfoEnv.step(act)

        if 'kickable' in info and firstRecord:
            cnt = numTakenActions - numTakenActionCKPT
            firstRecord = False

        next_state = torch.tensor(hfoEnv.preprocessState(newObservation)).to(device)

        # all updates on the parameters shall acqure lock
        numTakenActions += 1

        if done:
            firstRecord = True
            status_lst.append(status)
            steps_in_episode.append(numTakenActions - numTakenActionCKPT)

            if cnt is None:
                steps_to_ball.append(numTakenActions - numTakenActionCKPT)
            else:
                steps_to_ball.append(cnt)
                cnt = None

            episodeNumber += 1
            numTakenActionCKPT = numTakenActions

            # hfoEnv alreay call 'preprocessState' in 'reset'
            state = torch.tensor(hfoEnv.reset()).to(device)

    filename = os.path.join(args.log_dir, 'log_eval.pkl')
    saveLog(log, filename)