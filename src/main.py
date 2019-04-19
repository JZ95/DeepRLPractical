#!/usr/bin/env python3
# encoding utf-8
import torch.multiprocessing as mp
import torch
import argparse
import os
from Networks import ValueNetwork
from Worker import runTrain, runEval
from SharedAdam import SharedAdam


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train',
                        help='run in train/eval mode')
    parser.add_argument('--use-gpu', action='store_true',
                        help='use gpu or not.')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='path for the experiment result.')
    parser.add_argument('--reward-opt', type=str, default='baseline',
                        help='reward option, see reward_fun.py')
    parser.add_argument('--ckpt-interval', type=int, default=1000000,
                        help='checkpoint saving intervals')
    parser.add_argument('--t-max', type=float, default=32e6,
                        help='training step upper-bound')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='number of threads')
    parser.add_argument('--i-async_update', type=int, default=500,
                        help='interval for updating value network.')
    parser.add_argument('--i-target', type=int, default=7500,
                        help='interval for updating target network.')
    parser.add_argument('--discountFactor', type=float, default=0.99,
                        help='discount factor, in the range of (0, 1].')
    parser.add_argument('--eps-end', type=float, default=0.02,
                        help='epsilon lower bound.')
    parser.add_argument('--eps-start', type=float, default=0.99,
                        help='epsilon upper bound.')
    parser.add_argument('--eps-decay', type=float, default=1e6,
                        help='epsilon decay rate.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Example on how to initialize global locks for processes
    # and counters.
    mp.set_start_method('spawn')
    args = get_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if args.mode == 'train':
        num_processes = args.n_jobs
        counter = mp.Value('i', 0)
        lock = mp.Lock()

        valueNetwork = ValueNetwork()
        targetNetwork = ValueNetwork()

        if args.use_gpu:
            targetNetwork = targetNetwork.cuda()
            valueNetwork = valueNetwork.cuda()

        targetNetwork.load_state_dict(valueNetwork.state_dict())
        targetNetwork.eval()

        valueNetwork.share_memory()
        targetNetwork.share_memory()

        # create Shared Adam optimizer
        optimizer = SharedAdam(valueNetwork.parameters())
        optimizer.share_memory()

        processes = []
        for idx in range(0, num_processes):
            trainingArgs = (idx, args, valueNetwork, targetNetwork, optimizer, lock, counter)
            p = mp.Process(target=runTrain, args=(idx, args, valueNetwork, targetNetwork, optimizer, lock, counter))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    elif args.mode == 'eval':
        ckpt_path = os.path.join(args.log_dir, 'ckpt')
        lastest_ckpt_id = sorted([int(file.split('_')[-1]) for file in os.listdir(ckpt_path)])[-1]
        lastest_ckpt = os.path.join(ckpt_path, 'params_%d' % lastest_ckpt_id)
        print('loading ckpt %s' % lastest_ckpt)

        map_loc = 'cpu'
        valueNetwork = ValueNetwork()
        if args.use_gpu:
            valueNetwork = valueNetwork.cuda()
            map_loc = 'gpu'

        valueNetwork.load_state_dict(torch.load(os.path.join(ckpt_path, lastest_ckpt), map_location=map_loc))
        valueNetwork.eval()
        runEval(args, valueNetwork)

    else:
        raise ValueError('unknown mode %s, mode shall be train/eval' % args.mode)
