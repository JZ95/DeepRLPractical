#!/usr/bin/env python3
# encoding utf-8
import torch.multiprocessing as mp
import torch
import argparse
import os
from Networks import ValueNetwork
from Worker import train, evaluation
from SharedAdam import SharedAdam
# Use this script to handle arguments and
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--reward-opt', type=str, default='baseline')
    parser.add_argument('--ckpt-interval', type=int, default=1000000)

    parser.add_argument('--t-max', type=float, default=32e6)
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--i-async_update', type=int, default=250)
    parser.add_argument('--i-target', type=int, default=2000)
    parser.add_argument('--discountFactor', type=float, default=0.99)

    parser.add_argument('--eps-end', type=float, default=0.02)
    parser.add_argument('--eps-start', type=float, default=0.99)
    parser.add_argument('--eps-decay', type=float, default=1e6)

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

        processes = []
        for idx in range(0, num_processes):
            trainingArgs = (idx, args, valueNetwork, targetNetwork, optimizer, lock, counter)
            p = mp.Process(target=train, args=(idx, args, valueNetwork, targetNetwork, optimizer, lock, counter))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    elif args.mode == 'eval':
        ckpt_path = os.path.join(args.log_dir, 'ckpt')
        lastest_ckpt = sorted(os.listdir(ckpt_path))[-1]
        print('loading ckpt %s' % lastest_ckpt)

        valueNetwork = ValueNetwork()
        if args.use_gpu:
            valueNetwork = valueNetwork.cuda()

        valueNetwork.load_state_dict(torch.load(os.path.join(ckpt_path, lastest_ckpt)))
        valueNetwork.eval()
        evaluation(args, valueNetwork)
