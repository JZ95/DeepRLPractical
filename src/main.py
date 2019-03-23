#!/usr/bin/env python3
# encoding utf-8
import torch.multiprocessing as mp
import argparse
import os
from Networks import ValueNetwork
from Worker import train
from SharedAdam import SharedAdam
# Use this script to handle arguments and
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--port', type=int, default=6000)
    # parser.add_argument('--logfile', type=str, default='')

    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--ckpt-interval', type=int, default=1000000)

    parser.add_argument('--t-max', type=int, default=200000000)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--i-async_update', type=int, default=150)
    parser.add_argument('--i-target', type=int, default=3000)
    parser.add_argument('--discountFactor', type=float, default=0.99)

    parser.add_argument('--eps-end', type=float, default=0.02)
    parser.add_argument('--eps-start', type=float, default=0.95)
    parser.add_argument('--eps-decay', type=float, default=50000.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Example on how to initialize global locks for processes
    # and counters.
    args = get_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    num_processes = args.n_jobs
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    valueNetwork = ValueNetwork()
    targetNetwork = ValueNetwork()

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
