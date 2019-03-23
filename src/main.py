#!/usr/bin/env python3
# encoding utf-8
import torch.multiprocessing as mp
import argparse
from Networks import ValueNetwork
from Worker import train
from SharedAdam import SharedAdam
# Use this script to handle arguments and
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--logfile', type=str, default='')

    parser.add_argument('--n-threads', type=int, default=1)
    parser.add_argument('--i-async_update', type=int, default=150)
    parser.add_argument('--i-target', type=int, default=3000)
    parser.add_argument('--discountFactor', type=float, default=0.99)
    parser.add_argument('--eps-end', type=float, default=0.02)
    parser.add_argument('--eps-start', type=float, default=0.95)
    parser.add_argument('--eps-decay', type=float, default=50000.0)
    parser.add_argument('--model-param-path', type=str, default='./')
    parser.add_argument('--ckpt-interval', type=int, default=1000000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Example on how to initialize global locks for processes
    # and counters.
    args = get_args()
    num_processes = args.n_threads
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    valueNetwork = ValueNetwork()
    targetNetwork = ValueNetwork()

    targetNetwork.load_state_dict(valueNetwork.state_dict())
    targetNetwork.eval()

    valueNetwork.share_memory()
    targetNetwork.share_memory()
    optimizer = SharedAdam(valueNetwork.parameters())

    processes = []
    for idx in range(0, num_processes):
        trainingArgs = (idx, args, valueNetwork, targetNetwork, optimizer, lock, counter)
        p = mp.Process(target=train, args=(idx, args, valueNetwork, targetNetwork, optimizer, lock, counter))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
