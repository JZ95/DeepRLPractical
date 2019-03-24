import argparse
import os
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str)
    args = parser.parse_args()

    data = []
    for file in os.listdir(args.log_dir):
        filename = os.path.join(args.log_dir, file)
        with open(filename, 'rb') as f:
            data.append(pickle.load(f))


if __name__ == '__main__':
    main()
