import argparse
import os
import pickle
import numpy as np

IN_GAME, GOAL, CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME, SERVER_DOWN = list(range(6))


def read_data(logdir):
    train_data = []
    for file in os.listdir(logdir):
        filename = os.path.join(logdir, file)
        if os.path.isdir(filename):
            continue
        with open(filename, 'rb') as f:
            if 'eval' in filename:
                eval_data = [pickle.load(f)]
            else:
                train_data.append(pickle.load(f))
    return eval_data, train_data


def describe(data, name):
    tmp_status = []
    tmp_steps_to_ball = []
    tmp_n_steps_episode = []
    for i in range(len(data)):
        tmp_status.append(data[i]['status_lst'][-100:])
        tmp_steps_to_ball.append(data[i]['steps_to_ball'][-100:])
        tmp_n_steps_episode.append(data[i]['steps_in_episode'][-100:])

    print('=========== INFO : %s ===========' % name)
    print('%d threads in %s.' % (len(data), name))

    for title, FLAG in zip(['GOAL', 'CAPTURED_BY_DEFENSE','OUT_OF_BOUNDS', 'OUT_OF_TIME'],
                    [GOAL, CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME]):

        ratio_per_thread = np.mean(np.array(tmp_status) == FLAG, axis=1)
        std_ = np.std(ratio_per_thread)
        mean_ = np.mean(ratio_per_thread)
        print('%-25s:  %.3f ± %.3f' % (title, mean_, std_))

    print('-' * 45)
    for title, lst in zip(['AVG STEPS TO BALL', 'AVG STEPS TO FINISH'], [tmp_steps_to_ball, tmp_n_steps_episode]):
        mean_per_thread = np.mean(np.array(lst), axis=1)
        std_ = np.std(mean_per_thread)
        mean_ = np.mean(mean_per_thread)
        print('%-25s:  %.3f ± %.3f' % (title, mean_, std_))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str)
    args = parser.parse_args()

    eval_data, train_data = read_data(args.log_dir)
    print('STATS FOR EVAL:')
    describe(eval_data, os.path.split(args.log_dir)[-1])

    print('\n')
    print('STATS FOR TRAIN:')
    describe(train_data, os.path.split(args.log_dir)[-1])


if __name__ == '__main__':
    main()
