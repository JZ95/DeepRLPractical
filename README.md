# DeepRLPractical
An Implementation for Asynchronous Q-Learning on [HFO](https://github.com/LARG/HFO) game. This is a coursework for [Reinforcement Learning](https://www.inf.ed.ac.uk/teaching/courses/rl/) in UoE.

## Task Description
The goal of this task is to let the agent (footplayer) to score as much as possible when defended by a goalkeeper (a part of the enironment). During each episode, the agent is initialized at a random place in the field, and it has to move to the ball (also randomly initialized), and then dribble towards the penalty area and kick the ball to score. The episode may end up with 4 possible status, which are `GOAL`, `OUT_OF_BOUND`, `CAPTURED_BY_GOALKEEPER`, `OUT_OF_TIME` (agent does not shoot with 500 timesteps) respectively.

## Environment Setup for this Toy Project
### Option 1. Step by Step on your local machine
#### 1.1 install HFO
follow this [link](https://github.com/LARG/HFO) for installing HFO experiment environment.
#### 1.2 clone this repo
```
git clone https://github.com/JZ95/DeepRLPractical
```
#### 1.3 set environment variable
```
export HFO_PATH=/path/to/your/HFO/dir
```
#### 1.4 run
```
cd DeepRLPractical
python main.py --help  # see the help info
python main.py  # run by default setting
```

### Option 2. Use Docker (recommended)
#### 2.1 build image
```
# use cpu
docker build -f dockerfile.cpu -t deep_rl_env .
# or use gpu
docker build -f dockerfile.gpu deep_rl_env .
```
#### 2.2 create a volume to serialize data
```
docker volume create rl-vol
```
#### 2.3 run container
```
# cpu version
docker run --rm -it -v rl-vol:/home/workspace/logs deep_rl_env
# rember to use nvidia docker runtime if you wanna use gpu version
docker run --runtime=nvidia --rm -it -v rl-vol:/home/workspace/logs deep_rl_env
```

## Method
We implement an Asynchronous Q-learning Agent for this task. We use a two-layer neural network (see `src/Networks.py`) to do Q-value approximation, where the state is representated by feature vector produced by HFO environment (contains the location of the agent, the ball and the goalkeeper and other information, see [HFO manual](https://github.com/LARG/HFO/blob/master/doc/manual.pdf) for more info). The general idea of Asynchronous Q-learning is to use multi-threads to 
increase the diversity of exploration, and make online updates in parallel less correlated in time. Furthermore, under the same number of training episodes, with more parallel threads to use, the agent is able to explore more and learn a more accurate estimate for the Q-value, thus is expected to achieve a better performance. See the [paper](https://arxiv.org/pdf/1602.01783.pdf) for detailed explaintion.

### Eps-Greedy Policy
We use <img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\epsilon" title="\large \epsilon" />-greedy for policy improvement, and the <img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;\epsilon" title="\large \epsilon" /> decreases exponentially w.r.t. the training steps as follows:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\epsilon = \epsilon_{end} + (\epsilon_{start} - \epsilon_{end}) \times exp(- \frac{t}{d})" title="\Large \epsilon = \epsilon_{end} + (\epsilon_{start} - \epsilon_{end}) \times exp(- \frac{t}{d})" />

where <img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;t" title="\large t" /> is the training step and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\large&space;d" title="\large d" /> is the decay rate.

### Design on reward
Different rewards are designed in this work as follows:

- GOAL: give the agent reward `Rg` if it scores.
- hold the ball: give a reward `Rb` if the agent transits from not having the ball to having the ball.
- goal distance: give a reward `Rd` on the distance between the agent and the goal, which encourages the agent to approach the goal

We believe rewarding on intermediate sub-goals (e.g. approaching the ball, or the goal) would provide a better guide to the agent than just telling it to score as much as possbile (i.e. only rewarding on GOAL). That is why we disign 3 reward types in this task. 

Different combinations of the given components would be explored as follows:

- Rg -- baseline
- Rg + Rb
- Rg + Rd
- Rg + Rb + Rd

## Results
### Effect of reward

------------
## Reference:
[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)