# DeepRLPractical
An Implementation for Asynchronous Q-Learning on HFO game. This is a coursework for [Reinforcement Learning](https://www.inf.ed.ac.uk/teaching/courses/rl/) in UoE.

## Two Options to Run This Toy Project
### Optition 1. Setup environment on your own machine
#### 1.1 install HFO
follow this [link](https://github.com/LARG/HFO) for guidline.
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

### Optition 2. Use Docker (recommended)
#### 1.1 build image
```
# use cpu
docker build -f dockerfile.cpu -t deep_rl_env .
# use gpu
docker build -f dockerfile.gpu deep_rl_env .
```
#### 1.2 create a volume to serialize log data
```
docker volume create rl-vol
```
#### 1.3 run container
```
# cpu version
docker run --rm -it -v rl-vol:/home/workspace/logs deep_rl_env
# rember to use nvidia docker runtime if you wanna use gpu version
docker run --runtime=nvidia --rm -it -v rl-vol:/home/workspace/logs deep_rl_env
```
#### 1.4 enter container and do wtf you want

------------
## Reference:
[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
