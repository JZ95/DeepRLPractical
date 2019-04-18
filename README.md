<head>
  <script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML" async>
</script>
</head>

# DeepRLPractical
An Implementation for Asynchronous Q-Learning on HFO game. This is a coursework for [Reinforcement Learning](https://www.inf.ed.ac.uk/teaching/courses/rl/) in UoE.

## Environment Setup for this Toy Project
### Optition 1. Step by Step on your local machine
#### 1.1 install HFO
follow this [link](https://github.com/LARG/HFO) for install HFO experiment environment.
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

## Eps-Greedy Policy

<p>
    We use \(\epsilon \)-greedy for policy improvement, and the \(\epsilon \) decreases exponentially w.r.t. the training steps as follows: $$\epsilon = \epsilon_{end} + (\epsilon_{start} - \epsilon_{end})$$
    where \(t \) is the training step and \(d \) is the decay rate.
</p>

------------
## Reference:
[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
