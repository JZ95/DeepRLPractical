docker build -f dockerfile.gpu -t rl_env_gpu .
docker run --runtime=nvidia --rm -it -v my-vol:/home/workspace/logs rl_env_gpu
sudo ls /var/lib/docker/volumes/my-vol/_data/baseline