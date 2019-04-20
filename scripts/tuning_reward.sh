#!/bin/sh
export CODE_PATH=/home/workspace/src
export LOG_DIR=/home/workspace/logs

python $CODE_PATH/main.py \
   --use-gpu \
   --log-dir $LOG_DIR/'baseline' \
   --t-max 1002200 \
   --n-jobs 2 \
   --eps-decay 40000 \
   --ckpt-interval 50000

python $CODE_PATH/main.py \
    --log-dir $LOG_DIR/'baseline' \
    --t-max 100012 \
    --mode eval
