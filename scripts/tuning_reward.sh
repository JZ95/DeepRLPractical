export CODE_PATH=/home/workspace/src
export LOG_DIR=/home/workspace/logs

python $CODE_PATH/main.py \
   --log-dir $LOG_DIR/'baseline' \
   --t-max 1000200 \
   --n-jobs 8 \
   --eps-decay 1500000 \
   --ckpt-interval 250000

python $CODE_PATH/main.py \
    --log-dir $LOG_DIR/'baseline' \
    --t-max 100012 \
    --mode eval
