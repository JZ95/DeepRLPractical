export CODE_PATH=/home/workspace/src
export LOG_DIR=/home/workspace/logs

python $CODE_PATH/main.py \
   --log-dir $LOG_DIR/'baseline' \
   --t-max 32002200 \
   --n-jobs 8 \
   --eps-decay 4000000 \
   --ckpt-interval 500000

python $CODE_PATH/main.py \
    --log-dir $LOG_DIR/'baseline' \
    --t-max 100012 \
    --mode eval
