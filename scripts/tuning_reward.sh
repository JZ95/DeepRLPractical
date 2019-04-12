export CODE_PATH=/home/workspace/src
export LOG_DIR=/home/workspace/logs

python $CODE_PATH/main.py \
   --use-gpu \
   --log-dir $LOG_DIR/'baseline' \
   --t-max 32002200 \
   --n-jobs 32 \
   --eps-decay 500000 \
   --ckpt-interval 250000

python $CODE_PATH/main.py \
    --log-dir $LOG_DIR/'baseline-test' \
    --t-max 100012 \
    --mode eval
