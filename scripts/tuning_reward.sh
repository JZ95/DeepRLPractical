export CODE_PATH=/home/workspce/src
export LOG_DIR=/home/workspce/logs

python $CODE_PATH/main.py \
   --use-cuda \
   --log-dir $LOG_DIR/'test' \
   --t-max 32000012 \
   --n-jobs 1 \
   --eps-decay 1500000 \
   --ckpt-interval 1000000

# python $CODE_PATH/main.py \
#     --log-dir $LOG_DIR/cloud_data/closer2ball-0.1-final \
#     --t-max 100012 \
#     --mode eval
