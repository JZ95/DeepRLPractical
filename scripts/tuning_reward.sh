export HFO_PATH=/Users/j.zhou/coursework_rl/HFO
export CODE_PATH=/Users/j.zhou/DeepRLPractical/src
export LOG_DIR=/Users/j.zhou/DeepRLPractical/logs

python $CODE_PATH/main.py \
   --log-dir $LOG_DIR/'test' \
   --t-max 10000 \
   --n-jobs 2 \
   --eps-decay 1500000 \
   --ckpt-interval 1000

# python $CODE_PATH/main.py \
#     --log-dir $LOG_DIR/cloud_data/closer2ball-0.1-final \
#     --t-max 100012 \
#     --mode eval
