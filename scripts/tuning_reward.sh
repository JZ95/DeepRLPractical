export HFO_PATH=/Users/j.zhou/coursework_rl/HFO
export CODE_PATH=/Users/j.zhou/DeepRLPractical/src
export LOG_DIR=/Users/j.zhou/DeepRLPractical/logs

python $CODE_PATH/main.py \
    --log-dir $LOG_DIR/'final_sub' \
    --t-max 32000012 \
    --n-jobs 16 \
    --eps-decay 1500000 \
    --ckpt-interval 1000000
