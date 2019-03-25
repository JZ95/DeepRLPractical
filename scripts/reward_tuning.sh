export CODE_PATH=/home/workspace/src
export LOG_DIR=/home/workspace/logs
python $CODE_PATH/main.py --log-dir $LOG_DIR/'baseline' --t-max 1000000 --n-jobs 8 --eps-decay 30000
python $CODE_PATH/main.py --log-dir $LOG_DIR/'baseline-closer2ball-0.5' --t-max 1000000 --n-jobs 8 --eps-decay 30000 --reward-opt 'baseline-closer2ball-0.5'
python $CODE_PATH/main.py --log-dir $LOG_DIR/'baseline-closer2ball-0.25' --t-max 1000000 --n-jobs 8 --eps-decay 30000 --reward-opt 'baseline-closer2ball-0.25'
