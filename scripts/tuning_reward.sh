export HFO_PATH=/Users/j.zhou/coursework_rl/HFO
export CODE_PATH=/Users/j.zhou/DeepRLPractical/src
export LOG_DIR=/Users/j.zhou/DeepRLPractical/logs
# python $CODE_PATH/main.py --log-dir $LOG_DIR/'baseline' --t-max 1000000 --n-jobs 8 --eps-decay 30000
# python $CODE_PATH/main.py --log-dir $LOG_DIR/'baseline-closer2ball-0.5' --t-max 1000000 --n-jobs 8 --eps-decay 30000 --reward-opt 'baseline-closer2ball-0.5'
# python $CODE_PATH/main.py --log-dir $LOG_DIR/'baseline-closer2ball-0.25' --t-max 1000000 --n-jobs 8 --eps-decay 30000 --reward-opt 'baseline-closer2ball-0.25'
# python $CODE_PATH/main.py \
#     --log-dir $LOG_DIR/'closer2ball-0.25-closer2-goal-0.5-goal-dist-lim-0.75' \
#     --t-max 10000 \
#     --n-jobs 2 \
#     --eps-decay 30000 \
#     --reward-opt 'closer2ball-0.25-closer2-goal-0.5-goal-dist-lim-0.75'

python $CODE_PATH/main.py \
    --log-dir $LOG_DIR/'baseline' \
    --t-max 1000000 \
    --n-jobs 8 \
    --eps-decay 40000 \

python $CODE_PATH/main.py \
    --log-dir $LOG_DIR/'closer2ball-0.25-v2' \
    --t-max 1000000 \
    --n-jobs 8 \
    --eps-decay 40000 \
    --reward-opt 'closer2ball-0.25'

python $CODE_PATH/main.py \
    --log-dir $LOG_DIR/'closer2ball-0.25-closer2-goal-0.25-goal-dist-lim-0.75-v2' \
    --t-max 1000000 \
    --n-jobs 8 \
    --eps-decay 40000 \
    --reward-opt 'closer2ball-0.25-closer2-goal-0.25-goal-dist-lim-0.75'

    #python $CODE_PATH/main.py \
#    --log-dir $LOG_DIR/'closer2ball-0.25-closer2-goal-0.5-goal-dist-lim-0.5' \
#    --t-max 1000000 \
#    --n-jobs 8 \
#    --eps-decay 30000 \
#    --reward-opt 'closer2ball-0.25-closer2-goal-0.5-goal-dist-lim-0.5'
