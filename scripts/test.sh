export HFO_PATH=/Users/j.zhou/coursework_rl/HFO
export CODE_PATH=/Users/j.zhou/DeepRLPractical/src
export LOG_DIR=/Users/j.zhou/DeepRLPractical/logs
export EXP_NAME=mix_reward_ball_0.5
python $CODE_PATH/main.py --log-dir $LOG_DIR/$EXP_NAME --t-max 100000 -n-jobs 1