export HFO_PATH=/Users/j.zhou/coursework_rl/HFO
export CODE_PATH=/Users/j.zhou/DeepRLPractical/src
export LOG_DIR=/Users/j.zhou/DeepRLPractical/logs
export EXP_NAME=mix_reward_ball_shoot_ball_vel_angle_close_to_goal_but_not_too_close
python $CODE_PATH/main.py --log-dir $LOG_DIR/$EXP_NAME --t-max 100000 --n-jobs 1
