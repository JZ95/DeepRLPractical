export HFO_PATH=/Users/j.zhou/coursework_rl/HFO
export CODE_PATH=/Users/j.zhou/DeepRLPractical/src
export LOG_DIR=/Users/j.zhou/DeepRLPractical/logs
export EXP_NAME=baseline
python $CODE_PATH/main.py --log-dir $LOG_DIR/$EXP_NAME --t-max 10000 --n-jobs 2