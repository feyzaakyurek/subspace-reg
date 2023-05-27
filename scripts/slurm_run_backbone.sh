#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-6
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=attreg_back_query_sizes



DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
DATA_PATH="/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/data"
BACKBONES_FOLDER=${DUMPED_PATH}/backbones/tieredImageNet/linear_trial
mkdir -p $BACKBONES_FOLDER


EXP_NAME=bias_false
EXP_FOLDER=$BACKBONES_FOLDER/$EXP_NAME
mkdir -p $EXP_FOLDER
LOG_STDOUT="${EXP_FOLDER}/log.out"
python -u train_supervised.py --trial pretrain \
                              --model_path $EXP_FOLDER  \
                              --data_root $DATA_PATH \
                              --epochs 60 \
                              --no_linear_bias \
                              --augment_pretrain_wtrainb \
                              --lr_decay_epochs "30,45" \
                              --dataset tieredImageNet \
                              --classifier linear # &> $LOG_STDOUT
                              
                
# EXP_NAME=bias_false
# EXP_FOLDER=$BACKBONES_FOLDER/$EXP_NAME
# mkdir -p $EXP_FOLDER
# LOG_STDOUT="${EXP_FOLDER}/log.out"
# python -u train_supervised.py --trial pretrain \
#                               --model_path $EXP_FOLDER  \
#                               --data_root data \
#                               --no_linear_bias \
#                               --data_root $DATA_PATH \
#                               --classifier linear &> $LOG_STDOUT
                              
                              
