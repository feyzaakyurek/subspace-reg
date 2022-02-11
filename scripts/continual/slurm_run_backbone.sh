#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-10
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=resnet18



CURRENT="$PWD"
DUMPED_PATH="$CURRENT/dumped"
DATA_PATH="$CURRENT/data"
BACKBONE_FOLDER=${DUMPED_PATH}/backbones/continual/resnet18
mkdir -p $BACKBONE_FOLDER

cnt=0
for SEED in {1..10}; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then 
    EXP_NAME=continual_backbone_seed_${SEED}
    EXP_FOLDER=$BACKBONE_FOLDER/$SEED
    mkdir -p $EXP_FOLDER
    LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
    LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
    python train_supervised.py --trial pretrain \
                               --tb_path tb \
                               --data_root $DATA_PATH \
                               --classifier linear \
                               --model_path $EXP_FOLDER \
                               --continual \
                               --model resnet18 \
                               --no_dropblock \
                               --save_freq 100 \
                               --no_linear_bias \
                               --set_seed $SEED > $LOG_STDOUT 2> $LOG_STDERR
fi
done


# If running for a single seed use below (comment out above
# keeping the variable definitions such as BACKBONE_FOLDER):

# python train_supervised.py --trial pretrain \
#                                --tb_path tb \
#                                --data_root $DATA_PATH \
#                                --classifier linear \
#                                --model_path $BACKBONE_FOLDER/20 \
#                                --continual \
#                                --model resnet18 \
#                                --no_dropblock \
#                                --save_freq 100 \
#                                --no_linear_bias \
#                                --set_seed 20
