#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-5
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=fmini1ft


DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
EXP_FOLDER=$DUMPED_PATH/"converge/finetune_new_episodes"
DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
# BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet12_miniImageNet_linear_classifier_wbias/resnet12_last.pth"
BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_linear_8075566/resnet12_last.pth"
mkdir -p $EXP_FOLDER

# cnt=0
# for LR in 0.002; do
# for LMBD in 0.03 0.1 0.2 0.4 0.8; do
# (( cnt++ ))
# if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
#     EXP_NAME=lmbd_${LMBD}_${SLURM_ARRAY_TASK_ID}
#     LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
#     LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
#     python eval_incremental.py --model_path $BACKBONE_PATH \
#                                --data_root $DATA_PATH \
#                                --n_shots 5 \
#                                --eval_mode few-shot-incremental-fine-tune \
#                                --classifier linear \
#                                --min_novel_epochs 20 \
#                                --max_novel_epochs 1000 \
#                                --learning_rate $LR \
#                                --use_episodes \
#                                --num_workers 0 \
#                                --skip_val \
#                                --lmbd_reg_transform_w $LMBD \
#                                --freeze_backbone_at 1 \
#                                --target_train_loss 0.0 > $LOG_STDOUT 2> $LOG_STDERR
# fi
# done
# done

# For debugging.                           


# No language fine tuning few-shot
python eval_incremental.py --model_path $BACKBONE_PATH \
                           --data_root $DATA_PATH \
                           --n_shots 5 \
                           --classifier linear \
                           --eval_mode few-shot-incremental-fine-tune \
                           --min_novel_epochs 20 \
                           --learning_rate 0.002 \
                           --freeze_backbone_at 1 \
                           --use_episodes \
                           --num_workers 0 \
                           --lmbd_reg_transform_w 0.03 \
                           --target_train_loss 0.0 \
                           --save_preds_0 \
                           --skip_val
                           


# Checklist to run an array job.
# 1. Make sure total number of experiments matches the array param in sbatch.
# 2. Make sure the order that params are written to file matches the reassignment.
# 3. 
