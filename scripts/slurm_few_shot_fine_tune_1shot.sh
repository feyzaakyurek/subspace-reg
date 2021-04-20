#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-6
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=fmini1ft


DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
EXP_FOLDER=$DUMPED_PATH/"1shot/converge/finetune_new_episodes/"
DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_linear_8075566/resnet12_last.pth"

mkdir -p $EXP_FOLDER

cnt=0
for LMBD in 0.02 0.1 0.2; do
for LR in 0.003 0.006; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
    EXP_NAME=lmbd_${LMBD}_lr_${LR}_maxnovelep_1000_${SLURM_ARRAY_TASK_ID}
    LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
    LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
    python eval_incremental.py --model_path $BACKBONE_PATH \
                               --data_root $DATA_PATH \
                               --n_shots 1 \
                               --eval_mode few-shot-incremental-fine-tune \
                               --classifier linear \
                               --min_novel_epochs 20 \
                               --learning_rate $LR \
                               --use_episodes \
                               --max_novel_epochs 1000 \
                               --lmbd_reg_transform_w $LMBD \
                               --freeze_backbone_at 1 \
                               --num_workers 0 \
                               --skip_val \
                               --target_train_loss 0.0 > $LOG_STDOUT 2> $LOG_STDERR
fi
done
done

# For debugging.                           


# No language fine tuning few-shot
# python eval_incremental.py --model_path $BACKBONE_PATH \
#                            --data_root $DATA_PATH \
#                            --n_shots 1 \
#                            --classifier linear \
#                            --eval_mode few-shot-incremental-fine-tune \
#                            --novel_epochs 20 \
#                            --learning_rate 0.003 \
#                            --neval_episodes 15 \
#                            --freeze_backbone_at 1 \
#                            --lmbd_reg_transform_w 0.2 \
#                            --target_train_loss 0.1
                           


# Checklist to run an array job.
# 1. Make sure total number of experiments matches the array param in sbatch.
# 2. Make sure the order that params are written to file matches the reassignment.
# 3. 
