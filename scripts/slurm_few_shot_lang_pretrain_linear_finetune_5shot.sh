#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-18
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=lang-pre-tune-lin-5shot


# Create the combinations of params for each array task,
# and save them to a temp params file.sc
DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped"
FILE="$DUMPED_PATH/${SLURM_ARRAY_TASK_ID}_temp_hyperparameters.txt"
rm $FILE 

for LMBD in 0.4 0.5 0.3; do
    for TRLOSS in 0.5 0.5 0.7; do
        for NOVELEPOCH in 20; do
            for LR in 0.001 0.0005; do 
                echo "${LMBD} ${TRLOSS} ${NOVELEPOCH} ${LR}" >> $FILE
            done
        done
    done
done


# Read the SLURM_ARRAY_TASK_ID line from the params file.
LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $FILE) 
read -ra PARAMS<<< "$LINE"

LMBD="${PARAMS[0]}"
TRLOSS="${PARAMS[1]}"
NOVELEPOCH="${PARAMS[2]}"
LR="${PARAMS[3]}"

# Create log files
LOG_STDOUT="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
LOG_STDERR="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# LABEL ONLY FEW-SHOT LANG PRETRAINING LINEAR FINETUNING
DATA_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/data"
BACKBONE_PATH="${DUMPED_PATH}/backbones/label/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_lang_linear_multip_0.05/resnet12_lastFalse.pth"
python eval_incremental.py --model_path $BACKBONE_PATH \
                           --data_root $DATA_PATH \
                           --n_shots 5 \
                           --eval_mode few-shot-incremental-language-pretrain-linear-tune \
                           --classifier lang-linear \
                           --novel_epochs $NOVELEPOCH \
                           --learning_rate $LR \
                           --freeze_backbone_at 1 \
                           --num_workers 10 \
                           --lmbd_reg_transform_w $LMBD \
                           --target_train_loss $TRLOSS \
                           --multip_fc 0.05 \
                           --orig_alpha 1.0 > $LOG_STDOUT 2> $LOG_STDERR
                           


# # For debugging.                           


# # # No language fine tuning few-shot
# export DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped"
# export DATA_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/data"
# export BACKBONE_PATH="${DUMPED_PATH}/backbones/label/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_lang_linear_multip_0.05/resnet12_lastFalse.pth"
# python eval_incremental.py --model_path $BACKBONE_PATH \
#                            --data_root $DATA_PATH \
#                            --n_shots 5 \
#                            --eval_mode few-shot-incremental-language-pretrain-linear-tune \
#                            --classifier lang-linear \
#                            --novel_epochs 20 \
#                            --learning_rate 0.002 \
#                            --freeze_backbone_at 1 \
#                            --lmbd_reg_transform_w 0.5 \
#                            --target_train_loss 0.7 \
#                            --multip_fc 0.05 \
#                            --orig_alpha 1.0
                           
# Checklist to run an array job.
# 1. Make sure total number of experiments matches the array param in sbatch.
# 2. Make sure the order that params are written to file matches the reassignment.
# 3. 

