#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-2
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=tiered_1ft


DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
EXP_FOLDER=$DUMPED_PATH/"tiered/finetune_1shot_converge"
DATA_PATH="/home/gridsan/akyurek/git/rfs-incremental/data"
BACKBONE_PATH="${DUMPED_PATH}/backbones/tieredImageNet/linear/resnet18/bias_false/resnet18_last.pth"

mkdir -p $EXP_FOLDER

cnt=0
for LMBD in 0.3; do
for TRLOSS in 0.0; do
for WD in 5e-4 5e-3; do
for LR in 0.006; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
    EXP_NAME=lmbd_${LMBD}_trloss_${TRLOSS}_lr_${LR}_wd_${WD}_${SLURM_ARRAY_TASK_ID}
    LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
    LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
    python eval_incremental.py --model_path $BACKBONE_PATH \
                               --data_root $DATA_PATH \
                               --n_shots 1 \
                               --model resnet18 \
                               --eval_mode few-shot-incremental-fine-tune \
                               --classifier linear \
                               --min_novel_epochs 5 \
                               --dataset tieredImageNet \
                               --learning_rate $LR \
                               --lmbd_reg_transform_w $LMBD \
                               --freeze_backbone_at 1 \
                               --skip_val \
                               --num_workers 0 \
                               --weight_decay $WD \
                               --target_train_loss $TRLOSS > $LOG_STDOUT 2> $LOG_STDERR
fi
done
done
done
done

# For debugging.


# No language fine tuning few-shot
# python eval_incremental.py --model_path $BACKBONE_PATH \
#                            --data_root $DATA_PATH \
#                            --n_shots 1 \
#                            --classifier linear \
#                            --eval_mode few-shot-incremental-fine-tune \
#                            --novel_epochs 5 \
#                            --learning_rate 0.001 \
#                            --dataset tieredImageNet \
#                            --freeze_backbone_at 1 \
#                            --lmbd_reg_transform_w 0.5 \
#                            --weight_decay 5e-3 \
#                            --target_train_loss 0.3



# Checklist to run an array job.
# 1. Make sure total number of experiments matches the array param in sbatch.
# 2. Make sure the order that params are written to file matches the reassignment.
# 3.
