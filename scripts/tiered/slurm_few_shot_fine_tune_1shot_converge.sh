#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=18-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-12
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=tiered_1ft


DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped"
DATA_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/data"
BACKBONE_PATH="${DUMPED_PATH}/backbones/tiered_backbone_feyza/resnet18_last.pth"
EXP_FOLDER=$DUMPED_PATH/"tiered/finetune_1shot_converge_delta"

mkdir -p $EXP_FOLDER

cnt=0
for LMBD in 0.2 0.3; do
for TRLOSS in 0.0; do
for LR in 0.002 0.003; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
    EXP_NAME=lmbd_${LMBD}_trloss_${TRLOSS}_lr_${LR}_${SLURM_ARRAY_TASK_ID}
    LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
    LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
    python eval_incremental.py --model_path $BACKBONE_PATH \
                               --data_root $DATA_PATH \
                               --n_shots 1 \
                               --eval_mode few-shot-incremental-fine-tune \
                               --classifier linear \
                               --min_novel_epochs 5 \
                               --model resnet18 \
                               --num_workers 0 \
                               --dataset tieredImageNet \
                               --learning_rate $LR \
                               --lmbd_reg_transform_w $LMBD \
                               --freeze_backbone_at 1 \
                               --weight_decay 5e-3 \
			                         --skip_val \
                               --target_train_loss $TRLOSS > $LOG_STDOUT 2> $LOG_STDERR
fi
done
done
done

# For debugging.


# No language fine tuning few-shot
# python eval_incremental.py --model_path $BACKBONE_PATH \
#                            --data_root $DATA_PATH \
#                            --n_shots 1 \
#                            --model resnet18 \
#                            --classifier linear \
#                            --eval_mode few-shot-incremental-fine-tune \
#                            --min_novel_epochs 5 \
#                            --learning_rate 0.003 \
#                            --dataset tieredImageNet \
#                            --freeze_backbone_at 1 \
#                            --lmbd_reg_transform_w 0.5 \
#                            --weight_decay 5e-3 \
#                            --target_train_loss 0.1



# Checklist to run an array job.
# 1. Make sure total number of experiments matches the array param in sbatch.
# 2. Make sure the order that params are written to file matches the reassignment.
# 3.