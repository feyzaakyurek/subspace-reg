#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-1
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=tiered_5ftD


DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
EXP_FOLDER=$DUMPED_PATH/"tiered/finetune_5shot_converge_delta"
# DATA_PATH="/home/gridsan/akyurek/git/rfs-incremental/data"
DATA_PATH="/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/data"
BACKBONE_PATH="${DUMPED_PATH}/backbones/tieredImageNet/linear/resnet18/bias_false/resnet18_last.pth"
mkdir -p $EXP_FOLDER

# for running slurm jobs

# cnt=0
# for TRLOSS in 0.0; do
# for LMBD in 0.3; do
# for WD in 5e-3; do
# for LR in 0.001; do
# (( cnt++ ))
# if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
#     EXP_NAME=trloss_${TRLOSS}_lmbd_${LMBD}_lr_${LR}_wd_${WD}_${SLURM_ARRAY_TASK_ID}
#     LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
#     LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
#     python eval_incremental.py --model_path $BACKBONE_PATH \
#                                --data_root $DATA_PATH \
#                                --n_shots 5 \
#                                --dataset tieredImageNet \
#                                --model resnet18 \
#                                --eval_mode few-shot-incremental-fine-tune \
#                                --classifier linear \
#                                --min_novel_epochs 10 \
#                                --lmbd_reg_transform_w $LMBD \
#                                --learning_rate $LR \
#                                --freeze_backbone_at 1 \
#                                --weight_decay $WD \
#                                --num_workers 0 \
#                                --skip_val \
#                                --target_train_loss $TRLOSS > $LOG_STDOUT 2> $LOG_STDERR
# fi
# done
# done
# done
# done

# For debugging.
TRLOSS=0.0
LMBD=0.3
WD=5e-3
LR=0.001
EXP_NAME=trloss_${TRLOSS}_lmbd_${LMBD}_lr_${LR}_wd_${WD}_deneme
LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
python eval_incremental.py --model_path $BACKBONE_PATH \
                            --data_root $DATA_PATH \
                            --n_shots 5 \
                            --dataset tieredImageNet \
                            --model resnet18 \
                            --eval_mode few-shot-incremental-fine-tune \
                            --classifier linear \
                            --min_novel_epochs 10 \
                            --lmbd_reg_transform_w $LMBD \
                            --learning_rate $LR \
                            --freeze_backbone_at 1 \
                            --weight_decay $WD \
                            --num_workers 0 \
                            --skip_val \
                            --target_train_loss $TRLOSS


