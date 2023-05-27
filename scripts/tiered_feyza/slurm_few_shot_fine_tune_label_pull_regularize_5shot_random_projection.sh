#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-3
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=tiered_5SubD


DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
EXP_FOLDER=$DUMPED_PATH/"tiered/finetune_5shot_distance2subspace_converge_delta_1"
DATA_PATH="/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/data"
BACKBONE_PATH="${DUMPED_PATH}/backbones/tieredImageNet/linear/resnet18/bias_false/resnet18_last.pth"
mkdir -p $EXP_FOLDER

# For running slurm jobs.

# cnt=0
# for LMBD in 0.2; do
# for TRLOSS in 0.0; do
# for PULL in 0.01 0.03 0.05; do
# (( cnt++ ))
# if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
#     EXP_NAME=lambda_${LMBD}_trloss_${TRLOSS}_pull_${PULL}_${SLURM_ARRAY_TASK_ID}
#     LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
#     LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
#     python eval_incremental.py --model_path $BACKBONE_PATH \
#                                --data_root $DATA_PATH \
#                                --n_shots 5 \
#                                --dataset tieredImageNet \
#                                --eval_mode few-shot-incremental-fine-tune \
#                                --classifier linear \
#                                --min_novel_epochs 10 \
#                                --learning_rate 0.002 \
#                                --attraction_override "distance2subspace" \
#                                --freeze_backbone_at 1 \
#                                --label_pull $PULL \
#                                --model resnet18 \
#                                --glove \
#                                --num_workers 0 \
#                                --skip_val \
#                                --pulling regularize \
#                                --lmbd_reg_transform_w $LMBD \
#                                --target_train_loss $TRLOSS > $LOG_STDOUT 2> $LOG_STDERR
# fi
# done
# done
# done

# For debugging.
LMBD=0.2
TRLOSS=0.0
PULL=0.03
python eval_incremental.py --model_path $BACKBONE_PATH \
                               --data_root $DATA_PATH \
                               --n_shots 5 \
                               --dataset tieredImageNet \
                               --eval_mode few-shot-incremental-fine-tune \
                               --classifier linear \
                               --min_novel_epochs 10 \
                               --learning_rate 0.002 \
                               --attraction_override "distance2subspace" \
                               --freeze_backbone_at 1 \
                               --label_pull $PULL \
                               --model resnet18 \
                               --glove \
                               --num_workers 0 \
                               --skip_val \
                               --pulling regularize \
                               --lmbd_reg_transform_w $LMBD \
                               --target_train_loss $TRLOSS

