#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-10
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=cont_5label


DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
EXP_FOLDER=$DUMPED_PATH/"continual"/"finetune_label_pull_linear_mapping_converge_seeds"
DATA_PATH="/home/gridsan/akyurek/git/rfs-incremental/data"
mkdir -p $EXP_FOLDER

cnt=0
for TRLOSS in 0.0; do
for LR in 0.002; do
for LMBD in 0.2; do
for LMBDN in 0.1; do
for PULL in 0.1; do
for SEED in {1..10}; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
    EXP_NAME=seed_${SEED}_trloss_${TRLOSS}_lmbd_${LMBD}_lmbdN_${LMBDN}_pull_${PULL}_${SLURM_ARRAY_TASK_ID}
    LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
    LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
    BACKBONE_PATH="${DUMPED_PATH}/backbones/continual/resnet18/${SEED}/resnet18_last_with_mapping.pth"

    python eval_incremental.py --model_path $BACKBONE_PATH \
                           --model resnet18 \
                           --no_dropblock \
                           --data_root $DATA_PATH \
                           --n_shots 5 \
                           --classifier linear \
                           --eval_mode few-shot-incremental-fine-tune \
                           --min_novel_epochs 20 \
                           --learning_rate $LR \
                           --freeze_backbone_at 1 \
                           --test_base_batch_size 2000 \
                           --continual \
                           --num_workers 0 \
                           --n_queries 25 \
                           --lmbd_reg_transform_w $LMBD \
                           --target_train_loss $TRLOSS \
                           --label_pull $PULL \
                           --set_seed $SEED \
                           --lmbd_reg_novel $LMBDN \
                           --glove \
                           --attraction_override "mapping_linear_label2image" \
                           --save_preds_0 > $LOG_STDOUT 2> $LOG_STDERR
fi
done
done
done
done
done
done

# # For debugging.
# No language fine tuning few-shot
# BACKBONE_PATH="${DUMPED_PATH}/backbones/continual/resnet18/2/resnet18_last_with_mapping.pth"
# python eval_incremental.py --model_path $BACKBONE_PATH \
#                            --model resnet18 \
#                            --no_dropblock \
#                            --data_root $DATA_PATH \
#                            --n_shots 5 \
#                            --classifier linear \
#                            --eval_mode few-shot-incremental-fine-tune \
#                            --novel_epochs 20 \
#                            --max_novel_epochs 400 \
#                            --learning_rate 0.002 \
#                            --freeze_backbone_at 1 \
#                            --test_base_batch_size 2000 \
#                            --continual \
#                            --n_queries 25 \
#                            --num_workers 0 \
#                            --lmbd_reg_transform_w 0.2 \
#                            --target_train_loss 0.0 \
#                            --label_pull 1.0 \
#                            --set_seed 2 \
#                            --lmbd_reg_novel 0.1 \
#                            --glove \
#                            --attraction_override "mapping_linear_label2image"



# Checklist to run an array job.
# 1. Make sure total number of experiments matches the array param in sbatch.
# 2. Make sure the order that params are written to file matches the reassignment.
# 3.
