#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-1
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=tiered_pull5


DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped/"
EXP_FOLDER=$DUMPED_PATH/"tiered/finetune_5shot_linear_pull"
DATA_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/data"
BACKBONE_PATH="${DUMPED_PATH}/backbones/tiered_backbone_feyza/resnet18_last_with_mapping.pth"
mkdir -p $EXP_FOLDER
#lambda_0.2_trloss_1.0_pull_0.05_lr_0.001_2
cnt=0
# for LMBD in 0.2 0.3; do
# for TRLOSS in 1.0 1.2; do
# for PULL in 0.05 0.2 0.3; do
# for LR in 0.002 0.001; do
for LMBD in 0.2; do
for TRLOSS in 1.0; do
for PULL in 0.05; do
for LR in 0.001; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID  ]]; then
    EXP_NAME=lambda_${LMBD}_trloss_${TRLOSS}_pull_${PULL}_lr_${LR}_${SLURM_ARRAY_TASK_ID}
    LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
    LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
    python eval_incremental.py --model_path $BACKBONE_PATH \
                               --data_root $DATA_PATH \
                               --n_shots 5 \
                               --dataset tieredImageNet \
                               --eval_mode few-shot-incremental-fine-tune \
                               --classifier linear \
                               --min_novel_epochs 10 \
                               --model resnet18 \
                               --learning_rate $LR \
                               --freeze_backbone_at 1 \
                               --label_pull $PULL \
                               --glove \
                               --num_workers 0 \
                               --pulling regularize \
                               --lmbd_reg_transform_w $LMBD \
			       --skip_val \
                               --attraction_override "mapping_linear_label2image" \
                               --target_train_loss $TRLOSS > $LOG_STDOUT 2> $LOG_STDERR
fi
done
done
done
done
# For debugging.

# No language fine tuning few-shot with label pull
# python eval_incremental.py --model_path $BACKBONE_PATH \
#                                --data_root $DATA_PATH \
#                                --n_shots 5 \
#                                --dataset tieredImageNet \
#                                --eval_mode few-shot-incremental-fine-tune \
#                                --classifier linear \
#                                --novel_epochs 10 \
#                                --learning_rate 0.001 \
#                                --freeze_backbone_at 1 \
#                                --label_pull 0.3 \
#                                --pulling regularize \
#                                --lmbd_reg_transform_w 2.0 \
#                                --target_train_loss 2.0
# --use_episodes \

# # No language fine tuning few-shot
# export DUMPED_PATH="dumped"
# export DATA_PATH="data"
# # export BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet18_miniImageNet_linear_classifier_wbias/resnet18_last.pth"
# BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet18_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_linear_8075566/resnet18_last.pth"
# # CUDA_VISIBLE_DEVICES=6 python eval_incremental.py --model_path $BACKBONE_PATH \
# #                            --data_root $DATA_PATH \
# #                            --n_shots 5 \
# #                            --classifier linear \
# #                            --eval_mode few-shot-incremental-fine-tune \
# #                            --novel_epochs 20 \
# #                            --learning_rate 0.002 \
# #                            --freeze_backbone_at 1 \
# #                            --lmbd_reg_transform_w 0.2 \
# #                            --target_train_loss 0.5 \
# #                            --use_episodes \
# #                            --neval_episodes 1 \
# #                            --track_weights

# CUDA_VISIBLE_DEVICES=6 python eval_incremental.py --model_path $BACKBONE_PATH \
#                            --data_root $DATA_PATH \
#                            --n_shots 5 \
#                            --classifier linear \
#                            --eval_mode few-shot-incremental-fine-tune \
#                            --novel_epochs 20 \
#                            --learning_rate 0.002 \
#                            --freeze_backbone_at 1 \
#                            --lmbd_reg_transform_w 0.2 \
#                            --target_train_loss 0.6 \
#                            --use_episodes \
#                            --neval_episodes 300 \
#                            --label_pull 0.05 \
#                            --pulling last-mile
#                            --word_embed_size 500 \
#                            --track_label_inspired_weights \
#                            --track_weights \
#                            --word_embed_size 500 # > labelpullnon0.out
# #
# #                            --track_weights \
# --glove \
#
#
# Checklist to run an array job.
# 1. Make sure total number of experiments matches the array param in sbatch.
# 2. Make sure the order that params are written to file matches the reassignment.
# 3.


##export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# # Start (or restart) experiment
# date >> $LOG_STDOUT
# which python >> $LOG_STDOUT
# echo "---Beginning program ---" >> $LOG_STDOUT
# echo "Exp name     : rfs_base" >> $LOG_STDOUT
# echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT
# echo "SBATCH script: slurm_run.sh" >> $LOG_STDOUT

# python eval_incremental.py --model_path dumped/ekin_dumped/resnet18_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_layer_${LAYER}_multip_${MULTIPFC}/resnet18_last.pth --data_root data --n_shots 5 --eval_mode few-shot-language-incremental --classifier description-linear --novel_epochs 14 --learning_rate 0.001 --freeze_backbone --lmbd_reg_transform_w 0.01 --target_train_loss 0.6 --prefix_label --multip_fc $MULTIPFC --transformer_layer $LAYER

# export WANDB_MODE=dryrun
# python train_supervised.py --trial pretrain --model_path dumped --tb_path tb --data_root data --classifier description-linear --desc_embed_model bert-base-cased --prefix_label --multip_fc $MULTIPFC --transformer_layer $LAYER


# python eval_incremental.py --model_path dumped/ekin_dumped/resnet18_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_layer_6_multip_0.1/resnet18_last.pth --data_root data --n_shots 5 --eval_mode few-shot-language-incremental --classifier description-linear --novel_epochs 14 --learning_rate 0.001 --num_novel_combs 599 --freeze_backbone --lmbd_reg_transform_w 0.01 --target_train_loss 0.6 --prefix_label --multip_fc 0.1 --transformer_layer 6


# # Create the combinations of params for each array task,
# # and save them to a temp params file.sc
# DUMPED_PATH="/raid/lingo/akyurek/git/rfs-incremental/dumped"
# FILE="$DUMPED_PATH/${SLURM_ARRAY_TASK_ID}_temp_hyperparameters.txt"
# rm $FILE

# for LMBD in 0.2 0.3; do
#     for TRLOSS in 0.5 0.55 0.6; do
#         for NOVELEPOCH in 20; do
#             for LR in 0.002 0.001; do
#                 for PULL in 0.0 0.05 0.1; do
#                     echo "${LMBD} ${TRLOSS} ${NOVELEPOCH} ${LR} ${PULL}" >> $FILE
#                 done
#             done
#         done
#     done
# done


# # Read the SLURM_ARRAY_TASK_ID line from the params file.
# LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $FILE)
# read -ra PARAMS<<< "$LINE"

# LMBD="${PARAMS[0]}"
# TRLOSS="${PARAMS[1]}"
# NOVELEPOCH="${PARAMS[2]}"
# LR="${PARAMS[3]}"
# PULL="${PARAMS[4]}"

# # Create log files
# LOG_STDOUT="${DUMPED_PATH}/label_pull_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
# LOG_STDERR="${DUMPED_PATH}/label_pull_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"


# # LABEL ONLY FEW-SHOT FINETUNING
# BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet18_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_linear_8075566/resnet18_last.pth"
# # export BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet18_miniImageNet_linear_classifier_wbias/resnet18_last.pth"
# export DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
# python eval_incremental.py --model_path $BACKBONE_PATH \
#                            --data_root $DATA_PATH \
#                            --n_shots 5 \
#                            --eval_mode few-shot-incremental-fine-tune \
#                            --classifier linear \
#                            --novel_epochs $NOVELEPOCH \
#                            --learning_rate $LR \
#                            --use_episodes \
#                            --word_embed_size 500 \
#                            --label_pull $PULL \
#                            --freeze_backbone_at 1 \
#                            --lmbd_reg_transform_w $LMBD \
#                            --target_train_loss $TRLOSS > $LOG_STDOUT 2> $LOG_STDERR
