#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=18-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-6
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=tiered_5ft


DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped"
DATA_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/data"
BACKBONE_PATH="${DUMPED_PATH}/backbones/tiered_backbone_feyza/resnet18_last.pth"
EXP_FOLDER=$DUMPED_PATH/"tiered/finetune_5shot_converge_delta"

mkdir -p $EXP_FOLDER

cnt=0
for LMBD in 0.2 0.3; do
for TRLOSS in 0.0; do
for LR in 0.002 0.003; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
    EXP_NAME=trloss_${TRLOSS}_lmbd_${LMBD}_lr_${LR}_${SLURM_ARRAY_TASK_ID}
    LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
    LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
    python eval_incremental.py --model_path $BACKBONE_PATH \
                               --data_root $DATA_PATH \
                               --n_shots 5 \
                               --num_workers 0 \
                               --dataset tieredImageNet \
                               --eval_mode few-shot-incremental-fine-tune \
                               --classifier linear \
                               --min_novel_epochs 10 \
                               --model resnet18 \
                               --lmbd_reg_transform_w $LMBD \
                               --learning_rate $LR \
                               --freeze_backbone_at 1 \
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
#                            --n_shots 5 \
#                            --classifier linear \
#                            --eval_mode few-shot-incremental-fine-tune \
#                            --novel_epochs 10 \
#                            --learning_rate 0.001 \
#                            --dataset tieredImageNet \
#                            --freeze_backbone_at 1 \
#                            --lmbd_reg_transform_w 1.0 \
#                            --target_train_loss 1.5



# Checklist to run an array job.
# 1. Make sure total number of experiments matches the array param in sbatch.
# 2. Make sure the order that params are written to file matches the reassignment.
# 3.


# BACKBONE_PATH="${DUMPED_PATH}/backbones/label+desc/resnet18_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_layer_${LAYER}_multip_${MULTIPFC}/resnet18_last.pth" # label+desc
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

# for LMBD in 0.1 0.2 0.3; do
#     for TRLOSS in 0.5 0.6; do
#         for NOVELEPOCH in 20; do
#             for LR in 0.002; do
#                 echo "${LMBD} ${TRLOSS} ${NOVELEPOCH} ${LR}" >> $FILE
#             done
#         done
#     done
# done


# # Read the SLURM_ARRAY_TASK_ID line from the params file.
# LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $FILE)
# read -ra PARAMS<<< "$LINE"

# FILE="$DUMPED_PATH/${SLURM_ARRAY_TASK_ID}_temp_hyperparameters.txt"
# rm $FILE



# LMBD="${PARAMS[0]}"
# TRLOSS="${PARAMS[1]}"
# NOVELEPOCH="${PARAMS[2]}"
# LR="${PARAMS[3]}"
