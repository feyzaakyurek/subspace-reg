# #!/bin/bash
# #SBATCH --constraint=xeon-g6
# #SBATCH --time=15-00:00
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --gres=gpu:volta:1
# #SBATCH --array=1-2
# #SBATCH --output=dumped/%A_%a.out
# #SBATCH --error=dumped/%A_%a.err
# #SBATCH --job-name=ft_5push


DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
EXP_FOLDER=$DUMPED_PATH/"finetune_afterbug_pushaway"
DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
# BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet12_miniImageNet_linear_classifier_wbias/resnet12_last.pth"
BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_linear_8075566/resnet12_last.pth"
mkdir -p $EXP_FOLDER

# cnt=0
# for TRLOSS in 0.6; do
# for LR in 0.002; do
# for LMBD in 0.2; do
# (( cnt++ ))
# if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
#     EXP_NAME=trloss_${TRLOSS}_lmbd_${LMBD}_${SLURM_ARRAY_TASK_ID}
#     LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
#     LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
#     python eval_incremental.py --model_path $BACKBONE_PATH \
#                                --data_root $DATA_PATH \
#                                --n_shots 5 \
#                                --eval_mode few-shot-incremental-fine-tune \
#                                --classifier linear \
#                                --novel_epochs 20 \
#                                --learning_rate $LR \
#                                --use_episodes \
#                                --test_base_batch_size 1600 \
#                                --continual \
#                                --lmbd_reg_transform_w $LMBD \
#                                --freeze_backbone_at 1 \
#                                --target_train_loss $TRLOSS > $LOG_STDOUT 2> $LOG_STDERR
# fi
# done
# done
# done

# For debugging.                           


# No language fine tuning few-shot
python eval_incremental.py --model_path $BACKBONE_PATH \
                           --data_root $DATA_PATH \
                           --n_shots 5 \
                           --classifier linear \
                           --eval_mode few-shot-incremental-fine-tune \
                           --novel_epochs 20 \
                           --learning_rate 0.002 \
                           --freeze_backbone_at 1 \
                           --test_base_batch_size 2000 \
                           --continual \
                           --num_workers 0 \
                           --n_queries 25 \
                           --lmbd_reg_transform_w 0.2 \
                           --target_train_loss 0.6
                           


# Checklist to run an array job.
# 1. Make sure total number of experiments matches the array param in sbatch.
# 2. Make sure the order that params are written to file matches the reassignment.
# 3. 
