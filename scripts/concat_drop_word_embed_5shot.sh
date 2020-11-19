#!/bin/bash
#SBATCH --dependency afterok:8286105
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-18
#SBATCH --output=dumped/concat_drop_word_embed/%A_%a.out
#SBATCH --error=dumped/concat_drop_word_embed/%A_%a.err
#SBATCH --job-name=episod-concat-reg



# Create the combinations of params for each array task,
# and save them to a temp params file.sc
DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped/concat_drop_word_embed_drop"
DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
WORD_EMBEDS="/home/gridsan/groups/akyureklab/rfs-incremental/word_embeds"


# for LMBD in 0.5 0.6; do
#     for TRLOSS in 0.7; do
#         for NOVELEPOCH in 20; do
#             for LR in 0.002; do
#                 echo "${LMBD} ${TRLOSS} ${NOVELEPOCH} ${LR}" >> $FILE
#             done
#         done
#     done
# done

cnt=0
for LMBD in 0.4 0.5 0.6; do
        for TRLOSS in 0.65 0.7 0.75; do
                 for LR in 0.002 0.003; do
                         (( cnt++ ))
                        if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
                                exp_name=lambda_${LMBD}_trloss_${TRLOSS}_lr_${LR}
                                #     for backbone in '0.05_8286111' '0.075_8286112' '0.025_8286107' '0.075_8286113' '0.025_8286106' '0.025_8286108' '0.05_8286109' '0.05_8286110' '0.075_8286105'; do
                                #         BACKBONE_PATH="${DUMPED_PATH}/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_lang-linear_multip_${backbone}/resnet12_last.pth"
                                BACKBONE_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped/backbone_finetune_word_embed/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_lang-linear_multip_0.075_8406906/resnet12_last.pth"

                                LOG_STDOUT="${DUMPED_PATH}/finetune_${exp_name}.out"
                                LOG_STDERR="${DUMPED_PATH}/finetune_${exp_name}.err"
                                python -u eval_incremental.py --model_path $BACKBONE_PATH \
                                                   --data_root $DATA_PATH \
                                                   --n_shots 5 \
                                                   --eval_mode few-shot-language-incremental \
                                                   --classifier lang-linear \
                                                   --novel_epochs 20 \
                                                   --learning_rate $LR \
                                                   --freeze_backbone_at 1 \
                                                   --attention concat \
                                                   --use_episodes \
                                                   --word_embed_path $WORD_EMBEDS \
                                                   --word_embed_type "" \
                                                   --lmbd_reg_transform_w $LMBD \
                                                   --target_train_loss $TRLOSS > $LOG_STDOUT  2> $LOG_STDERR
                        fi
        done
    done
done

# FILE="$DUMPED_PATH/${SLURM_ARRAY_TASK_ID}_temp_hyperparameters.txt"
# rm $FILE

# for LMBD in 0.5 0.6 0.7 0.9; do
#     for TRLOSS in 0.7 0.75 0.8; do
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

# LMBD="${PARAMS[0]}"
# TRLOSS="${PARAMS[1]}"
# NOVELEPOCH="${PARAMS[2]}"
# LR="${PARAMS[3]}"

# Create log files

# LOG_STDERR="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
# BACKBONE_PATH="${DUMPED_PATH}/backbones/label+desc/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_layer_${LAYER}_multip_${MULTIPFC}/resnet12_last.pth" # label+desc


# LABEL ONLY FEW-SHOT FINETUNING USING PRE-SPECIFIED EPIDODES
# BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_linear_8075566/resnet12_last.pth"





# For debugging.

# export DUMPED_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/dumped/dumped_feyza"
# # export LOG_STDOUT="${DUMPED_PATH}/3453264.out" #random
# # export LOG_STDERR="${DUMPED_PATH}/3453264.err" #random
# export BACKBONE_PATH="${DUMPED_PATH}/backbones/c-x-sum/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_lang-linear_multip_0.05_858201/resnet12_last.pth"
# # export BACKBONE_PATH="${DUMPED_PATH}/backbones/c-x-sum/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_lang-linear_multip_0.05_858201/resnet12_last.pth"
# export DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
# export WORD_EMBEDS="/home/gridsan/groups/akyureklab/rfs-incremental/word_embeds"
# python eval_incremental.py --model_path $BACKBONE_PATH \
#                            --data_root $DATA_PATH \
#                            --n_shots 5 \
#                            --classifier lang-linear \
#                            --eval_mode few-shot-language-incremental \
#                            --novel_epochs 20 \
#                            --learning_rate 0.002 \
#                            --freeze_backbone_at 1 \
#                            --lmbd_reg_transform_w 0.5 \
#                            --attention sum \
#                            --use_episodes \
#                            --word_embed_path $WORD_EMBEDS \
#                            --target_train_loss 0.7 # > $LOG_STDOUT 2> $LOG_STDERR






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

# python eval_incremental.py --model_path dumped/ekin_dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_layer_${LAYER}_multip_${MULTIPFC}/resnet12_last.pth --data_root data --n_shots 5 --eval_mode few-shot-language-incremental --classifier description-linear --novel_epochs 14 --learning_rate 0.001 --freeze_backbone --lmbd_reg_transform_w 0.01 --target_train_loss 0.6 --prefix_label --multip_fc $MULTIPFC --transformer_layer $LAYER

# export WANDB_MODE=dryrun
# python train_supervised.py --trial pretrain --model_path dumped --tb_path tb --data_root data --classifier description-linear --desc_embed_model bert-base-cased --prefix_label --multip_fc $MULTIPFC --transformer_layer $LAYER


# python eval_incremental.py --model_path dumped/ekin_dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_layer_6_multip_0.1/resnet12_last.pth --data_root data --n_shots 5 --eval_mode few-shot-language-incremental --classifier description-linear --novel_epochs 14 --learning_rate 0.001 --num_novel_combs 599 --freeze_backbone --lmbd_reg_transform_w 0.01 --target_train_loss 0.6 --prefix_label --multip_fc 0.1 --transformer_layer 6
