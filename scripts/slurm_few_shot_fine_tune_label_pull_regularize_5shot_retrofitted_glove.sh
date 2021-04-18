#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-4
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=pull_retro5


DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
EXP_FOLDER=$DUMPED_PATH/"finetune_label_pull_retrofitted_glove_new_episodes"
DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
# # BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet12_miniImageNet_linear_classifier_wbias/resnet12_last.pth"
BACKBONE_PATH="${DUMPED_PATH}/backbones/linear/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_linear_8075566/resnet12_last.pth"

mkdir -p $EXP_FOLDER

PPDB="ppdb-xl"
WSYN="wordnet-synonyms"
WSYNP="wordnet-synonyms+"
RETRO="word_embeds_retrofitted"

cnt=0
for LMBD in 0.2; do
for TRLOSS in 0.6 0.7; do
for PULL in 0.03 0.05; do
for EMBED in $WSYNP; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
    EXP_NAME=glove_${EMBED}_lambda_${LMBD}_trloss_${TRLOSS}_pull_${PULL}_$SLURM_ARRAY_TASK_ID
    LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
    LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
    python eval_incremental.py --model_path $BACKBONE_PATH \
                               --data_root $DATA_PATH \
                               --n_shots 5 \
                               --eval_mode few-shot-incremental-fine-tune \
                               --classifier linear \
                               --min_novel_epochs 20 \
                               --glove \
                               --learning_rate 0.002 \
                               --use_episodes \
                               --freeze_backbone_at 1 \
                               --label_pull $PULL \
                               --pulling regularize \
                               --word_embed_path $RETRO/$EMBED \
                               --word_embed_size 300 \
                               --lmbd_reg_transform_w $LMBD \
                               --target_train_loss $TRLOSS > $LOG_STDOUT 2> $LOG_STDERR
fi
done
done
done
done
# For debugging. 

# No language fine tuning few-shot with label pull
# python eval_incremental.py --model_path $BACKBONE_PATH \
#                            --data_root $DATA_PATH \
#                            --n_shots 5 \
#                            --eval_mode few-shot-incremental-fine-tune \
#                            --classifier linear \
#                            --novel_epochs 20 \
#                            --learning_rate 0.002 \
#                            --use_episodes \
#                            --neval_episodes 1 \
#                            --freeze_backbone_at 1 \
#                            --label_pull 0.03 \
#                            --pulling regularize \
#                            --lmbd_reg_transform_w 0.3 \
#                            --target_train_loss 1.1

