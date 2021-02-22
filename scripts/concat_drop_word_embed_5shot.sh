#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-12
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=query_size300


DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped"
EXP_FOLDER=${DUMPED_PATH}/finetune_label_attention_concat_drop_diagreg_query_size_750/
DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
WORD_EMBEDS="/home/gridsan/groups/akyureklab/rfs-incremental/word_embeds"
mkdir -p $EXP_FOLDER


cnt=0
for MULTIPFC in 0.075; do
for DIAG_REG in 0.075; do
for QUERY_SIZE in 750; do
for LMBD in 0.2 0.3 0.4 0.5; do
for TRLOSS in 0.65 0.75 0.80; do
for LR in 0.002; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
    EXP_NAME=lambda_${LMBD}_trloss_${TRLOSS}_lr_${LR}
    BACKBONE_PATH=${DUMPED_PATH}/backbones/c-x-concat/simple_attention_concat_multipfc_${MULTIPFC}_diagreg_${DIAG_REG}_querysize_${QUERY_SIZE}/resnet12_last.pth
    LOG_STDOUT=${EXP_FOLDER}/${EXP_NAME}.out
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
                                  --diag_reg 0 \
                                  --lmbd_reg_transform_w $LMBD \
                                  --transform_query_size $QUERY_SIZE \
                                  --target_train_loss $TRLOSS &> $LOG_STDOUT
fi
done
done
done
done
done
done


    #     for backbone in '0.05_8286111' '0.075_8286112' '0.025_8286107' '0.075_8286113' '0.025_8286106' '0.025_8286108' '0.05_8286109' '0.05_8286110' '0.075_8286105'; do
    #         BACKBONE_PATH="${DUMPED_PATH}/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_lang-linear_multip_${backbone}/resnet12_last.pth"
#     BACKBONE_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped/backbone_finetune_word_embed/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_lang-linear_multip_0.075_8406906/resnet12_last.pth"
# DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped/concat_drop_word_embed_diag"
