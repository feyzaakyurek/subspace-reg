# #!/bin/bash
# #SBATCH --dependency afterok:8286105
# #SBATCH --constraint=xeon-g6
# #SBATCH --time=15-00:00
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --gres=gpu:volta:1
# #SBATCH --array=1-18
# #SBATCH --output=dumped/%A_%a.out
# #SBATCH --error=dumped/%A_%a.err
# #SBATCH --job-name=concat-replicate


MY_DUMPED="/home/gridsan/akyurek/git/rfs-incremental/dumped"
SHARED_DUMPED="/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/dumped"
DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
EXP_FOLDER=$MY_DUMPED/"label_attention_concat"
mkdir -p $EXP_FOLDER

WORD_EMBEDS="/home/gridsan/groups/akyureklab/rfs-incremental/word_embeds"
BACKBONE_PATH="${SHARED_DUMPED}/backbones/c-x-concat"
BACKBONE_PATH="${BACKBONE_PATH}/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_lang-linear_multip_0.05_630620/resnet12_last.pth"

# cnt=0
# for LMBD in 0.4 0.5 0.6; do
# for TRLOSS in 0.65 0.7 0.75; do
# for LR in 0.004; do
# (( cnt++ ))
# if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
#     EXP_NAME=lambda_${LMBD}_trloss_${TRLOSS}_lr_${LR}
#     LOG_STDOUT="${EXP_FOLDER}/finetune_${EXP_NAME}.out"
#     LOG_STDERR="${EXP_FOLDER}/finetune_${EXP_NAME}.err"
#     python -u eval_incremental.py --model_path $BACKBONE_PATH \
#                        --data_root $DATA_PATH \
#                        --n_shots 5 \
#                        --eval_mode few-shot-language-incremental \
#                        --classifier lang-linear \
#                        --novel_epochs 20 \
#                        --learning_rate $LR \
#                        --freeze_backbone_at 1 \
#                        --attention "concat" \
#                        --use_episodes \
#                        --word_embed_path $WORD_EMBEDS \
#                        --lmbd_reg_transform_w $LMBD \
#                        --target_train_loss $TRLOSS > $LOG_STDOUT  2> $LOG_STDERR
# fi
# done
# done
# done



# For debugging.
python -u eval_incremental.py --model_path $BACKBONE_PATH \
                       --data_root $DATA_PATH \
                       --n_shots 5 \
                       --eval_mode few-shot-language-incremental \
                       --classifier lang-linear \
                       --novel_epochs 20 \
                       --learning_rate 0.004 \
                       --freeze_backbone_at 1 \
                       --attention "concat" \
                       --use_episodes \
                       --word_embed_path $WORD_EMBEDS \
                       --lmbd_reg_transform_w 0.4






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
