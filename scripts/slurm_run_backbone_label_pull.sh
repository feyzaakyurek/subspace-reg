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
#SBATCH --job-name=bb_pull_nosyns




DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
BACKBONES_FOLDER=${DUMPED_PATH}/backbones/linear/label_pull
mkdir -p $BACKBONES_FOLDER

cnt=0
for PULL in 0.0; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
    EXP_NAME=no_synonyms_label_pull_${PULL}
    EXP_FOLDER=$BACKBONES_FOLDER/$EXP_NAME
    mkdir -p $EXP_FOLDER
    LOG_STDOUT="${EXP_FOLDER}/${EXP_NAME}.out"
    LOG_STDERR="${EXP_FOLDER}/${EXP_NAME}.err"
    python train_supervised.py --trial pretrain \
                               --model_path $EXP_FOLDER \
                               --data_root $DATA_PATH \
                               --label_pull $PULL \
                               --classifier linear > $LOG_STDOUT 2> $LOG_STDERR
                              
fi                
done



# EXP_NAME=bias_false
# EXP_FOLDER=$BACKBONES_FOLDER/$EXP_NAME
# mkdir -p $EXP_FOLDER
# LOG_STDOUT="${EXP_FOLDER}/log.out"
# python -u train_supervised.py --trial pretrain \
#                               --model_path $EXP_FOLDER  \
#                               --data_root data \
#                               --no_linear_bias \
#                               --data_root $DATA_PATH \
#                               --classifier linear &> $LOG_STDOUT
                              
                              
                              
                              
                              

# cnt=0
# for BIAS in 0.05 0.075 0.01; do
# (( cnt++ ))
# if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
#     EXP_NAME=multipfc_${MULTIPFC}
#     EXP_FOLDER=$BACKBONES_FOLDER/$EXP_NAME
#     mkdir -p $EXP_FOLDER
#     LOG_STDOUT="${DUMPED_PATH}/${EXP_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
#     python -u train_supervised.py --trial pretrain \
#                                   --model_path $EXP_FOLDER  \
#                                   --data_root data \
#                                   --data_root $DATA_PATH \
#                                   --classifier linear &> $LOG_STDOUT

# fi
# done

# --word_embed_type ".random" \


#
#
# # Read the SLURM_ARRAY_TASK_ID line from the params file.
# LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $FILE)
# read -ra PARAMS<<< "$LINE"
# MULTIPFC="${PARAMS[0]}"
# DIAG_REG="${PARAMS[1]}"
#

# Debugging
# # Concat with Reg.
# export DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped"
# export DATA_PATH="/home/gridsan/eakyurek/akyureklab_shared/rfs-incremental/data"
# export WORD_EMBEDS="/home/gridsan/eakyurek/akyureklab_shared/rfs-incremental/word_embeds"
# python -u train_supervised.py --trial pretrain \
#                                        --model_path $DUMPED_PATH  \
#                                        --data_root data \
#                                        --multip_fc 0.05 \
#                                        --data_root $DATA_PATH \
#                                        --classifier lang-linear \
#                                        --attention concat \
#                                        --diag_reg 0.05 \
#                                        --word_embed_path $WORD_EMBEDS
# # Context
# export DUMPED_PATH="/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/dumped/backbones/"
# export DATA_PATH="/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/data"
# export WORD_EMBEDS="/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/word_embeds"
# python train_supervised.py --trial pretrain \
#                             --model_path $DUMPED_PATH \
#                             --tb_path tb \
#                             --multip_fc 1.0 \
#                             --data_root $DATA_PATH \
#                             --classifier lang-linear \
#                             --attention context \
#                             --word_embed_path $WORD_EMBEDS
