#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-2 XXX
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=attreg_back_query_sizes



# Create the combinations of params for each array task,
# and save them to a temp params file.
# DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped/backbones/"
# DATA_PATH="/home/gridsan/groups/akyureklab/rfs-incremental/data"
# EXP_FOLDER=${DUMPED_PATH}/"attention_concat_smallkey"

# Create the combinations of params for each array task,
# and save them to a temp params file.
# DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped/backbones/c-x-concat/size_experiments"
# DATA_PATH="/home/gridsan/eakyurek/akyureklab_shared/rfs-incremental/data"
# WORD_EMBEDS="/home/gridsan/eakyurek/akyureklab_shared/rfs-incremental/word_embeds"


cnt=0
for MULTIPFC in 0.075 0.01; do
for DIAG_REG in 0.05 0.075; do
for QUERY_SIZE in 300 400 500 750; do
(( cnt++ ))
if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
#     echo "${MULTIPFC} ${DIAG_REG}"
    EXP_NAME=attention_concat_multipfc_${MULTIPFC}_diagreg_${DIAG_REG}_querysize_${QUERY_SIZE}
    EXP_FOLDER=$DUMPED_PATH/$EXP_NAME
    mkdir -p $EXP_FOLDER
    LOG_STDOUT="${DUMPED_PATH}/${EXP_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    python -u train_supervised.py --trial pretrain \
                                  --model_path $EXP_FOLDER  \
                                  --data_root data \
                                  --multip_fc $MULTIPFC \
                                  --data_root $DATA_PATH \
                                  --classifier lang-linear \
                                  --attention concat \
                                  --transform_query_size $QUERY_SIZE \
                                  --diag_reg $DIAG_REG \
                                  --word_embed_path $WORD_EMBEDS &> $LOG_STDOUT

fi
done
done
done

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
