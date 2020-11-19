#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-9
#SBATCH --output=%A_%a.out
#SBATCH --error=s%A_%a.err
#SBATCH --job-name=att_reg_back

# Create the combinations of params for each array task,
# and save them to a temp params file.
export DUMPED_PATH="/home/gridsan/eakyurek/gitother/rfs-incremental/dumped/backbone_finetune_random"
export DATA_PATH="/home/gridsan/eakyurek/akyureklab_shared/rfs-incremental/data"
export WORD_EMBEDS="/home/gridsan/eakyurek/akyureklab_shared/rfs-incremental/word_embeds"

# FILE="$DUMPED_PATH/${SLURM_ARRAY_TASK_ID}_temp_hyperparameters.txt"
# rm $FILE

cnt=0
for MULTIPFC in 0.075 0.01; do
    for DIAG_REG in 0.05 0.075; do
        (( cnt++ ))
        if [[ $cnt -eq $SLURM_ARRAY_TASK_ID ]]; then
            echo "${MULTIPFC} ${DIAG_REG}"
            # Create log files
            export LOG_STDOUT="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
#             export LOG_STDERR="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

            python -u train_supervised.py --trial pretrain \
                                       --model_path $DUMPED_PATH  \
                                       --data_root data \
                                       --multip_fc $MULTIPFC \
                                       --data_root $DATA_PATH \
                                       --classifier lang-linear \
                                       --attention concat \
                                       --diag_reg $DIAG_REG \
                                       --word_embed_type ".random" \
                                       --word_embed_path $WORD_EMBEDS &> $LOG_STDOUT # 2> $LOG_STDERR
            break
        fi
    done
done

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
