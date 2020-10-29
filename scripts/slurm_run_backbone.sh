#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-2
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=lil-backbone-context



# Create the combinations of params for each array task,
# and save them to a temp params file.
export DUMPED_PATH="/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/dumped/backbones/"
export DATA_PATH="/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/data"
export WORD_EMBEDS="/home/gridsan/akyurek/akyureklab_shared/rfs-incremental/word_embeds"

FILE="$DUMPED_PATH/${SLURM_ARRAY_TASK_ID}_temp_hyperparameters.txt"
rm $FILE 


for MULTIPFC in 0.05 0.1; do
    echo "${MULTIPFC}" >> $FILE
done



# Read the SLURM_ARRAY_TASK_ID line from the params file.
LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $FILE) 
read -ra PARAMS<<< "$LINE"

MULTIPFC="${PARAMS[0]}"


# Create log files
export LOG_STDOUT="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
export LOG_STDERR="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

python train_supervised.py --trial pretrain \
                           --model_path $DUMPED_PATH  \
                           --data_root data \
                           --multip_fc $MULTIPFC \
                           --data_root $DATA_PATH \
                           --classifier lang-linear \
                           --attention context \
                           --word_embed_path $WORD_EMBEDS > $LOG_STDOUT 2> $LOG_STDERR
                           
# Debugging                           
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
                           
