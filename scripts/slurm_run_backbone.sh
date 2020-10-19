#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=lil-backbone-no-bias



# Create the combinations of params for each array task,
# and save them to a temp params file.
export DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped/backbones/"
# FILE="$DUMPED_PATH/${SLURM_ARRAY_TASK_ID}_temp_hyperparameters.txt"
# rm $FILE 


# for MULTIPFC in 0.1 0.2 0.3; do
#     echo "${MULTIPFC}" >> $FILE
# done



# Read the SLURM_ARRAY_TASK_ID line from the params file.
# LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $FILE) 
# read -ra PARAMS<<< "$LINE"

# MULTIPFC="${PARAMS[0]}"


# Create log files
export LOG_STDOUT="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
export LOG_STDERR="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# python train_supervised.py --trial pretrain \
#                            --model_path dumped/backbones/label  \
#                            --data_root data \
#                            --multip_fc $MULTIPFC \
#                            --classifier lang-linear \
#                            --word_embed_size 500 > $LOG_STDOUT 2> $LOG_STDERR
export DUMPED_PATH="/afs/csail.mit.edu/u/a/akyurek/akyurek/feyza/git/rfs-incremental/dumped/backbones/c-x-concat"
export DATA_PATH="/afs/csail.mit.edu/u/a/akyurek/akyurek/git/rfs-incremental/data"
export WORD_EMBEDS="/afs/csail.mit.edu/u/a/akyurek/akyurek/feyza/git/rfs-incremental/word_embeds"
CUDA_VISIBLE_DEVICES=14 python train_supervised.py --trial pretrain \
                            --model_path $DUMPED_PATH \
                            --tb_path tb \
                            --data_root $DATA_PATH \
                            --classifier lang-linear \
                            --attention \
                            --word_embed_path $WORD_EMBEDS \
                            --no_linear_bias > $LOG_STDOUT 2> $LOG_STDERR
                           
