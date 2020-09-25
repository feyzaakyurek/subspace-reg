#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-8
#SBATCH --output=test_%A_%a.out
#SBATCH --error=test_%A_%a.err
#SBATCH --job-name=rfs-base
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

##export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# LINE=$(awk 'NR==$SLURM_ARRAY_TASK_ID' <<< "scripts/base_hyperparameters.txt")
FILE="scripts/base_hyperparameters.txt"

LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $FILE)
read -ra PARAMS<<< "$LINE"
LAYER="${PARAMS[0]}"
MULTIPFC="${PARAMS[1]}"
echo "*******LAYER: $LAYER MULTIPFC: $MULTIPFC *******"

# nvidia-smi

# LOG_STDOUT="$SLURM_JOB_ID.out"
# LOG_STDERR="$SLURM_JOB_ID.err"

# # Start (or restart) experiment
# date >> $LOG_STDOUT
# which python >> $LOG_STDOUT
# echo "---Beginning program ---" >> $LOG_STDOUT
# echo "Exp name     : rfs_base" >> $LOG_STDOUT
# echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT
# echo "SBATCH script: slurm_run.sh" >> $LOG_STDOUT
export WANDB_MODE=dryrun
python train_supervised.py --trial pretrain --model_path dumped --tb_path tb --data_root data --classifier description-linear --desc_embed_model bert-base-cased --prefix_label --multip_fc $MULTIPFC --transformer_layer $LAYER

