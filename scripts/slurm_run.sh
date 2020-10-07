#!/bin/bash
#SBATCH --constraint=xeon-g6
#SBATCH --time=15-00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-6
#SBATCH --output=dumped/%A_%a.out
#SBATCH --error=dumped/%A_%a.err
#SBATCH --job-name=lil



# Create the combinations of params for each array task,
# and save them to a temp params file.
DUMPED_PATH="/home/gridsan/akyurek/git/rfs-incremental/dumped"
FILE="$DUMPED_PATH/${SLURM_ARRAY_TASK_ID}_temp_hyperparameters.txt"
rm $FILE 

for LAYER in 6 8; do
    for TRLOSS in 0.6 0.7 0.8; do
        for MULTIPFC in 0.1; do
            echo "${LAYER} ${TRLOSS} ${MULTIPFC}" >> $FILE
        done
    done
done


# Read the SLURM_ARRAY_TASK_ID line from the params file.
LINE=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $FILE) 
read -ra PARAMS<<< "$LINE"
LAYER="${PARAMS[0]}"
TRLOSS="${PARAMS[1]}"
MULTIPFC="${PARAMS[2]}"


# Create log files.sc
LOG_STDOUT="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
LOG_STDERR="${DUMPED_PATH}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
BACKBONE_PATH="${DUMPED_PATH}/ekin_dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_layer_${LAYER}_multip_${MULTIPFC}/resnet12_last.pth"

python eval_incremental.py --model_path $BACKBONE_PATH \
                           --data_root data \
                           --n_shots 1 \
                           --eval_mode few-shot-language-incremental \
                           --classifier description-linear \
                           --novel_epochs 5 \
                           --learning_rate 0.01 \
                           --freeze_backbone_at 1 \
                           --lmbd_reg_transform_w 0.01 \
                           --target_train_loss $TRLOSS \
                           --prefix_label \
                           --multip_fc $MULTIPFC \
                           --transformer_layer $LAYER > $LOG_STDOUT 2> $LOG_STDERR
                           
                           

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

