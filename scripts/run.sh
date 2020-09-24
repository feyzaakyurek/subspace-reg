# ======================
# exampler commands on miniImageNet
# ======================

# supervised pre-training
# python train_supervised.py --trial pretrain --model_path dumped --tb_path tb --data_root data  

# distillation
# setting '-a 1.0' should give simimlar performance
#python train_distillation.py -r 0.5 -a 0.5 --path_t /path/to/teacher.pth --trial born1 --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

# evaluation
# python eval_fewshot.py --model_path /path/to/student.pth --data_root /path/to/data_root

# supervised pretraining with class labels
# python train_supervised.py --trial pretrain --model_path dumped  --data_root data --classifier lang-linear --word_embed_size 500

# supervised pretraining with class wordnet descriptions,  wandb/dryrun-20200922_202406-1ngu2zhc with prefixes
python train_supervised.py --trial pretrain --model_path dumped --tb_path tb --data_root data --classifier description-linear --desc_embed_model bert-base-cased --prefix_label --multip_fc 0.15 > train_supervised_desc_prefix_multipfc015.qlog 2> train_supervised_desc_prefix_multipfc015.err