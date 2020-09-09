# Zero shot evaluation with biased lang classifier
# python eval_incremental.py --model_path dumped/resnet12_last_langclassifier_wbias.pth --data_root data --n_shots 5 --eval_mode zero-shot-novel-only --word_embed_size 500 --lang_classifier_bias

# Zero shot evaluation with non-biased lang classifier
# python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_exp0/resnet12_lastFalse.pth --data_root data --n_shots 5 --eval_mode zero-shot-novel-only --word_embed_size 500

# Incremental evaluation with non-biased lang classifier
# python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_lastFalse.pth --data_root data --n_shots 5 --eval_mode zero-shot-incremental --classifier lang-linear --word_embed_size 500 --start_alpha 0.12 --end_alpha 0.25 --inc_alpha 0.02 --num_novel_combs 10

# Incremental evaluation with non-biased lang classifier
CUDA_VISIBLE_DEVICES=0 python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_lastFalse.pth --data_root data --n_shots 5 --eval_mode few-shot-language-incremental --classifier lang-linear --word_embed_size 500 

# Few-shot evaluation
# CUDA_VISIBLE_DEVICES=0 python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_lastFalse.pth --data_root data --n_shots 5 --eval_mode few-shot --classifier lang-linear --word_embed_size 500 