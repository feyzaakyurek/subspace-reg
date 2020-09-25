# Zero shot evaluation with biased lang classifier

# python eval_incremental.py --model_path dumped/resnet12_last_langclassifier_wbias.pth --data_root data --n_shots 5 --eval_mode zero-shot-novel-only --word_embed_size 500 --lang_classifier_bias

# Zero shot evaluation with non-biased lang classifier
# python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_exp0/resnet12_lastFalse.pth --data_root data --n_shots 5 --eval_mode zero-shot --word_embed_size 500 --classifier lang-linear >> zero-shot.log

# Zero-shot incremental evaluation with non-biased lang classifier
# python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_lastFalse.pth --data_root data --n_shots 5 --eval_mode zero-shot-incremental --classifier lang-linear --word_embed_size 500 --start_alpha 0.11 --end_alpha 0.15 --inc_alpha 0.01 > zero-shot-incremental-600.log

# Incremental few-shot (fine tuning) evaluation with non-biased lang classifier
# python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_lastFalse.pth --data_root data --n_shots 5 --eval_mode few-shot-language-incremental --classifier lang-linear --word_embed_size 500 --novel_epochs 14 --learning_rate 0.001 --num_novel_combs 10 --freeze_backbone --lmbd_reg_transform_w 0.01 --target_train_loss 0.6 > few-shot-language-incremental.log

# Incremental few-shot (fine tuning) evaluation with non-biased description classifier
python eval_incremental.py --model_path dumped/description_models/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --data_root data --n_shots 5 --eval_mode few-shot-language-incremental --classifier description-linear --novel_epochs 14 --learning_rate 0.001 --num_novel_combs 599 --freeze_backbone --lmbd_reg_transform_w 0.01 --target_train_loss 0.6

# Zero shot evaluation with non-biased lang classifier
# python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/ckpt_epoch_40.pth --data_root data --n_shots 5 --eval_mode zero-shot --classifier description-linear >> zero-shot.log

# Zero shot evaluation with non-biased description classifier
# python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_desc/resnet12_last.pth --data_root data --n_shots 5 --eval_mode zero-shot --classifier description-linear

# Few-shot-incremental
# python eval_incremental.py --model_path dumped/resnet12_miniImageNet_linear_classifier/resnet12_last.pth --data_root data --n_shots 5 --eval_mode few-shot-incremental --classifier linear > few-shot-incremental.log


# Few-shot evaluation
# CUDA_VISIBLE_DEVICES=0 python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_lastFalse.pth --data_root data --n_shots 5 --eval_mode few-shot --classifier lang-linear --word_embed_size 500 