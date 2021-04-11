from __future__ import print_function
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import argparse
import socket
import time
import ipdb
import os
import pickle
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import MetaImageNet, ImageNet
from dataset.tiered_imagenet import MetaTieredImageNet
from dataset.transform_cfg import transforms_test_options, transforms_list

from util import create_and_save_embeds, restricted_float, create_and_save_descriptions, create_and_save_synonyms
from eval.meta_eval import incremental_test, meta_hierarchical_incremental_test
from eval.zero_eval import zero_shot_test, zero_shot_incremental_test
from eval.language_eval import few_shot_language_incremental_test, few_shot_finetune_incremental_test
# import wandb
# print("!!!!! WANDB_MODE: ", os.environ['WANDB_MODE'])
# run = wandb.init(project="lil", mode="offline")
from configs import parse_option_eval

# torch.multiprocessing.set_sharing_strategy('file_system')
def main():

    opt = parse_option_eval()

    # Add git commit hash
    process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'], 
                               shell=False, 
                               stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    opt.git_head_hash = git_head_hash.decode()

    # Set seeds
    torch.manual_seed(opt.set_seed)
    np.random.seed(opt.set_seed)

    print("************* Training arguments *************")
#     run.config.update(opt)
    args = opt
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("End of arguments.\n")



    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]



        # load base evaluation dataset
#         if opt.use_episodes:
#             base_test_loader = DataLoader(MetaImageNet(args=opt, partition='test',
#                                                   train_transform=train_trans,
#                                                   test_transform=test_trans,
#                                                   fix_seed=True,
#                                                   pretrain=True),
#                                      batch_size=opt.test_batch_size, shuffle=True, drop_last=False,
#                                      num_workers=opt.num_workers)

#             base_val_loader = DataLoader(MetaImageNet(args=opt, partition='val',
#                                                       train_transform=train_trans,
#                                                       test_transform=test_trans,
#                                                       fix_seed=True,
#                                                       pretrain=True),
#                                          batch_size=opt.test_batch_size, shuffle=True, drop_last=False,
#                                          num_workers=opt.num_workers)
#             train_loader = base_val_loader

#         else:

#             train_loader = DataLoader(ImageNet(args=opt, partition='train', transform=train_trans), #FIXME: use train
#                                   batch_size=64, shuffle=True, drop_last=True,
#                                   num_workers=opt.num_workers)
#             base_val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
#                                          batch_size=opt.test_base_batch_size // 2,
#                                          shuffle=True,
#                                          drop_last=False,
#                                          num_workers=opt.num_workers // 2)

        base_test_loader = DataLoader(ImageNet(args=opt, split='train', phase='test', transform=test_trans),
                                      batch_size=opt.test_base_batch_size // 2,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=opt.num_workers // 2)

        base_support_loader = None
        if opt.n_base_support_samples > 0:
            ''' We'll use support samples from base classes. '''
            base_support_loader = DataLoader(MetaImageNet(args=opt, split='train', phase='train',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=True, use_episodes=False),
                                     batch_size=opt.test_batch_size, shuffle=True, drop_last=False, # False?
                                     num_workers=opt.num_workers)

#         base_test_loader = DataLoader(MetaImageNet(args=opt, split='train', phase='test',
#                                               train_transform=train_trans,
#                                               test_transform=test_trans,
#                                               fix_seed=True, use_episodes=opt.use_episodes),
#                                  batch_size=opt.test_batch_size, shuffle=True, drop_last=False,
#                                  num_workers=opt.num_workers)

#         base_val_loader = DataLoader(MetaImageNet(args=opt, split='train', phase='val',
#                                                   train_transform=train_trans,
#                                                   test_transform=test_trans,
#                                                   fix_seed=True, use_episodes=opt.use_episodes),
#                                      batch_size=opt.test_batch_size, shuffle=True, drop_last=False,
#                                      num_workers=opt.num_workers)

#         meta_testloader = DataLoader(MetaImageNet(args=opt, split='test',
#                                                   train_transform=train_trans,
#                                                   test_transform=test_trans,
#                                                   fix_seed=True, use_episodes=opt.use_episodes),
#                                      batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                      num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, split='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=True, use_episodes=opt.use_episodes, disjoint_classes=True),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 60
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]


        base_test_loader = DataLoader(MetaImageNet(args=opt, partition='test',
                                      train_transform=train_trans,
                                      test_transform=test_trans,
                                      fix_seed=True,
                                      pretrain=True),
                         batch_size=opt.test_batch_size, shuffle=True, drop_last=False,
                         num_workers=opt.num_workers)

        base_val_loader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=True,
                                                  pretrain=True),
                                     batch_size=opt.test_batch_size, shuffle=True, drop_last=False,
                                     num_workers=opt.num_workers)


        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans,
                                                       fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    else:
        raise NotImplementedError(opt.dataset)

    # Load model if available, check bias.
    ckpt = torch.load(opt.model_path)
    
    # If language classifiers are used, then we'll need the embeds recorded.
    if opt.classifier in ["lang-linear", "description-linear"]:
        opt.multip_fc = ckpt['opt'].multip_fc
        if opt.diag_reg is None:
            opt.diag_reg = ckpt['opt'].diag_reg
        opt.lang_classifier_bias = ckpt['opt'].lang_classifier_bias


        # Save full dataset vocab if not available
        vocab_train = [name for name in base_test_loader.dataset.label2human if name != '']
        vocab_test = [name for name in meta_testloader.dataset.label2human if name != '']
        vocab_val = [name for name in meta_valloader.dataset.label2human if name != '']
        vocab_all = vocab_train + vocab_val + vocab_test 
        if opt.classifier == "description-linear":
            create_and_save_descriptions(opt, vocab_all)
        else:
            create_and_save_embeds(opt, vocab_all)
        vocab = vocab_train
    else:
        vocab = None

    # This is another scenario in which we'll need the embeds.
    if opt.label_pull is not None:
#         ipdb.set_trace()
        vocab_train = [name for name in base_test_loader.dataset.label2human if name != '']
#         vocab_test = [name for name in meta_testloader.dataset.label2human if name != '']
        vocab_val = [name for name in meta_valloader.dataset.label2human if name != '']
        vocab_all = vocab_train + vocab_val # + vocab_test
        create_and_save_embeds(opt, vocab_all)

        if opt.use_synonyms:
            create_and_save_synonyms(opt, vocab_train, vocab_test, vocab_val)

    if opt.classifier =="linear":
        if 'classifier.bias' in ckpt['model'].keys():
            if ckpt['model']['classifier.bias'] is None:
                raise ValueError()
            opt.linear_bias = True
        else:
            opt.linear_bias = False


    model = create_model(opt.model, n_cls, opt, vocab=vocab, dataset=opt.dataset)
    print("Loading model...")
#     try:
    model.load_state_dict(ckpt['model'])
#     except Exception as e:
#         print(e)
#         model.load_state_dict(ckpt['model'], strict=False)

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True


#     run.watch(model)

    # evaluation

    if opt.eval_mode == "few-shot-incremental":
        best_alpha = opt.start_alpha
        best_score = 0.0
        for alpha in np.arange(opt.start_alpha,opt.end_alpha,opt.inc_alpha):
            start = time.time()
            novel, base = incremental_test(model, meta_valloader, base_val_loader,
                                               alpha, use_logit=True, is_norm=True)

            val_time = time.time() - start
            avg_score = (base[0]+novel[0])/2
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha
            print('alpha: ', alpha)
            print('val_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], val_time))
            print('val_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base[0], base[1], val_time))
            print('average: {:.4f}'.format((base[0]+novel[0])/2))

        start = time.time()
        novel, base = incremental_test(model, meta_testloader, base_test_loader, best_alpha, use_logit=True, is_norm=True)
        test_time = time.time() - start
        avg_test_score = (base[0]+novel[0])/2
        print('test_alpha: {0}'.format(best_alpha))
        print('test_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], test_time))
        print('test_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base[0], base[1], test_time))
        print('average: {:.4f}'.format((base[0]+novel[0])/2))
        df = incremental_test(model, meta_testloader, base_test_loader, best_alpha, use_logit=True, is_norm=True, vis=True)
        df.to_csv("inc_results.csv", index=False)


    elif opt.eval_mode == "zero-shot":
        assert opt.classifier in ["lang-linear", "description-linear"]
        start = time.time()
        novel = zero_shot_test(model, meta_valloader, opt, is_norm=False)
        val_time = time.time() - start
        print('val_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], val_time))
        print('val_score: {:.4f}'.format(novel[0]))

        start = time.time()
        novel = zero_shot_test(model, meta_testloader, opt, is_norm=False)
        test_time = time.time() - start
        print('test_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], test_time))
        print('test_score: {:.4f}'.format(novel[0]))

    elif opt.eval_mode == "zero-shot-incremental":
        assert opt.classifier == "lang-linear"
        best_alpha = 0.14 #opt.start_alpha
        best_score = 0.0
        for alpha in np.arange(opt.start_alpha,opt.end_alpha,opt.inc_alpha):
            start = time.time()
            novel, base = zero_shot_incremental_test(model, meta_valloader, base_val_loader, opt, alpha)
            val_time = time.time() - start
            avg_score = (base[0]+novel[0])/2
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha
            print('alpha: ', alpha)
            print('val_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], val_time))
            print('val_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base[0], base[1], val_time))
            print('average: {:.4f}'.format(avg_score))

        start = time.time()
        novel, base = zero_shot_incremental_test(model, meta_testloader, base_test_loader, opt, best_alpha)
        test_time = time.time() - start
        avg_score = (base[0]+novel[0])/2
        print('test_alpha: {0}'.format(best_alpha))
        print('test_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], test_time))
        print('test_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base[0], base[1], test_time))
        print('average: {:.4f}'.format(avg_score))


    elif opt.eval_mode == "few-shot-language-incremental":
        assert opt.classifier in ["lang-linear", "description-linear"]

        criterion = nn.CrossEntropyLoss()
        start = time.time()
        opt.split = "val"
        novel, base = few_shot_language_incremental_test(model,
                                                         ckpt,
                                                         criterion,
                                                         meta_valloader,
                                                         base_val_loader,
                                                         opt)
        val_time = time.time() - start
        avg_score = (base[0]+novel[0])/2
        print('val_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], val_time))
        print('val_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base[0], base[1], val_time))
        print('val_acc_average: {:.4f}'.format(avg_score))
#         run.log({
#            'val_acc_novel_avg': novel,
#            'val_acc_base_avg': base,
#            'val_acc_avg_both': avg_score})

        start = time.time()
        opt.split = "test"
        novel, base = few_shot_language_incremental_test(model,
                                                         ckpt,
                                                         criterion,
                                                         meta_testloader,
                                                         base_test_loader,
                                                         opt)
        test_time = time.time() - start
        avg_score = (base[0]+novel[0])/2
        print('test_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], test_time))
        print('test_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base[0], base[1], test_time))
        print('test_acc_average: {:.4f}'.format(avg_score))
#         run.log({
#            'test_acc_novel_avg': novel,
#            'test_acc_base_avg': base,
#            'test_acc_avg_both': avg_score})
    elif opt.eval_mode == "few-shot-incremental-language-pretrain-linear-tune":
        assert opt.classifier in ["lang-linear", "description-linear"]

        criterion = nn.CrossEntropyLoss()
        start = time.time()
        opt.split = "val"
        novel, base = few_shot_language_pretrain_linear_tune(model,
                                                         ckpt,
                                                         criterion,
                                                         meta_valloader,
                                                         base_val_loader,
                                                         opt)
        val_time = time.time() - start
        avg_score = (base[0]+novel[0])/2
        print('val_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], val_time))
        print('val_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base[0], base[1], val_time))
        print('val_acc_average: {:.4f}'.format(avg_score))
#         run.log({
#            'val_acc_novel_avg': novel[0],
#            'val_acc_base_avg': base[0],
#            'val_acc_avg_both': avg_score})

        start = time.time()
        opt.split = "test"
        novel, base = few_shot_language_pretrain_linear_tune(model,
                                                         ckpt,
                                                         criterion,
                                                         meta_testloader,
                                                         base_test_loader,
                                                         opt)
        test_time = time.time() - start
        avg_score = (base[0]+novel[0])/2
        print('test_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], test_time))
        print('test_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base[0], base[1], test_time))
        print('test_acc_average: {:.4f}'.format(avg_score))

    elif opt.eval_mode == 'few-shot-incremental-fine-tune':
        assert opt.classifier == "linear"
        criterion = nn.CrossEntropyLoss()


        start = time.time()
        opt.split = "val"
        original_nepisodes = opt.neval_episodes
        opt.neval_episodes = 300
        novel, base = few_shot_finetune_incremental_test(model,
                                                         ckpt,
                                                         criterion,
                                                         meta_valloader,
                                                         base_test_loader,
                                                         opt,
                                                         base_support_loader=base_support_loader)
        val_time = time.time() - start
        avg_score = (base+novel)/2
        print('val_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel, 0, val_time))
        print('val_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base, 0, val_time))
        print('val_acc_average: {:.4f}'.format(avg_score))

#         if opt.save_preds_0:
#             df = few_shot_finetune_incremental_test(model,
#                                                     ckpt,
#                                                     criterion,
#                                                     meta_valloader,
#                                                     base_val_loader,
#                                                     opt,
#                                                     vis=True)
#             df.to_csv(f"vis_{opt.eval_mode}_pulling_{opt.pulling}_{opt.label_pull}_target_loss_{opt.target_train_loss}_synonyms_{opt.use_synonyms}.csv", index=False)

#         if not opt.track_weights and not opt.track_label_inspired_weights:
#             start = time.time()
#             opt.split = "test" # TODO: run only for best val.
#             opt.neval_episodes = original_nepisodes
#             novel, base = few_shot_finetune_incremental_test(model,
#                                                              ckpt,
#                                                              criterion,
#                                                              meta_testloader,
#                                                              base_test_loader,
#                                                              opt)
#             test_time = time.time() - start
#             avg_score = (base+novel)/2
#             print('test_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel, 0, test_time))
#             print('test_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base, 0, test_time))
#             print('test_acc_average: {:.4f}'.format(avg_score))

    elif opt.eval_mode == "few-shot":
        start = time.time()
        val_acc, val_std = meta_test(model, meta_valloader)
        val_time = time.time() - start
        print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, val_std, val_time))

        start = time.time()
        val_acc_feat, val_std_feat = meta_test(model, meta_valloader, use_logit=False)
        val_time = time.time() - start
        print('val_acc_feat: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc_feat, val_std_feat, val_time))

        start = time.time()
        test_acc, test_std = meta_test(model, meta_testloader)
        test_time = time.time() - start
        print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std, test_time))

        start = time.time()
        test_acc_feat, test_std_feat = meta_test(model, meta_testloader, use_logit=False)
        test_time = time.time() - start
        print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat, test_std_feat, test_time))


    elif opt.eval_mode == "hierarchical-incremental-few-shot":
        start = time.time()
        val_acc = meta_hierarchical_incremental_test(model, meta_testloader, base_test_loader, opt)
        val_time = time.time() - start
        print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, 0, val_time))




    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
