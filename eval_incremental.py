# python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --data_root data --n_shots 5 --incremental_eval

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
from dataset.cifar import MetaCIFAR100
from dataset.transform_cfg import transforms_test_options, transforms_list

from eval.meta_eval import meta_test, incremental_test, zero_shot_test, zero_shot_incremental_test, \
few_shot_language_incremental_test, few_shot_finetune_incremental_test, few_shot_language_pretrain_linear_tune
from eval.cls_eval import incremental_validate
from util import create_and_save_embeds, restricted_float, create_and_save_descriptions, create_and_save_synonyms


# import wandb
# run = wandb.init(project="lil")

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # specify data_root
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=5, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')
    parser.add_argument('--test_base_batch_size', type=int, default=50, metavar='test_batch_size',
                        help='Size of test batch)')
    parser.add_argument('--set_seed', type=int, default=5, 
                        help='Seed for torch and np.')
    parser.add_argument('--eval_mode', type=str,
                        choices=["few-shot", 
                                 "few-shot-incremental", 
                                 "zero-shot", 
                                 "few-shot-incremental-fine-tune",
                                 "zero-shot-incremental", 
                                 "few-shot-language-incremental", 
                                 "few-shot-incremental-language-pretrain-linear-tune"])

    parser.add_argument('--classifier', type=str,
                        choices=['linear', 'lang-linear', 'description-linear'])
    parser.add_argument('--track_weights', action='store_true',
                            help='Save the classifier weights to a csv file.')
    parser.add_argument('--track_label_inspired_weights', action='store_true',
                            help='Save the label inspired weights to a csv file.')
    parser.add_argument('--save_preds_0', action='store_true', help='Save predictions for the first episode.' ) # TODO: This may not be available for every evalmode
    parser.add_argument('--use_synonyms', action='store_true', help='Use synonyms.') # TODO

    if parser.parse_known_args()[0].eval_mode in ["zero-shot-incremental","few-shot-incremental"]:

        parser.add_argument("--start_alpha", type=restricted_float, default="0.7",
                            help="Alpha is the fraction to multiply base scores with. Start is the beginning of the range to try.")
        parser.add_argument("--end_alpha", type=restricted_float, default="0.8",
                            help="Alpha is the fraction to multiply base scores with. End is the beginning of the range to try.")
        parser.add_argument("--inc_alpha", type=restricted_float, default="0.01",
                            help="Alpha is the fraction to multiply base scores with. Inc is increment.")

    if parser.parse_known_args()[0].eval_mode in ["zero-shot-incremental",
                                                  "few-shot-incremental",
                                                  "few-shot-language-incremental",
                                                  "few-shot-incremental-fine-tune",
                                                  "few-shot-incremental-language-pretrain-linear-tune"]:
        parser.add_argument("--neval_episodes", type=int, default=2000,
                            help="Number of evaluation episodes both for base and novel.")


    if parser.parse_known_args()[0].classifier in ["lang-linear", "description-linear"]:
        parser.add_argument('--word_embed_size', type=int, default=500,
                            help='Word embedding classifier')
        parser.add_argument('--word_embed_path', type=str, default="word_embeds",
                            help='Where to store word embeds pickles for dataset.')
        parser.add_argument('--lang_classifier_bias', action='store_true',
                            help='Use of bias in lang classifier.')
        parser.add_argument('--multip_fc', type=float, default=0.05)
        parser.add_argument('--attention', type=str, choices=["sum","concat","context"], default=None, help='Use of attention in lang classifier.')
        parser.add_argument('--orig_alpha', type=float, default=1.0)
        
    if parser.parse_known_args()[0].eval_mode in ["few-shot-incremental-fine-tune"]:
        parser.add_argument('--word_embed_size', type=int, default=500,
                            help='Word embedding classifier')
        parser.add_argument('--word_embed_path', type=str, default="word_embeds",
                            help='Where to store word embeds pickles for dataset.')
        parser.add_argument('--glove', action='store_true',
                            help='Use of Glove embeds instead of Vico.')
        parser.add_argument('--label_pull', type=float, default=None)
        
        if parser.parse_known_args()[0].label_pull is not None:
            parser.add_argument('--pulling', type=str, default="regularize",
                            help='How should we leverage label inspired weights?')

    if parser.parse_known_args()[0].eval_mode in ['zero-shot-incremental']:
        parser.add_argument('--num_novel_combs', type=int, default=0.05,
                            help='Number of combinations of novel/test classes to evaluate base samples against:)')

    if parser.parse_known_args()[0].eval_mode in ["few-shot-language-incremental", 
                                                  "few-shot-incremental-fine-tune",
                                                  "few_shot_language_pretrain_linear_tune",
                                                  "few-shot-incremental-language-pretrain-linear-tune"]:
        parser.add_argument('--novel_epochs', type=int, default=15, help='number of epochs for novel support set.')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--adam', action='store_true', help='use adam optimizer')
        parser.add_argument('--freeze_backbone_at', type=int, default=1, help='freeze backbone while updating classifier at the epoch X, epochs start at 1.')
        parser.add_argument('--lmbd_reg_transform_w',  type=float, default=None, help='regularization for the base classes.')
        parser.add_argument('--target_train_loss',  type=float, default=0.8, help='learning rate')
        parser.add_argument('--saliency',  action='store_true', help='append label to the beginning description')
        parser.add_argument('--use_episodes', type=bool, default=False, help='use exact XtarNet episodes.')

    if parser.parse_known_args()[0].classifier in ["description-linear"]:
        parser.add_argument('--description_embed_path', type=str, default="description_embeds")
        parser.add_argument('--desc_embed_model', type=str, default="bert-base-cased")
        parser.add_argument('--transformer_layer', type=str, default=6)
        parser.add_argument('--prefix_label', action='store_true', help='append label to the beginning description')

    opt = parser.parse_args()

    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
        opt.data_aug = True

    return opt


def main():

    opt = parse_option()

    # Add git commit hash
    process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=subprocess.PIPE)
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
        if opt.use_episodes:
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
            train_loader = base_val_loader
            
        else:
            
            train_loader = DataLoader(ImageNet(args=opt, partition='train', transform=train_trans), #FIXME: use train
                                  batch_size=64, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)        
            base_val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                         batch_size=opt.test_base_batch_size // 2,
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=opt.num_workers // 2)

            base_test_loader = DataLoader(ImageNet(args=opt, partition='test', transform=test_trans),
                                          batch_size=opt.test_base_batch_size // 2,
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=opt.num_workers // 2)

        

        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=True),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=True),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]
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
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_test_options['D']
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    else:
        raise NotImplementedError(opt.dataset)


    # If language classifiers are used, then we'll need the embeds recorded.
    if opt.classifier in ["lang-linear", "description-linear"]:
        # Save full dataset vocab if not available
        vocab_train = [name for name in train_loader.dataset.label2human if name != '']
        vocab_test = [name for name in meta_testloader.dataset.label2human if name != '']
        vocab_val = [name for name in meta_valloader.dataset.label2human if name != '']
        vocab_all = vocab_train + vocab_test + vocab_val
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
        vocab_train = [name for name in train_loader.dataset.label2human if name != '']
        vocab_test = [name for name in meta_testloader.dataset.label2human if name != '']
        vocab_val = [name for name in meta_valloader.dataset.label2human if name != '']
        vocab_all = vocab_train + vocab_test + vocab_val
        create_and_save_embeds(opt, vocab_all)
        
        if opt.use_synonyms:
            create_and_save_synonyms(opt, vocab_train, vocab_test, vocab_val)
    
    # Load model if available, check bias. 
    ckpt = torch.load(opt.model_path)
    if opt.classifier =="linear":
        try:
            bias = ckpt['model']['classifier.bias']
            opt.linear_bias = bias is not None
        except:
            print("!!!! Setting bias to true!")
            opt.linear_bias = False 
            
    model = create_model(opt.model, n_cls, opt, vocab=vocab, dataset=opt.dataset)
    print("Loading model...")
    try:
        model.load_state_dict(ckpt['model'])
    except Exception as e:
        print(e)
        model.load_state_dict(ckpt['model'], strict=False)
        
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
#            'val_acc_novel_avg': novel[0],
#            'val_acc_base_avg': base[0],
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
#            'test_acc_novel_avg': novel[0],
#            'test_acc_base_avg': base[0],
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
        novel, base = few_shot_finetune_incremental_test(model,
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
        
        if opt.save_preds_0:
            df = few_shot_finetune_incremental_test(model,
                                                    ckpt,
                                                    criterion,
                                                    meta_valloader,
                                                    base_val_loader,
                                                    opt,
                                                    vis=True)
            df.to_csv(f"vis_{opt.eval_mode}_pulling_{opt.pulling}_{opt.label_pull}_target_loss_{opt.target_train_loss}_synonyms_{opt.use_synonyms}.csv", index=False)
        
        if not opt.track_weights and not opt.track_label_inspired_weights:
            start = time.time()
            opt.split = "test" # TODO: run only for best val.

            novel, base = few_shot_finetune_incremental_test(model,
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

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
