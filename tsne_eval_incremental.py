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
from sklearn.manifold import TSNE

import pandas as pd
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

from eval.meta_eval import drop_a_dim #meta_test, incremental_test, zero_shot_test, zero_shot_incremental_test, few_shot_language_incremental_test, few_shot_finetune_incremental_test
from eval.cls_eval import incremental_validate
from util import create_and_save_embeds, restricted_float, create_and_save_descriptions


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
    parser.add_argument('--eval_mode', type=str,
                        choices=['few-shot', 'few-shot-incremental', 'zero-shot', 'few-shot-incremental-fine-tune',
                                 'zero-shot-incremental', 'few-shot-language-incremental'])

    parser.add_argument('--classifier', type=str,
                        choices=['linear', 'lang-linear', 'description-linear'])

    if parser.parse_known_args()[0].classifier in ["linear"]:
        parser.add_argument('--no_linear_bias', action='store_false', help='Use of bias in classifier.') #TODO
#     if parser.parse_known_args()[0].eval_mode in ["zero-shot-incremental","few-shot-incremental"]:

#         parser.add_argument("--start_alpha", type=restricted_float, default="0.7",
#                             help="Alpha is the fraction to multiply base scores with. Start is the beginning of the range to try.")
#         parser.add_argument("--end_alpha", type=restricted_float, default="0.8",
#                             help="Alpha is the fraction to multiply base scores with. End is the beginning of the range to try.")
#         parser.add_argument("--inc_alpha", type=restricted_float, default="0.01",
#                             help="Alpha is the fraction to multiply base scores with. Inc is increment.")

#     if parser.parse_known_args()[0].eval_mode in ["zero-shot-incremental",
#                                                   "few-shot-incremental",
#                                                   "few-shot-language-incremental",
#                                                   'few-shot-incremental-fine-tune']:
#         parser.add_argument("--neval_episodes", type=int, default=2000,
#                             help="Number of evaluation episodes both for base and novel.")


    if parser.parse_known_args()[0].classifier in ["lang-linear", "description-linear"]:
        parser.add_argument('--word_embed_size', type=int, default=None,
                            help='Word embedding classifier')
        parser.add_argument('--word_embed_path', type=str, default="word_embeds",
                            help='Where to store word embeds pickles for dataset.')

    if parser.parse_known_args()[0].classifier in ["lang-linear", "description-linear"]:
        parser.add_argument('--lang_classifier_bias', action='store_true',
                            help='Use of bias in lang classifier.')
        parser.add_argument('--multip_fc', type=float, default=0.05)

#     if parser.parse_known_args()[0].eval_mode in ['zero-shot-incremental']:
#         parser.add_argument('--num_novel_combs', type=int, default=0.05,
#                             help='Number of combinations of novel/test classes to evaluate base samples against:)')

#     if parser.parse_known_args()[0].eval_mode in ["few-shot-language-incremental", 'few-shot-incremental-fine-tune']:
#         parser.add_argument('--novel_epochs', type=int, default=15, help='number of epochs for novel support set.')
#         parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
#         parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
#         parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
#         parser.add_argument('--adam', action='store_true', help='use adam optimizer')
#         parser.add_argument('--freeze_backbone_at', type=int, default=1, help='freeze backbone while updating classifier at the epoch X, epochs start at 1.')
#         parser.add_argument('--lmbd_reg_transform_w',  type=float, default=None, help='learning rate')
#         parser.add_argument('--target_train_loss',  type=float, default=0.8, help='learning rate')
#         parser.add_argument('--saliency',  action='store_true', help='append label to the beginning description')

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

    print("************* Training arguments *************")
#     run.config.update(opt)
    args = opt
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("End of arguments.\n")



    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]

#         train_loader = DataLoader(ImageNet(args=opt, partition='train', transform=train_trans), #FIXME: use train
#                                   batch_size=64, shuffle=True, drop_last=True,
#                                   num_workers=opt.num_workers)

#         # load base evaluation dataset
#         base_val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
#                                      batch_size=opt.test_base_batch_size // 2,
#                                      shuffle=True,
#                                      drop_last=False,
#                                      num_workers=opt.num_workers // 2)

#         base_test_loader = DataLoader(ImageNet(args=opt, partition='test', transform=test_trans),
#                                       batch_size=opt.test_base_batch_size // 2,
#                                       shuffle=True,
#                                       drop_last=False,
#                                       num_workers=opt.num_workers // 2)
        
        base_meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=True,
                                                  pretrain=True),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)

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
                                    batch_size=opt.test_batch_size, shuffle=True, drop_last=False,
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

    vocab_train = [name for name in base_meta_testloader.dataset.label2human if name != '']
    vocab_test = [name for name in meta_testloader.dataset.label2human if name != '']
    vocab_val = [name for name in meta_valloader.dataset.label2human if name != '']
    vocab_all = vocab_train + vocab_test + vocab_val
    # load model
    if opt.classifier in ["lang-linear", "description-linear"]:
        # Save full dataset vocab if not available

        if opt.classifier == "description-linear":
            create_and_save_descriptions(opt, vocab_all)
        else:
            create_and_save_embeds(opt, vocab_all)
        vocab = vocab_train
    else:
        vocab = None

    
    ckpt = torch.load(opt.model_path)
#     if opt.classifier =="linear" and opt.no_linear_bias is None:
#         opt.no_linear_bias = ckpt['opt'].no_linear_bias
    model = create_model(opt.model, n_cls, opt, vocab=vocab, dataset=opt.dataset)
    print("Loading model...")
    model.load_state_dict(ckpt['model'], strict=False) #TODO

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # extract features for the first batches of novel and base
    random_batch_novel = 10 #np.random.choice(np.arange(len(meta_testloader)), 1, False)
    random_batch_base = 14 #np.random.choice(np.arange(len(base_meta_testloader)), 1, False)
    with torch.no_grad():
        for idx, data in enumerate(meta_testloader):
#             ipdb.set_trace()
            if idx == random_batch_novel:
                support_xs, support_ys, query_xs, novel_query_ys = drop_a_dim(data)
                query_xs = query_xs.cuda()
                feat_query, _ = model(query_xs.cuda(), is_feat=True)
                novel_query_features = feat_query[-1].view(query_xs.size(0), -1).detach().cpu().numpy()
                break

        for idx, data in enumerate(base_meta_testloader):
#             ipdb.set_trace()
            if idx == random_batch_base:
                support_xs, support_ys, query_xs, base_query_ys = drop_a_dim(data)
                query_xs = query_xs.cuda()
                feat_query, _ = model(query_xs.cuda(), is_feat=True)
                base_query_features = feat_query[-1].view(query_xs.size(0), -1).detach().cpu().numpy()
                break

        features = np.concatenate((novel_query_features, base_query_features), 0)
        embedded = TSNE().fit_transform(features)
        df = pd.DataFrame(embedded)
        df['labels'] = [vocab_all[i]+"_n" for i in novel_query_ys]+[vocab_all[i]+"_b" for i in base_query_ys]
        pth = os.path.join(os.path.split(opt.model_path)[0], "tsne_embedded.csv")
        print("Saved tsne embeds to {}.".format(pth))
        df.to_csv(pth)

if __name__ == '__main__':
    main()