import argparse
from dataset.transform_cfg import transforms_options, transforms_list
from models import model_pool

import os
import torch
import subprocess


def parse_option_eval():
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
    parser.add_argument('--n_test_runs', type=int, default=2000, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=5, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--n_base_aug_support_samples', default=0, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--n_base_support_samples', default=0, type=int,
                        help='The number of support base samples per base class.')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')
    parser.add_argument('--test_base_batch_size', type=int, default=50, metavar='test_batch_size',
                        help='Size of test batch)')
    parser.add_argument('--set_seed', type=int, default=5,
                        help='Seed for torch and np.')
    parser.add_argument('--eval_mode', type=str,
                        choices=["few-shot-incremental-fine-tune"])
    parser.add_argument('--classifier', type=str,
                        choices=['linear', 'lang-linear', 'description-linear'])
    parser.add_argument('--verbose', action='store_true',
                            help='Print novel epochs..')
    parser.add_argument('--track_weights', action='store_true',
                            help='Save the classifier weights to a csv file.')
    parser.add_argument('--track_label_inspired_weights', action='store_true',
                            help='Save the label inspired weights to a csv file.')
    parser.add_argument('--save_preds_0', action='store_true',
                        help='Save predictions for the first episode.')
    parser.add_argument('--use_synonyms', action='store_true', help='Use synonyms.') 
    parser.add_argument("--neval_episodes", type=int, default=2000,
                        help="Number of evaluation episodes both for base and novel.")
    parser.add_argument('--word_embed_size', type=int, default=500,
                        help='Word embedding classifier')
    parser.add_argument('--word_embed_path', type=str, default="word_embeds",
                        help='Where to store word embeds pickles for dataset.')
    parser.add_argument('--glove', action='store_true',
                        help='Use of Glove embeds instead of Vico.')
    parser.add_argument('--continual', action='store_true',
                        help='Evaluate like FSCIL.')
    parser.add_argument('--label_pull', type=float, default=None)
    parser.add_argument('--push_away', type=float, default=None)
    parser.add_argument('--no_dropblock', action='store_true',
                        help='Disable dropblock.')
    parser.add_argument('--attraction_override', type=str, default=None,
                        help='Instead of label pullers attract to elsewhere.')
    parser.add_argument('--lmbd_reg_novel',  type=float, default=None,
                        help='regularization for the novel classes in previous sessions.')

    parser.add_argument('--stable_epochs', type=int, default=10,
                        help='How many stable epochs before stopping.')
    parser.add_argument('--convergence_epsilon', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--memory_replay', type=int, default=0,
                        help="Keep samples from novel classes in memory.") # TODO: base
    if parser.parse_known_args()[0].label_pull is not None:
        parser.add_argument('--pulling', type=str, default="regularize",
                        help='How should we leverage label inspired weights?')
    parser.add_argument('--min_novel_epochs', type=int, default=15, help='min number of epochs for novel support set.')
    parser.add_argument('--max_novel_epochs', type=int, default=1000, help='max number of epochs for novel support set.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--freeze_backbone_at', type=int, default=1, help='freeze backbone while updating classifier at the epoch X, epochs start at 1.')
    parser.add_argument('--lmbd_reg_transform_w',  type=float, default=None, help='regularization for the base classes.')
    parser.add_argument('--target_train_loss',  type=float, default=1.0, help='When to stop fine-tuning.')
    parser.add_argument('--saliency',  action='store_true', help='append label to the beginning description')
    parser.add_argument('--use_episodes', action='store_true', help='use exact XtarNet episodes.')

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

def parse_option_supervised():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')


    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--eval_only', action='store_true', help='eval only on base classes')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--reload_path', type=str, default='', help='path to load model from')
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')
    parser.add_argument('--use_episodes', action='store_true', help='use exact XtarNet episodes.')
    parser.add_argument('--classifier', type=str,
                        choices=['linear', 'lang-linear', 'description-linear'])
    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')
    parser.add_argument('--continual', action='store_true',
                            help='Evaluate like FSCIL/ILVDQ.')
    parser.add_argument('--no_dropblock', action='store_true',
                            help='Disable dropblock.')
    parser.add_argument('--set_seed', type=int, default=5,
                        help='Seed for torch and np.')
    parser.add_argument('--no_linear_bias', action='store_true', help='Do not use bias in linear classifier.')
    parser.add_argument('--augment_pretrain_wtrainb', action='store_true', help='use train b classes too.')
    parser.add_argument('--eval_mode', type=str, default=None)
    parser.add_argument('--label_pull', type=float, default=None)
    if parser.parse_known_args()[0].label_pull is not None:
        parser.add_argument('--word_embed_size', type=int, default=500,
                            help='Word embedding classifier')
        parser.add_argument('--word_embed_path', type=str, default="word_embeds",
                            help='Where to store word embeds pickles for dataset.')
        parser.add_argument('--use_synonyms', action='store_true', help='Use synonyms.') # TODO
        parser.add_argument('--glove', action='store_true',
                            help='Use of Glove embeds instead of Vico.')
    opt = parser.parse_args()

    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_pretrained'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.linear_bias = not opt.no_linear_bias
    opt.model_name = '{}_{}_classifier_{}'.format(opt.dataset,
                                                    opt.model,
                                                    opt.classifier)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = opt.model_path # os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()
    print("Device count: ", opt.n_gpu)

    # Print opt

    # Add git commit hash
    process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    opt.git_head_hash = git_head_hash.decode()

    print("************* Training arguments *************")
    for arg in vars(opt):
        print(arg, getattr(opt, arg))
    print("End of arguments.\n")

    return opt
