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
                                 "few-shot-incremental-language-pretrain-linear-tune",
                                 "hierarchical-incremental-few-shot"])

    parser.add_argument('--classifier', type=str,
                        choices=['linear', 'lang-linear', 'description-linear'])
    parser.add_argument('--track_weights', action='store_true',
                            help='Save the classifier weights to a csv file.')
    parser.add_argument('--track_label_inspired_weights', action='store_true',
                            help='Save the label inspired weights to a csv file.')
    parser.add_argument('--save_preds_0', action='store_true', help='Save predictions for the first episode.' ) # TODO: This may not be available for every evalmode
    parser.add_argument('--use_synonyms', action='store_true', help='Use synonyms.') # TODO

    if parser.parse_known_args()[0].eval_mode in ["few-shot-incremental-fine-tune"]:
        parser.add_argument('--hierarchical_eval',  action='store_true', help='Use base/novel threshold during evaluation.')

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
                                                  "few-shot-incremental-language-pretrain-linear-tune",
                                                  "hierarchical-incremental-few-shot"]:
        parser.add_argument("--neval_episodes", type=int, default=2000,
                            help="Number of evaluation episodes both for base and novel.")


    if parser.parse_known_args()[0].classifier in ["lang-linear", "description-linear"]:
        parser.add_argument('--word_embed_size', type=int, default=500,
                            help='Word embedding classifier')
        parser.add_argument('--word_embed_path', type=str, default="word_embeds",
                            help='Where to store word embeds pickles for dataset.')
        parser.add_argument('--word_embed_type', type=str, default="")
#         parser.add_argument('--lang_classifier_bias', action='store_true',
#                             help='Use of bias in lang classifier.')
#         parser.add_argument('--multip_fc', type=float, default=0.05) # should be read from the loaded model.
        parser.add_argument('--diag_reg', type=float, default=None)
        parser.add_argument('--attention', type=str, choices=["sum","concat","context"],
                            default=None, help='Use of attention in lang classifier.')
        parser.add_argument('--orig_alpha', type=float, default=1.0)
        parser.add_argument('--transform_query_size', type=int, default=None, help='Output size of key, query, value in attention.')


    if parser.parse_known_args()[0].eval_mode in ["few-shot-incremental-fine-tune"]:
        parser.add_argument('--word_embed_size', type=int, default=500,
                            help='Word embedding classifier')
        parser.add_argument('--word_embed_path', type=str, default="word_embeds",
                            help='Where to store word embeds pickles for dataset.')
        parser.add_argument('--glove', action='store_true',
                            help='Use of Glove embeds instead of Vico.')
        parser.add_argument('--label_pull', type=float, default=None)
        parser.add_argument('--push_away', type=float, default=None)
        parser.add_argument('--attraction_override', type=str, default=None,
                            help='Instead of label pullers attract to elsewhere.')
        parser.add_argument('--pull_path_override', type=str, default=None,
                            help='Load embeds here.')
        parser.add_argument('--novel_initializer', type=str, default=None,
                            help='novel weight initialization rule if not random.')
        parser.add_argument('--stable_epochs', type=int, default=10,
                            help='How many stable epochs before stopping.')
        parser.add_argument('--convergence_epsilon', type=float, default=1e-4)
        parser.add_argument('--temperature', type=float, default=3)

        if parser.parse_known_args()[0].label_pull is not None:
            parser.add_argument('--pulling', type=str, default="regularize",
                            help='How should we leverage label inspired weights?')

    if parser.parse_known_args()[0].eval_mode in ['zero-shot-incremental']:
        parser.add_argument('--num_novel_combs', type=int, default=0.05,
                            help='Number of combinations of novel/test classes to evaluate base samples against:)')

    if parser.parse_known_args()[0].eval_mode in ["few-shot-language-incremental",
                                                  "few-shot-incremental-fine-tune",
                                                  "few_shot_language_pretrain_linear_tune",
                                                  "few-shot-incremental-language-pretrain-linear-tune",
                                                  "hierarchical-incremental-few-shot"]:
        parser.add_argument('--min_novel_epochs', type=int, default=15, help='min number of epochs for novel support set.')
        parser.add_argument('--max_novel_epochs', type=int, default=400, help='max number of epochs for novel support set.')
#         parser.add_argument('--novel_epochs', type=int, default=15, help='number of epochs for novel support set.')
#         parser.add_argument('--max_novel_epochs', type=int, default=400, help='number of epochs for novel support set.')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--adam', action='store_true', help='use adam optimizer')
        parser.add_argument('--freeze_backbone_at', type=int, default=1, help='freeze backbone while updating classifier at the epoch X, epochs start at 1.')
        parser.add_argument('--lmbd_reg_transform_w',  type=float, default=None, help='regularization for the base classes.')
        parser.add_argument('--target_train_loss',  type=float, default=1.0, help='When to stop fine-tuning.')
        parser.add_argument('--saliency',  action='store_true', help='append label to the beginning description')
        parser.add_argument('--use_episodes', action='store_true', help='use exact XtarNet episodes.')

    if parser.parse_known_args()[0].classifier in ["description-linear"]:
        parser.add_argument('--description_embed_path', type=str, default="description_embeds")
        parser.add_argument('--desc_embed_model', type=str, default="bert-base-cased")
        parser.add_argument('--transformer_layer', type=str, default=6)
        parser.add_argument('--prefix_label', action='store_true', help='append label to the beginning description')

    parser.add_argument('--skip_val', action='store_true', help='skip validation evaluation')
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
#     parser.add_argument('--save_folder', type=str, default='', help='path to save model')
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

    #if parser.parse_known_args()[0].classifier in ["linear"]:
    parser.add_argument('--no_linear_bias', action='store_true', help='Do not use bias in linear classifier.')

    parser.add_argument('--augment_pretrain_wtrainb', action='store_true', help='use train b classes too.')

    if parser.parse_known_args()[0].classifier in ["lang-linear"]:
        parser.add_argument('--word_embed_size', type=int, default=500, help='Word embedding classifier')

    if parser.parse_known_args()[0].classifier in ["lang-linear", "description-linear"]:
        parser.add_argument('--word_embed_path', type=str, default="word_embeds")
        parser.add_argument('--word_embed_type', type=str, default="")
        parser.add_argument('--lang_classifier_bias', action='store_true', help='Use of bias in lang classifier.')
        parser.add_argument('--multip_fc', type=float, default=0.05)
        parser.add_argument('--diag_reg', type=float, default=0.05)
        parser.add_argument('--attention', type=str, choices=["sum","concat","context"], default=None, help='Use of attention in lang classifier.')
        parser.add_argument('--transform_query_size', type=int, default=None, help='Output size of key, query, value in attention.')

    if parser.parse_known_args()[0].classifier in ["description-linear"]:
        parser.add_argument('--description_embed_path', type=str, default="description_embeds")
        parser.add_argument('--desc_embed_model', type=str, default="bert-base-cased")
        parser.add_argument('--transformer_layer', type=int, default=6, help="which layer to use from transformer.w")
        parser.add_argument('--prefix_label', action='store_true', help='append label to the beginning description')

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

#     opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model,
#                                                             opt.dataset,
#                                                             opt.learning_rate,
#                                                             opt.weight_decay,
#                                                             opt.transform)

#     if opt.cosine:
#         opt.model_name = '{}_cosine'.format(opt.model_name)

#     if opt.adam:
#         opt.model_name = '{}_useAdam'.format(opt.model_name)

#     if 'SLURM_JOB_ID' in os.environ:
#         job_id = os.environ['SLURM_JOB_ID']
#     else:
#         job_id = np.random.randint(100000, 999999, 1)[0]

    if opt.classifier == "description-linear":
        opt.model_name = '{}_{}_classifier_{}_layer_{}_multipfc_{}_prefix_{}'.format(opt.dataset,
                                                                                  opt.model,
                                                                                opt.classifier,
                                                                                opt.transformer_layer,
                                                                                opt.multip_fc,
                                                                                opt.prefix_label)
    elif opt.classifier == "lang-linear":
        opt.model_name = '{}_{}_classifier_{}_multipfc_{}'.format(opt.dataset, opt.model,
                                                                  opt.classifier,
                                                                  opt.multip_fc)
    else:
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
