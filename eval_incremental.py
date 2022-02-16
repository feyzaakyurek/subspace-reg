# This script is largely based on https://github.com/WangYueFt/rfs

from __future__ import print_function
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import time
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.util import create_model
from dataset.mini_imagenet import MetaImageNet, ImageNet
from dataset.transform_cfg import transforms_test_options

from util import create_and_save_embeds
from eval.language_eval import few_shot_finetune_incremental_test
from configs import parse_option_eval

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
    args = opt
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("End of arguments.\n")



    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]

        # Base test samples loader. "split=train" refers to the base set of classes
        # "phase=test" means we are interested in those samples that were not used in
        # training.
        base_test_loader = DataLoader(ImageNet(args=opt, split='train', phase='test', transform=test_trans),
                                      batch_size=opt.test_base_batch_size // 2,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=opt.num_workers // 2)

        # In case we are storing memory from base classes.
        base_support_loader = None
        if opt.n_base_support_samples > 0:
            ''' We'll use support samples from base classes. '''
            base_support_loader = DataLoader(MetaImageNet(args=opt, split='train', phase='train',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=True, use_episodes=False),
                                     batch_size=opt.test_batch_size, shuffle=True, drop_last=False, # False?
                                     num_workers=opt.num_workers)

        # Test samples from novel classes as they are introduced.
        # split=val means we are interested in novel classes.
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
    else:
        raise NotImplementedError(opt.dataset)

    # Load model if available, check bias.
    ckpt = torch.load(opt.model_path)

    vocab = None
    # In this scenario we'll need the label embeds saved.
    # Label pull is used interchangeably with semantic subspace reg.
    if opt.label_pull is not None: # label_pull refers to gamma in the paper.
        vocab_train = [name for name in base_test_loader.dataset.label2human if name != '']
        vocab_val = [name for name in meta_valloader.dataset.label2human if name != '']
        vocab_all = vocab_train + vocab_val # + vocab_test
        create_and_save_embeds(opt, vocab_all)

    # Linear layer bias is determined based on backbone.
    # Warning: we often assumed no linear bias.
    if opt.classifier =="linear":
        if 'classifier.bias' in ckpt['model'].keys():
            if ckpt['model']['classifier.bias'] is None:
                raise ValueError()
            opt.linear_bias = True
        else:
            opt.linear_bias = False

    # Load model.
    model = create_model(opt.model, n_cls, opt, vocab=vocab, dataset=opt.dataset)
    print("Loading model...")
    model.load_state_dict(ckpt['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # Evaluation
    assert opt.classifier == "linear"
    criterion = nn.CrossEntropyLoss()


    start = time.time()
    opt.split = "val"
    opt.neval_episodes = 8 # If multi-session, this is overridden later.
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



if __name__ == '__main__':
    main()
