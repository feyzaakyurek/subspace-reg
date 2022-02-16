from __future__ import print_function

import os
import socket
import time
import sys
import subprocess
import numpy as np

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model
from models.resnet_language import LangPuller

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, create_and_save_embeds, create_and_save_descriptions
from eval.util import accuracy, AverageMeter, validate

import ipdb
from configs import parse_option_supervised
#import wandb
# os.environ["WANDB_API_KEY"] = "1c6a939ef88d70da594fe947cdd93866d84bee87"
# os.environ["WANDB_MODE"] = "dryrun"
# wandb.init(project="rfs")

def main():
    opt = parse_option_supervised()
    # dataloader
#     train_partition = 'trainval' if opt.use_trainval else 'train'

    # Set seeds
    torch.manual_seed(opt.set_seed)
    np.random.seed(opt.set_seed)
    
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(ImageNet(args=opt, split="train", phase="train", transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(ImageNet(args=opt, split="train", phase="val", transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
#         meta_testloader = DataLoader(MetaImageNet(args=opt, split='test',
#                                                   train_transform=train_trans,
#                                                   test_transform=test_trans),
#                                      batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                      num_workers=opt.num_workers)
#         meta_valloader = DataLoader(MetaImageNet(args=opt, split='val',
#                                                  train_transform=train_trans,
#                                                  test_transform=test_trans),
#                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
#                                     num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
            if opt.continual:
                n_cls = 60

    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(TieredImageNet(args=opt, split="train", phase="train",
                                                 transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(TieredImageNet(args=opt, split="train", phase="val", transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, split='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, split='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351

    else:
        raise NotImplementedError(opt.dataset)
    
    lang_puller = None
    if opt.classifier in ["lang-linear", "description-linear"] or opt.label_pull is not None:
        # Save full dataset vocab if not available
        vocab_train = [name for name in train_loader.dataset.label2human if name != '']
#         vocab_test = [name for name in meta_testloader.dataset.label2human if name != '']
#         vocab_val = [name for name in meta_valloader.dataset.label2human if name != '']
        vocab = vocab_train # + vocab_val # + vocab_test 

        if opt.classifier == "lang-linear" or opt.label_pull:
            create_and_save_embeds(opt, vocab)

        if opt.classifier == "description-linear":
            create_and_save_descriptions(opt, vocab)

        vocab = vocab_train
        if opt.label_pull is not None:
            lang_puller = LangPuller(opt, vocab_train, vocab_train)
            vocab = None
        else:
            lang_puller = None

    else:
        vocab = None


    # model
    model = create_model(opt.model, n_cls, opt, vocab=vocab)

    # optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=0.0005)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    # routine: supervised pre-training
    for epoch in range(1, opt.epochs + 1):
#         print("Before Epoch {}".format(epoch))

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)

        if not opt.eval_only:
            print("==> training...")

            time1 = time.time()
            train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt, lang_puller)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
            }
            if opt.continual:
                state['training_classes'] = train_loader.dataset.basec_map
                state['label2human'] = train_loader.dataset.label2human
            save_file = os.path.join(opt.model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
    }
    if opt.continual:
        state['training_classes'] = train_loader.dataset.basec_map
        state['label2human'] = train_loader.dataset.label2human
    save_file = os.path.join(opt.model_path, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


def train(epoch, train_loader, model, criterion, optimizer, opt, lang_puller=None):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target,  _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda().long()

        # ===================forward=====================
        if opt.classifier != "linear" and opt.attention is not None:
            output, alphas = model(input, get_alphas=True)

            loss = criterion(output, target) + opt.diag_reg * criterion(alphas, target)
        else:
            output = model(input)
            loss = criterion(output, target)
        if opt.label_pull is not None:
            penalty = lang_puller.loss1(opt.label_pull,
                                        lang_puller(model.classifier.weight),
                                        model.classifier.weight)
            loss += penalty
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
