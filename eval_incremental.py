# python eval_incremental.py --model_path dumped/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain/resnet12_last.pth --data_root data --n_shots 5 --incremental_eval

from __future__ import print_function

import argparse
import socket
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import MetaImageNet, ImageNet
from dataset.tiered_imagenet import MetaTieredImageNet
from dataset.cifar import MetaCIFAR100
from dataset.transform_cfg import transforms_test_options, transforms_list

from eval.meta_eval import meta_test, incremental_test
from eval.cls_eval import incremental_validate

# import wandb
# wandb.init(project="rfs")

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
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')
    parser.add_argument('--test_base_batch_size', type=int, default=64, metavar='test_batch_size',
                        help='Size of test batch)')
    parser.add_argument('--incremental_eval', action='store_true', help='labels of novel samples will be incremental.')
    parser.add_argument('--use_word_embeddings', action='store_true', help='word embedding classifier')

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

    # test loader
    args = opt
    
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_test_options[opt.transform]

        train_loader = DataLoader(ImageNet(args=opt, partition='train', transform=train_trans), #FIXME: use train
                                  batch_size=64, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        
        # load base evaluation dataset
        base_val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                     batch_size=opt.test_base_batch_size // 2, shuffle=True, drop_last=False,
                                     num_workers=opt.num_workers // 2)
        
        base_test_loader = DataLoader(ImageNet(args=opt, partition='test', transform=test_trans),
                                      batch_size=opt.test_base_batch_size // 2, shuffle=False, drop_last=False,
                                      num_workers=opt.num_workers // 2)
        
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
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

    # load model
    if opt.use_word_embeddings:
        vocab = [name for name in train_loader.dataset.label2human if name != '']
    else:
        vocab = None
    model = create_model(opt.model, n_cls, opt.dataset, vocab=vocab)
    ckpt = torch.load(opt.model_path)
    model.load_state_dict(ckpt['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

#     wandb.watch(model)
    
    # evalation
        
    if opt.incremental_eval:
        best_alpha = 0.7
        best_score = 0.0
        for alpha in np.arange(0.7,0.9,0.02):
            start = time.time()
            novel, base = incremental_test(model, meta_valloader, base_val_loader, alpha, use_logit=True)
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
        novel, base = incremental_test(model, meta_testloader, base_test_loader, best_alpha, use_logit=True)
        test_time = time.time() - start
        avg_test_score = (base[0]+novel[0])/2
        print('test_alpha: {0}'.format(best_alpha))
        print('test_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel[0], novel[1], test_time))
        print('test_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base[0], base[1], test_time))
        print('average: {:.4f}'.format((base[0]+novel[0])/2))
        
#       wandb.log({"Novel val accuracy": novel_val_acc, "Base val accuracy": base_val_acc, "Alpha":alpha})

#         start = time.time()
#         val_acc_feat, val_std_feat = incremental_test(model, meta_valloader, use_logit=False)
#         val_time = time.time() - start
#         print('val_acc_feat_novel: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc_feat, val_std_feat, val_time))
        
#         start = time.time()
#         test_acc, test_acc_top5, test_loss = incremental_test(val_loader, model, criterion, opt)
#         test_time = time.time() - start
#         print('val_acc_base: {:.4f}, val_acc_top5: {:.4f}, time: {:.1f}, loss: {:.1f}'.format(test_acc, test_acc_top5, test_time, test_loss))
        

#         logger.log_value('test_acc', test_acc, epoch)
#         logger.log_value('test_acc_top5', test_acc_top5, epoch)
#         logger.log_value('test_loss', test_loss, epoch)
    
    else:
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

def get_wkprimes_val(novel_val_train_loaders, cac_model, kprimes_val, args):
    
    w_kprimes = defaultdict(list)
    cac_model.eval()
    
    with torch.no_grad():
        for kprime in kprimes_val:
            loader = novel_val_train_loaders[kprime]
            for data in loader:
                
                # send data to gpu
                device = torch.device('cuda') # TODO Hard coded.
                data[0] = [d.to(device, non_blocking=True) for d in data[0]]
                data[1] = data[1].to(device, non_blocking=True)
                
                w_kprime = cac_model(data).unsqueeze(0)
                w_kprimes[kprime].append(w_kprime)
                
    least_number_of_wkprime_per_kprime = min([len(ws) for kprime, ws in w_kprimes.items()])
    num_set_of_wkprimes = min(least_number_of_wkprime_per_kprime, args.num_set_of_wkprimes)
    print("Sampling {} sets of wkprimes for val evaluation...".format(num_set_of_wkprimes))
    
    # returns list of lists
    w_kprimes_list = [[] for i in range(num_set_of_wkprimes)]

    for kprime in kprimes_val:
        w_for_kprime = np.random.choice(w_kprimes[kprime], num_set_of_wkprimes, replace=False)
        for i,w in enumerate(w_for_kprime):
            w_kprimes_list[i].append(w)
    return w_kprimes_list

def validation_manager(base_val_loader,
                       novel_val_train_loaders,
                       novel_val_val_loaders,
                       cac_model,
                       backbone,
                       tau,
                       kprimes_val,
                       base_classifier_weights,
                       criterion,
                       optimizer,
                       mapping_val_with_base,
                       mapping_val_novel_only,
                       epoch,
                       args,
                       writer,
                       mode):
    
    global best_acc1
    global total_steps
    # evaluate in kprime_val classes

    ## 1. first determine a set of w_kprimes_val for novel classes (different from training)
    ## with only a forward pass.
    w_kprimes_val = get_wkprimes_val(novel_val_train_loaders, 
                                     cac_model, 
                                     kprimes_val, 
                                     args)

    ## 2. then evaluate on all validation sets available for base and kprimes_val classes. TODO 
    acc = []
    bo = []
    bb = []
    nb = []
    no = []

    for w_kprimes in w_kprimes_val:
        result = validate(base_val_loader, novel_val_val_loaders, backbone, 
                          w_kprimes, base_classifier_weights, criterion, 
                          mapping_val_with_base, mapping_val_novel_only, epoch, 
                          args, writer, mode="val", verbose=False)
        
        acc_, base_only, base_both, novel_both, novel_only = result
        acc.append(acc_)
        bo.append(base_only)
        bb.append(base_both)
        nb.append(novel_both)
        no.append(novel_only)

    acc1 = np.mean(acc)
    wandb.log({"NovelVal/Top1/Base-Base-Only": np.mean(bo),
               "NovelVal/Top1/Base-Both": np.mean(bb),
               "NovelVal/Top1/Novel-Both": np.mean(nb),
               "NovelVal/Top1/Novel-Novel-Only": np.mean(no),
               "NovelVal/Top1/Both": acc1}, step=total_steps)
    
    # evaluate in the novel kprime_val classes
    print("===Average of novel and base scores with base AND novel classifier weights===")
    print(acc1)

    # save the best model
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    save_checkpoint({
        'epoch': epoch + 1,
        'cac_state_dict': cac_model.state_dict(),
        'backbone_state_dict': backbone.state_dict(),
        'tau': tau,
        'base_classifier_weights': base_classifier_weights,
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)

    return acc1

def get_base_evaluation(base_val_loader, backbone, W, criterion, args, verbose):
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(base_val_loader),
        [batch_time, losses, top1, top5],
        prefix='Base Val: ')

    # eval mode
    backbone.eval()
    
    with torch.no_grad():
        end = time.time()    
        for i, (images, target) in enumerate(base_val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            _, features = backbone(images)
            scores = features @ W.transpose(0,1)
            loss = criterion(scores, target)
            
            
            # update losses
            losses.update(loss.item(), images.size(0))

            # if there is only one classifier weight - # TODO this is confusing
            #             if scores.shape[1] == 1:
            #                 acc1 = accuracy(scores, target, topk=(1,))
            #                 top1.update(acc1[0], images.size(0))
            #                 top5.update(100., images.size(0))
            #             else:
            acc1, acc5 = accuracy(scores, target, topk=(1,5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if verbose and i % args.print_freq == 0:
                progress.display(i)
        if verbose:
            print(' * BASE: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
    return(top1.avg.item())

def get_novel_evaluation(novel_val_loaders, label_mapping, backbone, W, criterion, args, verbose):
    top1_forall = AverageMeter('AccOverall@1', ':6.2f')
    for kprime, loader in novel_val_loaders.items():
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, top1, top5],
            prefix='Novel Val: ')

        with torch.no_grad():
            end = time.time()
            size = 0
            for j, images in enumerate(loader):
                images = images.cuda(non_blocking=True)
                target = torch.ones(len(images), dtype=torch.long).cuda(non_blocking=True) * label_mapping[kprime]
                _, features = backbone(images)
                scores = features @ W.transpose(0,1)
                loss = criterion(scores, target)
                
                # update losses
                losses.update(loss.item(), images.size(0))
                
                acc_alt = min(5, scores.shape[1])
                
#                 if scores.shape[1] <= 5:
#                     acc1 = accuracy(scores, target)
#                     top1.update(acc1[0], images.size(0))
#                     ipdb.set_trace()
#                     top5.update(tensor(100., device='cuda:0'), images.size(0))
#                 else:
                acc1, acc5 = accuracy(scores, target, topk=(1, acc_alt))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                
#                 # if there is only one novel class
#                 if scores.shape[1] == 1:
#                     top1.update(100., images.size(0))
#                     top5.update(100., images.size(0))
#                 elif scores.shape[1] <= 5:
#                     acc1 = accuracy(scores, target, topk=(1,))
#                     top1.update(acc1[0].item(), images.size(0))
#                     top5.update(100., images.size(0))
#                 else:
#                     acc1, acc5 = accuracy(scores, target, topk=(1,5))
#                     top1.update(acc1[0], images.size(0))
#                     top5.update(acc5[0], images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()
                
                size += len(images)
                
                if verbose and j % args.print_freq == 0:
                    progress.display(j)
                    #         ipdb.set_trace()
        top1_forall.update(top1.avg.item(), size)
        
        if verbose:
            print(' * NOVEL {kprime}: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5, kprime=kprime))
    if verbose: 
        print(' * NOVEL: Acc@1 {top1_forall.avg:.3f}'
              .format(top1_forall=top1_forall))
        
    return(top1_forall.avg)

def validate(base_val_loader, novel_val_loaders, backbone, w_kprimes,
             base_classifier_weights, criterion, mapping_with_base,
             mapping_novel_only, epoch, args, writer, mode, verbose):
    
    global total_steps
    
    wkprimes =  w_kprimes[0] if len(w_kprimes)==1 else torch.stack([w.squeeze(0) for w in w_kprimes])
    W = torch.cat((base_classifier_weights, wkprimes), dim=0)
    
    # if mode is val this method is called multiple times for different set of
    # wkprimes, thus verbose is set to false

    if verbose:
        print("===Base class evaluation with base classifier weights only===")
        base_only = get_base_evaluation(base_val_loader, backbone, 
                                        base_classifier_weights, criterion, args, 
                                        verbose)
    if verbose:
        print("===Base class evaluation with base AND novel classifier weights===")
        base_both = get_base_evaluation(base_val_loader, backbone, W, criterion, 
                                        args, verbose)
    if verbose:
        print("===Novel class evaluation with base AND novel classifier weights===")
        novel_both = get_novel_evaluation(novel_val_loaders, mapping_with_base, 
                                          backbone, W, criterion, args, verbose)
    if verbose:
        print("===Novel class evaluation with novel classifier weights only===")
        novel_only = get_novel_evaluation(novel_val_loaders, mapping_novel_only, 
                                          backbone, wkprimes, criterion, args, verbose)
        
    acc1 = (novel_both + base_both)/2
    if verbose:
        print("===Average of novel and base scores with base AND novel classifier weights===")
        print(acc1)
        
    # save to wandb
    if mode == "val":
        return acc1, base_only, base_both, novel_both, novel_only
    
    wandb.log({"NovelTrain/Top1/Base-Base-Only": base_only,
               "NovelTrain/Top1/Base-Both": base_both,
               "NovelTrain/Top1/Novel-Both": novel_both,
               "NovelTrain/Top1/Novel-Novel-Only": novel_only,
               "NovelTrain/Top1/Both": acc1}, step=total_steps)
    
    if args.tensorboard:
        writer.add_scalar('NovelTrain/Top1/Base-Base-Only', base_only, epoch)
        writer.add_scalar('NovelTrain/Top1/Base-Both', base_both, epoch)
        writer.add_scalar('NovelTrain/Top1/Novel-Both', novel_both, epoch)
        writer.add_scalar('NovelTrain/Top1/Novel-Novel-Only', novel_only, epoch)
        writer.add_scalar('NovelTrain/Top1/Both', acc1, epoch)
        
    return acc1

if __name__ == '__main__':
    main()
