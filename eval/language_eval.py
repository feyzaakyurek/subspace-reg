from __future__ import print_function
import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm
import ipdb
import os
import time
import copy
import pickle

import torch
import torch.nn as nn

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from .util import accuracy, NN, normalize, image_formatter, \
mean_confidence_interval, Cosine, get_vocabs, drop_a_dim, get_batch_cycle

from models.resnet_language import LangLinearClassifier

import pandas as pd
from PIL import Image
import io
import base64

def fine_tune_novel(epoch, support_xs, support_ys_id, net, criterion, optimizer, orig_classifier_weights, opt, label_inspired_weights=None, orig_classifier_bias=None):
    """One epoch training, single batch training."""

    support_xs = support_xs.float().cuda()
    support_ys_id = support_ys_id.cuda()

    # Compute output
    if opt.classifier in ["lang-linear", "description-linear"] and opt.attention is not None:
        output, alphas = net(support_xs, get_alphas=True)
        loss = criterion(output, support_ys_id) + opt.diag_reg * criterion(alphas, support_ys_id)
    else:
        output = net(support_xs)
        loss = criterion(output, support_ys_id)


    # output = net(support_xs)
    # loss = criterion(output, support_ys_id)
#     if opt.lmbd_reg_transform_w is not None:
#         loss = loss + opt.lmbd_reg_transform_w * torch.norm(net.classifier.transform_W - orig_transform_W)

    if opt.lmbd_reg_transform_w is not None:
        len_vocab,_ = orig_classifier_weights.size()
        loss += opt.lmbd_reg_transform_w * torch.norm(net.classifier.weight[:len_vocab,:] - orig_classifier_weights)
        if orig_classifier_bias is not None:
            loss += opt.lmbd_reg_transform_w * torch.norm(net.classifier.bias[:len_vocab] - orig_classifier_bias)**2

    if label_inspired_weights is not None and opt.pulling == "regularize":
        loss += opt.label_pull * torch.norm(net.classifier.weight[len_vocab:,:] - label_inspired_weights)**2

    # Train
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc1, acc5 = accuracy(output, support_ys_id, topk=(1,5))

    print('=======Novel Epoch {}=======\n'
          'Train\t'
          'Loss {:10.4f}\t'
          'Acc@1 {:10.3f}\t'
          'Acc@5 {:10.3f}'.format(
           epoch, loss.item(), acc1[0], acc5[0]))

    return acc1[0], loss.item()

def validate_fine_tune(query_xs, query_ys_id, net, criterion, opt):
    net.eval()
    with torch.no_grad():
        query_xs = query_xs.float()
        if torch.cuda.is_available():
            query_xs = query_xs.cuda()
            query_ys_id = query_ys_id.cuda()

            # compute output
            output = net(query_xs)
            loss = criterion(output, query_ys_id)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, query_ys_id, topk=(1, 5))
            query_ys_pred = torch.argmax(output, dim=1).detach().cpu().numpy()
            print('Test \t'
                  'Loss {:10.4f}\t'
                  'Acc@1 {:10.3f}\t'
                  'Acc@5 {:10.3f}'.format(
                   loss.item(), acc1[0], acc5[0]))

    return acc1[0], acc5[0], loss.item(), query_ys_pred

def eval_base(net, base_batch, criterion, vocab_all, df=None, return_preds=False):
    acc_base_ = []
    net.eval()
    with torch.no_grad():
#         for idb, (input, target, _) in enumerate(base_val_loader):
        input, target, *_ = base_batch
        input = input.squeeze(0).float().cuda()
        target = target.squeeze(0).cuda()

        output = net(input) #.detach().cpu()


        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc_base_.append(acc1[0].item())

        if df is not None:
            ys_pred = torch.argmax(output, dim=1).detach().cpu().numpy()
            imgdata = input.detach().numpy()
            base_info = [(0, vocab_all[target[i]], True, vocab_all[ys_pred[i]],
                          image_formatter(imgdata[i,:,:,:]))  for i in range(len(target))]
            df = df.append(pd.DataFrame(base_info, columns=df.columns), ignore_index=True)

    if return_preds:
        return np.mean(acc_base_), ys_pred
    return np.mean(acc_base_)

def few_shot_finetune_incremental_test(net, ckpt, criterion, meta_valloader, base_val_loader, opt,  vis=False):
    if vis:
        df = pd.DataFrame(columns=['idx', 'class', 'isbase', 'predicted', 'img'])

    acc_novel = []
    acc_base = []
    running_avg = []
    pull_running_avg = []

    # In case of hierarchical eval.
    novel_hacc = []
    base_hacc = []

    # Used for creation of confusion matrices.
    # preds_df = pd.DataFrame(columns = ["Episode", "Gold", "Prediction"])

    # Pretrained backbone, net will be reset before every episode.
    basenet = copy.deepcopy(net).cuda()

    # Linear layer of the backbone.
    base_weight = basenet.classifier.weight.clone().detach().requires_grad_(False)
    if basenet.classifier.bias is not None:
        base_bias   = basenet.classifier.bias.clone().detach().requires_grad_(False)
    else:
        base_bias = None

    # Loaders for fine tuning and testing.
    base_valloader_it = iter(base_val_loader)
    meta_valloader_it = iter(meta_valloader)

    if opt.track_weights:
        track_weights = pd.DataFrame(columns=["episode", "type", "label", "class", "fine_tune_epoch", "classifier_weight"])
    if opt.track_label_inspired_weights:
        track_inspired = pd.DataFrame(columns=["episode", "label", "fine_tune_epoch", "inspired_weight"])


    for idx in range(opt.neval_episodes):
        print("\n**** Iteration {}/{} ****\n".format(idx, opt.neval_episodes))

        try:
            data = next(meta_valloader_it)
        except StopIteration:
            meta_valloader_it = iter(meta_valloader)
            data = next(meta_valloader_it)

        support_xs, support_ys, query_xs, query_ys = drop_a_dim(data)
        novelimgs = query_xs.detach().numpy()

        # Get sorted numeric labels, create a mapping that maps the order to actual label
        vocab_base, vocab_all, vocab_novel, orig2id = get_vocabs(base_val_loader, meta_valloader, query_ys)
        novel_ids = np.sort(np.unique(query_ys))
        query_ys_id = torch.LongTensor([orig2id[y] for y in query_ys])
        support_ys_id = torch.LongTensor([orig2id[y] for y in support_ys])

        # Reset the network.
        net = copy.deepcopy(basenet)
        net.train()
        classifier = net.classifier

        # Create classifier weights for novel classes.
        dummy_classifier = nn.Linear(640, len(novel_ids), bias=(basenet.classifier.bias is not None)) # TODO!!
        novel_weight = dummy_classifier.weight.detach().cuda()

        # Augment the classifier.
        classifier.weight = nn.Parameter(torch.cat([base_weight, novel_weight], 0))
        if basenet.classifier.bias is not None:
            novel_bias = dummy_classifier.bias.detach().cuda()
            classifier.bias = nn.Parameter(torch.cat([base_bias, novel_bias]))

        # Label pulling is a regularization towards the label attractors.
        if opt.label_pull is not None:
            dim = opt.word_embed_size # TODO

            # Retrieve novel embeds
            embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim)) # TODO
            with open(embed_pth, "rb") as openfile:
                embeds = pickle.load(openfile)

            novel_embeds = [0] * len(vocab_novel)
            for (i,token) in enumerate(vocab_novel):
                words = token.split(' ')
                for w in words:
                    try:
                        novel_embeds[i] += embeds[w]
                    except KeyError:
                        novel_embeds[i] = np.zeros(dim)
                novel_embeds[i] /= len(words)
            novel_embeds = torch.cuda.FloatTensor(np.stack(novel_embeds, axis=0))

            # Retrieve base embeds
            if opt.use_synonyms:
                embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}_base_synonyms.pickle".format(opt.dataset, dim)) # TOdo
                with open(embed_pth, "rb") as openfile:
                    label_syn_embeds = pickle.load(openfile)
                base_embeds = []
                for base_label in vocab_base:
                    base_embeds.append(label_syn_embeds[base_label])
                base_embeds = torch.cuda.FloatTensor(np.stack(base_embeds, axis=0))

            else:
                embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim)) # TODO
                with open(embed_pth, "rb") as openfile:
                    embeds = pickle.load(openfile)
                base_embeds = [0] * len(vocab_base)
                for (i,token) in enumerate(vocab_base):
                    words = token.split(' ')
                    for w in words:
                        try:
                            base_embeds[i] += embeds[w]
                        except KeyError:
                            base_embeds[i] = np.zeros(dim)
                    base_embeds[i] /= len(words)
                base_embeds = torch.cuda.FloatTensor(np.stack(base_embeds, axis=0))

            # This will be used to compute label attractors.
            softmax = nn.Softmax(dim=1)

            # If Glove, use the first 300 TODO
            if opt.glove: #todo
                base_embeds = base_embeds[:,:300]
                novel_embeds = novel_embeds[:,:300]



        # Validate before training. TODO
        test_acc, test_acc_top5, test_loss, _ = validate_fine_tune(query_xs, query_ys_id, net, criterion, opt)
        print('{:25} {:.4f}\n'.format("Novel incremental acc before fine-tune:",test_acc.item()))

        # Optimizer
        if opt.adam:
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=opt.learning_rate,
                                         weight_decay=0.0005)
        else:
            optimizer = torch.optim.SGD(net.parameters(),
                                  lr=opt.learning_rate,
                                  momentum=opt.momentum,
                                  weight_decay=opt.weight_decay) # TODO anything to load from ckpt?



        # Routine: fine-tuning for novel classes
        train_loss = 15
        epoch = 1
        while train_loss > opt.target_train_loss or epoch < opt.novel_epochs + 1:
            freeze_backbone_weights(net, opt, epoch, exclude=["classifier"])
#             base_weight = classifier.weight.clone().detach().requires_grad_(False)[:len(vocab_base),:]
            if opt.label_pull is not None and opt.pulling == "regularize":
                # save softmax here.
                scores_over_labels = softmax(novel_embeds @ torch.transpose(base_embeds, 0, 1))
                label_inspired_weights = scores_over_labels @ base_weight # 5 x 640
#                 ipdb.set_trace()
            else:
                label_inspired_weights = None

            train_acc, train_loss = fine_tune_novel(epoch,
                                                    support_xs,
                                                    support_ys_id,
                                                    net,
                                                    criterion,
                                                    optimizer,
                                                    base_weight,
                                                    opt,
                                                    label_inspired_weights,
                                                    base_bias)

            test_acc, test_acc_top5, test_loss, query_ys_pred = validate_fine_tune(query_xs,
                                                                                   query_ys_id,
                                                                                   net,
                                                                                   criterion,
                                                                                   opt)

            if opt.track_label_inspired_weights:
                inspired_weights = label_inspired_weights.clone().cpu().numpy()
                for k,lbl in enumerate(vocab_novel):
                    track_inspired.loc[len(track_inspired)] = [idx, lbl, epoch, inspired_weights[k]]


            if opt.track_weights:
                classifier_weights = classifier.weight.clone().detach().cpu().numpy()
                for k,lbl in enumerate(vocab_base):
                    track_weights.loc[len(track_weights)] = [idx, "base", lbl, vocab_base[k], epoch, classifier_weights[k]]
                len_base = len(vocab_base)
                for k,lbl in enumerate(vocab_novel):
                    track_weights.loc[len(track_weights)] = [idx, "novel", lbl, vocab_novel[k], epoch, classifier_weights[len_base+k]]


            if vis and idx == 0:
                novel_info = [(idx, vocab_all[query_ys_id[i]], False, vocab_all[query_ys_pred[i]],
                               image_formatter(novelimgs[i,:,:,:]))  for i in range(len(query_ys_id))]
                df = df.append(pd.DataFrame(novel_info, columns=df.columns), ignore_index=True)
            epoch += 1

        try:
            base_batch = next(base_valloader_it)
        except StopIteration:
            base_valloader_it = iter(base_val_loader)
            base_batch = next(base_valloader_it)

        # Evaluate base samples with the updated network
        if vis and idx == 0:
            acc_base_ = eval_base(net, base_batch, criterion, vocab_all, df=df)
        else:
            acc_base_ = eval_base(net, base_batch, criterion, vocab_all) # TODO: vocab all should be irrelevant?

#         # To save preds (temporary, comment out below later) #
#         _, base_query_ys, *_ = base_batch
#         base_query_ys = base_query_ys.squeeze(0)
#         acc_base_, base_preds  = eval_base(net, base_batch, criterion, vocab_all, return_preds=True)
#         id2orig = {}

#         for k,v in orig2id.items():
#             id2orig[v] = k
#         print("!!! Mini imagenet hard coded 64. !!")
#         query_ys_pred = [id2orig[k.item()] if k>=64 else k.item() for k in query_ys_pred]
#         base_preds = [id2orig[k.item()] if k>=64 else k.item() for k in base_preds]
# #         ipdb.set_trace()
#         temp_df = pd.DataFrame({"Episode": np.repeat(idx, len(query_ys)+len(base_query_ys)),
#                                 "Gold": np.concatenate((query_ys, base_query_ys),0),
#                                 "Prediction": np.concatenate((query_ys_pred, base_preds),0).astype(int)})
#         preds_df = pd.concat([preds_df, temp_df], 0)

#         if idx == 5:
#             preds_df.to_csv("csv_files/finetuning_preds.csv", index=False)
#             exit(0)
#         # To save preds (temporary, comment out above later) #


        avg_score = (acc_base_ + test_acc.item())/2

        acc_base.append(acc_base_)
        acc_novel.append(test_acc.item())
        running_avg.append(avg_score)


        # Little pull here.
        pull_acc_base_ = 0
        pull_test_acc = 0
        if opt.label_pull is not None and opt.pulling == "last-mile": #TODO
            with torch.no_grad():
                scores_over_labels = softmax(novel_embeds @ torch.transpose(base_embeds, 0, 1))
                label_inspired_weights = scores_over_labels @ classifier.weight[:len(vocab_base)]
                novel_weights_pulled = classifier.weight[len(vocab_base):,:] + opt.label_pull * (label_inspired_weights - classifier.weight[len(vocab_base):,:])
                classifier.weight = nn.Parameter(torch.cat([base_weight,
                                                            novel_weights_pulled], 0))


            pull_acc_base_ = eval_base(net, base_batch, criterion, vocab_all)
            pull_test_acc, test_acc_top5, test_loss, _ = validate_fine_tune(query_xs,
                                                                       query_ys_id,
                                                                       net,
                                                                       criterion,
                                                                       opt)
            pull_avg_score = (pull_acc_base_ + pull_test_acc.item())/2
            pull_running_avg.append(pull_avg_score)

            print('{:25} {:.4f}\n'
              '{:25} {:.4f}\n'
              '{:25} {:.4f}\n'
              '{:25} {:.4f}\n'.format(
                  "Pull Novel incremental acc:",
                  pull_test_acc.item(),
                  "Pull Base incremental acc:",
                  pull_acc_base_,
                  "Pull Average:",
                  pull_avg_score,
                  "Pull Running Average:",
                  np.mean(pull_running_avg)), flush=True)

        print('\n{:25} {:}\n'
              '{:25} {:}\n'
              '{:25} {:}\n'
              '{:25} {:.4f}\n'
              '{:25} {:.4f}\n'
              '{:25} {:.4f}\n'
              '{:25} {:.4f}\n'.format(
                  "Novel classes are:",
                  novel_ids,
                  "Human labels are:",
                  vocab_novel,
                  "Novel training epochs:",
                  epoch-1,
                  "Novel incremental acc:",
                  test_acc.item(),
                  "Base incremental acc:",
                  acc_base_,
                  "Average:",
                  avg_score,
                  "Running Average:",
                  np.mean(running_avg),
                  ), flush=True)


    if opt.track_label_inspired_weights:
        track_inspired.to_csv(f"track_inspired_{opt.eval_mode}_pulling_{opt.pulling}_{opt.label_pull}_target_loss_{opt.target_train_loss}_synonyms_{opt.use_synonyms}.csv", index=False)

    if opt.track_weights:
        track_weights.to_csv(f"track_weights_{opt.eval_mode}_pulling_{opt.pulling}_{opt.label_pull}_target_loss_{opt.target_train_loss}_synonyms_{opt.use_synonyms}.csv", index=False)

    if vis:
        return df
    else:
        return mean_confidence_interval(acc_novel), mean_confidence_interval(acc_base)

def few_shot_language_incremental_test(net, ckpt, criterion, meta_valloader, base_val_loader, opt, vis=False):
    if vis:
        df = pd.DataFrame(columns=['idx', 'class', 'isbase', 'predicted', 'img'])
    attention = opt.attention
    assert net.classifier.attention == attention
    acc_novel = []
    acc_base = []
    running_avg = []
    basenet = copy.deepcopy(net).cuda()
    print("Copied to basenet.")
    if basenet.classifier.multip_fc == 0: # TODO
        print("A LARGE WARNING!!! Loaded multipfc is 0, setting it to {}!!!".format(opt.multip_fc))
        basenet.classifier.multip_fc = nn.Parameter(torch.FloatTensor([opt.multip_fc]), requires_grad=False)

    embed = basenet.classifier.embed.clone().detach().requires_grad_(False)

    if attention:
        if opt.lmbd_reg_transform_w:
            orig_classifier_weights = basenet.classifier.transform_W_output.clone().detach().requires_grad_(False)
        else:
            orig_classifier_weights = None
    else:
        trns = basenet.classifier.transform_W.clone().detach().requires_grad_(False)
        orig_classifier_weights = embed @ trns if opt.lmbd_reg_transform_w else None
        print("Retrieved original classifier weights.")

    base_valloader_it = iter(base_val_loader)
    meta_valloader_it = iter(meta_valloader)
    print("Created iterators.")
    print("len(base_valloader_it) ", len(base_valloader_it))
    print("len(meta_valloader_it) ", len(meta_valloader_it))
    print("opt.neval_episodes ", opt.neval_episodes)
#     for idx, data in enumerate(meta_valloader):
    for idx in range(opt.neval_episodes):
        print("\n**** Iteration {}/{} ****\n".format(idx, opt.neval_episodes))
        try:
            data = next(meta_valloader_it)
        except StopIteration:
            meta_valloader_it = iter(meta_valloader)
            data = next(meta_valloader_it)
#         ipdb.set_trace()
        support_xs, support_ys, query_xs, query_ys = drop_a_dim(data)
        novelimgs = query_xs.detach().numpy()

        # Get sorted numeric labels, create a mapping that maps the order to actual label

        vocab_base, vocab_all, vocab_novel, orig2id = get_vocabs(base_val_loader, meta_valloader, support_ys)
        novel_ids = np.sort(np.unique(query_ys))
        query_ys_id = torch.LongTensor([orig2id[y] for y in query_ys])
        support_ys_id = torch.LongTensor([orig2id[y] for y in support_ys])

        net = copy.deepcopy(basenet)
        classifier = net.classifier
        multip_fc = classifier.multip_fc.detach().clone().cpu().item()

        if opt.classifier == "lang-linear":

            # Create the language classifier which uses only novel class names' embeddings
            dim = opt.word_embed_size
            embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim))
            dummy_classifier = LangLinearClassifier(vocab_novel,
                                                    embed_pth,
                                                    dim=dim,
                                                    cdim=640,
                                                    bias=opt.lang_classifier_bias,
                                                    verbose=False,
                                                    multip_fc=multip_fc,
                                                    attention=attention,
                                                    transform_query_size=opt.transform_query_size) # TODO!!

        else: # Description linear classifier

            embed_pth = os.path.join(opt.description_embed_path,
                                     "{0}_{1}_layer{2}_prefix_{3}.pickle".format(opt.dataset,
                                                                                 opt.desc_embed_model,
                                                                                 opt.transformer_layer,
                                                                                 opt.prefix_label))
            dummy_classifier = LangLinearClassifier(vocab_novel,
                                                    embed_pth,
                                                    cdim=640,
                                                    dim=None,
                                                    bias=opt.lang_classifier_bias,
                                                    description=True,
                                                    verbose=False,
                                                    multip_fc=multip_fc,
                                                    attention=attention,
                                                    transform_query_size=opt.transform_query_size)

        novel_embeds = dummy_classifier.embed.detach().cuda()


        # Update the trained classifier of the network to accommodate for the new classes
        classifier.embed = nn.Parameter(torch.cat([embed, novel_embeds], 0),
                                        requires_grad=False) # TODO:CHECK DIM.
        if attention:
            classifier.transform_W_output = nn.Parameter(torch.cat([classifier.transform_W_output, dummy_classifier.weight.detach().cuda()], 0), requires_grad=True)

        # Validate before training.
        test_acc, test_acc_top5, test_loss, _ = validate_fine_tune(query_xs, query_ys_id, net, criterion, opt)
        print('{:25} {:.4f}\n'.format("Novel incremental acc before fine-tune:",test_acc.item()))


        # Evaluate base samples before updating the network
#         acc_base_ = eval_base(net, base_val_loader, criterion, vocab_all) # where to update df for this evaluations???

#         print('{:25} {:.4f}\n'.format("Base incremental acc before fine-tune:",acc_base_))

        # Retrieve original transform_W if regularization
#         orig_transform_W = ckpt['model']['classifier.transform_W'] if opt.lmbd_reg_transform_w else None
#         ipdb.set_trace()


        # routine: fine-tuning for novel classes
        train_loss = 15
        epoch = 1

        # optimizer
        if opt.adam:
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=opt.learning_rate,
                                         weight_decay=0.0005)
        else:
            optimizer = torch.optim.SGD(net.parameters(),
                                  lr=opt.learning_rate,
                                  momentum=opt.momentum,
                                  weight_decay=opt.weight_decay) # TODO anything to load from ckpt?


        while train_loss > opt.target_train_loss or epoch < opt.novel_epochs + 1:
            freeze_backbone_weights(net, opt, epoch)
            net.train()
            train_acc, train_loss = fine_tune_novel(epoch,
                                                    support_xs,
                                                    support_ys_id,
                                                    net,
                                                    criterion,
                                                    optimizer,
                                                    orig_classifier_weights,
                                                    opt)
            test_acc, test_acc_top5, test_loss, query_ys_pred = validate_fine_tune(query_xs,
                                                                                   query_ys_id,
                                                                                   net,
                                                                                   criterion,
                                                                                   opt)
            if vis and idx == 0:
                novel_info = [(idx, vocab_all[query_ys_id[i]], False, vocab_all[query_ys_pred[i]],
                               image_formatter(novelimgs[i,:,:,:]))  for i in range(len(query_ys_id))]
                df = df.append(pd.DataFrame(novel_info, columns=df.columns), ignore_index=True)
            epoch += 1

        try:
            base_batch = next(base_valloader_it)
        except StopIteration:
            base_valloader_it = iter(base_val_loader)
            base_batch = next(base_valloader_it)

        # Evaluate base samples with the updated network
        if vis and idx == 0:
            acc_base_ = eval_base(net, base_batch, criterion, vocab_all, df=df)
        else:
            acc_base_ = eval_base(net, base_batch, criterion, vocab_all)

        # Compute avg of base and novel.
        avg_score = (acc_base_ + test_acc.item())/2

        # Update trackers.
        acc_base.append(acc_base_)
        acc_novel.append(test_acc.item())
        running_avg.append(avg_score)

        print('\n{:25} {:}\n'
              '{:25} {:}\n'
              '{:25} {:}\n'
              '{:25} {:.4f}\n'
              '{:25} {:.4f}\n'
              '{:25} {:.4f}\n'
              '{:25} {:.4f}'.format(
                  "Novel classes are:",
                  novel_ids,
                  "Human labels are:",
                  vocab_novel,
                  "Novel training epochs:",
                  epoch-1,
                  "Novel incremental acc:",
                  test_acc.item(),
                  "Base incremental acc:",
                  acc_base_,
                  "Average:",
                  avg_score,
                  "Running Average:",
                  np.mean(running_avg)))

    if vis:
        return df
    else:
        return mean_confidence_interval(acc_novel), mean_confidence_interval(acc_base)
