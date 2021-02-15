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

def zero_shot_incremental_test(net, meta_valloader, base_val_loader, opt, alpha, is_norm=False, vis=False):
    net = net.eval()
    acc_novel = []
    acc_base = []
    if vis:
        df = pd.DataFrame(columns=['idx', 'class', 'isbase', 'predicted', 'img'])
    random_batches = np.random.choice(np.arange(len(meta_valloader)), opt.num_novel_combs, False)
    with torch.no_grad():
        for idx, data in tqdm(enumerate(meta_valloader)):
            if idx not in random_batches:
                continue
            _,_, query_xs, query_ys = drop_a_dim(data) # Read novel val batch
            novelimgs = query_xs.detach().numpy()

            # Extract features
            feat_query, _ = net(query_xs.cuda(), is_feat=True)
            query_features = feat_query[-1].view(query_xs.size(0), -1)

            # Normalize
            if is_norm:
                query_features = normalize(query_features)

            # Retrieve the vocab
            vocab_base, vocab_all, vocab_novel, orig2id = get_vocabs(base_val_loader, meta_valloader, query_ys)
            novel_ids = np.sort(np.unique(query_ys))
            print("len(vocab): ", len(vocab_all))
            query_ys_id = [orig2id[y] for y in query_ys]

            dim = opt.word_embed_size
            embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim))

            # Create classifier with base and novel class labels.
            classifier = LangLinearClassifier(vocab_all, embed_pth, dim=dim, cdim=640,
                                              bias=opt.lang_classifier_bias, verbose=False,
                                              multip_fc=net.classifier.multip_fc)


            # Replace the random transformations with the trained ones.
            classifier.transform_W = net.classifier.transform_W
            classifier.transform_B = net.classifier.transform_B
            classifier = classifier.cuda()


            # Get the classification scores for novel samples for both base and novel classes.
            novel_ys_pred_scores_ = classifier(query_features).detach().cpu().numpy()

            # Adjust scores by alpha
            novel_ys_pred_scores = np.concatenate([alpha * novel_ys_pred_scores_[:,:-len(vocab_novel)],
                                                  (1-alpha) * novel_ys_pred_scores_[:,-len(vocab_novel):]], 1)

            # Get predictions
            novel_ys_pred = np.argmax(novel_ys_pred_scores, 1)
            acc_novel.append(metrics.accuracy_score(novel_ys_pred, query_ys_id))

            if vis and idx == 0:
                novel_info = [(idx, vocab_all[query_ys_id[i]], False, vocab_all[novel_ys_pred[i]],  image_formatter(novelimgs[i,:,:,:]))  for i in range(len(query_ys_id))]
                df = df.append(pd.DataFrame(novel_info, columns=df.columns), ignore_index=True)
            # Important to evaluate base class samples against different combinations of novel classes.
#             if idx < opt.num_novel_combs:

            acc_base_ = []
            for idb, (input, target, _) in enumerate(base_val_loader):
                imgdata = input.detach().numpy()
                input = input.float().cuda()
                input = input.cuda()
                feat_query, _ = net(input, is_feat=True)
                base_query_features = feat_query[-1].view(input.size(0), -1)
                target = target.view(-1).numpy()

                # Get the classification scores for base samples for both base and novel classes and adjust
                base_ys_pred_scores_ = classifier(base_query_features).detach().cpu().numpy()
                base_ys_pred_scores = np.concatenate([alpha * base_ys_pred_scores_[:,:-len(vocab_novel)],
                                              (1-alpha) * base_ys_pred_scores_[:,-len(vocab_novel):]], 1)
                base_ys_pred = np.argmax(base_ys_pred_scores, 1)
                acc_base_.append(metrics.accuracy_score(base_ys_pred, target))
                if vis and idx == 0:
                    base_info = [(idx, vocab_all[target[i]], True, vocab_all[base_ys_pred[i]], image_formatter(imgdata[i,:,:,:]))  for i in range(len(target))]
                    df = df.append(pd.DataFrame(base_info, columns=df.columns), ignore_index=True)
            acc_base.append(np.mean(acc_base_))
#             else:
    if vis:
        return df
    return mean_confidence_interval(acc_novel), mean_confidence_interval(acc_base)


def zero_shot_test(net, loader, opt, is_norm=True, use_logit=False, novel_only=True, vis=False, **kwargs):
    if vis:
        df = pd.DataFrame(columns=['idx', 'class', 'isbase', 'predicted', 'img'])
    acc_novel = []
    net = net.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(loader)):
            _, _, query_xs, query_ys = drop_a_dim(data)
            novelimgs = query_xs.numpy()
            # Extract features. Q:TODO
            if use_logit:
                query_features = net(query_xs.cuda()).view(query_xs.size(0), -1)
            else:
                feat_query, _  = net(query_xs.cuda(), is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            # Normalize
            if is_norm:
                query_features = normalize(query_features)

            # Get sorted numeric labels, create a mapping that maps the order to actual label
            vocab_base, vocab_all, vocab_novel, orig2id = get_vocabs(base_loader=None, novel_loader=meta_valloader, novel_ids=support_ys)
            query_ys_id = [orig2id[y] for y in query_ys]

            # Retrieve the names of the classes in order
            # human_label_list = [label2human[y] for y in unique_sorted_lbls] # TODO: are you sure?

            if opt.classifier == "lang-linear":
                # Create the language classifier which uses only novel class names' embeddings
                dim = opt.word_embed_size
                embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim))
                classifier = LangLinearClassifier(human_label_list,
                                                  embed_pth,
                                                  dim=dim,
                                                  cdim=640,
                                                  bias=opt.lang_classifier_bias,
                                                  verbose=False,
                                                  multip_fc=net.classifier.multip_fc).cuda()

            else: # Description linear classifier
                embed_pth = os.path.join(opt.description_embed_path,
                                          "{0}_{1}_layer{2}.pickle".format(opt.dataset,
                                                                           opt.desc_embed_model,
                                                                           opt.transformer_layer))
                classifier = LangLinearClassifier(human_label_list,
                                                        embed_pth,
                                                        cdim=640,
                                                        dim=None,
                                                        bias=opt.lang_classifier_bias,
                                                        description=True,
                                                        verbose=False,
                                                        multip_fc=net.classifier.multip_fc).cuda()



            # Replace the transforms with the pretrained ones
            classifier.transform_W = net.classifier.transform_W
            classifier.transform_B = net.classifier.transform_B # TODO

            #classifier.bias = net.classifier.bias # TODO
            # Get predictions for novel samples over novel classes only (should be better than random).
            novel_only_ys_pred_scores = classifier(query_features).detach().cpu().numpy()
            novel_only_ys_pred = np.argmax(novel_only_ys_pred_scores, 1)
            acc_novel.append(metrics.accuracy_score(novel_only_ys_pred, query_ys_id))

            if vis and idx == 0:
                novel_info = [(idx, vocab_all[query_ys_id[i]], False, vocab_all[novel_only_ys_pred[i]],  image_formatter(novelimgs[i,:,:,:]))  for i in range(len(query_ys_id))]
                df = df.append(pd.DataFrame(novel_info, columns=df.columns), ignore_index=True)
        if vis:
            return df
        else:
            return mean_confidence_interval(acc_novel)
