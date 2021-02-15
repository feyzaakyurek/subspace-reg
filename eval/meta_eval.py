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

def incremental_test(net, testloader, val_loader, alpha, use_logit=False, is_norm=True, classifier='LR', vis=False):
    acc_novel = []
    acc_base = []
    if vis:
        df = pd.DataFrame(columns=['idx', 'class', 'isbase', 'predicted', 'img'])

    assert classifier == 'LR'

    net = net.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader)):
            if idx > 5:
                break
            support_xs, support_ys, query_xs, query_ys = drop_a_dim(data)
            novelimgs = query_xs.detach().numpy()

            if use_logit:
                support_features = net(support_xs.cuda()).view(support_xs.size(0), -1)
                query_features   = net(query_xs.cuda()).view(query_xs.size(0), -1)
            else:
                feat_support, _  = net(support_xs.cuda(), is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _    = net(query_xs.cuda(), is_feat=True)
                query_features   = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            # Get sorted numeric labels, create a mapping that maps the order to actual label
            vocab_base, vocab_all, vocab_novel, orig2id = get_vocabs(val_loader, testloader, support_ys)
            query_ys_id   = [orig2id[y] for y in query_ys]
            support_ys_id = [orig2id[y] for y in support_ys]

            # Convert numpy
            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            # Fit LR
            clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
            clf.fit(support_features, support_ys_id)

            # Prediction scores for novel query samples #TODO: support features are normalized??
            query_ys_pred_scores = np.concatenate([alpha * query_features,
                                                   (1-alpha) * query_features @ clf.coef_.transpose()], 1) # 75x64, (75x64 @ 64x5) # 75 x 69
            query_ys_pred = np.argmax(query_ys_pred_scores, 1)


            if vis and idx == 0:
                novel_info = [(idx, vocab_all[query_ys_id[i]], False, vocab_all[query_ys_pred[i]],  image_formatter(novelimgs[i,:,:,:]))  for i in range(len(query_ys_id))]
                df = df.append(pd.DataFrame(novel_info, columns=df.columns), ignore_index=True)
            acc_novel.append(metrics.accuracy_score(query_ys_id, query_ys_pred))

            # Evaluate base class samples.
            acc_base_ = []
            for idb, (input, target, _) in enumerate(val_loader):
                baseimgs = input.detach().numpy()
                input = input.float()
                base_query_features = net(input.cuda()).view(input.size(0), -1) # 32x64
                base_query_features = base_query_features.detach().cpu().numpy()
                target = target.view(-1).detach().numpy()
                base_query_ys_pred_scores = np.concatenate([alpha * base_query_features,
                                                            (1-alpha) * (base_query_features @ clf.coef_.transpose())], 1) # 32x64, (32x64 @ 64x5) # 32 x 69
                base_query_ys_pred = np.argmax(base_query_ys_pred_scores, 1)
                base_accs = metrics.accuracy_score(base_query_ys_pred, target)
                acc_base_.append(base_accs)
                if vis and idx == 0:
                    base_info = [(idx, vocab_all[target[i]], True, vocab_all[base_query_ys_pred[i]], image_formatter(baseimgs[i,:,:,:]))  for i in range(len(target))]
                    df = df.append(pd.DataFrame(base_info, columns=df.columns), ignore_index=True)
            acc_base.append(np.mean(acc_base_))
        if vis:
            return df
        else:
            return mean_confidence_interval(acc_novel), mean_confidence_interval(acc_base)

def rule_based_determine_base(novel_entropy, base_entropy):
    b1 = 0.4
    b2 = 0.02
    n1 = 0.95
    n2 = 0.7
    def rule(novel_ent, base_ent):
        if base_ent > b1:
            return False
        elif base_ent > b2:
            if novel_ent > n1:
                return True
            else:
                return False
        else:
            return True
#             if novel_ent > n2:
#                 return True
#             else:
#                 return False

    return np.array(list(map(rule, novel_entropy.tolist(), base_entropy.tolist()))) * 1.

def meta_hierarchical_incremental_test(net, testloader, baseloader, opt, is_norm=False, classifier='LR'):
    net = net.eval()
    novel_acc = []
    base_acc = []
    iteration = 0
    baseloader_it = iter(baseloader)

    # To save preds.
#     preds_df = pd.DataFrame(columns = ["Episode", "Gold", "Prediction"])

    # Create entropy df and parse vocabs.
#     entropy_df = pd.DataFrame(columns = ["Episode", "Type", "Class", "Label", "BaseEntropy", "NovelEntropy"])
#     vocab_novel = [(i,name) for i,name in enumerate(testloader.dataset.label2human) if name != '']
#     vocab_base = [(i,name) for i,name in enumerate(baseloader.dataset.label2human) if name != '']
#     vocabs = vocab_novel + vocab_base
#     vocab = dict(vocabs)


    with torch.no_grad():
        for idx, data in enumerate(testloader):
            support_xs, support_ys, query_xs, query_ys = drop_a_dim(data)
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()

            # Get the backbone features of the support and query sets of novel classes.
            support_features = net(support_xs.cuda()).view(support_xs.size(0), -1)
            query_features = net(query_xs.cuda()).view(query_xs.size(0), -1)

            # Normalize -- good for LR.
            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            # Get base samples for testing.
            base_data = get_batch_cycle(baseloader_it, baseloader)

            # Check the sizes of support and base.
            base_support_xs, base_support_ys, base_query_xs, base_query_ys = drop_a_dim(base_data)
            base_support_xs = base_support_xs.cuda()
            base_support_features = net(base_support_xs)
            base_query_xs = base_query_xs.cuda()
            base_query_features = net(base_query_xs)

            if is_norm:
                base_query_features = normalize(base_query_features)
                base_support_features = normalize(base_support_features)

            # Compute softmax activations.
            softmax = nn.Softmax(dim=1)
            support_features = softmax(support_features)
            query_features = softmax(query_features)
            base_support_features = softmax(base_support_features)
            base_query_features = softmax(base_query_features)

            # Send to cpu.
            base_query_features = base_query_features.detach().cpu().numpy()
            base_support_features = base_support_features.detach().cpu().numpy()
            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()


            # Fit classifier on solely novel support set.
            if classifier == 'LR':
                clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_features, support_ys)
            else:
                raise NotImplementedError('Classifier not supported: {}'.format(classifier))


            # Compute normalized entropy of both base scores and novel scores.
            novel_query_novel_probs = clf.predict_proba(query_features) # alternatively one can use 640 features.
            novel_query_base_probs = query_features
            novel_support_novel_probs = clf.predict_proba(support_features) # alternatively one can use 640 features.
            novel_support_base_probs = support_features
            base_support_novel_probs = clf.predict_proba(base_support_features)
            base_support_base_probs = base_support_features
            base_query_novel_probs = clf.predict_proba(base_query_features)
            base_query_base_probs = base_query_features

            assert opt.dataset == "miniImageNet" # 64 below.
            novel_query_novel_entropy = - np.sum(novel_query_novel_probs * np.log(novel_query_novel_probs) / np.log(opt.n_shots), axis=1) # should be of batch size
            novel_query_base_entropy = - np.sum(novel_query_base_probs * np.log(novel_query_base_probs) / np.log(64), axis=1)
            novel_support_novel_entropy = - np.sum(novel_support_novel_probs * np.log(0.00001+ novel_support_novel_probs) / np.log(opt.n_shots), axis=1) # should be of batch size
            novel_support_base_entropy = - np.sum(novel_support_base_probs * np.log(0.00001+ novel_support_base_probs) / np.log(64), axis=1)
            base_query_novel_entropy = - np.sum(base_query_novel_probs * np.log(base_query_novel_probs) / np.log(opt.n_shots), axis=1)
            base_query_base_entropy = - np.sum(base_query_base_probs * np.log(base_query_base_probs) / np.log(64), axis=1)
            base_support_novel_entropy = - np.sum(base_support_novel_probs * np.log(base_support_novel_probs) / np.log(opt.n_shots), axis=1)
            base_support_base_entropy = - np.sum(base_support_base_probs * np.log(base_support_base_probs) / np.log(64), axis=1)

#             novel_list = [pd.Series([idx, "Novel", query_ys[i], vocab[query_ys[i]], novel_query_base_entropy[i], ent], index=entropy_df.columns) for i,ent in enumerate(novel_query_novel_entropy)]
#             novel_support_list = [pd.Series([idx, "Novel_Support", support_ys[i], vocab[support_ys[i]], novel_support_base_entropy[i], ent], index=entropy_df.columns) for i,ent in enumerate(novel_support_novel_entropy)]
#             base_list = [pd.Series([idx, "Base", base_query_ys[i], vocab[base_query_ys[i]], base_query_base_entropy[i], ent], index=entropy_df.columns) for i,ent in enumerate(base_query_novel_entropy)]
#             entropy_df = entropy_df.append(novel_list, ignore_index=True)
#             entropy_df = entropy_df.append(novel_support_list, ignore_index=True)
#             entropy_df = entropy_df.append(base_list, ignore_index=True)


            # Check support ys (is it 0, 1, ..)
#             thres = np.min(support_features)
#             novel_is_base = (np.sum(query_features > thres, 1) >= 1) * 1.
#             base_is_base = (np.sum(base_query_features > thres, 1) >= 1) * 1
            novel_is_base = rule_based_determine_base(novel_query_novel_entropy, novel_query_base_entropy)
            base_is_base = rule_based_determine_base(base_query_novel_entropy, base_query_base_entropy)

            # Here print the accuracy of your rule.
            print("Based on the rule novels were marked novel {} of the time.".format(np.mean(novel_is_base == 0)))
            print("Based on the rule bases were marked base {} of the time.".format(np.mean(base_is_base == 1)))

            # Here fit a simple LR model on your query samples and print its training accuracy.
            X_novel = np.stack((novel_support_novel_entropy, novel_support_base_entropy), axis=1)
            X_base = np.stack((base_support_novel_entropy, base_support_base_entropy), axis=1)
            X = np.concatenate((X_novel, X_base), axis=0)
            y = np.concatenate((np.repeat(0, len(support_ys)), np.repeat(1, len(base_support_ys))), axis=0)
            clf_ent = LogisticRegression(random_state=0).fit(X, y)

            X_novel = np.stack((novel_query_novel_entropy, novel_query_base_entropy), axis=1)
            X_base = np.stack((base_query_novel_entropy, base_query_base_entropy), axis=1)
            y = np.concatenate((np.repeat(0, len(query_ys)), np.repeat(1, len(base_query_ys))), axis=0)

            novelbase_preds_ent_novel = clf_ent.predict(X_novel)
            novelbase_preds_ent_base = clf_ent.predict(X_base)
            novel_is_base = novelbase_preds_ent_novel
            base_is_base = novelbase_preds_ent_base
            print("Based on the LR on entropies novels were marked novel {} of the time.".format(np.mean(novelbase_preds_ent_novel == 0)))
            print("Based on the LR on entropies bases were marked base {} of the time.".format(np.mean(novelbase_preds_ent_base == 1)))


            # Preds _if_ a given sample is novel.
            novel_query_novel_pred = clf.predict(query_features)
            base_query_novel_pred = clf.predict(base_query_features)

            # Preds _if_ a given sample is base.
            novel_query_base_pred = np.argmax(query_features, 1)
            base_query_base_pred = np.argmax(base_query_features, 1)

            # Combine novel and base preds based on is_base
            novel_preds = novel_is_base * novel_query_base_pred + (1-novel_is_base) * novel_query_novel_pred
            base_preds = base_is_base * base_query_base_pred + (1-base_is_base) * base_query_novel_pred

            # Compute accuracies
            novel1 = np.mean(novel_preds == query_ys)
            base1 = np.mean(base_preds == base_query_ys)
            novel_acc.append(novel1)
            base_acc.append(base1)


#             temp_df = pd.DataFrame({"Episode": np.repeat(iteration, len(query_ys)+len(base_query_ys)),
#                                     "Gold": np.concatenate((query_ys, base_query_ys),0),
#                                     "Prediction": np.concatenate((novel_preds, base_preds),0).astype(int)})
#             preds_df = pd.concat([preds_df, temp_df], 0)

            print('Evaluation \t'
                  'Iteration {:4d}/{:4d}\t'
                  'Novel@1 {:10.3f}\t'
                  'Base@1 {:10.3f}\t'
                  'Avg@1 {:10.3f}\t'
                  'Running {:10.3f}\t'.format(iteration,
                                            opt.neval_episodes,
                                            novel1,
                                            base1,
                                            (novel1 + base1) / 2,
                                            (np.mean(novel_acc) + np.mean(base_acc))/2))

            iteration += 1
#             if iteration == 5:
#                 preds_df.to_csv("csv_files/hierarchical_preds.csv", index=False)
#                 return (np.mean(novel_acc) + np.mean(base_acc))/2

#     entropy_df.to_csv("csv_files/entropies_normalization.csv", index=False)

    return (np.mean(novel_acc) + np.mean(base_acc))/2
