from __future__ import print_function

import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm
import ipdb
import os

import torch
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from .util import accuracy
from models.resnet_language import LangLinearClassifier


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def zero_shot_incremental_test(net, meta_valloader, base_val_loader, opt, alpha, is_norm=False):
    net = net.eval()
    acc_novel = []
    acc_base = []
    label2human_base = base_val_loader.dataset.label2human
    label2human_novel = meta_valloader.dataset.label2human
    
    random_batches = np.random.choice(np.arange(len(meta_valloader)), opt.num_novel_combs, False)
    with torch.no_grad():
        for idx, data in tqdm(enumerate(meta_valloader)):
            if idx not in random_batches:
                continue
            _,_, query_xs, query_ys = data # Read novel val batch
            query_xs = query_xs.cuda()
            batch_size, _, height, width, channel = query_xs.size()
            query_xs = query_xs.view(-1, height, width, channel)
            query_ys = query_ys.view(-1).numpy()
            
            # Extract features
            feat_query, _ = net(query_xs, is_feat=True)
            query_features = feat_query[-1].view(query_xs.size(0), -1)
              
            # Normalize
            if is_norm:
                query_features = normalize(query_features)
                
            # Get sorted numeric labels, 
            unique_sorted_lbls = np.sort(np.unique(query_ys))

            # Retrieve the vocab
            vocab_base = [name for name in label2human_base if name != ''] #TODO, ORDER??
            vocab_novel = [label2human_novel[id] for id in unique_sorted_lbls]
            print(f"Novel class labels are {vocab_novel}.")
            vocab = vocab_base + vocab_novel
            dim = opt.word_embed_size
            embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim))
            
            # Create classifier with base and novel class labels.
            classifier = LangLinearClassifier(vocab, embed_pth, dim=dim, cdim=640,  
                                              bias=opt.lang_classifier_bias, verbose=False)
            
            # Replace the random transformations with the trained ones.
            classifier.transform_W = net.classifier.transform_W
            classifier.transform_B = net.classifier.transform_B
            classifier = classifier.cuda()
            
            # Create a mapping that maps the order to actual label
            orig2id = dict(zip(unique_sorted_lbls, len(vocab_base) + np.arange(len(unique_sorted_lbls))))
            query_ys_id = [orig2id[y] for y in query_ys]
            
            # Get the classification scores for novel samples for both base and novel classes.
            novel_ys_pred_scores_ = classifier(query_features).detach().cpu().numpy()
            
            # Adjust scores by alpha
            novel_ys_pred_scores = np.concatenate([alpha * novel_ys_pred_scores_[:,:-len(vocab_novel)], 
                                                  (1-alpha) * novel_ys_pred_scores_[:,-len(vocab_novel):]], 1)
            
            # Get predictions
            novel_ys_pred = np.argmax(novel_ys_pred_scores, 1)
            acc_novel.append(metrics.accuracy_score(novel_ys_pred, query_ys_id))
            
            # Important to evaluate base class samples against different combinations of novel classes. 
#             if idx < opt.num_novel_combs:
            acc_base_ = []
            for idb, (input, target, _) in enumerate(base_val_loader):
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
            acc_base.append(np.mean(acc_base_))
#             else:
    return mean_confidence_interval(acc_novel), mean_confidence_interval(acc_base)
                
def zero_shot_test(net, loader, opt, use_logit=False, is_norm=True, novel_only=True, **kwargs):
    net = net.eval()
    acc_novel = []
    label2human = loader.dataset.label2human
    
    with torch.no_grad():
        for idx, data in tqdm(enumerate(loader)):
            _, _, query_xs, query_ys = data
            query_xs = query_xs.cuda()
            batch_size, _, height, width, channel = query_xs.size()
            query_xs = query_xs.view(-1, height, width, channel)
            query_ys = query_ys.view(-1).numpy()
            
            # Extract features. Q:TODO 
            if use_logit:
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)
            
            # Normalize
            if is_norm:
                query_features = normalize(query_features)
            
            # Get sorted numeric labels, create a mapping that maps the order to actual label
            unique_sorted_lbls = np.sort(np.unique(query_ys))
            orig2id = dict(zip(unique_sorted_lbls, np.arange(len(unique_sorted_lbls))))
            query_ys_id = [orig2id[y] for y in query_ys]

            # Retrieve the names of the classes in order
            human_label_list = [label2human[y] for y in unique_sorted_lbls] # TODO: are you sure?
            
            # Create the language classifier which uses only novel class names' embeddings
            dim = opt.word_embed_size
            embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim))
            classifier = LangLinearClassifier(human_label_list, embed_pth, dim=dim, cdim=640,  
                                              bias=opt.lang_classifier_bias, verbose=False)
            
            # Replace the transforms with the pretrained ones
            classifier.transform_W = net.classifier.transform_W
            classifier.transform_B = net.classifier.transform_B # TODO
            #classifier.bias = net.classifier.bias # TODO
            classifier = classifier.cuda()
    
            # Get predictions for novel samples over novel classes only (should be better than random).
            novel_only_ys_pred_scores = classifier(query_features).detach().cpu().numpy()
            novel_only_ys_pred = np.argmax(novel_only_ys_pred_scores, 1)
            acc_novel.append(metrics.accuracy_score(novel_only_ys_pred, query_ys_id))
      
        return mean_confidence_interval(acc_novel)#, mean_confidence_interval(acc_base)

def incremental_test(net, testloader, val_loader, alpha, use_logit=False, is_norm=True, classifier='LR'):
    net = net.eval()
    acc_novel = []
    acc_base = []
    
    assert classifier == 'LR' 
    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader)):
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, height, width, channel = support_xs.size()
            support_xs = support_xs.view(-1, height, width, channel)
            query_xs = query_xs.view(-1, height, width, channel)

            if use_logit:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            # Convert numpy
            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()
            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            # Fit LR
            clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
            clf.fit(support_features, support_ys)

            # Prediction scores for novel query samples #TODO: support features are normalized??
            query_ys_pred_scores = np.concatenate([alpha * query_features, 
                                                   (1-alpha) * query_features @ clf.coef_.transpose()], 1) # 75x64, (75x64 @ 64x5) # 75 x 69
            query_ys_pred = np.argmax(query_ys_pred_scores, 1)
            
            # Evaluate base class samples.
            acc_base_ = []
            for idb, (input, target, _) in enumerate(val_loader):
                input = input.float()
                if torch.cuda.is_available():
                    input = input.cuda()
                base_query_features = net(input).view(input.size(0), -1) # 32x64
                base_query_features = base_query_features.detach().cpu().numpy()
                target = target.view(-1).numpy()

                base_query_ys_pred_scores = np.concatenate([alpha * base_query_features, 
                                                            (1-alpha) * (base_query_features @ clf.coef_.transpose())], 1) # 32x64, (32x64 @ 64x5) # 32 x 69
                base_query_ys_pred = np.argmax(base_query_ys_pred_scores, 1)
                acc_base_.append(metrics.accuracy_score(base_query_ys_pred, target))
                    
            acc_base.append(np.mean(acc_base_))
            acc_novel.append(metrics.accuracy_score(query_ys, query_ys_pred))
            
            if idx >= 50:
                return mean_confidence_interval(acc_novel), mean_confidence_interval(acc_base)


def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR'):
    net = net.eval()
    acc = []

    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader)):
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, height, width, channel = support_xs.size()
            support_xs = support_xs.view(-1, height, width, channel)
            query_xs = query_xs.view(-1, height, width, channel)

            if use_logit:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            if classifier == 'LR':
                clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
            elif classifier == 'NN':
                query_ys_pred = NN(support_features, support_ys, query_features)
            elif classifier == 'Cosine':
                query_ys_pred = Cosine(support_features, support_ys, query_features)
            else:
                raise NotImplementedError('classifier not supported: {}'.format(classifier))

            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    return mean_confidence_interval(acc)


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred
