from __future__ import print_function

import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm
import ipdb
import os
import time
import copy

import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from .util import accuracy
from models.resnet_language import LangLinearClassifier

import pandas as pd
from PIL import Image
import io
import base64

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    return x.div(norm)

def image_formatter(im):
    im = ((im / np.max(im, axis=(1,2), keepdims=True)) * 255).astype('uint8').transpose((1,2,0))
    im = Image.fromarray(im)
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG") # TODO: why this is required here ?
    rawBytes.seek(0)  # return to the start of the file
    decoded = base64.b64encode(rawBytes.read()).decode()
    return f'<img src="data:image/jpeg;base64,{decoded}">'

def freeze_backbone_weights(backbone, opt, epoch, exclude=['classifier.transform']):
    if opt.freeze_backbone_at == epoch:
        print("Freezing the backbone.")
        for name, param in backbone.named_parameters():
            param.requires_grad = False
            if any(map(lambda s: name.startswith(s), exclude)): # why not; name in exclude:
                print("Not frozen ", name)
                param.requires_grad = True

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



def get_vocabs(base_loader=None, novel_loader=None, query_ys=None):
    vocab_all = []
    vocab_base = None
    if base_loader is not None:
        label2human_base = base_loader.dataset.label2human
        vocab_base  = [name for name in label2human_base if name != '']
        vocab_all  += vocab_base

    vocab_novel, orig2id = None, None

    if novel_loader is not None:
        novel_ids = np.sort(np.unique(query_ys))
        label2human_novel = novel_loader.dataset.label2human
        vocab_novel = [label2human_novel[i] for i in novel_ids]
        orig2id = dict(zip(novel_ids, len(vocab_base) + np.arange(len(novel_ids))))
        vocab_all += vocab_novel
    print("len(vocab): ", len(vocab_all))
    return vocab_base, vocab_all, vocab_novel, orig2id

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


def drop_a_dim(data): #TODO why do we need this in the first place?
    support_xs, support_ys, query_xs, query_ys = data
    batch_size, _, height, width, channel = support_xs.size()
    support_xs = support_xs.view(-1, height, width, channel)
    query_xs = query_xs.view(-1, height, width, channel)
    support_ys = support_ys.view(-1).detach().numpy() # TODO
    query_ys = query_ys.view(-1).detach().numpy()
    return (support_xs, support_ys, query_xs, query_ys)

def few_shot_language_incremental_test(net, ckpt, criterion, meta_valloader, base_val_loader, opt,  vis=False):
    if vis:
        df = pd.DataFrame(columns=['idx', 'class', 'isbase', 'predicted', 'img'])

    acc_novel = []
    acc_base = []
    basenet = copy.deepcopy(net).cuda()

    if basenet.classifier.multip_fc == 0: # TODO
        print("A LARGE WARNING!!! Loaded multipfc is 0, setting it to {}!!!".format(opt.multip_fc))
        basenet.classifier.multip_fc = nn.Parameter(torch.FloatTensor([opt.multip_fc]), requires_grad=False)
#     ipdb.set_trace()

    embed = basenet.classifier.embed.clone().detach().requires_grad_(False)
    trns  = basenet.classifier.transform_W.clone().detach().requires_grad_(False)
    orig_classifier_weights = embed @ trns if opt.lmbd_reg_transform_w else None

    for idx, data in enumerate(meta_valloader):
        support_xs, support_ys, query_xs, query_ys = drop_a_dim(data)
        novelimgs = query_xs.detach().numpy()

        # Get sorted numeric labels, create a mapping that maps the order to actual label

        vocab_base, vocab_all, vocab_novel, orig2id = get_vocabs(base_val_loader, meta_valloader, support_ys)
        novel_ids = np.sort(np.unique(query_ys))
        query_ys_id = torch.LongTensor([orig2id[y] for y in query_ys])
        support_ys_id = torch.LongTensor([orig2id[y] for y in support_ys])

        net = copy.deepcopy(basenet)
        net.train()
        classifier = net.classifier

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
                                                    multip_fc=opt.multip_fc) # TODO!!

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
                                                    multip_fc=opt.multip_fc)

        novel_embeds = dummy_classifier.embed.detach().cuda()


        # Update the trained classifier of the network to accommodate for the new classes
        classifier.embed = nn.Parameter(torch.cat([embed, novel_embeds], 0),
                                        requires_grad=False) # TODO:CHECK DIM.

        # Validate before training.
        test_acc, test_acc_top5, test_loss, _ = validate_fine_tune(query_xs, query_ys_id, net, criterion, opt)
        print('{:25} {:.4f}\n'.format("Novel incremental acc before fine-tune:",test_acc.item()))


        # Evaluate base samples before updating the network
        acc_base_ = eval_base(net, base_val_loader, criterion, vocab_all) # where to update df for this evaluations???

        print('{:25} {:.4f}\n'.format("Base incremental acc before fine-tune:",acc_base_))

        # Retrieve original transform_W if regularization
#         orig_transform_W = ckpt['model']['classifier.transform_W'] if opt.lmbd_reg_transform_w else None
#         ipdb.set_trace()


        # routine: fine-tuning for novel classes
        train_loss = 15
        epoch = 1
        freeze_backbone_weights(net, opt, epoch)
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
                novel_info = [(idx, vocab_all[query_ys_id[i]], False, vocab_all[query_ys_pred[i]],  image_formatter(novelimgs[i,:,:,:]))  for i in range(len(query_ys_id))]
                df = df.append(pd.DataFrame(novel_info, columns=df.columns), ignore_index=True)
            epoch += 1
        acc_novel.append(test_acc.item())

        # Evaluate base samples with the updated network
        if vis and idx == 0:
            acc_base_ = eval_base(net, base_val_loader, criterion, vocab_all, df=df)
        else:
            acc_base_ = eval_base(net, base_val_loader, criterion, vocab_all)


        acc_base.append(acc_base_)

        avg_score = (acc_base_ + test_acc.item())/2

        print('\n{:25} {:}\n'
              '{:25} {:}\n'
              '{:25} {:}\n'
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
                  avg_score))

    if vis:
        return df
    else:
        return mean_confidence_interval(acc_novel), mean_confidence_interval(acc_base)




def fine_tune_novel(epoch, support_xs, support_ys_id, net, criterion, optimizer, orig_classifier_weights, opt):
    """One epoch training, single batch training."""
#     ipdb.set_trace()
    support_xs = support_xs.float().cuda()
    support_ys_id = support_ys_id.cuda()

    # Compute output
    output = net(support_xs)
    loss = criterion(output, support_ys_id)
#     if opt.lmbd_reg_transform_w is not None:
#         loss = loss + opt.lmbd_reg_transform_w * torch.norm(net.classifier.transform_W - orig_transform_W)

    if opt.lmbd_reg_transform_w is not None:
        len_vocab,_ = orig_classifier_weights.size()
        loss = loss + opt.lmbd_reg_transform_w * torch.norm(net.classifier.weight()[:len_vocab,:] - orig_classifier_weights)

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

def eval_base(net, base_val_loader, criterion, vocab_all, df=None):
    acc_base_ = []
    net.eval()
    with torch.no_grad():
        for idb, (input, target, _) in enumerate(base_val_loader):
            imgdata = input.detach().numpy()
            input  = input.float().cuda()
            output = net(input).detach().cpu()

            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc_base_.append(acc1[0].item())
            ys_pred = torch.argmax(output, dim=1).numpy()
            if df is not None:
                base_info = [(idx, vocab_all[target[i]], True, vocab_all[ys_pred[i]], image_formatter(imgdata[i,:,:,:]))  for i in range(len(target))]
                df = df.append(pd.DataFrame(base_info, columns=df.columns), ignore_index=True)
    return np.mean(acc_base_)

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



def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR'):
    net = net.eval()
    acc = []

    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader)):
            support_xs, support_ys, query_xs, query_ys = drop_a_dim(data)
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()

            if use_logit:
                support_features = net(support_xs.cuda()).view(support_xs.size(0), -1)
                query_features = net(query_xs.cuda()).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs.cuda(), is_feat=True)
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs.cuda(), is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

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
