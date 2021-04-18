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
import itertools

import torch
import torch.nn as nn

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from .util import accuracy, NN, normalize, image_formatter, \
mean_confidence_interval, Cosine, get_vocabs, drop_a_dim, \
get_optim, get_batch_cycle, freeze_backbone_weights, \
AverageMeter, log_episode

from models.resnet_language import LangLinearClassifier, LangPuller, Pusher

import pandas as pd
from PIL import Image
import io
import base64


def validate_fine_tune(query_xs, query_ys_id, net, criterion, opt, epoch):
    net.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            query_xs = query_xs.cuda()
            query_ys_id = query_ys_id.cuda()

            # compute output
            output = net(query_xs)
            loss = criterion(output, query_ys_id)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, query_ys_id, topk=(1, 5))
            query_ys_pred = torch.argmax(output, dim=1).detach().cpu().numpy()
#             if opt.verbose:
            if epoch % 10 == 0:
                print('Test \t'
                      'Loss {:10.4f}\t'
                      'Acc@1 {:10.3f}\t'
                      'Acc@5 {:10.3f}'.format(
                       loss.item(), acc1[0], acc5[0]))

    return acc1[0], acc5[0], loss.item(), query_ys_pred

def eval_base(net, base_batch, criterion, vocab_all=None, df=None, return_preds=False):
    acc_base_ = []
    net.eval()
    with torch.no_grad():
        input, target, _ = base_batch
        input = input.squeeze(0).cuda()
        target = target.squeeze(0).cuda()
        output = net(input) 
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

def few_shot_finetune_incremental_test(net, ckpt, criterion, meta_valloader, base_val_loader, opt,  vis=False, base_support_loader=None):
    if vis:
        df = pd.DataFrame(columns=['idx', 'class', 'isbase', 'predicted', 'img'])
    if opt.track_weights:
        track_weights = pd.DataFrame(columns=["episode", "type", "label", "class", "fine_tune_epoch", "classifier_weight"])
    if opt.track_label_inspired_weights:
        track_inspired = pd.DataFrame(columns=["episode", "label", "fine_tune_epoch", "inspired_weight"])
    
    # Create meters.
    acc_novel, acc_base = [AverageMeter() for _ in range(2)]
    weighted_avg_l = []
    acc_novel_list = []
    acc_base_list = []
    
    # Used for creation of confusion matrices.
    # preds_df = pd.DataFrame(columns = ["Episode", "Gold", "Prediction"])
    
    # Reset seeds.
    torch.manual_seed(opt.set_seed)
    np.random.seed(opt.set_seed)
    
    # Pretrained backbone, net will be reset before every episode.
    basenet = copy.deepcopy(net).cuda()
    base_weight, base_bias = basenet._get_base_weights()
    
    # Loaders for fine tuning and testing.
    base_valloader_it = itertools.cycle(iter(base_val_loader))
    meta_valloader_it = itertools.cycle(iter(meta_valloader))
    if base_support_loader is not None:
        base_support_it = itertools.cycle(iter(base_support_loader))
        # Use the same set of examples for every episode.
        # i.e. keep a fixed set of base examplars in memory.
        base_support_xs, base_support_ys, *_ = drop_a_dim(next(base_support_it))
        
    # Collect a set of query samples.
    novel_query_collection = None # XXX
    novel_query_collection_id = None # XXX
    base_batch = next(base_valloader_it) # XXXX Same base batch every time.
    
    # Initial validation on base samples.
    acc_base_ = eval_base(net, base_batch, criterion)
    weighted_avg_l.append(acc_base_)
    
    # How many episodes/sessions?
    iter_num = opt.neval_episodes
    if opt.continual:
        iter_num = 8
        
    for idx in range(iter_num):
#     for idx, data in enumerate(meta_valloader):
        print("\n**** Iteration {}/{} ****\n".format(idx, opt.neval_episodes))
        
        support_xs, support_ys, query_xs, query_ys = drop_a_dim(next(meta_valloader_it))
#         support_xs, support_ys, query_xs, query_ys = drop_a_dim(data)
        if base_support_loader is not None:
            support_xs = torch.cat([support_xs, base_support_xs],0)
            
        if vis: novelimgs = query_xs.detach().numpy() # for vis

        # Get vocabs for the loaders.
        if idx > 0:
            prev_vocab_base = vocab_base
            prev_vocab_novel = vocab_novel
        vocab_base, vocab_all, vocab_novel, orig2id = get_vocabs(base_val_loader, meta_valloader, query_ys)
        print("Vocab base: ", vocab_base)
        print("Vocab novel: ", vocab_novel)
#         ipdb.set_trace()
        if idx == 0:
            orig_base_num = len(vocab_base)
        if idx > 0:
            vocab_base = prev_vocab_base + prev_vocab_novel
        
        if idx == 1:
            novel_weight_to_reserve = net.classifier.weight.clone().detach()[-opt.n_ways:,:].requires_grad_(False)
            novel_bias_to_reserve = None
            if base_bias is not None:
                novel_bias_to_reserve = net.classifier.bias.clone().detach()[-opt.n_ways:].requires_grad_(False)
            print(f"Novel weight to reserve is of shape {novel_weight_to_reserve.shape} at session {idx}.")
        if idx > 1:
            new_novel_set = net.classifier.weight.clone().detach()[-opt.n_ways:,:].requires_grad_(False)
            novel_weight_to_reserve = torch.cat((novel_weight_to_reserve, new_novel_set), 0)
            if base_bias is not None:
                new_novel_set_bias = net.classifier.bias.clone().detach()[-opt.n_ways:].requires_grad_(False)
                novel_bias_to_reserve = torch.cat((novel_bias_to_reserve, new_novel_set_bias), 0)
            
            print(f"Novel weight to reserve is of shape {novel_weight_to_reserve.shape} at session {idx}.")
        
            
#         if idx > 0:
#             with torch.no_grad():
#                 base_weight = torch.cat((base_weight, 
#                                          net.classifier.weight.clone().detach()[-len(vocab_novel):].requires_grad_(False)), 0)

        # Get sorted numeric labels, create a mapping that maps the order to actual label
        novel_labels = np.sort(np.unique(query_ys)) # true labels of novel samples.
        print("Novel labels: ", novel_labels)
#         orig2id = dict(zip(novel_labels, len(vocab_base) + np.arange(len(novel_labels))))
        
        # Map the labels to their new form.
        query_ys_id = torch.LongTensor([orig2id[y]+idx*opt.n_ways for y in query_ys])
        support_ys_id = torch.LongTensor([orig2id[y]+idx*opt.n_ways for y in support_ys])
        
        # Add the new set of queries to the collection.
        if novel_query_collection_id is None:
            novel_query_collection = query_xs
            novel_query_collection_id = query_ys_id
        else:
            novel_query_collection = torch.cat((novel_query_collection, 
                                                  query_xs), 0)
            novel_query_collection_id = torch.cat((novel_query_collection_id, 
                                                     query_ys_id), 0)

        if base_support_loader is not None:
            support_ys_id = torch.cat([support_ys_id, torch.from_numpy(base_support_ys)])
            
        # Reset the network.
#         net = copy.deepcopy(basenet) XX
        net.train()
        classifier = net.classifier # function of net XX
        
        # Augment the net's classifier to accommodate new classes.
        net.augment_base_classifier_(len(novel_labels))
        
        # Label pulling is a regularization towards the label attractors.
        if opt.label_pull is not None and opt.pulling == "regularize":
            if idx == 0:
                lang_puller = LangPuller(opt, vocab_base, vocab_novel)
            else:
                lang_puller.update_novel_embeds(vocab_novel)
                
            if opt.attraction_override == "mapping_linear_label2image":
                lang_puller.create_pulling_mapping(ckpt[opt.attraction_override])

            pullers = lang_puller(base_weight[:orig_base_num, :])

            if opt.attraction_override == "random_uniform":
                with torch.no_grad():
                    num_base = base_weight.size(0)
                    device = base_weight.device
                    rand_weights = torch.from_numpy(np.random.uniform(0,1,num_base))
                    rand_weights /= torch.sum(rand_weights)
                    rand_weights = rand_weights.float().to(device)
                    pullers = rand_weights @ base_weight


            
        # Push away from the closest base classifier weight.
        if opt.push_away is not None:
            assert base_bias is None
            pusher = Pusher(opt, base_weight)
            
        
        # Validate before training. TODO
        test_acc, *_ = validate_fine_tune(novel_query_collection,
                                          novel_query_collection_id,
                                          net, 
                                          criterion, 
                                          opt,
                                          0)
        
        
        print('{:25} {:.4f}\n'.format("Novel incremental acc before fine-tune:",
                                      test_acc.item()))
        
        

        # Optimizer
        optimizer = get_optim(net, opt) # TODO anything to load from ckpt?

        # Fine tuning epochs.
        train_loss = 15
        epoch = 1
        
        # Stable epochs
        stable_epochs = 0
        while stable_epochs < opt.stable_epochs:
        
#         while (epoch < opt.max_novel_epochs) and (train_loss > opt.target_train_loss or epoch < opt.novel_epochs + 1):
#         while train_loss > opt.target_train_loss or epoch < opt.novel_epochs + 1:

            freeze_backbone_weights(net, opt, epoch, exclude=["classifier"])
#             base_weight = classifier.weight.clone().detach().requires_grad_(False)[:len(vocab_base),:] TODO
            support_xs = support_xs.cuda()
            support_ys_id = support_ys_id.cuda()
            
            # Compute output
            if opt.classifier in ["lang-linear", "description-linear"] and opt.attention is not None:
                output, alphas = net(support_xs, get_alphas=True)
                loss = criterion(output, support_ys_id) + opt.diag_reg * criterion(alphas, support_ys_id)
            else:
                output = net(support_xs)
                loss = criterion(output, support_ys_id)

            
            # Penalize the change in base classifier weights.
            if opt.lmbd_reg_transform_w is not None:
                lmbd_reg = net.regloss(opt.lmbd_reg_transform_w, base_weight, base_bias)
                if epoch % 10 == 0:
                    print("LMBD: ", lmbd_reg.item())
                loss += lmbd_reg
                
            if opt.lmbd_reg_novel is not None and idx > 0:
                lmbd_reg2 = net.reglossnovel(opt.lmbd_reg_novel,
                                             novel_weight_to_reserve,
                                             novel_bias_to_reserve)
                if epoch % 10 == 0:
                    print("LMBDN: ", lmbd_reg.item())
                loss += lmbd_reg2

            if opt.label_pull is not None and opt.pulling == "regularize":
                
                reg = lang_puller.loss1(opt.label_pull, 
                                        pullers,
                                        net.classifier.weight[len(vocab_base):,:])
                if epoch % 10 == 0:
                    print("PULL: ", reg.item())
                loss += reg
                
                
            if opt.push_away is not None:
                reg = pusher.loss1(opt.push_away, 
                                   net.classifier.weight[len(vocab_base):,:])
                print("PUSH: ", reg.item())
                loss += reg

            # Train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            

            with torch.no_grad():
                # Check if training converges
                if abs(loss.item() - train_loss) < opt.convergence_epsilon:
                    stable_epochs += 1
                else:
                    stable_epochs = 0
                
                acc1, acc5 = accuracy(output, support_ys_id, topk=(1,5))
                train_acc, train_loss = acc1[0], loss.item()
                if epoch % 10 == 0:
                    print('=======Novel Epoch {}=======\n'
                          'Train\t'
                          'Loss {:10.4f}\t'
                          'Acc@1 {:10.3f}\t'
                          'Acc@5 {:10.3f}'.format(
                           epoch, loss.item(), acc1[0], acc5[0]))
                


            test_acc, test_acc_top5, test_loss, query_ys_pred = validate_fine_tune(novel_query_collection,
                                                                                  novel_query_collection_id,
                                                                                  net, 
                                                                                  criterion, 
                                                                                  opt,
                                                                                  epoch)

            if opt.track_label_inspired_weights:
                inspired_weights = label_inspired_weights.clone().cpu().numpy()
                for k,lbl in enumerate(vocab_novel):
                    track_inspired.loc[len(track_inspired)] = [idx, lbl, epoch, inspired_weights[k]]


            if opt.track_weights:
                classifier_weights = net.classifier.weight.clone().detach().cpu().numpy()
                for k,lbl in enumerate(vocab_base):
                    track_weights.loc[len(track_weights)] = [idx, "base", lbl, vocab_base[k], 
                                                             epoch, classifier_weights[k]]
                len_base = len(vocab_base)
                for k,lbl in enumerate(vocab_novel):
                    track_weights.loc[len(track_weights)] = [idx, "novel", lbl, vocab_novel[k], 
                                                             epoch, classifier_weights[len_base+k]]


            if vis and idx == 0:
                novel_info = [(idx, vocab_all[query_ys_id[i]], False, vocab_all[query_ys_pred[i]],
                               image_formatter(novelimgs[i,:,:,:]))  for i in range(len(query_ys_id))]
                df = df.append(pd.DataFrame(novel_info, columns=df.columns), ignore_index=True)
            
            epoch += 1

        
        # Evaluate base samples with the updated network
#         base_batch = next(base_valloader_it) XXXX Same base batch every time.
        vis_condition = (vis and idx == 0)
        acc_base_ = eval_base(net, 
                              base_batch, 
                              criterion, 
                              vocab_all = vocab_all if vis_condition else None, 
                              df= df if vis_condition else None)
 
        # Update meters.
        acc_base.update(acc_base_)
        acc_novel.update(test_acc.item())


        # Little last-mile pull here.
        if opt.label_pull is not None and opt.pulling == "last-mile": #TODO
            pull_acc_base_ = 0
            pull_test_acc = 0
            with torch.no_grad():
                label_inspired_weights = lang_puller(classifier.weight[:len(vocab_base)])
                novel_weights_pulled = classifier.weight[len(vocab_base):,:] + \
                                       opt.label_pull * (label_inspired_weights - classifier.weight[len(vocab_base):,:])
                classifier.weight = nn.Parameter(torch.cat([base_weight,
                                                            novel_weights_pulled], 0))


            pull_acc_base_ = eval_base(net, base_batch, criterion)
            pull_test_acc, *_ = validate_fine_tune(query_xs,
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

            
        w1 = 60 if opt.dataset == "miniImageNet" else 200 # opt.test_base_batch_size
        w2 = len(vocab_base) + len(vocab_novel) - 60
        weighted_avg = (w1*acc_base_ + w2*test_acc.item())/(w1+w2)
        weighted_avg_l.append(round(weighted_avg,2))
        acc_novel_list.append(round(test_acc.item(),2))
        acc_base_list.append(round(acc_base_,2))
        

        print(f"***Running weighted avg: {weighted_avg}")
        
        # Log episode results.
        log_episode(novel_labels,
                    vocab_novel,
                    epoch,
                    test_acc.item(),
                    acc_base_,
                    acc_base.avg,
                    acc_novel.avg)
        
        
#         ######## To save preds (temporary, comment out below later) ########
#         _, base_query_ys, *_ = base_batch
#         base_query_ys = base_query_ys.squeeze(0)
#         acc_base_, base_preds  = eval_base(net, base_batch, criterion, vocab_all, return_preds=True)
#         id2orig = {}

#         for k,v in orig2id.items():
#             id2orig[v] = k
#         print("!!! Mini imagenet hard coded 64. !!")
#         query_ys_pred = [id2orig[k.item()] if k>=64 else k.item() for k in query_ys_pred]
#         base_preds = [id2orig[k.item()] if k>=64 else k.item() for k in base_preds]

#         temp_df = pd.DataFrame({"Episode": np.repeat(idx, len(query_ys)+len(base_query_ys)),
#                                 "Gold": np.concatenate((query_ys, base_query_ys),0),
#                                 "Prediction": np.concatenate((query_ys_pred, base_preds),0).astype(int)})
#         preds_df = pd.concat([preds_df, temp_df], 0)

#         if idx == 5:
#             preds_df.to_csv("csv_files/finetuning_preds.csv", index=False)
#             exit(0)
#         ######## To save preds (temporary, comment out above later) ########


    if opt.track_label_inspired_weights:
        track_inspired.to_csv(f"track_inspired_{opt.eval_mode}_pulling_{opt.pulling}_{opt.label_pull}_target_loss_{opt.target_train_loss}_synonyms_{opt.use_synonyms}.csv", index=False)

    if opt.track_weights:
        track_weights.to_csv(f"track_weights_{opt.eval_mode}_pulling_{opt.pulling}_{opt.label_pull}_target_loss_{opt.target_train_loss}_synonyms_{opt.use_synonyms}.csv", index=False)

    if vis:
        return df
    else:
        print("Overall continual accuracies: ", weighted_avg_l)
        print("Novel only incremental: ", acc_novel_list)
        print("Base only incremental: ", acc_base_list)
        return acc_novel.avg, acc_base.avg

def few_shot_language_incremental_test(net, ckpt, criterion, meta_valloader, base_val_loader, opt, vis=False):
    if vis:
        df = pd.DataFrame(columns=['idx', 'class', 'isbase', 'predicted', 'img'])
        
    # Backbone classifier should've been trained with attention.
    attention = opt.attention
    assert net.classifier.attention == attention
    
    # Create meters.
    acc_novel, acc_base = [AverageMeter() for _ in range(2)]
    
    # Save initial backbone.
    basenet = copy.deepcopy(net).cuda()
    base_weight, base_bias = basenet._get_base_weights()
    
    
    if basenet.classifier.multip_fc == 0: # TODO
        raise ValueError("We shouldn't use this backbone. It's multip_fc is unknown.") 

    embed = basenet.classifier.embed.clone().detach().requires_grad_(False) # XXXX

    # XXX
    if attention:
        if opt.lmbd_reg_transform_w:
            orig_classifier_weights = basenet.classifier.transform_W_output.clone().detach().requires_grad_(False)
        else:
            orig_classifier_weights = None
    else:
        trns = basenet.classifier.transform_W.clone().detach().requires_grad_(False)
        orig_classifier_weights = embed @ trns if opt.lmbd_reg_transform_w else None
        print("Retrieved original classifier weights.")

    # Create iterators.
    base_valloader_it = itertools.cycle(iter(base_val_loader))
    meta_valloader_it = itertools.cycle(iter(meta_valloader))
    
    for idx in range(opt.neval_episodes):
        print("\n**** Iteration {}/{} ****\n".format(idx, opt.neval_episodes))
        support_xs, support_ys, query_xs, query_ys = drop_a_dim(next(meta_valloader_it))
        novelimgs = query_xs.detach().numpy() # for vis

        # Get sorted numeric labels, create a mapping that maps the order to actual label
        vocab_base, vocab_all, vocab_novel, orig2id = get_vocabs(base_val_loader, meta_valloader, support_ys)
        
        # Map the labels to their new form.
        query_ys_id = torch.LongTensor([orig2id[y] for y in query_ys])
        support_ys_id = torch.LongTensor([orig2id[y] for y in support_ys])

        # Reset the network
        net = copy.deepcopy(basenet)
        classifier = net.classifier
        multip_fc = classifier.multip_fc.detach().clone().cpu().item()
        
        # Where to load the pre-saved embeddings from.
        if opt.classifier == "lang-linear":
            embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, 
                                                                                     opt.word_embed_size))


        elif opt.classifier == "description-linear":
            msg = "Check the classifier augmentation code in model for reading descriptions." 
            raise NotImplementedError(msg)
            embed_pth = os.path.join(opt.description_embed_path,
                                     "{0}_{1}_layer{2}_prefix_{3}.pickle".format(opt.dataset,
                                                                                 opt.desc_embed_model,
                                                                                 opt.transformer_layer,
                                                                                 opt.prefix_label))

        # Update the trained classifier of the network to accommodate for the new classes
        novel_labels = np.sort(np.unique(query_ys))
        net.classifier.augment_classifier_(vocab_novel, embed_pth)

        # Validate before training.
        test_acc, *_ = validate_fine_tune(query_xs, query_ys_id, net, criterion, opt)
        print('{:25} {:.4f}\n'.format("Novel incremental acc before fine-tune:",test_acc.item()))


        # Evaluate base samples before updating the network
        # TODO

        # Optimizer
        optimizer = get_optim(net, opt)

        # routine: fine-tuning for novel classes
        train_loss = 15
        epoch = 1
#         train_acc = 0
        while train_loss > opt.target_train_loss or epoch < opt.novel_epochs + 1:
#         while train_acc < opt.target_train_acc:
            freeze_backbone_weights(net, opt, epoch)
            net.train()
            support_xs = support_xs.cuda()
            support_ys_id = support_ys_id.cuda()

            # Compute output
            if opt.classifier in ["lang-linear", "description-linear"] and opt.attention is not None:
                output, alphas = net(support_xs, get_alphas=True)
                loss = criterion(output, support_ys_id) + opt.diag_reg * criterion(alphas, support_ys_id)
            else:
                output = net(support_xs)
                loss = criterion(output, support_ys_id)
              
            # Penalize the change in base classifier weights.
            if opt.lmbd_reg_transform_w is not None:
                loss += net.regloss(opt.lmbd_reg_transform_w, base_weight, base_bias)
                
            # Train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc1, acc5 = accuracy(output, support_ys_id, topk=(1,5))
                train_acc, train_loss = acc1[0], loss.item()
                print('=======Novel Epoch {}=======\n'
                      'Train\t'
                      'Loss {:10.4f}\t'
                      'Acc@1 {:10.3f}\t'
                      'Acc@5 {:10.3f}'.format(
                       epoch, loss.item(), acc1[0], acc5[0]))
            
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

    
        # Evaluate base samples with the updated network
        base_batch = next(base_valloader_it)
        vis_condition = (vis and idx == 0)
        acc_base_ = eval_base(net, 
                              base_batch, 
                              criterion, 
                              vocab_all = vocab_all if vis_condition else None, 
                              df= df if vis_condition else None)

        # Update meters.
        acc_base.update(acc_base_)
        acc_novel.update(test_acc.item())

        # Log episode results.
        log_episode(novel_labels,
                    vocab_novel,
                    epoch,
                    test_acc.item(),
                    acc_base_,
                    acc_base.avg,
                    acc_novel.avg)
        
        

    if vis:
        return df
    else:
        return mean_confidence_interval(acc_novel), mean_confidence_interval(acc_base)
