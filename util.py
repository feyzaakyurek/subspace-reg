import torch
import torch.nn as nn
import numpy as np
import pickle
# from torchnlp.word_to_vector import Vico
import ipdb
import os
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.corpus import wordnet


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, num_classes=64):
        super(BCEWithLogitsLoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, 
                                              size_average=size_average, 
                                              reduce=reduce, 
                                              reduction=reduction,
                                              pos_weight=pos_weight)
    def forward(self, input, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes)
        return self.criterion(input, target_onehot)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def create_and_save_embeds(opt, vocab):
    
    word_embeds = opt.word_embed_path
    dim = opt.word_embed_size
    embed_pth = "{0}_dim{1}.pickle".format(opt.dataset, dim)
    
    if not os.path.isdir(word_embeds):
        os.makedirs(word_embeds)
        
    words = []
    for token in vocab:
        words = words + token.split(' ')
        
    embed_pth = os.path.join(word_embeds, embed_pth)
    if os.path.exists(embed_pth):
        return
    else:
        print("Loading dictionary...")
        pretrained_embedding = Vico(name='linear',
                                    dim=dim,
                                    is_include=lambda w: w in set(words))
        
        embeds = []
        keys = words = pretrained_embedding.token_to_index.keys()
        for w in keys: 
            embeds.append(pretrained_embedding[w].numpy())
        d = dict(zip(keys, embeds))
        
        # Pickle the dictionary for later load
        print("Pickling word embeddings...")
        with open(embed_pth, 'wb') as f:
            pickle.dump(d, f)
        print("Pickled.")
        

def create_and_save_descriptions(opt, vocab):
    
    if not os.path.isdir(opt.description_embed_path):
        os.makedirs(opt.description_embed_path)
 
    embed_pth = os.path.join(opt.description_embed_path, 
                             "{0}_{1}_layer{2}_prefix_{3}.pickle".format(opt.dataset,
                                                             opt.desc_embed_model,
                                                             opt.transformer_layer,
                                                             opt.prefix_label))
    
    if os.path.exists(embed_pth):
        return
    else:
        with torch.no_grad():
            print("Creating tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(opt.desc_embed_model)
            print("Initializing {}...".format(opt.desc_embed_model))
            model = AutoModelForMaskedLM.from_pretrained(opt.desc_embed_model, output_hidden_states=True)

            # Create wordnet
            defs = [wordnet.synsets(v.replace(" ", "_"))[0].definition() for v in vocab]
    #         defs = torch.cat(defs, 0)
            embeds = []
            for i,d in enumerate(defs):
                inp = vocab[i]+" "+d if opt.prefix_label else d
                inp = tokenizer(inp, return_tensors="pt")
                outputs = model(**inp)
                hidden_states = outputs[1]
                embed = torch.mean(hidden_states[opt.transformer_layer], dim=(0,1))
                embeds.append(embed)

            d = dict(zip(vocab, embeds))
            # Pickle the dictionary for later load
            print("Pickling description embeddings from {}...".format(opt.desc_embed_model))
            with open(embed_pth, 'wb') as f:
                pickle.dump(d, f)
            print("Pickled.")
        
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x