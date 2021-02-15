import torch
import torch.nn as nn
import numpy as np
import pickle
from torchnlp.word_to_vector import Vico
import ipdb
import os
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.corpus import wordnet
from PyDictionary import PyDictionary

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

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

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
        print("Found {}.".format(embed_pth))
        return
    else:
        print("Loading dictionary...")
        pretrained_embedding = Vico(name='linear',
                                    dim=dim,
                                    is_include=lambda w: w in set(words))

        embeds = []
        keys = pretrained_embedding.token_to_index.keys()
        for w in keys:
            embeds.append(pretrained_embedding[w].numpy())
        d = dict(zip(keys, embeds))

        # Pickle the dictionary for later load
        print("Pickling word embeddings...")
        with open(embed_pth, 'wb') as f:
            pickle.dump(d, f)
        print("Pickled.")

def create_and_save_synonyms(opt, vocab_train, vocab_test, vocab_val):
    # For now save only the base.
    word_embeds = opt.word_embed_path
    dim = opt.word_embed_size
    embed_pth = "{0}_dim{1}_base_synonyms.pickle".format(opt.dataset, dim)
    embed_pth = os.path.join(word_embeds, embed_pth)

    if os.path.exists(embed_pth):
        print("Found {}.".format(embed_pth))
        return
    else:
        print("Loading dictionary for synonyms...")
        dictionary=PyDictionary([v.replace(" ", "_") for v in vocab_train])

        # For every v in vocab find synonyms and save to a dict
        synonyms = {}
        all_words = []
        for v in vocab_train:
            synonyms[v] = [v] + dictionary.synonym(v.replace(" ", "_")) #wordnet.synsets()
            for syn in synonyms[v]:
                all_words.extend(syn.split(' '))

        pretrained_embedding = Vico(name='linear',
                                    dim=dim,
                                    is_include=lambda w: w in set(all_words))

        dim = 300 if opt.glove else 500
        label_syn_embeds = {}
        for v in vocab_train:
            label_embed = np.zeros(dim)
            non_zero_syns = 0
            for syn in synonyms[v]:
                words = syn.split(' ')
                embed = np.zeros(dim) #TODO
                non_zero_words_in_syn = 0
                for w in words:
                    vec = pretrained_embedding[w].numpy()
                    if not np.equal(vec,np.zeros(dim)).all():
                        embed += vec
                        non_zero_words_in_syn += 1
                embed /= max(non_zero_words_in_syn,1)
                if not np.equal(embed,np.zeros(dim)).all() :
                    label_embed += embed
                    non_zero_syns += 1
            label_embed /= max(non_zero_syns,1) #len(synonyms[v])
            label_syn_embeds[v] = label_embed

        # Pickle the dictionary for later load
        print("Pickling label embeddings averaged over all synonyms including the label itself ...")
        with open(embed_pth, 'wb') as f:
            pickle.dump(label_syn_embeds, f)
        exit(0)

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
        print("Path {} not found.".format(embed_pth))
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
