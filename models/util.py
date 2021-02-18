from __future__ import print_function
import torch
import pickle
import ipdb
import numpy as np

def create_model(name, n_cls, opt, vocab=None, dataset='miniImageNet'):
    from . import model_dict
    """create model by name"""
    if dataset == 'miniImageNet' or dataset == 'tieredImageNet':
        if name.endswith('v2') or name.endswith('v3'):
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('resnet50'):
            print('use imagenet-style resnet50')
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, 
                                     dropblock_size=5, num_classes=n_cls, 
                                     vocab=vocab, opt=opt) #TODO
        elif name.startswith('wrn'):
            model = model_dict[name](num_classes=n_cls)
        elif name.startswith('convnet'):
            model = model_dict[name](num_classes=n_cls)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    elif dataset == 'CIFAR-FS' or dataset == 'FC100':
        if name.startswith('resnet') or name.startswith('seresnet'):
            model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_cls)
        elif name.startswith('convnet'):
            model = model_dict[name](num_classes=n_cls)
        else:
            raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    return model


def get_teacher_name(model_path):
    """parse to get teacher model name"""
    segments = model_path.split('/')[-2].split('_')
    if ':' in segments[0]:
        return segments[0].split(':')[-1]
    else:
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]
        

def get_embeds(embed_pth, vocab, dim=500):
    '''
    Takes in path to the embeds and vocab (list).
    Returns a list of embeds.
    '''
    with open(embed_pth, "rb") as openfile:
        embeds_ = pickle.load(openfile)
    embeds = [0] * len(vocab)
    for (i,token) in enumerate(vocab):
        words = token.split(' ')
        for w in words:
            try:
                embeds[i] += embeds_[w]
            except KeyError:
                embeds[i] = np.zeros(dim)
        embeds[i] /= len(words)
        
    return torch.stack([torch.from_numpy(e) for e in embeds], 0)