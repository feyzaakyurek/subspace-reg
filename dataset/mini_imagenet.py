import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import warnings
import re
import ipdb
torch.multiprocessing.set_sharing_strategy('file_system')
class ImageNet(Dataset):
    def __init__(self, 
                 args, 
                 split='train',
                 phase=None,
                 is_sample=False, 
                 k=4096,
                 transform=None):
        super(Dataset, self).__init__()
        self.split = split
        self.phase = phase
        self.data_aug = args.data_aug
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.unnormalize = transforms.Normalize(mean=-np.array(self.mean)/self.std, std=1/np.array(self.std))

        if transform is None:
            if self.split == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        
        if self.split == "train":
            file_pattern = 'miniImageNet_category_split_train_phase_{}.pickle'.format(phase)
        else:
            file_pattern = 'miniImageNet_category_split_{}.pickle'.format(split)
            
        self.data = {}
        with open(os.path.join(args.data_root, file_pattern), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            self.labels = data['labels']
            self.cat2label = data['catname2label']
            
        if args.continual and self.split == "val":
            file_pattern = 'miniImageNet_category_split_{}.pickle'.format("test")

            with open(os.path.join(args.data_root, file_pattern), 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                second_imgs = data['data']
                second_labels = data['labels']
                
                self.imgs = np.concatenate((self.imgs, second_imgs), axis=0)
                self.labels = np.concatenate((self.labels, second_labels), axis=0)
                self.cat2label.update(data['catname2label'])

        self.label2human = [""]*100 # Total of 100 classes in mini.

        # Labels are available by codes by default. Converting them into human readable labels.
        with open(os.path.join(args.data_root, 'class_labels.txt'), 'r') as f:
            for line in f.readlines():
                catname, humanname = line.strip().lower().split(' ')
                humanname = " ".join(humanname.split('_'))
                if catname in self.cat2label:
                    label = self.cat2label[catname]
                    self.label2human[label]= humanname

        self.global_labels = self.labels

        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, item):
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)
        if not self.is_sample:
            return img, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx

    def __len__(self):
        return len(self.labels)


class MetaImageNet(ImageNet):

    def __init__(self, 
                 args, 
                 split, 
                 phase=None,
                 train_transform=None, 
                 test_transform=None, 
                 fix_seed=True,
                 use_episodes=False,
                 disjoint_classes=False):
        
        super(MetaImageNet, self).__init__(args, split, phase)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.eval_mode = args.eval_mode
        self.n_aug_support_samples = args.n_aug_support_samples
        self.n_base_aug_support_samples = args.n_base_aug_support_samples
        self.n_base_support_samples = args.n_base_support_samples
        self.use_episodes = args.use_episodes
        self.phase = phase
        self.split = split
        self.disjoint_classes = disjoint_classes
        
        if self.split != "train":
            assert self.phase is None

        if self.use_episodes:
            
            self.episode_support_ids = []
            self.episode_query_ids = []

            with open(os.path.join(args.data_root, f"episodes_{self.n_ways}_{self.n_shots}.txt"), 'r') as f: # FIX: episodes is only for 5 shot x 5 way experiments
                is_val = True
                for line in f.readlines():
                    if line.startswith("TEST"):
                        is_val = False

                    if (split=="train" and phase == "val" and is_val) or (split=="train" and phase == "test" and not is_val):

                        if line.startswith("Base Query"):
                            arr = re.split(': ', line)[1].rstrip()
                            arr = list(map(int,filter(None,
                                  arr.lstrip('[').rstrip(']').split(" "))))
                            self.episode_query_ids.append(arr)

                    if (split == "val" and is_val) or (split == "test" and not is_val):

                        if line.startswith("Novel"):
                            arr = re.split(': ', line)[1].rstrip()
                            arr = list(map(int,filter(None,
                                  arr.lstrip('[').rstrip(']').split(","))))
                            if line.startswith("Novel Support"):
                                self.episode_support_ids.append(arr)
                            else:
                                self.episode_query_ids.append(arr)


        if train_transform is None:
            self.train_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())
        print("Self.classes", self.classes)

    def __getitem__(self, item):
        if not self.use_episodes:
            assert self.n_base_support_samples == 0 # this is not implemented for base support =! n_shot.
            if self.fix_seed:
                np.random.seed(item)
            cls_sampled = np.random.choice(self.classes, self.n_ways, False)

            if self.disjoint_classes:
                self.classes = np.setdiff1d(self.classes, cls_sampled, True)
                print("!!! Classes: ", self.classes)
            support_xs = []
            support_ys = []
            query_xs = []
            query_ys = []
            for idx, cls in enumerate(np.sort(cls_sampled)):
                imgs = np.asarray(self.data[cls]).astype('uint8')
                support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
                support_xs.append(imgs[support_xs_ids_sampled])
                lbl = idx
    #             if self.eval_mode in ["few-shot-incremental"]:
    #                 lbl = 64+idx
                if self.eval_mode in ["few-shot-incremental",
                                      "zero-shot",
                                      "zero-shot-incremental",
                                      "few-shot-language-incremental",
                                      "few-shot-incremental-fine-tune",
                                      "few-shot-incremental-language-pretrain-linear-tune",
                                      "hierarchical-incremental-few-shot"]:
                    lbl = cls
                support_ys.append([lbl] * self.n_shots) #
                query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
                query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
                query_xs.append(imgs[query_xs_ids])
                query_ys.append([lbl] * query_xs_ids.shape[0]) #
            support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(query_xs), np.array(query_ys)
            num_ways, n_queries_per_way, height, width, channel = query_xs.shape
            
            query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
            query_ys = query_ys.reshape((num_ways * n_queries_per_way, ))

            support_xs = support_xs.reshape((-1, height, width, channel))
            if self.n_aug_support_samples > 1:
                support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
                support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
            support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
            query_xs = query_xs.reshape((-1, height, width, channel))
            query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

            support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
            query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))
            
        else:
            
                
            if self.split == "train" and self.phase == "train":
                    assert self.n_base_support_samples > 0
                    # These samples will be stored in memory for every episode.
                    support_xs = []
                    support_ys = []
                    if self.fix_seed:
                        np.random.seed(item)
                    cls_sampled = np.random.choice(self.classes, len(self.classes), False)
                    
                    for idx, cls in enumerate(np.sort(cls_sampled)):
                        imgs = np.asarray(self.data[cls]).astype('uint8')
                        support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_base_support_samples, False)
                        support_xs.append(imgs[support_xs_ids_sampled])
                        support_ys.append([cls] * self.n_base_support_samples)    
                    support_xs, support_ys = np.array(support_xs), np.array(support_ys)
                    num_ways, n_queries_per_way, height, width, channel = support_xs.shape
                    support_xs = support_xs.reshape((-1, height, width, channel))
                    if self.n_base_aug_support_samples > 1:
                        support_xs = np.tile(support_xs, (self.n_base_aug_support_samples, 1, 1, 1))
                        support_ys = np.tile(support_ys.reshape((-1, )), (self.n_base_aug_support_samples))
                    support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
                    support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))

                    # Dummy query.
                    query_xs = support_xs
                    query_ys = support_ys
                    
            else:
#                 assert self.n_base_support_samples == 0
                
                # Actual query.
                query_xs_ids = self.episode_query_ids[item]
                query_xs = np.array(self.imgs[query_xs_ids])
                query_ys = np.array([self.labels[i] for i in query_xs_ids])
                _, height, width, channel = query_xs.shape
                num_ways, n_queries_per_way = (self.n_ways, len(query_xs_ids) // self.n_ways)

                query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
                query_ys = query_ys.reshape((num_ways * n_queries_per_way, ))
                query_xs = query_xs.reshape((-1, height, width, channel))
                query_xs = np.split(query_xs, query_xs.shape[0], axis=0)
                query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))
            
            
                if self.split == "train" and self.phase in ["val", "test"] :
                    # Dummy support if phase is val or test.
                    support_xs = query_xs.squeeze(0)
                    support_ys = query_ys
                
                else:
                    # Actual support.
                    support_xs_ids_sampled = self.episode_support_ids[item]
                    support_xs = np.array(self.imgs[support_xs_ids_sampled])

                    support_ys = np.array([self.labels[i] for i in support_xs_ids_sampled])
                    assert len(np.unique(support_ys)) == self.n_ways

                    support_xs = support_xs.reshape((-1, height, width, channel))
                    if self.n_aug_support_samples > 1:
                        support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
                        support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
                    support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
                    support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
                

        return support_xs.float(), support_ys, query_xs.float(), query_ys

    def __len__(self):
        if (self.split == "train" and self.phase == "train"):
            if self.disjoint_classes:
                return 8
            return self.n_test_runs
        elif self.use_episodes:
            return len(self.episode_query_ids)
        else:
            return self.n_test_runs
            


if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    args.data_root = 'data'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    imagenet = ImageNet(args, 'val')
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)

    metaimagenet = MetaImageNet(args)
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)
