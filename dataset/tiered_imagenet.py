import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import ipdb

num_classes_offset = {"traina":0,
                      "trainb":200,
                      "val":351,
                      "test":448}

def get_offset(partition, super_partition, trainb=False):
    if super_partition == "train":
        return num_classes_offset["trainb"] if trainb else num_classes_offset["traina"]
    
    
    if partition == "val":
        return num_classes_offset["val"]
    elif partition == "test":
        return num_classes_offset["test"]
    else:
        raise ValueError()
        
        

class TieredImageNet(Dataset):
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None, super_partition='train'):
        super(Dataset, self).__init__()
        if partition == 'train':
            assert super_partition == 'train'
            
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.super_partition = super_partition
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
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

        if self.super_partition == 'train':
            self.image_file_pattern = 'train_a_train_a_phase_%s_images.npz'
            self.label_file_pattern = 'train_a_train_a_phase_%s_labels.pkl'
        else:
            self.image_file_pattern = '%s_images.npz'
            self.label_file_pattern = '%s_labels.pkl'

        self.data = {}

        # modified code to load tieredImageNet 
        image_file = os.path.join(self.data_root, self.image_file_pattern % partition)
        self.imgs = np.load(image_file)['images']
        label_file = os.path.join(self.data_root, self.label_file_pattern % partition)
        label_file_content = self._load_labels(label_file)
        self.labels = label_file_content['label_specific']
        
        # Determine label2human
        self.label2human = [""]*608
        self.offset = get_offset(partition, super_partition)
        unique_labels = self.offset + np.sort(np.unique(self.labels))
        label2human = dict(zip(unique_labels, 
                               label_file_content['label_specific_str']))

        for i,human in label2human.items():
            self.label2human[i] = human

        # if partition is train, we'll pool two files together to use all 351 classes in backbone training.
        if self.pretrain and partition == "train" and args.augment_pretrain_wtrainb is not None:
            second_image_file = os.path.join(self.data_root, 'train_b_images.npz')
            second_label_file = os.path.join(self.data_root, 'train_b_labels.pkl')
            self.offset = get_offset('train', 'train', trainb=True)
            second_imgs = np.load(second_image_file)['images']
            second_labels = self._load_labels(second_label_file)['label_specific']
            second_labels = self.offset + second_labels
            self.imgs = np.concatenate((self.imgs, second_imgs), axis=0)
            self.labels = np.concatenate((self.labels, second_labels), axis=0)

            unique_labels = self.offset + np.sort(np.unique(self.labels))
            label2human = dict(zip(unique_labels, 
                                   label_file_content['label_specific_str']))

            for i,human in label2human:
                self.label2human[i] = human
            
        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            assert False
            # num_classes is not checked after offset.
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

    @staticmethod
    def _load_labels(file):
        try:
            with open(file, 'rb') as fo:
                data = pickle.load(fo)
            return data
        except:
            with open(file, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data


class MetaTieredImageNet(TieredImageNet):

    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True, pretrain=False, super_partition='non_train'):
        super(MetaTieredImageNet, self).__init__(args, partition, pretrain, super_partition=super_partition)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.eval_mode = args.eval_mode
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
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
            given_label = self.offset + self.labels[idx]
            if given_label not in self.data:
                self.data[given_label] = []
            self.data[given_label].append(self.imgs[idx])
        self.classes = list(self.data.keys())
        
    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            
            lbl = idx
            if self.eval_mode in ["few-shot-incremental",
                                  "zero-shot",
                                  "zero-shot-incremental",
                                  "few-shot-language-incremental",
                                  "few-shot-incremental-fine-tune",
                                  "few-shot-incremental-language-pretrain-linear-tune",
                                  "hierarchical-incremental-few-shot"]:
                lbl = cls
            
            support_ys.append([lbl] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([lbl] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs


if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    # args.data_root = 'data'
    args.data_root = '/home/yonglong/Data/tiered-imagenet-kwon'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    imagenet = TieredImageNet(args, 'train')
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)

    metaimagenet = MetaTieredImageNet(args)
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)
