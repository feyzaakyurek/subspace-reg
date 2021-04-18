import pickle
import os

DATA_PATH = "../data/miniImageNet/"
split = "test"
phase = None
dataset = "mini" # or tiered



def get_mini_human_labels(data_path, labels_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        imgs = data['data']
        labels = data['labels']
        cat2label = data['catname2label']

    label2human = {}

    with open(labels_path, 'r') as f:
        for line in f.readlines():
            catname, humanname = line.strip().lower().split(' ')
            humanname = " ".join(humanname.split('_'))
            if catname in cat2label:
                label = cat2label[catname]
                label2human[label]= humanname
            
    return label2human
            
if __name__ == "__main__":
    
    if dataset == "mini":
        if phase is None:
            file_pattern = f"miniImageNet_category_split_{split}.pickle"
            name = "mini_{}_human_labels.pickle".format(split)
        else:
            file_pattern = f"miniImageNet_category_split_{split}_phase_{phase}.pickle"
            name = "mini_{}_{}_human_labels.pickle".format(split, phase)
        data_path = os.path.join(DATA_PATH, file_pattern)
        labels_path = os.path.join(DATA_PATH, 'class_labels.txt')
        d = get_mini_human_labels(data_path, labels_path)
        
        save_path = os.path.join(DATA_PATH, name)
        with open(save_path, 'wb') as f1:
            pickle.dump(d, f1)
            
    else:
        raise NotImplementedError()
