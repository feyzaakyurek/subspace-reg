import torch
import torch.nn as nn
import pickle
from models.util import get_embeds
import pdb
import os
from models.resnet_language import LinearMap

WORD_EMBED_PATH = "word_embeds/miniImageNet_dim500.pickle"
BASE_LABELS = "data/miniImageNet/mini_train_train_human_labels.pickle"
MODEL_HOME = "dumped/backbones/linear/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_trial_pretrain_classifier_linear_8075566"
MODEL_PATH = os.path.join(MODEL_HOME, "resnet12_last.pth")
SAVE_PATH =os.path.join(MODEL_HOME, "resnet12_last_with_mapping.pth")

DEVICE = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1.0
WD = 5e-4
EPOCHS = 1000



def load_pickle(pth):
    with open(pth, 'rb') as f:
        d = pickle.load(f)
    return d

def get_base_labels(pth):
    d = load_pickle(pth)
    ls = [""]*len(d)
    for k,v in d.items():
        ls[k] = v
    return ls

def get_classifier_weights(pth, device):
    ckpt = torch.load(pth, map_location=device)
    return ckpt, ckpt['model']['classifier.weight']

def save_model(ckpt, model, nickname, save_path):
    ckpt[nickname] = model.state_dict()
    torch.save(ckpt, save_path)

def main():
    base_labels = get_base_labels(BASE_LABELS)
    label_embeds = get_embeds(WORD_EMBED_PATH, vocab=base_labels).float() #Tensor
    ckpt, base_embeds = get_classifier_weights(MODEL_PATH, DEVICE) #Tensor

    model = LinearMap(label_embeds.size(1),
                      base_embeds.size(1)) # e.g. for glove 300x640
    
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=LR,
                          weight_decay=WD)
    criterion = nn.MSELoss()
    
    for ep in range(EPOCHS):
        
        output = model(label_embeds)
        loss = criterion(output, base_embeds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (ep+1) % 10 == 0:
            print(f"Epoch [{ep+1}/{EPOCHS}] Loss: {loss}")
    
    save_model(ckpt, model, "mapping_linear_label2image", SAVE_PATH)
        
if __name__ == "__main__":
    main()