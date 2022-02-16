from torch.utils.data import Dataset
import torch

class Memory(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.data = None
        self.labels = None
        
#     def additem(data, label):
#         self.data.append(data)
#         self.labels.append(label)
        
    def additems(self, data, label):
        if self.data is None:
            self.data = data
            self.labels = label
        else:
            self.data = torch.cat((self.data, data), dim=0)
            self.labels = torch.cat((self.labels, label), dim=0)
        
    def __getitem__(self, item):
        return (self.data[item], self.labels[item])
    
    def __len__(self):
        if self.labels is None:
            return 0
        return len(self.labels)
    