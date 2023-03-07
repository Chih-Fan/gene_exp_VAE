from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class SyNet_Dataset(Dataset):
    def __init__(self, split_file, gene_exp_file, cvset, transform=None):
        self.gene_exp_df = pd.read_csv(gene_exp_file, index_col=[0])  # Not sure if index should be specified or not. 
        self.split_df = pd.read_csv(split_file)

        self.idx = np.where(self.split_df['cvset'] == cvset)[0]

        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        gene_exp = np.array([self.gene_exp_df.iloc[self.idx[idx], 1:]])  # I don't understand this. 
        if self.transform:
            gene_exp = self.transform(gene_exp)  # Not sure if we need this transform argument here. 
        
        return gene_exp


def CSV_reader(split_file, gene_exp_file, batch_size=128, shuffle=True):
    train_data = SyNet_Dataset(split_file=split_file, gene_exp_file = gene_exp_file, cvset='train')
    val_data = SyNet_Dataset(split_file=split_file, gene_exp_file = gene_exp_file, cvset='val')
    test_data = SyNet_Dataset(split_file=split_file, gene_exp_file = gene_exp_file, cvset='test')

    train_dataloader = DataLoader(train_data, batch_size, shuffle)
    val_dataloader = DataLoader(val_data, batch_size, shuffle)
    test_dataloader = DataLoader(test_data, batch_size, shuffle)

    return train_dataloader, val_dataloader, test_dataloader


