import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class SyNet_Dataset(Dataset):
    def __init__(self, split_file, gene_exp_file, surv_labels_file, cvset, transform=None, target_transform=None):
        self.gene_exp_df = pd.read_csv(gene_exp_file, index_col=[0])  # Index should be specified. 
        self.split_df = pd.read_csv(split_file, index_col=[0])
        self.labels = pd.read_csv(surv_labels_file, index_col=[0])

        self.idx = np.where(self.split_df['cvset'] == cvset)[0]  # [0] get the row indices


        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idx)
    # This should be the length of each dataset? (train/val/test)

    def __getitem__(self, idx):
        gene_exp = np.array([self.gene_exp_df.iloc[self.idx[idx], :]])  # I kinda understand this now. 
        label = self.labels.iloc[self.idx[idx], 0]
        study = str(self.gene_exp_df.index[self.idx[idx]]).split(';')[1]
        id = str(self.gene_exp_df.index[self.idx[idx]]).split(';')[0]
        
        if self.transform:
            gene_exp = self.transform(gene_exp)  # Not sure if we need this transform argument here. 
        if self.target_transform:
            label = self.target_transform(label)
        
        return gene_exp, label, study, id

# "idx gets index from the sampler until the desired batch_size is reached. "

def CSV_reader(split_file, gene_exp_file, surv_labels_file, train_batch_size=128, val_batch_size=128, test_batch_size=32, final_test_batch_size=32, concat_train_batch_size=128, shuffle=True):
    train_data = SyNet_Dataset(split_file=split_file, gene_exp_file = gene_exp_file, surv_labels_file=surv_labels_file, cvset='train')
    val_data = SyNet_Dataset(split_file=split_file, gene_exp_file = gene_exp_file, surv_labels_file=surv_labels_file, cvset='val')
    test_data = SyNet_Dataset(split_file=split_file, gene_exp_file = gene_exp_file, surv_labels_file=surv_labels_file, cvset='test')
    final_test_data = SyNet_Dataset(split_file=split_file, gene_exp_file = gene_exp_file, surv_labels_file=surv_labels_file, cvset='final_test')
    concat_train_data = torch.utils.data.ConcatDataset([train_data, val_data, test_data])

    train_dataloader = DataLoader(train_data, train_batch_size, shuffle)
    val_dataloader = DataLoader(val_data, val_batch_size, shuffle)
    test_dataloader = DataLoader(test_data, test_batch_size, shuffle)
    final_test_dataloader = DataLoader(final_test_data, final_test_batch_size, shuffle)
    concat_train_dataloader = DataLoader(concat_train_data, concat_train_batch_size, shuffle)

    return train_dataloader, val_dataloader, test_dataloader, final_test_dataloader, concat_train_dataloader
    

def test_set_reader(split_file, gene_exp_file, surv_labels_file, test_batch_size=32, shuffle=True):
    test_data = SyNet_Dataset(split_file=split_file, gene_exp_file = gene_exp_file, surv_labels_file=surv_labels_file, cvset='test')
    test_dataloader = DataLoader(test_data, test_batch_size, shuffle)

    return test_dataloader

class no_split_Dataset(Dataset):
    def __init__(self, gene_exp_file, surv_labels_df, transform=None, target_transform=None):
        self.gene_exp_df = pd.read_csv(gene_exp_file, index_col=[0])
        self.labels = surv_labels_df
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gene_exp = np.array([self.gene_exp_df.iloc[idx, :]])  # I kinda understand this now. 
        label = self.labels.iloc[idx, 0]
        study = str(self.gene_exp_df.index[idx]).split(';')[1]
        id = str(self.gene_exp_df.index[idx]).split(';')[0]

        if self.transform:
            gene_exp = self.transform(gene_exp)  # Not sure if we need this transform argument here. 
        if self.target_transform:
            label = self.target_transform(label)

        return gene_exp, label, study, id
    
def no_split_reader(gene_exp_file, surv_labels_df, test_batch_size=32, shuffle=True):
    data = no_split_Dataset(gene_exp_file=gene_exp_file, surv_labels_df=surv_labels_df)
    dataloader = DataLoader(data, test_batch_size, shuffle)

    return dataloader