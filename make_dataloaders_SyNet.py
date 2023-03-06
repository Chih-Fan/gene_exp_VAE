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
        gene_exp = np.array([self.gene_exp_df.iloc[self.idx[idx], 1:]])
        if self.transform:
            gene_exp = self.transform(gene_exp)  # Not sure if we need this transform argument here. 
        
        return gene_exp


def CSV_reader(train_data_path, test_data_path, batch_size=128, shuffle=True):
    train_data = SyNet_Dataset(gene_exp_file = train_data_path)
    test_data = SyNet_Dataset(gene_exp_file = test_data_path)

    train_dataloader = DataLoader(train_data, batch_size, shuffle)
    test_dataloader = DataLoader(test_data, batch_size, shuffle)

    return train_dataloader, test_dataloader

# df_train_data = pd.read_csv("/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/df_train_data_gene_exp.csv")
# df_test_data = pd.read_csv("/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/df_test_data_gene_exp.csv")

# train_data = torch.from_numpy(df_train_data.to_numpy())
# test_data = torch.from_numpy(df_test_data.to_numpy())

