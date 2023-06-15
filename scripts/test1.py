import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from make_dataloaders_labels_SyNet import CSV_reader


# data = torch.utils.data.DataLoader(MNIST('./data',
#                transform=transforms.ToTensor(),
#                download=True),
#         batch_size=128,
#         shuffle=True)

split_file = "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/SyNet_batching_1.csv"
gene_exp_file = "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/SyNet_Scaled_Batch-un-corrected_Filtered_Data_Only.csv"
surv_labels_file = "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/Survival_labels.csv"

train_dataloader, val_dataloader, test_dataloader = CSV_reader(split_file=split_file, gene_exp_file=gene_exp_file, surv_labels_file=surv_labels_file, train_batch_size=128, val_batch_size=128, test_batch_size=32, shuffle=False)
for batch, (x, y) in enumerate(test_dataloader):
    print(x.shape)
    print(y.shape)

    break