from run_VAE_SyNet_ropLR_tanh_scalef_dyn_loss_scaler_absx import DEVICE, Encoder, Decoder, VAE
from make_dataloaders_labels_SyNet import CSV_reader, test_set_reader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import umap
import torch
import argparse
import time


parser_ = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_.add_argument('--ged', type=str, help='gene expression data path')
parser_.add_argument('--spld', type=str, help='split file path')
parser_.add_argument('--conf', type=str, default='unnamed_config', help='configuration name')
parser_.add_argument('--trbs', type=int, default=128, help='training batch size')
parser_.add_argument('--valbs', type=int, default=128, help='validation batch size')
parser_.add_argument('--tebs', type=int, default=128, help='testing batch size')
parser_.add_argument('--idx', type=int, default=0, help='index of batch to be plot')
parser_.add_argument('--sdp', type=str, help='state dict path')
parser_.add_argument('--cvbs', type=str, default='unnamed_batch_split', help='cross validation batch split')
parser_.add_argument('--first', type=int, help='dim of first linear layer')
parser_.add_argument('--second', type=int, help='dim of second linear layer')
parser_.add_argument('--third', type=int, help='dim of third linear layer')
parser_.add_argument('--scalef', type=float, default=2.5, help='scaling factor in encoder output layer')
parser_.add_argument('--klscale', type=float, default=1.0, help='scaling factor for KL divergence')
parser_.add_argument('--pseudoc', type=float, default=1.0, help='pseudocount for the loss scaling')
parser_.add_argument('--absxscale', type=float, default=1.0, help='scaling factor for the abs x in loss scaling')
parser_.add_argument('--bc', type=str, help='batch-correction label')
args_ = parser_.parse_args()
surv_labels_file = "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/Survival_labels.csv"

def scatter_comparison(x_samp, x_r_samp, pear_corr):
    print(x_samp.shape, x_r_samp.shape)
    fig, ax = plt.subplots()
    sns.regplot(x=x_samp, y=x_r_samp, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'}, ax=ax)
    ax.text(1, 3, 'Pearson corr = '+ str(pear_corr))

    fig.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/images/scatter_com_' + str(args_.bc) + '_' + str(args_.cvbs) + '_' + str(args_.conf) + '_batch' + str(args_.idx) + '_sample0.png')

def hexbin(x_samp, x_r_samp, pear_corr):
    x_samp = x_samp.cpu().detach().numpy()
    x_r_samp = x_r_samp.cpu().detach().numpy()
    fig, ax = plt.subplots()
    ax.hexbin(x_samp, x_r_samp, gridsize=60)
    ax.text(1, 3, 'Pearson corr = '+ str(pear_corr))
    ax.set_xlabel('Original values')
    ax.set_ylabel('Reconstructed values')
    fig.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/images/average_hexbin_' + str(args_.bc) + '_' + str(args_.cvbs) + '.png')

def PCA_latent(mu, mu_std, y): 
    mu = mu.cpu().detach().numpy()
    mu_std = mu_std.cpu().detach().numpy()
    print('mu_std shape for umap: ' + str(mu_std.shape))
    scaling = StandardScaler()  # Scale data before applying PCA. https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
    mu_std_scaled = scaling.fit_transform(mu_std)  # Use fit and transform method
    mu_scaled = scaling.fit_transform(mu)

    pca_mu_std = PCA(n_components=2)
    pca_mu = PCA(n_components=2)
    mu_std_scaled_pca = pca_mu_std.fit_transform(mu_std_scaled)
    mu_scaled_pca = pca_mu.fit_transform(mu_scaled)

    print("mu_std explained variance ratio (first two components): %s" % str(pca_mu_std.explained_variance_ratio_))
    print("mu explained variance ratio (first two components): %s" % str(pca_mu.explained_variance_ratio_))


    label_names = ["< 5 years", ">= 5 years"]
    colors = ["lightseagreen", "darkorange"]
    lw = 2

    # plt.figure(figsize=(8, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for color, i, label_name in zip(colors, [0, 1], label_names):
        ax1.scatter(mu_scaled_pca[y == i, 0], mu_scaled_pca[y == i, 1], color=color, alpha=0.6, lw=lw, label=label_name)
    # ax1.legend(loc="best", shadow=False, scatterpoints=1)
    ax1.set_title("PCA of latent space for mean")

    for color, i, label_name in zip(colors, [0, 1], label_names):
        ax2.scatter(mu_std_scaled_pca[y == i, 0], mu_std_scaled_pca[y == i, 1], color=color, alpha=0.6, lw=lw, label=label_name)
    ax2.legend(loc="best", shadow=False, scatterpoints=1)
    ax2.set_title("PCA of latent space for mean+std")

    plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/images/PCA_latent_' + str(args_.bc) + '_' +  str(args_.cvbs) + '_' + str(args_.conf) + '.png')

def umap_latent(mu, mu_std, y):
    mu = mu.cpu().detach().numpy()
    mu_std = mu_std.cpu().detach().numpy()

    print('mu_std shape for umap: ' + str(mu_std.shape))
    scaling = StandardScaler()  # Scale data before applying UMAP. https://umap-learn.readthedocs.io/en/latest/basic_usage.html, https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
    mu_std_scaled = scaling.fit_transform(mu_std)  # Use fit and transform method
    mu_scaled = scaling.fit_transform(mu)
    umap_reducer_mu_std = umap.UMAP()
    umap_reducer_mu = umap.UMAP()

    mu_umap_embedding = pd.DataFrame(umap_reducer_mu.fit_transform(mu_scaled), columns = ['UMAP1', 'UMAP2'])
    mu_std_umap_embedding = pd.DataFrame(umap_reducer_mu_std.fit_transform(mu_std_scaled), columns = ['UMAP1', 'UMAP2'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns_plot1 = sns.scatterplot(x='UMAP1', y='UMAP2', data=mu_umap_embedding,
                hue=y,  # Not sure if this is correct. 
                linewidth=0, s=5, ax=ax1)
    
    sns_plot2 = sns.scatterplot(x='UMAP1', y='UMAP2', data=mu_std_umap_embedding,
                hue=y,  # Not sure if this is correct. 
                linewidth=0, s=5, ax=ax2)
    
    # sns_plot1.legend(loc='center left', bbox_to_anchor=(1, .5))
    # sns_plot2.legend(loc='center left', bbox_to_anchor=(1, .5))

    plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/images/UMAP_latent_' + str(args_.bc) + '_' +  str(args_.cvbs) + '_' + str(args_.conf) + '.png', bbox_inches='tight', dpi=500)


def main():
    print('Configuration: ' + str(args_.conf) + ', Batch split: ' + str(args_.cvbs) + ', Batch_correction: ' + str(args_.bc))
    start = time.time()
    # train_dataloader, val_dataloader, test_dataloader = CSV_reader(split_file=args_.spld, gene_exp_file=args_.ged,  train_batch_size=args_.trbs, val_batch_size=args_.valbs, test_batch_size=32, surv_labels_file=surv_labels_file,  shuffle=False)
    test_dataloader = test_set_reader(split_file=args_.spld, gene_exp_file=args_.ged, surv_labels_file=surv_labels_file, test_batch_size=args_.tebs, shuffle=False) 
    test_inp_size = [x.shape[2] for batch_idx, (x, y) in enumerate(test_dataloader)][0]
    # print("This is the test_inp_size: " + str(test_inp_size))
    
    test_model = VAE(Encoder(input_size=test_inp_size), Decoder(input_size=test_inp_size))
    if DEVICE == torch.device("cuda"):
        checkpoint = torch.load(args_.sdp)
    elif DEVICE == torch.device("cpu"):
        checkpoint = torch.load(args_.sdp, map_location=DEVICE)
    else:
        print("Device is " , DEVICE)
        raise ValueError("Device is not cuda nor cpu.")
    
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.to(DEVICE)
    
    test_model.eval()
    with torch.no_grad():
        overall_test_loss = 0
        overall_test_rec_loss = 0
        sum_pearson = 0
        for batch_idx, (x, y) in enumerate(test_dataloader):
            # print("The current batch: " + str(batch_idx+1))
            # print("Number of samples in the current batch: " + str(len(x)))
            # print("This is the y size: " + str(y.shape))
            x = x.view(len(x), test_inp_size)  
            x = x.to(torch.float32)
            x = x.to(DEVICE)
            z, mu, log_var, mu_std, x_r= test_model(x)
            if batch_idx == 0:
                mu_cat = mu
                mu_std_cat = mu_std
                y_cat = y
            else:
                mu_cat = torch.cat((mu_cat, mu),0 )
                mu_std_cat = torch.cat((mu_std_cat, mu_std), 0)
                y_cat = torch.cat((y_cat, y), 0)
            test_rec_loss, test_loss = test_model.loss_function(x_r=x_r, x=x, mean=mu, log_var=log_var)
            overall_test_loss += test_loss.item()
            overall_test_rec_loss += test_rec_loss.item()

            for i, (x_sample, x_r_sample) in enumerate(zip(x, x_r)):
                sample_pearson = stats.pearsonr(x_sample.detach().cpu().numpy(), x_r_sample.detach().cpu().numpy())[0]
                sum_pearson += sample_pearson
                if batch_idx == args_.idx and i == 0:
                    # scatter_comparison(x_samp=x_sample, x_r_samp=x_r_sample, pear_corr=sample_pearson)
                    hexbin(x_samp=x_sample, x_r_samp=x_r_sample, pear_corr=sample_pearson)

        pear_average = sum_pearson/len(test_dataloader.dataset)

        print("length of the whole set: " + str(len(test_dataloader.dataset)))

        average_test_loss = overall_test_loss / len(test_dataloader.dataset)
        average_test_rec_loss = overall_test_rec_loss / len(test_dataloader.dataset)
    
    
    end = time.time()
    torch.save(mu_std_cat, '../data/new0418/embedding_tensors/embedding_' +  str(args_.cvbs) + '_' + str(args_.conf) +'.pt')
    PCA_latent(mu=mu_cat, mu_std=mu_std_cat, y=y_cat)
    umap_latent(mu=mu_cat, mu_std=mu_std_cat, y=y_cat)
    

    print('Configuration: ' + str(args_.conf) + ', Batch split: ' + str(args_.cvbs) + ", Average test loss per sample: " + str(average_test_loss) + ", Average test rec_loss per sample: " + str(average_test_rec_loss) + ", Average Pearson corr: " + str(pear_average) + " , Runtime: " + str(end - start))

    return mu_std_cat, average_test_loss, average_test_rec_loss, pear_average 

if __name__ == "__main__":
    main()