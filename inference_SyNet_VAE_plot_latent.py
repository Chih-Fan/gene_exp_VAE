from run_VAE_SyNet_ropLR import DEVICE, Encoder, Decoder, VAE
from make_dataloaders_labels_SyNet import CSV_reader, test_set_reader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
parser_.add_argument('--idx', type=int, default=0, help='index of sample to be plot')
parser_.add_argument('--sdp', type=str, help='state dict path')
parser_.add_argument('--cvbs', type=str, default='unnamed_batch_split', help='cross validation batch split')
args_ = parser_.parse_args()
surv_labels_file = "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/Survival_labels.csv"

def scatter_comparison(x, x_r, sample_idx, batch_size):
    if DEVICE == 'cuda':
        x = x.cpu().detach().numpy
        print(DEVICE)
    else:
        print(DEVICE)
    x = x.view(batch_size, 11748)
    x_r = x_r.view(batch_size, 11748)
    print("This is x.shape in scatter com: " + str(x.shape))

    print("This is the sample shape in scatter com: " + str(x[sample_idx].shape) + str(x_r[sample_idx].shape))
    fig, ax = plt.subplots()
    ax.scatter(x[sample_idx], x_r[sample_idx])
    fig.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/images/scatter_com_batch-un-corrected_' +  str(args_.cvbs) + '_' + str(args_.conf) + '_batch' + str(sample_idx) + '.png')

def PCA_latent(mu, y): 
    if DEVICE == 'cuda':
        mu = mu.cpu().detach().numpy()
        print(DEVICE)
    else:
        print(DEVICE)

    scaling = StandardScaler()  # Scale data before applying PCA. https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
    mu_scaled = scaling.fit_transform(mu)  # Use fit and transform method

    pca = PCA(n_components=2)
    mu_pca = pca.fit_transform(mu)
    mu_scaled_pca = pca.fit_transform(mu_scaled)
    # print("mu dimension: " + str(mu_pca.shape))

    print("explained variance ratio (first two components): %s" % str(pca.explained_variance_ratio_))

    label_names = ["< 5 years", ">= 5 years"]
    colors = ["lightseagreen", "darkorange"]
    lw = 2

    # plt.figure(figsize=(8, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    for color, i, label_name in zip(colors, [0, 1], label_names):
        ax1.scatter(mu_pca[y == i, 0], mu_pca[y == i, 1], color=color, alpha=0.8, lw=lw, label=label_name)
    # ax1.legend(loc="best", shadow=False, scatterpoints=1)
    ax1.set_title("PCA of latent space (no scaling before dim-red)")

    for color, i, label_name in zip(colors, [0, 1], label_names):
        ax2.scatter(mu_scaled_pca[y == i, 0], mu_scaled_pca[y == i, 1], color=color, alpha=0.8, lw=lw, label=label_name)
    ax2.legend(loc="best", shadow=False, scatterpoints=1)
    ax2.set_title("PCA of latent space (standard-scaling before dim-red)")

    # scatter = ax.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
    
    # ax.legend(*scatter.legend_elements(), title='Survival > 5 years')
    plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/images/PCA_latent_batch-un-corrected_' +  str(args_.cvbs) + '_' + str(args_.conf) + '.png')

def umap_latent(mu, y):
    scaling = StandardScaler()  # Scale data before applying UMAP. https://umap-learn.readthedocs.io/en/latest/basic_usage.html, https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
    mu_scaled = scaling.fit_transform(mu)  # Use fit and transform method
    umap_reducer = umap.UMAP()
    mu_umap_embedding = pd.DataFrame(umap_reducer.fit_transform(mu), columns = ['UMAP1', 'UMAP2'])

    mu_scaled_umap_embedding = pd.DataFrame(umap_reducer.fit_transform(mu_scaled), columns = ['UMAP1', 'UMAP2'])

    print("mu_umap shape: " + str(mu_umap_embedding.shape) + ", mu_scaled_umap shape: " + str(mu_scaled_umap_embedding.shape))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    sns_plot1 = sns.scatterplot(x='UMAP1', y='UMAP2', data=mu_umap_embedding,
                hue=y,  # Not sure if this is correct. 
                linewidth=0, s=1, ax=ax1)
    
    sns_plot2 = sns.scatterplot(x='UMAP1', y='UMAP2', data=mu_scaled_umap_embedding,
                hue=y,  # Not sure if this is correct. 
                linewidth=0, s=1, ax=ax2)
    
    # sns_plot1.legend(loc='center left', bbox_to_anchor=(1, .5))
    sns_plot2.legend(loc='center left', bbox_to_anchor=(1, .5))

    plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/images/UMAP_latent_batch-un-corrected_' +  str(args_.cvbs) + '_' + str(args_.conf) + '.png', bbox_inches='tight', dpi=500)
    # ax1.scatter(mu_umap_embedding[:, 0], mu_umap_embedding[:, 1], c=sns.color_palette()[x] for x in )



def main():
    start = time.time()
    train_dataloader, val_dataloader, test_dataloader = CSV_reader(split_file=args_.spld, gene_exp_file=args_.ged,  train_batch_size=args_.trbs, val_batch_size=args_.valbs, test_batch_size=32, surv_labels_file=surv_labels_file,  shuffle=False)
    # test_dataloader = test_set_reader(split_file=args_.spld, gene_exp_file=args_.ged, surv_labels_file=surv_labels_file, test_batch_size=154, shuffle=False) 
    test_inp_size = [x.shape[2] for batch_idx, (x, y) in enumerate(train_dataloader)][0]
    print("This is the test_inp_size: " + str(test_inp_size))
    

    test_model = VAE(Encoder(input_size=test_inp_size), Decoder(input_size=test_inp_size))
    if DEVICE == 'cuda':
        test_model.load_state_dict(torch.load(args_.sdp))
    elif DEVICE == 'cpu':
        test_model.load_state_dict(torch.load(args_.sdp, map_location=DEVICE))

    test_model.to(DEVICE)
    
    test_model.eval()
    with torch.no_grad():
        # test_loss_list = []
        overall_test_loss = 0
        idx = 0
        for batch_idx, (x, y) in enumerate(train_dataloader):
            print("The current batch: " + str(batch_idx+1))
            print("Number of samples in the current batch: " + str(len(x)))
            print("This is the y size: " + str(y.shape))
            x = x.view(len(x), test_inp_size)  
            x = x.to(torch.float32)
            x = x.to(DEVICE)
            z, mu, log_var, x_r= test_model(x)
            if batch_idx == 0:
                mu_cat = mu
                y_cat = y
            else:
                mu_cat = torch.cat((mu_cat, mu), 0)
                y_cat = torch.cat((y_cat, y), 0)
            test_loss = test_model.loss_function(pred=x_r, target=x, mean=mu, log_var=log_var)
            # test_loss_list.append(test_loss)
            print("test_loss for batch " + str(batch_idx) + " is: " + str(test_loss))
            overall_test_loss += test_loss.item()
            if batch_idx == idx:
                scatter_comparison(x=x, x_r=x_r, sample_idx=args_.idx, batch_size=128)
            else:
                pass

        # print(test_loss_list)
        print("length of the whole set: " + str(len(train_dataloader.dataset)))

        average_test_loss = overall_test_loss / len(train_dataloader.dataset)
    
    
    end = time.time()
    print("Shape of mu_cat: " + str(mu_cat.shape))
    PCA_latent(mu=mu_cat, y=y_cat)
    umap_latent(mu=mu_cat, y=y_cat)

    print('Configuration: ' + str(args_.conf) + ', Batch split: ' + str(args_.cvbs) + ", Overall test loss: " + str(overall_test_loss) + ", Average test loss per sample: " + str(average_test_loss) + ", Runtime: " + str(end - start))

    return average_test_loss

if __name__ == "__main__":
    main()