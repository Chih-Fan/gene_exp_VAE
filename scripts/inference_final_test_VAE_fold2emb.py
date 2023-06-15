from train_VAE_SyNet import DEVICE, Encoder, Decoder, VAE
from make_dataloaders_labels_SyNet import CSV_reader, test_set_reader, no_split_reader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
surv_labels_file = "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/Survival_labels_filtered.csv"

def scatter_comparison(x_samp, x_r_samp, pear_corr):
    print(x_samp.shape, x_r_samp.shape)
    fig, ax = plt.subplots()
    sns.regplot(x=x_samp, y=x_r_samp, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'}, ax=ax)
    ax.text(1, 3, 'Pearson corr = '+ str(pear_corr))

    fig.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/train_all/images/scatter_com_' + str(args_.bc) + '_' + str(args_.cvbs) + '_' + str(args_.conf) + '_batch' + str(args_.idx) + '_sample0.png')

def hexbin(x_samp, x_r_samp, pear_corr):
    x_samp = x_samp.cpu().detach().numpy()
    x_r_samp = x_r_samp.cpu().detach().numpy()
    fig, ax = plt.subplots()
    ax.hexbin(x_samp, x_r_samp, cmap='Blues', gridsize=60)
    ax.text(0.95, 0.01, 'Pearson corr = '+ str(pear_corr),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=10)
    ax.set_xlabel('Original values')
    ax.set_ylabel('Reconstructed values')
    # fig.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/train_all/images/average_hexbin_fold2ontrain' + str(args_.bc) + '_' + str(args_.cvbs) + '.png')

def PCA_latent(mu, mu_std, y, s, save): 
    mu = mu.cpu().detach().numpy()
    mu_std = mu_std.cpu().detach().numpy()
    y = y.detach().cpu().numpy()
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
    studies = ['Desmedt-June07', 'Hatzis-Pusztai', 'METABRIC', 'Pawitan', 'Schmidt', 'Symmans ', 'TCGA', 'WangY', 'WangY-ErasmusMC', 'Zhang ']
    colors_studies = cm.rainbow(np.linspace(0, 1, len(studies)))
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

    # plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/images/PCA_latent_surv_' + str(save) + '_' + str(args_.bc) + '_' +  str(args_.cvbs) + '_' + str(args_.conf) + '.png')

    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    for color, study in zip(colors_studies, studies):
        ax3.scatter(mu_scaled_pca[s == study, 0], mu_scaled_pca[s == study, 1], color=color, alpha=0.6, lw=lw, label=study)
    # ax1.legend(loc="best", shadow=False, scatterpoints=1)
    ax1.set_title("PCA of latent space for mean")

    for color, study in zip(colors_studies, studies):
        ax4.scatter(mu_std_scaled_pca[s == study, 0], mu_std_scaled_pca[s == study, 1], color=color, alpha=0.6, lw=lw, label=study)
    ax4.legend(loc="best", shadow=False, scatterpoints=1)
    ax4.set_title("PCA of latent space for mean+std")

    # plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/images/PCA_latent_study_' + str(save) + '_' + str(args_.bc) + '_' +  str(args_.cvbs) + '_' + str(args_.conf) + '.png')

def umap_latent(mu, mu_std, y, s, save):
    mu = mu.detach().cpu().numpy()
    mu_std = mu_std.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    print('mu_std shape for umap: ' + str(mu_std.shape))
    scaling = StandardScaler()  # Scale data before applying UMAP. https://umap-learn.readthedocs.io/en/latest/basic_usage.html, https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
    mu_std_scaled = scaling.fit_transform(mu_std)  # Use fit and transform method
    mu_scaled = scaling.fit_transform(mu)
    umap_reducer_mu_std = umap.UMAP()
    umap_reducer_mu = umap.UMAP()

    mu_umap_embedding = pd.DataFrame(umap_reducer_mu.fit_transform(mu_scaled), columns = ['UMAP1', 'UMAP2'])
    mu_std_umap_embedding = pd.DataFrame(umap_reducer_mu_std.fit_transform(mu_std_scaled), columns = ['UMAP1', 'UMAP2'])

    legend_map = {0:'< 5 years', 1: '>= 5 years'}
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig, ax2 = plt.subplots(figsize=(8, 6))

    # sns_plot1 = sns.scatterplot(x='UMAP1', y='UMAP2', data=mu_umap_embedding,
    #             hue=y,  # Not sure if this is correct. 
    #             linewidth=0, s=5, ax=ax1)
    
    sns_plot2 = sns.scatterplot(x='UMAP1', y='UMAP2', data=mu_std_umap_embedding,
                hue=y,   
                linewidth=0, s=5, ax=ax2, legend='full')
    
    # sns_plot1.legend(loc='center left', bbox_to_anchor=(1, .5))
    l = ax2.legend()
    l.get_texts()[0].set_text('< 5 years') # You can also change the legend title
    l.get_texts()[1].set_text('>= 5 years')
    sns_plot2.legend(loc='center left', bbox_to_anchor=(1, .5))

    # plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/train_all/images/UMAP_latent_surv_fold2ontrain' + str(save) + '_'  + str(args_.bc) + '_' +  str(args_.cvbs) + '_' + str(args_.conf) + '.png', bbox_inches='tight', dpi=500)

    # fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    fig, ax4 = plt.subplots(figsize=(8, 6))
    # sns_plot3 = sns.scatterplot(x='UMAP1', y='UMAP2', data=mu_umap_embedding,
    #             hue=s,  # Not sure if this is correct. 
    #             linewidth=0, s=5, ax=ax3)

    sns_plot4 = sns.scatterplot(x='UMAP1', y='UMAP2', data=mu_std_umap_embedding,
                hue=s,  # Not sure if this is correct. 
                linewidth=0, s=5, ax=ax4)
    sns_plot4.legend(loc='center left', bbox_to_anchor=(1, .5))

    # plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/train_all/images/UMAP_latent_study_fold2ontrain' + str(save) + '_'  + str(args_.bc) + '_' +  str(args_.cvbs) + '_' + str(args_.conf) + '.png', bbox_inches='tight', dpi=500)


def run_inference(gene_exp_file, dict_file, surv_labels_df, split_file=None):
    print('Configuration: ' + str(args_.conf) + ', Batch split: ' + args_.cvbs + ', Batch_correction: ' + str(args_.bc))
    start = time.time()
    test_dataloader = no_split_reader(gene_exp_file=gene_exp_file, surv_labels_df=surv_labels_df, shuffle=False)
    print(x for batch_idx, (x, y, s, _) in enumerate(test_dataloader))
    test_inp_size = [x.shape[2] for batch_idx, (x, y, s, _) in enumerate(test_dataloader)][0]
    print("This is the test_inp_size: " + str(test_inp_size))
    
    test_model = VAE(Encoder(input_size=test_inp_size), Decoder(input_size=test_inp_size))
    if DEVICE == torch.device("cuda"):
        checkpoint = torch.load(dict_file)
    elif DEVICE == torch.device("cpu"):
        checkpoint = torch.load(dict_file, map_location=DEVICE)
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
        for batch_idx, (x, y, s, _) in enumerate(test_dataloader):
            print("The current batch: " + str(batch_idx+1))
            print("Number of samples in the current batch: " + str(len(x)))
            print("This is the y size: " + str(y.shape))
            x = x.view(len(x), test_inp_size)  
            x = x.to(torch.float32)
            x = x.to(DEVICE)
            # y = y.view(y.shape[0])
            y = y.to(torch.float32).to(DEVICE)
            z, mu, log_var, mu_std, x_r= test_model(x)
            
            if batch_idx == 0:
                mu_cat = mu
                mu_std_cat = mu_std
                y_cat = y
                s_cat = list(s)
            else:
                mu_cat = torch.cat((mu_cat, mu),0 )
                mu_std_cat = torch.cat((mu_std_cat, mu_std), 0)
                y_cat = torch.cat((y_cat, y), 0)
                s_cat = s_cat + list(s)
            
            print(mu_std_cat.shape, y_cat.shape, len(s_cat))
            test_rec_loss, test_loss = test_model.loss_function(x_r=x_r, x=x, mean=mu, log_var=log_var)
            overall_test_loss += test_loss.item()
            overall_test_rec_loss += test_rec_loss.item()

            for i, (x_sample, x_r_sample) in enumerate(zip(x, x_r)):
                sample_pearson = stats.pearsonr(x_sample.detach().cpu().numpy(), x_r_sample.detach().cpu().numpy())[0]
                sum_pearson += sample_pearson
                if batch_idx == args_.idx and i == 0:
                #     # scatter_comparison(x_samp=x_sample, x_r_samp=x_r_sample, pear_corr=sample_pearson)
                    hexbin(x_samp=x_sample, x_r_samp=x_r_sample, pear_corr=sample_pearson)

        pear_average = sum_pearson/len(test_dataloader.dataset)

        print("length of the whole set: " + str(len(test_dataloader.dataset)))

        average_test_loss = overall_test_loss / len(test_dataloader.dataset)
        average_test_rec_loss = overall_test_rec_loss / len(test_dataloader.dataset)
    
    
    end = time.time()
    torch.save(mu_std_cat, '../data/new0501/embedding_tensors/embedding_fold2ontrain' + args_.cvbs + '_' + str(args_.conf) +'.pt')
    torch.save(y_cat, '../data/new0501/embedding_tensors/surv_labels_fold2ontrain' + args_.cvbs + '_' + str(args_.conf) +'.pt')
    torch.save(s_cat, '../data/new0501/embedding_tensors/study_labels_fold2ontrain' + args_.cvbs + '_' + str(args_.conf) +'.pt')
    # PCA_latent(mu=mu_cat, mu_std=mu_std_cat, y=y_cat, s=s_cat, save=args_.cvbs)
    umap_latent(mu=mu_cat, mu_std=mu_std_cat, y=y_cat, s=s_cat, save=args_.cvbs)
    

    print('Configuration: ' + str(args_.conf) + ', Batch split: ' + args_.cvbs + ", Average test loss per sample: " + str(average_test_loss) + ", Average test rec_loss per sample: " + str(average_test_rec_loss) + ", Average Pearson corr: " + str(pear_average) + " , Runtime: " + str(end - start))

    return mu_cat, mu_std_cat, y_cat, s_cat, average_test_loss, average_test_rec_loss, pear_average

def log_reg_all(model, x, y, x_test, y_test):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    x_test = x_test.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    model.fit(x, y)
    y_pred = model.predict(x_test)
    prc = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('Precision: ', prc, 'Recall: ', rec, 'Accuracy: ', acc, 'f1 score: ', f1)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('True negatives: ', tn, '\nFalse positives: ', fp, '\nFalse negatives: ', fn, '\nTrue Positives: ', tp)

    fig, ax = plt.subplots(figsize=(6, 6))  #AUPRC plot
    
    prc_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, lw=1, ax=ax) 
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="Recall",
        ylabel="Precision"
    )
    # ax.axis("square")
    ax.legend()
    ax.set_title('VAE (best model for all) + log_reg final test AUPRC')
    fig.savefig('../data/new0501/train_all/images/prc_fold2ontrain' + str(args_.conf) + '_' + str(args_.cvbs) + '.png')

    fig2, ax2 = plt.subplots(figsize=(6, 6))  # confusion matrix
    conmat_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues, ax=ax2)
    ax2.set_title(str(args_.cvbs) + ' confusion matrix')
    # fig2.savefig('../data/new0501/train_all/images/confusion_mat_fold2ontrain' + str(args_.conf) + '_' + str(args_.cvbs) + '.png')

    return 


def main():
    test_file = '../data/SyNet_bcnew_final_test.csv'
    surv_labels = pd.read_csv('../data/Survival_labels_filtered.csv', index_col=[0], header=[0])
    test_labels_df = surv_labels[surv_labels.index.str.contains('Miller|Minn')]
    # print(test_labels_df.shape)
    train_file = '../data/SyNet_bcnew_10studies.csv'
    train_labels_df = surv_labels.drop([idx for idx in surv_labels.index if 'Miller' in idx or 'Minn' in idx])
    print('train_labels_df shape: ' + str(train_labels_df.shape))
    
    # mu_test, mu_std_test, y_test, s_test = run_inference(gene_exp_file=test_file, dict_file=args_.sdp, surv_labels_df=test_labels_df)[0:4]
    mu_train, mu_std_train, y_train, s_train = run_inference(gene_exp_file=train_file, dict_file=args_.sdp, surv_labels_df=train_labels_df)[0:4]
    
    # torch.save(mu_std_test, '../data/new0501/embedding_tensors/test_mu_std_embedding_' + str(args_.conf) +'.pt')
    # torch.save(y_test, '../data/new0501/embedding_tensors/test_surv_labels_' + str(args_.conf) +'.pt')
    # torch.save(s_test, '../data/new0501/embedding_tensors/test_study_labels_' + str(args_.conf) +'.pt')

    # torch.save(mu_std_train, '../data/new0501/embedding_tensors/fold2ontrain_mu_std_embedding_' + str(args_.conf) +'.pt')
    # torch.save(y_train, '../data/new0501/embedding_tensors/fold2ontrain_surv_labels_' + str(args_.conf) +'.pt')
    # torch.save(s_train, '../data/new0501/embedding_tensors/fold2ontrain_study_labels_' + str(args_.conf) +'.pt')

    test_emb = torch.load('../data/new0501/embedding_tensors/test_mu_std_embedding_conf41.pt')
    test_lab = torch.load('../data/new0501/embedding_tensors/test_surv_labels_conf41.pt')

    # PCA_latent(mu=mu_full, mu_std=mu_std_full, y=y_full, s=s_full, save='full')
    # umap_latent(mu=mu_test, mu_std=mu_std_test, y=y_test, s=s_test, save='test')

    log_reg_all(model=LogisticRegression(penalty='l2',solver='lbfgs', max_iter=1500), 
                x=mu_std_train, y=y_train, x_test=test_emb, y_test=test_lab)
   

if __name__ == "__main__":
    main()