import pandas as pd
import argparse
import numpy as np
import umap
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from make_dataloaders_labels_SyNet import CSV_reader, test_set_reader

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--trbs', type=int, default=128, help='training batch size')
parser.add_argument('--valbs', type=int, default=128, help='validation batch size')
parser.add_argument('--tebs', type=int, default=128, help='testing batch size')
parser.add_argument('--ged', type=str, help='gene expression data path')
parser.add_argument('--spld', type=str, help='split file path')
parser.add_argument('--cvbs', type=str, default='unnamed_batch_split', help='cross validation batch split')
parser.add_argument('--conf', type=str, help='configuration')
parser.add_argument('--bc')

args = parser.parse_args()

surv_labels_file = "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/Survival_labels_filtered.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_reg_lbfgs = LogisticRegression(penalty='l2',solver='lbfgs', max_iter=1000)
log_reg_sag = LogisticRegression(penalty='l2',solver='sag', max_iter=800)
log_reg_saga = LogisticRegression(penalty='l2',solver='saga', max_iter=800)
log_reg_liblinear = LogisticRegression(penalty='l2',solver='liblinear', max_iter=800)
model_list = [log_reg_lbfgs, log_reg_liblinear, log_reg_sag, log_reg_saga]
testing_score = []
logo = LeaveOneGroupOut()
groups_df = pd.read_csv('../data/SyNet_fold_1.csv', index_col=[0], header=[0])
groups = groups_df.loc[(groups_df['study_name'] != 'Miller') & (groups_df['study_name'] != 'Minn')]['batch_label'].to_numpy()

def pca(x, x_test):
    scaler = StandardScaler()
    pca = PCA(n_components=256)
    # pipe = Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=256))])
    x_scaled = scaler.fit_transform(x)
    x_test_scaled = scaler.fit_transform(x_test)
    x_pca = pca.fit_transform(x_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    # print("explained variance ratio (first two components): %s" % str(pca.explained_variance_ratio_))

    return x_test_pca

def plot_pca(latent, y, s, save): 
    y=np.squeeze(y)
    # print('y: '+str(y.shape))
    # print('s: '+ str(s.shape))
    scaling = StandardScaler()  # Scale data before applying PCA. https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
    latent_scaled = scaling.fit_transform(latent)  # Use fit and transform method

    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_scaled)

    print("explained variance ratio (first two components): %s" % str(pca.explained_variance_ratio_))

    label_names = ["< 5 years", ">= 5 years"]
    colors = ["lightseagreen", "darkorange"]
    studies = ['Desmedt-June07', 'Hatzis-Pusztai', 'METABRIC', 'Pawitan', 'Schmidt', 'Symmans ', 'TCGA', 'WangY', 'WangY-ErasmusMC', 'Zhang ']
    colors_studies = cm.rainbow(np.linspace(0, 1, len(studies)))
    lw = 2

    # for color, i, label_name in zip(colors, [0, 1], label_names):
        # print(latent_pca.shape)
        # print(latent_pca)
        # print(latent_pca[y == 0, 1])
        # ax1.scatter(latent_pca[y == i, 0], latent_pca[y == i, 1], color=color, alpha=0.6, lw=lw, label=label_name)
    # ax1.legend(loc="best", shadow=False, scatterpoints=1)
    # ax1.set_title("PCA of latent space - survival")
    # plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/images/PCA_latent_surv_' + str(save) + '_' + str(args_.bc) + '_' +  str(args_.cvbs) + '_' + str(args_.conf) + '.png')
    # print('y: '+str(y.shape))
    # print('s: '+ str(s.shape))
    # fig1, ax1 = plt.subplots()
    # for color, i, surv in zip(colors, [0, 1], label_names):
    #     ax1.scatter(latent_pca[y == i, 0], latent_pca[y == i, 1], color=color, alpha=0.6, lw=lw, s=5, label=surv)
    # # ax1.legend(loc="best", shadow=False, scatterpoints=1)
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.0)
    # ax1.set_title("PCA of latent space - survival")

    # plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/images/PCA_latent_survival_s5_' + str(save) + '_' + str(args.bc) + '_' +  str(args.cvbs) + '_' + str(args.conf) + '.png')


    fig2, ax2 = plt.subplots()
    for color, study in zip(colors_studies, studies):
        ax2.scatter(latent_pca[s == study, 0], latent_pca[s == study, 1], color=color, alpha=0.6, s=5, lw=lw, label=study)
    # ax2.legend(loc="best", shadow=False, scatterpoints=1)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax2.set_title("PCA of latent space - study")

    plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/images/PCA_latent_study_s5_' + str(save) + '_' + str(args.bc) + '_' +  str(args.cvbs) + '_' + str(args.conf) + '.png', bbox_inches='tight')

def umap_latent(latent, y, s, save):
    y=np.squeeze(y)
    # print('shape for umap: ' + str(latent.shape))
    scaling = StandardScaler()  # Scale data before applying UMAP. https://umap-learn.readthedocs.io/en/latest/basic_usage.html, https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler
    latent_scaled = scaling.fit_transform(latent)  # Use fit and transform method
    umap_model = umap.UMAP()

    embedding = pd.DataFrame(umap_model.fit_transform(latent_scaled), columns = ['UMAP1', 'UMAP2'])

    legend_map = {0:'< 5 years', 1: '>= 5 years'}
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig, ax2 = plt.subplots(figsize=(8, 6))

    # sns_plot1 = sns.scatterplot(x='UMAP1', y='UMAP2', data=mu_umap_embedding,
    #             hue=y,  # Not sure if this is correct. 
    #             linewidth=0, s=5, ax=ax1)
    
    sns_plot2 = sns.scatterplot(x='UMAP1', y='UMAP2', data=embedding,
                hue=pd.Series(y).map(legend_map),   
                linewidth=0, s=5, ax=ax2)
    # new_labels = ['< 5 years', '>= 5 years']
    # for t, l in zip(sns_plot2._legend.texts, new_labels):
        # t.set_text(l)
    
    # sns_plot1.legend(loc='center left', bbox_to_anchor=(1, .5))
    # l = ax2.legend()
    # l.get_texts()[0].set_text('< 5 years') # You can also change the legend title
    # l.get_texts()[1].set_text('>= 5 years')
    sns_plot2.legend(loc='center left', bbox_to_anchor=(1, .5))

    plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/images/UMAP_latent_surv_' + str(save) + '_'  + str(args.bc) + '_' +  str(args.cvbs) + '_' + str(args.conf) + '.png', bbox_inches='tight', dpi=500)

    # fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    fig, ax4 = plt.subplots(figsize=(8, 6))
    # # sns_plot3 = sns.scatterplot(x='UMAP1', y='UMAP2', data=mu_umap_embedding,
    # #             hue=s,  # Not sure if this is correct. 
    # #             linewidth=0, s=5, ax=ax3)

    sns_plot4 = sns.scatterplot(x='UMAP1', y='UMAP2', data=embedding,
                hue=s,  # Not sure if this is correct. 
                linewidth=0, s=5, ax=ax4)
    sns_plot4.legend(loc='center left', bbox_to_anchor=(1, .5))

    plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/images/UMAP_latent_study_' + str(save) + '_'  + str(args.bc) + '_' +  str(args.cvbs) + '_' + str(args.conf) + '.png', bbox_inches='tight', dpi=500)


def log_reg(model, x, y):
    total_acc = 0
    total_f1 = 0
    fig, ax = plt.subplots(figsize=(12, 6))

    for fold, (train_index, test_index) in enumerate(logo.split(x, y, groups=groups)):
        print('Fold' + str(fold+1))
        x_train_fold = x[train_index]
        x_test_fold = x[test_index]
        y_train_fold = y[train_index]
        y_test_fold = y[test_index]

        model.fit(x_train_fold, y_train_fold)
        y_pred = model.predict(x_test_fold)
        prc = precision_score(y_test_fold, y_pred)
        rec = recall_score(y_test_fold, y_pred)
        acc = accuracy_score(y_test_fold, y_pred)
        f1 = f1_score(y_test_fold, y_pred)
        print('Precision: ', prc, 'Recall: ', rec, 'Accuracy: ', acc, 'f1 score: ', f1)
        tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).ravel()
        print('True negatives: ', tn, '\nFalse positives: ', fp, '\nFalse negatives: ', fn, '\nTrue Positives: ', tp)

        total_acc += acc
        total_f1 += f1

        # fig2, ax2 = plt.subplots(figsize=(6, 6))
        prc_display = PrecisionRecallDisplay.from_predictions(y_test_fold, y_pred, name=f"AUPRC fold {fold+1}", alpha=0.8, lw=1, ax=ax) 
        # conmat_display = ConfusionMatrixDisplay.from_predictions(y_test_fold, y_pred, cmap=plt.cm.Blues, ax=ax2)

        # ax2.set_title('Fold ' + str(fold+1) + ' confusion matrix')
        # fig2.savefig('../data/new0501/images/confusion_mat_pca_fold' + str(fold+1) + str(args.conf) + '_' + str(args.cvbs) + '.png')
    
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="Recall",
        ylabel="Precision"
    )
    ax.axis("square")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.0)

    average_acc = (total_acc / logo.get_n_splits(groups=groups))
    average_f1 = (total_f1 / logo.get_n_splits(groups=groups))
    print('Average accuracy:' + str(average_acc) + ', Average f1 score: ' + str(average_f1))
    fig.savefig('../data/new0501/images/0509prc_pca_' + str(args.conf) + '_' + str(args.cvbs) + '.png')
    
    return average_acc


def main(x, y, s):
    # x = x.to(DEVICE)
    # y = y.to(DEVICE)
    for fold, (train_index, test_index) in enumerate(logo.split(x, y, groups=groups)):
        print(fold)
        # print(len(train_index), len(test_index))
        ### Dividing data into folds
        x_train_fold = x[train_index]
        x_test_fold = x[test_index]
        y_train_fold = y[train_index]
        y_test_fold = y[test_index]
        s_train_fold = s[train_index]
        s_test_fold = s[test_index]
        # Can write the above part into a method if have time. 
        
        if fold == 0:
            latent = pca(x_train_fold, x_test_fold)
            # print(latent)
            # print(latent.shape)
            y_latent = y_test_fold
            s_latent = s_test_fold
            # print(y_latent.shape)
        elif fold>0:
            latent_ = pca(x_train_fold, x_test_fold)
            latent = np.concatenate((latent, latent_), axis=0)
            y_latent = np.concatenate((y_latent, y_test_fold), axis=0)
            s_latent = np.concatenate((s_latent, s_test_fold), axis=0) 
            # print(latent.shape)
            # print(y_latent.shape)
        print(latent.shape)

        # plot_pca(latent=latent, y=y_latent, s=s_latent, save='all_folds')
        umap_latent(latent=latent, y=y_latent, s=s_latent, save='all_folds')
            
    
    print('latent shape: ' + str(latent.shape))
    print('y shape:' + str(y.shape))
    print('s shape: ' + str(s.shape))

    # for model in model_list:
    #     score = log_reg(model=model, x=latent, y=y)
    #     print(str(model) + ': ' + str(score))

    # log_reg(model=log_reg_lbfgs, x=latent, y=y)


if __name__ == "__main__":
    x_array = pd.read_csv('../data/SyNet_bcnew_10studies.csv', index_col=[0], header=[0]).to_numpy()
    df_y = pd.read_csv(surv_labels_file, index_col=[0], header=[0])
    y_array = df_y.drop([idx for idx in df_y.index if 'Miller' in idx or 'Minn' in idx], axis=0).to_numpy()
    id_list = list(df_y.drop([idx for idx in df_y.index if 'Miller' in idx or 'Minn' in idx], axis=0).index.values)
    s_array = np.asarray([id.split(';')[1] for id in id_list])
    # print(s_list, len(s_list))

    print('Results of VAE Log_Reg')
    print('Configuration: ' + str(args.conf) + ', Batch split: ' + str(args.cvbs) + ', Batch_correction: ' + str(args.bc))
    print('arrays shape:' + str(x_array.shape) + str(y_array.shape))
    print('groups shape' + str(groups.shape))
    main(x=x_array, y=y_array, s=s_array)

