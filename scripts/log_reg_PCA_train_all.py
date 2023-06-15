import pandas as pd
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
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

    return x_pca, x_test_pca

def log_reg_all(model, x, y, x_test, y_test):
    # x = x.detach().cpu().numpy()
    # y = y.detach().cpu().numpy()
    # x_test = x_test.detach().cpu().numpy()
    # y_test = y_test.detach().cpu().numpy()

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
    ax.set_title('PCA + log_reg final test AUPRC')
    fig.savefig('../data/new0501/train_all/images/prc_PCA_final_test.png')

    fig2, ax2 = plt.subplots(figsize=(6, 6))  # confusion matrix
    conmat_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues, ax=ax2)
    ax2.set_title('PCA + log_reg confusion matrix')
    fig2.savefig('../data/new0501/train_all/images/confusion_mat_PCA_final_test.png')

    return 



def main():
    # x = x.to(DEVICE)
    # y = y.to(DEVICE)
    surv_labels = pd.read_csv('../data/Survival_labels_filtered.csv', index_col=[0], header=[0])
    test_data = pd.read_csv('../data/SyNet_bcnew_final_test.csv', index_col=[0], header=[0])
    test_labels_df = surv_labels[surv_labels.index.str.contains('Miller|Minn')].to_numpy()
    # print(test_labels_df.shape)
    train_data = pd.read_csv('../data/SyNet_bcnew_10studies.csv', index_col=[0], header=[0])
    train_labels_df = surv_labels.drop([idx for idx in surv_labels.index if 'Miller' in idx or 'Minn' in idx]).to_numpy()
    print('train_labels_df shape: ' + str(train_labels_df.shape)) 

    train_latent, test_latent = pca(train_data, test_data)

    print('train latent shape: ' + str(train_latent.shape) + 'test latent shape: ' + str(test_latent.shape))

    # for model in model_list:
    #     score = log_reg(model=model, x=latent, y=y)
    #     print(str(model) + ': ' + str(score))

    log_reg_all(model=log_reg_lbfgs, x=train_latent, y=train_labels_df, x_test=test_latent, y_test=test_labels_df)


if __name__ == "__main__":
    print('Results of PCA Log_Reg')
    print('Configuration: ' + str(args.conf) + ', Batch split: ' + str(args.cvbs) + ', Batch_correction: ' + str(args.bc))
    main()

