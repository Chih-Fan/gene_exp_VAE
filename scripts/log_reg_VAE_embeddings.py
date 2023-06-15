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

log_reg_lbfgs = LogisticRegression(penalty='l2',solver='lbfgs', max_iter=1500)
log_reg_sag = LogisticRegression(penalty='l2',solver='sag', max_iter=1500)
log_reg_saga = LogisticRegression(penalty='l2',solver='saga', max_iter=1500)
log_reg_liblinear = LogisticRegression(penalty='l2',solver='liblinear', max_iter=1500)
model_list = [log_reg_lbfgs, log_reg_liblinear, log_reg_sag, log_reg_saga]
# log_reg_saga = LogisticRegression(penalty='l2',solver='saga', max_iter=2000)
logo = LeaveOneGroupOut()

path1 = '../data/new0418/embedding_tensors/embedding_fold'
path2 = '_tanh_scale1_512_loss_scaler_absx1_conf41.pt'
path3 = '../data/new0418/embedding_tensors/surv_labels_fold'
path4 = '_tanh_scale1_512_loss_scaler_absx1_conf41.pt'

fold1 = np.full(shape=183, fill_value='A', dtype=str)
fold2 = np.full(shape=150, fill_value='B', dtype=str)
fold3 = np.full(shape=1981, fill_value='C', dtype=str)
fold4 = np.full(shape=147, fill_value='F', dtype=str)
fold5 = np.full(shape=169, fill_value='G', dtype=str)
fold6 = np.full(shape=224, fill_value='H', dtype=str)
fold7 = np.full(shape=532, fill_value='I', dtype=str)
fold8 = np.full(shape=52, fill_value='J', dtype=str)
fold9 = np.full(shape=257, fill_value='K', dtype=str)
fold10 = np.full(shape=121, fill_value='L', dtype=str)

groups = np.concatenate((fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10), axis=None)

def concat_tensor(): 
    for n in range(10):
        # print(n+1)
        if n == 0:
            all_emb = torch.load(path1 + str(n+1) + path2, map_location=DEVICE)
            all_lab = torch.load(path3 + str(n+1) + path4, map_location=DEVICE)
            # print(all_emb.shape, all_lab.shape)
        else:
            emb = torch.load(path1 + str(n+1) + path2, map_location=DEVICE)
            lab = torch.load(path3 + str(n+1) + path4, map_location=DEVICE)
            # print(emb.shape, lab.shape)
            all_emb = torch.cat((all_emb, emb), 0)
            all_lab = torch.cat((all_lab, lab))
    
    # print(all_emb.shape, all_lab.shape)

    return all_emb, all_lab

def log_reg(model, x, y):
    total_acc = 0
    total_f1 = 0
    fig, ax = plt.subplots(figsize=(12, 6))

    for fold, (train_index, test_index) in enumerate(logo.split(x, y, groups=groups)):
        print('Fold' + str(fold+1))
        x_train_fold = x[train_index].detach().cpu().numpy()
        x_test_fold = x[test_index].detach().cpu().numpy()
        y_train_fold = y[train_index].detach().cpu().numpy()
        y_test_fold = y[test_index].detach().cpu().numpy()

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
        # fig2.savefig('../data/new0501/images/confusion_mat_fold' + str(fold+1) + str(args.conf) + '_' + str(args.cvbs) + '.png')
        

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
    fig.savefig('../data/new0501/images/0509prc_' + str(args.conf) + '_' + str(args.cvbs) + '.png')

    return average_acc

def main():
    # all_emb, all_lab = concat_tensor()
    # log_reg(model=log_reg_saga, x=all_emb, y=all_lab)
    all_emb = torch.load('../data/new0501/embedding_tensors/full_mu_std_embedding_conf41.pt', map_location=DEVICE)
    all_lab = torch.load('../data/new0501/embedding_tensors/full_surv_labels_conf41.pt', map_location=DEVICE)
    log_reg(model=log_reg_lbfgs, x=all_emb, y=all_lab)
    # for model in model_list:
    #     score = log_reg(model=model, x=all_emb, y=all_lab)
    #     print(str(model) + ' average score: ' + str(score))


if __name__ == "__main__":
    print('Results of Log_Reg on embeddings')
    print('Configuration: ' + str(args.conf) + ', Batch split: ' + str(args.cvbs) + ', Batch_correction: ' + str(args.bc))
    # print(groups)
    # print(len(groups))
    main()