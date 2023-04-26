import pandas as pd
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
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

def log_reg(model, x, y):
    total_acc = 0
    
    for fold, (train_index, test_index) in enumerate(logo.split(x, y, groups=groups)):
        ### Dividing data into folds
        x_train_fold = x[train_index]
        x_test_fold = x[test_index]
        y_train_fold = y[train_index]
        y_test_fold = y[test_index]

        if fold == 0:
            print('x_train_fold shape: ' + str(x_train_fold.shape))
            print('x_train_fold shape: ' + str(y_train_fold.shape))
            print(f"Fold {fold}:")
            print(f"  Train: index={train_index}, group={groups[train_index]}")
            print(f"  Test:  index={test_index}, group={groups[test_index]}")

        model.fit(x_train_fold, y_train_fold)
        model.predict(x_test_fold)
        score = model.score(x_test_fold, y_test_fold)

        # train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        # test = torch.utils.data.TensorDataset(x_test_fold, y_test_fold)
        # train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
        # test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

        total_acc += score
        print('Fold ' + str(fold+1) + ' accuracy: ' + str(score))
    total_acc = (total_acc / logo.get_n_splits(groups=groups))
    # print('\n\nTotal accuracy cross validation: {:.3f}%'.format(total_acc))
    
    return total_acc


def main(x, y):
    # x = x.to(DEVICE)
    # y = y.to(DEVICE)
    for fold, (train_index, test_index) in enumerate(logo.split(x, y, groups=groups)):
        print(fold)
        print(len(train_index), len(test_index))
        ### Dividing data into folds
        x_train_fold = x[train_index]
        x_test_fold = x[test_index]
        y_train_fold = y[train_index]
        y_test_fold = y[test_index]
        if fold == 0:
            latent = pca(x_train_fold, x_test_fold)
            # print(latent)
            print(latent.shape)
            y_latent = y_test_fold
            print(y_latent.shape)
        elif fold>0:
            latent_ = pca(x_train_fold, x_test_fold)
            latent = np.concatenate((latent, latent_), axis=0)
            y_latent = np.concatenate((y_latent, y_test_fold), axis=0) 
            print(latent.shape)
            print(y_latent.shape)
            
    
    print('latent shape: ' + str(latent.shape))
    print('y shape:' + str(y.shape))

    # for model in model_list:
    #     score = log_reg(model=model, x=latent, y=y)
    #     print(str(model) + ': ' + str(score))

    score = log_reg(model=log_reg_saga, x=latent, y=y)
    print('Accuracy: ' + str(score))


if __name__ == "__main__":
    x_array = pd.read_csv('../data/SyNet_bcnew_10studies.csv', index_col=[0], header=[0]).to_numpy()
    df_y = pd.read_csv(surv_labels_file, index_col=[0], header=[0])
    y_array = df_y.drop([idx for idx in df_y.index if 'Miller' in idx or 'Minn' in idx], axis=0).to_numpy()
    print('Results of Log_Reg')
    print('Configuration: ' + str(args.conf) + ', Batch split: ' + str(args.cvbs) + ', Batch_correction: ' + str(args.bc))
    print('arrays shape:' + str(x_array.shape) + str(y_array.shape))
    print(groups)
    print('groups shape' + str(groups.shape))
    main(x=x_array, y=y_array)

