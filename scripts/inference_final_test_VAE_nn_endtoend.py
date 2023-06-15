import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import BinaryF1Score
from torcheval.metrics import BinaryAccuracy
from torcheval.metrics import BinaryConfusionMatrix
import argparse
from make_dataloaders_labels_SyNet import CSV_reader
import csv
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-e', '--epochs', type=int, default=350, help='epochs')
parser.add_argument('--trbs', type=int, default=128, help='training batch size')
parser.add_argument('--valbs', type=int, default=128, help='validation batch size')
parser.add_argument('--tebs', type=int, default=128, help='testing batch size')
parser.add_argument('--ged', type=str, help='gene expression data path')
parser.add_argument('--spld', type=str, help='split file path')
parser.add_argument('--dr', type=float, default=0.5, help='dropout rate')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate, default=0.0001')
parser.add_argument('--lrf', type=float, default=0.1, help='learning rate step factor')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay in optimizer, default=0.0005')
parser.add_argument('--ralpha', type=float, default=2e-1, help='negative_slope of Leaky ReLU, default=0.2')
parser.add_argument('--conf', type=str, default='unnamed_config', help='configuration name')
parser.add_argument('--cvbs', type=str, default='unnamed_batch_split', help='cross validation batch split')
parser.add_argument('--first', type=int, help='dim of first linear layer')
parser.add_argument('--second', type=int, help='dim of second linear layer')
parser.add_argument('--third', type=int, help='dim of third linear layer')
parser.add_argument('--fourth', type=int, default=None, help='dim of fourth linear layer')
parser.add_argument('--bc', type=str, help='batch-correction label')
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logo = LeaveOneGroupOut()
surv_labels_file = "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/Survival_labels_filtered.csv"



class NNclassifier(nn.Module):
    def __init__(self, input_size, output_dim=1):
        super(NNclassifier, self).__init__()
        self.inp_dim = input_size
        self.out_dim = output_dim

        self.nn_sequential = nn.Sequential(
                                nn.Linear(self.inp_dim, args.first),
                                nn.Dropout(args.dr),
                                nn.LeakyReLU(args.ralpha),
                                nn.BatchNorm1d(args.first),
    
                                nn.Linear(args.first, args.second),
                                nn.Dropout(args.dr),
                                nn.LeakyReLU(args.ralpha),
                                nn.BatchNorm1d(args.second),

                                nn.Linear(args.second, args.third),
                                nn.Dropout(args.dr),
                                nn.LeakyReLU(args.ralpha),
                                nn.BatchNorm1d(args.third),

                                # nn.Linear(args.third, args.fourth),
                                # nn.Dropout(args.dr),
                                # nn.LeakyReLU(args.ralpha),
                                # nn.BatchNorm1d(args.fourth),

                                nn.Linear(args.third, output_dim),
                                nn.Sigmoid()
                                )
    
    def forward(self, x):
        out = self.nn_sequential(x)

        return out
    
def count_correct(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    # acc = (correct / len(y_pred)) 
    return correct
    

def main():
    print('Configuration: ' + str(args.conf) + ', Batch_corr: ' + str(args.bc) + ', Epochs: ' + str(args.epochs) + ', Training batch size: ' + str(args.trbs) + ', Validation batch size: ' + str(args.valbs) + ', Initial lr: ' + str(args.lr) + ', Dropout: ' + str(args.dr) + ', Weight decay: ' + str(args.wd))
    
    x_array = pd.read_csv('../data/SyNet_bcnew_final_test.csv', index_col=[0], header=[0]).to_numpy()
    df_y = pd.read_csv(surv_labels_file, index_col=[0], header=[0])
    y_array = df_y[df_y.index.str.contains('Miller|Minn')].to_numpy()
    print(x_array.shape, y_array.shape)
    
    loss_func = nn.BCELoss()
    f1_metric = BinaryF1Score().to(DEVICE)
    acc_metric = BinaryAccuracy().to(DEVICE)

    data = torch.tensor(x_array)
    labels = torch.tensor(y_array)

    dataset = torch.utils.data.TensorDataset(data, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = False)
        
    inp_size = [x.shape[1] for batch_idx, (x, y) in enumerate(loader)][0]
    dict_file = '../data/nn/trained_model_state_dict/f1/state_dict_conf1_fold10_epoch200.pt'
    if inp_size != 11748:
        raise ValueError('inp_size is not 11748')
        
    model = NNclassifier(input_size=inp_size).to(DEVICE)
    if DEVICE == torch.device("cuda"):
        checkpoint = torch.load(dict_file)
    elif DEVICE == torch.device("cpu"):
        checkpoint = torch.load(dict_file, map_location=DEVICE)
    else:
        print("Device is " , DEVICE)
        raise ValueError("Device is not cuda nor cpu.")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        test_correct = 0
        total_test_acc = 0
        total_test_f1 = 0
        overall_test_loss =0


        for batch_id, (x, y) in enumerate(loader):
            x = x.view(len(x), inp_size)  
            x = x.to(torch.float32)
            x = x.to(DEVICE)
            y = y.to(torch.float32).to(DEVICE)
            y_pred = model(x)
            y_pred = y_pred.to(torch.float32)
            test_loss = loss_func(y_pred, y)
            y_pred_class = torch.round(y_pred)

            y = y.view(y.shape[0])
            y_pred_class = y_pred_class.view(y_pred_class.shape[0])
            if batch_id == 0:
                y_cat = y_pred_class
            else:
                y_cat = torch.cat((y_cat, y_pred_class))
    
    print(y_cat.shape)

    test_correct += count_correct(y_true=labels.to(DEVICE), y_pred=y_cat)

    labels = labels.detach().cpu().numpy()
    y_cat = y_cat.detach().cpu().numpy()
    prc = precision_score(labels, y_cat)
    rec = recall_score(labels, y_cat)
    acc = accuracy_score(labels, y_cat)
    f1 = f1_score(labels, y_cat)
    print('Precision: ', prc, 'Recall: ', rec, 'Accuracy: ', acc, 'f1 score: ', f1)
    tn, fp, fn, tp = confusion_matrix(labels, y_cat).ravel()
    print('True negatives: ', tn, '\nFalse positives: ', fp, '\nFalse negatives: ', fn, '\nTrue Positives: ', tp)
    
    total_test_acc += acc
    total_test_f1 += f1
    overall_test_loss += test_loss.item()
    fig, ax = plt.subplots()
    prc_display = PrecisionRecallDisplay.from_predictions(labels, y_cat, alpha=0.8, lw=1, ax=ax)
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="Recall",
        ylabel="Precision"
    ) 
    # ax.axis("square")
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.set_title('Neural network final test AUPRC')
    # ax.set_xlabel('recall')
    # ax.set_ylabel('precision')
    
    fig.savefig('../data/nn/images/0513auprc_nn_final_test_fold10e200.png')

    fig2, ax2 = plt.subplots()
    conmat_display = ConfusionMatrixDisplay.from_predictions(labels, y_cat, cmap=plt.cm.Blues, ax=ax2)
    ax2.set_title('Confusion matrix')
    fig2.savefig('../data/nn/images/confusion_mat_nn_final_test_fold10e200.png')

    average_test_correct = test_correct / len(loader.dataset)
    average_test_loss = overall_test_loss / len(loader.dataset)
    average_test_acc = total_test_acc / len(loader)
    average_test_f1 = total_test_f1 / len(loader)
    
    print('Average_test_loss: ' + str(average_test_loss) + 
          ', Average_test_acc_count: ' + str(average_test_correct) + 
          ', Average_test_acc_score: ' + str(average_test_acc) +
          ', Average_test_f1: ' + str(average_test_f1))

if __name__ == "__main__":
    main()  