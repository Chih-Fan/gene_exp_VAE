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
    
    groups_df = pd.read_csv('../data/SyNet_fold_1.csv', index_col=[0], header=[0])
    groups = groups_df.loc[(groups_df['study_name'] != 'Miller') & (groups_df['study_name'] != 'Minn')]['batch_label'].to_numpy()
    x_array = pd.read_csv('../data/SyNet_bcnew_10studies.csv', index_col=[0], header=[0]).to_numpy()
    df_y = pd.read_csv(surv_labels_file, index_col=[0], header=[0])
    y_array = df_y.drop([idx for idx in df_y.index if 'Miller' in idx or 'Minn' in idx], axis=0).to_numpy()
    
    loss_func = nn.BCELoss()
    f1_metric = BinaryF1Score().to(DEVICE)
    acc_metric = BinaryAccuracy().to(DEVICE)

    av_train_loss_history = []
    av_corrected_train_loss_history = []
    av_val_loss_history = []
    av_train_acc_history = []
    av_corrected_train_acc_history = []
    av_val_acc_history = []
    av_train_eval_acc_history = []
    av_corrected_train_eval_acc_history = []
    av_val_eval_acc_history = []
    av_train_f1_history = []
    av_corrected_train_f1_history = []
    av_val_f1_history = []

    # with open('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/log_files/' + str(args.bc)  + '_' + str(args.conf) + '.log', 'w') as log:
    #     csv_writer=csv.writer(log)
    #     csv_writer.writerow(['epoch', 'time_taken', 'train_loss', 'val_loss'])

    for fold, (train_index, val_index) in enumerate(logo.split(x_array, y_array, groups=groups)):
        print(f"Fold {fold+1}:")
        ### Dividing data into folds
        x_train_fold = torch.tensor(x_array[train_index])
        x_val_fold = torch.tensor(x_array[val_index])
        y_train_fold = torch.tensor(y_array[train_index])
        y_val_fold = torch.tensor(y_array[val_index])

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        val = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
        train_loader = torch.utils.data.DataLoader(train, batch_size = args.trbs, shuffle = False)
        val_loader = torch.utils.data.DataLoader(val, batch_size = args.valbs, shuffle = False)
        
        train_inp_size = [x.shape[1] for batch_idx, (x, y) in enumerate(train_loader)][0]
        
        model = NNclassifier(input_size=11748).to(DEVICE)
        # num_trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lrf, patience=10, verbose=True)

        best_epoch = -1
        best_acc = 0
        best_f1 = 0
        train_loss_history = []
        corrected_train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        corrected_train_acc_history = []
        val_acc_history = []
        train_eval_acc_history = []
        corrected_train_eval_acc_history = []
        val_eval_acc_history = []
        train_f1_history = []
        corrected_train_f1_history = []
        val_f1_history = []

        for epoch in range(args.epochs):
            model.train()
            overall_train_loss = 0
            overall_corrected_train_loss = 0
            overall_val_loss = 0
            train_correct = 0
            corrected_train_correct = 0
            val_correct = 0
            total_train_acc = 0
            total_corrected_train_acc = 0
            total_val_acc = 0
            total_train_f1 = 0
            total_corrected_train_f1 = 0
            total_val_f1 = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.view(len(x), train_inp_size)  # Returns a new tensor with the same data but of a different shape as given.
                x = x.to(torch.float32)
                x = x.to(DEVICE)
                y = y.to(torch.float32).to(DEVICE)

                y_pred = model(x)
                y_pred = y_pred.to(torch.float32)
                train_loss = loss_func(y_pred, y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                y_pred_class = torch.round(y_pred)
                y = y.view(y.shape[0])
                
                y_pred_class = y_pred_class.view(y_pred_class.shape[0])
                # if epoch == 0 and batch_idx == 0:
                #     print(y)
                #     print(y_pred_class)
                acc_metric.update(y, y_pred_class)
                acc = acc_metric.compute().detach().cpu().numpy()
                # print(acc)
                f1_metric.update(y, y_pred_class)
                f1 = f1_metric.compute().detach().cpu().numpy()
                # print(f1)
                
                train_correct += count_correct(y_true=y, y_pred=y_pred_class)  # Percentage of correct prediction. 
                total_train_acc += acc
                total_train_f1 += f1
                overall_train_loss += train_loss.item()

            model.eval()
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(train_loader):
                    x = x.view(len(x), train_inp_size)  # val_inp_size same as train_inp_size. 
                    x = x.to(torch.float32)
                    x = x.to(DEVICE)
                    y = y.to(torch.float32).to(DEVICE)
                    
                    y_pred = model(x)
                    y_pred = y_pred.to(torch.float32)
                    corrected_train_loss = loss_func(y_pred, y)

                    y_pred_class = torch.round(y_pred)
                    y = y.view(y.shape[0])
                    y_pred_class = y_pred_class.view(y_pred_class.shape[0])
                    acc_metric.update(y, y_pred_class)
                    acc = acc_metric.compute().detach().cpu().numpy()
                    f1_metric.update(y, y_pred_class)
                    f1 = f1_metric.compute().detach().cpu().numpy()

                    corrected_train_correct += count_correct(y_true=y, y_pred=y_pred_class)
                    total_corrected_train_acc += acc
                    total_corrected_train_f1 += f1
                    overall_corrected_train_loss += corrected_train_loss.item()    

                for batch_idx, (x, y) in enumerate(val_loader):
                    x = x.view(len(x), train_inp_size)  # val_inp_size same as train_inp_size. 
                    x = x.to(torch.float32)
                    x = x.to(DEVICE)
                    y = y.to(torch.float32).to(DEVICE)
                    
                    y_pred = model(x)
                    y_pred = y_pred.to(torch.float32)
                    val_loss = loss_func(y_pred, y)

                    y_pred_class = torch.round(y_pred)
                    y = y.view(y.shape[0])
                    y_pred_class = y_pred_class.view(y_pred_class.shape[0])
                    acc_metric.update(y, y_pred_class)
                    acc = acc_metric.compute().detach().cpu().numpy()
                    f1_metric.update(y, y_pred_class)
                    f1 = f1_metric.compute().detach().cpu().numpy()

                    val_correct += count_correct(y_true=y, y_pred=y_pred_class)
                    total_val_acc += acc
                    total_val_f1 += f1

                    overall_val_loss += val_loss.item()
                    
            lr_scheduler.step(overall_val_loss)   
            
            average_train_loss_ep = overall_train_loss / len(train)
            average_corrected_train_loss_ep = overall_corrected_train_loss / len(train)
            average_val_loss_ep = overall_val_loss / len(val)
            train_loss_history.append(average_train_loss_ep)
            corrected_train_loss_history.append(average_corrected_train_loss_ep)
            val_loss_history.append(average_val_loss_ep)

            average_train_acc_ep = train_correct / len(train) *100
            average_corrected_train_acc_ep = corrected_train_correct / len(train) *100
            average_val_acc_ep = val_correct / len(val) *100
            train_acc_history.append(average_train_acc_ep)
            corrected_train_acc_history.append(average_corrected_train_acc_ep)
            val_acc_history.append(average_val_acc_ep)

            average_train_eval_acc_ep = total_train_acc / len(train_loader) *100
            average_corrected_train_eval_acc_ep = total_corrected_train_acc / len (train_loader) *100
            average_val_eval_acc_ep = total_val_acc / len (val_loader) *100
            train_eval_acc_history.append(average_train_eval_acc_ep)
            corrected_train_eval_acc_history.append(average_corrected_train_eval_acc_ep)
            val_eval_acc_history.append(average_val_eval_acc_ep)

            average_train_f1_ep = total_train_f1 / len(train_loader)
            average_corrected_train_f1_ep = total_corrected_train_f1 / len(train_loader)
            average_val_f1_ep = total_val_f1 / len(val_loader)
            train_f1_history.append(average_train_f1_ep)
            corrected_train_f1_history.append(average_corrected_train_f1_ep)
            val_f1_history.append(average_val_f1_ep)


            if average_val_f1_ep > best_f1: 
                torch.save({
                    'fold': fold+1, 
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    '/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/nn/trained_model_state_dict/f1/state_dict_' + str(args.conf) + '_fold' + str(fold+1) + '_epoch' + str(epoch + 1) + '.pt'
                    )
                best_f1 = average_val_f1_ep
                best_epoch = epoch
            

        av_train_loss_history.append(np.mean(train_loss_history))
        av_corrected_train_loss_history.append(np.mean(corrected_train_loss_history))
        av_val_loss_history.append(np.mean(val_loss_history))
        av_train_acc_history.append(np.mean(train_acc_history))
        av_corrected_train_acc_history.append(np.mean(corrected_train_acc_history))
        av_val_acc_history.append(np.mean(val_acc_history))
        av_train_eval_acc_history.append(np.mean(train_eval_acc_history))
        av_corrected_train_eval_acc_history.append(np.mean(corrected_train_eval_acc_history))
        av_val_eval_acc_history.append(np.mean(val_eval_acc_history))
        av_train_f1_history.append(np.mean(train_f1_history))
        av_corrected_train_f1_history.append(np.mean(corrected_train_f1_history))
        av_val_f1_history.append(np.mean(val_f1_history))

        print('Best epoch:' + str(best_epoch+1) +', av_train_acc: ' + str(np.mean(train_acc_history))+
                  ', av_ctrain_acc: ' + str(np.mean(corrected_train_acc_history))+ 
                  ', av_val_acc: ' + str(np.mean(val_acc_history))+
                  ', av_train_eval_acc: ' + str(np.mean(train_eval_acc_history)) +
                  ', av_ctrain_eval_acc: ' + str(np.mean(corrected_train_eval_acc_history)) +
                  ', av_val_eval_acc: ' + str(np.mean(val_eval_acc_history)) +
                  ', av_train_f1: ' + str(np.mean(train_f1_history)) +
                  ', av_ctrain_f1: ' + str(np.mean(corrected_train_f1_history)) +
                  ', av_val_f1: ' + str(np.mean(val_f1_history)))
        # Plot losses
        fig1, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        ax1.plot(train_loss_history, label='train_loss')
        ax1.plot(corrected_train_loss_history, label='corrected_train_loss')
        ax1.plot(val_loss_history, label='val_loss')

        ax2.plot(train_f1_history, label='train_f1')
        ax2.plot(corrected_train_f1_history, label='corrected_train_f1')
        ax2.plot(val_f1_history, label='val_f1')

        ax3.plot(train_acc_history, label='train_acc')
        ax3.plot(corrected_train_acc_history, label='corrected_train_acc')
        ax3.plot(val_acc_history, label='val_acc')

        ax4.plot(train_eval_acc_history, label='train_eval_acc')
        ax4.plot(corrected_train_eval_acc_history, label='corrected_train_eval_acc')
        ax4.plot(val_eval_acc_history, label='val_eval_acc')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/nn/images/loss_history_nn_fold_' + str(fold+1) + str(args.bc)  + '_' + str(args.conf) +'.png')

    print('Performance of the {} fold cross validation'.format(logo.get_n_splits(groups=groups)))
    print("Average Training Loss: {:.3f} \t Average c-Training Loss: {:.3f} \t Average Validation Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average c-Training Acc: {:.2f} \t Average Validation Acc: {:.2f} \t Average Train eval Acc: {:.2f} \t Average c-train eval Acc: {:.2f} \t Average val eval Acc: {:.2f} \t Average train f1 Acc: {:.2f} \t Average c-train f1 Acc: {:.2f} \t Average val f1 Acc: {:.2f}".format(
        np.mean(av_train_loss_history), np.mean(av_corrected_train_loss_history), np.mean(av_val_loss_history), 
        np.mean(av_train_acc_history), np.mean(av_corrected_train_acc_history), np.mean(av_val_acc_history), 
        np.mean(av_train_eval_acc_history), np.mean(av_corrected_train_eval_acc_history), np.mean(av_val_eval_acc_history), 
        np.mean(av_train_f1_history), np.mean(av_corrected_train_f1_history), np.mean(av_val_f1_history)))
        

    

if __name__ == "__main__":
    main()  