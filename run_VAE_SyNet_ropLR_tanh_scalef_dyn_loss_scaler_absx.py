# std libs
import argparse
import csv
import time
import matplotlib.pyplot as plt
from scipy import stats


# torch libs
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim

from make_dataloaders_SyNet import CSV_reader

# Key word arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-e', '--epochs', type=int, default=350, help='epochs')
parser.add_argument('--trbs', type=int, default=128, help='training batch size')
parser.add_argument('--valbs', type=int, default=128, help='validation batch size')
parser.add_argument('--tebs', type=int, default=128, help='testing batch size')
parser.add_argument('--ged', type=str, help='gene expression data path')
parser.add_argument('--spld', type=str, help='split file path')
parser.add_argument('--dr', type=float, default=0.5, help='dropout rate')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate, default=0.0001')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay in optimizer, default=0.0005')
parser.add_argument('--ralpha', type=float, default=2e-1, help='negative_slope of Leaky ReLU, default=0.2')
parser.add_argument('--imd', type=str, default='/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/images/loss_history.png', help='loss history plot saving directory')
parser.add_argument('--lrf', type=float, default=0.1, help='learning rate step factor')
parser.add_argument('--conf', type=str, default='unnamed_config', help='configuration name')
parser.add_argument('--cvbs', type=str, default='unnamed_batch_split', help='cross validation batch split')
parser.add_argument('--idx', type=int, default=0, help='index of sample to be plot')
parser.add_argument('--sdp', type=str, help='state dict path')
parser.add_argument('--first', type=int, help='dim of first linear layer')
parser.add_argument('--second', type=int, help='dim of second linear layer')
parser.add_argument('--third', type=int, help='dim of third linear layer')
parser.add_argument('--fourth', type=int, default=None, help='dim of fourth linear layer')
parser.add_argument('--scalef', type=float, default=1, help='scaling factor in decoder output layer')
parser.add_argument('--klscale', type=float, default=1.0, help='scaling factor for KL divergence')
parser.add_argument('--pseudoc', type=float, default=1.0, help='pseudocount for the loss scaling')
parser.add_argument('--absxscale', type=float, default=1.0, help='scaling factor for the abs x in loss scaling')
parser.add_argument('--bc', type=str, help='batch-correction label')
args = parser.parse_args()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, encoder_dims = [args.first, args.second, args.third], input_size = None):
        
        if encoder_dims == None or input_size == None:
            raise ValueError('Must explicitly declare input size and latent space dimension')
            
        super(Encoder, self).__init__()
        self.inp_dim = input_size
        self.zdim = encoder_dims[-1]

        current_dim = input_size
        self.layers = nn.ModuleList()
        for hdim in encoder_dims:
            # print("current_hdim: " + str(hdim))
            if hdim == self.zdim:
                self.layers.append(nn.Linear(current_dim, 2*hdim))  # 2 because we have std and mean. 
                self.layers.append(nn.Dropout(args.dr))
                self.layers.append(nn.LeakyReLU(args.ralpha))
                self.layers.append(nn.BatchNorm1d(2*hdim)) # Consider changing epsilon to 1e-02 if accuracy shows periodic fluctuations. See https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8.
            else:
                self.layers.append(nn.Linear(current_dim, hdim))
                self.layers.append(nn.Dropout(args.dr))  # Default drop out probability = 0.5. 
                self.layers.append(nn.LeakyReLU(args.ralpha))
                self.layers.append(nn.BatchNorm1d(hdim))
            current_dim = hdim

    def forward(self, x):        
            """
            
            Forward pass of the encoder
            
            """
            out = x
            for sub_module in self.layers:
                out = sub_module(out)
            # get mean and variance 
            mu, variance = out.chunk(2, dim=1)  # It is actually log variance here. 
            
            return mu, variance, out
    
class Decoder(nn.Module):
    def __init__(self, decoder_dims = [args.third, args.second, args.first], input_size = None):  
        # input_size is the sample's input size which is also the output size of the decoder. 
        super(Decoder, self).__init__()
        
        
        if decoder_dims == None or input_size == None:
            raise ValueError('Must explicitly declare input size (==output size) and decoder layer size')

        current_dim = decoder_dims[0]
        self.layers = nn.ModuleList()
        for hdim in decoder_dims[1:]:
            # print("current_hdim: " + str(hdim))
            self.layers.append(nn.Linear(current_dim, hdim))
            self.layers.append(nn.Dropout(args.dr))
            self.layers.append(nn.LeakyReLU(args.ralpha))
            self.layers.append(nn.BatchNorm1d(hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, input_size))
        self.layers.append(nn.Tanh())
        self.layers = nn.Sequential(*self.layers)
        

    def forward(self, z):
        out = (self.layers(z))*args.scalef
        return out
    
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__() #initialises OrderedDicts to hold network parameters etc. https://stackoverflow.com/a/61288819 
        self.Encoder = Encoder  ### Encoder is an attribute
        self.Decoder = Decoder  ### Attribute created in .__init__() are called instance attributes. 

    def loss_function(self, x, x_r, mean, log_var,  average = False):
        
        reconstruction_loss = (x - x_r).view(x_r.size(0), -1)  # -1: size automatically inferred to match the dimension
        pseudo_tensor = (torch.ones(x.size())*args.pseudoc).to(DEVICE)
        loss_scaler = torch.abs(x)*args.absxscale + pseudo_tensor
        reconstruction_loss = torch.mul(reconstruction_loss, loss_scaler)
        reconstruction_loss = reconstruction_loss**2

        reconstruction_loss = torch.sum(reconstruction_loss, dim=-1)  # row sum --> loss of each sample
        
        if average:
            reconstruction_loss = reconstruction_loss.mean()
        else:
            reconstruction_loss = reconstruction_loss.sum()  
            # Here we take the sum of loss of all samples (in a batch). 
            # Later we add up loss from all batches and devided the loss by the number of all samples. 

        kl_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        kl_loss = kl_loss*args.klscale

        return reconstruction_loss, reconstruction_loss + kl_loss

    def pearson_corelation(self, x_flat, x_r_flat):
        pearson_co = stats.pearsonr(x_flat, x_r_flat)
        return pearson_co

    def reparameterization(self, mean, var):            # mean and log variance from the encoder's latent space
        epsilon = torch.randn_like(var).to(DEVICE)      # sampling epsilon from a normal distribution N(0,1)       
        z = mean + var*epsilon                          # reparameterization trick  ### WHY is it var not sigma(std)?
        return z
        
    def forward(self, x):
        mu, log_var, mu_std = self.Encoder(x)
        # the latent space mu and variance, but reparametrized
        z = self.reparameterization(mu, torch.exp(0.5 * log_var))
        # the reconstructed data
        x_r = self.Decoder(z)
            
        return  z, mu, log_var, mu_std, x_r

def scatter_comparison(x_samp, x_r_samp):
    x_samp = x_samp.cpu().detach().numpy()
    x_r_samp = x_r_samp.cpu().detach().numpy()
    # print("This is x.shape in scatter com: " + str(x.shape))

    # print("This is the sample shape in scatter com: " + str(x[sample_idx].shape) + str(x_r[sample_idx].shape))
    fig, ax = plt.subplots()
    ax.scatter(x_samp, x_r_samp, alpha=0.6)
    fig.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/images/val_scatter_' + str(args.bc) + '_' +  str(args.cvbs) + '_' + str(args.conf) + '_batch0_sample0' + '.png')

def hexbinplot(x_samp, x_r_samp):
    x_samp = x_samp.cpu().detach().numpy()
    x_r_samp = x_r_samp.cpu().detach().numpy()
    
    fig, ax = plt.subplots()
    ax.hexbin(x_samp, x_r_samp, gridsize=60)
    fig.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/images/val_hexbin_' + str(args.bc) + '_' +  str(args.cvbs) + '_' + str(args.conf) + '_batch0_sample0' + '.png')

def main():
    
    print('Configuration: ' + str(args.conf) + ', Batch split: ' + str(args.cvbs) + ', Batch_corr: ' + str(args.bc) + ', Epochs: ' + str(args.epochs) + ', Training batch size: ' + str(args.trbs) + ', Validation batch size: ' + str(args.valbs) + ', Initial lr: ' + str(args.lr) + ', Dropout: ' + str(args.dr) + ', Weight decay: ' + str(args.wd))
    train_dataloader, val_dataloader, test_dataloader, final_test_dataloader = CSV_reader(split_file=args.spld, gene_exp_file=args.ged,  train_batch_size=args.trbs, val_batch_size=args.valbs, test_batch_size=32, shuffle=True)
    # for idx, x in enumerate(val_dataloader):
        # print(idx)
        # print(x)
        # print(x.shape)
    inp_size = [batch[0].shape[1] for _, batch in enumerate(train_dataloader, 0)][0]  # enumerate count from 0


    model = VAE(Encoder(input_size=inp_size), Decoder(input_size=inp_size)).to(DEVICE) 
    num_trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lrf, patience=10, verbose=True)
    early_stop_thresh = 15
    best_pearson = 0
    best_epoch = -1
    
    train_reconstruction_loss_history = []
    train_loss_history = []
    train_reconstruction_loss_history_corrected = []
    train_loss_history_corrected = []
    val_reconstruction_loss_history = []
    val_loss_history = []
    pearson_history = []
    pearson_history_corrected = []
    pearson_history_val = []


    with open('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/log_files/' + str(args.bc)  + '_' + str(args.cvbs) + '_' + str(args.conf) + '.log', 'w') as log:
        csv_writer=csv.writer(log)
        csv_writer.writerow(['epoch', 'time_taken', 'train_loss', 'train_loss_corrected', 'val_loss', 'train_rec_loss', 'train_rec_loss_corrected', 'val_rec_loss', 'train_pearson', 'train_pearson_corrected', 'val_pearson'])

    for epoch in range(args.epochs):
        start = time.time()
        # Training loop
        model.train()
        overall_train_loss = 0
        epoch_sum_pearson = 0
        overall_train_reconstruction_loss = 0
        for batch_idx, x in enumerate(train_dataloader):  # How does the train_dataloader look like? We don't have labels. 
            train_inp_size = x.shape[2]
            x = x.view(len(x), train_inp_size)  # Returns a new tensor with the same data but of a different shape as given.
            x = x.to(torch.float32)
            x = x.to(DEVICE)

            

            z, mu, log_var, mu_std, x_r= model(x)
            train_reconstruction_loss, train_loss = model.loss_function(x_r=x_r, x=x, mean=mu, log_var=log_var)
                
            overall_train_loss += train_loss.item()
            overall_train_reconstruction_loss += train_reconstruction_loss.item()
            
            optimizer.zero_grad()    
            train_loss.backward()
            optimizer.step()

            for (x_sample, x_r_sample) in zip(x, x_r):
                sample_pearson = stats.pearsonr(x_sample.detach().cpu().numpy(), x_r_sample.detach().cpu().numpy())[0]
                epoch_sum_pearson += sample_pearson
            
        average_train_loss = overall_train_loss / (len(train_dataloader.dataset))
        train_loss_history.append(average_train_loss) 
        average_pearson = epoch_sum_pearson / len(train_dataloader.dataset)
        pearson_history.append(average_pearson)
        average_train_reconstruction_loss = overall_train_reconstruction_loss / (len(train_dataloader.dataset))
        train_reconstruction_loss_history.append(average_train_reconstruction_loss)

        # Get real training loss
        model.eval()
        with torch.no_grad():
            overall_train_reconstruction_loss_corrected = 0
            overall_train_loss_corrected = 0
            epoch_sum_pearson_corrected = 0
            for batch_idx, x in enumerate(train_dataloader):  
                x = x.view(len(x), train_inp_size)  # len(x) = batch size, inp_size = number of features
                x = x.to(torch.float32)
                x = x.to(DEVICE)

                z, mu, log_var, mu_std, x_r= model(x)
                train_reconstruction_loss_corrected, train_loss_corrected = model.loss_function(x_r=x_r, x=x, mean=mu, log_var=log_var)
                overall_train_reconstruction_loss_corrected += train_reconstruction_loss_corrected.item()
                overall_train_loss_corrected += train_loss_corrected.item()

                for (x_sample, x_r_sample) in zip(x, x_r):
                    sample_pearson_corrected = stats.pearsonr(x_sample.detach().cpu().numpy(), x_r_sample.detach().cpu().numpy())[0]
                    epoch_sum_pearson_corrected += sample_pearson_corrected

            average_train_loss_corrected = overall_train_loss_corrected / len(train_dataloader.dataset)
            train_loss_history_corrected.append(average_train_loss_corrected)
            average_pearson_corrected = epoch_sum_pearson_corrected / len(train_dataloader.dataset)
            pearson_history_corrected.append(average_pearson_corrected)
            average_train_reconstruction_loss_corrected = overall_train_reconstruction_loss_corrected / (len(train_dataloader.dataset))
            train_reconstruction_loss_history_corrected.append(average_train_reconstruction_loss_corrected)


            # Validation loop
            
            overall_val_reconstruction_loss = 0
            overall_val_loss = 0
            epoch_sum_pearson_val = 0
            for batch_idx, x in enumerate(val_dataloader):
                val_inp_size = x.shape[2]
                # print("This is the val_inp_size: " + str(val_inp_size))
                x = x.view(len(x), val_inp_size)  # Returns a new tensor with the same data but of a different shape as given.
                # print("Batch " + str(batch_idx) + "validation tensor shape: " + str(x.shape))
                x = x.to(torch.float32)
                # if batch_idx == 1:
                    # print(x.shape)
                x = x.to(DEVICE)

                z, mu, log_var, mu_std, x_r= model(x)
                val_reconstruction_loss, val_loss = model.loss_function(x_r=x_r, x=x, mean=mu, log_var=log_var)
                overall_val_reconstruction_loss += val_reconstruction_loss.item()
                overall_val_loss += val_loss.item()  # Is this correct?

                for i, (x_sample, x_r_sample) in enumerate(zip(x, x_r)):
                    if (epoch+1) == args.epochs and batch_idx == 0 and i == 0:
                        scatter_comparison(x_samp=x_sample, x_r_samp=x_r_sample)
                        hexbinplot(x_samp=x_sample, x_r_samp=x_r_sample)

                    sample_pearson_val = stats.pearsonr(x_sample.detach().cpu().numpy(), x_r_sample.detach().cpu().numpy())[0]
                    epoch_sum_pearson_val += sample_pearson_val

            average_val_loss = overall_val_loss / len(val_dataloader.dataset)
            val_loss_history.append(average_val_loss)
            average_pearson_val = epoch_sum_pearson_val / len(val_dataloader.dataset)
            pearson_history_val.append(average_pearson_val)
            average_val_reconstruction_loss = overall_val_reconstruction_loss / (len(train_dataloader.dataset))
            val_reconstruction_loss_history.append(average_val_reconstruction_loss)

        lr_scheduler.step(overall_val_loss)  # Look at the sum of loss of the epoch. 

        end = time.time()
        with open('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/log_files/' + str(args.bc)  + '_'  + str(args.cvbs) + '_' + str(args.conf) + '.log', 'a') as log:
            csv_writer = csv.writer(log)
            csv_writer.writerow([str(epoch + 1), 
                                 str(end-start), 
                                 str(average_train_loss), 
                                 str(average_train_loss_corrected), 
                                 str(average_val_loss), 
                                 str(average_train_reconstruction_loss), 
                                 str(average_train_reconstruction_loss_corrected), 
                                 str(average_val_reconstruction_loss), 
                                 str(average_pearson), 
                                 str(average_pearson_corrected), 
                                 str(average_pearson_val)])
            
        if average_pearson_val > best_pearson: 
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 
                '/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/trained_model_state_dict/state_dict_' + str(args.conf) + '_' + str(args.cvbs) + '_epoch' + str(epoch + 1) + '.pt'
                )
            best_pearson = average_pearson_val
            best_epoch = epoch
        

    print('min val_loss: ' + str("%.3f" % min(val_loss_history)) + ', Epoch: ' + str(val_loss_history.index(min(val_loss_history))+1) + ', min val_rec_loss: '+ str("%.3f" % min(val_reconstruction_loss_history)) + ', Epoch: ' + str(val_reconstruction_loss_history.index(min(val_reconstruction_loss_history))+1) + ', max val_pearson: ' + str("%.3f" % max(pearson_history_val)) + ', Epoch: ' + str(pearson_history_val.index(max(pearson_history_val))+1) + ', should be equal to ' + str(best_epoch+1) + ', Total trainable parameters: ' + str(num_trainable_param))
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    ax1.plot(train_loss_history, label='train_loss')
    ax1.plot(train_loss_history_corrected[10:], label='corrected_train_loss')
    ax1.plot(val_loss_history[10:],label='val_loss')

    ax2.plot(pearson_history, label='train_pearson') 
    ax2.plot(pearson_history_corrected, label='corrected_train_pearson')
    ax2.plot(pearson_history_val, label='val_pearson')

    ax3.plot(train_reconstruction_loss_history, label='train_rec_loss')
    ax3.plot(train_reconstruction_loss_history_corrected, label='corrected_train_rec_loss')
    ax3.plot(val_reconstruction_loss_history, label='val_rec_loss')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/images/loss_history_' + str(args.bc)  + '_' +  str(args.cvbs) + '_' + str(args.conf) +'.png')

if __name__ == "__main__":
    main()  
    