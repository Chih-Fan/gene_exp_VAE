# std libs
import argparse
import csv
import time
import matplotlib.pyplot as plt


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
args = parser.parse_args()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, encoder_dims = [1000, 500, 200], input_size = None):
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
            
            return mu, variance
    
class Decoder(nn.Module):
    def __init__(self, decoder_dims = [200, 500, 1000], input_size = None):  
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

    def forward(self, z):
        out = z
        for sub_module in self.layers:
            out = sub_module(out)
        # out = nn.Tanh()(out)  # Add a tanh activation function here. 
        return out
    
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__() #initialises OrderedDicts to hold network parameters etc. https://stackoverflow.com/a/61288819 
        self.Encoder = Encoder  ### Encoder is an attribute
        self.Decoder = Decoder  ### Attribute created in .__init__() are called instance attributes. 


    def loss_function(self, pred, target, mean, log_var, average = False):
        reconstruction_loss = (pred - target).view(pred.size(0), -1)  # -1: size automatically inferred to match the dimension
        reconstruction_loss = reconstruction_loss**2
        reconstruction_loss = torch.sum(reconstruction_loss, dim=-1)  # row sum --> loss of each sample
        if average:
            reconstruction_loss = reconstruction_loss.mean()
        else:
            reconstruction_loss = reconstruction_loss.sum()  
            # Here we take the sum of loss of all samples (in a batch). 
            # Later we add up loss from all batches and devided the loss by the number of all samples. 

        kl_loss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reconstruction_loss + kl_loss



    def reparameterization(self, mean, var):            # mean and log variance from the encoder's latent space
        epsilon = torch.randn_like(var).to(DEVICE)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick  ### WHY is it var not sigma(std)?
        return z
        
    def forward(self, x):
        mu, log_var = self.Encoder(x)
        # the latent space mu and variance, but reparametrized
        z = self.reparameterization(mu, torch.exp(0.5 * log_var))
        # the reconstructed data
        x_r = self.Decoder(z)
            
        return  z, mu, log_var, x_r


def main():
    
    print('Configuration: ' + str(args.conf) + ', Batch split: ' + str(args.cvbs) + ', Epochs: ' + str(args.epochs) + ', Training batch size: ' + str(args.trbs) + ', Validation batch size: ' + str(args.valbs) + ', Initial lr: ' + str(args.lr) + ', Dropout: ' + str(args.dr) + ', Weight decay: ' + str(args.wd))
    train_dataloader, val_dataloader, test_dataloader = CSV_reader(split_file=args.spld, gene_exp_file=args.ged,  train_batch_size=args.trbs, val_batch_size=args.valbs, test_batch_size=32, shuffle=True)
    # for idx, x in enumerate(val_dataloader):
        # print(idx)
        # print(x)
        # print(x.shape)
    inp_size = [batch[0].shape[1] for _, batch in enumerate(train_dataloader, 0)][0]  # enumerate count from 0
    # print("This is the train inp_size: "  + str(inp_size))


    model = VAE(Encoder(input_size=inp_size), Decoder(input_size=inp_size)).to(DEVICE) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lrf, patience=10, verbose=True)
    
    train_loss_history = []
    train_loss_history_corrected = []
    val_loss_history = []
    with open('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/batch_corrected_output/log_files/train_SyNet_VAE_batch-corrected_' + str(args.cvbs) + '_' + str(args.conf) + '.log', 'w') as log:
        csv_writer=csv.writer(log)
        csv_writer.writerow(['epoch', 'time_taken', 'train_loss', 'train_loss_corrected', 'val_loss'])

    for epoch in range(args.epochs):
        start = time.time()
        # Training loop
        model.train()
        overall_train_loss = 0
        for batch_idx, x in enumerate(train_dataloader):  # How does the train_dataloader look like? We don't have labels. 
            train_inp_size = x.shape[2]
            # print("This is the train_inp_size: " + str(train_inp_size))
            x = x.view(len(x), train_inp_size)  # Returns a new tensor with the same data but of a different shape as given.
            # print("Batch " + str(batch_idx) + "training tensor shape: " + str(x.shape))
            x = x.to(torch.float32)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            z, mu, log_var, x_r= model(x)
            train_loss = model.loss_function(pred=x_r, target=x, mean=mu, log_var=log_var)
                
            overall_train_loss += train_loss.item()
                
            train_loss.backward()
            optimizer.step()

        train_loss_history.append(overall_train_loss / (len(train_dataloader.dataset))) 

        # Get real training loss
        model.eval()
        with torch.no_grad():
            overall_train_loss_corrected = 0
            for batch_idx, x in enumerate(train_dataloader):  
                x = x.view(len(x), train_inp_size)  # len(x) = batch size, inp_size = number of features
                x = x.to(torch.float32)
                x = x.to(DEVICE)

                z, mu, log_var, x_r= model(x)
                train_loss_corrected = model.loss_function(pred=x_r, target=x, mean=mu, log_var=log_var)
                overall_train_loss_corrected += train_loss_corrected.item()

            average_train_loss_corrected = overall_train_loss_corrected / len(train_dataloader.dataset)
            train_loss_history_corrected.append(average_train_loss_corrected)

            # Validation loop
            
            overall_val_loss = 0
            for batch_idx, x in enumerate(val_dataloader):
                val_inp_size = x.shape[2]
                # print("This is the val_inp_size: " + str(val_inp_size))
                x = x.view(len(x), val_inp_size)  # Returns a new tensor with the same data but of a different shape as given.
                # print("Batch " + str(batch_idx) + "validation tensor shape: " + str(x.shape))
                x = x.to(torch.float32)
                # if batch_idx == 1:
                    # print(x.shape)
                x = x.to(DEVICE)

                z, mu, log_var, x_r= model(x)
                # if batch_idx == 1:
                    # print(x_r.shape)
                val_loss = model.loss_function(pred=x_r, target=x, mean=mu, log_var=log_var)
                overall_val_loss += val_loss.item()  # Is this correct?
            # print(batch_idx)
            average_val_loss = overall_val_loss / len(val_dataloader.dataset)
            val_loss_history.append(average_val_loss)

        lr_scheduler.step(overall_val_loss)  # Look at the sum of loss of the epoch. 

        end = time.time()
        if average_val_loss < 1000: 
            torch.save(model.state_dict(), '/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/batch_corrected_output/trained_model_state_dict/state_dict_' + str(args.conf) + '_' + str(args.cvbs) + '_epoch' + str(epoch + 1) + '_.pt')
        with open('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/batch_corrected_output/log_files/train_SyNet_VAE_batch-corrected_' + str(args.cvbs) + '_' + str(args.conf) + '.log', 'a') as log:
            csv_writer = csv.writer(log)
            csv_writer.writerow([str(epoch + 1), str(end-start), str(overall_train_loss / (len(train_dataloader.dataset))), str(average_train_loss_corrected), str(average_val_loss)])

    print('min val_loss: ' + str("%.3f" % min(val_loss_history)) + ', Epoch: ' + str(val_loss_history.index(min(val_loss_history))+1))
    plt.plot(train_loss_history, label='train_loss')
    plt.plot(train_loss_history_corrected, label='corrected_train_loss')
    plt.plot(val_loss_history,label='val_loss')
    plt.legend()
    plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/batch_corrected_output/images/loss_history_batch-corrected_' +  str(args.cvbs) + '_' + str(args.conf) +'.png')

if __name__ == "__main__":
    main()  
    