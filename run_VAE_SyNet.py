# std libs
import argparse


# torch libs
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim


from make_dataloaders_SyNet import CSV_reader

# Key word arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
parser.add_argument('-b', '--bs', type=str, default=128, help='batch size')
parser.add_argument('--trd', type=str, help='train data path')
parser.add_argument('--ted', type=str, help='test data path')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')

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
            print("current_hdim: " + str(hdim))
            if hdim == self.zdim:
                self.layers.append(nn.Linear(current_dim, 2*hdim))  # 2 because we have std and mean. 
                self.layers.append(nn.LeakyReLU(0.2))
                self.layers.append(nn.BatchNorm1d(2*hdim))
            else:
                self.layers.append(nn.Linear(current_dim, hdim))
                self.layers.append(nn.LeakyReLU(0.2))
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
            mu, variance = out.chunk(2, dim=1)      
            
            return mu, variance
    
class Decoder(nn.Module):
    def __init__(self, decoder_dims = [200, 500, 1000], input_size = None):
        """
            
        The decoder class
            
        """
        #if latent_dim == None or input_size == None:
         #   raise ValueError('Must explicitly declare input size and latent space dimension (2*latent_dim for enc)')
            
        super(Decoder, self).__init__();
        
        if decoder_dims == None or input_size == None:
            raise ValueError('Must explicitly declare input size (==output size) and decoder layer size')
            
        # self.inp_dim = decoder_dims[0]
        #self.zdim = decoder_dims[-1]

        current_dim = decoder_dims[0]
        self.layers = nn.ModuleList()
        for hdim in decoder_dims[1:]:
            print("current_hdim: " + str(hdim))
            self.layers.append(nn.Linear(current_dim, hdim))
            self.layers.append(nn.LeakyReLU(0.2))
            self.layers.append(nn.BatchNorm1d(hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, input_size))

    def forward(self, z):
        """
            
        Forward pass of the Decoder
            
        """
        out = z
        for sub_module in self.layers:
            out = sub_module(out)
        return out
    
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__() #initialises OrderedDicts to hold network parameters etc. https://stackoverflow.com/a/61288819 
        self.Encoder = Encoder  ### Encoder is an attribute
        self.Decoder = Decoder  ### Attribute created in .__init__() are called instance attributes. 


    def loss_function(self, pred, target, mean, log_var, average = False):
        reconstruction_loss = (pred - target).view(pred.size(0), -1)
        reconstruction_loss = reconstruction_loss**2
        reconstruction_loss = torch.sum(reconstruction_loss, dim=-1)
        if average:
            reconstruction_loss = reconstruction_loss.mean()
        else:
            reconstruction_loss = reconstruction_loss.sum()

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
    global args, model
    args = parser.parse_args()

    
    print(DEVICE) 

    train_dataloader, test_dataloader = CSV_reader(train_data_path=args.trd, test_data_path=args.ted, batch_size=args.bs, shuffle=True)
    inp_size = [batch[0].shape[1] for _, batch in enumerate(train_dataloader, 0)][0]
    # print([batch for _, batch in enumerate(train_dataloader, 0)][0])

    # encoder = Encoder(input_size=inp_size)
    # decoder = Decoder(input_size=200)  

    model = VAE(Encoder(input_size=inp_size), Decoder(input_size=inp_size)).to(DEVICE)  # Or give number of layers as arguments.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(args.epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_dataloader):  # How does the train_dataloader look like? We don't have labels. 

            x = x.view(len(x), inp_size)  # Returns a new tensor with the same data but of a different shape as given.
            x = x.to(torch.float32)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            z, mu, log_var, x_r= model(x)
            print(f'xdim: {x.shape}, x_r: {x_r.shape}')
            loss = model.loss_function(pred=x_r, target=x, mean=mu, log_var=log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*args.bs))
        
    print("Finish!!")
    

if __name__ == "__main__":
    main()  