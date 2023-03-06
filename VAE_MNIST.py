import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.optim import Adam


# Use cuda GPU if it is avalible. Otherwise use CPU. 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keyword arguments
parser = argparse.ArgumentParser(description='Hyperparameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--path', default='~/datasets', help='dataset path')
parser.add_argument('-b', '--batch', type=int, default=100, help='batch size')
parser.add_argument('-x', '--xdim', type=int, default=784, help='input dimension')
parser.add_argument('--hiddim', type=int, default=400, help='hidden layer dimension')
parser.add_argument('-l', '--ladim', type=int, default=200, help='latent space dimension')
parser.add_argument('-r', '--lr', type=np.float32, default=1e-3, help='learning rate')
parser.add_argument('-e', '--epochs', type=int, default=30, help='epochs')
parser.add_argument('-d', '--state_dict_dir', default= './VAE_MNIST_state_dict.pt', help='directory to save the learned parameters (state dict)')
args = parser.parse_args()

# VAE
###    Step 1. Load (or download) Dataset

# Create a transform to apply to each datapoint
mnist_transform = transforms.Compose([
        transforms.ToTensor(),
])

# Keyword arguments in dictionary
kwargs = {'num_workers': 1, 'pin_memory': True} 

# Download the MNIST datasets
train_dataset = MNIST(args.path, transform=mnist_transform, train=True, download=True)  # "train" specify training or test dataset
test_dataset  = MNIST(args.path, transform=mnist_transform, train=False, download=True)

# Create train and test dataloaders that retrieves the dataset's features and labels one sample at a time. 
# Batches are shuffled at every epoch (after iteration of all batches). 
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True, **kwargs)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=args.batch, shuffle=False, **kwargs)


### Step 2. Define our model: Variational AutoEncoder (VAE)

"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Encoder(nn.Module):
    
    # Lego pieces of the model 
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
    
    # How the lego pieces are put together. How the data is flowing through these layers. 
    # Define how your network is going to be run. 
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
        

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):            # mean and log variance from the encoder's latent space
        epsilon = torch.randn_like(var).to(DEVICE)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick  ### WHY is it var not sigma(std)?
        return z
    
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var


encoder = Encoder(input_dim=args.xdim, hidden_dim=args.hiddim, latent_dim=args.ladim)
decoder = Decoder(latent_dim=args.ladim, hidden_dim = args.hiddim, output_dim = args.xdim)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

### Step 3. Define Loss function (reprod. loss) and optimizer

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

optimizer = Adam(model.parameters(), lr=args.lr)


### Step 4. Train Variational AutoEncoder (VAE)
if __name__ == '__main__':
    print("Start training VAE...")
    model.train()
    for epoch in range(args.epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(args.batch, args.xdim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*args.batch))
        
    print("Finish!!")
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    
    torch.save(model.state_dict(), args.state_dict_dir)
#onnxruntime https://onnxruntime.ai/docs/tutorials/