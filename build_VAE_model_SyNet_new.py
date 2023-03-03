import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, encoder_dims = [1000, 500, 200], input_size = None):
        if encoder_dims == None or input_size == None:
            raise ValueError('Must explicitly declare input size and latent space dimension')
            
        super(Encoder, self).__init__()
        self.inp_dim = input_size  
        self.zdim = encoder_dims[-1]  # The last hidden layer of encoder. 

        current_dim = input_size  # The initial current dim is the input size. 
        self.layers = nn.ModuleList()  # This is a list of layers with their sizes. 
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
        
        #original code copied from ACTIVA
        # feed forward layers  
        # self.enc_sequential = nn.Sequential(
        #                         nn.Linear(self.inp_dim, 1024),
        #                         nn.ReLU(),
        #                         nn.BatchNorm1d(1024),
            
        #                         nn.Linear(1024, 512),
        #                         nn.ReLU(),
        #                         nn.BatchNorm1d(512),
            
        #                         nn.Linear(512, 256),
        #                         nn.ReLU(),
        #                         nn.BatchNorm1d(256),
            
        #                         nn.Linear(256, 2*self.zdim),
        #                         nn.ReLU(),
        #                         nn.BatchNorm1d(2*self.zdim)
        #                                    )
        
    def forward(self, x):        
        """
        
        Forward pass of the encoder
        
        """

        out = self.layers(x);
        # get mean and variance 
        mu, variance = out.chunk(2, dim=1)      # Why can chunk get us mu and var?
        
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
            
        self.inp_dim = input_size
        #self.zdim = decoder_dims[-1]

        current_dim = input_size
        self.layers = nn.ModuleList()
        for hdim in decoder_dims:
            print("current_hdim: " + str(hdim))
            self.layers.append(nn.Linear(current_dim, hdim))
            self.layers.append(nn.LeakyReLU(0.2))
            self.layers.append(nn.BatchNorm1d(hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, input_size))
        
        
        # All from ACTIVA which I copied for an idea + split into encoder and decoder
        
        # self.inp_dim = input_size;
        # self.zdim = latent_dim;
        # self.scale = scale;
        # self.lsn = LSN(scale=self.scale)
        
        # # here we decide the thresholding 
        # if threshold == 0:
        #     self.thres_layer = nn.ReLU()
        # else:
        #     # here we will replace all values smaller than threshold with 0
        #     print(f"==> thresholding with {threshold}")
        #     self.thres_layer = nn.Threshold(threshold, 0)
        
        
        # # feed forward layers
        # self.dec_sequential = nn.Sequential(
                                            
        #                         nn.Linear(self.zdim, 256),
        #                         nn.ReLU(),
        #                         nn.BatchNorm1d(256),
                                            
        #                         nn.Linear(256, 512),
        #                         nn.ReLU(),
        #                         nn.BatchNorm1d(512),
                                            
        #                         nn.Linear(512, 1024),
        #                         nn.ReLU(),
        #                         nn.BatchNorm1d(1024),
                                            
        #                         nn.Linear(1024, self.inp_dim),
        #                         self.thres_layer
                                
        #                         #this is from ACTIVA: https://github.com/SindiLab/ACTIVA 
        #                         # in our experiments, adding the LSN did not improve our results
        #                         ## but in case if you want to try it
        #                         #self.lsn
        #                                     )
    
    def forward(self, z):
        """
            
        Forward pass of the Decoder
            
        """
        
        out =  self.layers(z);
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
        """
        
        Forward pass through the network from the input data
        
        """
        mu, log_var = self.Encoder(x)
        # the latent space mu and variance, but reparametrized
        z = self.reparameterization(mu, torch.exp(0.5 * log_var))
        # the reconstructed data
        x_r = self.Decoder(z)
        
        return  z, mu, log_var, x_r

    # def reparameterize(self, mean, variance):
    #         """

    #         To do the reparametrization trick of the VAEs.
    #         Take in means and variances. Output samples from random normal distribution to add as noise (epsilon term in equations)

    #         """
        
    #         std = variance.mul(0.5).exp() 
    #         #okay so variance is a tensor or numpy array. I think. Then you multiply it by half and raise e to the power of that number.
    #         #Don't know how that results in std as standard deviation is the square root of variance?
    #         #see here https://towardsdatascience.com/reparameterization-trick-126062cfd3c3 
    #         # check to see which device we are running on
    #         if torch.cuda.is_available():
    #             epsilon = torch.cuda.FloatTensor(std.size(), requires_autograd = True).normal_() #see https://pytorch.org/docs/stable/generated/torch.Tensor.normal_.html 
    #         else:
    #             epsilon = torch.FloatTensor(std.size(), requires_autograd = True).normal_()     
                
    #         ##epsilon = Variable(eps) --> deprecated, now automatic autograd (= computational graph for gradient calculation) on tensors
    #         ## see: https://www.quora.com/What-is-the-difference-between-a-Tensor-and-a-Variable-in-Pytorch?share=1 
    #         return epsilon.mul(std).add(mean)


    




    ##Below is how ACTIVA incorporates the classifier loss in its training. Note that this external classifier really is not a very good
    ##or advanced classifier I think.
    ##We could in principle use this by taking a classifier that classifies breast cancer subtypes. But the whole goal is to do this on
    ##latent features. So you would have to make a training loop where you link a classifier for, say, breast cancer subtype and one for
    ##study of origin's training to this, and then add some part of the loss that this multi-task classifier has on useful properties to
    ##inform the latent space. Should not be able to predict study of origin (so the loss would be being able to do that), should be able to
    ##infor subtype for instance. 

    # def classification_loss(self, cf_prediction, cf_target, size_average=False):     
    # """
    
    # Cell type prediction loss between then generated cells and the real cells
    
    # """
    # error = (cf_prediction - cf_target).view(cf_prediction.size(0), -1)
    # error = error**2
    # error = torch.sum(error, dim=-1)
    
    # if size_average:
    #     error = error.mean()
    # else:
    #     error = error.sum()
            
    # return error

        
    #https://github.com/SindiLab/ACTIVA/blob/main/ACTIVA/networks/model.py
