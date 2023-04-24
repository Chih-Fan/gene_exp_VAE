from run_VAE_SyNet_ropLR_tanh_scalef import DEVICE, Encoder, Decoder, VAE
from make_dataloaders_SyNet import test_set_reader
import matplotlib.pyplot as plt
import torch
import argparse
import time


parser_ = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_.add_argument('--ged', type=str, help='gene expression data path')
parser_.add_argument('--spld', type=str, help='split file path')
parser_.add_argument('--conf', type=str, default='unnamed_config', help='configuration name')
# parser_.add_argument('--idx', type=int, default=0, help='index of sample to be plot')
# parser_.add_argument('--sdp', type=str, help='state dict path')
parser_.add_argument('--cvbs', type=str, default='unnamed_batch_split', help='cross validation batch split')
# parser_.add_argument('--dr', type=float, default=0.5, help='dropout rate')
# parser_.add_argument('--ralpha', type=float, default=2e-1, help='negative_slope of Leaky ReLU, default=0.2')
args_ = parser_.parse_args()


def scatter_comparison(x, x_r, idx, batch_size):
    x = x.view(batch_size, 11748)
    x_r = x_r.view(batch_size, 11748)

    print(x[idx].shape, x_r[idx].shape)

    plt.scatter(x[idx], x_r[idx])
    plt.savefig('/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/images/scatter_com_batch-un-corrected_' +  str(args_.cvbs) + '_' + str(args_.conf) + '_sample' + str(0) + '_rerun' + '.png')


def main():
    start = time.time()
    test_dataloader = test_set_reader(split_file=args_.spld, gene_exp_file=args_.ged, test_batch_size=154, shuffle=False) 
    test_inp_size = [x.shape[2] for batch_idx, x in enumerate(test_dataloader)][0]
    print("This is the test_inp_size: " + str(test_inp_size))

    test_model = VAE(Encoder(input_size=test_inp_size), Decoder(input_size=test_inp_size))
    if DEVICE == 'cuda':
        test_model.load_state_dict(torch.load("/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/trained_model_state_dict/best_models/state_dict_conf38_b1_epoch404.pt"))
    elif DEVICE == 'cpu':
        test_model.load_state_dict(torch.load("/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/trained_model_state_dict/best_models/state_dict_conf38_b1_epoch404.pt", map_location=DEVICE))

    # print(test_model.state_dict()) 
    test_model.to(DEVICE)
    
    test_model.eval()
    with torch.no_grad():
        test_loss_list = []
        overall_test_loss = 0
        for batch_idx, x in enumerate(test_dataloader):
            print("The current batch: " + str(batch_idx))
            print("Number of samples in the current batch: " + str(len(x)))
            x = x.view(len(x), test_inp_size)  
            x = x.to(torch.float32)
            x = x.to(DEVICE)
            z, mu, log_var, x_r= test_model(x)
            test_loss = test_model.loss_function(pred=x_r, target=x, mean=mu, log_var=log_var)
            test_loss_list.append(test_loss)
            print("test_loss for batch " + str(batch_idx) + " is: " + str(test_loss))
            overall_test_loss += test_loss.item()
        
        print(test_loss_list)
        print("length of the whole test set: " + str(len(test_dataloader.dataset)))
        

        average_test_loss = overall_test_loss / len(test_dataloader.dataset)
    
    scatter_comparison(x=x, x_r=x_r, idx=0, batch_size=154)
    end = time.time()

    print('Configuration: ' + str(args_.conf) + ', Batch split: ' + str(args_.cvbs) + ", Overall test loss: " + str(overall_test_loss) + ", Average test loss per sample: " + str(average_test_loss) + ", Runtime: " + str(end - start))

    return average_test_loss

if __name__ == "__main__":
    main()