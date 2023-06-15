#!/bin/bash

#SBATCH --time=5-00:00:00
#SBATCH -c 4
#SBATCH --partition gpu
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --mem=10G
#SBATCH --job-name=SyNet_infer
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raecbce1216@gmail.com
​
source /home/cog/cchang/.bashrc
source activate synet_vae

export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64

python /hpc/compgen/users/cchang/Projects/gene_exp_VAE/scripts/inference_SyNet_VAE_plot_latent_scalef.py \
--ged "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/SyNet_bcnew_filtered.csv" \
--spld "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/SyNet_fold_4.csv"  \
--conf 'conf41' \
--cvbs 'fold4_tanh_scale1_512_loss_scaler_absx1' \
--sdp "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0418/trained_model_state_dict/best_models/state_dict_conf41_fold4_tanh_scale1_512_loss_scaler_absx1_epoch377.pt" \
--tebs 128 \
--first 512 \
--second 256 \
--third 128 \
--bc bc \
--idx 0