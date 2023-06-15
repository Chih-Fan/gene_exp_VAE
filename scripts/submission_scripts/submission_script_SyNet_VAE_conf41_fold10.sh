#!/bin/bash

#SBATCH --time=5-00:00:00
#SBATCH -c 4
#SBATCH --partition gpu
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --mem=30G
#SBATCH --job-name=SyNet_VAE
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raecbce1216@gmail.com

source /home/cog/cchang/.bashrc
source activate synet_vae

export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64

python /hpc/compgen/users/cchang/Projects/gene_exp_VAE/scripts/run_VAE_SyNet_ropLR_tanh_scalef_dyn_loss_scaler_absx.py \
-e 700 \
--trbs 128 \
--valbs 128 \
--ged "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/SyNet_bcnew_filtered.csv" \
--spld "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/SyNet_fold_10.csv"  \
--wd 1e-5 \
--dr 0.2 \
--lr 1e-3 \
--conf 'conf41' \
--cvbs 'fold10_tanh_scale1_512_loss_scaler_absx1' \
--scalef 1 \
--first 512 \
--second 256 \
--third 128 \
--pseudoc 1 \
--bc 'bc'