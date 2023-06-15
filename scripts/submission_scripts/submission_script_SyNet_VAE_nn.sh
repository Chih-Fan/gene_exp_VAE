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

python /hpc/compgen/users/cchang/Projects/gene_exp_VAE/scripts/VAE_nn_endtoend_.py \
-e 600 \
--trbs 128 \
--valbs 32 \
--wd 1e-4 \
--dr 0.2 \
--lr 1e-3 \
--first 512 \
--second 256 \
--third 128 \
--conf 'conf1' \
--bc 'bc' 