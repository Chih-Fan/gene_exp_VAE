#!/bin/bash

#SBATCH --time=5-00:00:00
#SBATCH -c 4
#SBATCH --partition gpu
#SBATCH --gpus-per-node=RTX6000:1
#SBATCH --mem=64G
#SBATCH --job-name=SyNet_VAE
#SBATCH --mail-type=ALL
#SBATCH --mail-user=c.chang@umcutrecht.nl

source /home/cog/cchang/.bashrc
source activate synet_vae

export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64

python /hpc/compgen/users/cchang/Projects/gene_exp_VAE/scripts/run_VAE_SyNet.py \
-e 350 \
--ged "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/SyNet_Scaled_Batchcorrected_Labeled_Filtered_Data_Only.csv" \
--spld "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/SyNet_batching_1.csv"  \
--wd 1e-2