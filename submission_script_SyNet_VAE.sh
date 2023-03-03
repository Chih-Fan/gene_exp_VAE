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
-e 200 \
--trd "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/df_train_data_gene_exp.csv" \
--ted "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/df_test_data_gene_exp.csv" 