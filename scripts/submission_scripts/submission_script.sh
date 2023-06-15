#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH -c 4
#SBATCH --mem=10G
#SBATCH --job-name=VAE_SyNet
#SBATCH --mail-type=ALL
#SBATCH --mail-user=c.chang@umcutrecht.nl

source /home/cog/cchang/.bashrc
source activate sklearn-env2

python /home/cog/cchang/workfolder/Projects/gene_exp_VAE/scripts/run_VAE_SyNet.py \
--epochs 5 \
--trd /hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/df_train_data_gene_exp.csv \
--ted /hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/df_train_data_gene_exp.csv