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

python /hpc/compgen/users/cchang/Projects/gene_exp_VAE/scripts/inference_final_test_VAE.py \
--ged "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/SyNet_bcnew_filtered.csv" \
--spld "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/SyNet_fold_2.csv"  \
--conf 'conf41' \
--cvbs 'final_test' \
--sdp "/hpc/compgen/users/cchang/Projects/gene_exp_VAE/data/new0501/train_all/trained_model_state_dict/state_dict_conf1_unnamed_batch_split_fold2_epoch488.pt" \
--tebs 128 \
--first 512 \
--second 256 \
--third 128 \
--bc bc \
--idx 0