#!/bin/bash
#SBATCH -J MGA
#SBATCH -o logs/MGA_%j.out
#SBATCH -e logs/MGA_%j.err
#SBATCH --partition=simbiosyslab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH -t 48:00:00
#SBATCH --mail-user=xing.liu@northeastern.edu
#SBATCH --mail-type=ALL

module load cuda/12.8.0



#python check_ctsr_data_model_alignment.py \
#  --task ss \
#  --model-file Models_fullnew \
#  --sequence-file para_tv_esmsequences_1600.npz \
#  --data-file para_tv_esmss_1600.npz \
#  --edge-file para_tv_esmedges_1600.npz \
#  --num-classes 8 \
#  --num-workers 0
#
#python check_ctsr_data_model_alignment.py \
#  --task ctm \
#  --model-file Models_fullnew \
#  --sequence-file para_tv_esmsequences_1600.npz \
#  --data-file global_maps_para_esmtv.npz \
#  --edge-file para_tv_esmedges_1600.npz \
#  --num-classes 2 \
#  --num-workers 0
