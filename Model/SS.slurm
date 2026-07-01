#!/bin/bash
#SBATCH -J SS
#SBATCH -o logs/SS_%j.out
#SBATCH -e logs/SS_%j.err
#SBATCH --partition=simbiosyslab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH -t 48:00:00
#SBATCH --mail-user=xing.liu@northeastern.edu
#SBATCH --mail-type=ALL

module load cuda/12.8.0
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python SS_train.py
