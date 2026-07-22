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



CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python Train_model.py --out_dir checkpoints_MIPE_i --num_layers 1 --num_gnn_layers 6 --num_int_layers 3  --dropout 0.1 --drop_path_rate 0.1 --learning_rate 2e-4 
