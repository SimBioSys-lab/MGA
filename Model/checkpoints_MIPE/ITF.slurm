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
#CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python Train_model_pretrained.py --model_module Models_fullnew \
#   --num_layers 1 --num_gnn_layers 10 --num_int_layers 5 --dropout 0.15 --drop_path_rate 0.10  --learning_rate 1e-4  \
#   --pretrained_ckpt_pattern "ctsr_ss_l1_g10_i5_do0.10_dpr0.10_lr0.0001_fold1_final_core.pth"   --out_dir checkpoints_old_isi

ls iMIPE*pth > iMIPE_i_files
python  PT_test.py --checkpoint_list iMIPE_i_files  --model_module Models_fullnew --out_dir result_MIPE_i  --best_category All    --copy_best  
