#!/bin/bash
#SBATCH -J test_pipline
#SBATCH -o test_pipline.out
#SBATCH -e test_pipline.err
##SBATCH --partition=gpu-short
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -t 8:00:00
##SBATCH --export=ALL
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=xing.liu@northeastern.edu
#SBATCH --mail-type=ALL

#python make_prediction.py ../data/tmp/2b4c_seqs.npz ../data/tmp/2b4c_edges.npz

python run_pred.py \
  --csv /projects/SimBioSys/Xing/MGA/data/train_set.csv \
  --combined-seq-npz /projects/SimBioSys/Xing/MGA/data/esm_sequences_all.npz \
  --combined-edges-npz /projects/SimBioSys/Xing/MGA/data/edges_combined_all.npz \
  --preprocess-out-dir /projects/SimBioSys/Xing/MGA/pred_runs \
  --scripts-dir /projects/SimBioSys/Xing/MGA/data_preprocess \
  --sample-idx 0
