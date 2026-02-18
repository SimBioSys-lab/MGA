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
python data_preprocess.py \
  --pdb-dir /projects/SimBioSys/Xing/MGA/data/ \
  --csv /projects/SimBioSys/Xing/MGA/data/train_set.csv \
  --hh-db-path /projects/SimBioSys/Xing/Uniref30/UniRef30_2023_02  \
  --scripts-dir /projects/SimBioSys/Xing/MGA/data_preprocess  \
  --work-dir /projects/SimBioSys/Xing/MGA/data \
  --preprocess-out-dir /projects/SimBioSys/Xing/MGA/data \
  --merge \
  --key-prefix-mode pdb

python itf_pipeline.py \
  --work-dir /projects/SimBioSys/Xing/MGA/data \
  --csv /projects/SimBioSys/Xing/MGA/data/train_set.csv \
  --pdb-dir /projects/SimBioSys/Xing/MGA/data/ \
  --ext .pdb \
  --cutoff 4.5 \
  --scripts-dir /projects/SimBioSys/Xing/MGA/data_preprocess
