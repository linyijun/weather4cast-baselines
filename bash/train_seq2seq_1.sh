#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lin00786@umn.edu
#SBATCH -p v100

module load python3

source activate torch-env

cd /home/yaoyi/lin00786/weather4cast/weather4cast-baselines/

python train.py --batch_size 32 --kernel_size 3 --h_dim 64 --horizon 1 --source_var_idx 1 --target_var_idx 1 --model_name seq2seq --num_test 50
python train.py --batch_size 32 --kernel_size 1 --h_dim 64 --horizon 1 --source_var_idx 1 --target_var_idx 1 --model_name seq2seq --num_test 50
