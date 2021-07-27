#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=expanding
#SBATCH -n 1
#SBATCH --mem-per-cpu=5G
#SBATCH -o logs/print_expanding.txt
#SBATCH -e logs/error_expanding.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

module load anaconda/3.6
source activate /scratch/itee/uqszhuan/COIL/env
module load cuda/10.0.130
module load gnu/5.4.0
module load mvapich2

for i in $(seq -f "%02g" 0 99)
do
  python create_expanded_corpus_with_TILDE.py \
    --output_dir corpus_TILDE \
    --file_path corpus/split${i}
done
