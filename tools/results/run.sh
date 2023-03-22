#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J results
#SBATCH -o /data/jongmin/projects/mmaction2_paul_work/tools/results/results_color_ablation.out
#SBATCH -e /data/jongmin/outputs/results.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -x ai[5,10]
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 20G

. /data/jongmin/anaconda3/etc/profile.d/conda.sh
conda activate action-dg
bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_01.sh

bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_02.sh

bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_04.sh

bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_06.sh

bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_07.sh

bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_08.sh

bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_09.sh

bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_1.sh

# bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_2.sh

# bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_3.sh

# bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_4.sh

# bash /data/jongmin/projects/mmaction2_paul_work/tools/results/test_val_5.sh