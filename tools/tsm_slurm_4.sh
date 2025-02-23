#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J tsm-4
#SBATCH -o /data/abhishek/outputs/slurm_4.out
#SBATCH -e /data/abhishek/outputs/slurm_4.err
#SBATCH -t 10-00:00
#SBATCH -p batch
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 64G


. /data/abhishek/anaconda3/etc/profile.d/conda.sh
conda activate action-dg




exp_name="tsm-mmsada-projhead-color-contrastive-resnet-frozenstages-2"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_mono-contrastive"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_slowfast_contrastivehead_ekmmsada_rgb_simsiam_xdb.py"


PORT=7070 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100 model.domain='D1' --validate

# PORT=7070 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100 model.detach_pathway_num=2 --validate

# PORT=7070 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100 model.detach_pathway_num=2 --validate