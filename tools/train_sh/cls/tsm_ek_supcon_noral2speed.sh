#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J reproduce_cls
#SBATCH -o /data/shinpaul14/outputs/reproduce_cls.out
#SBATCH -e /data/shinpaul14/outputs/reproduce_cls.err
#SBATCH --time 1-0
#SBATCH -p batch_grad
#SBATCH -w ariel-v11
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg



# exp_name="tsm_k400_supcon_normal2color"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_reproduce"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/sup_con/tsm_r50_1x1x3_100e_ekmmsada_SupCon_normal2color.py"

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' model.contrastive_loss.temperature=0.3 model.contrastive_loss.type_loss='supervised' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' model.contrastive_loss.temperature=0.3 total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' model.contrastive_loss.temperature=0.3 total_epochs=100  --validate --deterministic

exp_name1="tsm_k400_supcon_normal2speed_avg"
exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_reproduce"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'
exp_name="tsm_k400_supcon_normal2speed_avg"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_reproduce_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="./work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="./work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/latest.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="./work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/latest.pth" total_epochs=100  --validate --deterministic
