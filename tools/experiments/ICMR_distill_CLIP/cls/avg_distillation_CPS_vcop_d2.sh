#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J train_distil_CPS__vcop_3_D2_cls
#SBATCH -o ./out/tsm_k400_CPS_CS.out
#SBATCH -e ./out/tsm_k400_CPS_CS.err
#SBATCH --time 1-0
#SBATCH -p batch_grad
#SBATCH -w ai6
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------
# exp_name="tsm_k400_distillation_CPS_vcop"
# exp_section="ICMR_distil"
# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/ICMR_distil/tsm_r50_1x1x3_100e_ekmmsada_distillation_CPS_CS.py'



# #PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' model.domain='D1' total_epochs=100  --validate --deterministic

# #PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' model.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' model.domain='D3' total_epochs=100 --validate --deterministic 

#/data/shinpaul14/projects/mmaction2/work_dirs/ICMR_distil_final_new/tsm_k400_distillation_CPS_vcop
exp_name1="tsm_k400_distillation_CPS_vcop_avg"
exp_section1="ICMR_distil_final_clip"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_train_cls_batch_12.py'
exp_name="tsm_k400_distillation_CPS_vcop_avg"
exp_section="ICMR_distil_final_clip_cls"

#PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/latest.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/latest.pth" total_epochs=100  --validate --deterministic

