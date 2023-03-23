#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J CLIP_distillation_all_frame_d3
#SBATCH -o ./out/CLIP_distillation_all_frame_d1.out
#SBATCH -e ./out/CLIP_distillation_all_frame_d1.err
#SBATCH --time 1-0
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------
exp_name="tsm_k400_distillation_CLIP_all_frame"
exp_section="Single_Distillation"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/Distillation/tsm_r50_1x1x3_100e_ekmmsada_distillation_CLIP.py'



#PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' model.domain='D1' model.type_loss='stack' model.clip_method='all' total_epochs=100 --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' model.domain='D2' total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' model.domain='D3' model.type_loss='stack' model.clip_method='all' total_epochs=100 --validate --deterministic 


# exp_name1="tsm_k400_multi_teacher_distillation_avg_all_clip_all_frames"
# exp_section1="Multi_Teacher_Distillation"
# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'
# exp_name="tsm_k400_multi_teacher_distillation_avg_all_clip_all_frames"
# exp_section="Multi_Teacher_Distillation_cls"

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/latest.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/latest.pth" total_epochs=100  --validate --deterministic

