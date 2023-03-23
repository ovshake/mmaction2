#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J hmdb_late_distill
#SBATCH -o ./out/hmdb_baseline.out
#SBATCH -e ./out/train_cosdsdsdsdncat_all.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -w ai5
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G

. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg

exp_name="tsm_k400_HMDB_late_fusion_all_clip_all_frame"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_latefusion_CPS_CS_concat_vcop_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' model.clip_method='all' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_late_fusion_CPS_VCOP_clip_all_frame"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_latefusion_CPS_concat_vcop_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' model.clip_method='all' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_late_fusion_all_clip_center"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_latefusion_CPS_CS_concat_vcop_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' model.clip_method='center' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_late_fusion_CPS_VCOP_center"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_latefusion_CPS_concat_vcop_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' model.clip_method='center' total_epochs=100 --validate --deterministic

#--------------------------------------------------
exp_name="tsm_k400_HMDB_late_fusion_all"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_latefusion_CPS_CS_concat_vcop.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_late_fusion_CPS_VCOP"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_latefusion_CPS_concat_vcop.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_late_fusion_CS_VCOP"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_latefusion_CS_concat_vcop.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' total_epochs=100 --validate --deterministic
#-------------------- distill -------------------
exp_name="tsm_k400_HMDB_distil_all"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_ekmmsada_distillation_CPS_CS_vcop_all.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_distil_CPS_VCOP"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_ekmmsada_distillation_CPS_vcop.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_distil_CS_VCOP"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_ekmmsada_distillation_CS_vcop.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' total_epochs=100 --validate --deterministic

#----------------------------clip----------
exp_name="tsm_k400_HMDB_distil_all_clip_all"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_ekmmsada_distillation_CPS_CS_vcop_all_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51'  model.clip_method='all' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_distil_CPS_VCOP_clip_all"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_ekmmsada_distillation_CPS_vcop_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51'  model.clip_method='all' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_distil_all_clip_center"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_ekmmsada_distillation_CPS_CS_vcop_all_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51'  model.clip_method='center' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_distil_CPS_VCOP_clip_center"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_ekmmsada_distillation_CPS_vcop_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51'  model.clip_method='center' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_distil_all_clip_avg"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_ekmmsada_distillation_CPS_CS_vcop_all_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51'  model.clip_method='avg' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_HMDB_distil_CPS_VCOP_clip_avg"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_ekmmsada_distillation_CPS_vcop_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51'  model.clip_method='avg' total_epochs=100 --validate --deterministic


#-------------------------------------

 #------------------ cls head train ----------------
exp_name1="tsm_k400_HMDB_distil_all"
exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_distil_all"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/latest.pth" total_epochs=100  --validate --deterministic

exp_name1="tsm_k400_HMDB_distil_CPS_VCOP"
exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_distil_CPS_VCOP"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/latest.pth" total_epochs=100  --validate --deterministic


exp_name1="tsm_k400_HMDB_distil_CS_VCOP"
exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_distil_CS_VCOP"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/latest.pth" total_epochs=100  --validate --deterministic

#-------- clip cls head train ----------------
exp_name1="tsm_k400_HMDB_distil_all_clip_all"
exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_distil_all_clip_all"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/latest.pth" total_epochs=100  --validate --deterministic

exp_name1="tsm_k400_HMDB_distil_CPS_VCOP_clip_all"
exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_distil_CPS_VCOP_clip_all"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/latest.pth" total_epochs=100  --validate --deterministic

#-----------
exp_name1="tsm_k400_HMDB_distil_all_clip_center"
exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_distil_all_clip_center"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/latest.pth" total_epochs=100  --validate --deterministic

exp_name1="tsm_k400_HMDB_distil_CPS_VCOP_clip_center"
exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_distil_CPS_VCOP_clip_center"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/latest.pth" total_epochs=100  --validate --deterministic

#----------
exp_name1="tsm_k400_HMDB_distil_all_clip_avg"
exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_distil_all_clip_avg"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/latest.pth" total_epochs=100  --validate --deterministic

exp_name1="tsm_k400_HMDB_distil_CPS_VCOP_clip_avg"
exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_distil_CPS_VCOP_clip_avg"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/latest.pth" total_epochs=100  --validate --deterministic
