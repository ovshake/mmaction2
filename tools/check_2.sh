exp_name="tsm_k400_UCF_distil_CPS_VCOP"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_latefusion_CPS_concat_vcop_clip.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' model.domain='hmdb51' total_epochs=100 --validate --deterministic








