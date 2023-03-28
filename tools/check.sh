

exp_name="trash"
exp_section="trash"
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/ensemble_teacher/tsm_r50_1x1x3_100e_ekmmsada_ensemble_all.py"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' data.test.domain='D1' total_epochs=1  --validate --deterministic
#PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' model.contrastive_loss.temperature=0.3 model.contrastive_loss.type_loss='supervised' total_epochs=100  --validate --deterministic