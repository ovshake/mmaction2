
train_k400_test_uh_baseline_output="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/baseline/train_kinetics_test_ucf-hmdb/output.pkl"

train_uh_test_k400_baseline_output="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/baseline/train_ucf-hmdb_test_kinetics/output.pkl"

exp_name="vcop-3"
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ucf_hmdb_vcop_rgb.py"
work_dir="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/vcop-3/train_ucf-hmdb_test_kinetics/videos"
ckpt_path="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/vcop-3/train_ucf-hmdb_test_ucf-hmdb/best_top1_acc_epoch_20.pth"
python analysis/compare_two_models.py -m1 /data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/$exp_name/train_ucf-hmdb_test_kinetics/output.pkl -m2 $train_uh_test_k400_baseline_output -t "kinetics" -w $work_dir --config-name $config --ckpt-path $ckpt_path -wd "Train UH Test Kinetics _ $exp_name-train-uh-test-k400"  
