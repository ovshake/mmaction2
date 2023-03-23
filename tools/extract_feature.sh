config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/feature_extraction/tsm_r50_1x1x3_100e_ekmmsada_baseline_batch_12_extract.py'
exp_name='tsm-k400-vcop_clip4_batch_12_cls'
exp_section='tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls'

# /data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls

python ./tools/test.py $config ./work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_80.pth --out ./work_dirs/$exp_section/$exp_name/train_D1_test_D1/test_pkl.pkl 
python ./tools/test.py $config ./work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_55.pth --out ./work_dirs/$exp_section/$exp_name/train_D2_test_D2/test_pkl.pkl 
python ./tools/test.py $config ./work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_60.pth --out ./work_dirs/$exp_section/$exp_name/train_D3_test_D3/test_pkl.pkl  