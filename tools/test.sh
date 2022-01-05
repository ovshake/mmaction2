
# exp_name="baseline"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ucf_hmdb_rgb.py"
# test_domain="kinetics"
# ckpt="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/baseline/train_ucf-hmdb_test_ucf-hmdb/best_top1_acc_epoch_55.pth"
# out_path="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/$exp_name/train_ucf-hmdb_test_kinetics"


# python tools/test.py $config $ckpt --out "$out_path/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain=$test_domain


# exp_name="colorjitter-contrastive-head"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ucf_hmdb_colorjitter_contrastive_head_rgb.py"
# test_domain="kinetics"
# out_path="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/$exp_name/train_ucf-hmdb_test_kinetics"
# ckpt="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/colorjitter-contrastive-head/train_ucf-hmdb_test_ucf-hmdb/best_top1_acc_epoch_85.pth"


# python tools/test.py $config $ckpt --out "$out_path/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain=$test_domain


# exp_name="slow-fast-contrastive-head"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ucf_hmdb_slowfast_contrastive_head_rgb.py"
# test_domain="kinetics"
# ckpt="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/slow-fast-contrastive-head/train_ucf-hmdb_test_ucf-hmdb/best_top1_acc_epoch_45.pth"
# out_path="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/$exp_name/train_ucf-hmdb_test_kinetics"

# python tools/test.py $config $ckpt --out "$out_path/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain=$test_domain




# exp_name="vcop-3"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ucf_hmdb_vcop_rgb.py"
# test_domain="kinetics"
# ckpt="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/vcop-3/train_ucf-hmdb_test_ucf-hmdb/best_top1_acc_epoch_20.pth"

# out_path="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/$exp_name/train_ucf-hmdb_test_kinetics"

# python tools/test.py $config $ckpt --out "$out_path/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain=$test_domain




# exp_name="colorjitter-contrastive-head"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ucf_hmdb_colorjitter_contrastive_head_rgb.py"
# test_domain="ucf-hmdb"
# out_path="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/$exp_name/train_kinetics_test_ucf-hmdb"
# ckpt="/data/abhishek/projects/mmactioˀn2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/colorjitter-contrastive-head/train_kinetics_test_kinetics/best_top1_acc_epoch_60.pth"


# python tools/test.py $config $ckpt --out "$out_path/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain=$test_domain


exp_name="baseline"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ucf_hmdb_rgb.py"
test_domain="ucf-hmdb"
out_path="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/$exp_name/train_kinetics_test_ucf-hmdb"
ckpt="/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb/baseline/train_kinetics_test_kinetics/best_top1_acc_epoch_60.pth"


python tools/test.py $config $ckpt --out "$out_path/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain=$test_domain