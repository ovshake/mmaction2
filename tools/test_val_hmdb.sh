
#config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_UCF_HMDB_baseline_rgb.py'


config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_latefusion_CPS_CS_concat_vcop.py'

d1_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline/tsm_k400_HMDB_late_fusion_CPS_VCOP/best_top1_acc_epoch_5.pth"




python tools/test.py $config $d1_ckpt --out "./ucf_output_eval.pkl" --eval top_k_accuracy  --cfg-options model.domain='hmdb51' data.test.domain='hmdb51'
python tools/test.py $config $d1_ckpt --out "./hmdb_output_eval.pkl" --eval top_k_accuracy  --cfg-options  model.domain='hmdb51' data.test.domain='ucf101'

# python tools/test.py $config $d1_ckpt --out "./ucf_output_eval.pkl" --eval top_k_accuracy  --cfg-options model.domain='ucf101' data.test.domain='hmdb51'
# python tools/test.py $config $d1_ckpt --out "./hmdb_output_eval.pkl" --eval top_k_accuracy  --cfg-options  model.domain='ucf101' data.test.domain='ucf101'