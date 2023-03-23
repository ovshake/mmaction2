# # COLOR
# domain="D1"
# work_dir="/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/color"
# ckpt="/data/jongmin/projects/mmaction2/work_/color/$domain/latest.pth"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/color_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_color_frozen_cls_temp_5.py"
# python tools/test.py $config $ckpt --out "$work_dir/$domain/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain=$domain

# domain="D2"
# work_dir="/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/color"
# ckpt="/data/jongmin/projects/mmaction2/work_/color/$domain/latest.pth"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/color_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_color_frozen_cls_temp_5.py"
# python tools/test.py $config $ckpt --out "$work_dir/$domain/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain=$domain

# domain="D3"
# work_dir="/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/color"
# ckpt="/data/jongmin/projects/mmaction2/work_/color/$domain/latest.pth"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/color_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_color_frozen_cls_temp_5.py"
# python tools/test.py $config $ckpt --out "$work_dir/$domain/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain=$domain


# # SPEED

# domain="D1"
# work_dir="/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/speed"
# ckpt="/data/jongmin/projects/mmaction2/work_/speed/$domain/latest.pth"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_speed_frozen_cls_temp_5.py"
# python tools/test.py $config $ckpt --out "$work_dir/$domain/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain=$domain

# domain="D2"
# work_dir="/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/speed"
# ckpt="/data/jongmin/projects/mmaction2/work_/speed/$domain/latest.pth"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_speed_frozen_cls_temp_5.py"
# python tools/test.py $config $ckpt --out "$work_dir/$domain/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain=$domain

# domain="D3"
# work_dir="/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/speed"
# ckpt="/data/jongmin/projects/mmaction2/work_/speed/$domain/latest.pth"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_speed_frozen_cls_temp_5.py"
# python tools/test.py $config $ckpt --out "$work_dir/$domain/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain=$domain

# VCOP

domain="D1"
work_dir="/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/vcop"
ckpt="/data/jongmin/projects/mmaction2/work_/vcop/$domain/latest.pth"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/VCOP/tsm_r50_1x1x3_100e_ekmmsada_vcop_rgb_frozen_cls.py"
python tools/test.py $config $ckpt --out "$work_dir/$domain/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain=$domain

domain="D2"
work_dir="/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/vcop"
ckpt="/data/jongmin/projects/mmaction2/work_/vcop/$domain/latest.pth"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/VCOP/tsm_r50_1x1x3_100e_ekmmsada_vcop_rgb_frozen_cls.py"
python tools/test.py $config $ckpt --out "$work_dir/$domain/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain=$domain

domain="D3"
work_dir="/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/vcop"
ckpt="/data/jongmin/projects/mmaction2/work_/vcop/$domain/latest.pth"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/VCOP/tsm_r50_1x1x3_100e_ekmmsada_vcop_rgb_frozen_cls.py"
python tools/test.py $config $ckpt --out "$work_dir/$domain/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain=$domain