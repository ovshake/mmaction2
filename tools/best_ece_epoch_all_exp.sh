#!/bin/bash

section_name="tsm_r50_1x1x3_100e_ekmmsada_rgb_v2" 
run_name="color-contrastive-head-all-gather"
config_name="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_contrastivehead_ekmmsada_rgb.py"

bash tools/find_best_ece_epoch.sh $section_name $run_name $config_name 



section_name="tsm_r50_1x1x3_100e_ekmmsada_rgb_v2" 
run_name="speed-contrastive-head-all-gather"
config_name="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_slowfast_contrastivehead_ekmmsada_rgb.py"

bash tools/find_best_ece_epoch.sh $section_name $run_name $config_name 



section_name="tsm_r50_1x1x3_100e_ekmmsada_rgb_v2" 
run_name="multiple-contrastive-head-all-gather"
config_name="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ekmmsada_multiple_contrastive_space.py"

bash tools/find_best_ece_epoch.sh $section_name $run_name $config_name 


section_name="tsm_r50_1x1x3_100e_ekmmsada_rgb_v2" 
run_name="colorjitter-augself-10x"
config_name="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_augself_ekmmsada_rgb.py"

bash tools/find_best_ece_epoch.sh $section_name $run_name $config_name 


section_name="tsm_r50_1x1x3_100e_ekmmsada_rgb_v2" 
run_name="color-jitter-augself-contrastive-10x"
config_name="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_augself_contrastivehead_ekmmsada_rgb.py"

bash tools/find_best_ece_epoch.sh $section_name $run_name $config_name 