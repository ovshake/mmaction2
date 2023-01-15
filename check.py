import pickle

with open('/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2_cls/tsm-k400-color-contrastive_xd_sgd_color_temp_50/train_D1_test_D1/output_eval.pkl', 'rb') as f:
	data = pickle.load(f)
print(data[9])