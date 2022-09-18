import pickle
import argparse
import os
import os.path as osp
from mmaction.datasets import EpicKitchensMMSADA
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='Combining predictions from two different experiments')

    parser.add_argument('--exp1', help='experiment folder of the first experiment', type=str, required=True)
    parser.add_argument('--exp2', help='experiment folder of the second experiment', type=str, required=True)

    args = parser.parse_args()
    return args


def read_pkl(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        return file

def combine_predictions_average(exp1_preds, exp2_preds, gt_labels):
    if isinstance(exp1_preds, list):
        exp1_preds = np.asarray(exp1_preds)
    if isinstance(exp2_preds, list):
        exp2_preds = np.asarray(exp2_preds)
    if isinstance(gt_labels, list):
        gt_labels = np.asarray(gt_labels)

    return (np.argmax((exp1_preds + exp2_preds) / 2, axis=1) == gt_labels ).sum() / len(gt_labels)

def combine_predictions_OR(exp1_preds, exp2_preds, gt_labels):
    if isinstance(exp1_preds, list):
        exp1_preds = np.asarray(exp1_preds)
    if isinstance(exp2_preds, list):
        exp2_preds = np.asarray(exp2_preds)
    if isinstance(gt_labels, list):
        gt_labels = np.asarray(gt_labels)
    acc1 = np.argmax(exp1_preds, axis=1) == gt_labels
    acc2 = np.argmax(exp2_preds, axis=1) == gt_labels
    return (acc1 | acc2).sum() / len(gt_labels)

def main(args):
    domains = ['D1', 'D2', 'D3']
    labeldict = {k: [] for k in domains}
    for d in domains:
        dataset = EpicKitchensMMSADA(pipeline=[], domain=d, test_mode=True)
        gt_labels = [ann['label'] for ann in dataset.video_infos]
        labeldict[d] = gt_labels

    for d1 in domains:
        for d2 in domains:
            if d1 == d2:
                continue
            exp1_path = osp.join(args.exp1, f'train_{d1}_test_{d2}', 'output.pkl')
            exp2_path = osp.join(args.exp2, f'train_{d1}_test_{d2}', 'output.pkl')
            assert osp.exists(exp1_path)
            assert osp.exists(exp2_path)
            exp1_preds = read_pkl(exp1_path)
            exp2_preds = read_pkl(exp2_path)
            acc = combine_predictions_OR(exp1_preds, exp2_preds, labeldict[d2])
            print(f'train_{d1}_test_{d2} | acc: {acc:.4f}')

if __name__ == '__main__':
    args = parse_args()
    main(args)





