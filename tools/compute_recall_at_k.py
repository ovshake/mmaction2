import pickle
import pandas as pd
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def read_annotation_file(file_path):
    df = pd.read_pickle(file_path)
    return df

def parse_args():
    parser = argparse.ArgumentParser(description='Compute recall@k')
    parser.add_argument(
        '--annotation-file',
        help='annotation file path',
        required=True)
    parser.add_argument(
        '--output_file',
        help='output file path',
       required=True)
    parser.add_argument(
        '--top-k', type=int, default=5, help='top k predictions')
    args = parser.parse_args()
    return args

# Write a function to load file from pickle file
def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# write a function to compute cosine similarity
# matrix from a row of vectors

def compute_recall_at_k(ann_info, output_pkl, top_k=5):
    similarity_matrix = cosine_similarity(output_pkl)
    tp = 0
    for index, row in ann_info.iterrows():
        label = row['verb_class']
        prediction = output_pkl[index]
        similarity = similarity_matrix[index]
        similarity[index] = -2
        top_k_similarities = similarity.argsort()[-top_k:][::-1]
        top_k_labels = ann_info.iloc[top_k_similarities]['verb_class']
        if label in top_k_labels:
            tp += 1
    return tp / len(ann_info)



args = parse_args()
output_pkl = load_pickle_file(args.output_file)
ann_info = read_annotation_file(args.annotation_file)
recall_at_k = compute_recall_at_k(ann_info, output_pkl)
print(f"Recall at {args.top_k} is {recall_at_k}")